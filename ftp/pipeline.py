"""SFT data-generation pipeline for the single-CUSIP BG/FTP market-making experiment.

Faithful to the Bergault-Gueant setup: a desk that makes markets in ONE bond. The bond's
parameters (sigma, kappa, spread, base price) are FIXED. The only variation across episodes
is the **regime realization** -- a fresh MMPP path and flow history each episode -- exactly
what the desk faces day to day on its one bond.

  bond      = ONE fixed CUSIP (sampled once, or pinned via GenConfig)
  oracle    = BG full-information quotes for that bond (sees true MMPP regime)
  episode   = one finite-horizon day of that bond's RFQ flow under one MMPP path
  task axis = number of EPISODES (distinct regime realizations), not number of bonds
  label     = the oracle's discrete action (offset bin) at each RFQ
  example   = serialized (history -> action) string; SFT loss masked to action tokens

Generalization (held-out test) is to UNSEEN regime realizations on the SAME bond -- fresh
MMPP paths -- not to unseen bonds. The MMPP regime is censored from the serialized
observation; the agent gets sector-flow context to infer it.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from rfqsim.config import SimConfig
from rfqsim.mmpp import build_generator
from .env import BondParams, generate_episode
from .oracle import BGOracle, BGOracleParams, FillCurve
from .rollout import OraclePolicy, RandomPolicy, run_episode, pooled_gap
from .serialize import serialize_episode


@dataclass
class GenConfig:
    # Single-CUSIP: variation is over EPISODES (regime realizations), not bonds.
    n_train_episodes: int = 1600     # distinct regime realizations for training
    n_test_episodes: int = 320       # held-out regime realizations (same bond)
    horizon_days: float = 1.0
    gamma: float = 0.5               # inventory risk aversion (Scope-A primary)
    gamma_term: float = 0.0          # pure-BG terminal condition
    z: float = 1.0                   # reference trade size
    sector_intensity: float = 800.0
    cusip_frac: float = 0.06
    sector_halflife: float = 0.03
    oracle_n_t: int = 1200
    bond_seed: int = 8               # the fixed bond's identity (seed 8: kappa~0.33,
                                     # belief-gap ~0.72 -> regime matters, healthy band)
    seed: int = 11                   # episode-generation stream seed

    # Optional: pin the bond's parameters explicitly instead of sampling.
    bond_base_price: float | None = None
    bond_spread: float | None = None
    bond_kappa: float | None = None
    bond_sigma: float | None = None


def make_bond(cfg: SimConfig, gc: GenConfig) -> BondParams:
    """Construct THE single bond. Either pinned via GenConfig fields, or sampled once
    from a dedicated bond_seed so the same bond is reproducible across runs."""
    if all(v is not None for v in (gc.bond_base_price, gc.bond_spread,
                                   gc.bond_kappa, gc.bond_sigma)):
        return BondParams(base_price=gc.bond_base_price, spread=gc.bond_spread,
                          kappa=gc.bond_kappa, sigma=gc.bond_sigma,
                          sector_intensity=gc.sector_intensity, cusip_frac=gc.cusip_frac)
    return BondParams.sample(cfg, np.random.default_rng(gc.bond_seed),
                             gc.sector_intensity, gc.cusip_frac)


def build_oracle(cfg: SimConfig, bond: BondParams, gc: GenConfig) -> BGOracle:
    """Construct the BG oracle for the (single) bond from rfqsim-calibrated parameters."""
    Q = build_generator(cfg.mmpp)
    lo = cfg.mmpp.lam_low_frac * bond.sector_intensity
    hi = cfg.mmpp.lam_high_frac * bond.sector_intensity
    lam_b = np.array([lo, lo, hi, hi]); lam_a = np.array([lo, hi, lo, hi])
    fill = FillCurve(alpha=cfg.outcome.logit_alpha, beta=cfg.outcome.logit_beta,
                     delta0=bond.spread)
    cb, ca = lam_b * bond.cusip_frac, lam_a * bond.cusip_frac
    params = BGOracleParams(lam_b=cb, lam_a=ca, Q=Q, kappa=bond.kappa,
                            sigma=cfg.price.sigma_issuer_daily, gamma=gc.gamma,
                            z=gc.z, fill=fill, T=gc.horizon_days,
                            gamma_term=gc.gamma_term)
    return BGOracle(params, n_t=gc.oracle_n_t)


# back-compat alias used by evaluate.py / trace.py
_build_oracle = build_oracle


def generate_split(cfg: SimConfig, gc: GenConfig, bond: BondParams, oracle: BGOracle,
                   n_episodes: int, rng, split: str):
    """Generate serialized oracle episodes (one fixed bond, varying regime). JSONL records."""
    records, gap_O, gap_R = [], [], []
    ei = 0
    attempts = 0
    while ei < n_episodes and attempts < n_episodes * 4:
        attempts += 1
        ep = generate_episode(cfg, bond, gc.horizon_days, rng)
        if len(ep.steps) < 2:
            continue
        seed = int(rng.integers(1 << 30))
        res_o = run_episode(np.random.default_rng(seed), ep, OraclePolicy(oracle),
                            oracle, cfg, z=gc.z, sector_halflife=gc.sector_halflife)
        res_r = run_episode(np.random.default_rng(seed), ep,
                            RandomPolicy(np.random.default_rng(seed + 1)),
                            oracle, cfg, z=gc.z, sector_halflife=gc.sector_halflife)
        header = (f"<EPISODE bond={split} ep={ei} horizon={gc.horizon_days} "
                  f"n_rfq={len(ep.steps)}>")
        text = serialize_episode(res_o["obs_records"], res_o["actions"],
                                 res_o["rewards"], res_o["q_path"], z=gc.z, header=header)
        records.append({
            "text": text + "\n<END>",
            "meta": {"split": split, "episode_idx": ei, "n_rfq": len(ep.steps),
                     "bg_objective_oracle": round(res_o["bg_objective"], 4),
                     "bg_objective_random": round(res_r["bg_objective"], 4)},
        })
        gap_O.append(res_o["bg_objective"]); gap_R.append(res_r["bg_objective"])
        ei += 1
    stats = {"split": split, "n_episodes": len(records),
             "pooled_gap_oracle": round(pooled_gap(gap_O, gap_O, gap_R), 4),
             "pooled_gap_random": round(pooled_gap(gap_R, gap_O, gap_R), 4)}
    return records, stats


def run_pipeline(out_dir: str = "./sft_data", cfg: SimConfig | None = None,
                 gc: GenConfig | None = None):
    cfg = cfg or SimConfig()
    gc = gc or GenConfig()
    os.makedirs(out_dir, exist_ok=True)

    # ONE bond, ONE oracle -- fixed across all episodes (the single-CUSIP desk).
    bond = make_bond(cfg, gc)
    oracle = build_oracle(cfg, bond, gc)
    rng = np.random.default_rng(gc.seed)

    train, s_tr = generate_split(cfg, gc, bond, oracle, gc.n_train_episodes, rng, "train")
    test, s_te = generate_split(cfg, gc, bond, oracle, gc.n_test_episodes, rng, "test")

    for name, recs in [("train", train), ("test", test)]:
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    manifest = {"gen_config": asdict(gc), "bond": {k: round(v, 4) for k, v in asdict(bond).items()},
                "train_stats": s_tr, "test_stats": s_te,
                "action_bins": list(__import__("ftp.serialize", fromlist=["ACTION_BINS"])
                                    .ACTION_BINS)}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=float)
    return manifest
