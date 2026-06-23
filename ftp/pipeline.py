"""SFT data-generation pipeline for the BG/FTP market-making experiment (Scope A).

Produces fine-tuning data for an LLM that imitates the BG quadratic-approximation oracle:

  task  = one CUSIP (its own sigma, kappa, spread, base price)  ~ universe distribution
  oracle= BG full-information quotes for that CUSIP (sees true MMPP regime)
  episode= finite-horizon single-CUSIP RFQ stream; oracle quotes, inventory threads
  label = the oracle's discrete action (offset bin) at each RFQ
  example= serialized (history -> action) string; SFT loss masked to action tokens

CUSIPs are split into train/test so generalization is to UNSEEN bonds (the paper's
"new task" axis). The MMPP regime is censored from the serialized observation; the agent
gets sector-flow context to infer it. Output: JSONL of {text, meta} records + a manifest.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from rfqsim.config import SimConfig
from .env import BondParams, generate_episode
from .oracle import BGOracle, BGOracleParams, FillCurve
from .rollout import OraclePolicy, RandomPolicy, run_episode, pooled_gap
from .serialize import serialize_episode


@dataclass
class GenConfig:
    n_train_cusips: int = 200
    n_test_cusips: int = 40
    episodes_per_cusip: int = 8
    horizon_days: float = 1.0
    gamma: float = 0.5            # inventory risk aversion (Scope-A primary)
    gamma_term: float = 0.0      # pure-BG terminal condition
    z: float = 1.0               # reference trade size
    sector_intensity: float = 800.0
    cusip_frac: float = 0.06
    sector_halflife: float = 0.03
    oracle_n_t: int = 1200
    seed: int = 11


def _build_oracle(cfg: SimConfig, bond: BondParams, gc: GenConfig) -> BGOracle:
    """Construct the BG oracle for one CUSIP from rfqsim-calibrated parameters."""
    from rfqsim.mmpp import build_generator
    Q = build_generator(cfg.mmpp)
    lo = cfg.mmpp.lam_low_frac * bond.sector_intensity
    hi = cfg.mmpp.lam_high_frac * bond.sector_intensity
    lam_b = np.array([lo, lo, hi, hi])
    lam_a = np.array([lo, hi, lo, hi])
    fill = FillCurve(alpha=cfg.outcome.logit_alpha, beta=cfg.outcome.logit_beta,
                     delta0=bond.spread)
    # the oracle's per-CUSIP intensities scale by this bond's flow share
    cb, ca = lam_b * bond.cusip_frac, lam_a * bond.cusip_frac
    params = BGOracleParams(lam_b=cb, lam_a=ca, Q=Q, kappa=bond.kappa,
                            sigma=cfg.price.sigma_issuer_daily, gamma=gc.gamma,
                            z=gc.z, fill=fill, T=gc.horizon_days,
                            gamma_term=gc.gamma_term)
    return BGOracle(params, n_t=gc.oracle_n_t)


def generate_split(cfg: SimConfig, gc: GenConfig, n_cusips: int, rng, split: str):
    """Generate serialized oracle episodes for a set of CUSIPs. Yields JSONL records."""
    records, gap_O, gap_R, gap_E = [], [], [], []
    for ci in range(n_cusips):
        bond = BondParams.sample(cfg, rng, gc.sector_intensity, gc.cusip_frac)
        oracle = _build_oracle(cfg, bond, gc)
        for ei in range(gc.episodes_per_cusip):
            ep = generate_episode(cfg, bond, gc.horizon_days, rng)
            if len(ep.steps) < 2:
                continue
            # oracle rollout = the training trajectory (labels)
            seed = int(rng.integers(1 << 30))
            res_o = run_episode(np.random.default_rng(seed), ep, OraclePolicy(oracle),
                                oracle, cfg, z=gc.z, sector_halflife=gc.sector_halflife)
            # random rollout on the SAME stream = optimality-gap floor
            res_r = run_episode(np.random.default_rng(seed), ep,
                                RandomPolicy(np.random.default_rng(seed + 1)),
                                oracle, cfg, z=gc.z, sector_halflife=gc.sector_halflife)
            header = (f"<EPISODE cusip={split}_{ci} horizon={gc.horizon_days} "
                      f"n_rfq={len(ep.steps)}>")
            text = serialize_episode(res_o["obs_records"], res_o["actions"],
                                     res_o["rewards"], res_o["q_path"], z=gc.z,
                                     header=header)
            records.append({
                "text": text + "\n<END>",
                "meta": {"split": split, "cusip_idx": ci, "episode_idx": ei,
                         "n_rfq": len(ep.steps),
                         "bond": {k: round(v, 4) for k, v in asdict(bond).items()},
                         "bg_objective_oracle": round(res_o["bg_objective"], 4),
                         "bg_objective_random": round(res_r["bg_objective"], 4)},
            })
            gap_O.append(res_o["bg_objective"]); gap_R.append(res_r["bg_objective"])
            gap_E.append(res_o["bg_objective"])  # oracle vs oracle == 0 (sanity)
    stats = {"split": split, "n_cusips": n_cusips, "n_episodes": len(records),
             "pooled_gap_oracle": round(pooled_gap(gap_E, gap_O, gap_R), 4),
             "pooled_gap_random": round(pooled_gap(gap_R, gap_O, gap_R), 4)}
    return records, stats


def run_pipeline(out_dir: str = "/mnt/user-data/outputs/sft_data",
                 cfg: SimConfig | None = None, gc: GenConfig | None = None):
    import os
    cfg = cfg or SimConfig()
    gc = gc or GenConfig()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(gc.seed)

    train, s_tr = generate_split(cfg, gc, gc.n_train_cusips, rng, "train")
    test, s_te = generate_split(cfg, gc, gc.n_test_cusips, rng, "test")

    for name, recs in [("train", train), ("test", test)]:
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    manifest = {"gen_config": asdict(gc), "train_stats": s_tr, "test_stats": s_te,
                "action_bins": list(__import__("ftp.serialize", fromlist=["ACTION_BINS"])
                                    .ACTION_BINS)}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=float)
    return manifest
