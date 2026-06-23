"""Evaluation harness: the headline optimality-gap result (single-CUSIP).

Rolls each policy on the SAME held-out episodes (fresh MMPP regime realizations on the ONE
fixed bond) and reports the pooled BG optimality gap:

    full-info oracle   = 0       (ceiling, by construction)
    SFT LLM            = ?        (the result -- how much of the gap it closes)
    ICL (few-shot)     = ?        (baseline -- in-context only, no fine-tuning)
    belief-optimal     = irreducible inference loss (best possible flow-only policy)
    random             = 1       (floor)

All policies are rolled on identical streams (same seed) so fills are common-random-number
matched. Stateful LLM policies are .reset() per episode.
"""
from __future__ import annotations

import numpy as np

from rfqsim.config import SimConfig
from rfqsim.mmpp import build_generator
from .belief import BeliefOptimalPolicy, MMPPBeliefFilter
from .env import generate_episode
from .pipeline import GenConfig, build_oracle, make_bond
from .rollout import (OraclePolicy, RandomPolicy, pooled_gap, run_episode)


def evaluate(cfg: SimConfig, gc: GenConfig, n_eval_episodes=120, llm_policy=None,
             icl_policy=None, seed=999):
    """Roll all policies on held-out regime realizations of the fixed bond; return the
    pooled-gap decomposition."""
    rng = np.random.default_rng(seed)
    Q = build_generator(cfg.mmpp)

    # ONE bond, ONE oracle (the same desk as in training).
    bond = make_bond(cfg, gc)
    oracle = build_oracle(cfg, bond, gc)
    lo = cfg.mmpp.lam_low_frac * bond.sector_intensity
    hi = cfg.mmpp.lam_high_frac * bond.sector_intensity
    lam_b = np.array([lo, lo, hi, hi]); lam_a = np.array([lo, hi, lo, hi])

    objs = {k: [] for k in ["oracle", "random", "belief", "sft", "icl"]}
    done = 0
    attempts = 0
    while done < n_eval_episodes and attempts < n_eval_episodes * 4:
        attempts += 1
        ep = generate_episode(cfg, bond, gc.horizon_days, rng)
        if len(ep.steps) < 2:
            continue
        s = int(rng.integers(1 << 30))

        def roll(pol):
            return run_episode(np.random.default_rng(s), ep, pol, oracle, cfg,
                               z=gc.z, sector_halflife=gc.sector_halflife)

        objs["oracle"].append(roll(OraclePolicy(oracle))["bg_objective"])
        objs["random"].append(
            roll(RandomPolicy(np.random.default_rng(s + 1)))["bg_objective"])
        bf = MMPPBeliefFilter(Q, lam_b, lam_a)
        objs["belief"].append(roll(BeliefOptimalPolicy(oracle, bf))["bg_objective"])
        if llm_policy is not None:
            llm_policy.reset()
            objs["sft"].append(roll(llm_policy)["bg_objective"])
        if icl_policy is not None:
            icl_policy.reset()
            objs["icl"].append(roll(icl_policy)["bg_objective"])
        done += 1

    O, R = objs["oracle"], objs["random"]
    out = {"full_info_oracle": 0.0,
           "random": round(pooled_gap(R, O, R), 4),
           "belief_optimal": round(pooled_gap(objs["belief"], O, R), 4)}
    if objs["sft"]:
        out["sft_llm"] = round(pooled_gap(objs["sft"], O, R), 4)
    if objs["icl"]:
        out["icl_fewshot"] = round(pooled_gap(objs["icl"], O, R), 4)
    out["n_episodes"] = len(O)
    return out


if __name__ == "__main__":
    import argparse
    import json

    from .llm_policy import (ICLPolicy, LLMPolicy, build_support_prefix,
                            load_sft_model)
    from .sft_data import load_jsonl

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="./ckpt/final")
    ap.add_argument("--data", default="./sft_data")
    ap.add_argument("--n_episodes", type=int, default=120)
    ap.add_argument("--with_icl", action="store_true")
    args = ap.parse_args()

    cfg, gc = SimConfig(), GenConfig()
    model, tok = load_sft_model(args.adapter)
    sft_pol = LLMPolicy(model, tok, z=gc.z)
    icl_pol = None
    if args.with_icl:
        support = build_support_prefix(load_jsonl(f"{args.data}/train.jsonl"), k=2)
        icl_pol = ICLPolicy(model, tok, support, z=gc.z)
    res = evaluate(cfg, gc, n_eval_episodes=args.n_episodes, llm_policy=sft_pol,
                   icl_policy=icl_pol)
    print(json.dumps(res, indent=2))
