"""
Fixed-MDP analysis (CPU). With ONE MDP there is a single c*; 'recovery' is how well a
method infers that one c* from the support rollouts, and how much utility it loses.

Aggregation is over EVAL INSTANCES (which differ only by support-rollout noise), not over
tasks. This module runs the closed-form OLS stand-in -> the in-context SHORTCUT baseline
that the real ICL / SFT runs (fixed_mdp/eval_llm.py) are compared against.

Usage:
  python -m fixed_mdp.analysis
"""
import json, numpy as np
from core.metrics import estimator_ols, crn_gap_for_task
from fixed_mdp.config import REGIME_SPREAD

def recovery_and_gap(eval_path, estimator_fn, n_gap_paths=6000):
    inst = [json.loads(l) for l in open(eval_path)]
    c_star = inst[0]["c_star"]
    c_hats, gaps = [], []
    for t in inst:
        ch = estimator_fn(t["support_text"])
        if np.isnan(ch): continue
        c_hats.append(ch)
        gaps.append(abs(crn_gap_for_task(t, ch, n_gap_paths, seed=t["instance_id"])))
    c_hats = np.array(c_hats); gaps = np.array(gaps)
    wins = np.clip(gaps, None, np.percentile(gaps, 95))
    return dict(c_star=c_star, n=len(c_hats),
                c_hat_mean=float(c_hats.mean()), c_hat_std=float(c_hats.std(ddof=1)),
                coeff_rmse=float(np.sqrt(np.mean((c_hats - c_star)**2))),
                coeff_bias=float(c_hats.mean() - c_star),
                gap_median=float(np.median(gaps)), gap_wins_mean=float(np.mean(wins)))

if __name__ == "__main__":
    print("Fixed-MDP stand-in baseline (OLS = in-context shortcut). This is the NULL that")
    print("the real ICL / SFT models must beat (lower coeff RMSE) to show in-context learning.\n")
    for regime in ["benign", "sharp"]:
        path = f"data/fixed_mdp/{regime}_eval.jsonl"
        try:
            r = recovery_and_gap(path, estimator_ols)
        except FileNotFoundError:
            print(f"{regime}: (no data -- run fixed_mdp.gen_data first)"); continue
        print(f"[{regime}] c*={r['c_star']:.4f}  n={r['n']}")
        print(f"   coeff: mean(c_hat)={r['c_hat_mean']:.4f}  RMSE={r['coeff_rmse']:.4f}  "
              f"bias={r['coeff_bias']:+.4f}  spread(std)={r['c_hat_std']:.4f}")
        print(f"   gap:   median={100*r['gap_median']:.3f}%  wins-mean={100*r['gap_wins_mean']:.3f}%")
