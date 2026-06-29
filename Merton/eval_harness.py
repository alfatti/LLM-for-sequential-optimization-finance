"""
Evaluation harness for the Merton replication. Two metrics:
  (1) Coefficient-recovery MSE  -- PRIMARY (no flatness problem; what claim 2 is about):
        the in-context inference of the hidden scalar c from the support demonstrations.
  (2) CRN optimality gap        -- SECONDARY (reported in both regimes):
        realized utility loss of acting with the inferred c vs the oracle c*, evaluated
        with common random numbers (paired) for low variance.

The 'policy' is abstracted as an estimator_fn(support_text) -> c_hat. The closed-form
OLS estimator is the STAND-IN used to validate this harness without an LLM. The LLM
plugs in by replacing estimator_fn with a routine that queries the model and reads off
its implied coefficient (action / wealth), or by acting step-by-step (hook provided).
"""
import json, numpy as np
from merton_serialize import parse_first_action
from merton import crra_utility
from rollout import rollout_paths

import re
_pair_re = re.compile(r"<W>\s*(-?\d+\.\d+)\s*<A>\s*(-?\d+\.\d+)")

def parse_support_pairs(support_text):
    """Recover (wealth, action) demonstration pairs from a serialized support context."""
    Ws, As = [], []
    for m in _pair_re.finditer(support_text):
        Ws.append(float(m.group(1))); As.append(float(m.group(2)))
    return np.array(Ws), np.array(As)

def estimator_ols(support_text):
    """In-context inference of c by OLS of demonstrated action on wealth (the stand-in)."""
    W, A = parse_support_pairs(support_text)
    if W.size == 0:
        return np.nan
    return float(np.sum(A*W)/np.sum(W*W))

def crn_gap_for_task(task, c_hat, n_paths=8000, seed=0):
    """Paired CRN optimality gap: oracle c* vs c_hat policy on identical Brownian paths."""
    m = task["market"]; mu, sigma, r, rho = m["mu"], m["sigma"], m["r"], m["rho"]
    spread = task["spread"]; c_star = task["c_star"]
    dt = 1.0/10; rng = np.random.default_rng(seed)
    dW = np.sqrt(dt)*rng.standard_normal((n_paths, 10))
    _, Wo, _, _ = rollout_paths(mu, sigma, r, c_star, 1.0, 1.0, 10, n_paths, rng, spread, dW=dW)
    _, Wh, _, _ = rollout_paths(mu, sigma, r, c_hat,  1.0, 1.0, 10, n_paths, rng, spread, dW=dW)
    Uo = crra_utility(Wo[:, -1], rho); Uh = crra_utility(Wh[:, -1], rho)
    return float((Uo - Uh).mean()/abs(Uo.mean()))

def evaluate(eval_path, estimator_fn, n_gap_paths=8000):
    tasks = [json.loads(l) for l in open(eval_path)]
    sq_err, gaps, c_hats, c_stars = [], [], [], []
    for t in tasks:
        ch = estimator_fn(t["support_text"]); cs = t["c_star"]
        sq_err.append((ch - cs)**2)
        gaps.append(abs(crn_gap_for_task(t, ch, n_gap_paths, seed=t["task_id"])))
        c_hats.append(ch); c_stars.append(cs)
    gaps = np.array(gaps)
    wins = np.clip(gaps, None, np.percentile(gaps, 95))   # winsorize tail (CRRA U=-1/x blowups)
    return dict(mse=float(np.mean(sq_err)),
                rmse=float(np.sqrt(np.mean(sq_err))),
                gap_median=float(np.median(gaps)),
                gap_wins_mean=float(np.mean(wins)),
                n=len(tasks), c_hats=c_hats, c_stars=c_stars)

if __name__ == "__main__":
    for regime in ["benign", "sharp"]:
        for split in ["id", "ood"]:
            path = f"data_{regime}_eval_{split}.jsonl"
            try:
                res = evaluate(path, estimator_ols, n_gap_paths=6000)
            except FileNotFoundError:
                print(f"{regime}/{split}: (no file)"); continue
            print(f"{regime:6s}/{split:3s}  n={res['n']:3d}  "
                  f"coeff RMSE={res['rmse']:.4f}  "
                  f"gap median={100*res['gap_median']:.3f}%  "
                  f"gap wins-mean={100*res['gap_wins_mean']:.3f}%")
