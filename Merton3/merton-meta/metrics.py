"""Metrics and summaries (D6).

Primary: RMSE(pi_hat, pi*) with two-sided Student-t CIs (over tasks for Q4,
over CRN draws for Q1/Q2, matching the paper's CI convention). Reported as
RMSE with CI endpoints obtained by the monotone transform of the MSE CI.

Mechanism: RMSE(pi_hat, oracle) for the Bayes-under-training-prior oracle and
the prior-free sample-mean oracle, per (tier, ktest) -- the "which oracle does
the model track" readout.

Shrinkage: OLS fit pi_hat = alpha + beta * pi* per tier; beta < 1 with the
Bayes oracle's beta as reference indicates prior-shrinkage.

Secondary: exact log-CE gap (env.log_ce_gap) aggregated the same way.

Random baseline (analytic, uniform over the action grid): for grid U[lo,hi],
E[(A - p)^2] = Var + (mean - p)^2 = (hi-lo)^2/12 + ((lo+hi)/2 - p)^2, and the
per-rollout log-CE gap is 0.5*gamma*sigma^2*dt*T*E[(A-p)^2].
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import stats


def mse_with_ci(sq_errors: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    se2 = np.asarray(sq_errors, dtype=float)
    n = se2.size
    mse = float(se2.mean())
    out = {"n": n, "mse": mse, "rmse": math.sqrt(max(mse, 0.0))}
    if n >= 2:
        sem = se2.std(ddof=1) / math.sqrt(n)
        tcrit = stats.t.ppf(0.5 + conf / 2, df=n - 1)
        lo, hi = max(mse - tcrit * sem, 0.0), mse + tcrit * sem
        out.update({"mse_ci_lo": lo, "mse_ci_hi": hi,
                    "rmse_ci_lo": math.sqrt(lo), "rmse_ci_hi": math.sqrt(hi)})
    return out


def mean_with_ci(values: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    n = v.size
    out = {"n": n, "mean": float(v.mean())}
    if n >= 2:
        sem = v.std(ddof=1) / math.sqrt(n)
        tcrit = stats.t.ppf(0.5 + conf / 2, df=n - 1)
        out.update({"ci_lo": float(v.mean() - tcrit * sem),
                    "ci_hi": float(v.mean() + tcrit * sem)})
    return out


def shrinkage_fit(pistars: np.ndarray, pihats: np.ndarray) -> Dict[str, float]:
    x = np.asarray(pistars, float)
    y = np.asarray(pihats, float)
    if x.size < 3 or np.allclose(x.var(), 0.0):
        return {"n": int(x.size), "beta": float("nan"), "alpha": float("nan")}
    res = stats.linregress(x, y)
    return {"n": int(x.size), "beta": float(res.slope), "alpha": float(res.intercept),
            "beta_se": float(res.stderr), "r2": float(res.rvalue ** 2)}


def loglog_slope(ks: np.ndarray, mses: np.ndarray) -> Dict[str, float]:
    """Fit log(MSE) = c + slope * log(K). Pre-registered: slope ~ -1 for the
    in-context term at fixed n_steps."""
    ks = np.asarray(ks, float)
    mses = np.asarray(mses, float)
    ok = (ks > 0) & (mses > 0)
    if ok.sum() < 2:
        return {"slope": float("nan")}
    res = stats.linregress(np.log(ks[ok]), np.log(mses[ok]))
    return {"slope": float(res.slope), "slope_se": float(res.stderr),
            "intercept": float(res.intercept), "r2": float(res.rvalue ** 2)}


def random_baseline_mse(grid_lo: float, grid_hi: float, pistar: float) -> float:
    mid = 0.5 * (grid_lo + grid_hi)
    return (grid_hi - grid_lo) ** 2 / 12.0 + (mid - pistar) ** 2


def summarize_results(rows: Iterable[dict], grid_lo: float, grid_hi: float,
                      gamma: float, sigma: float, dt: float, n_steps: int) -> dict:
    """Aggregate eval-result rows (from eval_llm.py) into per-(exp, tier, ktest)
    summaries plus slope fits and shrinkage regressions.

    Expected row fields: exp, tier, ktest, pistar, pi_hat, bayes_mean,
    sample_mean, log_ce_gap.
    """
    groups: Dict[Tuple[str, str, int], List[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["exp"], r["tier"], int(r["ktest"]))].append(r)

    summary: dict = {"groups": {}, "slopes": {}, "shrinkage": {}}
    for (exp, tier, k), rs in sorted(groups.items()):
        p = np.array([r["pistar"] for r in rs])
        ph = np.array([r["pi_hat"] for r in rs])
        g = {
            "exp": exp, "tier": tier, "ktest": k,
            "rmse_true": mse_with_ci((ph - p) ** 2),
            "log_ce_gap": mean_with_ci(np.array([r["log_ce_gap"] for r in rs])),
            "random_baseline_mse": float(np.mean(
                [random_baseline_mse(grid_lo, grid_hi, pi) for pi in p])),
            "random_baseline_log_ce_gap": float(
                0.5 * gamma * sigma ** 2 * dt * n_steps * np.mean(
                    [random_baseline_mse(grid_lo, grid_hi, pi) for pi in p])),
        }
        if rs[0].get("bayes_mean") is not None:
            b = np.array([r["bayes_mean"] for r in rs])
            s = np.array([r["sample_mean"] for r in rs])
            g["rmse_to_bayes"] = mse_with_ci((ph - b) ** 2)
            g["rmse_to_samplemean"] = mse_with_ci((ph - s) ** 2)
            g["bayes_rmse_true"] = mse_with_ci((b - p) ** 2)
            g["samplemean_rmse_true"] = mse_with_ci((s - p) ** 2)
            g["tracks"] = ("bayes" if g["rmse_to_bayes"]["mse"] <
                           g["rmse_to_samplemean"]["mse"] else "sample_mean")
        summary["groups"][f"{exp}|{tier}|k{k:02d}"] = g

    # log-log slope of MSE vs ktest, per (exp, tier) with >= 2 ktest values
    bytier: Dict[Tuple[str, str], List[Tuple[int, float]]] = defaultdict(list)
    for (exp, tier, k), rs in groups.items():
        if k > 0:
            p = np.array([r["pistar"] for r in rs])
            ph = np.array([r["pi_hat"] for r in rs])
            bytier[(exp, tier)].append((k, float(np.mean((ph - p) ** 2))))
    for (exp, tier), pts in bytier.items():
        if len(pts) >= 2:
            ks, ms = zip(*sorted(pts))
            summary["slopes"][f"{exp}|{tier}"] = loglog_slope(np.array(ks), np.array(ms))

    # shrinkage regressions per (exp, tier, ktest) where pi* varies (Q4 tiers)
    for (exp, tier, k), rs in groups.items():
        p = np.array([r["pistar"] for r in rs])
        if np.ptp(p) > 1e-9:
            ph = np.array([r["pi_hat"] for r in rs])
            fit = shrinkage_fit(p, ph)
            if rs[0].get("bayes_mean") is not None:
                fit["beta_bayes"] = shrinkage_fit(
                    p, np.array([r["bayes_mean"] for r in rs]))["beta"]
            summary["shrinkage"][f"{exp}|{tier}|k{k:02d}"] = fit
    return summary


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
