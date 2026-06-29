"""
Outcome analysis (Test S3): does the model carry a WORLD MODEL (the transition kernel),
or only a policy? We probe with COUNTERFACTUAL actions.

Merton dynamics give a clean, falsifiable signature. With u=A held at a counterfactual
value at state (S,W):
    E[R | W, A] = (A(mu-r) + rW - pen) dt / W ,   pen = spread*max(0, A-W)
=> conditional-mean return is LINEAR in A with slope (mu-r)dt/W, and in the SHARP regime
   it KINKS at A=W (slope drops to (mu-r-spread)dt/W as borrowing cost switches on).

A policy-only learner has no reason to predict returns correctly for OFF-POLICY actions.
A dynamics-aware model reproduces the slope (and, in sharp, the kink). Read-out is in
prediction space (return-prediction), so it is NOT subject to the flatness limitation.

The model is abstracted as predict_return(prefix_text) -> float, so mock models and a
real Llama (see llm_return_predictor in eval_llm-style wrapper) both plug in. This file
is CPU-validated with mock models; point it at Llama on a GPU box.
"""
import numpy as np
from merton_serialize import fmt, SW_DECIMALS, A_DECIMALS, R_DECIMALS

DT = 1.0/10

# ----------------------------- theory -----------------------------
def theoretical_return(mu, sigma, r, W, A, spread):
    pen = spread*max(0.0, A - W)
    return (A*(mu - r) + r*W - pen)*DT / W

def theoretical_slopes(mu, sigma, r, W, spread):
    """Return (slope_below_kink, slope_above_kink). Kink at A=W (sharp only)."""
    below = (mu - r)*DT / W
    above = (mu - r - spread)*DT / W
    return below, above

# ------------------------- probe + analysis -----------------------
def probe_prefix(context, S, W, A, sep=" <SEP> "):
    """Prefix ending at '<R>' so the model predicts the step return for counterfactual A."""
    step = (f"<S> {fmt(S,SW_DECIMALS)} <W> {fmt(W,SW_DECIMALS)} "
            f"<A> {fmt(A,A_DECIMALS)} <R>")
    return context + sep + step

def return_response(predict_return, context, S, W, A_grid):
    """Greedy-predict R-hat for each counterfactual action A in A_grid."""
    out = []
    for A in A_grid:
        r = predict_return(probe_prefix(context, S, W, A))
        out.append(np.nan if r is None else r)
    return np.array(out)

def analyze_task(predict_return, task, S=1.0, W=1.0, A_grid=None, n_pts=13):
    m = task["market"]; mu, sigma, r = m["mu"], m["sigma"], m["r"]
    spread = task["spread"]
    if A_grid is None:
        A_grid = np.linspace(0.2, 2.0, n_pts)
    Rhat = return_response(predict_return, task["support_text"], S, W, A_grid)
    ok = ~np.isnan(Rhat)
    Ag, Rg = A_grid[ok], Rhat[ok]
    if Ag.size < 3:
        return None
    # overall slope (single line) and fit quality
    slope, intercept = np.polyfit(Ag, Rg, 1)
    Rtrue = np.array([theoretical_return(mu, sigma, r, W, A, spread) for A in Ag])
    ss_res = np.sum((Rg - Rtrue)**2); ss_tot = np.sum((Rg - Rg.mean())**2) + 1e-12
    slope_theory_below, slope_theory_above = theoretical_slopes(mu, sigma, r, W, spread)
    # kink test (sharp): slope below vs above A=W
    below = Ag <= W; above = Ag > W
    sb = np.polyfit(Ag[below], Rg[below], 1)[0] if below.sum() >= 2 else np.nan
    sa = np.polyfit(Ag[above], Rg[above], 1)[0] if above.sum() >= 2 else np.nan
    return dict(slope=slope, intercept=intercept,
                slope_theory=slope_theory_below,
                slope_err=abs(slope - slope_theory_below),
                pred_vs_theory_rmse=float(np.sqrt(np.mean((Rg - Rtrue)**2))),
                slope_below=sb, slope_above=sa,
                kink_drop=(sb - sa) if (np.isfinite(sb) and np.isfinite(sa)) else np.nan,
                kink_drop_theory=(slope_theory_below - slope_theory_above))

def analyze_eval_set(predict_return, tasks, **kw):
    recs = [analyze_task(predict_return, t, **kw) for t in tasks]
    recs = [r for r in recs if r is not None]
    agg = lambda k: float(np.nanmean([r[k] for r in recs]))
    return dict(n=len(recs),
                mean_slope=agg("slope"), mean_slope_below=agg("slope_below"),
                mean_slope_theory=agg("slope_theory"),
                mean_slope_err=agg("slope_err"),
                mean_pred_rmse=agg("pred_vs_theory_rmse"),
                mean_kink_drop=agg("kink_drop"), mean_kink_drop_theory=agg("kink_drop_theory"),
                records=recs)
