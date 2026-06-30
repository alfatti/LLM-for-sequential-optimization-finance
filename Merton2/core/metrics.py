"""
Shared evaluation metrics, used by both the fixed-MDP and many-tasks experiments.
  - parse_support_pairs : recover (wealth, action) demo pairs from serialized context
  - estimator_ols       : closed-form in-context coefficient estimate (the SHORTCUT/stand-in)
  - crn_gap_for_task    : common-random-numbers optimality gap for one (market, c*, spread)
"""
import re, numpy as np
from core.dynamics import crra_utility
from core.rollout import rollout_paths

_pair_re = re.compile(r"<W>\s*(-?\d+\.\d+)\s*<A>\s*(-?\d+\.\d+)")

def parse_support_pairs(support_text):
    Ws, As = [], []
    for m in _pair_re.finditer(support_text):
        Ws.append(float(m.group(1))); As.append(float(m.group(2)))
    return np.array(Ws), np.array(As)

def estimator_ols(support_text):
    W, A = parse_support_pairs(support_text)
    if W.size == 0:
        return np.nan
    return float(np.sum(A*W)/np.sum(W*W))

def crn_gap(mu, sigma, r, rho, spread, c_star, c_hat, n_paths=8000, seed=0,
            x0=1.0, T=1.0, n_steps=10):
    """Paired CRN optimality gap: oracle c* vs c_hat on identical Brownian paths."""
    dt = T/n_steps
    rng = np.random.default_rng(seed)
    dW = np.sqrt(dt)*rng.standard_normal((n_paths, n_steps))
    _, Wo, _, _ = rollout_paths(mu, sigma, r, c_star, x0, T, n_steps, n_paths, rng, spread, dW=dW)
    _, Wh, _, _ = rollout_paths(mu, sigma, r, c_hat,  x0, T, n_steps, n_paths, rng, spread, dW=dW)
    Uo = crra_utility(Wo[:, -1], rho); Uh = crra_utility(Wh[:, -1], rho)
    return float((Uo - Uh).mean()/abs(Uo.mean()))

def crn_gap_for_task(task, c_hat, n_paths=8000, seed=0):
    m = task["market"]
    return crn_gap(m["mu"], m["sigma"], m["r"], m["rho"], task["spread"],
                   task["c_star"], c_hat, n_paths=n_paths, seed=seed)
