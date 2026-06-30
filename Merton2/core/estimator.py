"""
Closed-form in-context estimators of the Merton coefficient c (u* = c x),
the analytic stand-in for what a trained (linear) attention layer computes.

Paper 1, eq. (5):   Qhat(x_q) = x_q^T  Gamma^{-1} ( 1/M  sum_i y_i x_i ),
with Gamma = (1 + 1/N) Lambda + (tr Lambda / N) I,  Lambda = E[x x^T].

In the Merton task the feature is scalar (d=1): x_i = wealth at a demo step,
y_i = recorded oracle action (= c x_i, possibly with multiplicative label noise).
So Lambda = E[x^2] =: S, and the coefficient estimators are:

  OLS (exact recovery if noiseless):  c_ols  = (sum y_i x_i) / (sum x_i^2)
  Shrinkage predictor (eq. 5):        c_shr  = (1/M sum y_i x_i) / [ S (1 + 2/N) ]

The shrinkage predictor carries the O(1/N) bias whose square is Paper 1's
training-length term; OLS does not, but is exact only when labels are noiseless.
"""

import numpy as np
from core.dynamics import Market, crra_coeff, simulate_wealth


def build_support_pairs(mkt, rho, x0, T, n_steps, k_test, rng, label_noise=0.0):
    """Roll k_test optimal-policy trajectories; return stacked (x_i, y_i) demo pairs."""
    c = crra_coeff(mkt, rho)
    pol = lambda t, x: c * x
    xs, ys = [], []
    for _ in range(k_test):
        tr = simulate_wealth(mkt, pol, x0, T, n_steps, rng, label_noise=label_noise)
        # demo pairs are (wealth at step k, recorded action at step k)
        xs.append(tr["x"][:-1])
        ys.append(tr["u_recorded"])
    return np.concatenate(xs), np.concatenate(ys)


def estimate_ols(xs, ys):
    return float(np.sum(ys * xs) / np.sum(xs * xs))


def estimate_shrinkage(xs, ys, N, S):
    M = xs.size
    moment = np.sum(ys * xs) / M
    Gamma = S * (1.0 + 2.0 / N)       # d=1 specialization of eq.(4)
    return float(moment / Gamma)


def feature_second_moment(mkt, rho, x0, T, n_steps, rng, n_traj=400):
    """Estimate S = E[x^2] over the support-state distribution (process-noise driven)."""
    xs, _ = build_support_pairs(mkt, rho, x0, T, n_steps, n_traj, rng, label_noise=0.0)
    return float(np.mean(xs**2))
