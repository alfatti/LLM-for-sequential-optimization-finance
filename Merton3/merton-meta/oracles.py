"""Two-oracle mechanism test (D5) + analytic error predictions.

Observation model for the oracles: demo actions parsed FROM THE PROMPT TEXT,
a_i = pi* + eps_i (2dp-rounded), eps ~ N(0, sigma_label^2) iid, i = 1..M with
M = K_test * n_steps. The wealth-transition channel carries Fisher information
about pi* of gamma^2 sigma^2 dt per step vs 1/sigma_label^2 per step for the
label channel (ratio ~1e-3 at default parameters; computed and asserted in
proxy_validate.py), so the pre-registered oracles use labels only.

Oracle A (Bayes under the TRAINING prior): pi* ~ U[lo, hi] prior with Gaussian
likelihood => posterior is N(abar, s^2) truncated to [lo, hi], s^2 =
sigma_label^2 / M. Posterior mean has the standard truncated-normal closed form
    E[pi* | abar] = abar + s * (phi(alpha) - phi(beta)) / (Phi(beta) - Phi(alpha)),
    alpha = (lo - abar)/s,  beta = (hi - abar)/s.
This oracle SHRINKS toward [lo, hi]: it is the "learned-the-prior" reference.

Oracle B (prior-free sample mean): abar. Unbiased, extrapolates freely: the
"learned-the-algorithm" reference.

Which oracle the model tracks, per tier and per K_test, is the mechanism
readout (Raventos-style prior-vs-algorithm transition across the N_task sweep).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def sample_mean_oracle(demo_actions: np.ndarray) -> float:
    a = np.asarray(demo_actions, dtype=float)
    if a.size == 0:
        raise ValueError("no demo actions")
    return float(a.mean())


def bayes_posterior_mean(demo_actions: np.ndarray, sigma_label: float,
                         lo: float, hi: float) -> float:
    """Posterior mean of pi* under Uniform[lo,hi] prior.

    Closed form: posterior = N(abar, s^2) truncated to [lo, hi], mean
        abar + s * (phi(alpha) - phi(beta)) / (Phi(beta) - Phi(alpha)).
    Evaluated via scipy.stats.truncnorm.mean, which is numerically stable in
    the deep-tail regime (sample mean far outside the support; the naive
    Phi-difference cancels catastrophically there -- this matters for Tier-3
    sign-flip tasks). Non-finite fallback pins to the nearest edge.
    """
    a = np.asarray(demo_actions, dtype=float)
    m = a.size
    if m == 0:
        return 0.5 * (lo + hi)
    abar = float(a.mean())
    s = sigma_label / np.sqrt(m)
    alpha = (lo - abar) / s
    beta = (hi - abar) / s
    mean = float(stats.truncnorm.mean(alpha, beta, loc=abar, scale=s))
    if not np.isfinite(mean):
        return float(lo if abar < lo else hi)
    return float(np.clip(mean, lo, hi))


def bayes_posterior_mean_numeric(demo_actions: np.ndarray, sigma_label: float,
                                 lo: float, hi: float, n_grid: int = 200001) -> float:
    """Numeric-integration cross-check of the closed form (used in validation)."""
    a = np.asarray(demo_actions, dtype=float)
    h = (hi - lo) / n_grid
    theta = lo + (np.arange(n_grid) + 0.5) * h   # midpoint rule (kills O(h/2) bias)
    loglik = -0.5 * ((a[None, :] - theta[:, None]) ** 2).sum(axis=1) / sigma_label ** 2
    loglik -= loglik.max()
    w = np.exp(loglik)
    return float((theta * w).sum() / w.sum())


def predicted_bayes_rmse_interior(sigma_label: float, m: int) -> float:
    """Interior-task prediction (truncation not binding): RMSE ~ sigma_label/sqrt(M).
    Pre-registered slope of log RMSE^2 vs log K_test at fixed n_steps: -1."""
    return sigma_label / np.sqrt(m)


def transition_channel_info_ratio(gamma: float, sigma: float, dt: float,
                                  sigma_label: float) -> float:
    """Per-step Fisher information ratio (transition channel / label channel)
    about pi*: (gamma^2 sigma^2 dt) / (1 / sigma_label^2)."""
    return (gamma ** 2) * (sigma ** 2) * dt * (sigma_label ** 2)
