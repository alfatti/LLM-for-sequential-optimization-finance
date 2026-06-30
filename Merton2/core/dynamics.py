"""
Core Merton-problem machinery, set up as in Fathi (Paper 2), recast as a
sequential-decision task in the style of Zhang/Aghaei/Saghafian (Paper 1).

Two oracle controls:
  - Exponential utility U(x) = -exp(-gamma x):
        u*(t) = (lambda / (gamma sigma)) * exp(-r (T - t))      [Paper 2, eq. 26]
        => STATE-INDEPENDENT (depends on t and market params only).
  - CRRA power utility U(x) = x^alpha / alpha, RRA rho = 1 - alpha:
        u*(t,x) = c * x,  c = (mu - r) / (sigma^2 * rho)
        => STATE-DEPENDENT (the Merton fraction). The per-task hidden scalar
           the model must infer from context is c.

Here u_t is the *dollar amount* held in the risky asset (Paper 2's convention),
and the discounted wealth SDE (Paper 2, eq. 16) is:
    dX_t = (u_t (mu - r) + r X_t) dt + u_t sigma dW_t.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Market:
    mu: float       # drift of risky asset
    sigma: float    # volatility
    r: float        # risk-free rate

    @property
    def lam(self):  # Sharpe-like quantity lambda = (mu - r)/sigma
        return (self.mu - self.r) / self.sigma


# ----------------------------- oracle controls -----------------------------

def crra_coeff(mkt: Market, rho: float) -> float:
    """Optimal Merton coefficient c such that u*(t,x) = c x. rho = relative risk aversion."""
    return (mkt.mu - mkt.r) / (mkt.sigma**2 * rho)


def crra_action(mkt: Market, rho: float, x: float) -> float:
    return crra_coeff(mkt, rho) * x


def exp_action(mkt: Market, gamma: float, t: float, T: float) -> float:
    """Paper 2 eq.(26): state-independent, depends only on t and market params."""
    return (mkt.lam / (gamma * mkt.sigma)) * np.exp(-mkt.r * (T - t))


# ------------------------------- dynamics ----------------------------------

def simulate_wealth(mkt: Market, policy, x0, T, n_steps, rng, *, label_noise=0.0):
    """
    Roll a single wealth trajectory under `policy`.

    policy: callable (t, x) -> dollar amount u in risky asset.
    label_noise: multiplicative Gaussian noise applied to the *recorded* action
                 (demonstration noise); does NOT affect the realized dynamics
                 unless you pass the noised action back in. We keep realized
                 dynamics on the clean action and record the noised one, matching
                 "noisy observation of the oracle's intended action".

    Returns dict of arrays: t, x (wealth), u_clean, u_recorded.
    """
    dt = T / n_steps
    ts = np.linspace(0.0, T, n_steps + 1)
    x = np.empty(n_steps + 1)
    u_clean = np.empty(n_steps)
    u_rec = np.empty(n_steps)
    x[0] = x0
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        t = ts[k]
        u = policy(t, x[k])
        u_clean[k] = u
        if label_noise > 0.0:
            u_rec[k] = u * (1.0 + label_noise * rng.standard_normal())
        else:
            u_rec[k] = u
        dW = sqrt_dt * rng.standard_normal()
        x[k + 1] = x[k] + (u * (mkt.mu - mkt.r) + mkt.r * x[k]) * dt + u * mkt.sigma * dW
    return {"t": ts, "x": x, "u_clean": u_clean, "u_recorded": u_rec}


# ------------------------------- utilities ---------------------------------

def crra_utility(x, rho):
    alpha = 1.0 - rho
    if abs(alpha) < 1e-12:          # log utility limit
        return np.log(np.maximum(x, 1e-12))
    return np.sign(x) * np.abs(x) ** alpha / alpha if alpha > 0 else \
        -(np.maximum(x, 1e-12) ** alpha) / abs(alpha)


def exp_utility(x, gamma):
    return -np.exp(-gamma * x)


# --------------------------- optimality gap --------------------------------

def expected_utility(mkt, policy, x0, T, n_steps, util_fn, n_mc, rng, label_noise=0.0):
    """Monte-Carlo E[U(x_T)] under a policy."""
    vals = np.empty(n_mc)
    for i in range(n_mc):
        traj = simulate_wealth(mkt, policy, x0, T, n_steps, rng, label_noise=label_noise)
        vals[i] = util_fn(traj["x"][-1])
    return vals.mean(), vals.std(ddof=1) / np.sqrt(n_mc)


def certainty_equivalent_crra(EU, rho):
    """Map E[U] back to a wealth-equivalent so gaps are in interpretable units."""
    alpha = 1.0 - rho
    if alpha > 0:
        return (alpha * EU) ** (1.0 / alpha)
    return np.nan
