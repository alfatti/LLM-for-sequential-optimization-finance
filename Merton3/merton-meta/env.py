"""Merton environment: dynamics, oracle policy, and EXACT certainty-equivalent gap.

Dynamics (exact per-step GBM under a piecewise-constant control a_t):
    x_{t+1} = x_t + (r + a_t (mu - r) - 0.5 a_t^2 sigma^2) dt + a_t sigma sqrt(dt) Z_t
where x_t = log(W_t / W_0) and Z_t ~ N(0,1) iid. This "log-Euler" scheme is the
EXACT solution of the wealth SDE when the fraction is held constant within the
step, and it guarantees W_t > 0 for any a in R.

Oracle. With CRRA utility U(W) = W^{1-gamma}/(1-gamma) and the scheme above,
per-step log-CE contributions ADD, and the per-step optimizer is exactly the
Merton fraction. Derivation: for one step of length dt under constant a,
    log E[W_1^{1-gamma}] = (1-gamma)(r + a(mu-r) - 0.5 a^2 sigma^2) dt
                           + 0.5 (1-gamma)^2 a^2 sigma^2 dt,
so log CE = (1/(1-gamma)) log E[W^{1-gamma}]
          = [ r + a(mu-r) - 0.5 gamma sigma^2 a^2 ] dt  =:  g(a) dt.
g is strictly concave in a with argmax a* = (mu - r) / (gamma sigma^2) = pi*,
independent of the state -- so the constant policy pi* is exactly optimal for
the DISCRETE-TIME problem as well, and:

    EXACT CE GAP identity (used as the secondary metric, no Monte Carlo):
        log CE(pi*) - log CE({a_t}) = sum_t [g(pi*) - g(a_t)] dt
                                    = 0.5 gamma sigma^2 dt * sum_t (a_t - pi*)^2.
    Relative CE gap = 1 - exp(-(log-CE gap)).

This identity is verified against Monte Carlo in scripts/proxy_validate.py.
It also quantifies the known flatness: |a - pi*| = 0.10 over a 1-year horizon
costs 0.5*2*0.04*1*0.01 = 4.0e-4 in log-CE, i.e. ~0.04% of wealth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import MarketConfig


@dataclass(frozen=True)
class Task:
    """A single Merton task, identified by pi* (D1: only mu varies)."""
    task_id: str
    tier: str          # "train" | "tier1" | "tier2_low" | "tier2_high" | "tier3"
    pistar: float

    def mu(self, m: MarketConfig) -> float:
        return m.r + self.pistar * m.gamma * m.sigma ** 2


def pistar_to_mu(pistar: float, m: MarketConfig) -> float:
    return m.r + pistar * m.gamma * m.sigma ** 2


def step_log_wealth(x: np.ndarray, a: np.ndarray, z: np.ndarray,
                    mu: float, m: MarketConfig) -> np.ndarray:
    """One exact log-Euler step. Vectorized over leading dims."""
    drift = (m.r + a * (mu - m.r) - 0.5 * a * a * m.sigma ** 2) * m.dt
    diff = a * m.sigma * np.sqrt(m.dt) * z
    return x + drift + diff


def rollout(actions: Sequence[float], z: np.ndarray, mu: float,
            m: MarketConfig) -> np.ndarray:
    """Roll a single trajectory. Returns log-wealth states x_0..x_T (len T+1),
    x_0 = 0. `actions` has length T = m.n_steps; z has shape (T,)."""
    actions = np.asarray(actions, dtype=float)
    assert actions.shape == (m.n_steps,) and z.shape == (m.n_steps,)
    xs = np.zeros(m.n_steps + 1)
    for t in range(m.n_steps):
        xs[t + 1] = step_log_wealth(xs[t], actions[t], z[t], mu, m)
    return xs


def ce_growth(a: np.ndarray, mu: float, m: MarketConfig) -> np.ndarray:
    """Per-year certainty-equivalent growth rate g(a) (see module docstring)."""
    a = np.asarray(a, dtype=float)
    return m.r + a * (mu - m.r) - 0.5 * m.gamma * m.sigma ** 2 * a * a


def log_ce_gap(actions: Sequence[float], pistar: float, m: MarketConfig) -> float:
    """EXACT log-CE gap of an executed action sequence vs the optimal constant.
    Equals 0.5 * gamma * sigma^2 * dt * sum_t (a_t - pi*)^2 (derivation above)."""
    a = np.asarray(actions, dtype=float)
    return float(0.5 * m.gamma * m.sigma ** 2 * m.dt * np.sum((a - pistar) ** 2))


def relative_ce_gap(actions: Sequence[float], pistar: float, m: MarketConfig) -> float:
    return float(1.0 - np.exp(-log_ce_gap(actions, pistar, m)))


def crra_utility(w: np.ndarray, gamma: float) -> np.ndarray:
    w = np.maximum(np.asarray(w, dtype=float), 1e-300)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(w)
    return w ** (1.0 - gamma) / (1.0 - gamma)


def mc_certainty_equivalent(a_const: float, mu: float, m: MarketConfig,
                            n_paths: int, rng: np.random.Generator) -> float:
    """Monte-Carlo CE of terminal wealth under a constant fraction (validation
    utility only; the experiment's CE-gap metric is analytic)."""
    z = rng.standard_normal((n_paths, m.n_steps))
    drift = (m.r + a_const * (mu - m.r) - 0.5 * a_const ** 2 * m.sigma ** 2) * m.dt
    x_T = m.n_steps * drift + a_const * m.sigma * np.sqrt(m.dt) * z.sum(axis=1)
    eu = crra_utility(np.exp(x_T), m.gamma).mean()
    if abs(m.gamma - 1.0) < 1e-12:
        return float(np.exp(eu))
    return float(((1.0 - m.gamma) * eu) ** (1.0 / (1.0 - m.gamma)))
