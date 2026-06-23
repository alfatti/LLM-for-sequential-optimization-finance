"""Belief filter and belief-optimal policy: the honest benchmark for a flow-only agent.

The full-information BGOracle quotes against the *true* MMPP regime, which a real desk
(and our LLM) cannot observe. The best ANY flow-only policy can do is quote against the
Bayesian posterior over regimes given the observed RFQ stream. That posterior is exactly
BG Eq. (2): for observed RFQ times t_1<...<t_n with sides s_1..s_n,

    pi_t propto pi_0' [ prod_r exp((Q - Lam~_b - Lam~_a)(t_r - t_{r-1})) Lam~_{s_r} ]
                     * exp((Q - Lam~_b - Lam~_a)(t - t_n)) e_state

where Lam~_b = diag(lam_b), Lam~_a = diag(lam_a), and Lam~_{s} selects the side that
fired. Between events the no-event operator exp((Q - Lam~_b - Lam~_a) dt) propagates and
down-weights; at each event we multiply by the intensity of the side that arrived.

The belief-optimal quote mixes the per-state value functions by the posterior. Because the
BG quadratic approximation makes theta quadratic in q with per-state (A,B,C), the
posterior-expected value function is itself quadratic with coefficients
(E_pi[A], E_pi[B], E_pi[C]), so the exact quote map applies to the mixed coefficients.

Scoring the LLM against THIS policy (not the full-info oracle) separates:
  gap(LLM vs full-info)      = inference loss + learning loss
  gap(belief-opt vs full-info) = irreducible inference loss
  difference                  = what SFT can actually close.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .oracle import BGOracle, N_STATES


class MMPPBeliefFilter:
    """Online exact posterior over MMPP states from observed RFQ (time, side) events."""

    def __init__(self, Q, lam_b, lam_a, pi0=None):
        self.Q = np.asarray(Q, float)
        self.lam_b = np.asarray(lam_b, float)
        self.lam_a = np.asarray(lam_a, float)
        self.A_noevent = self.Q - np.diag(self.lam_b) - np.diag(self.lam_a)
        self.pi = (np.ones(N_STATES) / N_STATES) if pi0 is None else np.asarray(pi0, float)
        self.pi = self.pi / self.pi.sum()
        self.t = 0.0
        self._expm_cache = {}

    def _prop(self, dt):
        key = round(dt, 9)
        M = self._expm_cache.get(key)
        if M is None:
            M = expm(self.A_noevent * dt)
            self._expm_cache[key] = M
        return M

    def update(self, t, side):
        """Advance belief to time t and condition on an RFQ at `side` (0=bid,1=ask)."""
        dt = max(t - self.t, 0.0)
        v = self.pi @ self._prop(dt)              # propagate through no-event interval
        lam = self.lam_b if side == 0 else self.lam_a
        v = v * lam                               # multiply by intensity of the side seen
        s = v.sum()
        self.pi = v / s if s > 0 else np.ones(N_STATES) / N_STATES
        self.t = t
        return self.pi

    def propagate_to(self, t):
        """Belief at time t with no new event (for querying between/after events)."""
        dt = max(t - self.t, 0.0)
        v = self.pi @ self._prop(dt)
        s = v.sum()
        return v / s if s > 0 else self.pi


class BeliefOptimalPolicy:
    """Quotes against the posterior-mixed BG value function (best flow-only policy)."""

    def __init__(self, oracle: BGOracle, filt: MMPPBeliefFilter):
        self.o = oracle
        self.f = filt

    def __call__(self, step, q, t):
        from .serialize import quantize_offset
        # condition belief on the RFQ we are about to answer (its side is observable)
        pi = self.f.update(float(step["t_days"]), int(step["side"]))
        i = self.o._ti(t)
        # posterior-mixed quadratic coefficients
        A = float(pi @ self.o.A[i]); B = float(pi @ self.o.B[i]); C = float(pi @ self.o.C[i])
        z = self.o.p.z
        theta = lambda qq: -qq**2 * A - qq * B - C
        p_bid = (theta(q) - theta(q + z)) / z
        p_ask = (theta(q) - theta(q - z)) / z
        delta = (self.o.p.fill.delta_bar(p_bid) if step["side"] == 0
                 else self.o.p.fill.delta_bar(p_ask))
        half = 0.5 * (step["composite_ask"] - step["composite_bid"])
        return quantize_offset(float(delta), half)
