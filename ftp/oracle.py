"""Bergault-Gueant finite-horizon market-making oracle (quadratic approximation).

This is the *baseline / oracle* policy for the LLM-imitation experiment. It solves
the BG single-asset RFQ market-making control under the Fair-Transfer-Price
objective, in the **full-information** regime: the theoretical market maker knows
the current MMPP liquidity state (jb, ja) at every instant.

Model (BG "Liquidity Dynamics in RFQ Markets", Sec 3.2.1, Sec 4.4):

    dS_t   = sigma dW_t + kappa (lam^a_t - lam^b_t) dt          (reference price)
    dq_t   = z dN^b_t - z dN^a_t                                 (inventory)
    fill prob at offset delta:  f(delta) = 1 / (1 + exp(alpha + beta * delta/delta0))

    maximize  E[ int_0^T ( z lam^b delta^b f(delta^b) + z lam^a delta^a f(delta^a)
                           + kappa(lam^a - lam^b) q - (gamma/2) sigma^2 q^2 ) dt ]

Per-state value functions theta^{jb,ja}(t,q) satisfy the HJB system (BG p.12/p.26).
We use the **quadratic approximation a la Bergault et al.** (BG p.26-27): replace the
Hamiltonians H^{b/a}(p) = sup_delta f(delta)(delta - p) by their order-2 Taylor
expansion at p=0, which makes theta quadratic in q,

    theta^{jb,ja}(t,q) = -q^2 A_{jb,ja}(t) - q B_{jb,ja}(t) - C_{jb,ja}(t),

with A, B, C solving the coupled ODE system (BG p.27), integrated backward from
A(T)=B(T)=C(T)=0 (pure BG terminal condition; a terminal inventory penalty would
set A(T)=gamma_term). The *exact* quote map is retained (BG keep the true f^{-1}):

    delta^{b,*} = delta_bar( (theta(t,q) - theta(t,q+z)) / z )
    delta^{a,*} = delta_bar( (theta(t,q) - theta(t,q-z)) / z )
    delta_bar(p) = f^{-1}( -H'(p) ).

For the symmetric logistic f used here (f^b=f^a=f), H, H', f^{-1} are closed form
(see Hamiltonian below), so the quote map is analytic given A,B,C.

State indexing matches rfqsim.mmpp: 0=LL, 1=LH, 2=HL, 3=HH, lexicographic over
(bid-level, ask-level). lam^b uses the bid level, lam^a the ask level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MMPP state layout (mirror of rfqsim.mmpp)
LL, LH, HL, HH = 0, 1, 2, 3
N_STATES = 4


# --------------------------------------------------------------------------- #
# Logistic fill curve and its Hamiltonian                                     #
# --------------------------------------------------------------------------- #
# f(delta) = 1 / (1 + exp(alpha + b * delta)),  with b = beta_logit / delta0.
# This is decreasing from f(-inf)=1 to f(+inf)=0, as required for an S-curve.
#
# Hamiltonian  H(p) = sup_{delta} f(delta) (delta - p).
# With u = f(delta) in (0,1), delta = (logit-inverse): delta(u) = (1/b)(log((1-u)/u) - alpha).
# The first-order condition for H gives, for the logistic, the standard closed form
# used in OTC market-making (Gueant 2017): the optimal fill prob u*(p) solves
#   delta(u*) - p = -u* delta'(u*) = u*/(b u*(1-u*)) ... -> 1 - u* = exp(b p + alpha + 1)?? 
# Rather than rely on a memorized closed form, we compute H, H', H'' by the robust
# route: H'(p) = -u*(p) (envelope theorem), and u*(p) is found by 1-D root finding on
# the strictly monotone FOC. H''(p) follows analytically once u* is known. The Taylor
# coefficients at p=0 (alpha0,alpha1,alpha2 in BG) use H(0), H'(0), H''(0).


@dataclass(frozen=True)
class FillCurve:
    """Logistic fill curve f(delta) = 1/(1+exp(alpha + b*delta)), b = beta/delta0."""

    alpha: float
    beta: float
    delta0: float  # composite bid-ask spread scale (the BG delta^0)

    @property
    def b(self) -> float:
        return self.beta / self.delta0

    def f(self, delta):
        return 1.0 / (1.0 + np.exp(self.alpha + self.b * np.asarray(delta, float)))

    def delta_of_u(self, u):
        """Inverse fill curve: offset delta that yields fill prob u in (0,1)."""
        u = np.clip(np.asarray(u, float), 1e-12, 1 - 1e-12)
        return (np.log((1.0 - u) / u) - self.alpha) / self.b

    # -- Hamiltonian H(p) = sup_delta f(delta)(delta - p) ------------------- #
    # Parameterize the sup by u = f(delta) in (0,1). Objective in u:
    #   g(u; p) = u * (delta_of_u(u) - p).
    # FOC: d/du [u (delta(u) - p)] = (delta(u) - p) + u delta'(u) = 0,
    # with delta'(u) = -1 / (b u (1-u)).  So FOC:  delta(u) - p = 1 / (b (1-u)).
    # LHS strictly decreasing in u, RHS strictly increasing in u -> unique root.
    def _u_star(self, p):
        p = np.asarray(p, float)
        out = np.empty_like(p, dtype=float)
        flat = out.reshape(-1)
        pv = p.reshape(-1)
        for i, pi in enumerate(pv):
            lo, hi = 1e-9, 1 - 1e-9
            # FOC residual r(u) = delta(u) - p - 1/(b(1-u)); decreasing in u.
            def r(u):
                return self.delta_of_u(u) - pi - 1.0 / (self.b * (1.0 - u))
            rlo, rhi = r(lo), r(hi)
            if rlo <= 0:           # corner: tiny fill prob optimal
                flat[i] = lo
                continue
            if rhi >= 0:
                flat[i] = hi
                continue
            for _ in range(80):    # bisection (monotone, robust)
                mid = 0.5 * (lo + hi)
                if r(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            flat[i] = 0.5 * (lo + hi)
        return out

    def H(self, p):
        u = self._u_star(p)
        return u * (self.delta_of_u(u) - p)

    def H_prime(self, p):
        # Envelope theorem: dH/dp = -u*(p).
        return -self._u_star(p)

    def H_double_prime(self, p, h=1e-5):
        # u*(p) is smooth; differentiate the envelope value numerically.
        return -(self._u_star(p + h) - self._u_star(p - h)) / (2 * h)

    def taylor_coeffs(self):
        """(alpha0, alpha1, alpha2) = (H(0), H'(0), H''(0)) for the quad approx."""
        a0 = float(self.H(0.0))
        a1 = float(self.H_prime(0.0))
        a2 = float(self.H_double_prime(0.0))
        return a0, a1, a2

    def delta_bar(self, p):
        """Exact optimal offset given the dual variable p:  f^{-1}(-H'(p)).

        -H'(p) = u*(p) is exactly the optimal fill prob, so delta_bar(p) = delta(u*(p)).
        """
        return self.delta_of_u(self._u_star(p))


# --------------------------------------------------------------------------- #
# Oracle: solve the A/B/C Riccati system, expose quotes/skew/FTP              #
# --------------------------------------------------------------------------- #
@dataclass
class BGOracleParams:
    lam_b: np.ndarray        # (4,) bid intensity per MMPP state (RFQs/day)
    lam_a: np.ndarray        # (4,) ask intensity per MMPP state
    Q: np.ndarray            # (4,4) MMPP generator (rows sum to 0)
    kappa: float             # price drift per unit imbalance ($/ (RFQ/day))
    sigma: float             # reference-price vol ($/sqrt(day))
    gamma: float             # inventory risk aversion
    z: float                 # reference trade size (BG single fixed size)
    fill: FillCurve
    T: float                 # horizon (trading days)
    gamma_term: float = 0.0  # terminal inventory penalty: A(T)=gamma_term


class BGOracle:
    """Finite-horizon BG quadratic-approx oracle, full-information over MMPP state."""

    def __init__(self, p: BGOracleParams, n_t: int = 2000):
        self.p = p
        self.n_t = n_t
        self.t_grid = np.linspace(0.0, p.T, n_t + 1)
        self._solve()

    # ---- backward integration of A, B, C (BG p.27) ------------------------ #
    def _solve(self):
        p = self.p
        z = p.z
        a0, a1, a2 = p.fill.taylor_coeffs()

        # Delta_i,k = alpha_i * z^k  (BG notation). Symmetric curve: b-side == a-side.
        d = {(i, k): (a0 if i == 0 else a1 if i == 1 else a2) * z ** k
             for i in (0, 1, 2) for k in (1, 2, 3)}
        lb, la, Q = p.lam_b, p.lam_a, p.Q
        gs2 = p.gamma * p.sigma ** 2

        def rhs(A, B, C):
            # A,B,C are length-4 vectors over MMPP states. Equations BG p.27,
            # with b-side and a-side coefficients equal (single S-curve).
            dA = np.empty(4); dB = np.empty(4); dC = np.empty(4)
            for s in range(4):
                lbs, las = lb[s], la[s]
                # A'(t)
                dA[s] = (2 * (lbs * d[(2, 1)] + las * d[(2, 1)]) * A[s] ** 2
                         - 0.5 * gs2
                         - Q[s] @ A)
                # B'(t)
                dB[s] = (2 * (lbs * d[(1, 1)] - las * d[(1, 1)]) * A[s]
                         + 2 * (lbs * d[(2, 2)] - las * d[(2, 2)]) * A[s] ** 2
                         + p.kappa * (las_intensity := (la[s] - lb[s]))
                         + 2 * (lbs * d[(2, 1)] + las * d[(2, 1)]) * A[s] * B[s]
                         - Q[s] @ B)
                # C'(t)
                dC[s] = ((lbs * d[(0, 1)] + las * d[(0, 1)])
                         + (lbs * d[(1, 2)] + las * d[(1, 2)]) * A[s]
                         + (lbs * d[(1, 1)] - las * d[(1, 1)]) * B[s]
                         + 0.5 * (lbs * d[(2, 3)] + las * d[(2, 3)]) * A[s] ** 2
                         + 0.5 * (lbs * d[(2, 1)] + las * d[(2, 1)]) * B[s] ** 2
                         + (lbs * d[(2, 2)] - las * d[(2, 2)]) * A[s] * B[s]
                         - Q[s] @ C)
            return dA, dB, dC

        nt = self.n_t
        A = np.zeros((nt + 1, 4)); B = np.zeros((nt + 1, 4)); C = np.zeros((nt + 1, 4))
        A[nt] = p.gamma_term  # terminal conditions
        dt = p.T / nt
        # backward RK4 (integrate from t=T down to t=0)
        for n in range(nt, 0, -1):
            An, Bn, Cn = A[n], B[n], C[n]
            k1 = rhs(An, Bn, Cn)
            k2 = rhs(An - 0.5 * dt * k1[0], Bn - 0.5 * dt * k1[1], Cn - 0.5 * dt * k1[2])
            k3 = rhs(An - 0.5 * dt * k2[0], Bn - 0.5 * dt * k2[1], Cn - 0.5 * dt * k2[2])
            k4 = rhs(An - dt * k3[0], Bn - dt * k3[1], Cn - dt * k3[2])
            A[n - 1] = An - dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            B[n - 1] = Bn - dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            C[n - 1] = Cn - dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.A, self.B, self.C = A, B, C

    # ---- value function and quotes ---------------------------------------- #
    def _ti(self, t):
        return int(np.clip(round(t / self.p.T * self.n_t), 0, self.n_t))

    def theta(self, t, q, state):
        i = self._ti(t)
        return -q ** 2 * self.A[i, state] - q * self.B[i, state] - self.C[i, state]

    def quotes(self, t, q, state):
        """Optimal (delta_bid, delta_ask) offsets at (t, q, MMPP state)."""
        z = self.p.z
        p_bid = (self.theta(t, q, state) - self.theta(t, q + z, state)) / z
        p_ask = (self.theta(t, q, state) - self.theta(t, q - z, state)) / z
        return float(self.p.fill.delta_bar(p_bid)), float(self.p.fill.delta_bar(p_ask))

    def skew(self, t, q, state):
        """Quote skew = delta_ask - delta_bid (the BG asymmetry projected to price)."""
        db, da = self.quotes(t, q, state)
        return da - db

    def ftp_minus_mid(self, t, state, q=0.0):
        """FTP offset from reference mid = 0.5*(delta_ask - delta_bid) at q=0 (BG eq.)."""
        db, da = self.quotes(t, q, state)
        return 0.5 * (da - db)
