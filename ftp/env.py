"""Single-CUSIP episode environment for the FTP/BEGV market-making experiment (Scope A).

Recycles the rfqsim generative mechanics, restructured from the engine's vectorized
(week, sector) chunk into a *sequential single-CUSIP episode*: a stream of RFQs on one
bond over a finite horizon, against which a policy quotes and accumulates inventory.

What is lifted verbatim from rfqsim.engine (step 5 mechanics), so the oracle is optimal
for the ACTUAL generative process (no model-mismatch):
  - logistic fill curve  p_trade = 1/(1+exp(alpha + beta * delta_best/delta0))
  - best-of-(k-1) competitor cover: p_min ~ Beta(1, m), d_cov = half*(1 + comp_noise*z_min)
  - win == our offset beats cover; trade gated by client intent
The MMPP regime, arrival intensities, and the kappa-imbalance price drift come straight
from rfqsim.mmpp.SectorChain. Bond statics (sigma, kappa, spread, base price) come from
a single row of rfqsim.universe (one CUSIP = one task).

The environment is policy-agnostic: a policy is a callable (step, q, t) -> action_index.
The rollout (ftp.rollout) drives it; this module just produces the RFQ stream and the fill.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfinv

from rfqsim.config import SimConfig
from rfqsim.mmpp import simulate_sector_chain


def _ndtri(p):
    return np.sqrt(2.0) * erfinv(2.0 * np.clip(p, 1e-7, 1 - 1e-7) - 1.0)


@dataclass
class BondParams:
    """Statics for ONE CUSIP (a single task), sampled to match rfqsim.universe."""
    base_price: float
    spread: float          # composite full spread delta0
    kappa: float           # imbalance->price drift sensitivity
    sigma: float           # cusip idiosyncratic vol ($/sqrt(day))
    sector_intensity: float  # sector-aggregate RFQs/day (drives the MMPP scale)
    cusip_frac: float      # this bond's share of sector flow (thinning prob)

    @staticmethod
    def sample(cfg: SimConfig, rng: np.random.Generator,
               sector_intensity: float = 800.0, cusip_frac: float = 0.06):
        """Draw a bond's statics from the same distributions as universe.build_universe."""
        pc = cfg.price
        amt = np.exp(rng.normal(19.6, 0.85))
        size_z = (np.log(amt) - 19.6) / 0.85
        maturity = float(np.random.choice([2.0, 5.0, 7.0, 10.0, 30.0]))
        spread = (pc.spread_base * np.exp(-0.18 * size_z) * np.exp(rng.normal(0, 0.25))
                  * (1.0 + 0.12 * max(maturity - 5.0, 0) / 25.0))
        return BondParams(
            base_price=float(rng.normal(pc.base_price_mean, pc.base_price_sigma)),
            spread=float(spread),
            kappa=float(np.clip(rng.normal(pc.kappa_mean, pc.kappa_sigma), 0.01, None)),
            sigma=float(pc.sigma_cusip_daily * np.exp(rng.normal(0, 0.3))),
            sector_intensity=sector_intensity,
            cusip_frac=cusip_frac,
        )


@dataclass
class Episode:
    """One CUSIP's RFQ stream over the horizon, plus the sector stream behind it."""
    steps: list          # per-RFQ dicts for THIS cusip (the agent's decision points)
    sector_stream: list  # all sector RFQs (time, side, mmpp_state) for belief/sector ctx
    bond: BondParams
    horizon: float


def generate_episode(cfg: SimConfig, bond: BondParams, horizon_days: float,
                     rng: np.random.Generator) -> Episode:
    """Generate one single-CUSIP episode over [0, horizon_days].

    Sector MMPP drives a dense sector stream; the agent's CUSIP is a thinned subsample
    (cusip_frac). Mid follows base + kappa*imbalance_integral + idio Brownian increments,
    matching the engine's construction (simplified to single-CUSIP: no issuer grid, the
    issuer Brownian bridge becomes a per-CUSIP random walk).
    """
    oc, pc = cfg.outcome, cfg.price
    chain = simulate_sector_chain(cfg.mmpp, bond.sector_intensity, horizon_days, rng)

    # --- dense sector stream (all CUSIPs in the sector): times, sides, regime ---
    sector_stream = []
    for k in range(len(chain.states)):
        s = int(chain.states[k]); t0 = chain.times[k]
        t1 = chain.times[k + 1] if k + 1 < len(chain.times) else horizon_days
        dur = max(t1 - t0, 0.0)
        for side, lam in [(0, chain.lam_b[s]), (1, chain.lam_a[s])]:
            n = int(rng.poisson(lam * dur))
            for _ in range(n):
                sector_stream.append({"t_days": t0 + rng.random() * dur,
                                      "side": side, "mmpp_state": s})
    sector_stream.sort(key=lambda r: r["t_days"])

    # --- thin to this CUSIP, attach marks (size, tier, k, mid) ---
    steps = []
    # mid: precompute an idio random-walk anchor; kappa-drift added per-RFQ exactly
    flow_scale = cfg.flow.rfqs_per_day_target / cfg.universe.n_sectors
    for ev in sector_stream:
        if rng.random() >= bond.cusip_frac:
            continue
        t = ev["t_days"]
        # exact MMPP imbalance integral at t (drives the reference-price drift)
        imb_int = chain.imbalance_at(np.array([t]))[0]
        idio = rng.standard_normal() * bond.sigma * np.sqrt(max(t, 1e-6))
        mid = bond.base_price + bond.kappa * imb_int / flow_scale + idio
        half = 0.5 * bond.spread
        # client tier (size-correlated tiers as in universe: rough 10/30/60 split)
        u = rng.random()
        tier = 0 if u < 0.10 else (1 if u < 0.40 else 2)
        size = float(max(np.exp(rng.standard_normal() * oc.size_lognorm_sigma
                                + oc.size_lognorm_mu), oc.odd_lot_floor))
        # competition: k dealers (generic asset_mgr-ish band), large-size decrement
        kd = int(rng.integers(4, 9))
        if size > oc.large_size_threshold:
            kd = max(kd - oc.large_size_k_decrement, 1)
        steps.append({
            "t_days": t, "mmpp_state": ev["mmpp_state"], "side": ev["side"],
            "client_tier": tier, "size": size, "k_dealers": kd,
            "composite_mid": mid, "composite_bid": mid - half, "composite_ask": mid + half,
        })
    return Episode(steps=steps, sector_stream=sector_stream, bond=bond,
                   horizon=horizon_days)


def simulate_fill(rng: np.random.Generator, step: dict, delta: float,
                  cfg: SimConfig) -> tuple[bool, float]:
    """Engine-faithful fill: best-of-(k-1) cover x logistic trade gate.

    Returns (filled, signed_inventory_change_per_unit). Lifted from engine.py step 5.
    `delta` is OUR offset on the requested side ($). Client intent is folded into a
    per-RFQ Bernoulli (the engine uses per-client intent; here a generic intent prob).
    """
    oc = cfg.outcome
    d0 = step["composite_ask"] - step["composite_bid"]
    half = 0.5 * d0
    kd = max(int(step["k_dealers"]), 1)
    m = max(kd - 1, 0)
    if m >= 1:
        u_min = rng.random()
        p_min = 1.0 - (1.0 - u_min) ** (1.0 / m)
        z_min = _ndtri(p_min)
        d_cov = max(half * (1.0 + oc.competitor_noise * z_min), 0.02 * d0)
    else:
        d_cov = np.inf
    we_win = delta <= d_cov
    d_best = min(delta, d_cov)
    p_trade = 1.0 / (1.0 + np.exp(oc.logit_alpha + oc.logit_beta * d_best / d0))
    # generic intent prob (engine uses per-client; ~0.8 asset-mgr-like)
    intent = rng.random() < 0.80
    traded = intent and (rng.random() < p_trade)
    filled = bool(traded and we_win)
    sgn = +1.0 if step["side"] == 0 else -1.0   # side 0 = client sells -> we buy (+)
    return filled, sgn
