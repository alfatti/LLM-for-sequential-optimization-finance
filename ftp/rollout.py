"""Episode rollout driver for the single-CUSIP BG/FTP market-making MDP (Scope A).

Threads dealer inventory through a finite-horizon episode, lets a policy choose a quote
offset per RFQ, simulates the fill via the engine-faithful mechanics (ftp.env), and
accumulates the BG objective. Policies share the signature policy(step, q, t) -> action.

Sector-flow context (ftp.sector) is attached to each observation so a flow-only agent can
infer the censored MMPP regime; the oracle reads the true regime out-of-band.
"""
from __future__ import annotations

import bisect

import numpy as np

from rfqsim.config import SimConfig
from .env import Episode, simulate_fill
from .oracle import BGOracle
from .sector import build_sector_context
from .serialize import N_ACTIONS, offset_of_action, quantize_offset


# --------------------------------------------------------------------------- #
# Policies                                                                     #
# --------------------------------------------------------------------------- #
class OraclePolicy:
    """Full-information BG oracle: sees the true MMPP regime, plays the optimal offset."""

    def __init__(self, oracle: BGOracle):
        self.o = oracle

    def __call__(self, step, q, t):
        state = int(step["mmpp_state"])
        db, da = self.o.quotes(t, q, state)
        delta = db if step["side"] == 0 else da
        half = 0.5 * (step["composite_ask"] - step["composite_bid"])
        return quantize_offset(delta, half)


class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def __call__(self, step, q, t):
        return int(self.rng.integers(N_ACTIONS))


# --------------------------------------------------------------------------- #
# Rollout                                                                      #
# --------------------------------------------------------------------------- #
def run_episode(rng, episode: Episode, policy, oracle: BGOracle, cfg: SimConfig,
                z: float = 1.0, sector_halflife: float = 0.03,
                with_sector_ctx: bool = True):
    """Roll one single-CUSIP episode. Returns BG objective, inventory path, and the
    per-step records (obs/action/reward) needed for serialization."""
    p = oracle.p
    steps = episode.steps
    # precompute causal sector context aligned to the sector stream
    if with_sector_ctx:
        snaps = build_sector_context(episode.sector_stream, halflife_days=sector_halflife)
        sec_t = [e["t_days"] for e in episode.sector_stream]

    q = 0.0
    q_path = [0.0]
    actions, rewards, obs_records = [], [], []
    bg_obj = 0.0
    flow_imb = 0.0
    prev_t = 0.0

    for step in steps:
        t = float(step["t_days"])
        state = int(step["mmpp_state"])
        dt = max(t - prev_t, 0.0)
        # running inventory penalty + realized imbalance drift-on-inventory (BG objective)
        bg_obj += -0.5 * p.gamma * p.sigma ** 2 * q ** 2 * dt
        bg_obj += p.kappa * (p.lam_a[state] - p.lam_b[state]) * q * dt
        prev_t = t

        s = dict(step, flow_imb=flow_imb)
        if with_sector_ctx:
            j = bisect.bisect_left(sec_t, t) - 1
            if 0 <= j < len(snaps):
                s.update(snaps[j])

        a = policy(s, q, t)
        half = 0.5 * (step["composite_ask"] - step["composite_bid"])
        delta = offset_of_action(a, half)
        filled, sgn = simulate_fill(rng, step, delta, cfg)
        pnl = delta * z if filled else 0.0
        bg_obj += pnl
        if filled:
            q += sgn * z
        # stateful LLM policies need the realized reward appended to their history
        if hasattr(policy, "observe_outcome"):
            policy.observe_outcome(filled, pnl, q)
        q_path.append(q)
        actions.append(a)
        rewards.append({"filled": filled, "pnl": pnl})
        obs_records.append(s)   # enriched obs (flow_imb + sector ctx) actually seen
        sign = +1.0 if step["side"] == 1 else -1.0
        flow_imb = 0.85 * flow_imb + 0.15 * sign

    bg_obj += -p.gamma_term * q ** 2
    return {"bg_objective": bg_obj, "q_path": q_path, "actions": actions,
            "rewards": rewards, "steps": steps, "obs_records": obs_records}


def optimality_gap(eval_obj, oracle_obj, random_obj):
    denom = oracle_obj - random_obj
    return float("nan") if abs(denom) < 1e-9 else float((oracle_obj - eval_obj) / denom)


def pooled_gap(eval_objs, oracle_objs, random_objs):
    """Pooled gap across a test set (sum then ratio) -- stable on short episodes."""
    O = float(np.sum(oracle_objs)); E = float(np.sum(eval_objs)); R = float(np.sum(random_objs))
    denom = O - R
    return float("nan") if abs(denom) < 1e-9 else (O - E) / denom
