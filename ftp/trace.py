"""Traced rollouts for outcome analysis.

evaluate.py returns only episode-level BG objectives. For the benchmark-against-oracle
analysis we need finer logging: at each decision point, the oracle's action, the policy's
action, the (true) censored regime, the belief-filter posterior, inventory, and the
realized pnl. collect_traces() rolls a policy alongside the oracle on identical streams and
returns a tidy per-decision record list plus per-episode objective pairs, ready to load
into a notebook / DataFrame.
"""
from __future__ import annotations

import bisect

import numpy as np

from rfqsim.config import SimConfig
from rfqsim.mmpp import build_generator
from .belief import MMPPBeliefFilter
from .env import BondParams, generate_episode, simulate_fill
from .pipeline import GenConfig, _build_oracle
from .rollout import OraclePolicy, RandomPolicy
from .sector import build_sector_context
from .serialize import N_ACTIONS, offset_of_action, quantize_offset


def _traced_episode(rng, ep, policy, oracle, cfg, gc, log_belief=True):
    """Roll `policy` and the oracle in lockstep on the same stream; log per decision."""
    p = oracle.p
    Q = build_generator(cfg.mmpp)
    lo = cfg.mmpp.lam_low_frac * ep.bond.sector_intensity
    hi = cfg.mmpp.lam_high_frac * ep.bond.sector_intensity
    lam_b = np.array([lo, lo, hi, hi]); lam_a = np.array([lo, hi, lo, hi])
    bf = MMPPBeliefFilter(Q, lam_b, lam_a) if log_belief else None

    snaps = build_sector_context(ep.sector_stream, halflife_days=gc.sector_halflife)
    sec_t = [e["t_days"] for e in ep.sector_stream]

    orc_pol = OraclePolicy(oracle)
    if hasattr(policy, "reset"):
        policy.reset()

    q_pol = 0.0
    q_orc = 0.0
    obj_pol = 0.0
    obj_orc = 0.0
    flow_imb = 0.0
    prev_t = 0.0
    rows = []

    for step in ep.steps:
        t = float(step["t_days"]); state = int(step["mmpp_state"])
        dt = max(t - prev_t, 0.0)
        # accrue inventory penalties for both books on their own inventories
        obj_pol += -0.5 * p.gamma * p.sigma ** 2 * q_pol ** 2 * dt \
            + p.kappa * (p.lam_a[state] - p.lam_b[state]) * q_pol * dt
        obj_orc += -0.5 * p.gamma * p.sigma ** 2 * q_orc ** 2 * dt \
            + p.kappa * (p.lam_a[state] - p.lam_b[state]) * q_orc * dt
        prev_t = t

        belief = None
        if bf is not None:
            belief = bf.update(t, int(step["side"])).copy()

        s = dict(step, flow_imb=flow_imb)
        j = bisect.bisect_left(sec_t, t) - 1
        if 0 <= j < len(snaps):
            s.update(snaps[j])

        # oracle action (full info) and policy action (flow-only) on identical state
        a_orc = orc_pol(step, q_orc, t)
        a_pol = policy(s, q_pol, t) if not isinstance(policy, OraclePolicy) \
            else orc_pol(step, q_pol, t)

        half = 0.5 * (step["composite_ask"] - step["composite_bid"])
        # simulate fills independently for each book (common random number on the draw)
        seed_fill = int(rng.integers(1 << 30))
        f_orc, sgn = simulate_fill(np.random.default_rng(seed_fill), step,
                                   offset_of_action(a_orc, half), cfg)
        f_pol, _ = simulate_fill(np.random.default_rng(seed_fill), step,
                                 offset_of_action(a_pol, half), cfg)
        pnl_orc = offset_of_action(a_orc, half) * gc.z if f_orc else 0.0
        pnl_pol = offset_of_action(a_pol, half) * gc.z if f_pol else 0.0
        obj_orc += pnl_orc; obj_pol += pnl_pol
        if f_orc: q_orc += sgn * gc.z
        if f_pol: q_pol += sgn * gc.z
        if hasattr(policy, "observe_outcome"):
            policy.observe_outcome(f_pol, pnl_pol, q_pol)

        rows.append({
            "t": t, "regime": state, "side": int(step["side"]),
            "tier": int(step["client_tier"]), "k": int(step["k_dealers"]),
            "q_oracle": q_orc, "q_policy": q_pol,
            "a_oracle": a_orc, "a_policy": a_pol,
            "match": int(a_orc == a_pol), "absdiff": abs(a_orc - a_pol),
            "pnl_oracle": pnl_orc, "pnl_policy": pnl_pol,
            "belief_true": float(belief[state]) if belief is not None else np.nan,
            "belief_argmax_correct": int(np.argmax(belief) == state)
                if belief is not None else -1,
        })
        flow_imb = 0.85 * flow_imb + 0.15 * (1.0 if step["side"] == 1 else -1.0)

    return rows, {"obj_oracle": obj_orc, "obj_policy": obj_pol}


def collect_traces(cfg: SimConfig, gc: GenConfig, policy_factory, n_cusips=20,
                   episodes_per=4, seed=999, log_belief=True):
    """Roll a policy (built per-call by policy_factory(oracle, cfg) or None=random) and
    the oracle in lockstep across held-out CUSIPs. Returns (rows, episode_objs).

    policy_factory: callable -> a policy with (step,q,t)->action. If None, uses Random.
                    For the oracle-vs-oracle sanity, pass lambda o,c: OraclePolicy(o).
    """
    rng = np.random.default_rng(seed)
    all_rows, ep_objs = [], []
    eid = 0
    for _ in range(n_cusips):
        bond = BondParams.sample(cfg, rng, gc.sector_intensity, gc.cusip_frac)
        oracle = _build_oracle(cfg, bond, gc)
        for _ in range(episodes_per):
            ep = generate_episode(cfg, bond, gc.horizon_days, rng)
            if len(ep.steps) < 2:
                continue
            if policy_factory is None:
                pol = RandomPolicy(np.random.default_rng(int(rng.integers(1 << 30))))
            else:
                pol = policy_factory(oracle, cfg)
            rows, objs = _traced_episode(rng, ep, pol, oracle, cfg, gc,
                                         log_belief=log_belief)
            for r in rows:
                r["episode"] = eid
            all_rows.extend(rows)
            objs["episode"] = eid
            ep_objs.append(objs)
            eid += 1
    return all_rows, ep_objs
