"""Serialization of RFQ market-making episodes into LLM-readable strings.

One RFQ = one decision step (the granularity you chose). The schema mirrors the
SFT paper's deterministic encoder E: an observation block, an action token, then a
reward block. Crucially, the **MMPP regime label is censored** -- it is NEVER written
into the observation. The agent must infer liquidity state from the flow it can see
(recent sides, fill outcomes, mid drift). The oracle, by contrast, gets the true
state out-of-band when generating training labels.

Action space (discrete, gamma=0.5 regime): the dealer's quoted half-spread offset
delta on the requested side, expressed in units of the composite half-spread, then
binned. Bins must straddle two scales at gamma=0.5: the ~0.13 inventory-driven move
and the ~0.58 regime-driven move (both in $ at delta0~0.9). We bin in half-spread
units spanning roughly [0.2, 1.8] so the oracle's optimal offset lands inside the grid
in both symmetric and imbalanced states.
"""
from __future__ import annotations

import numpy as np

# Discrete offset grid in units of the composite *half*-spread (delta0/2 ~ 0.45$).
# 9 bins. Edges chosen so symmetric-state optimum (~0.95 half-spreads) sits mid-grid
# and the imbalanced defensive/aggressive offsets remain representable.
ACTION_BINS = np.array([0.20, 0.45, 0.70, 0.85, 0.95, 1.10, 1.30, 1.55, 1.85])
N_ACTIONS = len(ACTION_BINS)


def quantize_offset(delta, half_spread):
    """Map a continuous $ offset to the nearest action-bin index."""
    u = delta / max(half_spread, 1e-9)          # offset in half-spread units
    return int(np.argmin(np.abs(ACTION_BINS - u)))


def offset_of_action(a, half_spread):
    """Inverse: action index -> $ offset (bin center * half-spread)."""
    return ACTION_BINS[int(a)] * half_spread


# --------------------------------------------------------------------------- #
# Observation features (regime-censored)                                      #
# --------------------------------------------------------------------------- #
def _inv_bucket(q, z):
    """Coarse signed inventory bucket in units of trade size z."""
    n = int(round(q / max(z, 1e-9)))
    return max(-5, min(5, n))


def _size_bucket(size):
    if size < 250_000: return "S"
    if size < 1_000_000: return "M"
    if size < 5_000_000: return "L"
    return "XL"


def encode_observation(step, q, z):
    """One observation line. step is a dict with RFQ fields from the rfqsim schema.

    Regime label (step['mmpp_state']) is deliberately NOT included. The agent infers
    it from observable flow. Two tiers of flow context:
      - CUSIP-local 'flow_imb': recent signed pressure on THIS bond (thin, noisy).
      - Sector context (the enhancement): dense read on the latent sector regime,
        computed over a finite recent window so it stays a noisy estimator of the
        true intensities (the window length is the observation-fidelity knob).
    """
    side = "BUY" if step["side"] == 1 else "SELL"
    tier = int(step["client_tier"])
    base = (f"<OBS> q={_inv_bucket(q, z):+d} side={side} sz={_size_bucket(step['size'])} "
            f"tier={tier} mid={step['composite_mid']:.2f} "
            f"flow_imb={step.get('flow_imb', 0.0):+.2f} k={int(step['k_dealers'])}")
    # sector context (present iff the rollout supplies it). Rates normalized to a
    # small bounded ratio (sec_ask/sec_bid skew + total-rate z-ish) for clean tokens.
    if "sec_imb" in step:
        sb = step.get("sec_bid", 0.0); sa = step.get("sec_ask", 0.0)
        tot = sb + sa
        rate_skew = (sa - sb) / tot if tot > 1e-9 else 0.0   # in [-1,1]
        base += (f" | sec_imb={step['sec_imb']:+.2f} "
                 f"sec_skew={rate_skew:+.2f}")
    return base


def encode_action(a):
    return f"<ACT> {int(a)}"


def encode_reward(filled, pnl, q_next, z):
    out = "win" if filled else "miss"
    return f"<REW> {out} pnl={pnl:+.4f} q={_inv_bucket(q_next, z):+d}"


def serialize_episode(steps, actions, rewards, q_path, z, header=None):
    """Assemble a full episode string from per-step records."""
    lines = []
    if header:
        lines.append(header)
    for i in range(len(steps)):
        lines.append(encode_observation(steps[i], q_path[i], z))
        lines.append(encode_action(actions[i]))
        lines.append(encode_reward(rewards[i]["filled"], rewards[i]["pnl"],
                                   q_path[i + 1], z))
    return "\n".join(lines)
