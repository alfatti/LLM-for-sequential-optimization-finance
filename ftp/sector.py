"""Sector-level flow context with a tunable observation window.

The agent quotes a single CUSIP, but a real desk watches the whole sector's tape. This
tracker consumes the FULL sector RFQ stream (every CUSIP) and maintains windowed
estimates of side imbalance and per-side arrival rate. These are noisy estimators of the
latent MMPP intensities -- noisy because they're finite-window counts, not true rates --
so they let the agent infer the sector regime well without it being handed to them.

The window (EWMA half-life in trading-day units) is the observation-fidelity knob, the
analogue of the SFT paper's q in {0.5, 0.8, 1.0}:
  short half-life -> noisy, reactive, harder inference, no lag
  long  half-life -> smooth, low-noise, easier inference, but laggy after a switch
"""
from __future__ import annotations

import numpy as np


class SectorFlowTracker:
    """Windowed running stats over a sector's RFQ stream (all CUSIPs)."""

    def __init__(self, halflife_days=0.05):
        self.hl = halflife_days
        self.t = None
        self.imb = 0.0          # EWMA of signed side (+1 buy, -1 sell)
        self.bid_rate = 0.0     # EWMA arrival rate of bid-side (client sell) RFQs
        self.ask_rate = 0.0     # EWMA arrival rate of ask-side (client buy) RFQs

    def _decay(self, dt):
        return 0.5 ** (max(dt, 0.0) / self.hl)

    def observe(self, t, side):
        """Update with one sector RFQ at time t (days), side 0=bid/sell,1=ask/buy."""
        if self.t is None:
            self.t = t
        dt = t - self.t
        w = self._decay(dt)
        # instantaneous rate contribution ~ 1 event / window normalized by halflife
        inst_rate = (np.log(2) / self.hl)   # an event raises the EWMA rate by ~this
        self.imb = w * self.imb + (1 - w) * (1.0 if side == 1 else -1.0)
        self.bid_rate = w * self.bid_rate + (0.0 if side == 1 else inst_rate)
        self.ask_rate = w * self.ask_rate + (inst_rate if side == 1 else 0.0)
        self.t = t

    def snapshot(self):
        return {"sec_imb": float(self.imb),
                "sec_rate": float(self.bid_rate + self.ask_rate),
                "sec_bid": float(self.bid_rate),
                "sec_ask": float(self.ask_rate)}


def build_sector_context(sector_stream, halflife_days=0.05):
    """Precompute the sector-context snapshot visible JUST BEFORE each RFQ.

    sector_stream: time-ordered list of dicts with 't_days' and 'side' for the WHOLE
    sector (every CUSIP). Returns a list of snapshots aligned to sector_stream, each
    reflecting flow strictly prior to that event (causal -- no lookahead).
    """
    trk = SectorFlowTracker(halflife_days)
    snaps = []
    for ev in sector_stream:
        snaps.append(trk.snapshot())          # state BEFORE this event (causal)
        trk.observe(ev["t_days"], int(ev["side"]))
    return snaps
