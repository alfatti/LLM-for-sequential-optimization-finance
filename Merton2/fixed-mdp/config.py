"""
The single FIXED MDP for the warm-up ICL-vs-SFT test. One market, one optimal policy
(c*). The LLM sees multiple optimal-policy ROLLOUTS of this same MDP (varying only the
Brownian path). No task distribution -> no 'which market' inference; the question is
narrow: for THIS MDP, is ICL or SFT more effective at learning to act optimally?

Two regimes carry over from Phase 0 (the curvature contrast still governs detectability):
  benign (spread=0): flat objective   -> utility gap tiny; lead with coefficient recovery
  sharp  (spread=0.4): borrowing kink  -> utility gap has contrast
"""
FIXED_MDP = dict(mu=0.12, sigma=0.20, r=0.03, rho=2.0, T=1.0, n_steps=10, x0=1.0)
REGIME_SPREAD = dict(benign=0.0, sharp=0.4)

# label noise on SUPPORT demos (clean query targets). Under one MDP this makes the
# in-context step genuine averaging rather than copying a single (W,A) pair.
LABEL_NOISE = 0.25
