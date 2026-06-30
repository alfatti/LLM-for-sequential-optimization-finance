"""
Does S1 (non-stationary c*(t)) have DISCRIMINATING POWER, or does flatness absorb it?
sigma ramps over the episode -> c*(t)=(mu-r)/(sigma(t)^2 rho) varies with t.
Compare (CRN optimality gap vs the time-varying optimum):
  (a) the time-varying optimal policy u=c*(t)x          [structure oracle, gap=0 by def]
  (b) the OLS-shortcut constant policy u=c_bar x         [what our scalar null produces]
  (c) the BEST possible constant policy                   [lower bound on any constant]
If (b)/(c) gaps are large -> the test can detect a structure-learner. If ~0 -> vacuous.
Run in BOTH regimes (curvature may govern detectability -- ties to the flatness finding).
"""
import numpy as np
from core.dynamics import crra_utility

mu, r, rho = 0.12, 0.03, 2.0
T, x0, n_steps = 1.0, 1.0, 10; dt = T/n_steps

# sigma schedule: ramp 0.15 -> 0.30 across the episode
sig = np.linspace(0.15, 0.30, n_steps)
cstar_t = (mu - r) / (sig**2 * rho)              # time-varying optimal coefficient

def rollout(coeff_fn, spread, dW):
    """coeff_fn(k)->c at step k. Returns terminal wealth per path."""
    n = dW.shape[0]; W = np.full(n, x0)
    for k in range(n_steps):
        c = coeff_fn(k); u = c * W
        pen = spread * np.maximum(0.0, u - W)
        s = sig[k]
        W = np.maximum(W + (u*(mu-r) + r*W - pen)*dt + u*s*dW[:, k], 1e-9)
    return W

def gap_vs_optimal(coeff_fn, spread, n=60000, seed=0):
    rng = np.random.default_rng(seed)
    dW = np.sqrt(dt)*rng.standard_normal((n, n_steps))
    Wopt = rollout(lambda k: cstar_t[k], spread, dW)
    Wcf  = rollout(coeff_fn, spread, dW)
    Uo = crra_utility(Wopt, rho); Uc = crra_utility(Wcf, rho)
    return (Uo - Uc).mean()/abs(Uo.mean())

# OLS-shortcut c_bar: what OLS recovers from optimal-policy demo pairs (A_t=c*(t)W_t).
# Under the optimal policy, A_t/W_t = c*(t); OLS slope = sum(c*(t)W_t^2)/sum(W_t^2),
# i.e. an E[W^2]-weighted average of c*(t). Approximate weights via a quick rollout.
def ols_cbar(spread, n=40000, seed=1):
    rng = np.random.default_rng(seed); dW=np.sqrt(dt)*rng.standard_normal((n,n_steps))
    W = np.full(n, x0); num=0.0; den=0.0
    for k in range(n_steps):
        c=cstar_t[k]; num += (c*W**2).sum(); den += (W**2).sum()
        u=c*W; pen=spread*np.maximum(0.0,u-W)
        W=np.maximum(W+(u*(mu-r)+r*W-pen)*dt+u*sig[k]*dW[:,k],1e-9)
    return num/den

for spread, name in [(0.0,"benign"),(0.4,"sharp")]:
    cbar = ols_cbar(spread)
    # best constant by 1-D search
    cs = np.linspace(0.3, 2.2, 40)
    gaps = [gap_vs_optimal(lambda k,c=c: c, spread, n=20000) for c in cs]
    cbest = cs[int(np.argmin(gaps))]; gbest = min(gaps)
    g_ols = gap_vs_optimal(lambda k: cbar, spread, n=60000)
    print(f"[{name}] c*(t) ramps {cstar_t[0]:.2f}->{cstar_t[-1]:.2f} | "
          f"OLS c_bar={cbar:.2f}  gap(OLS const)={100*g_ols:.2f}%  "
          f"best const c={cbest:.2f} gap={100*gbest:.2f}%")
