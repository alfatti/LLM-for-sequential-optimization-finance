"""
Optimality gap with COMMON RANDOM NUMBERS (paired evaluation). For each eval path
we drive BOTH the optimal policy and the c_hat policy with the same Brownian
increments, so the gap is a low-variance paired difference. This (a) resolves the
true K_test dependence and (b) is the variance-reduction we'll need for LLM eval.

Also surfaces a substantive finding: how FLAT is the CRRA objective near the
optimum? Flatness compresses optimality gaps and lowers the contrast of claim 1.
"""
import numpy as np
from core.dynamics import Market, crra_coeff, crra_utility
from core.estimator import build_support_pairs, estimate_ols

mkt = Market(mu=0.12, sigma=0.20, r=0.03); rho = 2.0
T, x0, n_steps = 1.0, 1.0, 10
c_true = crra_coeff(mkt, rho)
util = lambda x: crra_utility(x, rho)

def paired_wealth(c1, c2, x0, dW):
    """Roll two constant-fraction policies on the SAME Brownian path dW."""
    dt = T / n_steps
    x1 = x0; x2 = x0
    for k in range(n_steps):
        u1 = c1 * x1; u2 = c2 * x2
        x1 += (u1*(mkt.mu-mkt.r) + mkt.r*x1)*dt + u1*mkt.sigma*dW[k]
        x2 += (u2*(mkt.mu-mkt.r) + mkt.r*x2)*dt + u2*mkt.sigma*dW[k]
    return x1, x2

def crn_gap(c_hat, n_paths, rng):
    dt = T/n_steps
    diffs = np.empty(n_paths)
    EUopt = np.empty(n_paths)
    for i in range(n_paths):
        dW = np.sqrt(dt)*rng.standard_normal(n_steps)
        x_opt, x_hat = paired_wealth(c_true, c_hat, x0, dW)
        diffs[i] = util(x_opt) - util(x_hat)
        EUopt[i] = util(x_opt)
    return diffs.mean()/abs(EUopt.mean())   # relative gap, low-variance

rng = np.random.default_rng(31)

# (1) curvature: gap vs fixed coefficient error (no estimation, pure objective shape)
print("=== Objective flatness near optimum (CRN, 200k paths) ===")
for frac in [0.5, 0.8, 0.9, 1.1, 1.2, 1.5]:
    g = crn_gap(frac*c_true, 200000, np.random.default_rng(99))
    print(f"  c_hat = {frac:.1f} c*  ->  optimality gap = {100*g:.3f}%")

# (2) gap vs K_test with estimation, CRN eval, label noise on demos
print("\n=== Optimality gap vs K_test (CRN eval, noisy demos) ===")
K = [1,2,5,10,20,50]
for kt in K:
    gtask=[]
    for _ in range(200):
        xs, ys = build_support_pairs(mkt, rho, x0, T, n_steps, kt, rng, label_noise=0.25)
        ch = estimate_ols(xs, ys)
        gtask.append(crn_gap(ch, 4000, rng))
    arr=np.array(gtask)
    print(f"  K_test={kt:3d}  gap={100*arr.mean():.3f}%  (+/- {100*arr.std(ddof=1)/np.sqrt(len(arr)):.3f})")
