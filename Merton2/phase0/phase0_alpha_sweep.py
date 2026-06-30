"""
Does alpha (CRRA U(x)=x^alpha/alpha) give enough optimality-gap CONTRAST to test
claim 1 fairly? We sweep alpha and, for each, measure the CRN optimality-gap "bowl"
vs relative coefficient error. Contrast = gap at +/-30% error. We also flag the
leverage/degeneracy boundary where c* explodes and wealth can go non-positive.

Risk aversion rho = 1 - alpha.  c* = (mu-r)/(sigma^2 * rho).
"""
import numpy as np
from core.dynamics import Market, crra_coeff, crra_utility

mkt = Market(mu=0.12, sigma=0.20, r=0.03)
T, x0, n_steps = 1.0, 1.0, 10
dt = T/n_steps

def crn_gap_general(alpha, frac, n=60000):
    rho = 1.0 - alpha
    c = (mkt.mu-mkt.r)/(mkt.sigma**2*rho)
    util = lambda x: crra_utility(x, rho)
    rng = np.random.default_rng(13)
    d = np.empty(n); o = np.empty(n); bad = 0
    for i in range(n):
        dW = np.sqrt(dt)*rng.standard_normal(n_steps)
        xo = x0; xh = x0
        for k in range(n_steps):
            uo = c*xo; uh = (frac*c)*xh
            xo += (uo*(mkt.mu-mkt.r)+mkt.r*xo)*dt + uo*mkt.sigma*dW[k]
            xh += (uh*(mkt.mu-mkt.r)+mkt.r*xh)*dt + uh*mkt.sigma*dW[k]
        if xo<=0 or xh<=0: bad += 1; xo=max(xo,1e-9); xh=max(xh,1e-9)
        d[i]=util(xo)-util(xh); o[i]=util(xo)
    return d.mean()/abs(o.mean()), c, bad/n

print(f"{'alpha':>6} {'rho':>5} {'c*':>7} {'gap@-30%':>9} {'gap@+30%':>9} {'contrast':>9} {'%wealth<=0':>11}")
for alpha in [-5.0, -3.0, -2.0, -1.0, -0.5, 0.2, 0.5, 0.7, 0.85, 0.95]:
    rho = 1.0-alpha
    g_lo, c, b_lo = crn_gap_general(alpha, 0.7)
    g_hi, _, b_hi = crn_gap_general(alpha, 1.3)
    contrast = 100*max(g_lo, g_hi)
    flag = "  <-- leverage/degeneracy" if (c>3 or max(b_lo,b_hi)>0.001) else ""
    print(f"{alpha:6.2f} {rho:5.2f} {c:7.3f} {100*g_lo:8.3f}% {100*g_hi:8.3f}% {contrast:8.3f}% {100*max(b_lo,b_hi):10.2f}%{flag}")
