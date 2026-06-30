"""
Sharp-Merton via a borrowing spread (two interest rates). Lend cash at r, borrow at
r+s when the position u exceeds wealth x. Dynamics:
   dX = (u(mu-r) + r x - s*max(0, u - x)) dt + u sigma dW.
For u=c x the penalty is s*x*max(0,c-1): a KINK at c=1 (fully invested). If c*>1 the
optimum sits near the kink and over-leverage is first-order costly -> breaks flatness.
Oracle found by numerical EU-maximization (exact over policy class); check linearity.
"""
import numpy as np
from core.dynamics import Market, crra_utility
mkt = Market(mu=0.12, sigma=0.20, r=0.03); rho = 2.0
T, x0, n_steps = 1.0, 1.0, 10; dt = T/n_steps
util = lambda x: crra_utility(x, rho)

def roll_EU(c, spread, n, rng, b=0.0):
    U = np.empty(n)
    for i in range(n):
        dW = np.sqrt(dt)*rng.standard_normal(n_steps); x = x0
        for k in range(n_steps):
            u = c*x + b
            pen = spread*max(0.0, u - x)
            x += (u*(mkt.mu-mkt.r) + mkt.r*x - pen)*dt + u*mkt.sigma*dW[k]
        U[i] = util(max(x,1e-9))
    return U.mean()

def find_cstar(spread, n=50000):
    grid = np.linspace(0.3, 2.2, 40)
    vals = np.array([roll_EU(c, spread, n, np.random.default_rng(100+j)) for j,c in enumerate(grid)])
    j = np.argmax(vals); lo,hi = max(0,j-3),min(len(grid),j+4)
    co = np.polyfit(grid[lo:hi], vals[lo:hi], 2)
    return float(-co[1]/(2*co[0])), grid[j]

def crn_contrast(spread, cstar, n=80000):
    def paired(frac):
        rng=np.random.default_rng(21); d=np.empty(n); o=np.empty(n)
        for i in range(n):
            dW=np.sqrt(dt)*rng.standard_normal(n_steps); xo=x0; xh=x0
            for k in range(n_steps):
                uo=cstar*xo; uh=frac*cstar*xh
                po=spread*max(0.0,uo-xo); ph=spread*max(0.0,uh-xh)
                xo+=(uo*(mkt.mu-mkt.r)+mkt.r*xo-po)*dt+uo*mkt.sigma*dW[k]
                xh+=(uh*(mkt.mu-mkt.r)+mkt.r*xh-ph)*dt+uh*mkt.sigma*dW[k]
            d[i]=util(max(xo,1e-9))-util(max(xh,1e-9)); o[i]=util(max(xo,1e-9))
        return d.mean()/abs(o.mean())
    return abs(paired(0.7)), abs(paired(1.3))

print(f"{'spread':>7} {'c*':>7} {'gap@-30%':>9} {'gap@+30%':>9} {'contrast':>9}  linearity(best b)")
for s in [0.0, 0.05, 0.15, 0.40, 1.00]:
    cstar, cgrid = find_cstar(s)
    g_lo, g_hi = crn_contrast(s, cstar)
    bvals = {bb: roll_EU(cstar, s, 25000, np.random.default_rng(200), b=bb) for bb in [-0.1,0.0,0.1]}
    best_b = max(bvals, key=bvals.get)
    print(f"{s:7.2f} {cstar:7.3f} {100*g_lo:8.3f}% {100*g_hi:8.3f}% {100*max(g_lo,g_hi):8.3f}%   b={best_b:+.1f}")
