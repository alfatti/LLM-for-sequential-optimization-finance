"""
High-curvature regime via a quadratic holding/financing cost on the position:
   J(c) = E[U(x_T)] - (kappa/2) E[ sum_k u_k^2 dt ]
The cost penalizes large/levered positions, sharpening the optimum WITHOUT needing
large leverage. We (1) confirm the optimal policy stays ~linear u=c*x so the
"infer a scalar" task is preserved, (2) find c*(kappa) by numerical EU-maximization
(exact over the policy class), (3) measure optimality-gap contrast vs kappa.
"""
import numpy as np
from core.dynamics import Market, crra_utility
mkt = Market(mu=0.12, sigma=0.20, r=0.03); rho = 2.0
T, x0, n_steps = 1.0, 1.0, 10; dt = T/n_steps
util = lambda x: crra_utility(x, rho)

def objective(c, kappa, n, rng, b=0.0):
    """E[U(x_T)] - (kappa/2) E[sum u^2 dt], policy u = c*x + b."""
    U = np.empty(n); C = np.empty(n)
    for i in range(n):
        dW = np.sqrt(dt)*rng.standard_normal(n_steps); x = x0; cost = 0.0
        for k in range(n_steps):
            u = c*x + b; cost += u*u*dt
            x += (u*(mkt.mu-mkt.r)+mkt.r*x)*dt + u*mkt.sigma*dW[k]
        U[i] = util(max(x,1e-9)); C[i] = cost
    return U.mean() - 0.5*kappa*C.mean()

def find_cstar(kappa, grid=None, n=40000):
    rng0 = np.random.default_rng(0)
    if grid is None: grid = np.linspace(0.05, 1.2, 40)
    vals = np.array([objective(c, kappa, n, np.random.default_rng(100+j))
                     for j, c in enumerate(grid)])
    j = np.argmax(vals)
    # quadratic refine
    lo, hi = max(0,j-3), min(len(grid),j+4)
    co = np.polyfit(grid[lo:hi], vals[lo:hi], 2)
    return -co[1]/(2*co[0]), grid, vals

def contrast(kappa, cstar, n=60000):
    """CRN optimality gap (penalized objective) at +/-30% coefficient error."""
    def Jpaired(frac):
        rng = np.random.default_rng(21); d=np.empty(n); o=np.empty(n)
        for i in range(n):
            dW=np.sqrt(dt)*rng.standard_normal(n_steps); xo=x0; xh=x0; co_=0.0; ch_=0.0
            for k in range(n_steps):
                uo=cstar*xo; uh=frac*cstar*xh; co_+=uo*uo*dt; ch_+=uh*uh*dt
                xo+=(uo*(mkt.mu-mkt.r)+mkt.r*xo)*dt+uo*mkt.sigma*dW[k]
                xh+=(uh*(mkt.mu-mkt.r)+mkt.r*xh)*dt+uh*mkt.sigma*dW[k]
            Jo=util(max(xo,1e-9))-0.5*kappa*co_; Jh=util(max(xh,1e-9))-0.5*kappa*ch_
            d[i]=Jo-Jh; o[i]=Jo
        return d.mean()/abs(o.mean())
    return max(abs(Jpaired(0.7)), abs(Jpaired(1.3)))

print(f"{'kappa':>7} {'c*':>7} {'contrast@30%':>13}  linearity check (best b at c*)")
for kappa in [0.0, 0.5, 2.0, 5.0, 10.0, 20.0]:
    cstar, grid, vals = find_cstar(kappa)
    ct = contrast(kappa, cstar)
    # linearity: at fixed c*, does adding an intercept b help?
    rng = np.random.default_rng(7)
    bvals = {b: objective(cstar, kappa, 20000, np.random.default_rng(200), b=b)
             for b in [-0.1, 0.0, 0.1]}
    best_b = max(bvals, key=bvals.get)
    print(f"{kappa:7.1f} {cstar:7.3f} {100*ct:11.2f}%   best b={best_b:+.1f} "
          f"(J: {bvals[-0.1]:.4f}/{bvals[0.0]:.4f}/{bvals[0.1]:.4f})")
