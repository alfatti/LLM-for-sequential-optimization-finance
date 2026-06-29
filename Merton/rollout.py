"""
Vectorized rollout + oracle solver for all three regimes.
  regime='benign' : plain Merton, oracle c* = (mu-r)/(sigma^2 rho)  [closed form]
  regime='sharp'  : borrowing-spread Merton, oracle c* found numerically (golden section)
  regime='exp'    : exponential utility, state-INDEPENDENT schedule u*(t)
All rollouts are vectorized over paths for speed.
"""
import numpy as np
from merton import crra_utility, exp_utility

def rollout_paths(mu, sigma, r, c, x0, T, n_steps, n_paths, rng, spread=0.0, dW=None):
    """Vectorized: roll n_paths under policy u=c*x with optional borrowing spread.
       If dW is given (n_paths,n_steps) it is reused (common random numbers).
       Returns prices(n_paths,n_steps+1), wealth(n_paths,n_steps+1), actions, returns."""
    dt = T / n_steps; sq = np.sqrt(dt)
    if dW is None:
        dW = sq * rng.standard_normal((n_paths, n_steps))
    W = np.empty((n_paths, n_steps + 1)); W[:, 0] = x0
    S = np.empty((n_paths, n_steps + 1)); S[:, 0] = 1.0
    A = np.empty((n_paths, n_steps)); Rt = np.empty((n_paths, n_steps))
    for k in range(n_steps):
        dWk = dW[:, k]
        u = c * W[:, k]
        A[:, k] = u
        pen = spread * np.maximum(0.0, u - W[:, k])
        Wn = W[:, k] + (u*(mu - r) + r*W[:, k] - pen)*dt + u*sigma*dWk
        Wn = np.maximum(Wn, 1e-9)
        W[:, k+1] = Wn
        Rt[:, k] = (Wn - W[:, k]) / np.maximum(W[:, k], 1e-9)   # step return
        S[:, k+1] = S[:, k] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dWk)
    return S, W, A, Rt

def expected_utility_c(mu, sigma, r, c, rho, x0, T, n_steps, n_paths, rng, spread=0.0):
    _, W, _, _ = rollout_paths(mu, sigma, r, c, x0, T, n_steps, n_paths, rng, spread)
    return crra_utility(W[:, -1], rho).mean()

def oracle_c(mu, sigma, r, rho, regime, x0=1.0, T=1.0, n_steps=10, spread=0.4,
             n_paths=20000, seed=0):
    """Return the oracle coefficient c* for the given regime."""
    if regime == "benign":
        return (mu - r) / (sigma**2 * rho)
    if regime == "sharp":
        # CRN grid search: evaluate EU(c) for all c on the SAME Brownian paths, so
        # the empirical EU(c) is a smooth function of c and the argmax is stable.
        dt = T/n_steps
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt)*rng.standard_normal((n_paths, n_steps))
        grid = np.linspace(0.3, 1.6, 53)
        eus = np.empty_like(grid)
        for j, c in enumerate(grid):
            _, W, _, _ = rollout_paths(mu, sigma, r, c, x0, T, n_steps, n_paths,
                                       rng, spread, dW=dW)
            eus[j] = crra_utility(W[:, -1], rho).mean()
        j = int(np.argmax(eus))
        lo_i, hi_i = max(0, j-3), min(len(grid), j+4)
        co = np.polyfit(grid[lo_i:hi_i], eus[lo_i:hi_i], 2)
        c_ref = -co[1]/(2*co[0]) if co[0] < 0 else grid[j]
        return float(np.clip(c_ref, grid[0], grid[-1]))
    raise ValueError(regime)

def exp_schedule(mu, sigma, r, gamma, T, n_steps):
    """State-independent optimal dollar schedule u*(t) for exponential utility."""
    lam = (mu - r)/sigma
    ts = np.linspace(0, T, n_steps+1)[:-1]
    return (lam/(gamma*sigma)) * np.exp(-r*(T - ts))   # length n_steps

if __name__ == "__main__":
    # sanity: sharp oracle reproduces Phase-0 baseline ~0.91
    mu, sigma, r, rho = 0.12, 0.20, 0.03, 2.0
    print("benign c* =", round(oracle_c(mu,sigma,r,rho,"benign"),4), "(expect 1.125)")
    print("sharp  c* =", round(oracle_c(mu,sigma,r,rho,"sharp",spread=0.4,n_paths=40000),4),
          "(expect ~0.91)")
