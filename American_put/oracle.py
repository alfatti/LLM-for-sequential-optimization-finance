"""
Oracle for a single American put, via CRR binomial tree.

Instance (frozen): S0=K=40, T=1, sigma=0.4, r=0.06, risk-neutral measure.

Provides:
  - the experiment's ground-truth value v* (the 50-step Bermudan price),
  - the optimal exercise boundary S*(t_k) at each lattice time level,
  - a recombining-lattice representation so rollouts land exactly on nodes
    and oracle labels are exact tree decisions (no interpolation).

Validation:
  - Richardson-style convergence of the price toward the literature value (~5.311),
  - an independent Longstaff-Schwartz price for cross-check.
"""

import numpy as np

# ----- frozen instance ------------------------------------------------------
S0    = 40.0
K     = 40.0
T     = 1.0
SIGMA = 0.4
R     = 0.06
PUT   = lambda S: np.maximum(K - S, 0.0)


# ----- CRR tree -------------------------------------------------------------
def crr_tree(n_steps, S0=S0, K=K, T=T, sigma=SIGMA, r=R):
    """
    Build a CRR tree for the American put.

    Returns a dict with:
      dt, u, d, p           : lattice parameters (risk-neutral up-prob p)
      price                 : American put value at (t=0, S0)  == v*
      S_levels[k]           : array of node prices at time level k (length k+1),
                              index j = number of up-moves, S = S0 * u^j * d^(k-j)
      cont[k], exer[k]      : continuation value and exercise value at each node
      exercise[k]           : bool array, True where immediate exercise is optimal
      Sstar[k]              : critical (highest) stock price at level k for which
                              exercise is optimal; np.nan if exercise is optimal
                              nowhere at that level
    """
    dt = T / n_steps
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    disc = np.exp(-r * dt)
    p  = (np.exp(r * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(f"risk-neutral prob out of (0,1): p={p}")

    # node prices per level: j up-moves out of k steps
    S_levels = []
    for k in range(n_steps + 1):
        j = np.arange(k + 1)
        S_levels.append(S0 * u**j * d**(k - j))

    # terminal payoff
    V = PUT(S_levels[n_steps])

    cont     = [None] * (n_steps + 1)
    exer     = [None] * (n_steps + 1)
    exercise = [None] * (n_steps + 1)
    Sstar    = [np.nan] * (n_steps + 1)

    # terminal level bookkeeping (continuation undefined at T; exercise = payoff)
    exer[n_steps]     = PUT(S_levels[n_steps])
    cont[n_steps]     = np.zeros(n_steps + 1)      # nothing to continue into
    exercise[n_steps] = exer[n_steps] > 0.0        # "exercise iff in the money"
    Sstar[n_steps]    = K  # S*(T) = K for r>0; in the money <=> S <= K at expiry

    # backward induction
    for k in range(n_steps - 1, -1, -1):
        # continuation = discounted risk-neutral expectation of next level
        c = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        e = PUT(S_levels[k])
        # exercise is optimal only where it is weakly better AND in the money;
        # the "in the money" guard removes the spurious 0>=0 ties at deep-OTM
        # nodes (payoff=cont=0), which are not part of the stopping region.
        ex = (e >= c) & (e > 0.0)
        V  = np.where(ex, e, c)

        cont[k]     = c
        exer[k]     = e
        exercise[k] = ex

        # boundary = highest node price where exercise is optimal (upper edge of
        # the low-price stopping region for a put).
        if ex.any():
            Sstar[k] = S_levels[k][ex].max()
        else:
            Sstar[k] = np.nan

    price = float(V[0])  # single node at k=0

    return dict(dt=dt, u=u, d=d, p=p, disc=disc, price=price,
                S_levels=S_levels, cont=cont, exer=exer,
                exercise=exercise, Sstar=np.array(Sstar),
                n_steps=n_steps, S0=S0, K=K, T=T, sigma=sigma, r=r)


def european_put_tree(n_steps, **kw):
    """European put on the same lattice, for the early-exercise premium reference."""
    t = crr_tree(n_steps, **kw)
    dt, u, d, disc, p = t["dt"], t["u"], t["d"], t["disc"], t["p"]
    n = n_steps
    S_levels = t["S_levels"]
    V = PUT(S_levels[n])
    for k in range(n - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
    return float(V[0])


# ----- independent cross-check: Longstaff-Schwartz --------------------------
def longstaff_schwartz(n_paths=200_000, n_steps=50, seed=0,
                       S0=S0, K=K, T=T, sigma=SIGMA, r=R, deg=3):
    """
    Regression Monte Carlo (Longstaff-Schwartz) American put price.
    Independent of the tree; used only to confirm the tree price.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    disc = np.exp(-r * dt)

    # simulate GBM under Q
    Z = rng.standard_normal((n_paths, n_steps))
    incr = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    logS = np.log(S0) + np.cumsum(incr, axis=1)
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    S[:, 1:] = np.exp(logS)

    # backward induction on cashflows
    cf = PUT(S[:, n_steps])            # cashflow if held to maturity
    for k in range(n_steps - 1, 0, -1):
        payoff = PUT(S[:, k])
        itm = payoff > 0.0
        if itm.sum() >= deg + 1:
            x = S[itm, k]
            y = cf[itm] * disc        # discounted continuation value one step
            # regress continuation on polynomial basis of x
            A = np.vander(x, deg + 1, increasing=True)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            cont_est = A @ coef
            exercise_now = payoff[itm] >= cont_est
            idx = np.where(itm)[0][exercise_now]
            cf[idx] = payoff[idx[..., None]].ravel() if False else payoff[itm][exercise_now]
            # discount the un-exercised cashflows by one step
            mask = np.ones(n_paths, dtype=bool)
            mask[idx] = False
            cf[mask] = cf[mask] * disc
        else:
            cf = cf * disc
    price = cf.mean() * disc
    stderr = cf.std(ddof=1) / np.sqrt(n_paths) * disc
    return float(price), float(stderr)


if __name__ == "__main__":
    print("=== CRR American put: convergence ===")
    for n in [10, 25, 50, 100, 250, 500, 1000, 2000, 5000]:
        t = crr_tree(n)
        print(f"  n_steps={n:5d}   price={t['price']:.5f}")

    t50 = crr_tree(50)
    eur50 = european_put_tree(50)
    print()
    print(f"v* (50-step American) = {t50['price']:.5f}   <-- experiment ground truth")
    print(f"European put (50-step) = {eur50:.5f}")
    print(f"early-exercise premium = {t50['price']-eur50:.5f}")
    print(f"risk-neutral up-prob p = {t50['p']:.5f}   u={t50['u']:.5f}  d={t50['d']:.5f}")

    print()
    print("=== independent Longstaff-Schwartz cross-check (50 steps) ===")
    lsp, lse = longstaff_schwartz(n_paths=200_000, n_steps=50, seed=1)
    print(f"  LS price = {lsp:.5f} +/- {lse:.5f}  (tree 50-step = {t50['price']:.5f})")

    print()
    print("=== optimal exercise boundary S*(t_k), 50-step lattice ===")
    Sstar = t50["Sstar"]
    for k in range(0, 51, 5):
        tk = k * t50["dt"]
        print(f"  t={tk:.2f}  S*={Sstar[k]:.4f}")
    print(f"  S*(T) = {Sstar[50]:.4f}  (should equal K={K})")
