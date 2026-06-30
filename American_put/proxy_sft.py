"""
CPU PROXY for the American-put SFT experiment.

Purpose: validate the experimental DESIGN end-to-end without a GPU. The model
here is a small stop-go classifier (logistic regression on (t, S) features),
standing in for the Llama-2-7B SFT policy. Everything AROUND the model -- the
serialized data, the evaluation rollouts, and the three metrics -- is identical
to what the real H200 run will use. Only the policy's predict() swaps out.

This de-risks:
  - the evaluation harness (roll a policy forward on fresh lattice paths),
  - the three metrics (value gap / boundary error / stopping-time error) and
    their decoupling,
  - the data-coverage prediction (boundary learnable on its upper edge,
    unconstrained deep in the stopping region),
  - the GO/STOP imbalance effect.

Conditions implemented (mirror the real experiment):
  - "SFT"   : classifier trained on optimal-policy rollout labels.
  - "ICL"   : k-NN in (t,S) over a few demonstration trajectories -- a
              training-free, in-context analog (predict STOP/GO by nearest
              demonstrated decisions). Stands in for the LLM's ICL-only policy.
  - "random": stop w.p. 0.5 each step.
  - references: oracle (gap 0) and never-exercise (European).
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from oracle import crr_tree, european_put_tree, K
from rollouts import simulate_lattice_paths, node_price


# ----- feature representation ----------------------------------------------
def steps_to_XY(trajs, n_steps):
    """
    Flatten trajectories into per-step (features, label) for the classifier.
    Features: normalized time k/N, stock S, moneyness (K-S), log(S/K).
    Label: optimal decision (1=STOP, 0=GO).
    Terminal step (k=N) is excluded from training -- the decision there is the
    trivial "exercise iff ITM" and is handled deterministically at eval.
    """
    X, Y = [], []
    for t in trajs:
        for (k, j, S, payoff, dec) in t:
            if k == n_steps:
                continue
            X.append([k / n_steps, S, K - S, np.log(S / K)])
            Y.append(dec)
    return np.array(X, dtype=float), np.array(Y, dtype=int)


def featurize_node(k, S, n_steps):
    return np.array([[k / n_steps, S, K - S, np.log(S / K)]], dtype=float)


# ----- policies -------------------------------------------------------------
class SFTProxyPolicy:
    """Logistic-regression stop-go classifier -- proxy for the SFT LLM policy."""
    def __init__(self, n_steps, class_weight=None):
        self.n_steps = n_steps
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, class_weight=class_weight)

    def fit(self, trajs):
        X, Y = steps_to_XY(trajs, self.n_steps)
        # guard: if a class is absent (rare at tiny N), fall back gracefully
        self.single_class = None
        if len(np.unique(Y)) < 2:
            self.single_class = int(Y[0])
            return self
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, Y)
        return self

    def decide(self, k, S):
        if k == self.n_steps:
            return int((K - S) > 0.0)               # terminal: exercise iff ITM
        if self.single_class is not None:
            return self.single_class
        x = self.scaler.transform(featurize_node(k, S, self.n_steps))
        return int(self.clf.predict(x)[0])

    def boundary(self, tree):
        """Reconstruct implied S*(t_k): highest S the policy would STOP at."""
        return _reconstruct_boundary(self.decide, tree)


class ICLProxyPolicy:
    """k-NN over demonstration steps -- training-free in-context analog."""
    def __init__(self, n_steps, k_neighbors=5):
        self.n_steps = n_steps
        self.k = k_neighbors
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=k_neighbors)

    def fit(self, demo_trajs):
        X, Y = steps_to_XY(demo_trajs, self.n_steps)
        self.single_class = None
        if len(np.unique(Y)) < 2:
            self.single_class = int(Y[0])
            return self
        self.k = min(self.k, len(Y))
        self.knn.set_params(n_neighbors=self.k)
        Xs = self.scaler.fit_transform(X)
        self.knn.fit(Xs, Y)
        return self

    def decide(self, k, S):
        if k == self.n_steps:
            return int((K - S) > 0.0)
        if self.single_class is not None:
            return self.single_class
        x = self.scaler.transform(featurize_node(k, S, self.n_steps))
        return int(self.knn.predict(x)[0])

    def boundary(self, tree):
        return _reconstruct_boundary(self.decide, tree)


class RandomPolicy:
    def __init__(self, n_steps, seed=0):
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)
    def decide(self, k, S):
        if k == self.n_steps:
            return int((K - S) > 0.0)
        return int(self.rng.random() < 0.5)


class OraclePolicy:
    """Exact tree-optimal decision; value gap must be ~0 (MC noise only)."""
    def __init__(self, tree):
        self.tree = tree
        self.n_steps = tree["n_steps"]
    def decide_node(self, k, j, S):
        if k == self.n_steps:
            return int((K - S) > 0.0)
        return int(self.tree["exercise"][k][j])


def _reconstruct_boundary(decide_fn, tree):
    """
    For each time level k, the implied critical price = highest lattice price
    at which the policy decides STOP. NaN if it never stops at that level.
    Lets us compute boundary error against the oracle S*(t_k).
    """
    n = tree["n_steps"]
    Sstar_hat = np.full(n + 1, np.nan)
    for k in range(n + 1):
        stop_prices = []
        for j in range(k + 1):
            S = node_price(tree, k, j)
            if decide_fn(k, S) == 1:
                stop_prices.append(S)
        if stop_prices:
            Sstar_hat[k] = max(stop_prices)
    return Sstar_hat


# ----- evaluation harness (identical for every policy, incl. the real LLM) --
def precompute_decisions(policy, tree):
    """
    Evaluate the policy's decision at every lattice node ONCE, returning a
    list-of-arrays dec_table[k][j] in {0,1}. Rollout then looks up decisions
    by (k,j) instead of calling the model per step -- the same trick the real
    LLM eval will use (batch the queries, don't re-run per path).
    """
    n = tree["n_steps"]
    is_oracle = isinstance(policy, OraclePolicy)
    dec_table = []
    for k in range(n + 1):
        row = np.empty(k + 1, dtype=int)
        for j in range(k + 1):
            S = node_price(tree, k, j)
            if is_oracle:
                row[j] = policy.decide_node(k, j, S)
            else:
                row[j] = policy.decide(k, S)
        dec_table.append(row)
    return dec_table


def evaluate_policy(policy, eval_trajs, tree, oracle_stop_steps=None):
    """
    Roll each eval path forward under `policy`, recording discounted payoff and
    stop time. eval_trajs are pre-simulated lattice paths (SAME fresh paths for
    every policy, for paired comparison). Decisions are precomputed on the
    lattice and looked up by (k,j) -- no per-step model calls.
    """
    n = tree["n_steps"]
    dt, r = tree["dt"], tree["r"]
    v_star = tree["price"]
    dec_table = precompute_decisions(policy, tree)

    pvs = np.empty(len(eval_trajs))
    stop_steps = np.empty(len(eval_trajs), dtype=int)
    for i, t in enumerate(eval_trajs):
        stopped = False
        for (k, j, S, payoff, dec) in t:
            if dec_table[k][j] == 1:
                pvs[i] = np.exp(-r * k * dt) * max(K - S, 0.0)
                stop_steps[i] = k
                stopped = True
                break
        if not stopped:
            k, j, S, payoff, dec = t[-1]
            pvs[i] = np.exp(-r * k * dt) * max(K - S, 0.0)
            stop_steps[i] = k

    out = dict(
        mean_pv=float(pvs.mean()),
        se_pv=float(pvs.std(ddof=1) / np.sqrt(len(pvs))),
        value_gap=float((v_star - pvs.mean()) / v_star),
        mean_stop=float(stop_steps.mean()),
        stop_steps=stop_steps,
    )
    if oracle_stop_steps is not None:
        st_err = stop_steps - oracle_stop_steps
        out["stoptime_err_mean"] = float(st_err.mean())
        out["stoptime_err_abs_mean"] = float(np.abs(st_err).mean())
        out["stoptime_err_frac_exact"] = float((st_err == 0).mean())
    return out


def boundary_error(Sstar_hat, Sstar_true):
    """L1/Linf boundary error over levels where BOTH are defined."""
    mask = ~(np.isnan(Sstar_hat) | np.isnan(Sstar_true))
    if mask.sum() == 0:
        return dict(l1=np.nan, linf=np.nan, n_levels=0)
    diff = np.abs(Sstar_hat[mask] - Sstar_true[mask])
    return dict(l1=float(diff.mean()), linf=float(diff.max()),
                n_levels=int(mask.sum()))


# ----- oracle stop steps on the eval paths (for stopping-time error) --------
def oracle_stop_steps_on(eval_trajs, tree):
    """The optimal stop step for each eval path (paths were generated optimally,
    so the encoded first-STOP is the oracle stop)."""
    steps = []
    n = tree["n_steps"]
    for t in eval_trajs:
        s = n
        for (k, j, S, payoff, dec) in t:
            if dec == 1:
                s = k
                break
        steps.append(s)
    return np.array(steps)


def run_proxy(n_steps=50, seed=0, N_sweep=(50, 100, 200, 500, 1000, 2000),
              n_eval=20000, k_icl_demos=2, icl_neighbors=5):
    tree = crr_tree(n_steps)
    Sstar_true = tree["Sstar"]
    eur = european_put_tree(n_steps)

    # fresh held-out eval paths, shared across all policies (paired).
    # CRITICAL: eval paths must be FULL untruncated walks to maturity, so every
    # policy can be rolled forward to ITS OWN stop decision. (Truncating at the
    # oracle's stop would bias every other policy.)
    eval_trajs = simulate_lattice_paths(tree, n_eval, seed=seed + 99, style="full")
    oracle_steps = oracle_stop_steps_on(eval_trajs, tree)

    print(f"instance: S0=K={K}, sigma={tree['sigma']}, r={tree['r']}, "
          f"n_steps={n_steps}")
    print(f"v* = {tree['price']:.5f}   European = {eur:.5f}   "
          f"early-ex premium = {tree['price']-eur:.5f}")
    print(f"eval paths: {n_eval} (shared, paired)\n")

    # --- reference policies ---
    orac = OraclePolicy(tree)
    oracle_eval = evaluate_policy(orac, eval_trajs, tree, oracle_steps)
    print(f"{'ORACLE':<22} value_gap={oracle_eval['value_gap']:+.4f}  "
          f"mean_pv={oracle_eval['mean_pv']:.4f}  "
          f"(sanity: gap ~ MC noise)")

    rnd = RandomPolicy(n_steps, seed=seed)
    rnd_eval = evaluate_policy(rnd, eval_trajs, tree, oracle_steps)
    print(f"{'RANDOM':<22} value_gap={rnd_eval['value_gap']:+.4f}  "
          f"mean_pv={rnd_eval['mean_pv']:.4f}  "
          f"stoptime_abs_err={rnd_eval['stoptime_err_abs_mean']:.2f}")

    # never-exercise (European) is a degenerate policy: gap vs v*
    print(f"{'NEVER-EXERCISE (Eur)':<22} value_gap="
          f"{(tree['price']-eur)/tree['price']:+.4f}  mean_pv={eur:.4f}\n")

    # --- ICL-only (training-free), fixed small number of demos ---
    demo_trajs = simulate_lattice_paths(tree, k_icl_demos, seed=seed + 1,
                                        style="optimal")
    icl = ICLProxyPolicy(n_steps, k_neighbors=icl_neighbors).fit(demo_trajs)
    icl_eval = evaluate_policy(icl, eval_trajs, tree, oracle_steps)
    icl_bd = boundary_error(icl.boundary(tree), Sstar_true)
    print(f"{'ICL ('+str(k_icl_demos)+' demos)':<22} "
          f"value_gap={icl_eval['value_gap']:+.4f}  "
          f"bd_err_L1={icl_bd['l1']:.3f}  bd_err_Linf={icl_bd['linf']:.3f}  "
          f"stoptime_abs_err={icl_eval['stoptime_err_abs_mean']:.2f}\n")

    # --- SFT proxy across training-size sweep ---
    print("SFT proxy (logistic stop-go), training-size sweep:")
    print(f"  {'N_train':>8}  {'value_gap':>10}  {'bd_L1':>7}  {'bd_Linf':>8}  "
          f"{'st_abs':>7}  {'st_exact':>8}")
    rng_master = np.random.default_rng(seed + 7)
    for N in N_sweep:
        train_trajs = simulate_lattice_paths(
            tree, N, seed=int(rng_master.integers(1e9)), style="optimal")
        sft = SFTProxyPolicy(n_steps).fit(train_trajs)
        ev = evaluate_policy(sft, eval_trajs, tree, oracle_steps)
        bd = boundary_error(sft.boundary(tree), Sstar_true)
        print(f"  {N:>8}  {ev['value_gap']:>+10.4f}  {bd['l1']:>7.3f}  "
              f"{bd['linf']:>8.3f}  {ev['stoptime_err_abs_mean']:>7.2f}  "
              f"{ev['stoptime_err_frac_exact']:>8.3f}")

    # detailed boundary comparison at largest N
    print("\nlearned vs true boundary (largest N), every 5 levels:")
    Sstar_hat = sft.boundary(tree)
    print(f"  {'t':>5}  {'S*_true':>8}  {'S*_hat':>8}")
    for k in range(0, n_steps + 1, 5):
        tt = f"{k/n_steps:.2f}"
        a = Sstar_true[k]; b = Sstar_hat[k]
        sa = f"{a:.3f}" if not np.isnan(a) else "  --  "
        sb = f"{b:.3f}" if not np.isnan(b) else "  --  "
        print(f"  {tt:>5}  {sa:>8}  {sb:>8}")


if __name__ == "__main__":
    run_proxy()
