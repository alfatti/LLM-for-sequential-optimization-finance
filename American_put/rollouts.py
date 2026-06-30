"""
Rollout generator for the single American put MDP.

Generates on-lattice risk-neutral paths and labels each step with the EXACT
tree-optimal decision (STOP / GO) read from the oracle's per-node exercise map.
Because paths live on the recombining lattice, labels are exact -- no boundary
interpolation, no labeling noise (the clean core run).

Two trajectory styles:
  - "optimal": follow the optimal policy, truncate at first STOP (faithful
    analog of Paper 1's optimal-policy rollouts; this is the SFT/ICL data).
  - "full"   : ignore the policy and walk to maturity, labeling every visited
    node. Used only to give coverage of the deep stopping-region interior if
    the optimal-policy data proves too thin (optional; off by default).

Serialization (version A, per-step binary):
    <t_k> <S_k> <DECISION>
  rendered as a compact line per step. State is (t, S) only -- the put MDP is
  Markovian and fully observed.

Outputs:
  - rollouts_optimal.jsonl : one JSON object per trajectory (raw, for analysis)
  - sft_text.jsonl         : one serialized training string per trajectory
  - meta.json              : instance + lattice params + v* + boundary
"""

import json
import numpy as np
from oracle import crr_tree, european_put_tree, S0, K, T, SIGMA, R, PUT


def node_price(tree, k, j):
    return tree["S0"] * tree["u"]**j * tree["d"]**(k - j)


def simulate_lattice_paths(tree, n_paths, seed, style="optimal"):
    """
    Simulate paths on the recombining lattice under the risk-neutral measure.

    A path is a random walk in the up-count j: at each step j -> j+1 w.p. p,
    else j stays (down move keeps j, since S_{k+1,down}=S0 u^j d^{k+1-j}).
    Equivalently we track cumulative up-moves.

    Returns a list of trajectories; each trajectory is a list of
    (k, j, S, payoff, optimal_decision) tuples, where optimal_decision is
    1 for STOP (exercise) and 0 for GO (continue), read from tree["exercise"].
    For style="optimal" the trajectory is truncated at the first STOP.
    """
    rng = np.random.default_rng(seed)
    p = tree["p"]
    n = tree["n_steps"]
    exercise = tree["exercise"]   # list of bool arrays indexed [k][j]

    trajs = []
    for _ in range(n_paths):
        j = 0
        traj = []
        for k in range(n + 1):
            S = node_price(tree, k, j)
            payoff = float(max(K - S, 0.0))
            if k < n:
                dec = int(exercise[k][j])     # 1=STOP, 0=GO
            else:
                # terminal: exercise iff in the money
                dec = int(payoff > 0.0)
            traj.append((k, j, float(S), payoff, dec))

            if style == "optimal" and dec == 1:
                break   # optimal policy stops here
            if k < n:
                # advance lattice: up-move increments j
                if rng.random() < p:
                    j += 1
                # down-move: j unchanged
        trajs.append(traj)
    return trajs


def serialize(traj, fmt="compact"):
    """
    Serialize a trajectory into a training string (version A, per-step binary).
    Each step -> "t=<k> S=<S> -> <DECISION>". Decisions: STOP / GO.
    """
    toks = []
    for (k, j, S, payoff, dec) in traj:
        d = "STOP" if dec == 1 else "GO"
        toks.append(f"t={k} S={S:.4f} -> {d}")
    return " ; ".join(toks)


def discounted_payoff_of_traj(traj, tree):
    """Realized discounted payoff of a trajectory under its (here optimal) policy."""
    # find the stop step (first dec==1); optimal-style trajectories end at stop
    for (k, j, S, payoff, dec) in traj:
        if dec == 1:
            return np.exp(-tree["r"] * k * tree["dt"]) * payoff
    # never stopped before terminal -> terminal decision already encoded as last dec
    k, j, S, payoff, dec = traj[-1]
    return np.exp(-tree["r"] * k * tree["dt"]) * (payoff if dec == 1 else 0.0)


def build_dataset(n_train=2000, n_eval=5000, n_steps=50, seed=0):
    tree = crr_tree(n_steps)
    eur = european_put_tree(n_steps)

    # training trajectories (optimal policy, truncated at stop)
    train = simulate_lattice_paths(tree, n_train, seed=seed, style="optimal")
    # held-out eval trajectories (fresh paths, same instance)
    eval_ = simulate_lattice_paths(tree, n_eval, seed=seed + 10_000, style="optimal")

    # sanity: Monte-Carlo price of the optimal policy on eval paths == v*
    pvs = np.array([discounted_payoff_of_traj(t, tree) for t in eval_])
    mc_price = pvs.mean()
    mc_se = pvs.std(ddof=1) / np.sqrt(len(pvs))

    meta = dict(
        instance=dict(S0=S0, K=K, T=T, sigma=SIGMA, r=R, measure="risk-neutral"),
        lattice=dict(n_steps=n_steps, u=tree["u"], d=tree["d"], p=tree["p"],
                     dt=tree["dt"]),
        v_star=tree["price"],
        european_put=eur,
        early_exercise_premium=tree["price"] - eur,
        boundary_Sstar=[None if np.isnan(x) else float(x) for x in tree["Sstar"]],
        optimal_policy_mc_price=float(mc_price),
        optimal_policy_mc_stderr=float(mc_se),
        n_train=n_train, n_eval=n_eval,
    )

    # write raw rollouts
    with open("rollouts_optimal.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(dict(
                steps=[dict(k=k, j=j, S=S, payoff=pay, decision=dec)
                       for (k, j, S, pay, dec) in t])) + "\n")

    # write serialized SFT strings
    with open("sft_text.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(dict(text=serialize(t))) + "\n")

    with open("meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return tree, train, eval_, meta


if __name__ == "__main__":
    tree, train, eval_, meta = build_dataset(n_train=2000, n_eval=5000, n_steps=50)

    print("=== dataset built ===")
    print(f"v* (ground truth)            : {meta['v_star']:.5f}")
    print(f"optimal-policy MC price (eval): {meta['optimal_policy_mc_price']:.5f} "
          f"+/- {meta['optimal_policy_mc_stderr']:.5f}")
    print(f"  -> these should agree, confirming rollouts + labels are consistent")
    print(f"European put / early-ex premium: {meta['european_put']:.5f} / "
          f"{meta['early_exercise_premium']:.5f}")
    print()

    # trajectory length distribution (how long until optimal stop)
    lens = np.array([len(t) for t in train])
    stops = np.array([t[-1][4] for t in train])  # last decision
    print(f"train trajectories           : {len(train)}")
    print(f"  stop-step (len) mean/median/min/max: "
          f"{lens.mean():.1f}/{np.median(lens):.0f}/{lens.min()}/{lens.max()}")
    print(f"  fraction ending in STOP     : {stops.mean():.3f}")
    frac_exercised = np.mean([any(s[4]==1 for s in t) for t in train])
    print(f"  fraction ever exercising    : {frac_exercised:.3f}")
    print(f"  (rest expire worthless OTM at maturity)")
    print()
    print("=== sample serialized trajectories ===")
    for t in train[:3]:
        s = serialize(t)
        print("  " + (s if len(s) < 200 else s[:200] + " ..."))
    print()
    print("Files: rollouts_optimal.jsonl, sft_text.jsonl, meta.json")
