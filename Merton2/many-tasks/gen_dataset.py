"""
Generate SFT datasets and eval task specs for the Merton replication.

Training example = [K_demo noisy support trajectories] ++ [1 clean query trajectory].
  - support actions carry multiplicative label noise (Finding 2: makes K_test informative)
  - query actions are clean optimal (the imitation target)
  - 'query_char_start' marks where the query begins so the trainer masks loss to it.

Eval task = market + c_star + K_test serialized noisy support trajectories (context).
"""
import json, numpy as np, argparse
from core.rollout import rollout_paths, oracle_c
from core.serialize import serialize_trajectory, A_DECIMALS

ID_RANGES  = dict(mu=(0.06, 0.14), sigma=(0.15, 0.25), r=(0.01, 0.05))
OOD_RANGES = dict(mu=(0.14, 0.20), sigma=(0.25, 0.35), r=(0.01, 0.05))  # disjoint, c* bounded
RHO = 2.0
T, X0, N_STEPS = 1.0, 1.0, 10
SEP = " <SEP> "

def sample_market(ranges, rng):
    return (rng.uniform(*ranges["mu"]), rng.uniform(*ranges["sigma"]), rng.uniform(*ranges["r"]))

def one_trajectory(mu, sigma, r, c, regime, rng, label_noise):
    spread = 0.4 if regime == "sharp" else 0.0
    S, W, A, Rt = rollout_paths(mu, sigma, r, c, X0, T, N_STEPS, 1, rng, spread=spread)
    S, W, A, Rt = S[0], W[0], A[0], Rt[0]
    if label_noise > 0:
        A = A * (1.0 + label_noise * rng.standard_normal(A.shape))   # noisy recorded action
    return serialize_trajectory(S, W, A, Rt)

def make_training_examples(n_tasks, regime, k_demo, queries_per_task, label_noise, seed):
    rng = np.random.default_rng(seed)
    out = []
    for tid in range(n_tasks):
        mu, sigma, r = sample_market(ID_RANGES, rng)
        c = oracle_c(mu, sigma, r, RHO, regime, x0=X0, T=T, n_steps=N_STEPS,
                     spread=0.4, n_paths=20000, seed=tid)
        # K_demo noisy support trajectories
        support = [one_trajectory(mu, sigma, r, c, regime, rng, label_noise)
                   for _ in range(k_demo)]
        ctx = SEP.join(support)
        for _ in range(queries_per_task):
            query = one_trajectory(mu, sigma, r, c, regime, rng, label_noise=0.0)  # clean target
            text = ctx + SEP + query
            out.append(dict(task_id=tid, regime=regime,
                            market=dict(mu=mu, sigma=sigma, r=r, rho=RHO),
                            c_star=c, spread=(0.4 if regime=="sharp" else 0.0),
                            text=text, query_char_start=len(ctx + SEP)))
    return out

def make_eval_tasks(n_tasks, regime, k_test, label_noise, ranges, seed):
    rng = np.random.default_rng(seed)
    out = []
    for tid in range(n_tasks):
        mu, sigma, r = sample_market(ranges, rng)
        c = oracle_c(mu, sigma, r, RHO, regime, x0=X0, T=T, n_steps=N_STEPS,
                     spread=0.4, n_paths=20000, seed=10_000+tid)
        support = [one_trajectory(mu, sigma, r, c, regime, rng, label_noise)
                   for _ in range(k_test)]
        out.append(dict(task_id=tid, regime=regime,
                        market=dict(mu=mu, sigma=sigma, r=r, rho=RHO),
                        c_star=c, spread=(0.4 if regime=="sharp" else 0.0),
                        support_text=SEP.join(support), k_test=k_test))
    return out

def dump(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="benign", choices=["benign","sharp","exp"])
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--k_demo", type=int, default=4)
    ap.add_argument("--k_test", type=int, default=4)
    ap.add_argument("--queries_per_task", type=int, default=2)
    ap.add_argument("--label_noise", type=float, default=0.25)
    ap.add_argument("--out_prefix", default="data/many_tasks/data")
    args = ap.parse_args()

    tr = make_training_examples(args.n_train, args.regime, args.k_demo,
                                args.queries_per_task, args.label_noise, seed=1)
    ev_id = make_eval_tasks(args.n_eval, args.regime, args.k_test, args.label_noise,
                            ID_RANGES, seed=2)
    ev_ood = make_eval_tasks(args.n_eval, args.regime, args.k_test, args.label_noise,
                             OOD_RANGES, seed=3)
    dump(tr, f"{args.out_prefix}_{args.regime}_train.jsonl")
    dump(ev_id, f"{args.out_prefix}_{args.regime}_eval_id.jsonl")
    dump(ev_ood, f"{args.out_prefix}_{args.regime}_eval_ood.jsonl")
    cs = [r["c_star"] for r in tr]
    print(f"[{args.regime}] train ex={len(tr)} eval_id={len(ev_id)} eval_ood={len(ev_ood)}")
    print(f"  c* range over train tasks: [{min(cs):.3f}, {max(cs):.3f}]  mean {np.mean(cs):.3f}")
    print(f"  sample text (first 240 chars):\n  {tr[0]['text'][:240]}")
