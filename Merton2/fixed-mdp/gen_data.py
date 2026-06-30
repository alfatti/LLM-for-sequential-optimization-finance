"""
Generate data for the fixed-MDP warm-up. Everything is rollouts of ONE MDP.

SFT example  = [K_demo noisy support rollouts] ++ [1 clean query rollout]   (loss on query)
Eval instance= [K_test noisy support rollouts]                              (ICL context)

Train and eval differ only by rollout SEED (disjoint), not by market -- there is no market
to hold out. c* is identical across all examples; the model learns THIS MDP's policy.

Usage:
  python -m fixed_mdp.gen_data --regime benign --n_train 600 --n_eval 150
  python -m fixed_mdp.gen_data --regime sharp  --n_train 600 --n_eval 150
"""
import json, argparse, numpy as np
from core.rollout import rollout_paths, oracle_c
from core.serialize import serialize_trajectory, SEP
from fixed_mdp.config import FIXED_MDP, REGIME_SPREAD, LABEL_NOISE

def one_rollout_text(mkt, c, spread, rng, label_noise):
    S, W, A, Rt = rollout_paths(mkt["mu"], mkt["sigma"], mkt["r"], c,
                                mkt["x0"], mkt["T"], mkt["n_steps"], 1, rng, spread=spread)
    S, W, A, Rt = S[0], W[0], A[0], Rt[0]
    if label_noise > 0:
        A = A * (1.0 + label_noise * rng.standard_normal(A.shape))
    return serialize_trajectory(S, W, A, Rt)

def make_training_examples(mkt, c, spread, n_train, k_demo, queries_per_ctx, label_noise, seed):
    rng = np.random.default_rng(seed); out = []
    n_ctx = max(1, n_train // queries_per_ctx)
    for _ in range(n_ctx):
        support = [one_rollout_text(mkt, c, spread, rng, label_noise) for _ in range(k_demo)]
        ctx = SEP.join(support)
        for _ in range(queries_per_ctx):
            query = one_rollout_text(mkt, c, spread, rng, label_noise=0.0)   # clean target
            text = ctx + SEP + query
            out.append(dict(regime_spread=spread, c_star=c,
                            market=dict(**{k: mkt[k] for k in ("mu","sigma","r","rho")}),
                            spread=spread, text=text, query_char_start=len(ctx + SEP)))
    return out

def make_eval_instances(mkt, c, spread, n_eval, k_test, label_noise, seed):
    rng = np.random.default_rng(seed); out = []
    for i in range(n_eval):
        support = [one_rollout_text(mkt, c, spread, rng, label_noise) for _ in range(k_test)]
        out.append(dict(instance_id=i, c_star=c,
                        market=dict(**{k: mkt[k] for k in ("mu","sigma","r","rho")}),
                        spread=spread, support_text=SEP.join(support), k_test=k_test))
    return out

def dump(rows, path):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="benign", choices=["benign","sharp"])
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_eval", type=int, default=150)
    ap.add_argument("--k_demo", type=int, default=4)
    ap.add_argument("--k_test", type=int, default=4)
    ap.add_argument("--queries_per_ctx", type=int, default=2)
    ap.add_argument("--label_noise", type=float, default=LABEL_NOISE)
    ap.add_argument("--outdir", default="data/fixed_mdp")
    args = ap.parse_args()

    mkt = FIXED_MDP; spread = REGIME_SPREAD[args.regime]
    c = oracle_c(mkt["mu"], mkt["sigma"], mkt["r"], mkt["rho"], args.regime,
                 x0=mkt["x0"], T=mkt["T"], n_steps=mkt["n_steps"], spread=spread, n_paths=40000)
    tr  = make_training_examples(mkt, c, spread, args.n_train, args.k_demo,
                                 args.queries_per_ctx, args.label_noise, seed=1)
    ev  = make_eval_instances(mkt, c, spread, args.n_eval, args.k_test, args.label_noise, seed=2)
    p1 = dump(tr, f"{args.outdir}/{args.regime}_train.jsonl")
    p2 = dump(ev, f"{args.outdir}/{args.regime}_eval.jsonl")
    print(f"[fixed-MDP {args.regime}] c*={c:.4f}  spread={spread}")
    print(f"  train examples={len(tr)} -> {p1}")
    print(f"  eval instances={len(ev)} -> {p2}")
    print(f"  sample (first 200 chars): {tr[0]['text'][:200]}")
