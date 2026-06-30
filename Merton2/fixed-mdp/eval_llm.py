"""
Fixed-MDP ICL vs SFT vs random (GPU + HF). Aggregates over eval INSTANCES (one MDP, one c*).
  --adapter PATH   -> SFT (LoRA on base);   omit -> ICL-only (base model)
  --random         -> random-coefficient control
  --no_context     -> SFT ablation: act WITHOUT the support context (did it memorize c*?)

Usage:
  python -m fixed_mdp.eval_llm --eval data/fixed_mdp/sharp_eval.jsonl --adapter adapters/sharp
  python -m fixed_mdp.eval_llm --eval data/fixed_mdp/sharp_eval.jsonl                  # ICL-only
  python -m fixed_mdp.eval_llm --eval data/fixed_mdp/sharp_eval.jsonl --adapter adapters/sharp --no_context
"""
import argparse, json, numpy as np
from core.metrics import crn_gap_for_task

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--no_context", action="store_true")
    ap.add_argument("--n_gap_paths", type=int, default=8000)
    args = ap.parse_args()

    inst = [json.loads(l) for l in open(args.eval)]
    c_star = inst[0]["c_star"]
    if args.random:
        rng = np.random.default_rng(0); est = lambda s: float(rng.uniform(0.1, 2.0)); tag = "random"
    else:
        from core.llm import load, make_coeff_probe
        tok, model = load(args.adapter)
        est = make_coeff_probe(tok, model, use_context=not args.no_context)
        tag = ("SFT" if args.adapter else "ICL-only") + ("/no-ctx" if args.no_context else "")

    c_hats, gaps = [], []
    for t in inst:
        ch = est(t["support_text"])
        if np.isnan(ch): continue
        c_hats.append(ch); gaps.append(abs(crn_gap_for_task(t, ch, args.n_gap_paths, seed=t["instance_id"])))
    c_hats = np.array(c_hats); gaps = np.array(gaps)
    wins = np.clip(gaps, None, np.percentile(gaps, 95))
    print(f"[{tag}] {args.eval}: n={len(c_hats)}  c*={c_star:.4f}")
    print(f"   coeff RMSE={np.sqrt(np.mean((c_hats-c_star)**2)):.4f}  "
          f"mean(c_hat)={c_hats.mean():.4f}  bias={c_hats.mean()-c_star:+.4f}")
    print(f"   gap median={100*np.median(gaps):.3f}%  wins-mean={100*np.mean(wins):.3f}%")

if __name__ == "__main__":
    main()
