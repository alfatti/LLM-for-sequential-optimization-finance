"""Many-tasks eval harness: coeff RMSE + CRN gap over a task DISTRIBUTION.
Primitives live in core.metrics; this adds the task-loop + winsorized reporting."""
import json, numpy as np
from core.metrics import estimator_ols, crn_gap_for_task

def evaluate(eval_path, estimator_fn, n_gap_paths=6000):
    tasks = [json.loads(l) for l in open(eval_path)]
    sq, gaps = [], []
    for t in tasks:
        ch = estimator_fn(t["support_text"]); 
        if np.isnan(ch): continue
        sq.append((ch - t["c_star"])**2)
        gaps.append(abs(crn_gap_for_task(t, ch, n_gap_paths, seed=t["task_id"])))
    gaps = np.array(gaps); wins = np.clip(gaps, None, np.percentile(gaps, 95))
    return dict(rmse=float(np.sqrt(np.mean(sq))), n=len(sq),
                gap_median=float(np.median(gaps)), gap_wins_mean=float(np.mean(wins)))

if __name__ == "__main__":
    for regime in ["benign","sharp"]:
        for split in ["id","ood"]:
            p = f"data/many_tasks/data_{regime}_eval_{split}.jsonl"
            try: r = evaluate(p, estimator_ols)
            except FileNotFoundError: print(f"{regime}/{split}: (no file)"); continue
            print(f"{regime:6s}/{split:3s} n={r['n']:3d} coeff RMSE={r['rmse']:.4f} "
                  f"gap med={100*r['gap_median']:.3f}% wins={100*r['gap_wins_mean']:.3f}%")
