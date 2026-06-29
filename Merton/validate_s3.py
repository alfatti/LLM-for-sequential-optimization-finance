"""Validate S3 outcome analysis with mock models (no LLM). Each mock parses the
counterfactual (W,A) from the probe prefix and returns a predicted step return.
We confirm the analysis separates: ideal world-model / shortcut / no-kink."""
import json, re, numpy as np
from outcome_analysis import analyze_eval_set, theoretical_return

# parse the LAST "<W> w <A> a <R>" in the probe prefix
_tail = re.compile(r"<W>\s*(-?\d+\.\d+)\s*<A>\s*(-?\d+\.\d+)\s*<R>\s*$")
def parse_WA(prefix):
    m = _tail.search(prefix.strip())
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)

def make_mocks(mkt, spread, c_star):
    mu, sigma, r = mkt["mu"], mkt["sigma"], mkt["r"]
    def ideal(prefix):           # knows the true transition kernel
        W, A = parse_WA(prefix)
        if W is None: return None
        return theoretical_return(mu, sigma, r, W, A, spread) + 0.0
    def shortcut(prefix):        # only knows optimal-policy return; ignores counterfactual A
        W, A = parse_WA(prefix)
        if W is None: return None
        return theoretical_return(mu, sigma, r, W, c_star*W, spread)   # uses A=c*W, not the probe A
    def no_kink(prefix):         # linear dynamics but unaware of borrowing cost (no kink)
        W, A = parse_WA(prefix)
        if W is None: return None
        return (A*(mu - r) + r*W)*0.1 / W                              # drops the penalty term
    return dict(ideal=ideal, shortcut=shortcut, no_kink=no_kink)

for regime in ["benign", "sharp"]:
    tasks = [json.loads(l) for l in open(f"data_{regime}_eval_id.jsonl")]
    print(f"\n=== regime: {regime} (spread={tasks[0]['spread']}) ===")
    for name in ["ideal", "shortcut", "no_kink"]:
        # per-task mock closes over that task's market; wrap to dispatch by task
        def predict(prefix, _tasks=tasks):
            # identify task by its support_text presence in the prefix
            for t in _tasks:
                if t["support_text"] in prefix:
                    return make_mocks(t["market"], t["spread"], t["c_star"])[name](prefix)
            return None
        res = analyze_eval_set(predict, tasks, n_pts=13)
        print(f"  [{name:8s}] slope={res['mean_slope']:.5f} (theory {res['mean_slope_theory']:.5f}) "
              f"slope_err={res['mean_slope_err']:.5f}  pred_rmse={res['mean_pred_rmse']:.5f}  "
              f"kink_drop={res['mean_kink_drop']:.5f} (theory {res['mean_kink_drop_theory']:.5f})")
