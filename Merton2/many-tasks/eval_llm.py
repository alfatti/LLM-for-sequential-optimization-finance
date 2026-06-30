"""Many-tasks ICL/SFT/random eval (GPU). Aggregates over the task distribution."""
import argparse, json, numpy as np
from core.metrics import crn_gap_for_task
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--eval",required=True)
    ap.add_argument("--adapter",default=None); ap.add_argument("--random",action="store_true")
    ap.add_argument("--n_gap_paths",type=int,default=8000); a=ap.parse_args()
    tasks=[json.loads(l) for l in open(a.eval)]
    if a.random:
        rng=np.random.default_rng(0); est=lambda s: float(rng.uniform(0.1,2.0)); tag="random"
    else:
        from core.llm import load, make_coeff_probe
        tok,model=load(a.adapter); est=make_coeff_probe(tok,model); tag="SFT" if a.adapter else "ICL-only"
    sq,gaps=[],[]
    for t in tasks:
        ch=est(t["support_text"]); 
        if np.isnan(ch): continue
        sq.append((ch-t["c_star"])**2); gaps.append(abs(crn_gap_for_task(t,ch,a.n_gap_paths,seed=t["task_id"])))
    gaps=np.array(gaps); wins=np.clip(gaps,None,np.percentile(gaps,95))
    print(f"[{tag}] {a.eval}: n={len(sq)} coeff RMSE={np.sqrt(np.mean(sq)):.4f} "
          f"gap med={100*np.median(gaps):.3f}% wins={100*np.mean(wins):.3f}%")
if __name__=="__main__": main()
