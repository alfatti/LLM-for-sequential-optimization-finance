"""Validate the harness can MEASURE the 1/K_test law on real generated data:
subsample 1..12 support trajectories from each task, estimate c, report RMSE vs K_test."""
import json, numpy as np
from eval_harness import estimator_ols
SEP=" <SEP> "
tasks=[json.loads(l) for l in open("data_benign_eval_kt12.jsonl")]
def sub(support_text, n):  # take first n support trajectories
    return SEP.join(support_text.split(SEP)[:n])
print("K_test   coeff RMSE")
Ks=[1,2,3,4,6,8,12]; rmses=[]
for k in Ks:
    se=[]
    for t in tasks:
        ch=estimator_ols(sub(t["support_text"],k)); se.append((ch-t["c_star"])**2)
    rmses.append(np.sqrt(np.mean(se)))
    print(f"{k:5d}    {np.sqrt(np.mean(se)):.4f}")
# slope of MSE vs Ktest
sl=np.polyfit(np.log(Ks),np.log(np.array(rmses)**2),1)[0]
print(f"\nlog-log slope of MSE vs K_test = {sl:.2f}  (theory -1)")
