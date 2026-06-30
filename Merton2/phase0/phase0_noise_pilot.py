"""
Phase 0 noise pilot. Question: under what noise condition does the in-context
estimation error of the Merton coefficient actually FALL with K_test (so claim 2's
1/K_test term is observable), rather than saturating immediately?

We compute MSE(c_hat) vs K_test and vs N (training length) under three conditions:
  (a) noiseless,  (b) process noise only,  (c) process + label noise.
For each we run OLS and the eq.(5) shrinkage predictor.
"""
import numpy as np
from core.dynamics import Market, crra_coeff
from core.estimator import (build_support_pairs, estimate_ols, estimate_shrinkage,
                       feature_second_moment)

mkt = Market(mu=0.12, sigma=0.20, r=0.03)
rho = 2.0
T, x0, n_steps = 1.0, 1.0, 10        # horizon discretized into 10 decision steps
c_true = crra_coeff(mkt, rho)
N_demo_pairs = lambda k_test: k_test * n_steps   # M = K_test * T

rng = np.random.default_rng(7)
S = feature_second_moment(mkt, rho, x0, T, n_steps, rng, n_traj=2000)
print(f"c_true={c_true:.4f}  S=E[x^2]={S:.4f}")

K_TESTS = [1, 2, 5, 10, 20, 50]
N_FIXED = 1000          # training length when sweeping K_test
N_TASKS = 4000          # MC replications (here: re-draws of the support set)

def mse_curve(label_noise, estimator, N_train):
    out = []
    for kt in K_TESTS:
        errs = np.empty(N_TASKS)
        for j in range(N_TASKS):
            xs, ys = build_support_pairs(mkt, rho, x0, T, n_steps, kt, rng,
                                         label_noise=label_noise)
            if estimator == "ols":
                ch = estimate_ols(xs, ys)
            else:
                ch = estimate_shrinkage(xs, ys, N_train, S)
            errs[j] = (ch - c_true)**2
        out.append(errs.mean())
    return np.array(out)

conditions = {
    "noiseless"      : dict(label_noise=0.00),
    "process_only"   : dict(label_noise=0.00),   # same as noiseless here (process noise always on via GBM)
    "process+label"  : dict(label_noise=0.25),
}

results = {}
for cond, kw in conditions.items():
    for est in ["ols", "shrinkage"]:
        key = f"{cond}|{est}"
        results[key] = mse_curve(kw["label_noise"], est, N_FIXED)
        print(f"{key:28s}  MSE@Ktest={K_TESTS}: " +
              " ".join(f"{v:.2e}" for v in results[key]))

np.savez("pilot_ktest.npz", K_TESTS=np.array(K_TESTS), c_true=c_true, S=S,
         **{k.replace("|","__").replace("+","p"): v for k, v in results.items()})
print("saved pilot_ktest.npz")
