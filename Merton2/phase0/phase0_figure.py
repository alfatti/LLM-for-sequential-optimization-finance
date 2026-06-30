import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.dynamics import Market, crra_coeff, crra_utility, expected_utility
from core.estimator import build_support_pairs, estimate_ols, estimate_shrinkage, feature_second_moment

kt_d   = np.load("pilot_ktest.npz")
g      = np.load("pilot_gaussian.npz")

mkt = Market(mu=0.12, sigma=0.20, r=0.03); rho = 2.0
T, x0, n_steps = 1.0, 1.0, 10
c_true = crra_coeff(mkt, rho)
rng = np.random.default_rng(21)

# ---- Panel D data: MC optimality gap (%) vs K_test, with/without label noise ----
S = feature_second_moment(mkt, rho, x0, T, n_steps, rng, n_traj=2000)
util = lambda x: crra_utility(x, rho)
# optimal expected utility (reference)
opt_pol = lambda t, x: c_true * x
EU_opt, _ = expected_utility(mkt, opt_pol, x0, T, n_steps, util, 40000, rng)

def gap_curve(label_noise, n_tasks=120, n_mc=1500):
    K = [1,2,5,10,20,50]; gaps=[]
    for kt in K:
        gtask=[]
        for _ in range(n_tasks):
            xs, ys = build_support_pairs(mkt, rho, x0, T, n_steps, kt, rng, label_noise=label_noise)
            ch = estimate_ols(xs, ys)
            pol = lambda t, x, ch=ch: ch * x
            EU, _ = expected_utility(mkt, pol, x0, T, n_steps, util, n_mc, rng)
            # gap in expected-utility terms, normalized; CRRA U<0 here (alpha=-1), so
            # use relative gap on |EU|
            gtask.append(abs((EU_opt - EU)/EU_opt))
        gaps.append(np.mean(gtask))
    return np.array(K), np.array(gaps)

Kg, gap_noise = gap_curve(0.25)
_,  gap_clean = gap_curve(0.0)

# ----------------------------- figure -----------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

# A: design-decision plot (GBM features)
K = kt_d["K_TESTS"]
ax[0,0].loglog(K, kt_d["noiseless__ols"], 'o-', color="#c0392b", label="noiseless, OLS")
ax[0,0].loglog(K, kt_d["processplabel__ols"], 's-', color="#2980b9", label="label-noise, OLS")
ax[0,0].loglog(K, kt_d["processplabel__shrinkage"], '^-', color="#27ae60", label="label-noise, shrinkage")
ax[0,0].loglog(K, kt_d["processplabel__ols"][0]/K, 'k--', alpha=.5, label="1/K_test ref")
ax[0,0].set_title("A. Design decision: estimation error vs K_test\n(GBM features, realistic Merton)")
ax[0,0].set_xlabel("K_test (support trajectories)"); ax[0,0].set_ylabel("MSE of coefficient estimate")
ax[0,0].legend(fontsize=8); ax[0,0].grid(True, which="both", alpha=.3)
ax[0,0].annotate("noiseless OLS saturates\n(~1e-32): K_test does nothing",
                 xy=(5, 3e-32), xytext=(2, 1e-25), fontsize=8, color="#c0392b")

# B: training-length bias (matched Gaussian)
N = g["N_TRAIN"]
ax[0,1].loglog(N, g["emp_bias"]**2, 'o-', color="#8e44ad", label="empirical bias$^2$")
ax[0,1].loglog(N, g["ana_bias"]**2, 'k--', alpha=.6, label="analytic $(2c/N)^2$")
ax[0,1].loglog(N, g["emp_bias"][0]**2*(N[0]/N)**2, ':', color="gray", label="$1/N^2$ ref")
ax[0,1].set_title(f"B. Training-length term (Paper1 eq.6, 2nd term)\nslope={float(g['sl_bias']):.2f} (expect -2)")
ax[0,1].set_xlabel("N = K·T (training length)"); ax[0,1].set_ylabel("squared bias of estimate")
ax[0,1].legend(fontsize=8); ax[0,1].grid(True, which="both", alpha=.3)

# C: in-context variance term (matched Gaussian + label noise)
Kt = g["K_TESTS"]
ax[1,0].loglog(Kt, g["var_curve"], 'o-', color="#16a085", label="Var(estimate), label noise")
ax[1,0].loglog(Kt, g["var_curve"][0]*(Kt[0]/Kt), 'k--', alpha=.6, label="$1/K_{test}$ ref")
ax[1,0].set_title(f"C. In-context term (Paper1 eq.6, 1st term)\nslope={float(g['sl_var']):.2f} (expect -1)")
ax[1,0].set_xlabel("K_test (support trajectories)"); ax[1,0].set_ylabel("variance of estimate")
ax[1,0].legend(fontsize=8); ax[1,0].grid(True, which="both", alpha=.3)

# D: ties to Paper 1's actual metric
ax[1,1].plot(Kg, 100*gap_clean, 's-', color="#7f8c8d", label="noiseless demos")
ax[1,1].plot(Kg, 100*gap_noise, 'o-', color="#2980b9", label="noisy demos (label=0.25)")
ax[1,1].set_title("D. Translates to the optimality gap\n(MC, the metric Paper 1 reports)")
ax[1,1].set_xlabel("K_test (support trajectories)"); ax[1,1].set_ylabel("optimality gap (%)")
ax[1,1].legend(fontsize=8); ax[1,1].grid(True, alpha=.3)

plt.tight_layout()
plt.savefig("phase0_pilot.png", dpi=130, bbox_inches="tight")
print("saved phase0_pilot.png")
print(f"\nOptimality gap vs K_test (noisy demos): {dict(zip(Kg.tolist(), np.round(100*gap_noise,2).tolist()))}")
print(f"Optimality gap vs K_test (clean demos): {dict(zip(Kg.tolist(), np.round(100*gap_clean,2).tolist()))}")
