"""
Matched-Gaussian harness (Paper 1, Appendix E.2 / Fig. 4): features x_i ~ N(0, S)
with KNOWN S, so the only bias is the shrinkage bias. This isolates the 1/N^2
training-length term cleanly. Two checks:
  (1) E[c_hat] = c/(1+2/N)   (bias formula exact)
  (2) bias^2 ~ 1/N^2         (log-log slope ~ -2)
Also re-confirm the 1/M variance term in the same clean setup.
"""
import numpy as np

c_true = 1.125
S_true = 1.0                       # x ~ N(0,1) => E[x^2]=1 exactly
rng = np.random.default_rng(3)

def c_hat(M, N, label_noise, rng):
    x = rng.standard_normal(M) * np.sqrt(S_true)
    y = c_true * x
    if label_noise > 0:
        y = y * (1 + label_noise * rng.standard_normal(M))
    moment = np.mean(y * x)
    Gamma = S_true * (1 + 2.0 / N)
    return moment / Gamma

# ---- (1)+(2): bias vs N at large M (variance negligible in the MEAN) ----
N_TRAIN = [50, 100, 200, 500, 1000, 2000, 5000]
M_BIG, REPS = 20000, 4000
emp_bias, ana_bias = [], []
for N in N_TRAIN:
    chs = np.array([c_hat(M_BIG, N, 0.0, rng) for _ in range(REPS)])
    emp_bias.append(chs.mean() - c_true)
    ana_bias.append(-2*c_true/(N+2))
emp_bias = np.array(emp_bias); ana_bias = np.array(ana_bias)
sl_bias = np.polyfit(np.log(N_TRAIN), np.log(emp_bias**2), 1)[0]
print("=== Training-length bias (matched-Gaussian, known S) ===")
print("N        emp_bias      analytic -2c/(N+2)")
for N, eb, ab in zip(N_TRAIN, emp_bias, ana_bias):
    print(f"{N:6d}   {eb:+.5e}   {ab:+.5e}")
print(f"log-log slope bias^2 vs N = {sl_bias:.2f}   (expect -2)\n")

# ---- variance term vs M (large N so bias ~0), with label noise ----
K_TESTS = [1, 2, 5, 10, 20, 50, 100]
n_steps = 10
N_BIG, REPS2 = 100000, 4000
var_curve = []
for kt in K_TESTS:
    M = kt * n_steps
    chs = np.array([c_hat(M, N_BIG, 0.25, rng) for _ in range(REPS2)])
    var_curve.append(chs.var())
var_curve = np.array(var_curve)
sl_var = np.polyfit(np.log(K_TESTS), np.log(var_curve), 1)[0]
print("=== In-context variance term (label noise=0.25, large N) ===")
print("K_test   Var(c_hat)")
for kt, v in zip(K_TESTS, var_curve):
    print(f"{kt:6d}   {v:.3e}")
print(f"log-log slope Var vs K_test = {sl_var:.2f}   (expect -1)")

np.savez("pilot_gaussian.npz", N_TRAIN=np.array(N_TRAIN), emp_bias=emp_bias,
         ana_bias=ana_bias, K_TESTS=np.array(K_TESTS), var_curve=var_curve,
         c_true=c_true, sl_bias=sl_bias, sl_var=sl_var)
print("saved pilot_gaussian.npz")
