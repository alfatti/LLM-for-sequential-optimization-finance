"""
Isolate the training-length term properly: it is the BIAS of the shrinkage
predictor, not total MSE. Analytic prediction:
    E[c_hat] = c / (1 + 2/N)   =>   bias = E[c_hat] - c = -2c/(N+2) ~ -2c/N
    bias^2 ~ 1/N^2   <-- Paper 1's training-length term.
We estimate E[c_hat] with many reps (variance averages out of the MEAN).
"""
import numpy as np
from core.dynamics import Market, crra_coeff
from core.estimator import build_support_pairs, estimate_shrinkage, feature_second_moment

mkt = Market(mu=0.12, sigma=0.20, r=0.03); rho = 2.0
T, x0, n_steps = 1.0, 1.0, 10
c_true = crra_coeff(mkt, rho)
rng = np.random.default_rng(11)
S = feature_second_moment(mkt, rho, x0, T, n_steps, rng, n_traj=4000)

N_TRAIN = [50, 100, 200, 500, 1000, 2000, 5000]
K_TEST = 100
REPS = 6000

emp_bias, ana_bias = [], []
for N in N_TRAIN:
    chs = np.empty(REPS)
    for j in range(REPS):
        xs, ys = build_support_pairs(mkt, rho, x0, T, n_steps, K_TEST, rng, label_noise=0.0)
        chs[j] = estimate_shrinkage(xs, ys, N, S)
    b = chs.mean() - c_true
    emp_bias.append(b)
    ana_bias.append(-2*c_true/(N+2))
emp_bias = np.array(emp_bias); ana_bias = np.array(ana_bias)

sl = np.polyfit(np.log(N_TRAIN), np.log(emp_bias**2), 1)[0]
print("N_train   emp_bias     analytic -2c/(N+2)   emp_bias^2")
for N, eb, ab in zip(N_TRAIN, emp_bias, ana_bias):
    print(f"{N:6d}   {eb:+.4e}   {ab:+.4e}        {eb**2:.3e}")
print(f"\nlog-log slope of bias^2 vs N = {sl:.2f}  (expect ~ -2)")

np.savez("pilot_bias.npz", N_TRAIN=np.array(N_TRAIN),
         emp_bias=emp_bias, ana_bias=ana_bias, c_true=c_true)
print("saved pilot_bias.npz")
