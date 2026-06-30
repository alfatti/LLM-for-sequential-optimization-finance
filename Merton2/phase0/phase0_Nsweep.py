"""
Training-length bias: Paper 1's second term, ~1/N^2. It is the SQUARED bias of the
shrinkage predictor, E[c_hat] ~ c/(1+2/N) ~ c(1 - 2/N), so bias^2 ~ (2c/N)^2 ~ 1/N^2.
We isolate it by using large K_test (kills the 1/M variance) and reading the floor.
"""
import numpy as np
from core.dynamics import Market, crra_coeff
from core.estimator import build_support_pairs, estimate_shrinkage, feature_second_moment

mkt = Market(mu=0.12, sigma=0.20, r=0.03); rho = 2.0
T, x0, n_steps = 1.0, 1.0, 10
c_true = crra_coeff(mkt, rho)
rng = np.random.default_rng(11)
S = feature_second_moment(mkt, rho, x0, T, n_steps, rng, n_traj=2000)

N_TRAIN = [50, 100, 200, 500, 1000, 2000, 5000]
K_TEST_BIG = 400          # large support set -> variance negligible, bias dominates
REPS = 800

mse, bias2_pred = [], []
for N in N_TRAIN:
    chs = np.empty(REPS)
    for j in range(REPS):
        xs, ys = build_support_pairs(mkt, rho, x0, T, n_steps, K_TEST_BIG, rng, label_noise=0.0)
        chs[j] = estimate_shrinkage(xs, ys, N, S)
    mse.append(np.mean((chs - c_true)**2))
    bias2_pred.append((2*c_true/N)**2)          # analytic prediction
mse = np.array(mse); bias2_pred = np.array(bias2_pred)

# log-log slope of MSE vs N
sl = np.polyfit(np.log(N_TRAIN), np.log(mse), 1)[0]
print("N_train     MSE(c_hat)     predicted (2c/N)^2")
for N, m, b in zip(N_TRAIN, mse, bias2_pred):
    print(f"{N:6d}   {m:.3e}     {b:.3e}")
print(f"\nlog-log slope of MSE vs N = {sl:.2f}  (expect ~ -2)")

np.savez("pilot_Nsweep.npz", N_TRAIN=np.array(N_TRAIN), mse=mse,
         bias2_pred=bias2_pred, c_true=c_true)
print("saved pilot_Nsweep.npz")
