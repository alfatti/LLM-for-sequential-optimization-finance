import numpy as np
from merton import Market, crra_coeff, crra_utility, expected_utility

rng = np.random.default_rng(0)
mkt = Market(mu=0.12, sigma=0.20, r=0.03)
rho = 2.0          # relative risk aversion
T, x0, n_steps, n_mc = 1.0, 1.0, 50, 60000

c_star = crra_coeff(mkt, rho)
print(f"Analytic optimal coefficient c* = {c_star:.4f}")

# Sweep candidate constant-fraction policies u = c' * x and find the EU-maximizer.
cs = np.linspace(0.3*c_star, 1.9*c_star, 17)
EUs, ses = [], []
for c in cs:
    pol = lambda t, x, c=c: c * x
    eu, se = expected_utility(mkt, pol, x0, T, n_steps,
                              lambda x: crra_utility(x, rho), n_mc, rng)
    EUs.append(eu); ses.append(se)
EUs = np.array(EUs); ses = np.array(ses)

c_emp = cs[np.argmax(EUs)]
print(f"Empirical EU-maximizing c  = {c_emp:.4f}  (grid spacing {cs[1]-cs[0]:.4f})")
print(f"Ratio c_emp / c*           = {c_emp/c_star:.3f}")
# fit a quadratic near the top to locate the peak more precisely
top = np.argsort(EUs)[-5:]
coef = np.polyfit(cs[top], EUs[top], 2)
c_peak = -coef[1]/(2*coef[0])
print(f"Quadratic-fit peak c       = {c_peak:.4f}   ratio {c_peak/c_star:.3f}")
