import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from merton import Market, crra_coeff, crra_utility

mkt = Market(mu=0.12, sigma=0.20, r=0.03); rho=2.0
T,x0,n_steps = 1.0,1.0,10; c_true=crra_coeff(mkt,rho)
util=lambda x: crra_utility(x,rho)

kt_d=np.load("pilot_ktest.npz"); g=np.load("pilot_gaussian.npz")

# flatness bowl via CRN (fast)
def paired(c1,c2,dW):
    dt=T/n_steps; x1=x0; x2=x0
    for k in range(n_steps):
        u1=c1*x1; u2=c2*x2
        x1+=(u1*(mkt.mu-mkt.r)+mkt.r*x1)*dt+u1*mkt.sigma*dW[k]
        x2+=(u2*(mkt.mu-mkt.r)+mkt.r*x2)*dt+u2*mkt.sigma*dW[k]
    return x1,x2
def gap_at(frac,n=60000):
    rng=np.random.default_rng(7); dt=T/n_steps
    d=np.empty(n); o=np.empty(n)
    for i in range(n):
        dW=np.sqrt(dt)*rng.standard_normal(n_steps)
        xo,xh=paired(c_true,frac*c_true,dW); d[i]=util(xo)-util(xh); o[i]=util(xo)
    return d.mean()/abs(o.mean())
fracs=np.linspace(0.4,1.6,21)
bowl=np.array([gap_at(f) for f in fracs])

fig,ax=plt.subplots(2,2,figsize=(12,9))

K=kt_d["K_TESTS"]
ax[0,0].loglog(K,np.maximum(kt_d["noiseless__ols"],1e-33),'o-',color="#c0392b",label="noiseless, OLS")
ax[0,0].loglog(K,kt_d["processplabel__ols"],'s-',color="#2980b9",label="label-noise, OLS")
ax[0,0].loglog(K,kt_d["processplabel__shrinkage"],'^-',color="#27ae60",label="label-noise, shrinkage")
ax[0,0].loglog(K,kt_d["processplabel__ols"][0]/K,'k--',alpha=.5,label="$1/K_{test}$")
ax[0,0].set_title("A. DESIGN DECISION — coefficient MSE vs $K_{test}$\n(realistic GBM features)",fontweight='bold')
ax[0,0].set_xlabel("$K_{test}$ (support trajectories)"); ax[0,0].set_ylabel("MSE of $\\hat c$")
ax[0,0].legend(fontsize=8); ax[0,0].grid(True,which="both",alpha=.3)
ax[0,0].text(1.5,1e-30,"noiseless OLS recovers $c$ from\none demo → $K_{test}$ does nothing.\nLabel noise restores the $1/K_{test}$ law.",
             fontsize=7.5,color="#c0392b",va='top')

N=g["N_TRAIN"]
ax[0,1].loglog(N,g["emp_bias"]**2,'o-',color="#8e44ad",label="empirical bias$^2$")
ax[0,1].loglog(N,g["ana_bias"]**2,'k--',alpha=.6,label="analytic $(2c/N)^2$")
ax[0,1].set_title(f"B. Training-length term [eq.6, 2nd]\nslope={float(g['sl_bias']):.2f} (theory $-2$)")
ax[0,1].set_xlabel("$N=K\\cdot T$ (training length)"); ax[0,1].set_ylabel("squared bias of $\\hat c$")
ax[0,1].legend(fontsize=8); ax[0,1].grid(True,which="both",alpha=.3)

Kt=g["K_TESTS"]
ax[1,0].loglog(Kt,g["var_curve"],'o-',color="#16a085",label="Var($\\hat c$), label noise")
ax[1,0].loglog(Kt,g["var_curve"][0]*(Kt[0]/Kt),'k--',alpha=.6,label="$1/K_{test}$")
ax[1,0].set_title(f"C. In-context term [eq.6, 1st]\nslope={float(g['sl_var']):.2f} (theory $-1$)")
ax[1,0].set_xlabel("$K_{test}$ (support trajectories)"); ax[1,0].set_ylabel("variance of $\\hat c$")
ax[1,0].legend(fontsize=8); ax[1,0].grid(True,which="both",alpha=.3)

ax[1,1].plot(fracs,100*bowl,'o-',color="#e67e22")
ax[1,1].axvline(1.0,color='gray',ls=':',alpha=.6)
ax[1,1].set_title("D. KEY FINDING — the objective is FLAT near optimum\n(CRN; gap vs coefficient error)",fontweight='bold')
ax[1,1].set_xlabel("$\\hat c / c^*$  (coefficient error)"); ax[1,1].set_ylabel("optimality gap (%)")
ax[1,1].grid(True,alpha=.3)
ax[1,1].text(0.42,1.0,"±20% coefficient error\n→ only ~0.2% gap.\nClaim-1 contrast will be\nsmall in utility terms;\nlead with MSE instead.",fontsize=7.5,va='top')

plt.tight_layout(); plt.savefig("phase0_pilot.png",dpi=130,bbox_inches="tight")
print("saved phase0_pilot.png")
