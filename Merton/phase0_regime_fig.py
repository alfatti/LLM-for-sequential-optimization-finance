import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from merton import Market, crra_utility
mkt=Market(mu=0.12,sigma=0.20,r=0.03); rho=2.0
T,x0,n_steps=1.0,1.0,10; dt=T/n_steps; util=lambda x: crra_utility(x,rho)

def bowl(spread,cs,fracs,n=20000):
    out=[]
    for f in fracs:
        rng=np.random.default_rng(21); d=np.empty(n); o=np.empty(n)
        for i in range(n):
            dW=np.sqrt(dt)*rng.standard_normal(n_steps); xo=x0; xh=x0
            for k in range(n_steps):
                uo=cs*xo; uh=f*cs*xh; po=spread*max(0.0,uo-xo); ph=spread*max(0.0,uh-xh)
                xo+=(uo*(mkt.mu-mkt.r)+mkt.r*xo-po)*dt+uo*mkt.sigma*dW[k]
                xh+=(uh*(mkt.mu-mkt.r)+mkt.r*xh-ph)*dt+uh*mkt.sigma*dW[k]
            d[i]=util(max(xo,1e-9))-util(max(xh,1e-9)); o[i]=util(max(xo,1e-9))
        out.append(d.mean()/abs(o.mean()))
    return np.array(out)

# reuse known c* (benign ~1.125 analytic; sharp from prior run)
cs0, cs_sharp = 1.125, 0.913
fracs=np.linspace(0.5,1.5,17)
b0=bowl(0.0,cs0,fracs); bS=bowl(0.4,cs_sharp,fracs)

# contrast vs spread (light: known approx c* per spread, n small)
spreads=[0.0,0.05,0.1,0.2,0.4,0.7,1.0]
cs_by_s={0.0:1.125,0.05:0.945,0.1:0.93,0.2:0.92,0.4:0.913,0.7:0.911,1.0:0.910}
contr=[float(100*np.max(np.abs(bowl(s,cs_by_s[s],np.array([0.7,1.3]),n=20000)))) for s in spreads]

fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].plot(spreads,contr,'o-',color="#c0392b")
ax[0].axhline(0.5,color='gray',ls=':',label='benign flat floor ~0.5%')
ax[0].set_title("Curvature knob: borrowing spread $s$ sets the contrast")
ax[0].set_xlabel("borrowing spread $s$"); ax[0].set_ylabel("gap at ±30% error (%)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

ax[1].plot(100*(fracs-1),100*b0,'s-',color="#7f8c8d",label=f"benign $s=0$")
ax[1].plot(100*(fracs-1),100*bS,'o-',color="#e67e22",label=f"sharp $s=0.4$")
ax[1].axvline(0,color='gray',ls=':',alpha=.6)
ax[1].set_title("Benign vs sharp objective (one-sided sharpness)")
ax[1].set_xlabel("coefficient error (%)"); ax[1].set_ylabel("optimality gap (%)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("phase0_regimes.png",dpi=130,bbox_inches="tight")
print("c* benign=%.3f  c* sharp=%.3f"%(cs0,cs_sharp))
print("contrast vs spread:", [round(c,2) for c in contr])
print("saved phase0_regimes.png")
