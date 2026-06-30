import numpy as np
from core.dynamics import Market, crra_coeff, crra_utility
mkt=Market(mu=0.12,sigma=0.20,r=0.03); rho=2.0
T,x0,n_steps=1.0,1.0,10; c_true=crra_coeff(mkt,rho); util=lambda x: crra_utility(x,rho)

# ---- (1) action-grid quantization floor (CRN), isolating rounding only ----
def gap_quantized(decimals,n=80000):
    rng=np.random.default_rng(5); dt=T/n_steps; d=np.empty(n); o=np.empty(n)
    for i in range(n):
        dW=np.sqrt(dt)*rng.standard_normal(n_steps)
        xo=x0; xh=x0
        for k in range(n_steps):
            uo=c_true*xo
            uh=np.round(c_true*xh,decimals) if decimals is not None else c_true*xh
            xo+=(uo*(mkt.mu-mkt.r)+mkt.r*xo)*dt+uo*mkt.sigma*dW[k]
            xh+=(uh*(mkt.mu-mkt.r)+mkt.r*xh)*dt+uh*mkt.sigma*dW[k]
        d[i]=util(xo)-util(xh); o[i]=util(xo)
    return d.mean()/abs(o.mean())
print("=== Action-grid quantization floor (optimal c, rounded actions) ===")
for dec in [0,1,2,3]:
    print(f"  round to {dec} decimals (res={10.0**-dec:.3f})  ->  gap floor = {100*gap_quantized(dec):.4f}%")
print(f"  no rounding (continuous)              ->  gap floor = {100*gap_quantized(None):.4f}%")

# ---- (2) context-length budget for Llama-3-8B (8192-token window) ----
# per-step serialization e.g.: "<S_3> 1.07 <W_3> 1.21 <A_3> 1.36 <R_3> 0.041 "
# rough token estimate: 4 tags (~1 tok each) + 4 numbers (~3 toks each, e.g. '1.36'->['1','.','36']) + spaces
# add history window h of (price,wealth) pairs: +2 numbers per past step shown
print("\n=== Context budget (Llama-3-8B, 8192 ctx) ===")
CTX=8192
def toks_per_step(window_h):
    base = 4*1 + 4*3              # tags + 4 numbers
    hist = window_h*2*3          # h past (price,wealth) pairs, ~3 toks each
    return base + hist + 4       # spaces/sep
for h in [0,3,5]:
    tps=toks_per_step(h); per_traj=tps*n_steps
    print(f"  window h={h}: ~{tps} tok/step, ~{per_traj} tok/trajectory(T=10)")
    for kt in [2,5,10,20,50]:
        total=kt*per_traj + per_traj   # support + query
        flag="" if total<CTX else "  << EXCEEDS 8192"
        print(f"      K_test={kt:3d}: ~{total:5d} tok{flag}")
