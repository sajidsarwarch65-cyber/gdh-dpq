#!/usr/bin/env python3
# OPTION B: ALL methods RE-SOLVE on the failed topology (failed edges' lifetime=0).
# Isolates inherent robustness from staleness/path-length. Same failed sets as stale sweep.
# HD=30, Cg=8, Cisl=10. Metrics: SR/PLR/hops (+time).
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, torch
import hypatia_pop_instance as HP, neural_solver_MCpath as MP, unroll_solver as U
import gdh_dpq_v6e as DUAL, hypatia_baselines as BL
dev='cpu'; K=4
SCALE=1.0; CG=8.0; CISL=10.0; HD=30.0
FAIL_PCT=[0.0,0.02,0.05,0.10,0.15,0.20]
OUT='/home/msarwa02/resolve_sweep_results.txt'

# RL policy
try: cfg=MP.default_cfg() if hasattr(MP,'default_cfg') else MP.CFG
except: import train_unroll_vr as TR; cfg=TR.cfg
pol=MP.ConstructionPolicy(cfg).to(dev)
sd=torch.load('policy_unroll.pt',map_location=dev)
import torch.nn as _nn
ck=sd['enc.node_in.weight'].shape[1]; cur=pol.enc.node_in
if cur.in_features!=ck:
    nl=_nn.Linear(ck,cur.out_features,bias=(cur.bias is not None)).to(dev)
    with torch.no_grad():
        m=min(cur.in_features,ck); nl.weight[:,:m].copy_(cur.weight[:,:m])
        if cur.bias is not None: nl.bias.copy_(cur.bias)
    pol.enc.node_in=nl
pol.load_state_dict(sd,strict=False); pol=U.widen_encoder(pol,dev); pol.eval()

HP.rng_seed=7; HP.HOTSPOT_DEMAND=HD
inst,DST,GWS=HP.build_instance(scale=SCALE,hotspot=True,use_std_gw=True)
inst.Cg=np.full(inst.Ng,CG); inst.Cisl=CISL
orig_life=inst.isl_lifetime_s.copy()
E=inst.eidx_mat

def rl_solve():
    srs,fs,_=U.unrolled(pol,inst,K=K,greedy=True)
    sr=float(srs[-1]) if hasattr(srs,'__len__') else float(srs)
    hs=[]
    for s,gd in fs.items():
        for gid,(a,p,e) in gd.items():
            if a>0: hs.append(len(p)-1)
    return sr,1-sr,(np.mean(hs) if hs else 0.0)

def dual_solve(ul,un):
    r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=ul,use_nu=un)
    return r['served_ratio'],r['plr'],r['mean_hops']

hdr="%-7s %-10s %8s %8s %7s %8s"%("fail%","method","SR","PLR","hops","time_s")
print(hdr,flush=True)
open(OUT,'w').write("# OPTION B: all methods RE-SOLVE on failed topology (lifetime=0). HD=%.0f Cg=%.0f Cisl=%.0f\n%s\n"%(HD,CG,CISL,hdr))
rng=np.random.default_rng(123)   # SAME seed as stale sweep -> same failed sets
for fp in FAIL_PCT:
    nf=int(fp*inst.n_edges)
    failed=rng.choice(inst.n_edges,size=nf,replace=False) if nf>0 else []
    # apply failures: lifetime=0 for failed edges (both directions)
    life2=orig_life.copy()
    for e in failed:
        i,j=inst.isl_edges[e]; life2[i,j]=0.0; life2[j,i]=0.0
    inst.isl_lifetime_s=life2
    # all methods RE-SOLVE on this failed topology
    for nm,fn in [("dual",lambda:dual_solve(1,1)),("nu-only",lambda:dual_solve(0,1)),
                  ("lam-only",lambda:dual_solve(1,0)),("no-dual",lambda:dual_solve(0,0)),
                  ("RL",rl_solve)]:
        try:
            t0=time.time(); sr,plr,hp=fn(); t=time.time()-t0
            line="%-7.0f %-10s %8.4f %8.4f %7.2f %8.1f"%(fp*100,nm,sr,plr,hp,t)
        except Exception as e:
            line="%-7.0f %-10s ERR %s"%(fp*100,nm,str(e)[:40])
        print(line,flush=True); open(OUT,'a').write(line+"\n")
    for bn,bf in BL.ALL_BASELINES.items():
        try:
            t0=time.time(); r=bf(inst); t=time.time()-t0
            line="%-7.0f %-10s %8.4f %8.4f %7.2f %8.1f"%(fp*100,bn,r['served_ratio'],r['plr'],r['avg_hops'],t)
        except Exception as e:
            line="%-7.0f %-10s ERR %s"%(fp*100,bn,str(e)[:40])
        print(line,flush=True); open(OUT,'a').write(line+"\n")
    inst.isl_lifetime_s=orig_life  # restore
    open(OUT,'a').write("\n"); print("",flush=True)
print("[done]",flush=True); open(OUT,'a').write("[done]\n")
