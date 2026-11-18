#!/usr/bin/env python3
# HOTSPOT RADIUS sweep, FIXED TOTAL hotspot demand (Option 2 - concentration test).
# Total hotspot = 300 Gbps spread over N sats (per-sat = 300/N). Cg=8, Cisl=10.
# All 11 methods, full metrics. Back-fills RL/dual/no-dual delay+hops too.
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, torch
import hypatia_pop_instance as HP, neural_solver_MCpath as MP, unroll_solver as U
import gdh_dpq_v6e as DUAL, hypatia_baselines as BL
dev='cpu'; K=4
TOTAL_HOTSPOT=300.0; CG=8.0; CISL=10.0; SCALE=1.0
RADII=[200.0,300.0,400.0,500.0,700.0,1000.0]
OUT='/home/msarwa02/radius_sweep_results.txt'
HP.HOTSPOT_MAX_SATS=999   # lift cap so radius controls count

# load RL policy
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

def rl_m(inst):
    t0=time.time(); srs,fs,_=U.unrolled(pol,inst,K=K,greedy=True); dt=time.time()-t0
    sr=float(srs[-1]) if hasattr(srs,'__len__') else float(srs)
    gwl={}; isll={}
    for s,gd in fs.items():
        for gid,(a,p,e) in gd.items():
            if a<=0: continue
            gwl[gid]=gwl.get(gid,0.0)+a
            for ek in e: isll[ek]=isll.get(ek,0.0)+a
    ds=[]; hs=[]
    for s,gd in fs.items():
        for gid,(a,p,e) in gd.items():
            if a>0:
                hs.append(len(p)-1)
                try: ds.append(inst.full_delay(p,gid,gwl,isll))
                except: pass
    return dict(served_ratio=sr,plr=1-sr,avgD_ms=(np.mean(ds)*1000 if ds else 0),avg_hops=(np.mean(hs) if hs else 0)),dt
def dual_m(inst,ul,un):
    t0=time.time(); r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=ul,use_nu=un); dt=time.time()-t0
    return dict(served_ratio=r['served_ratio'],plr=r['plr'],avgD_ms=r['avgD_ms'],avg_hops=r['mean_hops']),dt
def gr_m(inst):
    t0=time.time(); g=MP.greedy_nearest_baseline(inst); dt=time.time()-t0
    return dict(served_ratio=g['served_ratio'],plr=g['plr'],avgD_ms=g['avgD_ms'],avg_hops=g['avg_hops']),dt

hdr="%-7s %-6s %-9s %-12s %8s %8s %9s %7s %8s"%("radius","nsat","persat","method","SR","PLR","delay_ms","hops","time_s")
print(hdr,flush=True)
with open(OUT,'w') as f: f.write("# Radius sweep, FIXED total hotspot=%.0f Gbps, Cg=%.0f Cisl=%.0f\n%s\n"%(TOTAL_HOTSPOT,CG,CISL,hdr))
for r in RADII:
    HP.HOTSPOT_RADIUS_KM=r; HP.rng_seed=7
    # first pass: count sats at this radius (probe demand=1)
    HP.HOTSPOT_DEMAND=1.0
    tmp,_,_=HP.build_instance(scale=SCALE,hotspot=True,use_std_gw=True)
    n=int((tmp.rho>=0.99).sum()); persat=TOTAL_HOTSPOT/max(n,1)
    # real build with per-sat = total/n
    HP.HOTSPOT_DEMAND=persat
    inst,DST,GWS=HP.build_instance(scale=SCALE,hotspot=True,use_std_gw=True)
    inst.Cg=np.full(inst.Ng,CG); inst.Cisl=CISL
    methods=[("dual",lambda i:dual_m(i,1,1)),("nu-only",lambda i:dual_m(i,0,1)),
             ("lam-only",lambda i:dual_m(i,1,0)),("no-dual",lambda i:dual_m(i,0,0)),
             ("RL",rl_m),("greedy",gr_m)]
    for mname,fn in methods:
        try:
            res,t=fn(inst)
            line="%-7.0f %-6d %-9.2f %-12s %8.4f %8.4f %9.1f %7.2f %8.1f"%(r,n,persat,mname,res['served_ratio'],res['plr'],res['avgD_ms'],res['avg_hops'],t)
        except Exception as e:
            line="%-7.0f %-6d %-9.2f %-12s ERR %s"%(r,n,persat,mname,str(e)[:40])
        print(line,flush=True)
        with open(OUT,'a') as f: f.write(line+"\n")
    # baselines
    for bname,bfn in BL.ALL_BASELINES.items():
        try:
            t0=time.time(); br=bfn(inst); t=time.time()-t0
            line="%-7.0f %-6d %-9.2f %-12s %8.4f %8.4f %9.1f %7.2f %8.1f"%(r,n,persat,bname,br['served_ratio'],br['plr'],br['avgD_ms'],br['avg_hops'],t)
        except Exception as e:
            line="%-7.0f %-6d %-9.2f %-12s ERR %s"%(r,n,persat,bname,str(e)[:40])
        print(line,flush=True)
        with open(OUT,'a') as f: f.write(line+"\n")
    with open(OUT,'a') as f: f.write("\n")
    print("",flush=True)
print("[done] saved to %s"%OUT,flush=True)
with open(OUT,'a') as f: f.write("[done]\n")
