#!/usr/bin/env python3
# TRUE STALENESS: slow methods solve on snapshot-t topology, then X% of ISLs break (t+1).
# Their STALE paths crossing broken edges FAIL -> flows lost. RL re-routes on broken topology.
# Same broken edge set for all methods per level. Metrics: SR/PLR/delay/hops.
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, torch
import hypatia_pop_instance as HP, neural_solver_MCpath as MP, unroll_solver as U
import gdh_dpq_v6e as DUAL, hypatia_baselines as BL
dev='cpu'; K=4
SCALE=1.0; CG=8.0; CISL=10.0; HD=30.0
BREAK_PCT=[0.0,0.02,0.05,0.10,0.15,0.20]
OUT='/home/msarwa02/staleness_sweep_results.txt'

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
n_edges=inst.n_edges; E=inst.eidx_mat; offered=float(inst.rho.sum())

def path_broken(path, broken):
    for a,b in zip(path[:-1],path[1:]):
        if int(E[a,b]) in broken: return True
    return False

def eval_stale(paths_dict, served_dict, broken):
    """Flow s contributes its ACTUAL admitted amount (served_dict[s]) iff its path survives."""
    surv=[]
    for s,path in paths_dict.items():
        if path is None or len(path)<1: continue
        if path_broken(path, broken): continue    # stale path failed -> flow lost
        amt=float(served_dict.get(s,0.0)) if served_dict else float(inst.rho[s])
        surv.append((s,path,amt))
    served=sum(a for _,_,a in surv)
    sr=served/offered if offered>0 else 0.0
    hs=[len(p)-1 for _,p,_ in surv]
    return sr, 1-sr, (np.mean(hs) if hs else 0.0)

# ---- solve each SLOW method ONCE on full topology, capture paths ----
print("solving slow methods on full topology (capturing paths)...", flush=True)
slow={}
for nm,fl in [("dual",(1,1)),("nu-only",(0,1)),("lam-only",(1,0)),("no-dual",(0,0))]:
    r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=fl[0],use_nu=fl[1],return_paths=True)
    slow[nm]=(r['paths'] or {}, r.get('served_flow') or {})
    print("  %s: %d paths, SR(full)=%.4f"%(nm,len(slow[nm][0]),r['served_ratio']),flush=True)
for nm,fn in BL.ALL_BASELINES.items():
    r=fn(inst); slow[nm]=(r.get('paths') or {}, r.get('served_flow') or {})
    print("  %s: %d paths, SR(full)=%.4f"%(nm,len(slow[nm][0]),r['served_ratio']),flush=True)
# greedy
g=MP.greedy_nearest_baseline(inst)
# greedy may not return paths; skip if absent
print("  (greedy has no path export - skipped in stale eval)",flush=True)

hdr="%-7s %-10s %8s %8s %7s"%("break%","method","SR","PLR","hops")
print("\n"+hdr,flush=True)
open(OUT,'w').write("# Staleness: slow=stale paths on broken topo; RL=re-routed. HD=%.0f Cg=%.0f Cisl=%.0f\n%s\n"%(HD,CG,CISL,hdr))
rng=np.random.default_rng(123)
for bp in BREAK_PCT:
    nb=int(bp*n_edges)
    broken=set(int(x) for x in rng.choice(n_edges,size=nb,replace=False)) if nb>0 else set()
    # slow methods: STALE paths on broken topology
    for nm,(pd,sf) in slow.items():
        sr,plr,hp=eval_stale(pd, sf, broken)
        line="%-7.0f %-10s %8.4f %8.4f %7.2f"%(bp*100,nm,sr,plr,hp)
        print(line,flush=True); open(OUT,'a').write(line+"\n")
    # RL: RE-ROUTE on broken topology
    srs,fs,_=U.unrolled(pol,inst,K=K,greedy=True,broken_links=broken)
    rl_paths={}; rl_served={}
    for s,gd in fs.items():
        for gid,(a,p,e) in gd.items():
            if a>0: rl_paths[s]=p; rl_served[s]=rl_served.get(s,0.)+a
    sr,plr,hp=eval_stale(rl_paths, rl_served, set())  # RL routed around breaks
    line="%-7.0f %-10s %8.4f %8.4f %7.2f"%(bp*100,"RL(fresh)",sr,plr,hp)
    print(line,flush=True); open(OUT,'a').write(line+"\n")
    open(OUT,'a').write("\n"); print("",flush=True)
print("[done]",flush=True); open(OUT,'a').write("[done]\n")
