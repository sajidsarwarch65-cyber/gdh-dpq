#!/usr/bin/env python3
# LOAD GENERALIZATION on Hypatia. RL trained ~0.6-1.5, tested held-out to 3.0.
import time, numpy as np, torch
import hypatia_pop_instance as HP
import neural_solver_MCpath as MP
import unroll_solver as U
import gdh_dpq_v6e as DUAL
dev='cpu'; K=4; USE_STD_GW=True

try:
    cfg = MP.default_cfg() if hasattr(MP,'default_cfg') else MP.CFG
except Exception:
    import train_unroll_vr as TR; cfg = TR.cfg
pol = MP.ConstructionPolicy(cfg).to(dev)
sd = torch.load('policy_unroll.pt', map_location=dev)
import torch.nn as _nn
ck_in = sd['enc.node_in.weight'].shape[1]
cur = pol.enc.node_in
if cur.in_features != ck_in:
    nl=_nn.Linear(ck_in,cur.out_features,bias=(cur.bias is not None)).to(dev)
    with torch.no_grad():
        m=min(cur.in_features,ck_in); nl.weight[:,:m].copy_(cur.weight[:,:m])
        if cur.bias is not None: nl.bias.copy_(cur.bias)
    pol.enc.node_in=nl
pol.load_state_dict(sd, strict=False)
pol = U.widen_encoder(pol, dev)   # 8->10 to match build_gt8 features
pol.eval()
print("[policy] loaded + widened", flush=True)

def rl_metrics(inst):
    t0=time.time()
    srs, fs, _ = U.unrolled(pol, inst, K=K, greedy=True)
    sr = float(srs[-1]) if hasattr(srs,'__len__') else float(srs)
    # reconstruct gwl/isll loads from full_sol so full_delay works
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
    dl=float(np.mean(ds))*1000 if ds else 0.0
    hp=float(np.mean(hs)) if hs else 0.0
    return sr,1-sr,dl,hp,time.time()-t0

print("\n%-6s %-8s %8s %8s %9s %7s"%("scale","method","SR","PLR","delay_ms","hops"),flush=True)
print("-"*52,flush=True)
for scale in [1.0,2.0,2.5,3.0]:
    inst,DST,GWS=HP.build_instance(scale=scale,hotspot=True,use_std_gw=USE_STD_GW)
    # RL
    try:
        sr,plr,dl,hp,t=rl_metrics(inst)
        print("%-6.1f %-8s %8.4f %8.4f %9.1f %7.2f  (%.0fs)"%(scale,"RL",sr,plr,dl,hp,t),flush=True)
    except Exception as e:
        print("%-6.1f %-8s ERR %s"%(scale,"RL",str(e)[:55]),flush=True)
    # dual
    try:
        t0=time.time(); r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=True,use_nu=True)
        print("%-6.1f %-8s %8.4f %8.4f %9.1f %7.2f  (%.0fs)"%(scale,"dual",r['served_ratio'],r['plr'],r['avgD_ms'],r['mean_hops'],time.time()-t0),flush=True)
    except Exception as e:
        print("%-6.1f %-8s ERR %s"%(scale,"dual",str(e)[:55]),flush=True)
    # greedy
    try:
        g=MP.greedy_nearest_baseline(inst)
        print("%-6.1f %-8s %8.4f %8.4f %9.1f %7.2f"%(scale,"greedy",g['served_ratio'],g['plr'],g['avgD_ms'],g['avg_hops']),flush=True)
    except Exception as e:
        print("%-6.1f %-8s ERR %s"%(scale,"greedy",str(e)[:55]),flush=True)
    print("",flush=True)
print("[done]",flush=True)
