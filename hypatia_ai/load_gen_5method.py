#!/usr/bin/env python3
# LOAD GENERALIZATION - 5 methods (dual, nu-only, lam-only, no-dual, RL) x scales 0.5-4.0.
# Regenerates the "SR/PLR/delay/hops vs traffic scale" 4-panel figure. Reproducible.
import time, numpy as np, torch
import hypatia_pop_instance as HP, neural_solver_MCpath as MP, unroll_solver as U
import gdh_dpq_v6e as DUAL
dev='cpu'; K=4
SCALES=[0.5,1.0,2.0,2.5,3.0,4.0]
OUT='/home/msarwa02/load_gen_5method_results.txt'

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

def rl_metrics(inst):
    srs,fs,_=U.unrolled(pol,inst,K=K,greedy=True)
    sr=float(srs[-1]) if hasattr(srs,'__len__') else float(srs)
    gwl={}; isll={}
    for s,gd in fs.items():
        for gid,(a,p,e) in gd.items():
            if a<=0: continue
            gwl[gid]=gwl.get(gid,0.)+a
            for ek in e: isll[ek]=isll.get(ek,0.)+a
    ds=[]; hs=[]
    for s,gd in fs.items():
        for gid,(a,p,e) in gd.items():
            if a>0:
                hs.append(len(p)-1)
                try: ds.append(inst.full_delay(p,gid,gwl,isll))
                except: pass
    return sr,1-sr,(np.mean(ds)*1000 if ds else 0),(np.mean(hs) if hs else 0)

def dual_metrics(inst,ul,un):
    r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=ul,use_nu=un)
    return r['served_ratio'],r['plr'],r['avgD_ms'],r['mean_hops']

hdr="%-6s %-10s %8s %8s %9s %7s"%("scale","method","SR","PLR","delay_ms","hops")
print(hdr,flush=True)
open(OUT,'w').write("# Load generalization 5-method x scale. Cg=8 Cisl=10.\n%s\n"%hdr)
for sc in SCALES:
    HP.rng_seed=7; HP.HOTSPOT_DEMAND=30.0
    inst,DST,GWS=HP.build_instance(scale=sc,hotspot=True,use_std_gw=True)
    inst.Cg=np.full(inst.Ng,8.0); inst.Cisl=10.0
    for nm,fn in [("dual",lambda i:dual_metrics(i,1,1)),("nu-only",lambda i:dual_metrics(i,0,1)),
                  ("lam-only",lambda i:dual_metrics(i,1,0)),("no-dual",lambda i:dual_metrics(i,0,0))]:
        try:
            t0=time.time(); sr,plr,dl,hp=fn(inst); t=time.time()-t0
            line="%-6.1f %-10s %8.4f %8.4f %9.1f %7.2f"%(sc,nm,sr,plr,dl,hp)
        except Exception as e:
            line="%-6.1f %-10s ERR %s"%(sc,nm,str(e)[:40])
        print(line,flush=True); open(OUT,'a').write(line+"\n")
    try:
        sr,plr,dl,hp=rl_metrics(inst)
        line="%-6.1f %-10s %8.4f %8.4f %9.1f %7.2f"%(sc,"RL",sr,plr,dl,hp)
    except Exception as e:
        line="%-6.1f %-10s ERR %s"%(sc,"RL",str(e)[:40])
    print(line,flush=True); open(OUT,'a').write(line+"\n")
    open(OUT,'a').write("\n"); print("",flush=True)
print("[done]",flush=True); open(OUT,'a').write("[done]\n")
