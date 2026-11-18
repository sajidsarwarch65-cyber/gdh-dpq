#!/usr/bin/env python3
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, torch
import hypatia_pop_instance as HP, neural_solver_MCpath as MP, unroll_solver as U, gdh_dpq_v6e as DUAL
dev='cpu'; K=4
SCALE=1.0
CG_LOW=15.0                              # low gateway cap (gap-shrinking regime)
CISL_GRID=[4.0,6.0,8.0,10.0,12.0,16.0]   # tight -> loose ISL capacity
OUT='/home/msarwa02/cisl_sweep_results.txt'

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

def rl_eval(inst):
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
    return sr,1-sr,(np.mean(ds)*1000 if ds else 0.0),(np.mean(hs) if hs else 0.0),dt

def dual_eval(inst,ul,un):
    t0=time.time(); r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=ul,use_nu=un); dt=time.time()-t0
    return r['served_ratio'],r['plr'],r['avgD_ms'],r['mean_hops'],dt

def greedy_eval(inst):
    t0=time.time(); g=MP.greedy_nearest_baseline(inst); dt=time.time()-t0
    return g['served_ratio'],g['plr'],g['avgD_ms'],g['avg_hops'],dt

methods=[('RL','rl'),('dual',(1,1)),('lam-only',(1,0)),('nu-only',(0,1)),('no-dual',(0,0)),('greedy','gr')]
hdr="%-6s %-9s %8s %8s %9s %7s %9s"%("Cisl","method","SR","PLR","delay_ms","hops","time_s")
print(hdr,flush=True)
with open(OUT,'w') as f: f.write("# Cisl sweep, Cg=%.0f scale=%.1f, all methods fresh per point (RL NOT held-out)\n%s\n"%(CG_LOW,SCALE,hdr))
for cisl in CISL_GRID:
    HP.rng_seed=7
    inst,DST,GWS=HP.build_instance(scale=SCALE,hotspot=True,use_std_gw=True)
    inst.Cg=np.full(inst.Ng,CG_LOW); inst.Cisl=cisl      # <-- FIXED: Cg array, Cisl scalar
    for mname,flag in methods:
        try:
            if flag=='rl': sr,plr,dl,hp,t=rl_eval(inst)
            elif flag=='gr': sr,plr,dl,hp,t=greedy_eval(inst)
            else: sr,plr,dl,hp,t=dual_eval(inst,flag[0],flag[1])
            line="%-6.1f %-9s %8.4f %8.4f %9.1f %7.2f %9.1f"%(cisl,mname,sr,plr,dl,hp,t)
        except Exception as e:
            line="%-6.1f %-9s ERR %s"%(cisl,mname,str(e)[:50])
        print(line,flush=True)
        with open(OUT,'a') as f: f.write(line+"\n")
    with open(OUT,'a') as f: f.write("\n")
    print("",flush=True)
print("[done] saved to %s"%OUT,flush=True)
with open(OUT,'a') as f: f.write("[done]\n")
