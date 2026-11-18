#!/usr/bin/env python3
# FOCUSED: does nu-only fall below RL at high HD + tight gw + roomy ISL?
# RL + nu-only only (skip slow lam-only, and dual/greedy/nodual already known).
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, torch
import hypatia_pop_instance as HP, neural_solver_MCpath as MP, unroll_solver as U, gdh_dpq_v6e as DUAL
dev='cpu'; K=4; SCALE=1.0
HD_GRID=[45.0,60.0]; CG=8.0; CISL_GRID=[7.0,10.0]
OUT='/home/msarwa02/focused_rl_nu_results.txt'

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
def nu_eval(inst):
    t0=time.time(); r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=0,use_nu=1); dt=time.time()-t0
    return r['served_ratio'],r['plr'],r['avgD_ms'],r['mean_hops'],dt

hdr="%-4s %-4s %-5s %-8s %8s %8s %9s %7s %9s"%("HD","Cg","Cisl","method","SR","PLR","delay","hops","time_s")
print(hdr,flush=True)
with open(OUT,'w') as f: f.write("# Focused RL vs nu-only @ Cg=8, high HD, roomy ISL\n%s\n"%hdr)
for hd in HD_GRID:
    for cisl in CISL_GRID:
        HP.HOTSPOT_DEMAND=hd; HP.rng_seed=7
        inst,DST,GWS=HP.build_instance(scale=SCALE,hotspot=True,use_std_gw=True)
        inst.Cg=np.full(inst.Ng,CG); inst.Cisl=cisl
        for mname,fn in [("RL",rl_eval),("nu-only",nu_eval)]:
            try:
                sr,plr,dl,hp,t=fn(inst)
                line="%-4.0f %-4.0f %-5.1f %-8s %8.4f %8.4f %9.1f %7.2f %9.1f"%(hd,CG,cisl,mname,sr,plr,dl,hp,t)
            except Exception as e:
                line="%-4.0f %-4.0f %-5.1f %-8s ERR %s"%(hd,CG,cisl,mname,str(e)[:40])
            print(line,flush=True)
            with open(OUT,'a') as f: f.write(line+"\n")
        with open(OUT,'a') as f: f.write("\n")
print("[done]",flush=True)
with open(OUT,'a') as f: f.write("[done]\n")
