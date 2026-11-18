#!/usr/bin/env python3
# HARD-REGIME HUNT: does RL EVER beat no-dual on SR? Sweep HD x Cg x Cisl.
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, torch
import hypatia_pop_instance as HP, neural_solver_MCpath as MP, unroll_solver as U, gdh_dpq_v6e as DUAL
dev='cpu'; K=4; SCALE=1.0
OUT='/home/msarwa02/hard_hunt_results.txt'
# coordination-pressure grid
HD_GRID=[20.0, 30.0, 45.0, 60.0]      # hotspot demand (was 30 default)
CG_GRID=[8.0, 12.0]                    # tight gateway
CISL_GRID=[5.0, 7.0, 10.0]             # tight-ish ISL

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

def rl_sr(inst):
    t0=time.time(); srs,fs,_=U.unrolled(pol,inst,K=K,greedy=True); dt=time.time()-t0
    sr=float(srs[-1]) if hasattr(srs,'__len__') else float(srs)
    return sr,dt
def nod_sr(inst):
    t0=time.time(); r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=0,use_nu=0); dt=time.time()-t0
    return r['served_ratio'],dt
def dual_sr(inst):
    t0=time.time(); r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=1,use_nu=1); dt=time.time()-t0
    return r['served_ratio'],dt

hdr="%-5s %-5s %-6s %8s %8s %8s %10s"%("HD","Cg","Cisl","RL_SR","noD_SR","dual_SR","RL-noD")
print(hdr,flush=True)
with open(OUT,'w') as f: f.write("# HARD HUNT: RL vs no-dual SR. positive RL-noD = RL WINS.\n%s\n"%hdr)
rl_wins=[]
for hd in HD_GRID:
    for cg in CG_GRID:
        for cisl in CISL_GRID:
            HP.HOTSPOT_DEMAND=hd; HP.rng_seed=7
            inst,DST,GWS=HP.build_instance(scale=SCALE,hotspot=True,use_std_gw=True)
            inst.Cg=np.full(inst.Ng,cg); inst.Cisl=cisl
            try:
                rl,trl=rl_sr(inst); nod,tnod=nod_sr(inst); du,tdu=dual_sr(inst)
                diff=rl-nod
                mark=" <-- RL WINS" if diff>0.005 else ""
                if diff>0.005: rl_wins.append((hd,cg,cisl,diff))
                line="%-5.0f %-5.0f %-6.1f %8.4f %8.4f %8.4f %+10.4f%s"%(hd,cg,cisl,rl,nod,du,diff,mark)
            except Exception as e:
                line="%-5.0f %-5.0f %-6.1f ERR %s"%(hd,cg,cisl,str(e)[:40])
            print(line,flush=True)
            with open(OUT,'a') as f: f.write(line+"\n")
print("\n=== SUMMARY ===",flush=True)
if rl_wins:
    print("RL beats no-dual (>0.5pp) in %d/%d cells:"%(len(rl_wins),len(HD_GRID)*len(CG_GRID)*len(CISL_GRID)),flush=True)
    for w in rl_wins: print("  HD=%.0f Cg=%.0f Cisl=%.1f  RL-noD=+%.4f"%w,flush=True)
else:
    print("RL NEVER beats no-dual by >0.5pp across all %d cells. No niche found."%(len(HD_GRID)*len(CG_GRID)*len(CISL_GRID)),flush=True)
with open(OUT,'a') as f:
    f.write("\n# SUMMARY: RL wins in %d cells\n"%len(rl_wins))
    for w in rl_wins: f.write("#  HD=%.0f Cg=%.0f Cisl=%.1f +%.4f\n"%w)
    f.write("[done]\n")
print("[done] -> %s"%OUT,flush=True)
