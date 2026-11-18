#!/usr/bin/env python3
# Shell generalization EXTENSION: add end-to-end DELAY (ms) + TIMING for RL and Classical.
# RL time = construction/inference (fixed policy). Classical time = dual re-solve (price-ascent).
# 3 shells x 6 scales, Cg=8 Cisl=10. Complements shell_generalization_results.txt (SR/PLR/hops).
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, time
import hypatia_pop_instance as HP, neural_solver_MCpath as MP, unroll_solver as U
import gdh_dpq_v6e as DUAL
dev='cpu'; K=4
SCALES=[0.5,1.0,1.5,2.0,2.5,3.0]
CG=8.0; CISL=10.0
GENS={
 'S1_53':  "/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_gdh_dpq_over_isls",
 'S3_70':  "/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data/starlink_shell3_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls",
 'S4_976': "/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data/starlink_shell4_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls",
}
OUT='/home/msarwa02/shell_gen_delay_time_results.txt'

# RL policy (trained on Shell-1)
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
    t0=time.time()
    srs,fs,_=U.unrolled(pol,inst,K=K,greedy=True)
    t_rl=time.time()-t0
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
    return sr,1-sr,(np.mean(ds)*1000 if ds else 0),(np.mean(hs) if hs else 0),t_rl

def dual_metrics(inst):
    t0=time.time()
    r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=1,use_nu=1)
    t_d=time.time()-t0
    return r['served_ratio'],r['plr'],r['avgD_ms'],r['mean_hops'],t_d

hdr="%-8s %-7s %-10s %8s %8s %9s %7s %9s"%("shell","scale","method","SR","PLR","delay_ms","hops","time_s")
print(hdr,flush=True)
open(OUT,'w').write("# Shell generalization: DELAY + TIMING. RL=inference, Classical=dual re-solve. Cg=%.0f Cisl=%.0f\n%s\n"%(CG,CISL,hdr))
for sk,gen in GENS.items():
    for sc in SCALES:
        HP.rng_seed=7; HP.HOTSPOT_DEMAND=30.0
        inst,DST,GWS=HP.build_instance(scale=sc,hotspot=True,use_std_gw=True,gen_path=gen)
        inst.Cg=np.full(inst.Ng,CG); inst.Cisl=CISL
        try:
            sr,plr,dl,hp,t=rl_metrics(inst)
            line="%-8s %-7.1f %-10s %8.4f %8.4f %9.1f %7.2f %9.1f"%(sk,sc,"GDH-RL",sr,plr,dl,hp,t)
        except Exception as e:
            line="%-8s %-7.1f %-10s ERR %s"%(sk,sc,"GDH-RL",str(e)[:35])
        print(line,flush=True); open(OUT,'a').write(line+"\n")
        try:
            sr,plr,dl,hp,t=dual_metrics(inst)
            line="%-8s %-7.1f %-10s %8.4f %8.4f %9.1f %7.2f %9.1f"%(sk,sc,"Classical",sr,plr,dl,hp,t)
        except Exception as e:
            line="%-8s %-7.1f %-10s ERR %s"%(sk,sc,"Classical",str(e)[:35])
        print(line,flush=True); open(OUT,'a').write(line+"\n")
        open(OUT,'a').write("")
    open(OUT,'a').write("\n"); print("",flush=True)
print("[done]",flush=True); open(OUT,'a').write("[done]\n")
