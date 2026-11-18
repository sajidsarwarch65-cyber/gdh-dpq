#!/usr/bin/env python3
# Cross-shell generalization: RL (trained on Shell-1 @53) tested on held-out
# Shell-3 (@70) and Shell-4 (@97.6 polar), across traffic scales, vs classical baseline.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, time
import hypatia_pop_instance as HP, neural_solver_MCpath as MP, unroll_solver as U
import gdh_dpq_v6e as DUAL
dev='cpu'; K=4
SCALES=[0.5,1.0,1.5,2.0,2.5,3.0]
CG=8.0; CISL=10.0
GENS={
 'S1_53':   "/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_gdh_dpq_over_isls",
 'S3_70':   "/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data/starlink_shell3_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls",
 'S4_976':  "/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data/starlink_shell4_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls",
}
OUT='/home/msarwa02/shell_generalization_results.txt'

# RL policy (trained on Shell-1)
try: cfg=MP.default_cfg() if hasattr(MP,'default_cfg') else MP.CFG
except: import train_unroll_vr as TR; cfg=TR.cfg
pol=MP.ConstructionPolicy(cfg)
sd=torch.load('policy_unroll.pt',map_location=dev)
import torch.nn as _nn
ck=sd['enc.node_in.weight'].shape[1]; cur=pol.enc.node_in
if cur.in_features!=ck:
    nl=_nn.Linear(ck,cur.out_features,bias=(cur.bias is not None))
    with torch.no_grad():
        m=min(cur.in_features,ck); nl.weight[:,:m].copy_(cur.weight[:,:m])
        if cur.bias is not None: nl.bias.copy_(cur.bias)
    pol.enc.node_in=nl
pol.load_state_dict(sd,strict=False); pol=U.widen_encoder(pol,dev); pol.eval()

hdr="%-8s %-7s %-8s %8s %8s %7s"%("shell","scale","method","SR","PLR","hops")
print(hdr,flush=True)
open(OUT,'w').write("# Cross-shell generalization: RL(Shell-1) + classical on 3 shells x scales. Cg=%.0f Cisl=%.0f\n%s\n"%(CG,CISL,hdr))
for sk,gen in GENS.items():
    for sc in SCALES:
        HP.rng_seed=7; HP.HOTSPOT_DEMAND=30.0
        inst,DST,GWS=HP.build_instance(scale=sc,hotspot=True,use_std_gw=True,gen_path=gen)
        inst.Cg=np.full(inst.Ng,CG); inst.Cisl=CISL
        # RL (trained on Shell-1, no retrain)
        try:
            srs,fs,_=U.unrolled(pol,inst,K=K,greedy=True)
            sr=float(srs[-1]) if hasattr(srs,'__len__') else float(srs)
            hs=[len(p)-1 for s,gd in fs.items() for gid,(a,p,e) in gd.items() if a>0]
            line="%-8s %-7.1f %-8s %8.4f %8.4f %7.2f"%(sk,sc,"RL",sr,1-sr,np.mean(hs) if hs else 0)
        except Exception as e:
            line="%-8s %-7.1f %-8s ERR %s"%(sk,sc,"RL",str(e)[:35])
        print(line,flush=True); open(OUT,'a').write(line+"\n")
        # Classical: dual (the optimizer, "classical" reference)
        try:
            r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=1,use_nu=1)
            line="%-8s %-7.1f %-8s %8.4f %8.4f %7.2f"%(sk,sc,"Classical",r['served_ratio'],r['plr'],r['mean_hops'])
        except Exception as e:
            line="%-8s %-7.1f %-8s ERR %s"%(sk,sc,"Classical",str(e)[:35])
        print(line,flush=True); open(OUT,'a').write(line+"\n")
        open(OUT,'a').write("")
    open(OUT,'a').write("\n"); print("",flush=True)
print("[done]",flush=True); open(OUT,'a').write("[done]\n")
