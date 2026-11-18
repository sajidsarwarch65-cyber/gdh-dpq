#!/usr/bin/env python3
# Same 24-cell grid (HD x Cg x Cisl) for nu-only, lambda-only, greedy.
# Full metrics: SR, PLR, delay_ms, hops, time_s.
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np
import hypatia_pop_instance as HP, neural_solver_MCpath as MP, gdh_dpq_v6e as DUAL
SCALE=1.0
HD_GRID=[20.0,30.0,45.0,60.0]
CG_GRID=[8.0,12.0]
CISL_GRID=[5.0,7.0,10.0]
OUT='/home/msarwa02/hard_hunt_ablations_results.txt'

def dual_eval(inst,ul,un):
    t0=time.time(); r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=ul,use_nu=un); dt=time.time()-t0
    return r['served_ratio'],r['plr'],r['avgD_ms'],r['mean_hops'],dt
def greedy_eval(inst):
    t0=time.time(); g=MP.greedy_nearest_baseline(inst); dt=time.time()-t0
    return g['served_ratio'],g['plr'],g['avgD_ms'],g['avg_hops'],dt

methods=[('nu-only',(0,1)),('lam-only',(1,0)),('greedy','gr')]
hdr="%-4s %-4s %-5s %-9s %8s %8s %9s %7s %9s"%("HD","Cg","Cisl","method","SR","PLR","delay_ms","hops","time_s")
print(hdr,flush=True)
with open(OUT,'w') as f: f.write("# Hard-hunt ablations: nu-only, lam-only, greedy. Full metrics.\n%s\n"%hdr)
for hd in HD_GRID:
    for cg in CG_GRID:
        for cisl in CISL_GRID:
            HP.HOTSPOT_DEMAND=hd; HP.rng_seed=7
            inst,DST,GWS=HP.build_instance(scale=SCALE,hotspot=True,use_std_gw=True)
            inst.Cg=np.full(inst.Ng,cg); inst.Cisl=cisl
            for mname,flag in methods:
                try:
                    if flag=='gr': sr,plr,dl,hp,t=greedy_eval(inst)
                    else: sr,plr,dl,hp,t=dual_eval(inst,flag[0],flag[1])
                    line="%-4.0f %-4.0f %-5.1f %-9s %8.4f %8.4f %9.1f %7.2f %9.1f"%(hd,cg,cisl,mname,sr,plr,dl,hp,t)
                except Exception as e:
                    line="%-4.0f %-4.0f %-5.1f %-9s ERR %s"%(hd,cg,cisl,mname,str(e)[:45])
                print(line,flush=True)
                with open(OUT,'a') as f: f.write(line+"\n")
            with open(OUT,'a') as f: f.write("\n")
            print("",flush=True)
print("[done] saved to %s"%OUT,flush=True)
with open(OUT,'a') as f: f.write("[done]\n")
