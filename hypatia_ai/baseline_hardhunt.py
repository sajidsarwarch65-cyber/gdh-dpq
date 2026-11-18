#!/usr/bin/env python3
# 5 literature baselines on the HARD-HUNT grid: HD{20,30,45,60} x Cg{8,12} x Cisl{5,7,10}
# Metrics: SR, PLR, delay, hops, time. Merges with the RL/dual/no-dual hard-hunt figure.
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np
import hypatia_pop_instance as HP
import hypatia_baselines as BL
SCALE=1.0
HD_GRID=[20.0,30.0,45.0,60.0]
CG_GRID=[8.0,12.0]
CISL_GRID=[5.0,7.0,10.0]
OUT='/home/msarwa02/baseline_hardhunt_results.txt'

hdr="%-4s %-4s %-5s %-12s %8s %8s %9s %7s %8s"%("HD","Cg","Cisl","method","SR","PLR","delay_ms","hops","time_s")
print(hdr,flush=True)
with open(OUT,'w') as f: f.write("# Baselines on hard-hunt grid HDxCgxCisl\n%s\n"%hdr)
for hd in HD_GRID:
    for cg in CG_GRID:
        for cisl in CISL_GRID:
            HP.rng_seed=7; HP.HOTSPOT_DEMAND=hd
            inst,DST,GWS=HP.build_instance(scale=SCALE,hotspot=True,use_std_gw=True)
            inst.Cg=np.full(inst.Ng,cg); inst.Cisl=cisl
            for name,fn in BL.ALL_BASELINES.items():
                try:
                    t0=time.time(); r=fn(inst); t=time.time()-t0
                    line="%-4.0f %-4.0f %-5.1f %-12s %8.4f %8.4f %9.1f %7.2f %8.1f"%(hd,cg,cisl,name,r['served_ratio'],r['plr'],r['avgD_ms'],r['avg_hops'],t)
                except Exception as e:
                    line="%-4.0f %-4.0f %-5.1f %-12s ERR %s"%(hd,cg,cisl,name,str(e)[:40])
                print(line,flush=True)
                with open(OUT,'a') as f: f.write(line+"\n")
            with open(OUT,'a') as f: f.write("\n")
    print("",flush=True)
print("[done] saved to %s"%OUT,flush=True)
with open(OUT,'a') as f: f.write("[done]\n")
