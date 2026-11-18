import sys, numpy as np, time; sys.path.insert(0,'.')
import hypatia_pop_instance as HP, gdh_dpq_v6e as V6
rows=[]
for scale in [0.5,1.0,2.0]:
    inst,_,_=HP.build_instance(scale=scale, hotspot=True, use_std_gw=True)
    # time the dual (median of 3)
    ts=[]; 
    for _ in range(3):
        t=time.time(); r=V6.run_gdh_v6e(inst, inst.DST, inst.GW_SERVES, True, True); ts.append(time.time()-t)
    tt=np.median(ts)
    # hop count: need per-flow paths  recompute once with a path trace if exposed, else estimate from delay
    # run_gdh_v6e returns avgD_ms; hop count not in result  extract from a fresh admit if available
    sr=r['served_ratio']; plr=r['plr']; delay=r['avgD_ms']
    rows.append((scale, inst.rho.sum(), r['served'], sr, plr, delay, tt, r.get('nz_nu'), r.get('nz_lam')))
    print('scale %.1f: offered=%.0f served=%.0f SR=%.4f PLR=%.4f delay=%.1fms time=%.1fs nz_nu=%d nz_lam=%d' % (
        scale, inst.rho.sum(), r['served'], sr, plr, delay, tt, r.get('nz_nu',0), r.get('nz_lam',0)), flush=True)
np.save('/home/msarwa02/dual_metrics_3scale.npy', np.array([r[:7] for r in rows]))
print('saved dual_metrics_3scale.npy')
