import sys, numpy as np; sys.path.insert(0,'.')
import hypatia_pop_instance as HP, gdh_dpq_v6e as M
for scale in [0.5,0.75,1.0]:
    inst,_,_=HP.build_instance(scale=scale, hotspot=True, use_std_gw=True)
    M._LOG_CONV=True; M._CONV_LOG=[]
    M.DUAL_MAX_ITERS=200; M.DUAL_EARLY_STOP=999   # full 200 iters, no early stop
    r=M.run_gdh_v6e(inst, inst.DST, inst.GW_SERVES, use_lambda=True, use_nu=True)
    log=np.array(M._CONV_LOG)
    np.save('/home/msarwa02/hypatia_conv_scale%.2f.npy'%scale, log)
    M._LOG_CONV=False
    print('scale %.2f: SR=%.4f logged=%d iters | final: max_gw=%.1f max_isl=%.1f nz_nu=%d served=%.0f delay=%.1fms' % (
        scale, r['served_ratio'], len(log), log[-1,0], log[-1,1], int(log[-1,2]), log[-1,4], log[-1,5]), flush=True)
print('saved hypatia_conv_scale*.npy', flush=True)
