import sys, numpy as np, time; sys.path.insert(0,'.')
import hypatia_pop_instance as HP
import gdh_dpq_v6e as V6
from collections import defaultdict

# Re-implement the ascent loop with per-iteration logging, mirroring run_gdh_v6e exactly.
def run_logged(inst, DST, GW_SERVES, n_iters=500):
    import gdh_dpq_v6e as M
    # pull the module-level constants + build the closures by calling into a patched copy
    # Simplest: monkey-capture by re-running the loop here using the same _route/_admit via a trace hook.
    # We instead call run_gdh_v6e but with a logging callback injected via a global list.
    M._CONV_LOG=[]   # each entry: (max_gw_viol, max_isl_viol, nz_nu, tot_viol, served, avgd_ms)
    orig_iters=M.DUAL_MAX_ITERS
    M.DUAL_MAX_ITERS=n_iters
    orig_stop=M.DUAL_EARLY_STOP
    M.DUAL_EARLY_STOP=n_iters+1   # disable early stop so we see full convergence
    # patch: wrap run to log. We add logging by setting an env flag the function checks.
    M._LOG_CONV=True
    r=M.run_gdh_v6e(inst, DST, GW_SERVES, use_lambda=True, use_nu=True)
    log=np.array(M._CONV_LOG) if M._CONV_LOG else np.zeros((0,6))
    M.DUAL_MAX_ITERS=orig_iters; M.DUAL_EARLY_STOP=orig_stop; M._LOG_CONV=False
    return r, log

if __name__=="__main__":
    for scale in [0.5,0.75,1.0]:
        inst,_,_=HP.build_instance(scale=scale, hotspot=True, use_std_gw=True)
        r,log=run_logged(inst, inst.DST, inst.GW_SERVES, n_iters=500)
        np.save('/home/msarwa02/hypatia_conv_scale%.2f.npy'%scale, log)
        print('scale %.2f: SR=%.4f iters_logged=%d  final served=%.0f nz_nu=%d' % (
            scale, r['served_ratio'], len(log), r['served'], r['nz_nu']), flush=True)
    print('saved hypatia_conv_scale*.npy', flush=True)
