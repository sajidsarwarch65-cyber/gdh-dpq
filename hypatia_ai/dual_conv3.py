import sys, numpy as np; sys.path.insert(0,'.')
import hypatia_pop_instance as HP, gdh_dpq_v6e as M
scale=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
niter=int(sys.argv[2]) if len(sys.argv)>2 else 150
print("building scale %.2f..."%scale, flush=True)
inst,_,_=HP.build_instance(scale=scale, hotspot=True, use_std_gw=True)
print("built. running %d iters with logging..."%niter, flush=True)
M._LOG_CONV=True; M._CONV_LOG=[]
M.DUAL_MAX_ITERS=niter; M.DUAL_EARLY_STOP=niter+1
r=M.run_gdh_v6e(inst, inst.DST, inst.GW_SERVES, use_lambda=True, use_nu=True)
log=np.array(M._CONV_LOG)
np.save('/home/msarwa02/hypatia_conv_scale%.2f.npy'%scale, log)
print("DONE scale %.2f: SR=%.4f logged=%d final_nz_nu=%d served=%.0f"%(scale,r['served_ratio'],len(log),int(log[-1,2]),log[-1,4]), flush=True)
