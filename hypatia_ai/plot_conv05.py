import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log=np.load('/home/msarwa02/hypatia_conv_scale0.50.npy')  # cols: max_gw, max_isl, nz_nu, tot_viol, served, delay_ms
it=np.arange(len(log))
max_gw, max_isl, nz_nu, tot_viol, served, delay = [log[:,i] for i in range(6)]
def runmin(x): return np.minimum.accumulate(x)
def runbest(x): return np.maximum.accumulate(x)   # for served (higher better)

fig,ax=plt.subplots(2,3,figsize=(15,8))
ax[0,0].semilogy(it,max_gw,'r-',alpha=.4,label='per-iter'); ax[0,0].semilogy(it,runmin(max_gw),'r-',lw=2,label='running min')
ax[0,0].set_title('(i) max GW violation'); ax[0,0].set_xlabel('iteration'); ax[0,0].set_ylabel('Gbps'); ax[0,0].legend(); ax[0,0].grid(True,alpha=.3)

ax[0,1].semilogy(it,np.maximum(max_isl,1e-3),'r-',alpha=.4,label='per-iter'); ax[0,1].semilogy(it,np.maximum(runmin(max_isl),1e-3),'r-',lw=2,label='running min')
ax[0,1].set_title('(ii) max ISL violation'); ax[0,1].set_xlabel('iteration'); ax[0,1].set_ylabel('Gbps'); ax[0,1].legend(); ax[0,1].grid(True,alpha=.3)

ax[0,2].plot(it,nz_nu,'r-',lw=2); ax[0,2].set_title('(iii) ISL-price activation |nu>0|'); ax[0,2].set_xlabel('dual iteration'); ax[0,2].set_ylabel('active ISL prices'); ax[0,2].grid(True,alpha=.3)

ax[1,0].semilogy(it,np.maximum(tot_viol,1e-2),'r-',alpha=.4,label='per-iter'); ax[1,0].semilogy(it,np.maximum(runmin(tot_viol),1e-2),'r-',lw=2,label='running min')
ax[1,0].set_title('(iv) total capacity violation'); ax[1,0].set_xlabel('dual iteration'); ax[1,0].set_ylabel('Gbps'); ax[1,0].legend(); ax[1,0].grid(True,alpha=.3)

ax[1,1].plot(it,served,'r-',alpha=.4,label='per-iter'); ax[1,1].plot(it,runbest(served),'r-',lw=2,label='running best')
ax[1,1].set_title('(v) served-traffic convergence'); ax[1,1].set_xlabel('dual iteration'); ax[1,1].set_ylabel('served (Gbps)'); ax[1,1].legend(); ax[1,1].grid(True,alpha=.3)

# delay: running-best = min delay so far among best-served
ax[1,2].plot(it,delay,'r-',alpha=.4,label='per-iter'); ax[1,2].plot(it,runmin(delay),'r-',lw=2,label='running best')
ax[1,2].set_title('(vi) mean delay convergence'); ax[1,2].set_xlabel('dual iteration'); ax[1,2].set_ylabel('mean E2E delay (ms)'); ax[1,2].legend(); ax[1,2].grid(True,alpha=.3)

plt.suptitle('Hypatia dual convergence  scale 0.5 (offered 379 Gbps)', fontsize=13)
plt.tight_layout()
plt.savefig('/home/msarwa02/hypatia_conv_scale05.png', dpi=120, bbox_inches='tight')
print('saved hypatia_conv_scale05.png')
print('--- final values ---')
print('max_gw=%.2f  max_isl=%.2f  nz_nu=%d  tot_viol=%.1f  served=%.1f  delay=%.1fms' % (
    max_gw[-1], max_isl[-1], int(nz_nu[-1]), tot_viol[-1], served[-1], delay[-1]))
print('served: start=%.1f  final=%.1f  converged-by-iter=%d' % (
    served[0], served[-1], int(np.argmax(runbest(served)>=served.max()-1e-6))))
print('nz_nu: start=%d  final=%d  (ISL-price activation)' % (int(nz_nu[0]), int(nz_nu[-1])))
print('max_gw: start=%.1f final=%.1f | max_isl: start=%.1f final=%.1f' % (max_gw[0],max_gw[-1],max_isl[0],max_isl[-1]))
