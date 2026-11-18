import sys, numpy as np, torch, time, copy; sys.path.insert(0,'.')
import hypatia_pop_instance as HP
import neural_solver_MCpath as MP

DUAL_SR=0.918
t0=time.time()
inst0,hs0,pop0=HP.build_instance(scale=1.0, hotspot=True, use_std_gw=True)
cfg=inst0.cfg; dev=getattr(cfg,'device','cpu')
print("[init] correct instance built in %.0fs: offered=%.0f Ns=%d Ng=%d" % (time.time()-t0, inst0.rho.sum(), inst0.Ns, inst0.Ng), flush=True)

_rho0=inst0.rho.copy()
def jitter(seed):
    # cheap: copy the prebuilt instance, jitter only rho (no raster reload)
    inst=copy.copy(inst0)
    inst.rho=_rho0*np.random.default_rng(seed).uniform(0.9,1.1,size=_rho0.shape)
    return inst

val_insts=[jitter(2000+k) for k in range(3)]
pol=MP.ConstructionPolicy(cfg).to(dev)

def eval_sr(p,inst):
    p.eval()
    with torch.no_grad():
        h,ge,bv,e2=p.encode(inst,dev)
        _,_,sol,gwl,isll=p.rollout(inst,h,ge,bv,e2,greedy=True)
    p.train()
    return MP.compute_reward(inst,sol,gwl,isll)[1]['served_ratio']

gbase=np.mean([eval_sr(pol,vi) for vi in val_insts])
print("[init] val greedy-decode SR %.4f  (dual=%.3f, greedy_baseline=0.763)" % (gbase, DUAL_SR), flush=True)

instance_fn=lambda seed: jitter(seed)
tt=time.time()
try:
    hist=MP.train(pol, cfg, instance_fn=instance_fn, log_every=10, val_insts=val_insts, val_every=5)
    fin=np.mean([eval_sr(pol,vi) for vi in val_insts])
    torch.save(pol.state_dict(),"policy_MCpath_correct.pt")
    print("[done] %.0fs final val SR %.4f vs greedy %.4f (+%.4f) vs dual %.3f saved" % (time.time()-tt, fin, gbase, fin-gbase, DUAL_SR), flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print("MP.train mismatch:", e, flush=True)
