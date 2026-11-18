import sys, numpy as np, torch, time, copy, dataclasses; sys.path.insert(0,'.')
import hypatia_pop_instance as HP
import neural_solver_MCpath as MP

DUAL_SR=0.918; t0=time.time()
inst0,_,_=HP.build_instance(scale=1.0, hotspot=True, use_std_gw=True)
cfg=inst0.cfg
# PATH LEVER ON, tighter exploration, more steps
try: cfg=dataclasses.replace(cfg, USE_LEARNED_EDGE_BIAS=True, entropy_coef=0.003, lr=1e-3, steps=250)
except Exception:
    cfg.USE_LEARNED_EDGE_BIAS=True; cfg.entropy_coef=0.003; cfg.lr=1e-3; cfg.steps=250
for k,v in [('PATH_TOPK',4),('PATH_NDET',2)]:
    try: setattr(cfg,k,v)
    except: pass
inst0.cfg=cfg
print("[init] path-lever ON built %.0fs offered=%.0f EDGE_BIAS=%s lr=%s ent=%s" % (
    time.time()-t0, inst0.rho.sum(), cfg.USE_LEARNED_EDGE_BIAS, cfg.lr, cfg.entropy_coef), flush=True)

_rho0=inst0.rho.copy()
def jitter(seed):
    inst=copy.copy(inst0); inst.rho=_rho0*np.random.default_rng(seed).uniform(0.9,1.1,size=_rho0.shape); return inst
val_insts=[jitter(3000+k) for k in range(3)]
dev=getattr(cfg,'device','cpu')
pol=MP.ConstructionPolicy(cfg).to(dev)

def eval_sr(p,inst):
    p.eval()
    with torch.no_grad():
        h,ge,bv,e2=p.encode(inst,dev); _,_,sol,gwl,isll=p.rollout(inst,h,ge,bv,e2,greedy=True)
    p.train(); return MP.compute_reward(inst,sol,gwl,isll)[1]['served_ratio']

gbase=np.mean([eval_sr(pol,vi) for vi in val_insts])
print("[init] val greedy-decode SR %.4f (dual=%.3f, greedy_baseline=0.763)" % (gbase, DUAL_SR), flush=True)

instance_fn=lambda seed: jitter(seed); tt=time.time()
try:
    hist=MP.train(pol, cfg, instance_fn=instance_fn, log_every=10, val_insts=val_insts, val_every=5)
    fin=np.mean([eval_sr(pol,vi) for vi in val_insts])
    torch.save(pol.state_dict(),"policy_MCpath_v2.pt")
    print("[done] %.0fs final val SR %.4f vs greedy %.4f (+%.4f) vs dual %.3f saved" % (time.time()-tt,fin,gbase,fin-gbase,DUAL_SR), flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print("mismatch:",e,flush=True)
