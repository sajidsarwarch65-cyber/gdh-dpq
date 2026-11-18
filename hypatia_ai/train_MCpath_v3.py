import sys, numpy as np, torch, time, dataclasses; sys.path.insert(0,'.')
import hypatia_pop_instance as HP
import neural_solver_MCpath as MP

DUAL_SR=0.918; t0=time.time()
inst,_,_=HP.build_instance(scale=1.0, hotspot=True, use_std_gw=True)
cfg=inst.cfg
try: cfg=dataclasses.replace(cfg, USE_LEARNED_EDGE_BIAS=True, entropy_coef=0.005, lr=1e-3, steps=250)
except Exception:
    cfg.USE_LEARNED_EDGE_BIAS=True; cfg.entropy_coef=0.005; cfg.lr=1e-3; cfg.steps=250
for k,v in [('PATH_TOPK',4),('PATH_NDET',2)]:
    try: setattr(cfg,k,v)
    except: pass
inst.cfg=cfg
dev=getattr(cfg,'device','cpu')
print("[init] FIXED instance, path lever ON, built %.0fs offered=%.0f EDGE_BIAS=%s lr=%s ent=%s" % (
    time.time()-t0, inst.rho.sum(), cfg.USE_LEARNED_EDGE_BIAS, cfg.lr, cfg.entropy_coef), flush=True)

pol=MP.ConstructionPolicy(cfg).to(dev)
def eval_sr(p):
    p.eval()
    with torch.no_grad():
        h,ge,bv,e2=p.encode(inst,dev); _,_,sol,gwl,isll=p.rollout(inst,h,ge,bv,e2,greedy=True)
    p.train(); return MP.compute_reward(inst,sol,gwl,isll)[1]['served_ratio']
print("[init] greedy-decode SR %.4f (dual=%.3f, greedy_baseline=0.763)" % (eval_sr(pol), DUAL_SR), flush=True)

# train + eval on the SAME fixed instance (instance_fn ignores seed, returns the fixed inst)
instance_fn=lambda seed: inst
val_insts=[inst]; tt=time.time(); best=[-1.0]
try:
    hist=MP.train(pol, cfg, instance_fn=instance_fn, log_every=10, val_insts=val_insts, val_every=5)
    fin=eval_sr(pol); torch.save(pol.state_dict(),"policy_MCpath_v3.pt")
    print("[done] %.0fs FINAL greedy-decode SR %.4f vs greedy 0.763 (+%.4f) vs dual %.3f (gap %.3f) saved" % (
        time.time()-tt, fin, fin-0.763, DUAL_SR, DUAL_SR-fin), flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print("mismatch:",e,flush=True)
