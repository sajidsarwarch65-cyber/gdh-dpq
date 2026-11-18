import sys, numpy as np, torch, copy, time; sys.path.insert(0,'.')
sys.path.insert(0,"/home/msarwa02/hypatia/satgenpy")
sys.path.insert(0,"/home/msarwa02/hypatia/paper/satellite_networks_state/gdh_dpq_rl")
import topology as topo, neural_solver_MCpath as MP
from satgen.dynamic_state import fstate_gdh_dpq as G
from instance_adapter import instance_from_hypatia, gw_positions_km, sat_positions_km
import multicommodity as MCmod

GEN="/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_gdh_dpq_over_isls"
gss,sats,epoch,isls=topo.read_inputs(GEN); gw=gw_positions_km(gss)
satpos=sat_positions_km(sats,str(epoch),str(epoch))
sg,nisl,nbif,vis=topo.build_state_at(sats,gss,epoch,isls,0)
Ns=len(sats); base=G._make_rho(Ns,1.0)
mc=MCmod.build_multicommodity(satpos,gw,base)
hs,_,_,_,_=MCmod.site_hotspot(satpos,gw,None,base,pctile=75,radius_km=1200.0) if False else (None,)*5

# build the FIXED Hypatia instance once; instance_fn returns it with traffic jitter per seed
def build(scale, seed=None):
    cfg=MP.CFG(MAX_GW_EVAL=16, USE_LEARNED_EDGE_BIAS=False, batch=6, steps=120, lr=3e-3)
    r=base.copy()*scale
    if seed is not None:                      # per-step traffic jitter
        r=r*np.random.default_rng(seed).uniform(0.9,1.1,size=r.shape)
    inst=instance_from_hypatia(MP.Instance,MP.CFG,sat_graph=sg,vis_list=vis,satellites=sats,
        ground_stations=gss,epoch_str=str(epoch),date_str=str(epoch),rho=r,
        Cg_val=40,Cisl_val=8,cfg=cfg,gw_pos_km=gw)
    m=MCmod.build_multicommodity(satpos,gw,r)
    inst.DST=m['DST']; inst.GW_SERVES=m['GW_SERVES']; inst.gw_region=m['gw_region']
    hs2,_,_,_,_=MCmod.site_hotspot(satpos,gw,inst.V,r,pctile=75,radius_km=1200.0)
    inst.rho[hs2]=100.0
    return inst

cfg=MP.CFG(MAX_GW_EVAL=16, USE_LEARNED_EDGE_BIAS=False, batch=6, steps=120, lr=3e-3,
           device="cpu", entropy_coef=0.01, grad_clip=1.0, seed=0)

# fixed validation instances (held-out traffic seeds) at scale 1.0
val_insts=[build(1.0, seed=1000+k) for k in range(3)]
# greedy baseline SR on val (for reference in the log)
pol0=MP.ConstructionPolicy(cfg); pol0.eval()
def gsr(inst):
    with torch.no_grad():
        h,ge,bv,e2=pol0.encode(inst); _,_,sol,gwl,isll=pol0.rollout(inst,h,ge,bv,e2,greedy=True)
    return MP.compute_reward(inst,sol,gwl,isll)[1]['served_ratio']
gbase=np.mean([gsr(vi) for vi in val_insts])
print(f"[init] val greedy SR {gbase:.4f}  (classical dual=0.479, gateway-choice RL was 0.445)",flush=True)

pol=MP.ConstructionPolicy(cfg)
instance_fn=lambda seed: build(1.0, seed=seed)   # scale 1.0, jittered traffic
t0=time.time()
hist=MP.train(pol, cfg, instance_fn=instance_fn, log_every=10, val_insts=val_insts, val_every=5)
torch.save(pol.state_dict(), "policy_MCpath.pt")
# final held-out SR
fsr=np.mean([gsr_pol(pol,vi) for vi in val_insts]) if False else None
def eval_pol(p,inst):
    with torch.no_grad():
        h,ge,bv,e2=p.encode(inst); _,_,sol,gwl,isll=p.rollout(inst,h,ge,bv,e2,greedy=True)
    return MP.compute_reward(inst,sol,gwl,isll)[1]['served_ratio']
fin=np.mean([eval_pol(pol,vi) for vi in val_insts])
print(f"[done] {time.time()-t0:.0f}s  final val SR {fin:.4f} vs greedy {gbase:.4f} (+{fin-gbase:.4f})  saved policy_MCpath.pt",flush=True)
print(f"   COMPARISON: path-RL {fin:.4f} | gateway-choice RL 0.445 | classical dual 0.479 | greedy {gbase:.4f}",flush=True)
