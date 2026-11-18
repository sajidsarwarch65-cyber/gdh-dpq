"""REINFORCE trainer for the temporal reservation policy."""
import sys, numpy as np, torch, time
sys.path.insert(0,'/home/msarwa02/hypatia_ai')
sys.path.insert(0,"/home/msarwa02/hypatia/satgenpy")
sys.path.insert(0,"/home/msarwa02/hypatia/paper/satellite_networks_state/gdh_dpq_rl")
import topology as topo, neural_solver_SR as SR
from satgen.dynamic_state import fstate_gdh_dpq as G
from instance_adapter import instance_from_hypatia, gw_positions_km, sat_positions_km
import multicommodity as MC
from temporal_env import TemporalRoutingEnv
from temporal_risk import edge_hotspot_risk
from temporal_learned import ResPolicy, run_episode_learned

GEN="/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_gdh_dpq_over_isls"
gss,sats,epoch,isls=topo.read_inputs(GEN); gw=gw_positions_km(gss)
satpos=sat_positions_km(sats,str(epoch),str(epoch))
sg,nisl,nbif,vis=topo.build_state_at(sats,gss,epoch,isls,0)
Ns=len(sats); base=G._make_rho(Ns,1.0)
cfg=SR.CFG(MAX_GW_EVAL=16,USE_LEARNED_EDGE_BIAS=False)
inst=instance_from_hypatia(SR.Instance,SR.CFG,sat_graph=sg,vis_list=vis,satellites=sats,
  ground_stations=gss,epoch_str=str(epoch),date_str=str(epoch),rho=base.copy(),
  Cg_val=40,Cisl_val=8,cfg=cfg,gw_pos_km=gw)
mc=MC.build_multicommodity(satpos,gw,base)
inst.DST=mc['DST']; inst.GW_SERVES=mc['GW_SERVES']; inst.gw_region=mc['gw_region']
hs,_,_,_,_=MC.site_hotspot(satpos,gw,inst.V,base,pctile=75,radius_km=1200.0)
risk=edge_hotspot_risk(inst,mc,hs)

def make_env(seed):
    return TemporalRoutingEnv(inst,T=12,rate=8.0,seed=seed,base_bursts_per_epoch=200,
        hotspot_bursts_per_epoch=12,base_vol=(20.,40.),hotspot_vol=(60.,120.),hotspot_sats=hs)

TRAIN_SEEDS=[1,2,3,4,5,6]; TEST_SEEDS=[101,102,103,104]
pol=ResPolicy(in_dim=4,hidden=32)
opt=torch.optim.Adam(pol.parameters(),lr=3e-3)
STEPS=int(__import__('os').environ.get("TSTEPS","120"))

class Fixed(torch.nn.Module):
    def __init__(self,a): super().__init__(); self.a=max(1e-3,min(1-1e-3,a))
    def forward(self,x): c=20.; return torch.tensor(self.a*c),torch.tensor((1-self.a)*c)
def eval_pol(p,seeds):
    return np.mean([run_episode_learned(make_env(s),inst,mc,risk,p,greedy=True)[0] for s in seeds])
def mean_a(p):
    import torch as _t
    # probe the policy mean aggressiveness across a grid of feature states
    xs=[[r,o,e,1.] for r in (0.0,0.15,0.3) for o in (0.0,0.5,1.0) for e in (0.0,0.5,1.0)]
    al,be=p(_t.tensor(xs,dtype=_t.float32))
    return float((al/(al+be)).mean())

g_greedy=eval_pol(Fixed(0.0),TEST_SEEDS)
g_fixed1=eval_pol(Fixed(1.0),TEST_SEEDS)
print(f"baselines: greedy(a=0)={g_greedy:.0f}  fixed(a=1)={g_fixed1:.0f}  oracle_ceiling~greedy+5674",flush=True)

best=-1e9
for step in range(STEPS):
    # BATCH: average advantage-weighted loss over several seeds -> cancels the bimodal ±1145 noise
    opt.zero_grad(); losses=[]; Rs_last=Rg_last=0.
    for bi in range(4):
        seed=TRAIN_SEEDS[(step*4+bi)%len(TRAIN_SEEDS)]
        env=make_env(seed)
        Rs,logps,rews=run_episode_learned(env,inst,mc,risk,pol,greedy=False,collect=True)
        with torch.no_grad():
            Rg,_,_=run_episode_learned(make_env(seed),inst,mc,risk,pol,greedy=True)
        if logps:
            losses.append(-((Rs-Rg)/max(abs(Rg),1.0)) * torch.stack(logps).sum())
        Rs_last,Rg_last=Rs,Rg
    Rs,Rg=Rs_last,Rg_last; adv=Rs-Rg
    if losses:
        (torch.stack(losses).mean()).backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(),1.0); opt.step()
    if step%10==0 or step==STEPS-1:
        v=eval_pol(pol,TEST_SEEDS)
        tag="*" if v>best else " "
        if v>best: best=v; torch.save(pol.state_dict(),"/home/msarwa02/hypatia_ai/policy_temporal.pt")
        print(f"[{step:3d}] Rs={Rs:.0f} Rg={Rg:.0f} adv={adv:+.0f} | test_value={v:.0f} "
              f"(vs greedy {v-g_greedy:+.0f}, vs fixed1 {v-g_fixed1:+.0f}) mean_a={mean_a(pol):.3f} {tag}",flush=True)
print(f"\nBEST test value={best:.0f}  (greedy {g_greedy:.0f}, fixed_a1 {g_fixed1:.0f}, oracle~{g_greedy+5674:.0f})",flush=True)
print(f"learned recovers {100*(best-g_greedy)/5674:.0f}% of the oracle gap",flush=True)
