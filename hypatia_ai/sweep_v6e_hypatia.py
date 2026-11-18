import sys, numpy as np, time; sys.path.insert(0,'/home/msarwa02/hypatia_ai')
sys.path.insert(0,"/home/msarwa02/hypatia/satgenpy")
sys.path.insert(0,"/home/msarwa02/hypatia/paper/satellite_networks_state/gdh_dpq_rl")
import topology as topo, neural_solver as NS
from satgen.dynamic_state import fstate_gdh_dpq as G
from instance_adapter import instance_from_hypatia, gw_positions_km, sat_positions_km
import multicommodity as MC, gdh_dpq_v6e as V6

GEN="/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_gdh_dpq_over_isls"
HOTSPOT_RADIUS=1200.0; HOTSPOT_DEMAND=100.0
SCALES=[0.5,1.0,1.5,2.0,2.5,3.0]

gss,sats,epoch,isls=topo.read_inputs(GEN); gw=gw_positions_km(gss)
satpos=sat_positions_km(sats,str(epoch),str(epoch))
sg,nisl,nbif,vis=topo.build_state_at(sats,gss,epoch,isls,0)
Ns=len(sats); base=G._make_rho(Ns,1.0)
cfg=NS.CFG(MAX_GW_EVAL=16,USE_LEARNED_EDGE_BIAS=False)
inst=instance_from_hypatia(NS.Instance,NS.CFG,sat_graph=sg,vis_list=vis,satellites=sats,
  ground_stations=gss,epoch_str=str(epoch),date_str=str(epoch),rho=base.copy(),
  Cg_val=40,Cisl_val=8,cfg=cfg,gw_pos_km=gw)
mc=MC.build_multicommodity(satpos,gw,base); DST=mc['DST']; GW_SERVES=mc['GW_SERVES']
hs,center,best,near,nvis=MC.site_hotspot(satpos,gw,inst.V,base,pctile=75,radius_km=HOTSPOT_RADIUS)
print(f"Hotspot: {len(hs)} sats @ ({center[0]:.1f},{center[1]:.1f}), nearest gw {near:.0f}km, vis={nvis}",flush=True)

VARIANTS=[("proposed",True,True),("lamonly",True,False),("nuonly",False,True),("nodual",False,False)]
res={k:{m:[] for m in ("sr","plr","avgD","nz_lam","nz_nu")} for k,_,_ in VARIANTS}
for sc in SCALES:
    rho=base.copy()*sc; rho[hs]=HOTSPOT_DEMAND; inst.rho=rho
    print(f"\n--- scale {sc} | offered {rho.sum():.0f} | util {rho.sum()/4000:.2f} ---",flush=True)
    for key,ul,un in VARIANTS:
        t=time.time(); r=V6.run_gdh_v6e(inst,DST,GW_SERVES,use_lambda=ul,use_nu=un)
        res[key]["sr"].append(r["served_ratio"]); res[key]["plr"].append(r["plr"])
        res[key]["avgD"].append(r["avgD_ms"]); res[key]["nz_lam"].append(r["nz_lam"])
        res[key]["nz_nu"].append(r["nz_nu"])
        print(f"  {key:9s}: SR={r['served_ratio']:.4f} PLR={r['plr']:.4f} avgD={r['avgD_ms']:.1f}ms "
              f"nz_lam={r['nz_lam']} nz_nu={r['nz_nu']} [{time.time()-t:.0f}s]",flush=True)

def arr(v): return " ".join(f"{x:.6f}" for x in v)
print("\n%% ===== HYPATIA v6e ABLATION (MATLAB) =====",flush=True)
print(f"hyp_x_scale = [{' '.join('%.2f'%s for s in SCALES)}];")
for key,_,_ in VARIANTS:
    print(f"hyp_{key}_sr   = [{arr(res[key]['sr'])}];")
    print(f"hyp_{key}_plr  = [{arr(res[key]['plr'])}];")
    print(f"hyp_{key}_avgD = [{arr([d/1e3 for d in res[key]['avgD']])}];   %% seconds")
np.save('/home/msarwa02/hypatia_v6e_ablation.npy', res, allow_pickle=True)
print("\nsaved /home/msarwa02/hypatia_v6e_ablation.npy",flush=True)
