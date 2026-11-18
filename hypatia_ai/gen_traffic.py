#!/usr/bin/env python3
# TRAFFIC-PATTERN GENERALIZATION on Hypatia: fixed scale, fresh demand+dest draws
# (HP.rng_seed varied). Trained on rng_seed=7; test on unseen seeds. SR/PLR/delay/hops.
import time, numpy as np, torch
import hypatia_pop_instance as HP
import neural_solver_MCpath as MP
import unroll_solver as U
import gdh_dpq_v6e as DUAL
dev='cpu'; K=4; USE_STD_GW=True; SCALE=1.0
TEST_SEEDS=[11,22,33,44,55]   # unseen traffic draws (training used 7)

try:
    cfg = MP.default_cfg() if hasattr(MP,'default_cfg') else MP.CFG
except Exception:
    import train_unroll_vr as TR; cfg = TR.cfg
pol = MP.ConstructionPolicy(cfg).to(dev)
sd = torch.load('policy_unroll.pt', map_location=dev)
import torch.nn as _nn
ck=sd['enc.node_in.weight'].shape[1]; cur=pol.enc.node_in
if cur.in_features!=ck:
    nl=_nn.Linear(ck,cur.out_features,bias=(cur.bias is not None)).to(dev)
    with torch.no_grad():
        m=min(cur.in_features,ck); nl.weight[:,:m].copy_(cur.weight[:,:m])
        if cur.bias is not None: nl.bias.copy_(cur.bias)
    pol.enc.node_in=nl
pol.load_state_dict(sd,strict=False); pol=U.widen_encoder(pol,dev); pol.eval()
print("[policy] loaded + widened",flush=True)

def rl_metrics(inst):
    srs,fs,_=U.unrolled(pol,inst,K=K,greedy=True)
    sr=float(srs[-1]) if hasattr(srs,'__len__') else float(srs)
    gwl={}; isll={}
    for s,gd in fs.items():
        for gid,(a,p,e) in gd.items():
            if a<=0: continue
            gwl[gid]=gwl.get(gid,0.0)+a
            for ek in e: isll[ek]=isll.get(ek,0.0)+a
    ds=[]; hs=[]
    for s,gd in fs.items():
        for gid,(a,p,e) in gd.items():
            if a>0:
                hs.append(len(p)-1)
                try: ds.append(inst.full_delay(p,gid,gwl,isll))
                except: pass
    return sr,1-sr,(np.mean(ds)*1000 if ds else 0.0),(np.mean(hs) if hs else 0.0)

res={m:{'SR':[],'PLR':[],'delay':[],'hops':[]} for m in ['RL','dual','greedy']}
print("\n%-6s %-8s %8s %8s %9s %7s"%("seed","method","SR","PLR","delay_ms","hops"),flush=True)
print("-"*52,flush=True)
for sd_i in TEST_SEEDS:
    HP.rng_seed = sd_i                      # <-- fresh unseen traffic realization
    inst,DST,GWS=HP.build_instance(scale=SCALE,hotspot=True,use_std_gw=USE_STD_GW)
    # RL
    sr,plr,dl,hp=rl_metrics(inst)
    res['RL']['SR'].append(sr); res['RL']['PLR'].append(plr); res['RL']['delay'].append(dl); res['RL']['hops'].append(hp)
    print("%-6d %-8s %8.4f %8.4f %9.1f %7.2f"%(sd_i,"RL",sr,plr,dl,hp),flush=True)
    # dual
    r=DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=True,use_nu=True)
    res['dual']['SR'].append(r['served_ratio']); res['dual']['PLR'].append(r['plr']); res['dual']['delay'].append(r['avgD_ms']); res['dual']['hops'].append(r['mean_hops'])
    print("%-6d %-8s %8.4f %8.4f %9.1f %7.2f"%(sd_i,"dual",r['served_ratio'],r['plr'],r['avgD_ms'],r['mean_hops']),flush=True)
    # greedy
    g=MP.greedy_nearest_baseline(inst)
    res['greedy']['SR'].append(g['served_ratio']); res['greedy']['PLR'].append(g['plr']); res['greedy']['delay'].append(g['avgD_ms']); res['greedy']['hops'].append(g['avg_hops'])
    print("%-6d %-8s %8.4f %8.4f %9.1f %7.2f"%(sd_i,"greedy",g['served_ratio'],g['plr'],g['avgD_ms'],g['avg_hops']),flush=True)
    print("",flush=True)

print("=== MEAN +/- STD across %d unseen traffic draws (scale %.1f) ==="%(len(TEST_SEEDS),SCALE),flush=True)
for m in ['RL','dual','greedy']:
    a=res[m]
    print("%-8s SR %.4f+/-%.4f  PLR %.4f+/-%.4f  delay %.1f+/-%.1f  hops %.2f+/-%.2f"%(
        m, np.mean(a['SR']),np.std(a['SR']), np.mean(a['PLR']),np.std(a['PLR']),
        np.mean(a['delay']),np.std(a['delay']), np.mean(a['hops']),np.std(a['hops'])),flush=True)
print("[done]",flush=True)
