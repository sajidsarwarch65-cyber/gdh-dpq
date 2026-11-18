"""Lightweight learned temporal policy + REINFORCE trainer."""
import numpy as np, math, torch, torch.nn as nn, torch.nn.functional as F

class ResPolicy(nn.Module):
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden), nn.ReLU(),
                               nn.Linear(hidden,hidden), nn.ReLU(),
                               nn.Linear(hidden,2))
    def forward(self, x):
        out=self.net(x)
        mean=torch.sigmoid(out[...,0])          # aggressiveness mean in (0,1)
        conc=8.0+8.0*torch.sigmoid(out[...,1])    # concentration [8,16]: explore NEAR the mean (cleaner signal)
        alpha=mean*conc + 1e-3
        beta=(1-mean)*conc + 1e-3
        return alpha, beta

def _burst_feats(env, inst, risk, cost_eids, epoch):
    if not cost_eids:
        return torch.tensor([0.,0.,epoch/env.T,1.],dtype=torch.float32)
    r=float(np.mean([risk[e] for e in cost_eids]))
    occ=float(max(env.L_isl[e,env.t]/env.Cisl for e in cost_eids))
    return torch.tensor([r, occ, epoch/env.T, 1.],dtype=torch.float32)

def run_episode_learned(env, inst, mc, risk, policy, greedy=False, collect=False):
    from temporal_policies import _legal_gws, _route_burst
    env.reset(); val=0.0; logps=[]; step_rewards=[]
    for t in range(env.T):
        env.t=t
        for src,vol,is_hot in sorted(env.arrivals[t],key=lambda x:-(x[1]*(10 if x[2] else 1))):
            need=env.rate; tt=env.t; vpv=10.0 if is_hot else 1.0
            if is_hot:
                r=_route_burst(env,inst,mc,src,need)
                if r is not None:
                    b=env.delivered; env.commit(src,r[0],r[1],vol); dv=(env.delivered-b)*vpv
                    val+=dv
                continue
            r0=_route_burst(env,inst,mc,src,need)
            eids0=r0[0] if r0 else []
            feat=_burst_feats(env,inst,risk,eids0,t)
            alpha,beta=policy(feat)
            dist=torch.distributions.Beta(alpha,beta)
            a = dist.mean if greedy else dist.rsample()
            if collect: logps.append(dist.log_prob(a.clamp(1e-4,1-1e-4)))
            a_val=float(a.detach())
            def wf(u,v):
                e=int(inst.eidx_mat[u,v])
                if e<0: return math.inf
                if env.Cisl-env.L_isl[e,tt] < need-1e-9: return math.inf
                if float(inst.isl_lifetime_s[u,v])<inst.cfg.L_MIN: return math.inf
                pen = a_val * risk[e] * 10.0 * inst.isl_prop_s(u,v)
                return inst.isl_prop_s(u,v) + pen
            best=None
            for g in _legal_gws(inst,mc,src):
                if env.Cg[g]-env.L_gw[g,tt] < need-1e-9: continue
                path=inst.dijkstra(src, inst.exit_sats(g), wf)
                if path is None or not inst.is_path_valid(path,g): continue
                eids=env.edge_ids(path)
                c=sum(inst.isl_prop_s(a2,b2) for a2,b2 in zip(path[:-1],path[1:]))+inst.sg_prop_s(path[-1],g)
                if best is None or c<best[0]: best=(c,eids,g)
            r=None if best is None else (best[1],best[2])
            dv=0.0
            if r is not None:
                b=env.delivered; env.commit(src,r[0],r[1],vol); dv=(env.delivered-b)*vpv
                val+=dv
            if collect: step_rewards.append(dv)
    return val, logps, step_rewards
