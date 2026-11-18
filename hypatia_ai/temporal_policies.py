"""Greedy baseline + defer-capable episode runner over TemporalRoutingEnv."""
import numpy as np, math

def _legal_gws(inst, mc, src):
    if mc is None: return list(inst.exit_cands)
    d=int(mc['DST'][src]); gs=mc['GW_SERVES']
    return [g for g in inst.exit_cands if gs[g,d]]

def _route_burst(env, inst, mc, src, need_rate):
    t=env.t
    def wf(u,v):
        e=int(inst.eidx_mat[u,v])
        if e<0: return math.inf
        if env.Cisl-env.L_isl[e,t] < need_rate-1e-9: return math.inf
        if float(inst.isl_lifetime_s[u,v])<inst.cfg.L_MIN: return math.inf
        return inst.isl_prop_s(u,v)
    best=None
    for g in _legal_gws(inst,mc,src):
        if env.Cg[g]-env.L_gw[g,t] < need_rate-1e-9: continue
        path=inst.dijkstra(src, inst.exit_sats(g), wf)
        if path is None or not inst.is_path_valid(path,g): continue
        eids=env.edge_ids(path)
        cost=sum(inst.isl_prop_s(a,b) for a,b in zip(path[:-1],path[1:]))+inst.sg_prop_s(path[-1],g)
        if best is None or cost<best[0]: best=(cost,eids,g,path)
    return None if best is None else (best[1],best[2],best[3])

def run_episode(env, inst, mc, defer_fn=None, rng=None):
    env.reset(); dv=0.0
    for t in range(env.T):
        env.t=t
        evs=env.arrivals[t]
        order=sorted(range(len(evs)), key=lambda i:-(evs[i][1]*(10.0 if evs[i][2] else 1.0)))
        for i in order:
            src,vol,is_hot=evs[i]; value_per_vol=(10.0 if is_hot else 1.0)
            r=_route_burst(env,inst,mc,src,env.rate)
            if r is None: continue
            eids,g,path=r
            if defer_fn is not None:
                fut=max((env.L_isl[e,t:t+3].mean() for e in eids), default=0.0)
                if defer_fn(value_per_vol*vol, vol, is_hot, fut, rng):
                    continue
            before=env.delivered
            env.commit(src,eids,g,vol)
            dv += (env.delivered-before)*value_per_vol
    return dv
print("temporal_policies ready")
