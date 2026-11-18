"""Static per-edge hotspot-criticality: how often an edge lies on a shortest path from
a hotspot source to a legal hub gateway. This is the STABLE structural feature the
learned policy uses to infer which edges to reserve -- the observable proxy for the
oracle's schedule cheat (edges near hotspot routes are chronically hotspot-critical)."""
import numpy as np, math

def edge_hotspot_risk(inst, mc, hotspot_sats):
    E=inst.n_edges; risk=np.zeros(E)
    gs=mc['GW_SERVES']; DST=mc['DST']
    def wf(u,v):
        e=int(inst.eidx_mat[u,v])
        if e<0: return math.inf
        if float(inst.isl_lifetime_s[u,v])<inst.cfg.L_MIN: return math.inf
        return inst.isl_prop_s(u,v)
    n=0
    for src in hotspot_sats:
        src=int(src); d=int(DST[src])
        legal=[g for g in inst.exit_cands if gs[g,d]]
        best=None
        for g in legal:
            path=inst.dijkstra(src, inst.exit_sats(g), wf)
            if path is None: continue
            c=sum(inst.isl_prop_s(a,b) for a,b in zip(path[:-1],path[1:]))+inst.sg_prop_s(path[-1],g)
            if best is None or c<best[0]: best=(c,path)
        if best is None: continue
        n+=1
        for a,b in zip(best[1][:-1],best[1][1:]):
            risk[int(inst.eidx_mat[a,b])]+=1.0
    if n>0: risk/=n
    return risk
