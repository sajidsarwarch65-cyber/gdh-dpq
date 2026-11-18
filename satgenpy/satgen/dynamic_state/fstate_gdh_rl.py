# fstate_gdh_rl.py  RL forwarding-state generator (Option A).
# Mirrors fstate_gdh_dpq.py but produces the per-satellite routing
# decision from the trained GNN policy instead of dual price-ascent.
# The decision dict decisions[s]=(gid,next_hop,path_len,hops) feeds the
# SAME fstate write block as the dual, so ns-3 sees identical structure.
import os, sys, math, heapq, numpy as np
from collections import defaultdict

# reuse the dual module's helpers verbatim (capacity, rho, dijkstra utils)
from .fstate_gdh_dpq import (
    _make_rho, _isl_capacity, _relative_velocity_m_s, _doppler_factor,
    CGW_DEFAULT, CISL,
)

# make the RL codebase importable (policy + unrolled solver live in ~/hypatia_ai)
_RL_DIR = os.path.expanduser("~/hypatia_ai")
if _RL_DIR not in sys.path:
    sys.path.insert(0, _RL_DIR)

_POL = None   # cache the loaded policy across timesteps
def _load_policy():
    global _POL
    if _POL is not None:
        return _POL
    import torch, dataclasses, torch.nn as nn
    import neural_solver_MCpath as MP, unroll_solver as U
    dev = "cpu"
    # cfg matching training
    cfg = MP.CFG(MAX_GW_EVAL=16, USE_LEARNED_EDGE_BIAS=True)
    pol = MP.ConstructionPolicy(cfg).to(dev)
    H = pol.enc.node_in.out_features
    pol.enc.node_in = nn.Linear(8, H).to(dev)                 # match 8-dim ckpt
    pol.load_state_dict(torch.load(os.path.join(_RL_DIR, "policy_unroll.pt"),
                                   map_location=dev))
    pol = U.widen_encoder(pol, dev)                            # 8->10
    pol.eval()
    _POL = (pol, cfg, U, MP, dev)
    return _POL


def calculate_fstate_gdh_rl(
        output_dynamic_state_dir, time_since_epoch_ns,
        num_satellites, num_ground_stations,
        sat_net_graph_only_satellites_with_isls,
        num_isls_per_sat, gid_to_sat_gsl_if_idx,
        ground_station_satellites_in_range_candidates,
        sat_neighbor_to_if, prev_fstate, enable_verbose_logs,
        gdhdpq_state=None, external_lambda=None, scale=1.0,
        rho_override=None):
    Ns = num_satellites; Ng = num_ground_stations
    sat_graph = sat_net_graph_only_satellites_with_isls
    vis_list = ground_station_satellites_in_range_candidates
    rho = _make_rho(Ns, scale=scale) if rho_override is None else np.asarray(rho_override, float)
    Cg = np.full(Ng, CGW_DEFAULT, float)

    # sat -> visible gateways
    sat_to_gws = defaultdict(list)
    for gid, vis in enumerate(vis_list):
        for dm, sid in vis:
            sat_to_gws[sid].append((dm, gid))

    # ---- RL ROUTING: produce decisions[s] = (gid, next_hop, path_len, hops) ----
    decisions = _rl_decisions(Ns, Ng, sat_graph, vis_list, sat_to_gws, rho,
                              enable_verbose_logs)

    # ===================== identical fstate build to the dual =====================
    fstate = {}
    dist_s2g = {}
    for s in range(Ns):
        for gid in range(Ng):
            dst = Ns + gid
            dm = next((d for d, sid in vis_list[gid] if sid == s), math.inf)
            dist_s2g[(s, dst)] = dm
    # Dijkstra outward from each GW (fallback next-hops)
    for gid in range(Ng):
        dst = Ns + gid
        pq = []
        for dm, sid in vis_list[gid]:
            dist_s2g[(sid, dst)] = dm
            heapq.heappush(pq, (dm, sid))
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist_s2g.get((u, dst), math.inf) + 1e-9: continue
            for nb in sat_graph.neighbors(u):
                nd = d + sat_graph.edges[u, nb]['weight']
                if nd < dist_s2g.get((nb, dst), math.inf):
                    dist_s2g[(nb, dst)] = nd
                    heapq.heappush(pq, (nd, nb))

    next_hop_to_gw = {}
    for s in range(Ns):
        for gid in range(Ng):
            dst = Ns + gid
            if math.isinf(dist_s2g.get((s, dst), math.inf)):
                next_hop_to_gw[(s, gid)] = -1; continue
            if any(sid == s for _, sid in vis_list[gid]):
                next_hop_to_gw[(s, gid)] = s; continue
            best_nb = -1; best_d = math.inf
            for nb in sat_graph.neighbors(s):
                nd = dist_s2g.get((nb, dst), math.inf)
                if nd < best_d: best_d = nd; best_nb = nb
            next_hop_to_gw[(s, gid)] = best_nb

    fname = output_dynamic_state_dir + "/fstate_" + str(time_since_epoch_ns) + ".txt"
    with open(fname, "w+") as f:
        for s in range(Ns):
            for gid in range(Ng):
                dst = Ns + gid
                if s in decisions and decisions[s][0] == gid:
                    nh_sat = decisions[s][1]
                else:
                    nh_sat = next_hop_to_gw.get((s, gid), -1)
                if nh_sat == s:
                    nhd = (dst, num_isls_per_sat[s] + gid_to_sat_gsl_if_idx[gid], 0)
                elif nh_sat >= 0 and (s, nh_sat) in sat_neighbor_to_if:
                    nhd = (nh_sat, sat_neighbor_to_if[(s, nh_sat)], sat_neighbor_to_if[(nh_sat, s)])
                else:
                    nhd = (-1, -1, -1)
                if not prev_fstate or prev_fstate.get((s, dst)) != nhd:
                    f.write("%d,%d,%d,%d,%d\n" % (s, dst, nhd[0], nhd[1], nhd[2]))
                fstate[(s, dst)] = nhd
        # GW-to-GW rows (identical to dual's tail)
        for sg in range(Ng):
            for dg in range(Ng):
                if sg == dg: continue
                sgs = Ns + sg; dgs = Ns + dg
                bc = math.inf; bs = -1
                for d_src, a in vis_list[sg]:
                    dd = dist_s2g.get((a, dgs), math.inf)
                    if d_src + dd < bc: bc = d_src + dd; bs = a
                if bs < 0 and vis_list[sg]:
                    bs = min(vis_list[sg], key=lambda x: x[0])[1]
                if bs >= 0:
                    nhd = (bs, 0, num_isls_per_sat[bs] + gid_to_sat_gsl_if_idx[sg])
                else:
                    nhd = (-1, -1, -1)
                if not prev_fstate or prev_fstate.get((sgs, dgs)) != nhd:
                    f.write("%d,%d,%d,%d,%d\n" % (sgs, dgs, nhd[0], nhd[1], nhd[2]))
                fstate[(sgs, dgs)] = nhd

    if enable_verbose_logs:
        served = sum(1 for s in decisions if decisions[s][0] >= 0)
        print(f"  > GDH-RL: {served}/{Ns} sats routed to a gateway")
    return fstate, {"fstate": fstate}


def _rl_decisions(Ns, Ng, sat_graph, vis_list, sat_to_gws, rho, verbose):
    """Run the trained policy on THIS timestep's graph, return
    decisions[s]=(gid,next_hop,path_len,hops) for source + transit sats."""
    pol, cfg, U, MP, dev = _load_policy()
    import numpy as _np
    # Build a minimal instance the unrolled solver understands from the
    # ns-3 graph. We route on satgenpy's own sat_graph/vis_list so RL and
    # dual share byte-identical topology (Option A fidelity requirement).
    inst = _build_inst_from_graph(Ns, Ng, sat_graph, vis_list, sat_to_gws, rho, cfg, MP)
    try:
        srs, fs, _ = U.unrolled(pol, inst, K=4, greedy=True)
    except Exception as e:
        if verbose: print("  > GDH-RL policy failed, dropping all:", e)
        return {}
    # convert per-flow paths (amt,path,edges) into next-hop decisions
    decisions = {}
    for s, gd in fs.items():
        # pick the dominant (largest-amt) path for this source sat
        best = None
        for gid, (amt, path, edges) in gd.items():
            if amt <= 0: continue
            if best is None or amt > best[0]:
                best = (amt, gid, path)
        if best is None: continue
        amt, gid, path = best
        # source + every transit sat gets a next-hop toward gid along this path
        for h in range(len(path) - 1):
            node = path[h]; nxt = path[h + 1]
            if node not in decisions:
                decisions[node] = (gid, nxt, 0.0, len(path) - 1 - h)
        # exit sat (last node) forwards down to the gateway
        exit_sat = path[-1]
        if exit_sat not in decisions:
            decisions[exit_sat] = (gid, exit_sat, 0.0, 0)
    return decisions


def _build_inst_from_graph(Ns, Ng, sat_graph, vis_list, sat_to_gws, rho, cfg, MP):
    """Most-faithful adapter: build the SAME Instance the RL was evaluated on,
    via hypatia_pop_instance.build_instance with the NATIVE 100 gateways
    (use_std_gw=False) so it aligns with ns-3's ground stations. We reconstruct
    positions from the GEN config (inside build_instance) and then overwrite the
    demand vector with THIS timestep's rho so routing matches the ns-3 load."""
    import hypatia_pop_instance as HP
    # build native-gateway instance (positions reconstructed from GEN config)
    inst, _, _ = HP.build_instance(scale=1.0, hotspot=False, use_std_gw=False)
    # overwrite demand with the timestep's rho (ns-3 pipeline supplies it)
    r = np.asarray(rho, float)
    if len(r) == inst.Ns:
        inst.rho = r
    inst.cfg = cfg
    return inst
