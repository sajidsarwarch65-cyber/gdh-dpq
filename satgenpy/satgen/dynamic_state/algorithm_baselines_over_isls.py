# =============================================================
# algorithm_baselines_over_isls.py
# Implements 5 baseline routing schemes for comparison with GDH-DPQ:
#   1. direct_drop   - nearest visible gateway, drop if full
#   2. dsp           - shortest propagation delay path
#   3. ca_dsp        - capacity-aware shortest path
#   4. dbpr          - queue+distance balanced routing
#   5. gpsr          - geographic greedy routing
# All use improved Hypatia models:
#   - Population-based traffic (loaded from rho_starlink_s1.npy)
#   - Distance-dependent ISL capacity
#   - Elevation-dependent GSL cost
# =============================================================

import math
import heapq
import numpy as np
from collections import defaultdict

# ── Shared parameters (match fstate_gdh_dpq.py) ─────────────
CGW_DEFAULT  = 40.0
CISL_UNIFORM = 8.0
D_TX_GSL     = 0.0005
D_PROC_GSL   = 0.0002
D_TX_ISL     = 0.0002
D_PROC_ISL   = 0.0001
C_LIGHT      = 299792458.0
QUEUE_SCALE  = 0.002
QUEUE_MAX    = 0.05
CADSP_GW_PEN = 1.0
CADSP_ISL_PEN= 0.5
DBPR_ALPHA_Q = 1.0
DBPR_BETA_D  = 0.25
DBPR_ITERS   = 5
RHO_FILE     = '/home/msarwa02/rho_starlink_s1.npy'

def _load_rho(Ns):
    import os
    if os.path.exists(RHO_FILE):
        rho = np.load(RHO_FILE)
        if len(rho) == Ns:
            return rho
    rng = np.random.default_rng(7)
    s=0.9; mu=math.log(1.0)-0.5*s*s
    rho_raw = np.clip(rng.lognormal(mu,s,size=Ns),0.1,8.0)
    target = 0.85*CGW_DEFAULT/(Ns/100.0)
    return rho_raw*(target/rho_raw.mean())

def _isl_capacity(dist_m):
    D_MIN=500000.0; C_MAX=50.0
    return max(C_MAX*(D_MIN/max(dist_m,D_MIN))**2, 1.0)

def _gsl_elev_factor(dist_m, sat_alt_m=550000.0):
    R_e=6371000.0; h=sat_alt_m; d=dist_m
    cos_phi=(R_e**2+(R_e+h)**2-d**2)/(2*R_e*(R_e+h))
    cos_phi=max(-1.0,min(1.0,cos_phi))
    phi=math.acos(cos_phi)
    elev=math.atan2(math.cos(phi)*(R_e+h)-R_e,
                    math.sin(phi)*(R_e+h))
    gain=math.sin(elev)**2
    return max(gain, 0.01)

def _queue_delay(load, cap):
    leff=min(float(load), 0.999*float(cap))
    return min(QUEUE_SCALE/max(float(cap)-leff,1e-12), QUEUE_MAX)

def _build_env(sat_graph, vis_list, Ns, Ng):
    """Build shared environment structures."""
    # ISL capacities
    isl_cap = {}
    for u,v,data in sat_graph.edges(data=True):
        e=(min(u,v),max(u,v))
        isl_cap[e] = _isl_capacity(data['weight'])

    # sat->gw visibility with distances
    sat_to_gws = defaultdict(list)
    for gid, vis in enumerate(vis_list):
        for dm, sid in vis:
            sat_to_gws[sid].append((dm, gid))

    # neighbour list
    nbr = {u: list(sat_graph.neighbors(u)) for u in range(Ns)}

    return isl_cap, sat_to_gws, nbr

def _dijkstra_isl(src, sat_to_gws, nbr, sat_graph, wfunc_isl, wfunc_gw):
    """Dijkstra from src satellite to best gateway."""
    INF = float('inf')
    dist = {src: 0.0}
    par  = {src: -1}
    pq   = [(0.0, src)]
    best_gw = -1; best_cost = INF

    while pq:
        du, u = heapq.heappop(pq)
        if du > dist.get(u, INF)+1e-12: continue
        # Check gateways reachable from u
        for dm, gid in sat_to_gws.get(u, []):
            gc = du + wfunc_gw(u, gid, dm)
            if gc < best_cost:
                best_cost = gc; best_gw = gid
                best_gw_exit = u
        # Expand ISL neighbours
        for v in nbr.get(u, []):
            nd = du + wfunc_isl(u, v, sat_graph)
            if nd < dist.get(v, INF):
                dist[v] = nd; par[v] = u
                heapq.heappush(pq, (nd, v))

    if best_gw < 0:
        return None, -1

    # Reconstruct path
    path = [best_gw_exit]
    node = best_gw_exit
    while par.get(node, -1) >= 0:
        node = par[node]
        path.append(node)
    path.reverse()
    return path, best_gw

def _write_fstate(output_dir, t_ns, Ns, Ng,
                  decisions, next_hop_to_gw,
                  num_isls_per_sat, gid_to_sat_gsl_if_idx,
                  sat_neighbor_to_if, prev_fstate, sat_to_gws=None,
                  scheme_name=''):
    """Write fstate file - same format as GDH-DPQ including GS->GS entries."""
    import os, math
    fname = output_dir + "/fstate_" + str(t_ns) + ".txt"
    fstate = {}

    # Build gw->sats lookup
    gw_to_sats = {}
    if sat_to_gws:
        for sid, gw_list in sat_to_gws.items():
            for dm, gid in gw_list:
                if gid not in gw_to_sats:
                    gw_to_sats[gid] = []
                gw_to_sats[gid].append((dm, sid))

    import sys
    with open(fname, "w+") as f:
        # Sat -> GW entries
        for s in range(Ns):
            for gid in range(Ng):
                dst = Ns + gid
                if s in decisions and decisions[s][0] == gid:
                    nh_sat = decisions[s][1]
                    if nh_sat == s:
                        nhd = (dst,
                               num_isls_per_sat[s]+gid_to_sat_gsl_if_idx[gid],
                               0)
                    elif nh_sat >= 0 and (s, nh_sat) in sat_neighbor_to_if:
                        nhd = (nh_sat,
                               sat_neighbor_to_if[(s, nh_sat)],
                               sat_neighbor_to_if[(nh_sat, s)])
                    else:
                        nhd = (-1, -1, -1)
                else:
                    # For direct_drop: drop if not in decisions
                    # For other schemes: use shortest path next hop
                    if scheme_name == 'direct_drop':
                        nhd = (-1, -1, -1)
                    else:
                        nh_sat = next_hop_to_gw.get((s, gid), -1)
                        if nh_sat == s:
                            nhd = (dst,
                                   num_isls_per_sat[s]+gid_to_sat_gsl_if_idx[gid],
                                   0)
                        elif nh_sat >= 0 and (s, nh_sat) in sat_neighbor_to_if:
                            nhd = (nh_sat,
                                   sat_neighbor_to_if[(s, nh_sat)],
                                   sat_neighbor_to_if[(nh_sat, s)])
                        else:
                            nhd = (-1, -1, -1)
                f.write("%d,%d,%d,%d,%d\n" % (s, dst,
                        nhd[0], nhd[1], nhd[2]))
                fstate[(s, dst)] = nhd

        # GS -> GS entries (required by ns-3)
        import sys
        dist_s2g = {}
        for gid in range(Ng):
            dst = Ns + gid
            for dm, sid in gw_to_sats.get(gid, []):
                dist_s2g[(sid, dst)] = dm

        for sg in range(Ng):
            for dg in range(Ng):
                if sg == dg: continue
                sgs = Ns + sg; dgs = Ns + dg
                bc = math.inf; bs = -1
                for dm, sid in gw_to_sats.get(sg, []):
                    if scheme_name == 'direct_drop':
                        # Only use sat if it has direct view of dst GW
                        if fstate.get((sid, dgs), (-1,-1,-1))[0] == dgs:
                            if dm < bc:
                                bc = dm; bs = sid
                    else:
                        dd = dist_s2g.get((sid, dgs), math.inf)
                        if dm + dd < bc:
                            bc = dm + dd; bs = sid
                if bs < 0 and gw_to_sats.get(sg, []):
                    bs = min(gw_to_sats[sg], key=lambda x: x[0])[1]
                if bs >= 0:
                    nhd = (bs, 0,
                           num_isls_per_sat[bs]+gid_to_sat_gsl_if_idx[sg])
                else:
                    nhd = (-1, -1, -1)
                f.write("%d,%d,%d,%d,%d\n" % (sgs, dgs,
                        nhd[0], nhd[1], nhd[2]))
                fstate[(sgs, dgs)] = nhd
                if sg==0 and dg==1:
                    import sys
        import sys
        gs_count = sum(1 for k in fstate if k[0]>=1584)

    # Debug: check file line count
    import sys
    with open(fname) as _f:
        lines = _f.readlines()
    gs_lines = sum(1 for l in lines if int(l.split(",")[0])>=1584)
    return fstate

def _build_next_hop_to_gw(Ns, Ng, sat_graph, vis_list, sat_to_gws):
    """Build next-hop table for all sat->gw pairs via Dijkstra."""
    import heapq as hq
    dist_s2g = {}
    for gid in range(Ng):
        dst = Ns + gid
        pq = []
        for dm, sid in vis_list[gid]:
            dist_s2g[(sid, dst)] = dm
            hq.heappush(pq, (dm, sid))
        while pq:
            d, u = hq.heappop(pq)
            if d > dist_s2g.get((u,dst), float('inf'))+1e-9: continue
            for nb in sat_graph.neighbors(u):
                nd = d + sat_graph.edges[u,nb]['weight']
                if nd < dist_s2g.get((nb,dst), float('inf')):
                    dist_s2g[(nb,dst)] = nd
                    hq.heappush(pq, (nd, nb))

    next_hop = {}
    for s in range(Ns):
        for gid in range(Ng):
            dst = Ns + gid
            my_dist = dist_s2g.get((s,dst), float('inf'))
            if math.isinf(my_dist):
                next_hop[(s,gid)] = -1; continue
            vis = vis_list[gid]
            if any(sid==s for _,sid in vis):
                next_hop[(s,gid)] = s; continue
            best_nb=-1; best_d=float('inf')
            for nb in sat_graph.neighbors(s):
                nd = dist_s2g.get((nb,dst), float('inf'))
                if nd < best_d:
                    best_d=nd; best_nb=nb
            next_hop[(s,gid)] = best_nb
    return next_hop

# ============================================================
# SCHEME 1: Direct Drop
# ============================================================
def run_direct_drop(Ns, Ng, sat_graph, vis_list, sat_to_gws,
                    isl_cap, nbr, rho, Cg):
    decisions = {}
    gw_load = np.zeros(Ng)

    for s in range(Ns):
        # Find nearest visible gateway
        best_gw=-1; best_d=float('inf')
        for dm, gid in sat_to_gws.get(s, []):
            ef = _gsl_elev_factor(dm)
            cost = dm/C_LIGHT + D_TX_GSL/ef + D_PROC_GSL
            if cost < best_d:
                best_d=cost; best_gw=gid; best_dm=dm
        if best_gw < 0: continue
        if gw_load[best_gw] + rho[s] > Cg[best_gw]: continue
        decisions[s] = (best_gw, s, best_dm)  # (gw, next_hop=self, dist)
        gw_load[best_gw] += rho[s]

    return decisions, gw_load

# ============================================================
# SCHEME 2: DSP (Shortest Propagation Delay)
# ============================================================
def run_dsp(Ns, Ng, sat_graph, vis_list, sat_to_gws,
            isl_cap, nbr, rho, Cg):
    decisions = {}
    gw_load = np.zeros(Ng)
    isl_load = defaultdict(float)

    def wf_isl(u, v, g):
        return isl_prop_delay(u,v,g) + D_TX_ISL + D_PROC_ISL

    def isl_prop_delay(u,v,g):
        return g.edges[u,v]['weight']/C_LIGHT

    for s in range(Ns):
        # Find shortest delay path to any gateway
        best_path=None; best_gw=-1; best_cost=float('inf')

        dist={s:0.0}; par={s:-1}; pq=[(0.0,s)]
        exit_sat=-1

        while pq:
            du,u=heapq.heappop(pq)
            if du>dist.get(u,float('inf'))+1e-12: continue
            for dm,gid in sat_to_gws.get(u,[]):
                ef=_gsl_elev_factor(dm)
                gc=du+dm/C_LIGHT+D_TX_GSL/ef+D_PROC_GSL
                if gc<best_cost:
                    best_cost=gc; best_gw=gid
                    exit_sat=u; exit_dm=dm
            for v in nbr.get(u,[]):
                nd=du+sat_graph.edges[u,v]['weight']/C_LIGHT+D_TX_ISL+D_PROC_ISL
                if nd<dist.get(v,float('inf')):
                    dist[v]=nd; par[v]=u
                    heapq.heappush(pq,(nd,v))

        if best_gw<0: continue
        if gw_load[best_gw]+rho[s]>Cg[best_gw]: continue

        # Get next hop
        if exit_sat==s:
            nh=s
        else:
            node=exit_sat
            while par.get(node,-1)!=s and par.get(node,-1)>=0:
                node=par[node]
            nh=node if par.get(node,-1)==s else exit_sat

        decisions[s]=(best_gw, nh, exit_dm)
        gw_load[best_gw]+=rho[s]

    return decisions, gw_load

# ============================================================
# SCHEME 3: CA-DSP (Capacity-Aware Shortest Path)
# ============================================================
def run_ca_dsp(Ns, Ng, sat_graph, vis_list, sat_to_gws,
               isl_cap, nbr, rho, Cg):
    decisions = {}
    gw_load = np.zeros(Ng)
    isl_load = defaultdict(float)

    order = np.argsort(-rho)

    for s in order:
        best_path=None; best_gw=-1; best_cost=float('inf')
        exit_sat_best=-1; exit_dm_best=0

        dist={s:0.0}; par={s:-1}; pq=[(0.0,s)]

        while pq:
            du,u=heapq.heappop(pq)
            if du>dist.get(u,float('inf'))+1e-12: continue
            for dm,gid in sat_to_gws.get(u,[]):
                if gw_load[gid]+rho[s]>Cg[gid]: continue
                ef=_gsl_elev_factor(dm)
                gw_pen=CADSP_GW_PEN*(gw_load[gid]/max(Cg[gid],1e-9))
                gc=du+dm/C_LIGHT+D_TX_GSL/ef+D_PROC_GSL+gw_pen
                if gc<best_cost:
                    best_cost=gc; best_gw=gid
                    exit_sat_best=u; exit_dm_best=dm
            for v in nbr.get(u,[]):
                e=(min(u,v),max(u,v))
                cap=isl_cap.get(e,CISL_UNIFORM)
                if isl_load[e]+rho[s]>cap: continue
                isl_pen=CADSP_ISL_PEN*(isl_load[e]/max(cap,1e-9))
                nd=du+sat_graph.edges[u,v]['weight']/C_LIGHT+D_TX_ISL+D_PROC_ISL+isl_pen
                if nd<dist.get(v,float('inf')):
                    dist[v]=nd; par[v]=u
                    heapq.heappush(pq,(nd,v))

        if best_gw<0: continue

        # Next hop
        if exit_sat_best==s:
            nh=s
        else:
            node=exit_sat_best
            while par.get(node,-1)!=s and par.get(node,-1)>=0:
                node=par[node]
            nh=node if par.get(node,-1)==s else exit_sat_best

        decisions[s]=(best_gw, nh, exit_dm_best)
        gw_load[best_gw]+=rho[s]
        # Update ISL loads along path
        node=exit_sat_best
        while par.get(node,-1)>=0:
            prev=par[node]
            e=(min(prev,node),max(prev,node))
            isl_load[e]+=rho[s]
            node=prev

    return decisions, gw_load

# ============================================================
# SCHEME 4: DBPR (Queue+Distance Balanced)
# ============================================================
def run_dbpr(Ns, Ng, sat_graph, vis_list, sat_to_gws,
             isl_cap, nbr, rho, Cg):
    decisions = {}
    gw_load = np.zeros(Ng)
    sq = np.zeros(Ns)

    for _ in range(DBPR_ITERS):
        decisions={}; gw_load=np.zeros(Ng)
        isl_load=defaultdict(float)

        for s in range(Ns):
            # Direct gateway
            if sat_to_gws.get(s,[]):
                best_gw=-1; best_cost=float('inf')
                best_dm=0
                for dm,gid in sat_to_gws[s]:
                    cost=DBPR_ALPHA_Q*gw_load[gid]+DBPR_BETA_D*dm/1000
                    if cost<best_cost:
                        best_cost=cost; best_gw=gid; best_dm=dm
                if best_gw>=0:
                    decisions[s]=(best_gw, s, best_dm)
                    gw_load[best_gw]+=rho[s]
                    continue

            # Multi-hop path
            dist={s:0.0}; par={s:-1}; pq=[(0.0,s)]
            best_gw=-1; best_cost=float('inf'); exit_sat=-1; exit_dm=0

            while pq:
                du,u=heapq.heappop(pq)
                if du>dist.get(u,float('inf'))+1e-12: continue
                for dm,gid in sat_to_gws.get(u,[]):
                    gc=du+DBPR_ALPHA_Q*gw_load[gid]+DBPR_BETA_D*dm/1000
                    if gc<best_cost:
                        best_cost=gc; best_gw=gid
                        exit_sat=u; exit_dm=dm
                for v in nbr.get(u,[]):
                    nd=du+(DBPR_ALPHA_Q*(sq[v]+isl_load[(min(u,v),max(u,v))])
                           +DBPR_BETA_D*sat_graph.edges[u,v]['weight']/1000)
                    if nd<dist.get(v,float('inf')):
                        dist[v]=nd; par[v]=u
                        heapq.heappush(pq,(nd,v))

            if best_gw<0: continue
            if exit_sat==s: nh=s
            else:
                node=exit_sat
                while par.get(node,-1)!=s and par.get(node,-1)>=0:
                    node=par[node]
                nh=node if par.get(node,-1)==s else exit_sat

            decisions[s]=(best_gw, nh, exit_dm)
            gw_load[best_gw]+=rho[s]
            node=exit_sat
            while par.get(node,-1)>=0:
                prev=par[node]
                isl_load[(min(prev,node),max(prev,node))]+=rho[s]
                node=prev

        sq[:]=0
        for s,(gw,nh,dm) in decisions.items():
            if nh!=s: sq[nh]+=rho[s]

    return decisions, gw_load

# ============================================================
# SCHEME 5: GPSR (Geographic Greedy - using ISL hop distances)
# Simplified: greedy routing toward nearest visible gateway
# using ISL graph hop count as distance metric
# ============================================================
def run_gpsr(Ns, Ng, sat_graph, vis_list, sat_to_gws,
             isl_cap, nbr, rho, Cg, sat_positions_km=None):
    decisions = {}
    gw_load = np.zeros(Ng)

    for s in range(Ns):
        # Direct gateway connection
        if sat_to_gws.get(s, []):
            best_gw=-1; best_cost=float('inf'); best_dm=0
            for dm,gid in sat_to_gws[s]:
                if gw_load[gid]+rho[s]>Cg[gid]: continue
                ef=_gsl_elev_factor(dm)
                cost=dm/C_LIGHT+D_TX_GSL/ef+D_PROC_GSL
                if cost<best_cost:
                    best_cost=cost; best_gw=gid; best_dm=dm
            if best_gw>=0:
                decisions[s]=(best_gw, s, best_dm)
                gw_load[best_gw]+=rho[s]
            continue

        # Multi-hop: BFS to find nearest exit satellite
        # then greedy forward along shortest ISL path
        from collections import deque
        visited={s}; queue=deque([(s,s,0)])  # (curr, first_hop, hops)
        best_gw=-1; best_nh=s; best_dm=0; best_cost=float('inf')

        while queue:
            cur, first_hop, hops = queue.popleft()
            if hops > 10: continue  # max hop limit
            # Check gateways reachable from cur
            for dm,gid in sat_to_gws.get(cur,[]):
                if gw_load[gid]+rho[s]>Cg[gid]: continue
                ef=_gsl_elev_factor(dm)
                cost=hops*(D_TX_ISL+D_PROC_ISL)+dm/C_LIGHT+D_TX_GSL/ef+D_PROC_GSL
                if cost<best_cost:
                    best_cost=cost; best_gw=gid
                    best_nh=first_hop; best_dm=dm
            # Expand neighbours
            for v in nbr.get(cur,[]):
                if v not in visited:
                    visited.add(v)
                    nh = v if cur==s else first_hop
                    queue.append((v, nh, hops+1))

        if best_gw<0: continue
        decisions[s]=(best_gw, best_nh, best_dm)
        gw_load[best_gw]+=rho[s]

    return decisions, gw_load

# ============================================================
# MAIN ENTRY POINT
# ============================================================
def algorithm_baselines_over_isls(
        scheme_name,
        output_dynamic_state_dir,
        time_since_epoch_ns,
        satellites,
        ground_stations,
        sat_net_graph_only_satellites_with_isls,
        ground_station_satellites_in_range,
        num_isls_per_sat,
        sat_neighbor_to_if,
        list_gsl_interfaces_info,
        prev_output,
        enable_verbose_logs,
        sat_positions_km=None):

    Ns = len(satellites)
    Ng = len(ground_stations)
    sat_graph = sat_net_graph_only_satellites_with_isls
    vis_list  = ground_station_satellites_in_range

    rho = _load_rho(Ns)
    Cg  = np.full(Ng, CGW_DEFAULT)

    isl_cap, sat_to_gws, nbr = _build_env(sat_graph, vis_list, Ns, Ng)

    if scheme_name == "direct_drop":
        decisions, gw_load = run_direct_drop(
            Ns, Ng, sat_graph, vis_list, sat_to_gws,
            isl_cap, nbr, rho, Cg)
    elif scheme_name == "dsp":
        decisions, gw_load = run_dsp(
            Ns, Ng, sat_graph, vis_list, sat_to_gws,
            isl_cap, nbr, rho, Cg)
    elif scheme_name == "ca_dsp":
        decisions, gw_load = run_ca_dsp(
            Ns, Ng, sat_graph, vis_list, sat_to_gws,
            isl_cap, nbr, rho, Cg)
    elif scheme_name == "dbpr":
        decisions, gw_load = run_dbpr(
            Ns, Ng, sat_graph, vis_list, sat_to_gws,
            isl_cap, nbr, rho, Cg)
    elif scheme_name == "gpsr":
        decisions, gw_load = run_gpsr(
            Ns, Ng, sat_graph, vis_list, sat_to_gws,
            isl_cap, nbr, rho, Cg, sat_positions_km)
    else:
        raise ValueError(f"Unknown scheme: {scheme_name}")

    # Build full next-hop table
    next_hop_to_gw = _build_next_hop_to_gw(
        Ns, Ng, sat_graph, vis_list, sat_to_gws)

    gid_to_sat_gsl_if_idx = [0]*Ng
    prev_fstate = prev_output.get("fstate") if prev_output else None

    fstate = _write_fstate(
        output_dynamic_state_dir, time_since_epoch_ns,
        Ns, Ng, decisions, next_hop_to_gw,
        num_isls_per_sat, gid_to_sat_gsl_if_idx,
        sat_neighbor_to_if, prev_fstate, sat_to_gws,
        scheme_name)

    # KPI computation
    served = sum(rho[s] for s in decisions)
    offered = rho.sum()
    plr = 1.0 - served/max(offered,1e-9)
    max_gw_load = gw_load.max()

    if enable_verbose_logs:
        print(f"  > {scheme_name.upper()}: "
              f"routed={len(decisions)}/{Ns} "
              f"PLR={plr:.4f} "
              f"served={served:.1f}/{offered:.1f} "
              f"maxGW={max_gw_load:.2f}/{CGW_DEFAULT}")

    # Write bandwidth state
    bw_fname = (output_dynamic_state_dir
                + "/gsl_if_bandwidth_"
                + str(time_since_epoch_ns) + ".txt")
    if time_since_epoch_ns == 0:
        with open(bw_fname, "w+") as f:
            for node_id in range(Ns):
                f.write("%d,%d,%f\n" % (
                    node_id, num_isls_per_sat[node_id],
                    list_gsl_interfaces_info[node_id]["aggregate_max_bandwidth"]))
            for node_id in range(Ns, Ns+Ng):
                f.write("%d,%d,%f\n" % (
                    node_id, 0,
                    list_gsl_interfaces_info[node_id]["aggregate_max_bandwidth"]))

    return {
        "fstate": fstate,
        "scheme_kpis": {
            "plr": plr,
            "served": served,
            "offered": offered,
            "max_gw_load": max_gw_load,
            "gw_load": gw_load.tolist(),
        }
    }
