"""
fstate_writer.py -- turn the neural policy's solution into Hypatia forwarding state.

The policy returns:   solution = {sat_id: (gateway_id, path_list)}
Hypatia needs:        fstate[(s, Ns+gid)] = (next_hop_node, my_if, next_if)
                      for EVERY (satellite, gateway) pair, plus gateway->gateway.

Rules (mirroring calculate_fstate_gdh_dpq):
  * For a satellite's ASSIGNED gateway  -> use the policy's own first hop.
  * For every OTHER gateway             -> min-propagation-distance tree
                                           (so the table is complete).
  * Dropped satellites                  -> (-1,-1,-1) to their gateways.
"""
import heapq
import math


def _build_distance_trees(sat_graph, vis_list, Ns, Ng):
    """dist_s2g[(s, Ns+gid)] = shortest ISL distance from s to gateway gid.
    Dijkstra outward from each gateway's visible satellites."""
    dist_s2g = {}
    for gid in range(Ng):
        dst = Ns + gid
        pq = []
        for dm, sid in vis_list[gid]:
            sid = int(sid)
            if dm < dist_s2g.get((sid, dst), math.inf):
                dist_s2g[(sid, dst)] = float(dm)
                heapq.heappush(pq, (float(dm), sid))
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist_s2g.get((u, dst), math.inf) + 1e-9:
                continue
            for nb in sat_graph.neighbors(u):
                nd = d + float(sat_graph.edges[u, nb]["weight"])
                if nd < dist_s2g.get((nb, dst), math.inf):
                    dist_s2g[(nb, dst)] = nd
                    heapq.heappush(pq, (nd, nb))
    return dist_s2g


def _next_hop_tree(sat_graph, vis_list, dist_s2g, Ns, Ng):
    """next_hop_to_gw[(s,gid)] = neighbour minimising distance to gateway,
    or s itself if s is directly visible to gid, or -1 if unreachable."""
    nh = {}
    vis_sets = [set(int(sid) for _dm, sid in vis_list[gid]) for gid in range(Ng)]
    for s in range(Ns):
        for gid in range(Ng):
            dst = Ns + gid
            if s in vis_sets[gid]:
                nh[(s, gid)] = s                    # exit satellite
                continue
            if math.isinf(dist_s2g.get((s, dst), math.inf)):
                nh[(s, gid)] = -1
                continue
            best_nb, best_d = -1, math.inf
            for nb in sat_graph.neighbors(s):
                d = dist_s2g.get((nb, dst), math.inf)
                if d < best_d:
                    best_d, best_nb = d, nb
            nh[(s, gid)] = best_nb
    return nh


def write_fstate_from_solution(
        output_dynamic_state_dir, time_since_epoch_ns,
        Ns, Ng, sat_graph, vis_list,
        num_isls_per_sat, gid_to_sat_gsl_if_idx, sat_neighbor_to_if,
        solution, prev_fstate=None):
    """
    solution : {sat_id: (gateway_id, path)} from policy.rollout(greedy=True)
    Returns the fstate dict and writes the fstate_<t>.txt file.
    """
    dist_s2g = _build_distance_trees(sat_graph, vis_list, Ns, Ng)
    nh_tree = _next_hop_tree(sat_graph, vis_list, dist_s2g, Ns, Ng)

    # policy's own next hop, only for the gateway it assigned
    assigned = {}          # sat -> (gid, next_hop_sat)
    for s, (gid, path) in solution.items():
        s = int(s); gid = int(gid)
        # path[0] == s ; exit satellite is path[-1]
        nh_sat = int(path[1]) if len(path) >= 2 else s
        assigned[s] = (gid, nh_sat)

    fstate = {}
    fname = f"{output_dynamic_state_dir}/fstate_{time_since_epoch_ns}.txt"
    with open(fname, "w+") as f:
        # --- satellite -> gateway ---
        for s in range(Ns):
            for gid in range(Ng):
                dst = Ns + gid

                if s in assigned and assigned[s][0] == gid:
                    nh_sat = assigned[s][1]            # policy decision
                elif s in solution:
                    # routed, but to a DIFFERENT gateway: complete the table
                    nh_sat = nh_tree.get((s, gid), -1)
                else:
                    # DROPPED satellite: no forwarding entry to any gateway
                    nh_sat = -1

                if nh_sat == s:
                    # s is the exit satellite -> hand off on its GSL interface
                    nhd = (dst,
                           num_isls_per_sat[s] + gid_to_sat_gsl_if_idx[gid],
                           0)
                elif nh_sat >= 0 and (s, nh_sat) in sat_neighbor_to_if:
                    nhd = (nh_sat,
                           sat_neighbor_to_if[(s, nh_sat)],
                           sat_neighbor_to_if[(nh_sat, s)])
                else:
                    nhd = (-1, -1, -1)

                if (not prev_fstate) or prev_fstate.get((s, dst)) != nhd:
                    f.write("%d,%d,%d,%d,%d\n" % (s, dst, nhd[0], nhd[1], nhd[2]))
                fstate[(s, dst)] = nhd

        # --- gateway -> gateway (relay via closest visible sat) ---
        for sg in range(Ng):
            for dg in range(Ng):
                if sg == dg:
                    continue
                sgs, dgs = Ns + sg, Ns + dg
                bc, bs = math.inf, -1
                for d_src, a in vis_list[sg]:
                    a = int(a)
                    dd = dist_s2g.get((a, dgs), math.inf)
                    if d_src + dd < bc:
                        bc, bs = d_src + dd, a
                if bs < 0 and vis_list[sg]:
                    bs = int(min(vis_list[sg], key=lambda x: x[0])[1])
                if bs >= 0:
                    nhd = (bs, 0,
                           num_isls_per_sat[bs] + gid_to_sat_gsl_if_idx[sg])
                else:
                    nhd = (-1, -1, -1)
                if (not prev_fstate) or prev_fstate.get((sgs, dgs)) != nhd:
                    f.write("%d,%d,%d,%d,%d\n" % (sgs, dgs, nhd[0], nhd[1], nhd[2]))
                fstate[(sgs, dgs)] = nhd

    return fstate
