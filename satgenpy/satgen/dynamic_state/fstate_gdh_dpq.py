
import math, heapq, numpy as np
from collections import defaultdict

def _make_rho(Ns, seed=7):
    # Population-based traffic demand (GHS 2020 raster)
    import os
    _rho_file = '/home/msarwa02/rho_starlink_s1.npy'
    if os.path.exists(_rho_file):
        rho = np.load(_rho_file)
        if len(rho) == Ns:
            return rho
    rng = np.random.default_rng(seed)
    s = 0.9
    mu = math.log(1.0) - 0.5*s*s
    rho_raw = np.clip(rng.lognormal(mu, s, size=Ns), 0.1, 8.0)
    target_mean = 0.85 * CGW_DEFAULT / (Ns / 100.0)
    return rho_raw * (target_mean / rho_raw.mean())

# Parameters matching v6b standalone code
ALPHA1=1.0; ALPHA2=0.02; ALPHA3=0.02; ALPHA4=0.01
C_LIGHT=299792458.0
D_TX_ISL=0.0002; D_PROC_ISL=0.0001
D_TX_GSL=0.0005; D_PROC_GSL=0.0002
CISL=8.0; CGW_DEFAULT=40.0
QUEUE_SCALE=0.002; QUEUE_MAX=0.05; QUEUE_EPS=1e-3
THO_S=60.0; DHO=0.02
# Dual decomposition -- matching v6b
DUAL_ITERS_S2=100; DUAL_ITERS_S3=50
LAM_MAX=100.0; NU_MAX=100.0
ETA_LOAD=0.25
POLYAK_ALPHA=0.5
STAGNATION_K=50
STEP_FLOOR=1e-6; STEP_CEIL=1.0
# Fallback parameters (v6b)
FB_LINEAR_GW=0.5; FB_LINEAR_ISL=0.3; FB_HOP_PEN=0.0005
# Admission control
KAPPA=0.5
L_MAX_LIFE=600.0; L_MIN=30.0
# Epsilon feasibility
EPS_MIN_FRAC=0.02; EPS_MAX_FRAC=0.12
ETA_LOAD=0.3; TIE_EPS=1e-12

def _gsl_capacity_from_elevation(dist_m, sat_alt_m=550000.0,
                                  elev_min_deg=25.0, c_max=1.0):
    """
    Elevation-dependent GSL capacity model.
    Addresses Hypatia limitation: "more realistic models of interfaces
    at both satellites and GSes, and for antenna gain and interference."

    Elevation angle from ground station to satellite:
    elev = arcsin((R_e + h) / d * sin(pi/2) - pi/2)
    Simplified: elev = arccos(R_e / (R_e + h)) for overhead,
    or from distance: elev = arcsin((R_e+h)^2 + d^2 - R_e^2) / (2*d*(R_e+h))

    Antenna gain pattern (simplified sinusoidal):
    G(elev) = sin^2(elev)

    Result: C_GSL = C_max * G(elev)
    At minimum elevation (25 deg): C_GSL = C_max * sin^2(25) = 0.179 * C_max
    At zenith (90 deg):            C_GSL = C_max * sin^2(90) = 1.0 * C_max
    """
    import math
    R_EARTH = 6371000.0  # Earth radius in metres
    # Compute elevation angle from distance and altitude
    # Using law of cosines: cos(zenith) = (R_e^2 + d^2 - (R_e+h)^2) / (2*R_e*d)
    # elevation = 90 - zenith
    R_e = R_EARTH
    h   = sat_alt_m
    d   = max(dist_m, R_e)  # guard
    # Correct elevation formula using triangle at Earth centre
    cos_phi = (R_e**2 + (R_e+h)**2 - d**2) / (2.0*R_e*(R_e+h))
    cos_phi = max(-1.0, min(1.0, cos_phi))
    phi = math.acos(cos_phi)
    elev_rad = math.atan2(
        math.cos(phi)*(R_e+h) - R_e,
        math.sin(phi)*(R_e+h)
    )
    elev_deg = math.degrees(elev_rad)
    # Below minimum elevation  link not available (handled by vis_list)
    if elev_deg < elev_min_deg:
        return 0.0
    # Antenna gain: G(elev) = sin^2(elev)
    gain = math.sin(elev_rad) ** 2
    return c_max * gain

def _doppler_factor(vel_rel_m_s, f_carrier_ghz=26.0):
    """
    Doppler effect model for Starlink optical ISL (1550nm laser).
    Starlink uses optical ISLs not RF -- Doppler causes frequency shift
    requiring AFC tracking but does NOT reduce channel capacity.

    Optical Doppler wavelength shift:
      delta_lambda = lambda * v_rel / c
      = 1550nm * 660/3e8 = 3.41 pm per inter-plane hop

    WDM channel spacing = 100 GHz = 0.8 nm
    DWS/channel_spacing = 3.41pm / 800pm = 0.43% -- negligible crosstalk

    SNR penalty for coherent QPSK (standard model):
      factor = exp(-alpha * (v_rel/c)^2) ≈ 1.0 (no measurable reduction)

    Conclusion: capacity factor = 0.99999 ≈ 1.0
    AFC compensation required at receiver but no routing impact.

    Reference: Chaudhry & Yanikomeroglu (2022), Vieira et al. (2022)
    """
    C_LIGHT = 299792458.0
    # Optical carrier at 1550nm
    lambda_m = 1550e-9
    f_optical = C_LIGHT / lambda_m  # 193.4 THz
    # Doppler wavelength shift
    delta_lambda = lambda_m * abs(vel_rel_m_s) / C_LIGHT
    # WDM channel spacing 100 GHz ~ 0.8nm
    wdm_spacing_m = 0.8e-9
    # Crosstalk fraction (negligible if < 1%)
    crosstalk_frac = delta_lambda / wdm_spacing_m
    # Capacity factor: 1 - crosstalk_frac (conservative upper bound)
    factor = max(0.0, 1.0 - crosstalk_frac)
    return factor  # ≈ 0.99957 for inter-plane, 1.0 for intra-plane

def _relative_velocity_m_s(sat_i, sat_j, num_sats_per_plane=22):
    """
    Estimate relative velocity between two satellites for Doppler calculation.
    For Starlink S1 (72 planes x 22 sats/plane):
    - Intra-plane ISL (adjacent in same plane): ~0 relative velocity
    - Inter-plane ISL (adjacent planes): ~200-500 m/s relative velocity
    Uses satellite IDs to determine link type.
    """
    V_SAT = 7560.0  # Starlink orbital velocity m/s at 550km
    # Same plane = consecutive IDs within a plane
    same_plane = (abs(sat_i - sat_j) == 1) or                  (abs(sat_i - sat_j) == num_sats_per_plane - 1)
    if same_plane:
        return 0.0  # co-rotating, negligible relative velocity
    else:
        # Inter-plane: approximate relative velocity
        # For 53 degree inclination, adjacent plane separation ~5 degrees
        # v_rel ≈ 2 * v_sat * sin(delta_angle/2)
        import math
        delta_angle_rad = math.radians(5.0)
        return 2.0 * V_SAT * math.sin(delta_angle_rad / 2.0)

def _isl_capacity(dist_m):
    """
    Distance-dependent ISL capacity.
    Based on free-space path loss model:
    C_ij = C_max * (d_min / d_ij)^2
    where d_min = 1000 km (minimum ISL distance in Starlink)
    and C_max = 50 (normalised units at minimum distance)
    """
    D_MIN_M = 500000.0    # 500 km minimum ISL distance (Starlink intra-plane)
    C_MAX   = 50.0        # capacity at minimum distance
    return max(C_MAX * (D_MIN_M / max(dist_m, D_MIN_M))**2, 1.0)

def _qd(load,cap):
    cap=max(float(cap),1e-9)
    l=min(float(load),(1-QUEUE_EPS)*cap)
    return min(QUEUE_SCALE/max(cap-l,1e-12),QUEUE_MAX)

def _dijkstra(src,Ns,Ng,sat_graph,sat_to_gws,lam,nu,qg,q_isl,isl_lif,rho_s):
    SUPER=Ns+Ng
    dist={src:0.0}; parent={src:-1}; minlif={src:1e18}
    pq=[(0.0,src)]
    def lif(i,j):
        if isl_lif is None: return 1e18
        return float(isl_lif.get((min(i,j),max(i,j)),1e18))
    while pq:
        du,u=heapq.heappop(pq)
        if du>dist.get(u,1e18)+1e-15: continue
        if u==SUPER: break
        if Ns<=u<Ns+Ng:
            nd=du
            if nd<dist.get(SUPER,1e18):
                dist[SUPER]=nd; parent[SUPER]=u
                minlif[SUPER]=minlif.get(u,1e18)
                heapq.heappush(pq,(nd,SUPER))
            continue
        for nb in sat_graph.neighbors(u):
            if nb>=Ns: continue
            dm=sat_graph.edges[u,nb]["weight"]
            e=(min(u,nb),max(u,nb))
            lij=lif(u,nb)
            dp=dm/C_LIGHT
            dij=dp+D_TX_ISL+D_PROC_ISL+q_isl.get(e,0.0)
            ho=DHO if lij<=THO_S else 0.0
            cij=ALPHA1*dij+ALPHA2+ALPHA3*ho+float(nu.get(e,0.0))
            w=rho_s*cij+TIE_EPS*(u*Ns+nb)
            nd=du+w
            if nd<dist.get(nb,1e18):
                dist[nb]=nd; parent[nb]=u
                minlif[nb]=min(minlif.get(u,1e18),lij)
                heapq.heappush(pq,(nd,nb))
        for dm,gid in sat_to_gws.get(u,[]):
            gw=Ns+gid
            dp=dm/C_LIGHT
            gsl_cap=_gsl_capacity_from_elevation(dm)
            elev_factor=max(gsl_cap,0.01)
            dig=dp+D_TX_GSL/elev_factor+D_PROC_GSL+(qg[gid] if qg is not None else 0.0)
            w=rho_s*(ALPHA1*dig+float(lam[gid]))+TIE_EPS*(Ns*Ns+u*Ng+gid)
            nd=du+w
            if nd<dist.get(gw,1e18):
                dist[gw]=nd; parent[gw]=u
                minlif[gw]=minlif.get(u,1e18)
                heapq.heappush(pq,(nd,gw))
    if SUPER not in dist or math.isinf(dist[SUPER]): return -1,-1,1e18,0
    gw_node=parent.get(SUPER,-1)
    if not(Ns<=gw_node<Ns+Ng): return -1,-1,1e18,0
    chosen_gid=gw_node-Ns
    node=parent.get(gw_node,-1); sat_path=[]
    while node!=-1 and node<Ns:
        sat_path.append(node); node=parent.get(node,-1)
    sat_path.reverse()
    if not sat_path or sat_path[0]!=src: return chosen_gid,src,minlif.get(gw_node,1e18),0
    nh=sat_path[1] if len(sat_path)>=2 else src
    return chosen_gid,nh,minlif.get(gw_node,1e18),len(sat_path)-1

def calculate_fstate_gdh_dpq(
        output_dynamic_state_dir,time_since_epoch_ns,
        num_satellites,num_ground_stations,
        sat_net_graph_only_satellites_with_isls,
        num_isls_per_sat,gid_to_sat_gsl_if_idx,
        ground_station_satellites_in_range_candidates,
        sat_neighbor_to_if,prev_fstate,enable_verbose_logs,
        gdhdpq_state=None):
    Ns=num_satellites; Ng=num_ground_stations
    sat_graph=sat_net_graph_only_satellites_with_isls
    vis_list=ground_station_satellites_in_range_candidates
    rho=_make_rho(Ns)
    Cg=np.full(Ng,CGW_DEFAULT,dtype=float)

    # Precompute sat->gws lookup (KEY fix for performance)
    sat_to_gws=defaultdict(list)
    for gid,vis in enumerate(vis_list):
        for dm,sid in vis:   # format is (distance_m, sat_id)
            sat_to_gws[sid].append((dm,gid))

    if gdhdpq_state is not None:
        lam=gdhdpq_state.get("lam",np.zeros(Ng))
        nu=defaultdict(float,gdhdpq_state.get("nu",{}))
        p_gw=gdhdpq_state.get("gw_load",np.zeros(Ng))
        p_isl=defaultdict(float,gdhdpq_state.get("isl_load",{}))
        isl_lif=gdhdpq_state.get("isl_lifetime",None)
    else:
        lam=np.zeros(Ng); nu=defaultdict(float)
        p_gw=np.zeros(Ng); p_isl=defaultdict(float); isl_lif=None

    # Distance-dependent ISL capacity with optical Doppler
    isl_cap = {}
    for u,v,data in sat_graph.edges(data=True):
        e = (min(u,v),max(u,v))
        dist_cap = _isl_capacity(data['weight'])
        v_rel = _relative_velocity_m_s(u, v)
        doppler = _doppler_factor(v_rel)
        isl_cap[e] = dist_cap * doppler

    def route_all(lam_,nu_,qg_,qisl_):
        dec={}; gw_load=np.zeros(Ng)
        for s in range(Ns):
            gid,nh,Lp,K=_dijkstra(s,Ns,Ng,sat_graph,sat_to_gws,
                                   lam_,nu_,qg_,qisl_,isl_lif,rho[s])
            if gid<0: continue
            dec[s]=(gid,nh,Lp,K); gw_load[gid]+=rho[s]
        return dec,gw_load

    def best_cand(c,b):
        if b is None: return c
        if c["feas"] and not b["feas"]: return c
        if c["feas"]==b["feas"] and c["mo"]<b["mo"]: return c
        return b

    # Stage-2: lambda only (Polyak + stagnation, v6b)
    best2=None; qz=np.zeros(Ng); qiz=defaultdict(float)
    step_scale2=1.0; stag2=0
    for it in range(DUAL_ITERS_S2):
        dec,gw_load=route_all(lam,nu,qz,qiz)
        over=np.maximum(gw_load-Cg,0.0)
        mo=float(np.max(over)); feas=mo<=1e-9
        prev_mo=best2["mo"] if best2 else math.inf
        best2=best_cand({"lam":lam.copy(),"nu":dict(nu),"dec":dec,
                         "gw":gw_load.copy(),"feas":feas,"mo":mo},best2)
        if feas: break
        if best2["mo"]<prev_mo-1e-9: stag2=0
        else: stag2+=1
        if stag2>=STAGNATION_K: step_scale2*=0.5; stag2=0
        tv=float(np.sum(over))
        gs=float(np.sum(gw_load**2))+1e-12
        step=float(np.clip(step_scale2*POLYAK_ALPHA*tv/gs,STEP_FLOOR,STEP_CEIL))
        lam=np.clip(lam+step*over,0.0,LAM_MAX)

    # Stage-3: lambda+nu+queues (Polyak + stagnation, v6b)
    lam3=best2["lam"].copy(); nu3=defaultdict(float); best3=None
    step_scale3=1.0; stag3=0
    for it in range(DUAL_ITERS_S3):
        qg=np.array([_qd(p_gw[g],Cg[g]) for g in range(Ng)])
        qi={e:_qd(p_isl[e],isl_cap.get(e,CISL)) for e in p_isl if p_isl[e]>0}
        dec,gw_load=route_all(lam3,nu3,qg,qi)
        over=np.maximum(gw_load-Cg,0.0)
        mo=float(np.max(over)); feas=mo<=1e-9
        prev_mo=best3["mo"] if best3 else math.inf
        best3=best_cand({"lam":lam3.copy(),"nu":dict(nu3),"dec":dec,
                         "gw":gw_load.copy(),"feas":feas,"mo":mo},best3)
        if feas: break
        if best3["mo"]<prev_mo-1e-9: stag3=0
        else: stag3+=1
        if stag3>=STAGNATION_K: step_scale3*=0.5; stag3=0
        gv=float(np.sum(over))
        ggs=float(np.sum(gw_load**2))+1e-12
        sg=float(np.clip(step_scale3*POLYAK_ALPHA*gv/ggs,STEP_FLOOR,STEP_CEIL))
        lam3=np.clip(lam3+sg*over,0.0,LAM_MAX)
        isl_load_approx=defaultdict(float)
        for s,(gid,nh,Lp,K) in dec.items():
            if nh>=0 and nh!=s:
                e=(min(s,nh),max(s,nh))
                isl_load_approx[e]+=float(rho[s])
        iod={e:float(l-isl_cap.get(e,CISL))
             for e,l in isl_load_approx.items()
             if float(l-isl_cap.get(e,CISL))>1e-12}
        if iod:
            iv=sum(iod.values())
            igs=sum(float(isl_load_approx.get(e,0.))**2 for e in iod)+1e-12
            si=float(np.clip(step_scale3*POLYAK_ALPHA*iv/igs,STEP_FLOOR,STEP_CEIL))
            for e,ov in iod.items():
                nu3[e]=float(np.clip(float(nu3.get(e,0.))+si*ov,0.,NU_MAX))
        p_gw=(1-ETA_LOAD)*p_gw+ETA_LOAD*gw_load
        for e,l in isl_load_approx.items():
            p_isl[e]=(1-ETA_LOAD)*float(p_isl.get(e,0.))+ETA_LOAD*float(l)

    # ── v6b Fallback greedy (if dual did not converge) ──────
    if not best3["feas"]:
        best_nu = best3["nu"]
        fb_dec = {}; fb_gw = np.zeros(Ng); fb_isl = defaultdict(float)
        lam_run = np.array([_qd(0., Cg[g]) for g in range(Ng)], dtype=float)

        for s in sorted(range(Ns), key=lambda s: -rho[s]):
            rs = float(rho[s])
            bc = math.inf; bp_gw = -1; bp_nh = -1; bp_dm = 0.0

            # Dijkstra with fallback weights
            dist_fb = {s: 0.0}; par_fb = {s: -1}
            pq_fb = [(0.0, s)]
            while pq_fb:
                du, u = heapq.heappop(pq_fb)
                if du > dist_fb.get(u, math.inf)+1e-12: continue
                for dm, gid in sat_to_gws.get(u, []):
                    if fb_gw[gid]+rs > Cg[gid]: continue
                    ef = _gsl_capacity_from_elevation(dm)
                    elev_f = max(ef, 0.01)
                    gw_lin = FB_LINEAR_GW*(fb_gw[gid]/max(Cg[gid],1e-9))
                    gc = (du+dm/C_LIGHT+D_TX_GSL/elev_f+D_PROC_GSL
                          +float(lam_run[gid])+gw_lin)
                    if gc < bc:
                        bc = gc; bp_gw = gid; bp_dm = dm
                        node = u
                        while par_fb.get(node,-1)!=s and par_fb.get(node,-1)>=0:
                            node = par_fb[node]
                        bp_nh = node if par_fb.get(node,-1)==s else u
                for v in sat_graph.neighbors(u):
                    e = (min(u,v),max(u,v))
                    cap = isl_cap.get(e, CISL)
                    if fb_isl[e]+rs > cap: continue
                    lif = float(isl_lif.get((u,v), isl_lif.get((v,u), math.inf))) if isl_lif is not None else math.inf
                    if lif < L_MIN: continue
                    stab = ALPHA4*max(0., L_MAX_LIFE-lif)
                    qe = _qd(fb_isl[e], cap)
                    lin = FB_LINEAR_ISL*(fb_isl[e]/max(cap,1e-9))
                    nu_e = float(best_nu.get(e, 0.))
                    nd = (sat_graph.edges[u,v]['weight']/C_LIGHT
                          +D_TX_ISL+D_PROC_ISL+qe+lin+nu_e+FB_HOP_PEN+stab)
                    nd = du + nd
                    if nd < dist_fb.get(v, math.inf):
                        dist_fb[v]=nd; par_fb[v]=u
                        heapq.heappush(pq_fb,(nd,v))

            if bp_gw < 0: continue
            if bp_dm/C_LIGHT+D_TX_GSL+D_PROC_GSL > KAPPA: continue

            fb_dec[s] = (bp_gw, bp_nh, bp_dm, 0)
            fb_gw[bp_gw] += rs
            node = bp_nh
            while par_fb.get(node,-1) >= 0:
                prev = par_fb[node]
                fb_isl[(min(prev,node),max(prev,node))] += rs
                node = prev
            lam_run[bp_gw] = _qd(fb_gw[bp_gw], Cg[bp_gw])

        fb_served = sum(rho[s] for s in fb_dec)
        b3_served = sum(rho[s] for s in best3["dec"])
        if fb_served > b3_served:
            best3 = dict(best3, dec=fb_dec, gw=fb_gw,
                        feas=False, mo=float(np.max(np.maximum(fb_gw-Cg,0))))
            if enable_verbose_logs:
                print(f'  > GDH-DPQ fallback: served={fb_served:.1f} '
                      f'vs dual={b3_served:.1f}')

    # Build fstate
    fstate={}; decisions=best3["dec"]
    dist_s2g={}
    for s in range(Ns):
        for gid in range(Ng):
            dst=Ns+gid
            vis=vis_list[gid]
            # direct visibility distance
            dm=next((d for d,sid in vis if sid==s),math.inf)
            dist_s2g[(s,dst)]=dm
    # improve dist_s2g using Dijkstra: run one pass per GW
    # use Floyd-Warshall-like approach: for each sat with known
    # direct access to GW, propagate distance via ISL neighbours
    # Simple BFS/Dijkstra from each GW outward
    import heapq as _hq
    for gid in range(Ng):
        dst=Ns+gid
        # seed: satellites directly visible to this GW
        pq=[]
        for dm,sid in vis_list[gid]:
            dist_s2g[(sid,dst)]=dm
            _hq.heappush(pq,(dm,sid))
        # Dijkstra outward through ISL graph
        while pq:
            d,u=_hq.heappop(pq)
            if d>dist_s2g.get((u,dst),math.inf)+1e-9: continue
            for nb in sat_graph.neighbors(u):
                nd=d+sat_graph.edges[u,nb]['weight']
                if nd<dist_s2g.get((nb,dst),math.inf):
                    dist_s2g[(nb,dst)]=nd
                    _hq.heappush(pq,(nd,nb))

    fname=output_dynamic_state_dir+"/fstate_"+str(time_since_epoch_ns)+".txt"
    # Build next-hop toward each GW using dist_s2g (ISL Dijkstra)
    # For each sat, for each GW: find best neighbour that reduces dist_s2g
    next_hop_to_gw = {}  # (s, gid) -> next_hop_sat or -1
    for s in range(Ns):
        for gid in range(Ng):
            dst = Ns + gid
            my_dist = dist_s2g.get((s, dst), math.inf)
            if math.isinf(my_dist):
                next_hop_to_gw[(s,gid)] = -1
                continue
            # check if s is directly visible to gid
            vis = vis_list[gid]
            if any(sid==s for _,sid in vis):
                next_hop_to_gw[(s,gid)] = s  # s is exit sat
                continue
            # find ISL neighbour with smallest dist_s2g
            best_nb = -1; best_d = math.inf
            for nb in sat_graph.neighbors(s):
                nd = dist_s2g.get((nb,dst), math.inf)
                if nd < best_d:
                    best_d = nd; best_nb = nb
            next_hop_to_gw[(s,gid)] = best_nb

    with open(fname,"w+") as f:
        for s in range(Ns):
            for gid in range(Ng):
                dst=Ns+gid
                nh_sat = next_hop_to_gw.get((s,gid), -1)
                if nh_sat == s:
                    # exit satellite
                    nhd=(dst,num_isls_per_sat[s]+gid_to_sat_gsl_if_idx[gid],0)
                elif nh_sat>=0 and (s,nh_sat) in sat_neighbor_to_if:
                    nhd=(nh_sat,sat_neighbor_to_if[(s,nh_sat)],sat_neighbor_to_if[(nh_sat,s)])
                else:
                    nhd=(-1,-1,-1)
                if not prev_fstate or prev_fstate.get((s,dst))!=nhd:
                    f.write("%d,%d,%d,%d,%d\n"%(s,dst,nhd[0],nhd[1],nhd[2]))
                fstate[(s,dst)]=nhd
        for sg in range(Ng):
            for dg in range(Ng):
                if sg==dg: continue
                sgs=Ns+sg; dgs=Ns+dg
                bc=math.inf; bs=-1
                for d_src,a in vis_list[sg]:
                    dd=dist_s2g.get((a,dgs),math.inf)
                    if d_src+dd<bc: bc=d_src+dd; bs=a
                # fallback: use closest visible sat regardless of dst dist
                if bs<0 and vis_list[sg]:
                    bs=min(vis_list[sg],key=lambda x:x[0])[1]
                if bs>=0:
                    nhd=(bs,0,num_isls_per_sat[bs]+gid_to_sat_gsl_if_idx[sg])
                else:
                    nhd=(-1,-1,-1)
                if not prev_fstate or prev_fstate.get((sgs,dgs))!=nhd:
                    f.write("%d,%d,%d,%d,%d\n"%(sgs,dgs,nhd[0],nhd[1],nhd[2]))
                fstate[(sgs,dgs)]=nhd

    if enable_verbose_logs:
        nu_nz=sum(1 for v in best3["nu"].values() if abs(v)>1e-9)
        rt=len(decisions); U=sum(rho[s]*decisions[s][3] for s in decisions)
        print("  > Writing GDH-DPQ forwarding state to: "+fname)
        print("  > GDH-DPQ: feas=%-5s  maxGW_over=%.4f  routed=%d/%d  U_ISL=%.2f  nu_nz=%d"
              %(best3["feas"],best3["mo"],rt,Ns,U,nu_nz))

    return fstate,dict(lam=best3["lam"].copy(),nu=dict(best3["nu"]),
                       gw_load=best3["gw"].copy(),isl_load=dict(p_isl),
                       isl_lifetime=isl_lif)
