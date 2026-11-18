"""
instance_adapter.py -- build the neural solver's `Instance` from Hypatia data.

The neural construction policy (GDH-DPQ Neural Solver v2) expects an `Instance`
with a specific data contract. Hypatia stores the same information in different
structures. This module bridges the two.

CRITICAL UNIT NOTE
------------------
`Instance` uses KILOMETRES and c_light = 299792.458 (km/s).
Hypatia uses METRES everywhere.
Every distance crossing this boundary is divided by 1000. Getting this wrong
makes all delays 1000x too large and the policy's D_MAX=0.5s filter rejects
every path.
"""
import math
import numpy as np

import ephem
from satgen.distance_tools import geodetic2cartesian


# ----------------------------------------------------------------------
# positions (ECEF, km)
# ----------------------------------------------------------------------
def _make_observer(epoch_str, date_str):
    """Observer at (0,0,0) -- mirrors satgen.distance_tools convention."""
    obs = ephem.Observer()
    obs.epoch = epoch_str
    obs.date = date_str
    obs.lat = 0
    obs.lon = 0
    obs.elevation = 0
    return obs


def sat_positions_km(satellites, epoch_str, date_str):
    """(Ns,3) ECEF km. Mirrors how satgen computes satellite state:
    sat.compute(observer) then read sublat/sublong/elevation."""
    obs = _make_observer(epoch_str, date_str)
    out = np.zeros((len(satellites), 3), dtype=float)
    for i, sat in enumerate(satellites):
        sat.compute(obs)
        x, y, z = geodetic2cartesian(
            math.degrees(sat.sublat),
            math.degrees(sat.sublong),
            sat.elevation,           # metres above sea level
        )
        out[i] = (x / 1000.0, y / 1000.0, z / 1000.0)   # m -> km
    return out


def gw_positions_km(ground_stations):
    """(Ng,3) ECEF km. read_ground_stations_extended already caches
    cartesian_x/y/z (metres), so no conversion is needed -- just scale."""
    out = np.zeros((len(ground_stations), 3), dtype=float)
    for g, gs in enumerate(ground_stations):
        out[g] = (gs["cartesian_x"] / 1000.0,
                  gs["cartesian_y"] / 1000.0,
                  gs["cartesian_z"] / 1000.0)
    return out


# ----------------------------------------------------------------------
# the adapter
# ----------------------------------------------------------------------
def instance_from_hypatia(Instance, CFG,
                          sat_graph, vis_list,
                          satellites, ground_stations,
                          epoch_str, date_str,
                          rho, Cg_val, Cisl_val,
                          cfg=None,
                          isl_lifetime_s=None,
                          sat_pos_km=None, gw_pos_km=None):
    """
    Parameters
    ----------
    Instance, CFG : the classes from the neural solver module (passed in to
                    avoid a hard import dependency / circular imports).
    sat_graph     : networkx graph, edge attr 'weight' in METRES.
    vis_list      : ground_station_satellites_in_range_candidates --
                    list (len Ng) of lists of (distance_m, sat_id).
    satellites    : list of ephem satellite objects.
    ground_stations : list of dicts from read_ground_stations_extended.
    epoch_str, date_str : strings for ephem (str(epoch), str(time)).
    rho           : (Ns,) offered demand, Gbps.
    Cg_val, Cisl_val : scalar capacities (Gbps).
    isl_lifetime_s : optional (Ns,Ns). If None, a CONSTANT L_MAX_LIFE is used
                     -- this matches build_shell() in the neural solver, which
                     treats lifetimes as a secondary edge feature. Hypatia's
                     own isl_lifetime is inert (closed None loop), so there is
                     nothing to inherit.
    sat_pos_km, gw_pos_km : optional precomputed positions (skip recompute).
    """
    cfg = cfg or CFG()
    Ns = sat_graph.number_of_nodes()
    Ng = len(ground_stations)

    # --- adjacency + dense distance matrix (METRES -> KM) ---
    NBR_LIST = [[int(v) for v in sat_graph.neighbors(i)] for i in range(Ns)]
    dist = np.zeros((Ns, Ns), dtype=float)
    for u, v, d in sat_graph.edges(data=True):
        km = float(d["weight"]) / 1000.0
        dist[u, v] = dist[v, u] = km

    # --- visibility matrix (Ns, Ng) ---
    V = np.zeros((Ns, Ng), dtype=np.uint8)
    for gid, lst in enumerate(vis_list):
        for _dm, sid in lst:
            V[int(sid), gid] = 1

    # --- positions (ECEF km) ---
    if sat_pos_km is None:
        sat_pos_km = sat_positions_km(satellites, epoch_str, date_str)
    if gw_pos_km is None:
        gw_pos_km = gw_positions_km(ground_stations)

    # --- lifetimes ---
    if isl_lifetime_s is None:
        isl_lifetime_s = np.full((Ns, Ns), float(cfg.L_MAX_LIFE))

    return Instance(
        Ns=Ns, Ng=Ng,
        V=V,
        rho=np.asarray(rho, dtype=float),
        NBR_LIST=NBR_LIST,
        dist=dist,
        isl_lifetime_s=isl_lifetime_s,
        Cg=np.full(Ng, float(Cg_val), dtype=float),
        Cisl=float(Cisl_val),
        sat_positions=sat_pos_km,
        gateway_positions=gw_pos_km,
        cfg=cfg,
    )
