"""
Multicommodity destination setup for Hypatia (ported from v6e standalone).
Each satellite's demand is bound for a destination REGION; a gateway can only
egress traffic for regions it serves (own region always; all regions if it's a
hub = top HUB_FRACTION by nearby population).
"""
import numpy as np, math

REGION_NAMES = ["Americas", "EMEA", "APAC"]
NREG = len(REGION_NAMES)
HUB_FRACTION = 0.35

def _ecef_to_latlon(xyz):
    x, y, z = xyz
    lon = math.degrees(math.atan2(y, x))
    p = math.sqrt(x*x + y*y)
    lat = math.degrees(math.atan2(z, p))
    return lat, lon

def region_of(lat, lon):
    if -170.0 <= lon < -30.0: return 0    # Americas
    if -30.0  <= lon <  60.0: return 1    # EMEA
    return 2                              # APAC

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def build_multicommodity(sat_pos_km, gw_pos_km, pop_per_sat, seed=7):
    """
    sat_pos_km : (Ns,3) ECEF km  OR  (Ns,2) lat/lon  OR list of (lat,lon)
    gw_pos_km  : (Ng,3) ECEF km  OR (Ng,2) lat/lon
    pop_per_sat: (Ns,) population under each satellite (for hub detection)
    Returns dict: DST (Ns,), GW_SERVES (Ng,NREG) bool, sat_region, gw_region, GW_IS_HUB
    """
    sat_pos_km = np.asarray(sat_pos_km, dtype=float)
    gw_pos_km  = np.asarray(gw_pos_km, dtype=float)
    Ns = len(sat_pos_km); Ng = len(gw_pos_km)

    # lat/lon for sats and gateways
    def to_latlon(arr):
        if arr.shape[1] == 2:                     # already lat/lon
            return arr[:,0], arr[:,1]
        lls = np.array([_ecef_to_latlon(arr[i]) for i in range(len(arr))])
        return lls[:,0], lls[:,1]
    sat_lat, sat_lon = to_latlon(sat_pos_km)
    gw_lat,  gw_lon  = to_latlon(gw_pos_km)

    sat_region = np.array([region_of(sat_lat[i], sat_lon[i]) for i in range(Ns)], dtype=int)
    gw_region  = np.array([region_of(gw_lat[g],  gw_lon[g])  for g in range(Ng)], dtype=int)

    # hub score = population within 1500 km of the gateway
    pop_per_sat = np.asarray(pop_per_sat, dtype=float)
    gw_pop = np.zeros(Ng)
    for g in range(Ng):
        d = np.array([_haversine_km(sat_lat[i], sat_lon[i], gw_lat[g], gw_lon[g])
                      for i in range(Ns)])
        gw_pop[g] = float(pop_per_sat[d <= 1500.0].sum())
    hub_thr = np.quantile(gw_pop, 1.0 - HUB_FRACTION)
    GW_IS_HUB = gw_pop >= hub_thr

    GW_SERVES = np.zeros((Ng, NREG), dtype=bool)
    for g in range(Ng):
        GW_SERVES[g, gw_region[g]] = True      # always serves own region
        if GW_IS_HUB[g]:
            GW_SERVES[g, :] = True              # hubs serve all regions

    # destination: 60% local, 40% to another region (fixed rule, not tuned)
    rng = np.random.default_rng(seed + 101)
    DST = np.zeros(Ns, dtype=int)
    for i in range(Ns):
        if rng.random() < 0.60:
            DST[i] = sat_region[i]
        else:
            others = [r for r in range(NREG) if r != sat_region[i]]
            DST[i] = int(rng.choice(others))

    return dict(DST=DST, GW_SERVES=GW_SERVES, sat_region=sat_region,
                gw_region=gw_region, GW_IS_HUB=GW_IS_HUB, gw_pop=gw_pop)

def summary(mc, Ng):
    d = mc
    print(f"Multicommodity: {NREG} regions {REGION_NAMES}")
    print(f"  gateways per region: {[int((d['gw_region']==r).sum()) for r in range(NREG)]}, "
          f"hubs = {int(d['GW_IS_HUB'].sum())}/{Ng}")
    print(f"  satellites per destination: {[int((d['DST']==r).sum()) for r in range(NREG)]}")
    print(f"  gateways usable per destination: {[int(d['GW_SERVES'][:,r].sum()) for r in range(NREG)]}")

def site_hotspot(sat_pos_km, gw_pos_km, V, pop_per_sat, pctile=75, radius_km=500.0):
    """v6e hotspot siting: dense (pop>=pctile) + prefer no-visible-gateway +
    farthest from any gateway. Returns (hs_indices, center_latlon, best, nearest_km, nvis)."""
    sat_pos_km=np.asarray(sat_pos_km,float); gw_pos_km=np.asarray(gw_pos_km,float)
    def to_latlon(arr):
        if arr.shape[1]==2: return arr[:,0],arr[:,1]
        lls=np.array([_ecef_to_latlon(arr[i]) for i in range(len(arr))])
        return lls[:,0],lls[:,1]
    sat_lat,sat_lon=to_latlon(sat_pos_km); gw_lat,gw_lon=to_latlon(gw_pos_km)
    Ns=len(sat_lat); Ng=len(gw_lat)
    gwmin=np.array([min(_haversine_km(sat_lat[i],sat_lon[i],gw_lat[g],gw_lon[g])
                        for g in range(Ng)) for i in range(Ns)])
    pop_per_sat=np.asarray(pop_per_sat,float)
    pos_pop=pop_per_sat[pop_per_sat>0]
    thr=float(np.percentile(pos_pop,pctile)) if len(pos_pop) else 0.0
    dense=np.where(pop_per_sat>=thr)[0]; dense=dense[pop_per_sat[dense]>0]
    novis=np.array([i for i in dense if V[i].sum()==0],dtype=int)
    pool=novis if len(novis) else dense
    best=int(pool[np.argmax(gwmin[pool])])
    center=(float(sat_lat[best]),float(sat_lon[best]))
    d=np.array([_haversine_km(sat_lat[i],sat_lon[i],center[0],center[1]) for i in range(Ns)])
    hs=np.where(d<=radius_km)[0]
    return hs, center, best, float(gwmin[best]), int(V[best].sum())
