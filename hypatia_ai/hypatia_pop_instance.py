"""Option B: Hypatia's own topology + the standalone's population/multicommodity traffic.
Hypatia gives sats/ISL-graph/gateways; we overlay the standalone's GHS-population Poisson
demand, region-gated multicommodity (GW_SERVES/DST), and population-sited hotspot.
Returns a neural_solver_MCpath.Instance so RL + dual run on the SAME realistic-traffic problem."""
import sys, os, math, numpy as np
sys.path.insert(0,'.')
sys.path.insert(0,"/home/msarwa02/hypatia/satgenpy")
sys.path.insert(0,"/home/msarwa02/hypatia/paper/satellite_networks_state/gdh_dpq_rl")
import topology as topo
from instance_adapter import instance_from_hypatia, gw_positions_km, sat_positions_km
import neural_solver_MCpath as MP
import tifffile

# ---- constants copied verbatim from standalone_v6e.py ----
POP_TIF="/home/msarwa02/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
rng_seed=7; rho_min=0.1; rho_max=8.0
R_COV_KM=400.0; ACTIVITY_RATE=0.00003; DEMAND_GBPS=0.1
HUB_FRACTION=0.35; NREG=3
HOTSPOT_DEMAND=30.0; HOTSPOT_RADIUS_KM=500.0; HOTSPOT_POP_PCTILE=75; HOTSPOT_MAX_SATS=10
LON_ORIGIN=-180.0; LAT_ORIGIN=90.0; PIXEL_DEG=30.0/3600.0; DEG_PER_KM=1.0/111.32

# ---- time-varying traffic (from traffic_gen.py output) ----
_TS_DF=None; _ZLL=None; _ZLOADS=None; _ZTIMES=None
def _load_trace():
    global _TS_DF,_ZLL,_ZLOADS,_ZTIMES
    if _TS_DF is not None: return
    import pandas as pd
    ts=pd.read_csv("/home/msarwa02/ground_traffic_timeseries.csv")
    meta=pd.read_csv("/home/msarwa02/ground_zone_metadata.csv")
    _ZTIMES=pd.to_datetime(ts["Timestamp_UTC"],utc=True).values
    lcols=[c for c in ts.columns if c.startswith("Load_Gbps_")]
    _ZLOADS=ts[lcols].values.astype(np.float32)          # (n_steps, n_zones)
    # zone ECEF (lat/lon on ground, radius = earth)
    RE=6371.0
    lat=np.radians(meta["Lat"].values); lon=np.radians(meta["Lon"].values)
    _ZLL=np.stack([RE*np.cos(lat)*np.cos(lon), RE*np.cos(lat)*np.sin(lon), RE*np.sin(lat)],axis=1)
    _TS_DF=True

def get_traffic_for_time(t_utc, satpos):
    """Assign each populated zone's load at the nearest timestamp to the nearest satellite."""
    _load_trace()
    import pandas as pd, numpy as _np
    tt=pd.Timestamp(t_utc); tt=(tt.tz_localize("UTC") if tt.tzinfo is None else tt.tz_convert("UTC")).to_datetime64()
    idx=int(_np.argmin(_np.abs(_ZTIMES - tt)))
    loads=_ZLOADS[idx]                                    # (n_zones,)
    Ns=len(satpos); rho=_np.zeros(Ns,dtype=float)
    # nearest sat per zone (vectorized in chunks to bound memory)
    for z0 in range(0,len(loads),512):
        zl=_ZLL[z0:z0+512]                                # (chunk,3)
        L=loads[z0:z0+512]
        d=_np.linalg.norm(satpos[None,:,:]-zl[:,None,:],axis=2)  # (chunk,Ns)
        nearest=_np.argmin(d,axis=1)
        for k,sidx in enumerate(nearest):
            if L[k]>0: rho[sidx]+=L[k]
    return _np.clip(rho, rho_min, rho_max)

_a=6378137.0; _e2=6.69437999014e-3

def ecef_to_latlon(xyz):
    x,y,z=float(xyz[0]),float(xyz[1]),float(xyz[2])
    lon=math.degrees(math.atan2(y,x)); p=math.sqrt(x*x+y*y)
    lat=math.degrees(math.atan2(z,p*(1-_e2)))
    for _ in range(5):
        sl=math.sin(math.radians(lat)); N=_a/math.sqrt(1-_e2*sl*sl)
        lat=math.degrees(math.atan2(z+_e2*N*sl,p))
    return lat,lon
def lon_deg_per_km(lat):
    cl=math.cos(math.radians(lat)); return 0. if cl<1e-6 else 1.0/(111.32*cl)
def _haversine_km(la1,lo1,la2,lo2):
    R=6371.0; p1,p2=math.radians(la1),math.radians(la2)
    dp=math.radians(la2-la1); dl=math.radians(lo2-lo1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))
def _region_of(lat,lon):
    if -170.0<=lon<-30.0: return 0
    if -30.0<=lon<60.0: return 1
    return 2

def compute_pop_per_sat(sat_pos, pop_arr, Ns):
    nrows,ncols=pop_arr.shape
    pop=np.zeros(Ns)
    for i in range(Ns):
        lat,lon=ecef_to_latlon(sat_pos[i])
        dlat=R_COV_KM*DEG_PER_KM; dlon=R_COV_KM*lon_deg_per_km(lat)
        r0=int((LAT_ORIGIN-(lat+dlat))/PIXEL_DEG); r1=int((LAT_ORIGIN-(lat-dlat))/PIXEL_DEG)+1
        c0=int((lon-dlon-LON_ORIGIN)/PIXEL_DEG); c1=int((lon+dlon-LON_ORIGIN)/PIXEL_DEG)+1
        r0=max(0,r0); r1=min(nrows,r1); c0=max(0,c0); c1=min(ncols,c1)
        if r0>=r1 or c0>=c1: continue
        patch=pop_arr[r0:r1,c0:c1]
        lons_p=LON_ORIGIN+(np.arange(c0,c1)+0.5)*PIXEL_DEG
        lats_p=LAT_ORIGIN-(np.arange(r0,r1)+0.5)*PIXEL_DEG
        LON_G,LAT_G=np.meshgrid(lons_p,lats_p)
        dlat_g=(LAT_G-lat)/DEG_PER_KM
        dkm=1.0/max(lon_deg_per_km(lat),1e-9); dlon_g=(LON_G-lon)*dkm
        dist_g=np.sqrt(dlat_g**2+dlon_g**2)
        pop[i]=float(np.sum(patch[dist_g<=R_COV_KM]))
    return pop


def _elevation_angle(sat_ecef, gw_ecef):
    """EXACT copy of standalone elevation_angle: true local-horizon elevation via ENU frame."""
    import math as _m, numpy as _np
    x,y,z = gw_ecef
    lon = _m.atan2(y,x); hyp = _m.sqrt(x*x+y*y); lat = _m.atan2(z,hyp)
    sl,cl = _m.sin(lat),_m.cos(lat); slo,clo = _m.sin(lon),_m.cos(lon)
    R = _np.array([[-slo, clo, 0],
                   [-sl*clo,-sl*slo, cl],
                   [ cl*clo, cl*slo, sl]],dtype=float)
    enu = R@(_np.asarray(sat_ecef)-_np.asarray(gw_ecef))
    return _m.atan2(enu[2], _m.sqrt(enu[0]**2+enu[1]**2)+1e-12)

def _elev_visibility(sat_pos_km, gw_pos_km, elev_deg=15.0):
    """EXACT copy of standalone build_visibility: V[i,g]=1 iff elevation_angle >= elev_deg."""
    import numpy as _np, math as _m
    Ns=len(sat_pos_km); Ng=len(gw_pos_km); thr=_m.radians(float(elev_deg))
    V=_np.zeros((Ns,Ng),dtype=_np.uint8)
    for i in range(Ns):
        for g in range(Ng):
            if _elevation_angle(sat_pos_km[i], gw_pos_km[g]) >= thr:
                V[i,g]=1
    return V

def build_instance(scale=1.0, hotspot=True, use_std_gw=False, traffic_time=None, gen_path=None):
    GEN=gen_path if gen_path is not None else "/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_gdh_dpq_over_isls"
    gss,sats,epoch,isls=topo.read_inputs(GEN); gw=gw_positions_km(gss)
    _std_gw = use_std_gw
    # NOTE: base instance always builds with Hypatia's 100 gateways; if _std_gw, we swap to
    # the 99 Mega gateways AFTER the build (consistently: V, gw, Cg, Ng, sgprop all rebuilt).
    satpos=sat_positions_km(sats,str(epoch),str(epoch))
    sg,nisl,nbif,vis=topo.build_state_at(sats,gss,epoch,isls,0)
    Ns=len(sats); Ng=len(gss)

    cfg=MP.CFG(MAX_GW_EVAL=16,USE_LEARNED_EDGE_BIAS=False)
    inst=instance_from_hypatia(MP.Instance,MP.CFG,sat_graph=sg,vis_list=vis,satellites=sats,
        ground_stations=gss,epoch_str=str(epoch),date_str=str(epoch),rho=np.ones(Ns),
        Cg_val=40,Cisl_val=8,cfg=cfg,gw_pos_km=gw)

    # ---- population traffic on HYPATIA sat positions ----
    pop_arr=tifffile.imread(POP_TIF).astype(np.float32)
    if pop_arr.ndim==3: pop_arr=pop_arr[0]
    pop_arr=np.where(pop_arr<0,0.,pop_arr)
    pop_per_sat=compute_pop_per_sat(satpos, pop_arr, Ns); del pop_arr
    np.random.seed(rng_seed)
    N_users=np.random.poisson(ACTIVITY_RATE*pop_per_sat)
    rho_base=np.clip(N_users.astype(float)*DEMAND_GBPS, rho_min, rho_max)
    rho_base[pop_per_sat==0]=rho_min
    rho=rho_base*scale
    # ---- time-varying traffic override ----
    if traffic_time is not None:
        rho = get_traffic_for_time(traffic_time, satpos).astype(float)*scale

    # ---- multicommodity on HYPATIA gateways ----
    if _std_gw:
        gw = np.load("/home/msarwa02/standalone_mega_gw_km.npy")   # (99,3) ECEF km
        Ng = int(len(gw))
        inst.gateway_positions = np.asarray(gw, float)
        inst.V = _elev_visibility(satpos, gw, elev_deg=15.0).astype(np.uint8)
        inst.Cg = np.full(Ng, 40.0)
        inst.Ng = Ng
        inst.__post_init__()
        print(f"[use_std_gw] {Ng} Mega gateways; V {inst.V.shape}; sats-with-gw={int((inst.V.sum(1)>0).sum())}/{len(satpos)}")
    gw_latlon=np.array([ecef_to_latlon(gw[g]) for g in range(Ng)])
    sat_latlon=np.array([ecef_to_latlon(satpos[i]) for i in range(Ns)])
    gw_region=np.array([_region_of(gw_latlon[g,0],gw_latlon[g,1]) for g in range(Ng)],int)
    gw_pop=np.zeros(Ng)
    for g in range(Ng):
        d=np.array([_haversine_km(sat_latlon[i,0],sat_latlon[i,1],gw_latlon[g,0],gw_latlon[g,1]) for i in range(Ns)])
        gw_pop[g]=float(pop_per_sat[d<=1500.0].sum())
    hub_thr=np.quantile(gw_pop,1.0-HUB_FRACTION); GW_IS_HUB=gw_pop>=hub_thr
    GW_SERVES=np.zeros((Ng,NREG),bool)
    for g in range(Ng):
        GW_SERVES[g,gw_region[g]]=True
        if GW_IS_HUB[g]: GW_SERVES[g,:]=True
    sat_region=np.array([_region_of(sat_latlon[i,0],sat_latlon[i,1]) for i in range(Ns)],int)
    np.random.seed(rng_seed+101)
    DST=np.zeros(Ns,int)
    for i in range(Ns):
        if np.random.rand()<0.60: DST[i]=sat_region[i]
        else: DST[i]=int(np.random.choice([r for r in range(NREG) if r!=sat_region[i]]))

    # ---- population-sited hotspot on HYPATIA sats ----
    hs=np.array([],int)
    if hotspot:
        V=inst.V
        gwmin=np.array([min(_haversine_km(sat_latlon[i,0],sat_latlon[i,1],gw_latlon[g,0],gw_latlon[g,1]) for g in range(Ng)) for i in range(Ns)])
        pos_pop=pop_per_sat[pop_per_sat>0]
        thr=float(np.percentile(pos_pop,HOTSPOT_POP_PCTILE)) if len(pos_pop) else 0.0
        midlat=np.abs(sat_latlon[:,0])<=60.0
        dense=np.where((pop_per_sat>=thr)&midlat&(pop_per_sat>0))[0]
        if len(dense)==0: dense=np.where((pop_per_sat>0)&midlat)[0]
        if len(dense)==0: dense=np.where(pop_per_sat>0)[0]
        novis=np.array([i for i in dense if V[i].sum()==0],int)
        pool=novis if len(novis)>0 else dense
        best=int(pool[np.argmax(gwmin[pool])])
        clat,clon=sat_latlon[best,0],sat_latlon[best,1]
        d=np.array([_haversine_km(sat_latlon[i,0],sat_latlon[i,1],clat,clon) for i in range(Ns)])
        within=np.where(d<=HOTSPOT_RADIUS_KM)[0]
        if len(within)>HOTSPOT_MAX_SATS: within=within[np.argsort(d[within])[:HOTSPOT_MAX_SATS]]
        hs=within; rho[hs]=HOTSPOT_DEMAND*scale   # scale hotspot with load too
        print(f"hotspot: {len(hs)} sats @({clat:.1f},{clon:.1f}) nearest_gw={gwmin[best]:.0f}km vis={int(V[best].sum())}")

    inst.rho=rho; inst.DST=DST; inst.GW_SERVES=GW_SERVES; inst.gw_region=gw_region
    # ---- predictive lookahead: t+120s future state ----
    try:
        _LA=120.0
        epoch_next = epoch + _LA/86400.0                      # ephem.Date is float days
        satpos_next = sat_positions_km(sats, str(epoch_next), str(epoch_next))
        _Cvl = 299792.458                                     # km/s
        inst.V_next = _elev_visibility(satpos_next, gw, elev_deg=15.0).astype(np.uint8)
        _diff = satpos_next[:,None,:] - np.asarray(gw,float)[None,:,:]
        inst.sgprop_next = (np.linalg.norm(_diff,axis=2)/_Cvl).astype(np.float64)
        inst.edge_prop_next = np.array([np.linalg.norm(satpos_next[i]-satpos_next[j])/_Cvl
                                        for (i,j) in inst.isl_edges], dtype=np.float64)
        if traffic_time is not None:
            from datetime import timedelta as _td
            import pandas as _pd
            t7=_pd.Timestamp(traffic_time); t7=(t7.tz_localize("UTC") if t7.tzinfo is None else t7.tz_convert("UTC")) - _td(days=7)
            inst.rho_next = get_traffic_for_time(t7, satpos_next).astype(float)*scale
        else:
            inst.rho_next = rho.copy()
    except Exception as _e:
        # fallback: no lookahead -> next = current
        inst.V_next = inst.V.copy()
        inst.sgprop_next = getattr(inst,'sgprop',np.zeros((Ns,Ng))).copy() if hasattr(inst,'sgprop') else np.zeros((Ns,inst.Ng))
        inst.edge_prop_next = np.asarray(inst.edge_prop,float).copy()
        inst.rho_next = rho.copy()
        print('lookahead fallback:', _e)
    print(f"Ns={Ns} Ng={Ng} offered={rho.sum():.1f} pop_sats={(pop_per_sat>0).sum()} hubs={int(GW_IS_HUB.sum())}/{Ng}")
    return inst, hs, pop_per_sat

if __name__=="__main__":
    inst,hs,pop=build_instance(scale=1.0)
