"""Time-varying traffic generator (docx spec): GHS population -> 2deg zones ->
4 regions with Pen_Index -> 31-day 5-min diurnal + region-weekend + AR(1) series.
Writes ground_traffic_timeseries.csv (Timestamp_UTC + Load_Gbps_<z>) and
ground_zone_metadata.csv (zone Lat/Lon/Region/Pen). Run once."""
import numpy as np, pandas as pd, tifffile
from datetime import datetime, timedelta, timezone

POP_TIF   = "/home/msarwa02/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
OUT_TS    = "/home/msarwa02/ground_traffic_timeseries.csv"
OUT_META  = "/home/msarwa02/ground_zone_metadata.csv"
ACTIVITY_RATE=0.00003; DEMAND_GBPS=0.1
ZONE_DEG=2.0
LON_ORIGIN=-180.0; LAT_ORIGIN=90.0; PIXEL_DEG=30.0/3600.0
START = datetime(2020,2,23,0,0,0,tzinfo=timezone.utc)   # match Hypatia epoch date
N_DAYS=31; STEP_MIN=5
np.random.seed(7)

# ---- 1. aggregate raster into 2deg zones ----
print("loading raster..."); pop=tifffile.imread(POP_TIF).astype(np.float32)
if pop.ndim==3: pop=pop[0]
pop=np.where(pop<0,0.,pop)
nrows,ncols=pop.shape
# 2deg zone index per pixel
pix_per_zone=int(round(ZONE_DEG/PIXEL_DEG))          # 240 pixels = 2 deg
nz_row=nrows//pix_per_zone; nz_col=ncols//pix_per_zone
print("zones grid:",nz_row,"x",nz_col," pix/zone",pix_per_zone)
zones=[]
for zr in range(nz_row):
    r0=zr*pix_per_zone; r1=r0+pix_per_zone
    row_pop=pop[r0:r1]
    for zc in range(nz_col):
        c0=zc*pix_per_zone; c1=c0+pix_per_zone
        P=float(row_pop[:,c0:c1].sum())
        if P<=0: continue                              # discard oceans/empty
        clat=LAT_ORIGIN-(r0+pix_per_zone/2.0)*PIXEL_DEG
        clon=LON_ORIGIN+(c0+pix_per_zone/2.0)*PIXEL_DEG
        zones.append((clat,clon,P))
print("populated zones:",len(zones))
zlat=np.array([z[0] for z in zones]); zlon=np.array([z[1] for z in zones])
zpop=np.array([z[2] for z in zones])

# ---- 2. region + Pen_Index (4 regions) ----
def region_of(lat,lon):
    # Middle East
    if 12<=lat<=42 and 25<=lon<=63: return "MiddleEast"
    # Mediterranean (S Europe / N Africa rim)
    if 30<=lat<=46 and -10<=lon<=40: return "Mediterranean"
    # East Asia
    if 18<=lat<=53 and 100<=lon<=150: return "EastAsia"
    return "Standard"
REG_PEN={"Standard":0.60,"Mediterranean":0.70,"MiddleEast":0.50,"EastAsia":0.90}
zreg=np.array([region_of(zlat[i],zlon[i]) for i in range(len(zones))])
zpen=np.array([REG_PEN[r] for r in zreg])

# base scale per zone (Gbps at diurnal multiplier 1.0)
zbase = zpop*ACTIVITY_RATE*zpen*DEMAND_GBPS
print("base traffic total (mult=1): %.1f Gbps across %d zones"%(zbase.sum(),len(zones)))

# ---- 3. time index ----
n_steps=N_DAYS*24*60//STEP_MIN
times=[START+timedelta(minutes=STEP_MIN*k) for k in range(n_steps)]
print("time steps:",n_steps,"(",N_DAYS,"days @",STEP_MIN,"min )")

# ---- 4. diurnal + weekend, per region, vectorized over zones ----
def diurnal_mult(local_hour, region):
    # base sinusoid + lunch & evening spikes; region tweaks amplitude/phase
    base=0.55+0.45*np.sin((local_hour-8.0)/24.0*2*np.pi)      # daily rise/fall
    lunch=0.25*np.exp(-((local_hour-13.0)**2)/(2*1.2**2))
    evening=0.45*np.exp(-((local_hour-21.0)**2)/(2*1.8**2))
    m=base+lunch+evening
    if region=="EastAsia": m*=1.10
    if region=="MiddleEast": m*=0.95
    return np.clip(m,0.15,None)

def is_weekend(dt, region):
    wd=dt.weekday()  # Mon=0..Sun=6
    if region=="MiddleEast": return wd in (3,4)   # Thu/Fri
    return wd in (5,6)                             # Sat/Sun

# AR(1) state per zone
PHI=0.99; SIGMA=0.03
ar=np.zeros(len(zones))
loads=np.zeros((n_steps,len(zones)),dtype=np.float32)
lon_hours=zlon/15.0   # longitude -> local-time offset
for k,dt in enumerate(times):
    utc_h=dt.hour+dt.minute/60.0
    local_h=(utc_h+lon_hours)%24.0
    # per-zone diurnal (loop over 4 regions for speed)
    m=np.empty(len(zones))
    for reg in REG_PEN:
        mask=(zreg==reg)
        m[mask]=diurnal_mult(local_h[mask],reg)
        if is_weekend(dt,reg): m[mask]*=0.80        # weekends lighter
    # AR(1) exponential noise
    ar=PHI*ar+np.random.normal(0,SIGMA,len(zones))
    noise=np.exp(ar)
    loads[k]=(zbase*m*noise).astype(np.float32)
    if k% (n_steps//10)==0: print("  step %d/%d"%(k,n_steps))

# ---- 5. write CSVs ----
cols=["Load_Gbps_%d"%z for z in range(len(zones))]
df=pd.DataFrame(loads,columns=cols)
df.insert(0,"Timestamp_UTC",[t.isoformat() for t in times])
df.to_csv(OUT_TS,index=False)
meta=pd.DataFrame({"Zone":range(len(zones)),"Lat":zlat,"Lon":zlon,"Region":zreg,"Pen":zpen,"Pop":zpop})
meta.to_csv(OUT_META,index=False)
print("WROTE",OUT_TS,"(%d steps x %d zones)"%(n_steps,len(zones)))
print("WROTE",OUT_META)
print("traffic total range: min=%.1f max=%.1f mean=%.1f Gbps"%(loads.sum(1).min(),loads.sum(1).max(),loads.sum(1).mean()))
