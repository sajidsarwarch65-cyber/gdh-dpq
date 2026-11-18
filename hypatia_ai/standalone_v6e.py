# ==============================================================
# GDH-DPC BENCHMARK v6e -- DUAL-PRIMARY FALLBACK (Path 1 rewrite)
# ==============================================================
# Critical fix vs v6d: the fallback now uses Phase-1/Phase-2 dual
# prices (best_lam[g] and best_nu[edge]) as the PRIMARY routing
# signal. Previously (v6b-v6d) best_lam was declared but never used,
# and best_nu entered only as a small bias term (~0.5x) on top of
# live queue-delay heuristics. This meant:
#   - The fallback was effectively an autonomous greedy admission
#     algorithm with dual prices as decoration.
#   - Full Proposed, P1-only, and P2-only ablation curves collapsed
#     to nearly identical PLR because they all used the same
#     live-queue-driven fallback.
#   - The paper's "dual decomposition" contribution wasn't the actual
#     routing signal.
#
# v6e treats dual prices as primary:
#   * ISL routing weight: prop_delay + best_nu[edge] + safety + stab
#   * Gateway cost:       prop_delay + best_lam[g] + gw_safety
#   * Live signals (queue_delay, linear load) REMOVED from routing.
#   * "safety" is a small step function activated only above 85% load
#     to protect against dual-blind edges that saturate during
#     admission accumulation.
#   * ALPHA4 reduced to 0.0001 so stability term doesn't overwhelm
#     dual nu magnitude.
#
# Expected outcome: full Proposed vs P1-only vs P2-only should now
# show meaningful ablation. If dual prices are informative, full
# Proposed should differentiate. If not, PLR gets worse than v6d,
# telling us the dual solver isn't producing tight prices.
# Change vs v6c:
#   1. REMOVED ELB (Taleb 2009). ELB was designed for sat-to-sat
#      routing without gateway selection, not for the opportunistic
#      routing problem that this paper addresses. Including it would
#      require heavy adaptation that no longer represents the
#      original algorithm.
#
#   2. REPLACED "pure BP" (Tassiulas 1990) with a *correct*
#      backpressure baseline following Krishnan et al. 2026 Eq. 14
#      (arXiv:2601.13662, "Reinforcement Learning for Opportunistic
#      Routing in Software-Defined LEO-Terrestrial Systems"), which
#      explicitly formulates BP for opportunistic routing in the
#      exact same problem class considered here.
#
#   3. ADDED LG-BP (Link-to-Ground-Aware Backpressure) baseline,
#      per Krishnan et al. 2026 Eq. 17. This is the state-of-the-art
#      analytical baseline for opportunistic routing.
#
#   3b. ADDED LSP-Chen baseline: Q. Chen et al. 2019 Acta Astronautica,
#      "A distributed congestion avoidance routing algorithm in
#      mega-constellation network with multi-gateway." Uses gateway
#      assignment by nearest hop count, Path Candidate Region, and
#      Long Side Priority greedy path build. Directly designed for
#      LEO+multi-gateway routing without adaptation of core logic.
#
#   4. FIXED Proposed (path-A tuning):
#      - Replaced stale Phase-2 nu with LIVE nu recomputed from
#        current ISL loads during fallback admission (biggest fix).
#      - Reduced ALPHA4 stability weight from 0.01 to 0.001 (was
#        forcing detours away from lifetime-short ISLs).
#      - Reduced FB_HOP_PEN from 0.0005 to 0.0001 (was penalising
#        hops even when needed).
#
# Baseline set (final): Direct-Drop, DSP, CA-DSP, DBPR, GPSR,
#                       BP (Krishnan Eq. 14),
#                       LG-BP (Krishnan Eq. 17),
#                       Proposed, Proposed-P1, Proposed-P2.
#
# All infrastructure from v6c preserved: multi-snapshot, ISL sweep,
# visibility diagnostic, KPI export, plotting.
# ==============================================================

import os, math, heapq, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import geopandas as gpd
from shapely.geometry import Point
from sgp4.api import Satrec, WGS72

# ============================================================
# A) FILE PATHS
# ============================================================
FOLDER   = r"/home/msarwa02"
CSV_FILE = os.path.join(FOLDER, "starlink_gp_2026_02_23_1035UTC.csv")
KML_FILE = os.path.join(FOLDER,
           "Unofficial Starlink Global Gateways & PoPs.kml")
POP_TIF  = os.path.join(FOLDER,
           "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif")

assert os.path.exists(CSV_FILE), f"CSV not found: {CSV_FILE}"
assert os.path.exists(KML_FILE), f"KML not found: {KML_FILE}"
assert os.path.exists(POP_TIF),  f"POP raster not found: {POP_TIF}"

# ============================================================
# B) CONTROLS
# ============================================================
# Walker +Grid dimensions
N_PLANES         = 30
N_SLOTS          = 40

rng_seed         = 7
prop_time        = datetime(2026, 2, 23, 16, 44, 27,
                            tzinfo=timezone.utc)

# Traffic
rho_min          = 0.1
rho_max          = 8.0

# ISL-tight capacity regime (default)
Cg_default       = 40.0
Cisl             = 8.0     # will be overridden during Cisl sweep

# Delay constants
c_light          = 299792.458
D_TX_ISL         = 0.0002
D_PROC_ISL       = 0.0001
D_TX_SG          = 0.0005
D_PROC_SG        = 0.0002
queue_scale      = 0.003   # ~3.4 ms at 90% load on 8 Gbps ISL (convex shape kept)
QUEUE_MAX        = 0.03     # saturated-link queue capped at a physical ~30 ms
                            # (partial admission pins bottleneck links at ~100%,
                            # where any convex M/M/1 term explodes; this ceiling
                            # keeps reported/routed queueing sane while the
                            # 80-95% gradient still drives congestion avoidance)
queue_eps_frac   = 1e-3

# Dual decomposition
alpha1           = 1.0
alpha2           = 0.02
DUAL_MAX_ITERS   = 80    # was 200; the loop never reaches strict feasibility
                         # here so it always spent the full budget. 80 + the
                         # early-stop below (break when the best iterate stops
                         # improving) cuts runtime ~3-4x with negligible quality
                         # loss, since the best iterate plateaus well before 80.
DUAL_EARLY_STOP  = 20    # stop a stage if the best feasibility metric has not
                         # improved for this many consecutive iterations
LAM_MAX          = 100.0
NU_MAX           = 100.0
eta_load         = 0.25
# Diminishing-step subgradient (replaces the Polyak step, which overshot and
# chased the per-iteration stampede so overload never fell). step_t decays as
# STEP0/(1+DECAY*it) -- provably convergent for subgradient ascent -- and the
# violation is measured on the EMA-smoothed load (eta_load) so prices respond
# to sustained congestion, not one iteration's bang-bang reassignment.
MAX_SPLIT_PATHS  = 8     # a satellite's aggregate demand may be split over up
                         # to this many paths (successive shortest paths under
                         # current prices/loads). Real per-sat demand is an
                         # aggregate of many sessions and is divisible; with
                         # Cisl=8 and per-sat demand 8-16 Gbps, an indivisible
                         # flow could not cross a single ISL at all.
DUAL_STEP0       = 0.15
DUAL_STEP_DECAY  = 0.03
TIE_EPS          = 1e-12
ALPHA_HO         = 0.02
THO_S            = 60.0
LIFE_DT_S        = 20.0
LIFE_HORIZON_S   = 1200.0
POLYAK_ALPHA     = 0.5
STAGNATION_K     = 50
STEP_FLOOR       = 1e-6
STEP_CEIL        = 1.0

# Baseline penalty weights
CADSP_GW_PEN     = 1.0
CADSP_ISL_PEN    = 0.5
DBPR_ALPHA_Q     = 1.0
DBPR_BETA_DIST   = 0.25
DBPR_ITERS       = 5

# ============================================================
# Baselines specific to opportunistic routing
# ------------------------------------------------------------
# BP and LG-BP parameters (Krishnan et al. 2026)
# BP score (Eq. 14):    s^BP(k->m) = (Q_k - Q_m) * C_ISL
# LG-BP score (Eq. 17): s^LG(k->m) = s^BP + lambda_LG * C_m^LG
# Adapted for single-snapshot non-splittable admission: "queue" = cumulative
# committed ISL load; "C_m^LG" = gateway availability & headroom at neighbor.
BP_ALPHA_ISL       = 1.0    # weight on ISL-load queue proxy
LGBP_LAMBDA_LG     = 0.5    # weight on gateway-availability term
LGBP_NO_GW_PENALTY = 0.5    # penalty for nodes with no visible gateway to target

# ------------------------------------------------------------
# Fallback tuning for Proposed (path-A fix: reduced hop penalty
# and reduced stability weight; live nu computed dynamically
# from current ISL loads instead of stale Phase-2 dual variables.)
FB_LINEAR_GW     = 0.5
FB_LINEAR_ISL    = 0.3
FB_HOP_PEN       = 0.0001    # was 0.0005; reduced 5x to allow needed hops
NU_ACTIVATION_TH = 0.5       # activate live-nu only when ISL load > 50%
LIVE_NU_MAX      = 0.20      # was 0.05; higher ceiling so nu affects routing

# Path validity filter
D_MAX            = 0.5       # 500 ms end-to-end
L_MIN            = 30.0      # 30 s per ISL

# v6b algorithmic improvements
ALPHA4           = 0.0001   # was 0.001 in v6d; reduced 10x more so
                            # stab doesn't overwhelm dual nu signal
                            # in v6e's dual-primary fallback
L_MAX_LIFE       = 600.0
KAPPA            = 0.5
EPS_MIN_FRAC     = 0.02
EPS_MAX_FRAC     = 0.10
CISL_EPS_REF     = 8.0   # freeze the dual ISL feasibility tolerance at the
                         # default Cisl. Without this, eps_ISL = frac*Cisl
                         # loosens the dual loop's stopping rule as Cisl grows,
                         # so it breaks before the gateway prices (the binding
                         # resource) converge -> non-monotone PLR in the Cisl
                         # sweep. Referencing a fixed Cisl keeps the stopping
                         # rule constant across the sweep. Equals the live Cisl
                         # at the default point, so the traffic sweep and every
                         # Cisl=8 result are unchanged.
CG_EPS_REF       = 40.0  # same idea for the gateway tolerance. In the Ng sweep
                         # per-gw Cg is boosted (up to 660) to preserve aggregate
                         # capacity; eps_GW = frac*mean(Cg) would then tolerate
                         # ~13 Gbps of gateway overload and Stage A would stop
                         # before lambda converges. Freezing at the default Cg
                         # keeps lambda tight regardless of the boosted Cg.
                         # Equals the live Cg at the default point, so all
                         # non-Ng-sweep results are unchanged.

# ============================================================
# SWEEP CONFIGURATION
# ============================================================
ELEV_FIXED        = 15
NG_SWEEP          = [6, 10, 20]   # sparse-gateway points only (ISL-bound); sparsest first
CG_SWEEP          = [20.0, 30.0, 40.0, 60.0, 80.0]
ELEV_SWEEP        = [10, 15, 20, 25, 30, 35]
SCALE_SWEEP       = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CISL_SWEEP        = [4.0, 6.0, 8.0, 12.0, 16.0, 20.0]   # NEW
SCALES_FIXED      = [1.0, 1.5]

# Which sweeps to run (all default TRUE to allow flip-toggling)
RUN_TRAFFIC_SWEEP = False   # already have these results; skip to Ng sweep
RUN_CISL_SWEEP    = False    # already have these results; skip to Ng sweep
RUN_NG_SWEEP      = False   # gateway-sparsity probe (done)
NG_PRESERVE_GW_CAP = True   # hold TOTAL gateway capacity fixed across the Ng
                            # sweep (per-gw Cg = Ng_full*Cg_default/Ng). This
                            # isolates gateway PLACEMENT/sparsity from aggregate
                            # gateway capacity: with fewer, higher-capacity
                            # gateways, traffic must backhaul over ISLs and the
                            # ISL links feeding each gateway become the binding
                            # cut -- the regime where the ISL dual can help.
                            # Set False for a plain "fewer stations, same Cg"
                            # sweep (aggregate GW capacity then also shrinks).

# ---- Realistic demand hotspot (dense region, sparse ground infrastructure) --
# Instead of an arbitrary ocean point, the hotspot is sited over a genuinely
# DENSE populated region that has NO local gateway coverage -- a real backhaul-
# stress scenario (a dense inland/island metro or event/data-centre egress with
# no nearby teleport). Such a region's aggregated demand must cross the ISL mesh
# to reach a distant gateway, which is where ISL queueing and the ISL dual
# actually matter. The center is chosen from the data at runtime (see siting
# block after population is loaded), not hard-coded to a guessed location.
# The radius is kept small so the crowd is a CONCENTRATED source (a few sats
# each carrying large aggregated load, like a stadium / DC egress), and the
# per-sat demand is set directly -- above the rho_max=8 residential cap -- to
# represent that aggregation.
RUN_HOTSPOT          = True
HOTSPOT_DEMAND       = 30.0           # crowd load per hotspot sat, held FIXED
                                       # (the hotspot is what separates the
                                       # variants: without it all four tie)
HOTSPOT_SCALE_SWEEP  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # background traffic scale
HOTSPOT_CENTER       = None            # auto-selected from data (dense + gw-sparse)
HOTSPOT_RADIUS_KM    = 500.0           # concentrated source (~a few sats), not a region
HOTSPOT_POP_PCTILE   = 75              # "dense" = population at/above this percentile
RUN_ELEV_SWEEP    = False   # legacy
RUN_CG_SWEEP      = False   # legacy
RUN_VIS_DIAGNOSTIC = True   # visibility-vs-elevation figure
RUN_DUAL_ABLATION  = True   # add lambda-only / nu-only / no-dual variants of
TRACE_DUAL         = True   # per-iteration dual trace (full Proposed only):
                            # served, gateway/ISL overload, price counts. Shows
                            # whether the ascent works and whether the ISL
                            # constraint is ever binding (the coupling test).
                            # Proposed to isolate each dual's contribution
ABLATION_ONLY      = True   # baselines already benchmarked: run ONLY the
                            # Proposed family (full + the three dual ablations)
                            # and skip Direct Drop/DSP/CA-DSP/DBPR/GPSR/BP/
                            # LG-BP/LSP-Chen. Set False to include baselines.

# ============================================================
# MULTI-SNAPSHOT CONFIGURATION
# ============================================================
# For time-averaged multi-snapshot studies. Default N_SNAPSHOTS=1
# gives identical behaviour to v6b (single snapshot at prop_time).
N_SNAPSHOTS         = 1
SNAPSHOT_DELTAS_MIN = [0]   # minute offsets from prop_time; first
                            # entry MUST be 0 to preserve v6b baseline
assert len(SNAPSHOT_DELTAS_MIN) >= N_SNAPSHOTS
assert SNAPSHOT_DELTAS_MIN[0] == 0

# ============================================================
# SCHEMES
# ============================================================
# The Proposed family: full method plus (optionally) the three dual ablations.
_PROPOSED_FAMILY = ["Proposed"]                      # single-phase dual (headline)
if RUN_DUAL_ABLATION:
    _PROPOSED_FAMILY += ["Proposed-LamOnly",  # gateway duals only (nu off)
                         "Proposed-NuOnly",   # ISL duals only (lambda off)
                         "Proposed-NoDual"]   # no duals (delay-greedy + partial admit)

_BASELINES = [
    "Direct Drop", "DSP", "CA-DSP", "DBPR", "GPSR",
    "BP", "LG-BP",                              # opp-routing baselines
    "LSP-Chen",                                 # Chen 2019 Acta Astronautica
]

if ABLATION_ONLY:
    SCHEMES = list(_PROPOSED_FAMILY)               # Proposed + ablations only
else:
    SCHEMES = _BASELINES + _PROPOSED_FAMILY        # full comparison

print("=" * 60)
print("v6e CONTROLS (dual-primary fallback)")
print("=" * 60)
print(f"Walker +Grid: P={N_PLANES} planes x S={N_SLOTS} slots "
      f"= {N_PLANES*N_SLOTS} sats")
print(f"Capacity: Cg={Cg_default} Gbps, Cisl={Cisl} Gbps (default)")
print(f"Path filter: D_MAX={D_MAX*1000:.0f}ms, L_MIN={L_MIN:.0f}s")
print(f"Proposed tuning (v6e): ALPHA4={ALPHA4} (was 0.001), "
      f"FB_HOP_PEN={FB_HOP_PEN}, KAPPA={KAPPA}")
print(f"  Fallback: DUAL-PRIMARY (best_lam and best_nu drive routing)")
print(f"  Safety net: activates only above 85% load, magnitude {0.1}s")
print(f"BP:    alpha_ISL={BP_ALPHA_ISL}")
print(f"LG-BP: lambda_LG={LGBP_LAMBDA_LG}, no_gw_pen={LGBP_NO_GW_PENALTY}")
print(f"N_SNAPSHOTS={N_SNAPSHOTS}, deltas={SNAPSHOT_DELTAS_MIN[:N_SNAPSHOTS]}")
print(f"Sweeps enabled: traffic={RUN_TRAFFIC_SWEEP}, "
      f"cisl={RUN_CISL_SWEEP}, ng={RUN_NG_SWEEP}, "
      f"elev={RUN_ELEV_SWEEP}, cg={RUN_CG_SWEEP}")
print(f"Schemes: {SCHEMES}")

# ============================================================
# C) COORDINATE HELPERS
# ============================================================
def datetime_to_jd_fr(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt  = dt.astimezone(timezone.utc)
    y,m,d = dt.year, dt.month, dt.day
    hh,mm = dt.hour, dt.minute
    sec   = dt.second + dt.microsecond*1e-6
    if m <= 2: y -= 1; m += 12
    A  = int(y/100); B = 2-A+int(A/4)
    jd = int(365.25*(y+4716))+int(30.6001*(m+1))+d+B-1524.5
    jd += (hh+(mm+sec/60.)/60.)/24.
    ji  = math.floor(jd)
    return ji, jd-ji

def gmst_rad(jd, fr):
    T  = ((jd+fr)-2451545.0)/36525.0
    gs = (67310.54841+(876600*3600+8640184.812866)*T
          +0.093104*T*T-6.2e-6*T**3) % 86400.
    if gs < 0: gs += 86400.
    return 2*np.pi*(gs/86400.)

def R3(th):
    c,s = np.cos(th),np.sin(th)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]],dtype=float)

_a = 6378.137; _f = 1/298.257223563; _e2 = _f*(2-_f)

def ecef_from_latlon(lat_deg, lon_deg, h_km=0.):
    la,lo = np.deg2rad(lat_deg),np.deg2rad(lon_deg)
    sl    = np.sin(la)
    N     = _a/np.sqrt(1-_e2*sl*sl)
    return np.array([(N+h_km)*np.cos(la)*np.cos(lo),
                     (N+h_km)*np.cos(la)*np.sin(lo),
                     ((1-_e2)*N+h_km)*sl],dtype=float)

def elevation_angle(sat_ecef, gw_ecef):
    x,y,z = gw_ecef
    lon   = math.atan2(y,x); hyp = math.sqrt(x*x+y*y)
    lat   = math.atan2(z,hyp)
    sl,cl = math.sin(lat),math.cos(lat)
    slo,clo = math.sin(lon),math.cos(lon)
    R   = np.array([[-slo, clo, 0],
                    [-sl*clo,-sl*slo, cl],
                    [ cl*clo, cl*slo, sl]],dtype=float)
    enu = R@(sat_ecef-gw_ecef)
    return math.atan2(enu[2], math.sqrt(enu[0]**2+enu[1]**2)+1e-12)

def queue_delay(load, cap):
    # M/M/1-style mean queueing delay: W_q = rho / (cap*(1-rho)) = rho/(cap-load).
    # rho-weighted so an empty link costs ~0 and the cost rises CONVEXLY toward
    # capacity with no artificial plateau (the old version clamped load at
    # 0.999*cap and capped at 50 ms, flattening exactly where a real queue
    # blows up). queue_scale is set so a ~90%-loaded 8 Gbps ISL costs ~9 ms --
    # comparable to per-hop propagation -- so the router actually feels
    # congestion in the 80-95% band instead of being blind to it.
    cap = float(cap); load = max(0., float(load))
    rho = min(load / max(cap, 1e-12), 1. - queue_eps_frac)
    wq  = queue_scale * rho / max(cap * (1. - rho), 1e-9)
    return min(wq, QUEUE_MAX)
# ============================================================
# D) WALKER CONSTELLATION
# ============================================================
df_csv = pd.read_csv(CSV_FILE)
print(f"\nTotal sats in CSV: {len(df_csv)}")

raan_all = df_csv["RA_OF_ASC_NODE"].to_numpy() % 360.0
ma_all   = df_csv["MEAN_ANOMALY"].to_numpy()   % 360.0

order_raan    = np.argsort(raan_all)
ranks         = np.empty_like(order_raan)
ranks[order_raan] = np.arange(len(df_csv))
plane_id_full = np.clip(
    (ranks * N_PLANES) // len(df_csv), 0, N_PLANES-1)

plane_raan_centers = np.array([
    raan_all[plane_id_full == p].mean() for p in range(N_PLANES)])
print(f"RAAN plane centers (deg): "
      f"min={plane_raan_centers.min():.1f} "
      f"max={plane_raan_centers.max():.1f}")

np.random.seed(rng_seed)
target_anomalies = np.linspace(0, 360, N_SLOTS, endpoint=False)
walker_pick      = np.full((N_PLANES, N_SLOTS), -1, dtype=int)

for p in range(N_PLANES):
    plane_idx = np.where(plane_id_full == p)[0]
    if len(plane_idx) < N_SLOTS:
        if len(plane_idx) == 0:
            nearest_p = max(0, p-1) if p > 0 else p+1
            plane_idx = np.where(plane_id_full == nearest_p)[0]
    pa = ma_all[plane_idx]
    for s in range(N_SLOTS):
        diff = np.abs(((pa - target_anomalies[s] + 180) % 360) - 180)
        pick = int(plane_idx[np.argmin(diff)])
        walker_pick[p, s] = pick

sat_csv_idx = walker_pick.reshape(-1)
Ns          = N_PLANES * N_SLOTS
print(f"Walker constellation: P={N_PLANES}, S={N_SLOTS}, Ns={Ns}")

def ps_to_i(p, s): return int(p) * N_SLOTS + int(s)
def i_to_ps(i):    return divmod(int(i), N_SLOTS)

JD_REF  = 2433281.5
mu_grav = 398600.4418
satrecs = [None] * Ns
n_bad   = 0
for i in range(Ns):
    row  = df_csv.iloc[int(sat_csv_idx[i])]
    mm   = float(row["MEAN_MOTION"])
    ecc  = float(row["ECCENTRICITY"])
    inc  = np.deg2rad(float(row["INCLINATION"]))
    raan = np.deg2rad(float(row["RA_OF_ASC_NODE"]))
    argp = np.deg2rad(float(row["ARG_OF_PERICENTER"]))
    M    = np.deg2rad(float(row["MEAN_ANOMALY"]))
    bstar= float(row["BSTAR"]) if ("BSTAR" in row
                                   and pd.notna(row["BSTAR"])) else 0.
    ep   = pd.to_datetime(row["EPOCH"],utc=True).to_pydatetime()
    ji,fr= datetime_to_jd_fr(ep)
    ep_d = (ji+fr)-JD_REF
    n_rps= mm*2*np.pi/(24*3600)
    a_km = (mu_grav/n_rps**2)**(1/3)
    if not np.isfinite(a_km) or a_km<6500 or a_km>9000:
        n_bad+=1; continue
    try:
        sat = Satrec()
        sat.sgp4init(WGS72,"i",
            int(row["NORAD_CAT_ID"]) if "NORAD_CAT_ID" in row
            else (100000+i),
            ep_d, bstar, 0.,0., ecc, argp, inc, M,
            mm*2*np.pi/1440., raan)
        satrecs[i] = sat
    except Exception:
        n_bad+=1
print(f"SGP4 init: bad={n_bad}/{Ns}")

if n_bad > 0:
    for i in range(Ns):
        if satrecs[i] is None:
            for j in range(Ns):
                if satrecs[j] is not None:
                    satrecs[i] = satrecs[j]; break

# ============================================================
# E) PROPAGATION + +Grid BUILDER (functions for multi-snapshot)
# ============================================================
def prop_positions(t_utc, sats):
    jd,fr = datetime_to_jd_fr(t_utc)
    Rec   = R3(gmst_rad(jd,fr))
    pos   = np.zeros((len(sats),3))
    good  = np.ones(len(sats),dtype=bool)
    for i,sat in enumerate(sats):
        e,r,v = sat.sgp4(jd,fr)
        if e!=0 or np.isnan(r).any(): good[i]=False; continue
        pos[i] = Rec@np.array(r,dtype=float)
    return pos, good

def build_plus_grid_and_dist(sat_pos):
    """Build +Grid adjacency and distance matrix from positions."""
    adj_local = np.zeros((Ns, Ns), dtype=np.uint8)
    for p in range(N_PLANES):
        for s in range(N_SLOTS):
            i = ps_to_i(p, s)
            j = ps_to_i(p, (s+1) % N_SLOTS); adj_local[i,j]=adj_local[j,i]=1
            j = ps_to_i((p+1) % N_PLANES, s); adj_local[i,j]=adj_local[j,i]=1
    dist_local = np.zeros((Ns, Ns), dtype=np.float64)
    for i in range(Ns):
        for j in np.where(adj_local[i]>0)[0]:
            if j > i:
                d = float(np.linalg.norm(sat_pos[i]-sat_pos[j]))
                dist_local[i,j] = dist_local[j,i] = d
    return adj_local, dist_local

# Initial propagation at base prop_time (snapshot 0)
sat_pos_raw, good0 = prop_positions(prop_time, satrecs)
if not good0.all():
    for i in np.where(~good0)[0]:
        p, s = i_to_ps(i)
        nbrs = [ps_to_i((p+1)%N_PLANES, s),
                ps_to_i((p-1)%N_PLANES, s),
                ps_to_i(p, (s+1)%N_SLOTS),
                ps_to_i(p, (s-1)%N_SLOTS)]
        valid = [n for n in nbrs if good0[n]]
        if valid:
            sat_pos_raw[i] = np.mean([sat_pos_raw[n] for n in valid],
                                     axis=0)
            good0[i] = True
sat_positions = sat_pos_raw
print(f"Valid sats after prop: {good0.sum()}/{Ns}")

adj, dist = build_plus_grid_and_dist(sat_positions)
deg = adj.sum(axis=1)
upper = np.triu(adj,1)
isl_edges = list(zip(*np.where(upper>0)))
MAX_ISL_KM = float(max(dist[i,j] for i,j in isl_edges))
print(f"+Grid: avg_deg={deg.mean():.3f}, |E|={len(isl_edges)}, "
      f"max_ISL={MAX_ISL_KM:.0f} km")

# ============================================================
# F) GATEWAYS
# ============================================================
layers     = gpd.list_layers(KML_FILE)
use_layers = [ln for ln in layers["name"]
              if ln not in ["PoPs & Backbone",
                            "Community Gateway satellite links"]]
all_rows = []
for ln in use_layers:
    try:
        gdf = gpd.read_file(KML_FILE, layer=ln, driver="KML")
        if len(gdf): all_rows.append(gdf[["Name","geometry"]])
    except Exception: pass
gateways = pd.concat(all_rows, ignore_index=True)
gateways = gateways[
    gateways["geometry"].apply(lambda g: isinstance(g,Point))].copy()
gateways["longitude"] = gateways.geometry.x
gateways["latitude"]  = gateways.geometry.y
mega = gateways[
    gateways["Name"].str.contains(r"\|\s*Mega",regex=True,na=False)].copy()
mega = mega[~mega["Name"].str.contains("Community",case=False,na=False)]
mega = mega.drop_duplicates(subset=["latitude","longitude"]).reset_index(drop=True)
mega["City"] = mega["Name"].apply(
    lambda n: n.split("|")[0].split(",")[0].split(" - ")[0].strip())
mega = mega.drop_duplicates(subset=["City"]).reset_index(drop=True)
gateway_positions = np.vstack([
    ecef_from_latlon(lat,lon,0.)
    for lat,lon in zip(mega["latitude"],mega["longitude"])])
Ng = gateway_positions.shape[0]
Cg = np.ones(Ng,dtype=float)*Cg_default
Ng_full = Ng                              # full deployment size (for Ng sweep)
gateway_positions_full = gateway_positions.copy()
print(f"Mega gateways: {Ng}, total GW cap = {Ng*Cg_default:.0f} Gbps")

def farthest_point_gateways(pos, k):
    """Deterministic max-min (farthest-point) subset of k gateways: keeps the
    subset geographically spread so shrinking Ng widens inter-gateway gaps
    rather than clustering. Returns sorted original-index array."""
    n = pos.shape[0]
    k = int(min(k, n))
    if k >= n:
        return np.arange(n, dtype=int)
    chosen = [0]                                  # fixed start -> reproducible
    d = np.linalg.norm(pos - pos[0], axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(pos - pos[nxt], axis=1))
    return np.array(sorted(chosen), dtype=int)

# ============================================================
# G) TRAFFIC (Poisson population model)
# ============================================================
import tifffile

R_COV_KM      = 400.0
ACTIVITY_RATE = 0.00003
DEMAND_GBPS   = 0.1

def ecef_to_latlon(xyz):
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    lon_deg = math.degrees(math.atan2(y, x))
    p = math.sqrt(x*x + y*y)
    lat_deg = math.degrees(math.atan2(z, p*(1-_e2)))
    for _ in range(5):
        sl = math.sin(math.radians(lat_deg))
        N  = _a / math.sqrt(1 - _e2*sl*sl)
        lat_deg = math.degrees(math.atan2(z + _e2*N*sl, p))
    return lat_deg, lon_deg

print(f"\nLoading population raster...")
pop_arr = tifffile.imread(POP_TIF).astype(np.float32)
if pop_arr.ndim == 3:
    pop_arr = pop_arr[0]
pop_arr = np.where(pop_arr < 0, 0., pop_arr)
nrows, ncols = pop_arr.shape

LON_ORIGIN = -180.0
LAT_ORIGIN =   90.0
PIXEL_DEG  = 30.0 / 3600.0
DEG_PER_KM = 1.0 / 111.32

def lon_deg_per_km(lat):
    cl = math.cos(math.radians(lat))
    return 0. if cl < 1e-6 else 1.0 / (111.32 * cl)

def compute_pop_per_sat(sat_pos):
    """Compute per-satellite population from raster; returns array of length Ns."""
    pop = np.zeros(Ns, dtype=np.float64)
    for i in range(Ns):
        lat, lon = ecef_to_latlon(sat_pos[i])
        dlat = R_COV_KM * DEG_PER_KM
        dlon = R_COV_KM * lon_deg_per_km(lat)
        r0 = int((LAT_ORIGIN - (lat+dlat)) / PIXEL_DEG)
        r1 = int((LAT_ORIGIN - (lat-dlat)) / PIXEL_DEG) + 1
        c0 = int((lon-dlon - LON_ORIGIN)   / PIXEL_DEG)
        c1 = int((lon+dlon - LON_ORIGIN)   / PIXEL_DEG) + 1
        r0 = max(0, r0); r1 = min(nrows, r1)
        c0 = max(0, c0); c1 = min(ncols, c1)
        if r0 >= r1 or c0 >= c1: continue
        patch    = pop_arr[r0:r1, c0:c1]
        lons_p   = LON_ORIGIN + (np.arange(c0,c1) + 0.5) * PIXEL_DEG
        lats_p   = LAT_ORIGIN - (np.arange(r0,r1) + 0.5) * PIXEL_DEG
        LON_G, LAT_G = np.meshgrid(lons_p, lats_p)
        dlat_g   = (LAT_G - lat) / DEG_PER_KM
        dkm      = 1.0 / max(lon_deg_per_km(lat), 1e-9)
        dlon_g   = (LON_G - lon) * dkm
        dist_g   = np.sqrt(dlat_g**2 + dlon_g**2)
        pop[i] = float(np.sum(patch[dist_g <= R_COV_KM]))
    return pop

pop_per_sat = compute_pop_per_sat(sat_positions)
del pop_arr

print(f"Pop stats: sats_with_pop={(pop_per_sat>0).sum()}/{Ns}, "
      f"mean={pop_per_sat.mean():,.0f}, max={pop_per_sat.max():,.0f}")

np.random.seed(rng_seed)
N_users  = np.random.poisson(ACTIVITY_RATE * pop_per_sat)
rho_raw  = N_users.astype(float) * DEMAND_GBPS
rho_base = np.clip(rho_raw, rho_min, rho_max)
rho_base[pop_per_sat == 0] = rho_min

print(f"Base traffic (scale=1.0): sum={rho_base.sum():.1f} Gbps")

# ---- Hotspot (flash-crowd) helpers ----
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2.0*R*math.asin(math.sqrt(a))

_sat_latlon = np.array([ecef_to_latlon(sat_positions[i]) for i in range(Ns)])

# ============================================================
# MULTICOMMODITY: traffic destinations + gateway reachability
# ------------------------------------------------------------
# Single-commodity ("any gateway will do") makes this a max-flow problem that
# successive-shortest-path solves EXACTLY, which is why dual pricing could not
# improve on greedy. Real traffic has DESTINATIONS: a session terminates at a
# specific terrestrial region (data centre / IXP / peering point), so it must
# egress at a gateway with connectivity to that region. Flows with different
# destinations then compete for the same ISLs while needing different exits --
# a multicommodity flow, which has no greedy optimum and whose standard
# solution method IS Lagrangian decomposition with link prices (lambda, nu).
#
# Regions are geographic (fixed by longitude/latitude bands), NOT tuned.
REGION_NAMES = ["Americas", "EMEA", "APAC"]
NREG = len(REGION_NAMES)

def _region_of(lat, lon):
    if -170.0 <= lon < -30.0:  return 0        # Americas
    if -30.0  <= lon <  60.0:  return 1        # Europe/Middle East/Africa
    return 2                                   # Asia-Pacific

# Gateway -> which destination regions it can egress traffic for.
# A gateway always serves its OWN region; it also serves other regions only if
# it is a major hub (has terrestrial long-haul capacity). We approximate hubs
# as gateways in the top HUB_FRACTION by population served nearby -- geography,
# not tuning. Non-hub gateways are region-local only.
HUB_FRACTION = 0.35
_gw_latlon = np.array([ecef_to_latlon(gateway_positions[g]) for g in range(Ng)])
_gw_region = np.array([_region_of(_gw_latlon[g,0], _gw_latlon[g,1])
                       for g in range(Ng)], dtype=int)
# hub score = population within 1500 km of the gateway (from the sat footprints)
_gw_pop = np.zeros(Ng)
for g in range(Ng):
    d = np.array([_haversine_km(_sat_latlon[i,0], _sat_latlon[i,1],
                                _gw_latlon[g,0], _gw_latlon[g,1])
                  for i in range(Ns)])
    _gw_pop[g] = float(pop_per_sat[d <= 1500.0].sum())
_hub_thr = np.quantile(_gw_pop, 1.0 - HUB_FRACTION)
GW_IS_HUB = _gw_pop >= _hub_thr

GW_SERVES = np.zeros((Ng, NREG), dtype=bool)
for g in range(Ng):
    GW_SERVES[g, _gw_region[g]] = True         # always serves its own region
    if GW_IS_HUB[g]:
        GW_SERVES[g, :] = True                 # hubs have long-haul to all

# Satellite traffic destination: most traffic is regional, some inter-region.
# Split by a fixed rule (not tuned): a satellite's demand is assigned to a
# destination region drawn with 60% local / 40% split across the others.
np.random.seed(rng_seed + 101)
_sat_region = np.array([_region_of(_sat_latlon[i,0], _sat_latlon[i,1])
                        for i in range(Ns)], dtype=int)
DST = np.zeros(Ns, dtype=int)
for i in range(Ns):
    if np.random.rand() < 0.60:
        DST[i] = _sat_region[i]
    else:
        others = [r for r in range(NREG) if r != _sat_region[i]]
        DST[i] = int(np.random.choice(others))

print(f"\nMulticommodity: {NREG} destination regions {REGION_NAMES}")
print(f"  gateways per region: "
      f"{[int((_gw_region==r).sum()) for r in range(NREG)]}, "
      f"hubs (serve all regions) = {int(GW_IS_HUB.sum())}/{Ng}")
print(f"  satellites per destination: "
      f"{[int((DST==r).sum()) for r in range(NREG)]}")
print(f"  gateways usable per destination: "
      f"{[int(GW_SERVES[:,r].sum()) for r in range(NREG)]}")
def hotspot_sats(center_lat, center_lon, radius_km):
    d = np.array([_haversine_km(_sat_latlon[i,0], _sat_latlon[i,1],
                                center_lat, center_lon) for i in range(Ns)])
    return np.where(d <= radius_km)[0]

rho = rho_base.copy()  # working array, overwritten per sweep point
# ============================================================
# H) VISIBILITY + ISL LIFETIMES (functions for multi-snapshot)
# ============================================================
def build_visibility(pos, elev_deg):
    thr = np.deg2rad(float(elev_deg))
    V   = np.zeros((pos.shape[0],Ng),dtype=np.uint8)
    for i in range(pos.shape[0]):
        for g in range(Ng):
            if elevation_angle(pos[i],gateway_positions[g]) >= thr:
                V[i,g]=1
    return V

def build_lifetimes(base_t, adj_l, max_km, dt=20., horizon=1200.):
    lif   = np.full((Ns,Ns),np.nan,dtype=float)
    steps = int(math.ceil(horizon/dt))
    times = [base_t+timedelta(seconds=(k+1)*dt) for k in range(steps)]
    pos_steps=[]
    for t in times:
        jd,fr = datetime_to_jd_fr(t)
        Rec   = R3(gmst_rad(jd,fr))
        pp    = np.zeros((len(satrecs),3))
        for i2,sat in enumerate(satrecs):
            e,r,v = sat.sgp4(jd,fr)
            pp[i2]= Rec@np.array(r,dtype=float) if e==0 else np.nan
        pos_steps.append(pp)
    edges = [(i,j) for i in range(Ns)
             for j in range(i+1,Ns) if adj_l[i,j]>0]
    for i,j in edges:
        L=horizon
        for k in range(steps):
            Pi,Pj=pos_steps[k][i],pos_steps[k][j]
            if not (np.isfinite(Pi).all() and np.isfinite(Pj).all()):
                L=(k+1)*dt; break
            if np.linalg.norm(Pi-Pj)>max_km:
                L=(k+1)*dt; break
        lif[i,j]=lif[j,i]=float(L)
    return lif

# Initial build at prop_time (snapshot 0)
print("\nBuilding ISL lifetimes...")
LIFETIME_MAX_KM = 1.5 * MAX_ISL_KM
isl_lifetime_s = build_lifetimes(prop_time, adj, LIFETIME_MAX_KM,
                                  LIFE_DT_S, LIFE_HORIZON_S)
V_fixed = build_visibility(sat_positions, ELEV_FIXED)
print(f"Visibility @ elev={ELEV_FIXED}deg: "
      f"{(V_fixed.sum(axis=1)>0).sum()}/{Ns} sats visible")

# ---- Realistic hotspot siting: densest region that must backhaul -----------
# Choose the hotspot center from the DATA: among "dense" satellites (population
# at/above HOTSPOT_POP_PCTILE), prefer those with NO directly-visible gateway
# (forced to backhaul); among those, take the one farthest from any gateway.
# This lands the crowd over a genuinely populated but gateway-sparse area --
# the realistic backhaul-stress scenario -- instead of an arbitrary ocean point.
if RUN_HOTSPOT and HOTSPOT_CENTER is None:
    _gwmin = np.array([min(sat_gw_km(i, g) for g in range(Ng))
                       for i in range(Ns)])           # nearest-gateway distance
    _pos_pop = pop_per_sat[pop_per_sat > 0]
    _pop_thr = float(np.percentile(_pos_pop, HOTSPOT_POP_PCTILE)) \
               if len(_pos_pop) else 0.0
    _dense = np.where(pop_per_sat >= _pop_thr)[0]
    _dense = _dense[pop_per_sat[_dense] > 0]
    _novis = np.array([i for i in _dense if V_fixed[i].sum() == 0], dtype=int)
    _pool = _novis if len(_novis) else _dense          # prefer must-backhaul
    _best = int(_pool[np.argmax(_gwmin[_pool])])       # farthest from a gateway
    HOTSPOT_CENTER = (float(_sat_latlon[_best, 0]), float(_sat_latlon[_best, 1]))
    print(f"\nHotspot sited at ({HOTSPOT_CENTER[0]:.1f}, {HOTSPOT_CENTER[1]:.1f}): "
          f"pop={pop_per_sat[_best]:,.0f}, "
          f"nearest gateway {_gwmin[_best]:.0f} km, "
          f"directly-visible gateways={int(V_fixed[_best].sum())} "
          f"(dense pctile>={HOTSPOT_POP_PCTILE}, pool={'no-vis' if len(_novis) else 'dense'})")

# ============================================================
# MULTI-SNAPSHOT: rebuild time-dependent quantities
# ============================================================
def rebuild_at_snapshot(delta_min):
    """Repropagate satellites and rebuild time-dependent quantities
    at prop_time + delta_min minutes. Updates module globals so that
    subsequent scheme functions see the new snapshot.

    delta_min=0 restores the base snapshot (no-op if already there).
    """
    global sat_positions, adj, dist, isl_edges, MAX_ISL_KM
    global isl_lifetime_s, V_fixed, pop_per_sat, rho_base, rho, NBR_LIST

    t_new = prop_time + timedelta(minutes=float(delta_min))
    print(f"[snapshot] rebuilding at t = {t_new.isoformat()}")

    # 1. propagate
    sp_new, good_new = prop_positions(t_new, satrecs)
    if not good_new.all():
        for i in np.where(~good_new)[0]:
            p, s = i_to_ps(i)
            nbrs_ = [ps_to_i((p+1)%N_PLANES, s),
                     ps_to_i((p-1)%N_PLANES, s),
                     ps_to_i(p, (s+1)%N_SLOTS),
                     ps_to_i(p, (s-1)%N_SLOTS)]
            valid = [n for n in nbrs_ if good_new[n]]
            if valid:
                sp_new[i] = np.mean([sp_new[n] for n in valid], axis=0)
                good_new[i] = True
    sat_positions = sp_new

    # 2. rebuild +Grid adjacency + distances
    adj, dist = build_plus_grid_and_dist(sat_positions)
    upper_new = np.triu(adj,1)
    isl_edges = list(zip(*np.where(upper_new>0)))
    MAX_ISL_KM = float(max(dist[i,j] for i,j in isl_edges))

    # 3. rebuild ISL lifetimes
    isl_lifetime_s = build_lifetimes(t_new, adj,
                                     1.5 * MAX_ISL_KM,
                                     LIFE_DT_S, LIFE_HORIZON_S)

    # 4. rebuild visibility at ELEV_FIXED
    V_fixed = build_visibility(sat_positions, ELEV_FIXED)

    # 5. rebuild neighbor list
    NBR_LIST = [list(np.where(adj[i]>0)[0]) for i in range(Ns)]

    # 6. re-derive traffic (population may have changed slightly if
    # ground footprint shifted; we resample Poisson with seed rng_seed
    # so that flow is deterministic within a snapshot)
    # NOTE: pop_arr already freed; skip re-derivation of rho_base if
    # you want same traffic profile across snapshots. Here we keep
    # rho_base fixed to isolate topology effects.
    # (Uncomment below to re-derive per snapshot)
    # pop_per_sat = compute_pop_per_sat(sat_positions)
    # np.random.seed(rng_seed)
    # N_users_ = np.random.poisson(ACTIVITY_RATE * pop_per_sat)
    # rho_base_ = np.clip(N_users_.astype(float) * DEMAND_GBPS,
    #                     rho_min, rho_max)
    # rho_base_[pop_per_sat == 0] = rho_min
    # rho_base = rho_base_

    print(f"  |E|={len(isl_edges)}, MAX_ISL={MAX_ISL_KM:.0f} km, "
          f"visible@{ELEV_FIXED}deg={(V_fixed.sum(axis=1)>0).sum()}")

# ============================================================
# I) SHARED HELPERS
# ============================================================
def edge_key(i,j): return (int(min(i,j)),int(max(i,j)))
def isl_prop_s(i,j): return float(dist[i,j]/c_light)
def sg_prop_s(s,g):
    return float(np.linalg.norm(
        sat_positions[s]-gateway_positions[g])/c_light)
def sat_gw_km(s,g):
    return float(np.linalg.norm(
        sat_positions[s]-gateway_positions[g]))

def nearest_vis_gw(V,s):
    vis=np.where(V[s]>0)[0]
    if not len(vis): return None
    return int(vis[np.argmin([sat_gw_km(s,g) for g in vis])])

def exit_sats(V,g): return np.where(V[:,g]>0)[0]
def any_exit(V):    return np.where(V.sum(axis=1)>0)[0]

NBR_LIST = [list(np.where(adj[i]>0)[0]) for i in range(Ns)]

def dijkstra(start, targets, wfunc):
    pq    = [(0.,int(start))]
    dv    = {int(start):0.}
    par   = {int(start):-1}
    tset  = set(int(t) for t in targets)
    hit   = None
    while pq:
        du,u = heapq.heappop(pq)
        if du > dv[u]+1e-15: continue
        if u in tset: hit=u; break
        for v in NBR_LIST[u]:
            v=int(v); nd=du+float(wfunc(u,v))
            if v not in dv or nd < dv[v]-1e-15:
                dv[v]=nd; par[v]=u
                heapq.heappush(pq,(nd,v))
    if hit is None: return None, None
    path=[int(hit)]
    while par[path[-1]]!=-1: path.append(int(par[path[-1]]))
    path.reverse(); return path, hit

def full_delay(path, g, gwl, isll, include_q=True):
    if path is None or g is None or g<0: return np.nan
    tot=0.
    for a,b in zip(path[:-1],path[1:]):
        e=edge_key(a,b)
        q=(queue_delay(float(isll.get(e,0.)),Cisl)
           if include_q else 0.)
        tot += isl_prop_s(a,b)+D_TX_ISL+D_PROC_ISL+q
    qg=(queue_delay(float(gwl[g]),float(Cg[g]))
        if include_q else 0.)
    tot += sg_prop_s(path[-1],g)+D_TX_SG+D_PROC_SG+qg
    return float(tot)

def check_caps(gwl, isll):
    gwo = np.maximum(np.asarray(gwl,dtype=float)-Cg,0.)
    mgw = float(np.max(gwo)) if np.any(gwo>1e-12) else 0.
    mis = 0.
    for e,l in isll.items():
        ov=float(l-Cisl)
        if ov>1e-12: mis=max(mis,ov)
    return mgw, mis

def can_admit(path,g,flow,gwl,isll):
    if g is None or g<0: return False
    if gwl[g]+flow > Cg[g]+1e-12: return False
    for a,b in zip(path[:-1],path[1:]):
        if isll[edge_key(a,b)]+flow > Cisl+1e-12: return False
    return True

def admit(path,g,flow,gwl,isll):
    gwl[g]+=flow
    for a,b in zip(path[:-1],path[1:]):
        isll[edge_key(a,b)]+=flow

def path_headroom(path, g, gwl, isll):
    """Max flow admissible on this path+gateway given current loads.
    Bottleneck = min(gateway headroom, min ISL headroom along the path)."""
    if g is None or g < 0:
        return 0.0
    head = max(Cg[g] - gwl[g], 0.0)
    for a, b in zip(path[:-1], path[1:]):
        head = min(head, max(Cisl - isll[edge_key(a, b)], 0.0))
    return float(head)

def admit_partial(path, g, flow, gwl, isll):
    """Admit min(flow, headroom) on this path+gateway, commit the load,
    and return the amount actually admitted. The overflow is simply
    dropped (single-path partial admission; no re-routing of remainder)."""
    take = min(float(flow), path_headroom(path, g, gwl, isll))
    if take <= 1e-12:
        return 0.0
    gwl[g] += take
    for a, b in zip(path[:-1], path[1:]):
        isll[edge_key(a, b)] += take
    return float(take)

def _commit_choice(s, flow, cand_list, gwl, isll,
                   paths, cgw, hc, served_flow):
    """Uniform partial-admission commit used by every scheme.
    cand_list: list of (cost, path, g), each already path-valid and
    kappa-passed. Prefer the lowest-cost candidate that fully fits; if
    none fits, partial-admit the lowest-cost candidate that has ANY
    headroom (so an over-capacity flow still contributes what it can
    instead of being rejected in full). Records paths/cgw/hc/served_flow
    (fd is filled by the caller over final loads). Returns amount admitted."""
    if not cand_list:
        return 0.0
    cand_list.sort(key=lambda x: x[0])
    full = None; part = None
    for _c, path, g in cand_list:
        if part is None and path_headroom(path, g, gwl, isll) > 1e-12:
            part = (path, g)
        if can_admit(path, g, flow, gwl, isll):
            full = (path, g); break
    use = full if full is not None else part
    if use is None:
        return 0.0
    path, g = use
    take = admit_partial(path, g, flow, gwl, isll)
    if take <= 1e-12:
        return 0.0
    paths[s] = [int(x) for x in path]
    cgw[s]   = int(g)
    hc[s]    = max(len(path) - 1, 0)
    served_flow[s] = take
    return take

def is_path_valid(path, g):
    if path is None or g is None or g < 0:
        return False
    d = 0.0
    for a, b in zip(path[:-1], path[1:]):
        d += isl_prop_s(a, b) + D_TX_ISL + D_PROC_ISL
        L_ij = float(isl_lifetime_s[a, b])
        if L_ij < L_MIN:
            return False
    d += sg_prop_s(path[-1], g) + D_TX_SG + D_PROC_SG
    if d > D_MAX:
        return False
    return True

# Per-scheme cost functions (for kappa gate)
def cost_dsp(path, g):
    c = 0.0
    for a, b in zip(path[:-1], path[1:]):
        c += isl_prop_s(a, b) + D_TX_ISL + D_PROC_ISL
    c += sg_prop_s(path[-1], g) + D_TX_SG + D_PROC_SG
    return c

def cost_cadsp(path, g, gwl, isll):
    c = 0.0
    for a, b in zip(path[:-1], path[1:]):
        c += (isl_prop_s(a, b) + D_TX_ISL + D_PROC_ISL
              + CADSP_ISL_PEN * (isll[edge_key(a, b)] / max(Cisl, 1e-12)))
    c += (sg_prop_s(path[-1], g) + D_TX_SG + D_PROC_SG
          + CADSP_GW_PEN * (gwl[g] / max(Cg[g], 1e-12)))
    return c

def cost_dbpr_for_kappa(path, g):
    c = 0.0
    for a, b in zip(path[:-1], path[1:]):
        c += isl_prop_s(a, b) + D_TX_ISL + D_PROC_ISL
    c += sg_prop_s(path[-1], g) + D_TX_SG + D_PROC_SG
    return c

def cost_gpsr(path, g):
    c = 0.0
    for a, b in zip(path[:-1], path[1:]):
        c += isl_prop_s(a, b) + D_TX_ISL + D_PROC_ISL
    c += sg_prop_s(path[-1], g) + D_TX_SG + D_PROC_SG
    return c

def cost_directdrop(s, g):
    return sg_prop_s(s, g) + D_TX_SG + D_PROC_SG

# Adaptive eps-feasibility thresholds
def eps_feas_thresholds(util):
    frac = EPS_MIN_FRAC + (EPS_MAX_FRAC - EPS_MIN_FRAC) * float(
        np.clip(util - 1.0, 0.0, 1.0))
    eps_gw = frac * float(CG_EPS_REF)      # was frac*mean(Cg) (Cg-coupled)
    eps_isl = frac * float(CISL_EPS_REF)   # was frac*Cisl (Cisl-coupled)
    return eps_gw, eps_isl

# ============================================================
# I2) FEASIBILITY UPPER BOUND
# ============================================================
def feasibility_upper_bound(V):
    # VALID upper bound via SINGLE-COMMODITY max-flow. With multicommodity
    # destinations this is a RELAXATION bound: it ignores the gateway-serves-
    # destination restriction, which can only increase servable flow, so it
    # remains a legitimate upper bound -- but a LOOSER one. Gaps to this bound
    # are therefore not tight under multicommodity; compare VARIANTS to each
    # other, not to the bound.
    # Relaxes per-flow routing, the delay budget, and link-lifetime (all of
    # which only REDUCE servable flow), so max-flow(source->sink) >= whatever
    # any scheme can serve. It still respects ISL-capacity and gateway-capacity
    # cuts, so unlike the old greedy heuristic it is a genuine bound AND tight
    # where the backhaul mesh (not just the gateways) is the bottleneck.
    #   nodes: 0=source | 1..Ns=sats | Ns+1..Ns+Ng=gateways | Ns+Ng+1=sink
    # (undirected ISL modelled as two Cisl arcs -> slightly loose but still a
    #  valid over-approximation, which is all an upper bound requires.)
    import collections
    S = 0; SINK = Ns + Ng + 1; Nn = Ns + Ng + 2
    graph = [dict() for _ in range(Nn)]
    def add(u, v, c):
        graph[u][v] = graph[u].get(v, 0.) + float(c)
        graph[v].setdefault(u, 0.)
    for i in range(Ns):
        if rho[i] > 0: add(S, 1+i, rho[i])                 # can't serve > demand
    for i in range(Ns):
        for j in NBR_LIST[i]:
            j = int(j)
            if j > i:
                add(1+i, 1+j, Cisl); add(1+j, 1+i, Cisl)   # ISL capacity
    for i in range(Ns):
        for g in np.where(V[i] > 0)[0]:
            add(1+i, 1+Ns+int(g), Cg[int(g)])              # downlink
    for g in range(Ng):
        add(1+Ns+g, SINK, Cg[g])                           # gateway capacity

    nbrs = [list(graph[u].keys()) for u in range(Nn)]
    def bfs():
        level = [-1]*Nn; level[S] = 0
        q = collections.deque([S])
        while q:
            u = q.popleft()
            for v, c in graph[u].items():
                if c > 1e-9 and level[v] < 0:
                    level[v] = level[u]+1; q.append(v)
        return level if level[SINK] >= 0 else None
    def dfs(u, f, level, it):
        if u == SINK: return f
        while it[u] < len(nbrs[u]):
            v = nbrs[u][it[u]]; c = graph[u][v]
            if c > 1e-9 and level[v] == level[u]+1:
                d = dfs(v, min(f, c), level, it)
                if d > 1e-9:
                    graph[u][v] -= d; graph[v][u] += d; return d
            it[u] += 1
        return 0.
    flow = 0.
    while True:
        level = bfs()
        if level is None: break
        it = [0]*Nn
        while True:
            f = dfs(S, float('inf'), level, it)
            if f <= 1e-9: break
            flow += f
    return float(flow), float(np.sum(rho))

# ============================================================
# J) KPI PACKER
# ============================================================
def pack_kpi(scheme, paths, chosen_gw, hop_count,
             gwl, isll, fd, served_flow=None):
    rho_l  = np.asarray(rho,dtype=float)
    ridx   = np.array(sorted(paths.keys()),dtype=int)
    offered= float(np.sum(rho_l))
    # Served amount per admitted satellite: partial if served_flow given,
    # else full rho (schemes that did not pass served_flow keep old
    # all-or-nothing accounting).
    if served_flow is not None:
        served_r = np.array([float(served_flow.get(int(s), 0.0))
                             for s in ridx]) if len(ridx) else np.zeros(0)
    else:
        served_r = rho_l[ridx] if len(ridx) else np.zeros(0)
    served = float(np.sum(served_r))
    plr    = 1.-served/offered if offered>0 else 0.
    mgw,mis= check_caps(gwl,isll)
    feas   = (mgw<=1e-12) and (mis<=1e-12)
    dv_all = fd[ridx] if len(ridx) else np.zeros(0)
    _fin   = np.isfinite(dv_all)
    dv     = dv_all[_fin]
    _dw    = served_r[_fin] if len(served_r)==len(dv_all) else None
    hv     = hop_count[ridx] if len(ridx) else np.zeros(0)
    avg_hops = float(np.average(hv,weights=served_r)) \
               if (len(hv) and served_r.sum()>0) else 0.
    # Traffic-weighted, to match the delay the dual actually optimises.
    # An unweighted mean over satellites weighted every satellite equally
    # regardless of how much traffic it carried, so the reported avgD did not
    # match best_avgD from the scoring pass.
    if len(dv) and _dw is not None and float(np.sum(_dw)) > 0:
        avg_d = float(np.average(dv, weights=_dw))
    else:
        avg_d = float(np.mean(dv)) if len(dv) else np.nan
    p95_d    = float(np.percentile(dv,95)) if len(dv) else np.nan
    gwl_arr  = np.asarray(gwl, dtype=float)
    Cg_arr   = np.asarray(Cg,  dtype=float)
    gw_util  = gwl_arr / np.maximum(Cg_arr, 1e-12)
    gw_over  = float(np.sum(np.maximum(gwl_arr - Cg_arr, 0.)))
    gw_u_max = float(np.max(gw_util)) if len(gw_util) else np.nan
    gw_q = np.array([
        queue_delay(float(gwl[int(chosen_gw[s])]),
                    float(Cg[int(chosen_gw[s])]))
        if int(chosen_gw[s]) >= 0 else np.nan
        for s in ridx
    ]) if len(ridx) else np.zeros(0)
    gw_q = gw_q[np.isfinite(gw_q)]
    p95_gwq  = float(np.percentile(gw_q, 95)) if len(gw_q) else np.nan
    isl_loads = np.array(list(isll.values()), dtype=float) \
                if isll else np.zeros(1)
    isl_u_max = float(np.max(isl_loads) / max(Cisl, 1e-12))
    isl_over  = float(np.sum(np.maximum(isl_loads - Cisl, 0.)))
    # Fraction of USED ISL links running above 90% capacity -- a real
    # congestion signal (max utilisation is pinned at 1.0 by any single
    # saturated funnel link and says nothing about how widespread it is).
    used = isl_loads[isl_loads > 1e-9]
    isl_busy = float(np.mean(used/max(Cisl,1e-12) > 0.90)) if len(used) else 0.
    if len(used):
        qd = np.array([queue_delay(float(l), Cisl) for l in used])
        isl_q_mean = float(np.mean(qd)); isl_q_p95 = float(np.percentile(qd,95))
    else:
        isl_q_mean = isl_q_p95 = 0.
    U_ISL = float(np.sum(served_r * hv)) if len(ridx) else 0.
    return dict(
        scheme=scheme, feas=bool(feas),
        routed=int(len(ridx)),
        offered_traffic=offered, served_traffic=served,
        served_ratio=float(served/offered) if offered>0 else 0.,
        packet_loss_ratio=float(plr),
        avgD_s=avg_d, p95D_s=p95_d,
        avg_hops=avg_hops, U_ISL=U_ISL,
        maxGW=float(mgw), maxISL=float(mis),
        gw_util_max=gw_u_max, gw_overload=gw_over,
        p95_gw_queue=p95_gwq,
        isl_util_max=isl_u_max, isl_overload=isl_over,
        isl_busy_frac=isl_busy,
        isl_qdelay_mean=isl_q_mean, isl_qdelay_p95=isl_q_p95,
        # per-satellite (served, delay) -- popped before the DataFrame is built.
        # Used to compare delay on the COMMON subset of satellites that every
        # variant served, which removes the traffic-mix confound: a variant
        # that drops far-from-gateway flows reports a low average simply
        # because it kept only the easy traffic.
        _persat={int(s_): (float(served_flow.get(int(s_), 0.0)
                                 if served_flow is not None else rho_l[int(s_)]),
                           float(fd[int(s_)]))
                 for s_ in ridx if np.isfinite(fd[int(s_)])},
    )
# ============================================================
# K) SCHEME FUNCTIONS
# ============================================================

# ---- K1) Direct Drop ----
def run_direct_drop(V):
    paths={}; cgw=-np.ones(Ns,int); hc=np.zeros(Ns,int)
    gwl=np.zeros(Ng); isll=defaultdict(float); served_flow={}
    for s in range(Ns):
        g=nearest_vis_gw(V,s)
        if g is None: continue
        if not is_path_valid([s], g): continue
        if cost_directdrop(s, g) > KAPPA: continue
        _commit_choice(s, float(rho[s]),
                       [(cost_directdrop(s, int(g)), [s], int(g))],
                       gwl, isll, paths, cgw, hc, served_flow)
    fd=np.full(Ns,np.nan)
    for s,p in paths.items(): fd[s]=full_delay(p,cgw[s],gwl,isll)
    return pack_kpi("Direct Drop",paths,cgw,hc,gwl,isll,fd,
                    served_flow=served_flow)

# ---- K2) DSP ----
def run_dsp(V):
    paths={}; cgw=-np.ones(Ns,int); hc=np.zeros(Ns,int)
    gwl=np.zeros(Ng); isll=defaultdict(float); served_flow={}
    cands=np.where(V.sum(axis=0)>0)[0]
    for s in range(Ns):
        flow=float(rho[s]); opts=[]
        for g in cands:
            ex=exit_sats(V,g)
            if not len(ex): continue
            wf=lambda u,v: isl_prop_s(u,v)+D_TX_ISL+D_PROC_ISL
            path,_=dijkstra(s,ex,wf)
            if path is None: continue
            if not is_path_valid(path, g): continue
            cost=cost_dsp(path, g)
            if cost > KAPPA: continue
            opts.append((cost,[int(x) for x in path],int(g)))
        _commit_choice(s, flow, opts, gwl, isll,
                       paths, cgw, hc, served_flow)
    fd=np.full(Ns,np.nan)
    for s,p in paths.items(): fd[s]=full_delay(p,cgw[s],gwl,isll)
    return pack_kpi("DSP",paths,cgw,hc,gwl,isll,fd,served_flow=served_flow)

# ---- K3) CA-DSP ----
def run_cadsp(V):
    paths={}; cgw=-np.ones(Ns,int); hc=np.zeros(Ns,int)
    gwl=np.zeros(Ng); isll=defaultdict(float); served_flow={}
    cands=np.where(V.sum(axis=0)>0)[0]
    order=np.argsort(-np.asarray(rho,dtype=float))
    for s in order:
        flow=float(rho[s]); opts=[]
        for g in cands:
            ex=exit_sats(V,g)
            if not len(ex): continue
            def wf(u,v,_g=g):
                return (isl_prop_s(u,v)+D_TX_ISL+D_PROC_ISL
                        +CADSP_ISL_PEN*(isll[edge_key(u,v)]/max(Cisl,1e-12)))
            path,_=dijkstra(s,ex,wf)
            if path is None: continue
            if not is_path_valid(path, g): continue
            cost=(sum(isl_prop_s(a,b)+D_TX_ISL+D_PROC_ISL
                      +CADSP_ISL_PEN*(isll[edge_key(a,b)]/max(Cisl,1e-12))
                      for a,b in zip(path[:-1],path[1:]))
                  +sg_prop_s(path[-1],g)+D_TX_SG+D_PROC_SG
                  +CADSP_GW_PEN*(gwl[g]/max(Cg[g],1e-12)))
            if cost > KAPPA: continue
            opts.append((cost,[int(x) for x in path],int(g)))
        _commit_choice(s, flow, opts, gwl, isll,
                       paths, cgw, hc, served_flow)
    fd=np.full(Ns,np.nan)
    for s,p in paths.items(): fd[s]=full_delay(p,cgw[s],gwl,isll)
    return pack_kpi("CA-DSP",paths,cgw,hc,gwl,isll,fd,served_flow=served_flow)

# ---- K4) DBPR ----
def run_dbpr(V):
    cgw=-np.ones(Ns,int); hc=np.zeros(Ns,int)
    gwl=np.zeros(Ng); isll=defaultdict(float); paths={}
    if not len(any_exit(V)):
        return pack_kpi("DBPR",{},cgw,hc,gwl,isll,np.full(Ns,np.nan))
    sq=np.zeros(Ns)
    for _ in range(DBPR_ITERS):
        paths={}; cgw[:]=-1; hc[:]=0
        gwl[:]=0.; isll=defaultdict(float)
        for s in range(Ns):
            vis=np.where(V[s]>0)[0]
            if len(vis):
                vis_ok = [int(g) for g in vis if is_path_valid([s], int(g))]
                if not vis_ok: continue
                bg=int(min(vis_ok,key=lambda g:
                    DBPR_ALPHA_Q*gwl[g]+DBPR_BETA_DIST*sat_gw_km(s,g)))
                paths[s]=[s]; cgw[s]=bg; gwl[bg]+=rho[s]; continue
            cands=np.where(V.sum(axis=0)>0)[0]
            bc=np.inf; bp=None; bg=None
            for g in cands:
                ex=exit_sats(V,g)
                if not len(ex): continue
                def wf(u,v):
                    return (DBPR_ALPHA_Q*(sq[v]+isll[edge_key(u,v)])
                            +DBPR_BETA_DIST*float(dist[u,v]))
                path,_=dijkstra(s,ex,wf)
                if path is None: continue
                if not is_path_valid(path, g): continue
                cost=(sum(DBPR_ALPHA_Q*(sq[b]+isll[edge_key(a,b)])
                          +DBPR_BETA_DIST*float(dist[a,b])
                          for a,b in zip(path[:-1],path[1:]))
                      +DBPR_ALPHA_Q*gwl[g]
                      +DBPR_BETA_DIST*sat_gw_km(path[-1],g))
                if cost<bc: bc=cost; bp=path; bg=int(g)
            if bp is None: continue
            paths[s]=[int(x) for x in bp]; cgw[s]=bg; hc[s]=max(len(bp)-1,0)
            gwl[bg]+=rho[s]
            for a,b in zip(bp[:-1],bp[1:]): isll[edge_key(a,b)]+=rho[s]
        sq[:]=0.
        for s,p in paths.items():
            for n in p[1:]: sq[n]+=rho[s]
    # Final admission over the converged routing, now PARTIAL: each flow
    # contributes min(rho, headroom) on its converged path (was: full flow
    # admitted only if it fit entirely, else dropped).
    fp={}; fg=np.zeros(Ng); fi=defaultdict(float); served_flow={}
    cgw2=-np.ones(Ns,int); hc2=np.zeros(Ns,int)
    for s in sorted(paths.keys()):
        p=paths[s]; g=int(cgw[s])
        if not is_path_valid(p, g): continue
        if cost_dbpr_for_kappa(p, g) > KAPPA: continue
        take=admit_partial(p, g, float(rho[s]), fg, fi)
        if take<=1e-12: continue
        fp[s]=p; cgw2[s]=g; hc2[s]=hc[s]; served_flow[s]=take
    fd=np.full(Ns,np.nan)
    for s,p in fp.items(): fd[s]=full_delay(p,cgw2[s],fg,fi)
    return pack_kpi("DBPR",fp,cgw2,hc2,fg,fi,fd,served_flow=served_flow)

# ---- K5) GPSR ----
def run_gpsr(V):
    paths={}; cgw=-np.ones(Ns,int); hc=np.zeros(Ns,int)
    gwl=np.zeros(Ng); isll=defaultdict(float); served_flow={}
    eany=any_exit(V); MAX_H=max(10,Ns)

    def gpsr_next(u, tgt, vis):
        nbrs=NBR_LIST[u]
        if not nbrs: return None
        du=np.linalg.norm(sat_positions[u]-tgt)
        bv=None; bd=du
        for v in nbrs:
            v=int(v)
            if v in vis: continue
            dv=np.linalg.norm(sat_positions[v]-tgt)
            if dv<bd-1e-12: bd=dv; bv=v
        return bv

    for s in range(Ns):
        flow=float(rho[s])
        if V[s].sum()>0:
            vg=sorted(np.where(V[s]>0)[0],key=lambda g:sat_gw_km(s,g))
            opts=[]
            for g in vg:
                g=int(g)
                if not is_path_valid([s], g): continue
                if cost_gpsr([s], g) > KAPPA: continue
                opts.append((cost_gpsr([s],g),[s],g))
            _commit_choice(s, flow, opts, gwl, isll,
                           paths, cgw, hc, served_flow)
            continue
        if not len(eany): continue
        d2=np.sum((sat_positions[eany]-sat_positions[s])**2,axis=1)
        tgt=sat_positions[int(eany[np.argmin(d2)])]
        path=[s]; vis={s}; cur=s; ok=False
        for _ in range(MAX_H):
            if V[cur].sum()>0: ok=True; break
            nx=gpsr_next(cur,tgt,vis)
            if nx is None: break
            path.append(nx); vis.add(nx); cur=nx
        if not ok and V[cur].sum()==0: continue
        vg=sorted(np.where(V[cur]>0)[0],key=lambda g:sat_gw_km(cur,g))
        opts=[]
        for g in vg:
            g=int(g)
            if not is_path_valid(path, g): continue
            if cost_gpsr(path, g) > KAPPA: continue
            opts.append((cost_gpsr(path,g),[int(x) for x in path],g))
        _commit_choice(s, flow, opts, gwl, isll,
                       paths, cgw, hc, served_flow)
    fd=np.full(Ns,np.nan)
    for s,p in paths.items(): fd[s]=full_delay(p,cgw[s],gwl,isll)
    return pack_kpi("GPSR",paths,cgw,hc,gwl,isll,fd,served_flow=served_flow)

# ---- K6) BP (Backpressure) baseline ----
# Reference: Tassiulas & Ephremides, "Stability Properties of Constrained
# Queueing Systems and Scheduling Policies for Maximum Throughput in
# Multihop Radio Networks," IEEE CDC 1990. Applied to opportunistic
# routing in LEO-terrestrial systems per Krishnan et al. Eq. (14),
# arXiv:2601.13662, 2026.
#
# Classical BP activates ISL k->m when (Q_k - Q_m) * C_ISL > 0 (queue
# differential). For single-snapshot non-splittable admission, we use
# the cumulative committed ISL load as the queue proxy: Q_e ~ load(e),
# so an edge weight of load/C_ISL captures the "downstream queue" that
# BP would avoid.
def run_bp(V):
    """
    Backpressure baseline (queue-differential, opportunistic-routing
    version per Krishnan et al. 2026 Eq. 14). NO propagation term,
    NO distance term -- purely queue-driven. Single path per satellite;
    same kappa-gate and path-validity filter as all other schemes.
    Partial admission: a flow that cannot fully fit still contributes
    its admissible portion instead of being dropped in full.
    """
    paths = {}
    cgw   = -np.ones(Ns, int)
    hc    = np.zeros(Ns, int)
    gwl   = np.zeros(Ng)
    isll  = defaultdict(float)
    served_flow = {}
    cands = np.where(V.sum(axis=0) > 0)[0]
    order = np.argsort(-np.asarray(rho, dtype=float))

    for s in order:
        flow = float(rho[s]); opts = []
        for g in cands:
            ex = exit_sats(V, g)
            if not len(ex): continue

            def wf(u, v, _g=g):
                return BP_ALPHA_ISL * (isll[edge_key(u, v)]
                                        / max(Cisl, 1e-12))

            path, _ = dijkstra(s, ex, wf)
            if path is None: continue
            if not is_path_valid(path, g): continue
            if cost_dsp(path, g) > KAPPA: continue

            cost = 0.0
            for a, b2 in zip(path[:-1], path[1:]):
                cost += BP_ALPHA_ISL * (isll[edge_key(a, b2)]
                                         / max(Cisl, 1e-12))
            cost += BP_ALPHA_ISL * (gwl[g] / max(Cg[g], 1e-12))
            opts.append((cost, [int(x) for x in path], int(g)))

        _commit_choice(s, flow, opts, gwl, isll,
                       paths, cgw, hc, served_flow)

    fd = np.full(Ns, np.nan)
    for s, p in paths.items():
        fd[s] = full_delay(p, cgw[s], gwl, isll)
    return pack_kpi("BP", paths, cgw, hc, gwl, isll, fd,
                    served_flow=served_flow)

# ---- K7) LG-BP (Link-to-Ground-Aware Backpressure) ----
# Reference: Krishnan, Gu, Park, Oh, Choi, "Reinforcement Learning for
# Opportunistic Routing in Software-Defined LEO-Terrestrial Systems,"
# arXiv:2601.13662, 2026. Eq. (17).
#
# LG-BP augments the BP score with a term rewarding neighbors that
# have downlink availability to a gateway with remaining capacity:
#   s^LG(k->m) = s^BP(k->m) + lambda_LG * C_m^LG
# In our single-snapshot framework, C_m^LG becomes a per-target-gateway
# term based on visibility and gateway headroom.
def run_lg_bp(V):
    """
    Link-to-Ground-Aware Backpressure (LG-BP) baseline, per
    Krishnan et al. 2026 Eq. 17. Prefers paths whose intermediate nodes
    can see gateways with remaining capacity. Partial admission as above.
    """
    paths = {}
    cgw   = -np.ones(Ns, int)
    hc    = np.zeros(Ns, int)
    gwl   = np.zeros(Ng)
    isll  = defaultdict(float)
    served_flow = {}
    cands = np.where(V.sum(axis=0) > 0)[0]
    order = np.argsort(-np.asarray(rho, dtype=float))

    def lg_term(node, g_target):
        """LG availability term at `node` for gateway `g_target`."""
        if V[node, g_target] > 0:
            gw_headroom = max(Cg[g_target] - gwl[g_target], 0.) \
                          / max(Cg[g_target], 1e-12)
            return LGBP_LAMBDA_LG * (1.0 - gw_headroom)
        else:
            return LGBP_NO_GW_PENALTY

    for s in order:
        flow = float(rho[s]); opts = []
        for g in cands:
            ex = exit_sats(V, g)
            if not len(ex): continue

            def wf(u, v, _g=g):
                bp_term = BP_ALPHA_ISL * (isll[edge_key(u, v)]
                                          / max(Cisl, 1e-12))
                return bp_term + lg_term(v, _g)

            path, _ = dijkstra(s, ex, wf)
            if path is None: continue
            if not is_path_valid(path, g): continue
            if cost_dsp(path, g) > KAPPA: continue

            cost = 0.0
            for a, b2 in zip(path[:-1], path[1:]):
                cost += BP_ALPHA_ISL * (isll[edge_key(a, b2)]
                                         / max(Cisl, 1e-12))
                cost += lg_term(b2, g)
            cost += BP_ALPHA_ISL * (gwl[g] / max(Cg[g], 1e-12))
            opts.append((cost, [int(x) for x in path], int(g)))

        _commit_choice(s, flow, opts, gwl, isll,
                       paths, cgw, hc, served_flow)

    fd = np.full(Ns, np.nan)
    for s, p in paths.items():
        fd[s] = full_delay(p, cgw[s], gwl, isll)
    return pack_kpi("LG-BP", paths, cgw, hc, gwl, isll, fd,
                    served_flow=served_flow)

# ---- K8) LSP-Chen (Chen et al. 2019 Acta Astronautica) ----
# Reference: Q. Chen, X. Chen, L. Yang, S. Wu, X. Tao,
# "A distributed congestion avoidance routing algorithm in
# mega-constellation network with multi-gateway,"
# Acta Astronautica, vol. 162, pp. 376-387, Sep. 2019.
# DOI: 10.1016/j.actaastro.2019.05.051
#
# Three-process algorithm (fits our problem directly, no adaptation
# of the core mechanism required):
#   1) Gateway assignment: nearest gateway by ISL hop count.
#   2) Path Candidate Region (PCR): rectangular region between
#      source and destination in the +Grid virtual topology.
#   3) Long Side Priority (LSP) greedy path generation:
#        if n_V > n_H : forward vertically (intra-plane)
#        if n_V < n_H : forward horizontally (inter-plane)
#        if tied      : pick lighter-load link
#      where n_V, n_H are remaining vertical/horizontal hops.
#
# Adaptations for our single-snapshot non-splittable framework
# (framework-level only, LSP core untouched):
#   * Admit satellites in descending-rho order (as with all schemes).
#   * Apply the same kappa-gate and path-validity filter.
#   * If admission fails at chosen gateway, satellite is dropped
#     (matches their "packet drop under congestion").
def run_lsp_chen(V):
    """
    LSP-Chen baseline: gateway assignment by nearest hop-count, then
    greedy Long-Side-Priority path within the Path Candidate Region.
    """
    paths = {}
    cgw   = -np.ones(Ns, int)
    hc    = np.zeros(Ns, int)
    gwl   = np.zeros(Ng)
    isll  = defaultdict(float)
    served_flow = {}

    # Pre-compute exit sats per gateway
    exit_sat_lists = {g: np.where(V[:, g] > 0)[0].tolist()
                      for g in range(Ng)}
    valid_gws = [g for g in range(Ng) if exit_sat_lists[g]]
    if not valid_gws:
        fd = np.full(Ns, np.nan)
        return pack_kpi("LSP-Chen", paths, cgw, hc, gwl, isll, fd)

    def grid_hops(a, b):
        """+Grid Manhattan hop count between sats a and b (cyclic in
        both directions)."""
        p_a, sl_a = i_to_ps(a); p_b, sl_b = i_to_ps(b)
        d_p  = min(abs(p_a - p_b), N_PLANES - abs(p_a - p_b))
        d_sl = min(abs(sl_a - sl_b), N_SLOTS - abs(sl_a - sl_b))
        return d_p + d_sl

    def assign_gateway(s):
        """Nearest gateway by ISL hop count to any of its exit sats.
        Returns (best_g, best_target_sat, hop_count)."""
        best_g, best_target, best_h = None, None, 10**9
        for g in valid_gws:
            for es in exit_sat_lists[g]:
                h = grid_hops(s, es)
                if h < best_h:
                    best_h = h; best_g = g; best_target = es
        return best_g, best_target, best_h

    def lsp_next_hop(cur, target, visited):
        """Long Side Priority next-hop selection in +Grid topology."""
        p_c, sl_c = i_to_ps(cur); p_t, sl_t = i_to_ps(target)
        # Remaining hops per axis (cyclic)
        n_H = min(abs(p_c - p_t), N_PLANES - abs(p_c - p_t))
        n_V = min(abs(sl_c - sl_t), N_SLOTS - abs(sl_c - sl_t))

        candidates = []
        # Vertical (slot) direction
        if n_V > 0:
            # Choose shorter cyclic direction
            fwd = ((sl_t - sl_c) % N_SLOTS) <= (N_SLOTS // 2)
            next_sl = (sl_c + 1) % N_SLOTS if fwd else (sl_c - 1) % N_SLOTS
            v_nb = ps_to_i(p_c, next_sl)
            if v_nb not in visited:
                candidates.append(("V", v_nb, n_V,
                                   isll[edge_key(cur, v_nb)]))
        # Horizontal (plane) direction
        if n_H > 0:
            fwd = ((p_t - p_c) % N_PLANES) <= (N_PLANES // 2)
            next_p = (p_c + 1) % N_PLANES if fwd else (p_c - 1) % N_PLANES
            h_nb = ps_to_i(next_p, sl_c)
            if h_nb not in visited:
                candidates.append(("H", h_nb, n_H,
                                   isll[edge_key(cur, h_nb)]))
        if not candidates:
            return None
        # LSP rule: longer side first, tie-break by lighter link load
        candidates.sort(key=lambda c: (-c[2], c[3]))
        return candidates[0][1]

    order = np.argsort(-np.asarray(rho, dtype=float))
    for s in order:
        flow = float(rho[s])

        # Step 1: nearest-gateway assignment
        g_a, target_sat, h_count = assign_gateway(s)
        if g_a is None or target_sat is None: continue

        # Steps 2 & 3: PCR-constrained LSP greedy path build
        path = [s]; cur = s; visited = {s}
        max_iters = h_count + 5
        reached = (s == target_sat)
        for _ in range(max_iters):
            if reached: break
            nx = lsp_next_hop(cur, target_sat, visited)
            if nx is None: break
            path.append(nx); visited.add(nx); cur = nx
            if cur == target_sat:
                reached = True
        if not reached: continue

        if not is_path_valid(path, g_a): continue
        if cost_dsp(path, g_a) > KAPPA: continue
        _commit_choice(s, flow,
                       [(cost_dsp(path, g_a),
                         [int(x) for x in path], int(g_a))],
                       gwl, isll, paths, cgw, hc, served_flow)

    fd = np.full(Ns, np.nan)
    for s, p in paths.items():
        fd[s] = full_delay(p, cgw[s], gwl, isll)
    return pack_kpi("LSP-Chen", paths, cgw, hc, gwl, isll, fd,
                    served_flow=served_flow)

# ---- K9) GDH-DPC with ablation flags ----
def run_gdh_dpq(V, use_lambda=True, use_nu=True):
    """
    Proposed scheme (GDH-DPC): SINGLE-PHASE Lagrangian dual.

    One subgradient loop performs JOINT ascent on the gateway prices
    lambda_g (GW-capacity duals) and the ISL prices nu_e (ISL-capacity
    duals), with queue-aware, lifetime-aware, price-directed routing over
    the augmented graph. The best dual iterate then drives a final
    PARTIAL-admission pass: every flow is admitted up to the bottleneck
    headroom on its price-optimal path and the overflow is dropped, so an
    over-capacity flow contributes what it can instead of being rejected
    in full. Feasibility holds by construction, so the old two-phase
    warm-start, strict-feasibility cleanup, and separate fallback are gone.

    Dual ablation flags:
      use_lambda=True,  use_nu=True   -> full method                 ("Proposed")
      use_lambda=True,  use_nu=False  -> gateway duals only          ("Proposed-LamOnly")
      use_lambda=False, use_nu=True   -> ISL duals only              ("Proposed-NuOnly")
      use_lambda=False, use_nu=False  -> no ascent, delay-greedy      ("Proposed-NoDual")
    A disabled dual is frozen at zero: its subgradient step is skipped, so
    it stays out of both the iterative routing and the final admission pass.
    """
    util_now = float(np.sum(rho)) / float(np.sum(Cg))
    eps_gw_abs, eps_isl_abs = eps_feas_thresholds(util_now)
    label = "Proposed"
    if use_lambda and not use_nu:      label = "Proposed-LamOnly"
    if use_nu and not use_lambda:      label = "Proposed-NuOnly"
    if not use_lambda and not use_nu:  label = "Proposed-NoDual"
    print(f"  [{label}] util={util_now:.2f}, "
          f"eps_GW={eps_gw_abs:.2f}, eps_ISL={eps_isl_abs:.2f}")

    # Inner routing function (same as v6b)
    def _route(lam, nu, prev_gw, prev_isl, use_q, use_nu):
        qg=np.zeros(Ng); q_isl={}
        if use_q:
            for g in range(Ng):
                qg[g]=queue_delay(prev_gw[g],Cg[g])
            for i in range(Ns):
                for j in NBR_LIST[i]:
                    if j<=i: continue
                    q_isl[(i,j)]=queue_delay(
                        float(prev_isl.get((i,j),
                              prev_isl.get((j,i),0.))),Cisl)
        paths_={}; cg_=np.full(Ns,-1,int); hc_=np.zeros(Ns,int)
        gwl_=np.zeros(Ng); isll_=defaultdict(float)
        fd_=np.full(Ns,np.inf)
        SUPER=Ns+Ng
        for s in range(Ns):
            rs=float(rho[s])
            dv=np.full(SUPER+1,np.inf); par=np.full(SUPER+1,-1,int)
            dv[s]=0.; pq=[(0.,s)]
            while pq:
                du,u=heapq.heappop(pq)
                if du!=dv[u]: continue
                if u==SUPER: break
                if Ns<=u<Ns+Ng:
                    if du<dv[SUPER]:
                        dv[SUPER]=du; par[SUPER]=u
                        heapq.heappush(pq,(du,SUPER))
                    continue
                for v in NBR_LIST[u]:
                    v=int(v)
                    L=float(isl_lifetime_s[u,v])
                    pen=ALPHA_HO*(1. if L<=THO_S else 0.)
                    stab=ALPHA4*max(0., L_MAX_LIFE-L)
                    Dij=(isl_prop_s(u,v)+D_TX_ISL+D_PROC_ISL
                         +(float(q_isl.get((min(u,v),max(u,v)),0.))
                           if use_q else 0.))
                    w=rs*(alpha1*Dij+alpha2+pen+stab
                          +(float(nu.get((min(u,v),max(u,v)),0.))
                            if use_nu else 0.))
                    w+=TIE_EPS*(u*Ns+v)
                    nd=du+w
                    if nd<dv[v]: dv[v]=nd; par[v]=u; heapq.heappush(pq,(nd,v))
                for g in np.where(V[u]>0)[0]:
                    g=int(g); gn=Ns+g
                    if not GW_SERVES[g, DST[s]]: continue   # MULTICOMMODITY
                    Dig=(sg_prop_s(u,g)+D_TX_SG+D_PROC_SG
                         +(float(qg[g]) if use_q else 0.))
                    w=rs*(alpha1*Dig+float(lam[g]))
                    w+=TIE_EPS*(Ns*Ns+u*Ng+g)
                    nd=du+w
                    if nd<dv[gn]: dv[gn]=nd; par[gn]=u; heapq.heappush(pq,(nd,gn))
            if not np.isfinite(dv[SUPER]): continue
            node=SUPER; rev=[]
            while node!=-1: rev.append(node); node=par[node]
            rev.reverse(); rev=rev[:-1]
            gn=int(rev[-1])
            if gn<Ns: continue
            g_best=gn-Ns
            sp=[int(x) for x in rev if int(x)<Ns]
            if not sp: continue
            if not is_path_valid(sp, g_best): continue
            sp_cost = cost_dsp(sp, g_best)
            if sp_cost > KAPPA: continue
            if len(sp)>=2:
                for a,b in zip(sp[:-1],sp[1:]): isll_[edge_key(a,b)]+=rs
            gwl_[g_best]+=rs; paths_[s]=sp; cg_[s]=g_best
            hc_[s]=max(len(sp)-1,0)
            tot=0.
            for a,b in zip(sp[:-1],sp[1:]):
                e=(min(a,b),max(a,b))
                tot+=(isl_prop_s(a,b)+D_TX_ISL+D_PROC_ISL
                      +(float(q_isl.get(e,0.)) if use_q else 0.))
            tot+=(sg_prop_s(sp[-1],g_best)+D_TX_SG+D_PROC_SG
                  +(float(qg[g_best]) if use_q else 0.))
            fd_[s]=tot
        return paths_,cg_,hc_,gwl_,isll_,fd_

    def _caps(gwl,isll):
        gwo=np.maximum(gwl-Cg,0.)
        mgw=float(np.max(gwo)) if np.any(gwo>1e-12) else 0.
        mis=0.
        for e,l in isll.items():
            ov=float(l-Cisl)
            if ov>1e-12: mis=max(mis,ov)
        return mgw,mis

    # ---- SINGLE-STAGE JOINT DUAL ------------------------------------------
    # One loop, both prices from zero, both updated every iteration against the
    # same routing. No Stage-2 warm-up: the staged version let lambda reshape
    # routing first, which erased the ISL overloads nu needed to see (measured:
    # |nu>0|=9 with Stage 2 vs 248 without it), so nu was structurally starved.
    #
    # Stopping rule: capacity-feasibility is UNREACHABLE under oversubscription
    # (at 147% load some resource is always saturated), so the old
    # "break when feasible" test could never fire and the loop just timed out.
    # We instead ascend for a fixed budget and stop when the SCORE stops
    # improving.
    #
    # Iterate selection: scored by SERVED TRAFFIC from the actual admission
    # pass -- not by minimum overload. Under overload the dual's job is not to
    # reach feasibility but to decide WHICH flows to drop and where to route the
    # rest; minimising overload just favours the iterate that routed least.
    SAFETY_MAGNITUDE = 0.1     # 100 ms when a resource is fully saturated
    SAFETY_TH        = 0.85    # activate only above 85% load
    def safety_penalty(load, cap):
        r = load / max(cap, 1e-12)
        if r <= SAFETY_TH: return 0.0
        return SAFETY_MAGNITUDE * (r - SAFETY_TH) / (1.0 - SAFETY_TH)

    SUPER = Ns + Ng
    def _path_for(s, lam, nu, fgwl, fisll, with_nu):
        """One augmented Dijkstra: best gateway under STATIC delay + DUAL PRICES.

        DISTRIBUTED SETTING: the routing metric deliberately contains NO live
        link-load term. Previously it added queue_delay()/safety_penalty()
        evaluated on the instantaneous loads accumulated during admission --
        i.e. it assumed a central controller that sees every link's exact load
        while flows are being placed one by one. No real LEO constellation has
        that; a satellite knows only the prices broadcast to it. That oracle
        feedback also made the dual prices redundant (measured: delay rose
        monotonically with price magnitude because a stale static price was
        being added on top of exact live state), which is why every ablation
        collapsed onto the same paths.

        Congestion now reaches the router ONLY through lambda (gateways) and
        nu (ISLs), so the ablation actually tests what the duals contribute:
          NoDual  -> pure shortest-path, congestion-blind
          LamOnly -> gateway congestion via lambda
          NuOnly  -> ISL congestion via nu
          Proposed-> both
        Capacity is still enforced exactly at admission (path_headroom /
        admit_partial); this changes only what the ROUTER can see.
        """
        dv=np.full(SUPER+1,np.inf); par=np.full(SUPER+1,-1,int)
        dv[s]=0.; pq=[(0.,s)]
        while pq:
            du,u=heapq.heappop(pq)
            if du!=dv[u]: continue
            if u==SUPER: break
            if Ns<=u<Ns+Ng:
                if du<dv[SUPER]:
                    dv[SUPER]=du; par[SUPER]=u; heapq.heappush(pq,(du,SUPER))
                continue
            for v in NBR_LIST[u]:
                v=int(v); ek=(min(u,v),max(u,v))
                # Saturated link = UNAVAILABLE (binary backpressure, which a
                # real router does get). This is feasibility, not a graded
                # congestion oracle, and it is what lets successive multipath
                # legs find alternate routes.
                if fisll[ek] >= Cisl - 1e-9: continue
                stab=ALPHA4*max(0.,L_MAX_LIFE-float(isl_lifetime_s[u,v]))
                w=(isl_prop_s(u,v)+D_TX_ISL+D_PROC_ISL
                   + FB_HOP_PEN + stab
                   + (float(nu.get(ek,0.0)) if with_nu else 0.0))
                w+=TIE_EPS*(u*Ns+v)
                nd=du+w
                if nd<dv[v]: dv[v]=nd; par[v]=u; heapq.heappush(pq,(nd,v))
            for g in np.where(V[u]>0)[0]:
                g=int(g); gn=Ns+g
                if fgwl[g] >= Cg[g] - 1e-9: continue     # full gateway
                if not GW_SERVES[g, DST[s]]: continue    # MULTICOMMODITY
                w=(sg_prop_s(u,g)+D_TX_SG+D_PROC_SG
                   + float(lam[g]))
                w+=TIE_EPS*(Ns*Ns+u*Ng+g)
                nd=du+w
                if nd<dv[gn]: dv[gn]=nd; par[gn]=u; heapq.heappush(pq,(nd,gn))
        if not np.isfinite(dv[SUPER]): return None,None
        node=SUPER; rev=[]
        while node!=-1: rev.append(node); node=par[node]
        rev.reverse(); rev=rev[:-1]
        gn=int(rev[-1])
        if gn<Ns: return None,None
        sp=[int(x) for x in rev if int(x)<Ns]
        if not sp: return None,None
        return sp, gn-Ns

    def _admit(lam, nu, with_nu, verbose=False):
        """MULTIPATH admission: a satellite's demand is an aggregate of many
        user sessions, so it is DIVISIBLE. Route it over successive best paths,
        each carrying what fits (min headroom along the path), until the demand
        is served or no path has room. Binary whole-flow admission was wrong:
        with per-sat demand (8-16 Gbps) >= Cisl (8 Gbps), an atomic flow could
        not cross ANY ISL, so the mesh sat idle (hops~0.3, islBusy~1%) and a
        100 Gbps hotspot was rejected outright. Splitting is also what the
        max-flow upper bound assumes, so this is what makes the bound
        attainable."""
        fp={}; cgwo=-np.ones(Ns,int)
        fgwl=np.zeros(Ng); fisll=defaultdict(float)
        fhc=np.zeros(Ns,int); served_flow={}
        ffd_local=np.full(Ns,np.nan)      # per-sat traffic-weighted leg delay
        d_wsum = 0.0                      # traffic-weighted delay accumulator
        order = sorted(range(Ns), key=lambda s: -rho[s])
        t0=time.time()
        for i_flow, s in enumerate(order):
            remaining = float(rho[s])
            hop_wsum = 0.0; taken_tot = 0.0; sat_dw = 0.0
            for _leg in range(MAX_SPLIT_PATHS):
                if remaining <= 1e-9: break
                p, g = _path_for(s, lam, nu, fgwl, fisll, with_nu)
                if p is None or not is_path_valid(p, g) or cost_dsp(p, g) > KAPPA:
                    break
                take = admit_partial(p, g, remaining, fgwl, fisll)
                if take <= 1e-9: break          # no headroom on the best path
                remaining -= take; taken_tot += take
                hop_wsum += take * max(len(p)-1, 0)
                _dleg = full_delay(p, g, fgwl, fisll)
                sat_dw   += take * _dleg
                d_wsum   += take * _dleg
                if s not in fp:                 # primary path for reporting
                    fp[s]=[int(x) for x in p]; cgwo[s]=int(g)
            if taken_tot > 1e-9:
                served_flow[s] = taken_tot
                fhc[s] = int(round(hop_wsum / taken_tot))   # traffic-wtd hops
                ffd_local[s] = sat_dw / taken_tot           # traffic-wtd delay
            if verbose and (i_flow+1) % 250 == 0:
                print(f"      [{label}] admitted {i_flow+1}/{Ns} flows "
                      f"({time.time()-t0:.1f}s)", flush=True)
        tot = float(sum(served_flow.values()))
        avg_d = (d_wsum / tot) if tot > 1e-9 else float('inf')
        return (fp, cgwo, fhc, fgwl, fisll, served_flow, tot, avg_d, ffd_local)

    lam = np.zeros(Ng); nu = defaultdict(float)
    pg  = np.zeros(Ng); pi = defaultdict(float)
    best_lam = lam.copy(); best_nu = {}
    best_served = -1.0; best_avgd = float('inf')
    no_gain = 0; it = 0; trace_rows = []
    do_ascend = bool(use_lambda or use_nu)
    n_iters = DUAL_MAX_ITERS if do_ascend else 1
    trace_label_on = True   # trace ALL variants: comparing lambda-on vs
                            # lambda-off traces is what exposed the staging bug

    for it in range(n_iters):
        # (i) inner Lagrangian routing at current prices (queues ALWAYS on)
        p_,cg_,hc_,gwl_,isll_,fd_ = _route(lam, nu, pg, pi,
                                           use_q=True, use_nu=use_nu)
        mgw,mis = _caps(gwl_,isll_)
        # (ii) score THIS price vector: served traffic FIRST, then DELAY.
        # Throughput saturates at the max-flow bound for essentially every
        # price vector, so scoring on served alone always returned iteration 0
        # (zero prices = greedy) and every variant collapsed to NoDual. The
        # paper's objective is DELAY minimisation subject to capacity, so among
        # solutions that serve the same traffic the lowest-delay one wins --
        # which is where lambda (gateway choice) and nu (ISL congestion) differ.
        _r = _admit(lam, nu, use_nu)
        served_now, avgd_now = _r[6], _r[7]
        SERVED_TOL = 1e-6 * max(float(np.sum(rho)), 1.0)   # relative tie band
        if (served_now > best_served + SERVED_TOL or
                (served_now > best_served - SERVED_TOL and
                 avgd_now < best_avgd - 1e-12)):
            best_served = max(best_served, served_now)
            best_avgd = avgd_now
            best_lam = lam.copy(); best_nu = dict(nu); no_gain = 0
        else:
            no_gain += 1
        if TRACE_DUAL and trace_label_on:
            tot_gwo = float(np.sum(np.maximum(gwl_-Cg,0.)))
            tot_iso = float(sum(max(l-Cisl,0.) for l in isll_.values()))
            n_iso   = int(sum(1 for l in isll_.values() if l-Cisl>1e-9))
            trace_rows.append((it, served_now, avgd_now, best_served,
                               best_avgd, tot_gwo, tot_iso, n_iso,
                               int(np.sum(lam>1e-9)), float(lam.max()),
                               sum(1 for v in nu.values() if v>1e-9),
                               (max(nu.values()) if nu else 0.)))
        if not do_ascend: break
        if no_gain >= DUAL_EARLY_STOP: break        # score plateaued -> stop
        # (iii) EMA-smooth loads, then ONE joint subgradient step on BOTH duals
        pg = (1.0-eta_load)*pg + eta_load*gwl_
        nprev = defaultdict(float)
        for e in set(pi)|set(isll_):
            nprev[e] = ((1.0-eta_load)*float(pi.get(e,0.))
                        + eta_load*float(isll_.get(e,0.)))
        pi = nprev
        alpha_t = DUAL_STEP0 / (1.0 + DUAL_STEP_DECAY*it)
        if use_lambda:
            lam = np.clip(lam + alpha_t*(np.maximum(pg-Cg,0.)/Cg), 0., LAM_MAX)
        if use_nu:
            for e, l in pi.items():
                ov = float(l - Cisl)
                if ov > 1e-12:
                    nu[e] = float(np.clip(float(nu.get(e,0.))
                                          + alpha_t*(ov/Cisl), 0., NU_MAX))

    if TRACE_DUAL and trace_label_on and trace_rows:
        print(f"    --- DUAL TRACE [{label}] single-stage joint "
              f"(lambda={use_lambda}, nu={use_nu}) ---")
        print(f"    {'it':>3} {'served':>8} {'avgD_ms':>8} {'bestSrv':>8} "
              f"{'bestD_ms':>8} {'totGWov':>8} {'totISLov':>8} {'nISL>':>5} "
              f"{'nzLam':>5} {'maxLam':>7} {'nzNu':>5} {'maxNu':>7}")
        for r in trace_rows:
            (i_,sv,ad,bs,bd,tg,ti,ni,nl,ml,nn,mn) = r
            print(f"    {i_:3d} {sv:8.0f} {ad*1e3:8.2f} {bs:8.0f} "
                  f"{bd*1e3:8.2f} {tg:8.1f} {ti:8.1f} {ni:5d} {nl:5d} "
                  f"{ml:7.2f} {nn:5d} {mn:7.2f}")
    print(f"  [dual] single-stage iters={it+1}, "
          f"|nu>0|={sum(1 for v in best_nu.values() if v>1e-9)}, "
          f"best_served={best_served:.0f} Gbps, "
          f"best_avgD={best_avgd*1e3:.2f} ms")

    # ---- Final admission at the best-scoring prices ----
    (fp, cgwo, fhc, fgwl, fisll, served_flow, _tot, _ad, ffd) = _admit(
        best_lam, best_nu, use_nu, verbose=True)
    return pack_kpi(label, fp, cgwo, fhc, fgwl, fisll, ffd,
                    served_flow=served_flow)
# ============================================================
# L) DRIVER
# ============================================================
# Dual-ablation wrappers (isolate each dual's contribution)
def run_gdh_lam_only(V): return run_gdh_dpq(V, use_lambda=True,  use_nu=False)
def run_gdh_nu_only(V):  return run_gdh_dpq(V, use_lambda=False, use_nu=True)
def run_gdh_no_dual(V):  return run_gdh_dpq(V, use_lambda=False, use_nu=False)

SCHEME_FUNCS = {
    "Direct Drop":  run_direct_drop,
    "DSP":          run_dsp,
    "CA-DSP":       run_cadsp,
    "DBPR":         run_dbpr,
    "GPSR":         run_gpsr,
    "BP":           run_bp,           # opp-routing BP (Krishnan Eq. 14)
    "LG-BP":        run_lg_bp,        # opp-routing LG-BP (Krishnan Eq. 17)
    "LSP-Chen":     run_lsp_chen,     # Chen et al. 2019 Acta Astronautica
    "Proposed":     run_gdh_dpq,
    # Dual ablations always registered; membership is controlled by SCHEMES.
    "Proposed-LamOnly": run_gdh_lam_only,
    "Proposed-NuOnly":  run_gdh_nu_only,
    "Proposed-NoDual":  run_gdh_no_dual,
}

ALL_KPI_COLS = [
    ("packet_loss_ratio", "plr"),
    ("served_ratio",      "sr"),
    ("feas",              "feas"),
    ("avgD_s",            "avgD"),
    ("p95D_s",            "p95D"),
    ("avg_hops",          "avg_hops"),
    ("U_ISL",             "U_ISL"),
    ("maxGW",             "maxGW"),
    ("maxISL",            "maxISL"),
    ("gw_util_max",       "gw_util"),
    ("gw_overload",       "gw_over"),
    ("p95_gw_queue",      "p95gwq"),
    ("isl_util_max",      "isl_util"),
    ("isl_overload",      "isl_over"),
]

# ============================================================
# M) VISIBILITY-VS-ELEVATION DIAGNOSTIC
# ============================================================
def compute_visibility_vs_elevation(elev_range=(10,15,20,25,30,35)):
    """
    Compute visible-satellite count for each elevation threshold
    using the CURRENT satellite positions.

    Returns dict {elev_deg: count_of_sats_with_at_least_one_visible_gw}.
    """
    counts = {}
    for elev in elev_range:
        V_elev = build_visibility(sat_positions, elev)
        counts[int(elev)] = int((V_elev.sum(axis=1) > 0).sum())
    return counts

def plot_visibility_diagnostic():
    """Standalone figure: visible sats vs elevation angle."""
    counts = compute_visibility_vs_elevation()
    elevs = sorted(counts.keys())
    vals = [counts[e] for e in elevs]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(e) for e in elevs], vals, color="#3498db",
                  edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v + max(vals)*0.01,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel(r"Minimum elevation angle $\theta_e$ ($^\circ$)")
    ax.set_ylabel("Sats with $\\geq 1$ visible gateway")
    ax.set_title(f"Gateway visibility diagnostic "
                 f"(Ns={Ns}, Ng={Ng}, snapshot at prop_time)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fname = "visibility_vs_elevation.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Visibility diagnostic saved: {fname}")
    print(f"Counts: {counts}")
    return counts

# ============================================================
# UNIFIED SWEEP RUNNER (handles single-snapshot execution)
# ============================================================
def run_one_sweep_point(V_use, snapshot_idx, sweep_name, sweep_val,
                        rows_out, ub_rows_out, extra_cols=None):
    """
    Run all schemes + UB at a single sweep point at the current
    snapshot. Appends results to rows_out and ub_rows_out.

    extra_cols: dict of columns to attach to each output row
                (e.g. {"scale_active": 1.0, "offered": 2910.5}).
    """
    ub_served, ub_offered = feasibility_upper_bound(V_use)
    ub_sr = ub_served / max(ub_offered, 1e-12)
    ub_row = dict(snapshot_idx=snapshot_idx,
                  sweep_name=sweep_name, sweep_val=float(sweep_val),
                  ub_served=ub_served, ub_offered=ub_offered,
                  ub_sr=ub_sr, ub_plr=1.0-ub_sr)
    if extra_cols: ub_row.update(extra_cols)
    ub_rows_out.append(ub_row)
    print(f"   UB_served(relaxation)={ub_served:.1f} ({ub_sr*100:.1f}%)")

    _persat_all = {}
    for scheme in SCHEMES:
        try:
            out = SCHEME_FUNCS[scheme](V_use)
            out["snapshot_idx"] = snapshot_idx
            out["sweep_name"]   = sweep_name
            out["sweep_val"]    = float(sweep_val)
            if extra_cols: out.update(extra_cols)
            _persat_all[scheme] = out.pop("_persat", {})
            rows_out.append(out)
            gap = (ub_sr - out["served_ratio"]) * 100
            print(f"    {scheme:14s} | "
                  f"PLR={out['packet_loss_ratio']:.4f} | "
                  f"sr={out['served_ratio']:.3f} (gap {gap:+.1f}pp) | "
                  f"hops={out['avg_hops']:.2f} | "
                  f"avgD={out['avgD_s']*1e3:.1f}ms | "
                  f"U_ISL={out['U_ISL']:.0f} | "
                  f"islBusy={out['isl_busy_frac']*100:.1f}% | "
                  f"islQ={out['isl_qdelay_mean']*1e3:.1f}/"
                  f"{out['isl_qdelay_p95']*1e3:.1f}ms")
        except Exception:
            import traceback; traceback.print_exc()

    # ---- CONTROLLED COMPARISON on the common served subset ----
    # Delay averages across variants are not comparable when the variants serve
    # different traffic (more served -> the extra traffic rides longer relay
    # paths -> higher average). Restricting to satellites ALL variants served,
    # and to the min served amount per satellite, makes it like-for-like.
    if len(_persat_all) > 1 and all(len(v) for v in _persat_all.values()):
        common = set.intersection(*[set(v.keys()) for v in _persat_all.values()])
        if common:
            base = {s_: min(_persat_all[sc][s_][0] for sc in _persat_all)
                    for s_ in common}
            wtot = sum(base.values())
            print(f"    -- controlled: {len(common)} sats served by ALL "
                  f"variants, {wtot:.0f} Gbps common traffic --")
            for sc in SCHEMES:
                if sc not in _persat_all: continue
                d = sum(base[s_]*_persat_all[sc][s_][1] for s_ in common)
                print(f"       {sc:16s} commonD="
                      f"{(d/wtot)*1e3 if wtot>0 else float('nan'):7.2f} ms")

# ============================================================
# MAIN DRIVER: MULTI-SNAPSHOT LOOP OVER SWEEPS
# ============================================================
all_rows = []
all_ub_rows = []

for k_snap, delta_min in enumerate(SNAPSHOT_DELTAS_MIN[:N_SNAPSHOTS]):
    print(f"\n{'#'*60}")
    print(f"# SNAPSHOT {k_snap+1}/{N_SNAPSHOTS}: delta = {delta_min} min")
    print(f"{'#'*60}")

    if k_snap > 0:
        rebuild_at_snapshot(delta_min)
        # V_fixed already rebuilt in rebuild_at_snapshot at ELEV_FIXED

    # ----- Traffic scale sweep (headline) -----
    if RUN_TRAFFIC_SWEEP:
        Cisl = 8.0    # restore default before sweep
        print(f"\n=== [SNAP {k_snap}] TRAFFIC SCALE SWEEP "
              f"(Cg={Cg_default}, Ng={Ng}, elev={ELEV_FIXED}, "
              f"Cisl={Cisl}) ===")
        for sc_active in SCALE_SWEEP:
            rho = rho_base * float(sc_active)
            offered = float(rho.sum())
            util_gw = offered / (Ng * Cg_default)
            print(f"\n--- traffic scale={sc_active:.2f} | "
                  f"offered={offered:.1f} Gbps | util={util_gw:.2f} ---")
            run_one_sweep_point(
                V_fixed, k_snap, "traffic", sc_active,
                all_rows, all_ub_rows,
                extra_cols={"scale_active": float(sc_active),
                            "offered": offered,
                            "Cisl_active": Cisl,
                            "Ng_active": Ng,
                            "Cg_active": Cg_default,
                            "elev_active": ELEV_FIXED})

    # ----- ISL hotspot (flash-crowd) sweep -----
    if RUN_HOTSPOT:
        Cisl = 8.0
        hs = hotspot_sats(HOTSPOT_CENTER[0], HOTSPOT_CENTER[1],
                          HOTSPOT_RADIUS_KM)
        nvis = (int(V_fixed[hs].sum(axis=1).astype(bool).sum())
                if len(hs) else 0)
        print(f"\n=== [SNAP {k_snap}] ISL HOTSPOT SWEEP "
              f"(center={HOTSPOT_CENTER}, R={HOTSPOT_RADIUS_KM:.0f}km, "
              f"Cg={Cg_default}, Ng={Ng}, Cisl={Cisl}) ===")
        print(f"  crowd = {len(hs)} sats; "
              f"{nvis}/{len(hs)} have a directly-visible gateway "
              f"(the rest must backhaul over ISLs)")
        for sc in HOTSPOT_SCALE_SWEEP:
            rho = rho_base.copy() * float(sc)
            if HOTSPOT_DEMAND > 0 and len(hs):
                rho[hs] = float(HOTSPOT_DEMAND)      # hotspot ON at every point
            offered = float(rho.sum())
            print(f"\n--- scale={sc:.2f} "
                  f"| crowd={HOTSPOT_DEMAND*len(hs):.0f} Gbps "
                  f"| total offered={offered:.1f} Gbps "
                  f"| GW cap={Ng*Cg_default:.0f} "
                  f"({offered/(Ng*Cg_default)*100:.0f}% subscribed) ---",
                  flush=True)
            _t0 = time.time()
            run_one_sweep_point(
                V_fixed, k_snap, "scale", sc,
                all_rows, all_ub_rows,
                extra_cols={"scale_active": float(sc), "offered": offered,
                            "hotspot_demand": HOTSPOT_DEMAND,
                            "Cisl_active": Cisl, "Ng_active": Ng,
                            "Cg_active": Cg_default, "elev_active": ELEV_FIXED})
            print(f"  (scale={sc:.2f} done in {time.time()-_t0:.1f}s)",
                  flush=True)
        rho = rho_base.copy()

    # ----- ISL capacity sweep (NEW) -----
    if RUN_CISL_SWEEP:
        print(f"\n=== [SNAP {k_snap}] ISL CAPACITY SWEEP "
              f"(scale=1.0, Cg={Cg_default}, Ng={Ng}, "
              f"elev={ELEV_FIXED}) ===")
        rho = rho_base.copy()      # scale=1.0
        offered = float(rho.sum())
        for cisl_val in CISL_SWEEP:
            Cisl = float(cisl_val)   # override module global
            print(f"\n--- Cisl={Cisl:.1f} Gbps | offered={offered:.1f} Gbps ---")
            run_one_sweep_point(
                V_fixed, k_snap, "cisl", cisl_val,
                all_rows, all_ub_rows,
                extra_cols={"scale_active": 1.0,
                            "offered": offered,
                            "Cisl_active": Cisl,
                            "Ng_active": Ng,
                            "Cg_active": Cg_default,
                            "elev_active": ELEV_FIXED})
        Cisl = 8.0    # restore default after sweep

    # ----- Legacy sweeps (off by default; enable if desired) -----
    if RUN_NG_SWEEP:
        cap_mode = "agg-cap preserved" if NG_PRESERVE_GW_CAP else "Cg fixed"
        NG_PROBE_SCALE = 0.5   # run below saturation so links sit ~80-95% full
                               # and nu has spare paths to divert onto (at
                               # scale 1.0 every ISL pegs at 100% and no dual
                               # can help -- there is no slack to route around)
        print(f"\n=== [SNAP {k_snap}] N_g SWEEP "
              f"(scale={NG_PROBE_SCALE}, Cisl={Cisl}, elev={ELEV_FIXED}, "
              f"{cap_mode}) ===")
        _gw_pos_full = gateway_positions_full
        for ng_val in NG_SWEEP:
            sub = farthest_point_gateways(_gw_pos_full, ng_val)
            gateway_positions = _gw_pos_full[sub]      # subset globals
            Ng = len(sub)
            cg_ng = (Cg_default * Ng_full / Ng) if NG_PRESERVE_GW_CAP else Cg_default
            Cg = np.ones(Ng, dtype=float) * cg_ng
            V_ng = build_visibility(sat_positions, ELEV_FIXED)
            rho = rho_base.copy() * NG_PROBE_SCALE
            offered = float(rho.sum())
            vis_sats = int((V_ng.sum(axis=1) > 0).sum())
            print(f"\n--- Ng={Ng} | Cg={cg_ng:.1f} Gbps/gw "
                  f"(agg {Ng*cg_ng:.0f}) | direct-vis sats={vis_sats}/{Ns} "
                  f"| offered={offered:.1f} Gbps ---")
            run_one_sweep_point(
                V_ng, k_snap, "ng", ng_val,
                all_rows, all_ub_rows,
                extra_cols={"scale_active": NG_PROBE_SCALE,
                            "offered": offered,
                            "Cisl_active": Cisl,
                            "Ng_active": Ng,
                            "Cg_active": cg_ng,
                            "elev_active": ELEV_FIXED})
        # restore full gateway deployment
        gateway_positions = _gw_pos_full
        Ng = Ng_full
        Cg = np.ones(Ng, dtype=float) * Cg_default

    if RUN_ELEV_SWEEP:
        print(f"\n=== [SNAP {k_snap}] ELEVATION SWEEP ===")
        for elev_val in ELEV_SWEEP:
            V_elev = build_visibility(sat_positions, elev_val)
            rho = rho_base.copy()
            offered = float(rho.sum())
            print(f"\n--- elev={elev_val} deg ---")
            run_one_sweep_point(
                V_elev, k_snap, "elev", elev_val,
                all_rows, all_ub_rows,
                extra_cols={"scale_active": 1.0,
                            "offered": offered,
                            "Cisl_active": Cisl,
                            "Ng_active": Ng,
                            "Cg_active": Cg_default,
                            "elev_active": elev_val})

    if RUN_CG_SWEEP:
        print(f"\n=== [SNAP {k_snap}] C_g SWEEP ===")
        for cg_val in CG_SWEEP:
            Cg = np.ones(Ng, dtype=float) * float(cg_val)
            rho = rho_base.copy()
            offered = float(rho.sum())
            print(f"\n--- Cg={cg_val} Gbps ---")
            run_one_sweep_point(
                V_fixed, k_snap, "cg", cg_val,
                all_rows, all_ub_rows,
                extra_cols={"scale_active": 1.0,
                            "offered": offered,
                            "Cisl_active": Cisl,
                            "Ng_active": Ng,
                            "Cg_active": cg_val,
                            "elev_active": ELEV_FIXED})
        Cg = np.ones(Ng, dtype=float) * Cg_default  # restore

# Restore working traffic
rho = rho_base.copy()

# Consolidate into DataFrames
df_all = pd.DataFrame(all_rows)
df_ub  = pd.DataFrame(all_ub_rows)
print(f"\nCollected {len(df_all)} scheme-runs, {len(df_ub)} UB points")

# For backward compatibility with v6b post-processing:
if RUN_TRAFFIC_SWEEP:
    df_ng = df_all[df_all["sweep_name"] == "traffic"].copy()
    df_ub_tr = df_ub[df_ub["sweep_name"] == "traffic"].copy()
elif RUN_HOTSPOT and (df_all["sweep_name"] == "scale").any():
    # hotspot-on traffic-scale sweep: x-axis is the background scale
    df_ng = df_all[df_all["sweep_name"] == "scale"].copy()
    df_ub_tr = df_ub[df_ub["sweep_name"] == "scale"].copy()
else:
    df_ng = df_all
    df_ub_tr = df_ub

# ============================================================
# VISIBILITY DIAGNOSTIC (standalone)
# ============================================================
if RUN_VIS_DIAGNOSTIC:
    print(f"\n=== VISIBILITY DIAGNOSTIC ===")
    plot_visibility_diagnostic()

# ============================================================
# N) MATLAB EXPORT (for traffic sweep only; extend if needed)
# ============================================================
print(f"\n=== MATLAB EXPORT: traffic sweep, snapshot-averaged ===")
if (RUN_TRAFFIC_SWEEP or RUN_HOTSPOT) and len(df_ng) > 0:
    ub_sorted = df_ub_tr.sort_values("sweep_val").groupby("sweep_val").mean(
        numeric_only=True).reset_index()
    if len(ub_sorted):
        print(f"\n% Feasibility UB")
        print(f"ub_tr_x_scale = "
              f"[{' '.join(f'{v:.2f}' for v in ub_sorted['sweep_val'])}];")
        print(f"ub_tr_sr      = "
              f"[{' '.join(f'{v:.6f}' for v in ub_sorted['ub_sr'])}];")
        print(f"ub_tr_plr     = "
              f"[{' '.join(f'{v:.6f}' for v in ub_sorted['ub_plr'])}];")

    for scheme in SCHEMES:
        sub = df_ng[df_ng["scheme"] == scheme].copy()
        if not len(sub): continue
        sub_avg = sub.groupby("sweep_val").mean(numeric_only=True).reset_index()
        sub_avg = sub_avg.sort_values("sweep_val")
        sn = scheme.lower().replace("-","_").replace(" ","_")
        print(f"\n% {scheme}")
        print(f"{sn}_tr_x_scale = "
              f"[{' '.join(f'{v:.2f}' for v in sub_avg['sweep_val'])}];")
        for col, tag in ALL_KPI_COLS:
            if col not in sub_avg.columns: continue
            vals = sub_avg[col].tolist()
            print(f"{sn}_tr_{tag} = "
                  f"[{' '.join(f'{v:.6f}' for v in vals)}];")

# ============================================================
# O) PLOTS (headline: traffic sweep, 10 curves)
# ============================================================
COLORS = {"Direct Drop":"#e74c3c","DSP":"#e67e22",
          "CA-DSP":"#f1c40f","DBPR":"#2ecc71",
          "GPSR":"#3498db",
          "BP":"#7f8c8d","LG-BP":"#16a085",
          "LSP-Chen":"#c0392b",   # dark red for Chen 2019
          "Proposed":"#8e44ad",
          "Proposed-LamOnly":"#9b59b6","Proposed-NuOnly":"#d35400",
          "Proposed-NoDual":"#95a5a6",
          "UB":"#000000"}
MARKERS = {"Direct Drop":"x","DSP":"s","CA-DSP":"^",
           "DBPR":"D","GPSR":"o",
           "BP":"P","LG-BP":"v","LSP-Chen":"<",
           "Proposed":"*","Proposed-LamOnly":"h","Proposed-NuOnly":"H",
           "Proposed-NoDual":"d",
           "UB":"+"}
LINES = {k:"--" for k in COLORS}
LINES["Proposed"]          = "-"
LINES["Proposed-LamOnly"]  = "-."
LINES["Proposed-NuOnly"]   = ":"
LINES["Proposed-NoDual"]   = (0,(3,1,1,1))
LINES["UB"]                = ":"

KPI_PLOTS = [
    ("packet_loss_ratio", "PLR",           "Packet Loss Rate",    False, False),
    ("served_ratio",      "SR",            "Served Ratio",        False, False),
    ("avgD_s",            "Delay(ms)",     "Avg Delay (ms)",      True,  True),
    ("p95D_s",            "p95D(ms)",      "p95 Delay (ms)",      True,  True),
    ("avg_hops",          "Hops",          "Avg Hop Count",       True,  False),
    ("U_ISL",             "U_ISL",         "U_ISL",               True,  False),
    ("gw_util_max",       "GW Util",       "Max GW Utilisation",  False, False),
    ("gw_overload",       "GW Overload",   "GW Overload",         False, False),
    ("p95_gw_queue",      "GWQ p95(ms)",   "GW Queue p95 (ms)",   False, True),
    ("isl_util_max",      "ISL Util",      "Max ISL Utilisation", True,  False),
]

def plot_sweep_panel(ax, df_use, ub_use, sweep_name, xlabel,
                     col, ylabel, title, skip_dd=False, scale_ms=False):
    """Plot one KPI column for one sweep, averaged across snapshots."""
    for scheme in SCHEMES:
        if skip_dd and scheme == "Direct Drop": continue
        sub = df_use[(df_use["scheme"] == scheme)
                     & (df_use["sweep_name"] == sweep_name)].copy()
        if not len(sub): continue
        sub_avg = sub.groupby("sweep_val").mean(numeric_only=True).reset_index()
        sub_avg = sub_avg.sort_values("sweep_val")
        if col not in sub_avg.columns: continue
        yv = sub_avg[col].to_numpy()
        if scale_ms: yv = yv * 1e3
        ax.plot(sub_avg["sweep_val"].to_numpy(), yv,
                color=COLORS.get(scheme,"gray"),
                marker=MARKERS.get(scheme,"."),
                linestyle=LINES.get(scheme,"--"),
                linewidth=2.5 if scheme=="Proposed" else 1.5,
                markersize=9 if scheme=="Proposed" else 5,
                label=scheme)
    if col in ("served_ratio", "packet_loss_ratio") and ub_use is not None:
        ubs = ub_use[ub_use["sweep_name"] == sweep_name].copy()
        if len(ubs):
            ubs_avg = ubs.groupby("sweep_val").mean(numeric_only=True).reset_index()
            ubs_avg = ubs_avg.sort_values("sweep_val")
            ub_col = "ub_sr" if col == "served_ratio" else "ub_plr"
            ax.plot(ubs_avg["sweep_val"].to_numpy(),
                    ubs_avg[ub_col].to_numpy(),
                    color=COLORS["UB"], marker=MARKERS["UB"],
                    linestyle=LINES["UB"], linewidth=2.0,
                    markersize=7, label="UB")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, framealpha=0.9, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.6)

# ---- Headline figure: traffic sweep ----
if RUN_TRAFFIC_SWEEP:
    ncols = 5; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    axes = axes.flatten()
    for idx, (col, ylabel, title, skip_dd, ms) in enumerate(KPI_PLOTS):
        plot_sweep_panel(axes[idx], df_all, df_ub, "traffic",
                         "Traffic scale", col, ylabel, title,
                         skip_dd=skip_dd, scale_ms=ms)
    for idx in range(len(KPI_PLOTS), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(
        f"Traffic scale sweep (headline) -- "
        f"Ns={Ns}, Ng={Ng}, Cg={Cg_default}, Cisl=8, elev={ELEV_FIXED}deg, "
        f"N_snap={N_SNAPSHOTS}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("v6c_traffic_sweep_all_kpis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: v6c_traffic_sweep_all_kpis.png")

# ---- Cisl sweep figure ----
if RUN_CISL_SWEEP:
    ncols = 5; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    axes = axes.flatten()
    for idx, (col, ylabel, title, skip_dd, ms) in enumerate(KPI_PLOTS):
        plot_sweep_panel(axes[idx], df_all, df_ub, "cisl",
                         "ISL capacity $C_{ISL}$ (Gbps)",
                         col, ylabel, title,
                         skip_dd=skip_dd, scale_ms=ms)
    for idx in range(len(KPI_PLOTS), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(
        f"ISL capacity sweep -- scale=1.0, "
        f"Ns={Ns}, Ng={Ng}, Cg={Cg_default}, elev={ELEV_FIXED}deg, "
        f"N_snap={N_SNAPSHOTS}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("v6c_cisl_sweep_all_kpis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: v6c_cisl_sweep_all_kpis.png")

print("\nDone. df_all has all sweep results; df_ub has UB.")
print(f"Access individual sweeps via df_all[df_all['sweep_name']=='traffic'] etc.")