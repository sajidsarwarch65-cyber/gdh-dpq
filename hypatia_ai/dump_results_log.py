"""Dump all generated results into one recoverable log + reload the saved .npy sweeps."""
import numpy as np, os, datetime

LOG="/home/msarwa02/results_log.txt"
f=open(LOG,"w")
def w(*a): print(*a); print(*a,file=f)

w("="*70)
w("GDH-DPQ HYPATIA + RL  RESULTS LOG   (dumped %s)" % datetime.datetime.now().isoformat(timespec='seconds'))
w("HD=12, scale 1.0, use_std_gw (99 Mega gateways) unless noted")
w("="*70)

# ---- diurnal dual ablation (from .npy) ----
w("\n### DIURNAL DUAL ABLATION (2020-02-24, every 2h) ###")
w("cols: offered, prop, nu, lam, nod, prop-nu, prop-nod, gw_util")
p="/home/msarwa02/diurnal_ablation.npy"
if os.path.exists(p):
    A=np.load(p); hrs=list(range(0,24,2))
    w("%-6s %7s %6s %6s %6s %6s | %8s %8s" % ("hour","offer","prop","nu","lam","nod","prop-nu","prop-nod"))
    for i,h in enumerate(hrs):
        r=A[i]
        w("%02d:00  %7.0f %6.3f %6.3f %6.3f %6.3f | %+8.4f %+8.4f" % (h,r[0],r[1],r[2],r[3],r[4],r[5],r[6]))
else: w("(diurnal_ablation.npy not found)")

# ---- diurnal RL (from .npy) ----
w("\n### DIURNAL RL SERVED RATIO (K=4 greedy) ###")
p="/home/msarwa02/diurnal_rl.npy"
if os.path.exists(p):
    A=np.load(p); hrs=list(range(0,24,2))
    w("%-6s %7s %7s" % ("hour","offer","RL_SR"))
    for i,h in enumerate(hrs):
        w("%02d:00  %7.0f %7.4f" % (h,A[i,0],A[i,1]))
else: w("(diurnal_rl.npy not found)")

# ---- static reference tables (typed from this session, not previously filed) ----
w("\n### HD=12 HEAD-TO-HEAD RL(K=4) vs DUAL  static Poisson (SR / mean_hops) ###")
w("scale  RL_SR  RL_hops | dual_SR dual_hops")
for row in [("0.5",0.967,5.04,1.000,2.19),("1.0",0.971,4.98,1.000,2.24),
            ("2.0",0.910,5.40,0.981,3.89),("2.5",0.865,5.41,0.942,4.21),
            ("3.0",0.832,5.25,0.898,3.14),("4.0",0.792,5.23,0.836,5.11)]:
    w("%-5s  %.3f  %5.2f  | %.3f   %5.2f" % row)

w("\n### HOTSPOT-INTENSITY SWEEP (gap = dual - RL) ###")
w("HD/sat  ISLs_needed  dual_SR  RL_SR   gap")
for row in [("8",1.0,1.000,0.999,0.001),("12",1.5,1.000,0.956,0.044),
            ("16",2.0,0.999,0.948,0.051),("20",2.5,0.988,0.919,0.069),
            ("30",3.75,0.918,0.817,0.101)]:
    w("%-6s  %-11s  %.3f   %.3f   %.3f" % row)

w("\n### LINK-FAILURE SWEEP HD=12 scale 1.0 (RL: SR / reroute_ms) ###")
w("fail%  RL_SR  reroute_ms | dual_SR  solve_s")
for row in [(0,0.971,28,1.000,36.6),(1,0.963,29,1.000,36.6),(2,0.971,29,1.000,48.1),
            (5,0.966,28,1.000,35.8),(10,0.971,25,0.990,37.6)]:
    w("%-5d  %.3f  %4d       | %.3f    %.1f" % row)

w("\n### PEAK vs TROUGH (degeneracy fix) ###")
w("TROUGH 22:20 offered 2242: prop 0.9525 nu 0.9610 prop-nu -0.0085 (lambda idle)")
w("PEAK   12:00 offered 2848: prop 0.8627 nu 0.7617 prop-nu +0.1010 (lambda BINDS)")

w("\n### TRAFFIC TRACE ###")
w("31-day 5-min, 4677 zones, total range min=4747 max=15395 mean=10263 Gbps (3.2x peak-trough)")

w("\n### KEY CONSTANTS ###")
w("Cg=40 Cisl=8 HOTSPOT_DEMAND=12(or30) HOTSPOT_RADIUS_KM=500 NREG=3 HUB_FRACTION=0.35")
w("dual: alpha1=1.0 alpha2=0.02 D_TX_ISL=2e-4 D_PROC_ISL=1e-4 QUEUE_SCALE=0.003 QUEUE_MAX=0.03")

w("\n### SAVED FILES ###")
for fn in ["diurnal_ablation.npy","diurnal_rl.npy","ground_traffic_timeseries.csv",
           "ground_zone_metadata.csv","policy_unroll.pt","standalone_mega_gw_km.npy"]:
    pp="/home/msarwa02/"+fn
    w("  %-32s %s" % (fn, "OK %d bytes"%os.path.getsize(pp) if os.path.exists(pp) else "MISSING"))
w("="*70)
f.close()
print("\nWROTE",LOG)
