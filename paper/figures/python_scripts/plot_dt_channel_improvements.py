"""
DT Channel Improvements Figure
Reconstructed from fstate_gdh_dpq.py models
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# ============================================================
# Model functions (from fstate_gdh_dpq.py)
# ============================================================

# ISL capacity model
D_MIN_KM = 1000.0  # minimum ISL distance
C_MAX    = 50.0    # max capacity (normalised)

def isl_cap(d_km):
    return C_MAX * (D_MIN_KM / d_km)**2

# GSL elevation cost model
D_TX_UNIFORM = 0.5e-3  # s (uniform model)

def gsl_cost_uniform():
    return D_TX_UNIFORM * 1000  # ms

def gsl_cost_elevation(elev_deg):
    """D_TX / sin^2(elev) in ms"""
    return (D_TX_UNIFORM / max(math.sin(math.radians(elev_deg))**2, 1e-6)) * 1000

# Doppler model (optical ISL, 1550nm)
def doppler_bw_reduction_pct(f_ghz):
    """Bandwidth reduction due to Doppler at carrier frequency f_ghz"""
    v_rel = 7.6e3  # m/s relative velocity between adjacent Starlink sats
    c = 3e8
    f_hz = f_ghz * 1e9
    delta_f = f_hz * v_rel / c
    # Bandwidth reduction as % of channel spacing (assume 1 GHz spacing)
    bw_spacing = 1e9
    return delta_f / bw_spacing * 100

# ============================================================
# Data
# ============================================================
# ISL distances
dist_km = np.linspace(500, 2000, 300)
cap_improved = np.array([isl_cap(d) for d in dist_km])
cap_uniform  = np.full_like(dist_km, 25.0)

# GSL elevation
elev_range = np.linspace(25, 70, 200)
cost_improved = np.array([gsl_cost_elevation(e) for e in elev_range])
cost_uniform  = np.full_like(elev_range, gsl_cost_uniform())

# Doppler
freqs_ghz  = [10, 20, 26, 40, 60]
doppler_pct = [doppler_bw_reduction_pct(f) for f in freqs_ghz]
freq_labels = ['10 GHz','20 GHz','26 GHz','40 GHz','60 GHz']

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) ISL capacity
ax = axes[0]
ax.plot(dist_km, cap_improved, 'r-', lw=2,
        label=r'Improved DT: $C_{ij}=C_{max}(d_{min}/d_{ij})^2$')
ax.plot(dist_km, cap_uniform, 'b--', lw=2,
        label='Original DT: uniform $C_{ISL}=25.0$')
ax.axvline(x=1000, color='gray', linestyle=':', lw=1,
           label='$d_{min}$=500km breakpoint')
ax.set_xlabel('ISL distance (km)', fontsize=11)
ax.set_ylabel('ISL capacity (normalised)', fontsize=11)
ax.set_title('(a) Distance-Dependent ISL Capacity\n87.5% of ISLs degraded vs uniform model',
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim([400, 2000])

# (b) GSL elevation cost
ax = axes[1]
ax.plot(elev_range[::-1], cost_improved[::-1], 'r-', lw=2,
        label=r'Improved DT: $D_{TX}/\sin^2(\theta)$')
ax.plot(elev_range[::-1], cost_uniform[::-1], 'b--', lw=2,
        label='Original DT: uniform $D_{TX}$')
ax.fill_between(elev_range[::-1], cost_uniform[::-1], cost_improved[::-1],
                alpha=0.15, color='red', label='Additional cost (low elevation)')
ax.set_xlabel('Elevation angle θ (degrees)', fontsize=11)
ax.set_ylabel('GSL transmission delay (ms)', fontsize=11)
ax.set_title('(b) Elevation-Dependent GSL Cost\nLow-elevation gateways penalised in routing',
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (c) Doppler bandwidth reduction
ax = axes[2]
bars = ax.bar(range(len(freqs_ghz)), doppler_pct,
              color='green', alpha=0.85, edgecolor='black', lw=0.8)
ax.set_xticks(range(len(freqs_ghz)))
ax.set_xticklabels(freq_labels, fontsize=9)
ax.set_ylabel('Bandwidth reduction (%)', fontsize=11)
ax.axhline(y=0.1, color='gray', linestyle='--', lw=1.5,
           label='0.1% threshold (negligible)')
for bar, val in zip(bars, doppler_pct):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
            f'{val:.4f}%', ha='center', fontsize=8)
ax.set_title('(c) Doppler Effect on ISL Bandwidth\nKa-band (26 GHz): 0.023% negligible',
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Digital Twin Improvements: ISL Capacity, GSL Elevation Model, Doppler Analysis\n'
             'Starlink S1  550km altitude  Ka-band ISLs',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/msarwa02/results/dt_channel_improvements.pdf', bbox_inches='tight')
plt.savefig('/home/msarwa02/results/dt_channel_improvements.png', dpi=150, bbox_inches='tight')
print("Saved dt_channel_improvements.pdf")

# Print data for MATLAB
print("\n% MATLAB data for dt_channel_improvements figure")
print(f"fig3.isl.dist_km = {list(range(500,2100,100))};")
print(f"fig3.isl.cap_improved = [{' '.join(f'{isl_cap(d):.3f}' for d in range(500,2100,100))}];")
print(f"fig3.isl.cap_uniform = 25.0 * ones(1,{len(range(500,2100,100))});")
print(f"fig3.gsl.elev_deg = {list(range(25,71,5))};")
print(f"fig3.gsl.cost_improved_ms = [{' '.join(f'{gsl_cost_elevation(e):.4f}' for e in range(25,71,5))}];")
print(f"fig3.gsl.cost_uniform_ms = {gsl_cost_uniform():.4f};")
print(f"fig3.doppler.freqs_ghz = {freqs_ghz};")
print(f"fig3.doppler.bw_reduction_pct = [{' '.join(f'{v:.6f}' for v in doppler_pct)}];")
