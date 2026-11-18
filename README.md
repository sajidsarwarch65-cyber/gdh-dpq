# GDHR: Gateway-aware Deliver-Hop Routing for LEO Satellite Networks

Congestion-aware routing for LEO mega-constellations, where the decision of
**when and where to leave the inter-satellite link (ISL) mesh** is optimized
jointly with the path through it.

Gateway capacity and ISL capacity are treated as shared, contended resources
and priced with Lagrangian dual variables (λ for gateways, ν for ISLs). Each
satellite then solves a shortest-path problem over a graph whose weights carry
those prices, so the exit gateway emerges from the same computation as the
route rather than being fixed in advance.

Built on [Hypatia](https://github.com/snkas/hypatia) (ETH Zurich, MIT).

> **Naming note.** This method was previously called GDH-DPQ. Several source
> files still carry the old name (`fstate_gdh_dpq.py`,
> `gdh-dpq-controller.cc`, `gdh_dpq_v6e.py`). They implement GDHR.

---

## Two components

The repository holds two separate implementations that serve different purposes.

**1. Hypatia / ns-3 integration** — packet-level validation on a
Starlink Shell-1 digital twin (1,584 satellites), with routing decisions
delivered to ns-3 either through forwarding-state files or through an OpenFlow
SDN controller.

**2. `hypatia_ai/` — standalone solver and learned extensions** — a
self-contained Python implementation of the dual-decomposition solver on an
analytic Walker constellation, plus the GNN/RL solvers benchmarked against it.
This is the implementation behind the paper's numerical results.

The two use different constellations and traffic instances, so their numbers
are not directly comparable. The ns-3 side establishes that the routing
decisions are realizable and correct at packet level; the standalone side is
where the algorithmic comparison against baselines is made.

---

## Repository layout

```
satgenpy/satgen/dynamic_state/
    fstate_gdh_dpq.py              GDHR solver -> Hypatia forwarding state
    fstate_gdh_rl.py               learned-policy routes -> forwarding state
    algorithm_gdh_dpq_over_isls.py GDHR routing hook
    algorithm_gdh_rl_over_isls.py  learned-policy routing hook
    algorithm_baselines_over_isls.py  Direct-Deliver, DSP, CA-DSP, DBPR, GPSR

ns3-sat-sim/simulator/src/openflow/model/
    gdh-dpq-controller.{h,cc}      OpenFlow controller pushing flow rules

paper/satellite_networks_state/
    main_starlink_550.py           Shell-1 forwarding-state generation
    main_starlink_shell3.py        Shell-3 (cross-shell generalization)
    main_starlink_shell4.py        Shell-4 (cross-shell generalization)

paper/ns3_experiments/starlink_gdhdpq/
    step_1_generate_runs.py        build run configurations
    step_2_run.py                  execute ns-3 runs

hypatia_ai/
    gdh_dpq_v6e.py                 dual-decomposition solver (reference)
    standalone_v6e.py              full standalone pipeline
    multicommodity.py              destination regions, hub gateways, GW_SERVES
    hypatia_pop_instance.py        population traffic on Hypatia topology
    hypatia_baselines.py           baseline schemes
    traffic_gen.py                 time-varying population traffic
    neural_solver_MC.py            GNN construction policy, multicommodity
    neural_solver_MCpath.py        adds (gateway, path) action space
    unroll_solver.py               K-step unrolled residual policy
    train_MCpath*.py               training drivers
    temporal_*.py                  cross-epoch temporal formulation
    *_sweep.py                     capacity / radius / staleness / shell sweeps
```

---

## Algorithm parameters

| Parameter | Value | Meaning |
|---|---|---|
| `alpha1` | 1.0 | end-to-end delay weight (seconds) |
| `alpha2` | 0.02 | per-ISL-hop penalty |
| `alpha3` (`ALPHA_HO`) | 0.02 | handover penalty below `T_HO` |
| `alpha4` | 1e-4 | link-lifetime stability weight |
| `LAM_MAX`, `NU_MAX` | 100 | dual price bounds |
| `C_g` | 40 Gbps | gateway capacity |
| `C_isl` | 8 Gbps | ISL capacity |
| `D_MAX` | 0.5 s | end-to-end path delay budget |
| `L_MIN` | 30 s | minimum residual ISL lifetime |
| `MAX_SPLIT_PATHS` | 8 | successive paths per divisible flow |
| `DUAL_MAX_ITERS` | 80 | subgradient ascent budget |
| `DUAL_EARLY_STOP` | 20 | stop after N iterations without improvement |

Traffic is population-driven, from the
[GHS-POP R2023A](https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php)
raster: users under each satellite's 400 km footprint are drawn as a Poisson
process and clipped to a per-satellite demand range.

---

## Data paths

Scripts in `hypatia_ai/` carry absolute paths under `/home/msarwa02` and must be
edited before they will run anywhere else. `grep -rn "/home/msarwa02" hypatia_ai/`
lists every line; they fall into four groups.

**Input data.** The files the solver reads:

| Path | Contents |
|---|---|
| `FOLDER = r"/home/msarwa02"` | directory holding the GHS-POP raster, the Starlink TLE/CSV file, and the gateway KML |
| `standalone_mega_gw_km.npy` | gateway positions, ECEF km, shape `(99, 3)` |
| `ground_zone_metadata.csv` | per-zone population metadata used by the traffic generator |

**Hypatia run directory.** `GEN` points at a generated forwarding-state
directory of the form
`hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_gdh_dpq_over_isls`,
produced by `main_starlink_550.py`. The scripts that bridge the standalone
solver and the Hypatia topology read it.

**Module path.** Several short drivers do
`sys.path.insert(0, '"'"'/home/msarwa02/hypatia_ai'"'"')` rather than relying on
`PYTHONPATH`. Either edit the literal or export
`PYTHONPATH=$PWD/hypatia_ai` and drop the insert.

**Results.** Written to the same home directory: `results_log.txt`,
`hypatia_conv_scale*.npy`, `hypatia_v6e_ablation.npy`, `dual_metrics_3scale.npy`,
the per-sweep `*_results.txt` files, and the trained checkpoint
`policy_temporal.pt`. Sweep and plotting scripts read these back by the same
absolute path, so a change to where results are written has to be made in both
the producing and the consuming script.

---

## Reproduction

### Standalone solver (no ns-3 build required)

```bash
conda activate hypatia
cd hypatia_ai
python standalone_v6e.py
```

Requires the GHS-POP raster, the Starlink TLE/CSV file, and the gateway KML.
Paths are set at the top of the script.

Individual sweeps:

```bash
python cisl_sweep.py          # ISL capacity
python radius_sweep.py        # hotspot radius
python staleness_sweep.py     # price staleness
python shell_generalization.py  # cross-shell transfer
```

### Hypatia / ns-3

```bash
# 1. forwarding state
cd paper/satellite_networks_state
python main_starlink_550.py

# 2. build ns-3 (see Troubleshooting for the OpenFlow prerequisites)
cd ../../ns3-sat-sim/simulator
./waf configure --with-openflow=../../openflow_src
./waf build

# 3. run
cd ../../paper/ns3_experiments/starlink_gdhdpq
python step_1_generate_runs.py
python step_2_run.py
```

Generated data (`gen_data/`, ~600 MB) and run outputs (`runs/`, ~200 MB) are
not tracked; both are regenerated by the steps above.

---

## What the ns-3 experiments show

The ns-3 campaign validates that GDHR's routing decisions are correctly
realized at packet level: forwarding state is delivered intact, flows follow
the intended paths, and the OpenFlow controller reproduces the file-based
routing exactly.

**It does not establish a packet-level throughput advantage.** In a seeded
comparison (5 seeds, gateway-concentrated traffic), the schemes were not
statistically distinguishable — the between-method gap was smaller than the
across-seed standard deviation, and single-seed orderings flipped between
runs. Earlier single-seed numbers should not be read as gains. The measurable
advantage of dual pricing appears in served ratio and packet loss in the
standalone evaluation, not in ns-3 TCP throughput, where the bottleneck and
simulation window dominate.

---

## Troubleshooting

**OpenFlow not enabled after configure** — check boost is found; if the
`signals` library is missing, edit `src/openflow/wscript` and set
`REQUIRED_BOOST_LIBS = ['system', 'filesystem']`.

**ns-3 build fails with g++13** — in `ns3-sat-sim/simulator/src/wscript` add
`DISABLED_MODULES = ['lte', 'netanim', 'test', 'visualizer']`, then
`rm build/optimized/c4che/_cache.py` and rebuild.

**`basic-sim` fails to compile on modern g++/clang** — the upstream
`basic-sim` submodule iterates a `std::map` with the wrong `value_type`, which
newer compilers reject. `basic-sim` is a third-party submodule, so the fix is
shipped here as a patch rather than committed:

```bash
cd ns3-sat-sim/simulator/contrib/basic-sim
git apply ../../../../patches/basic-sim-const-pair.patch
```

The change is one line in `model/core/basic-simulation.cc`:
`std::pair<std::string, std::string>` becomes
`std::pair<const std::string, std::string>` in
`ConfirmAllConfigParamKeysRequested()`.

**Python import errors** — `export PYTHONPATH=$PWD/satgenpy:$PYTHONPATH`

**Solver runtime** — a full solve at the 1,200-satellite scale takes roughly
90 s at light load and several minutes under overload on a single core. The
cost is per topology snapshot, not per packet, and the per-satellite
shortest-path passes are independent and parallelizable.

---

## Citation

```bibtex
@article{sarwar2026gdhr,
  author  = {Sarwar, Muhammad Sajid and Hossain, Jahangir},
  title   = {When to Exit the Mesh: Gateway-Aware Routing for
             Congestion Control in {LEO} Satellite Networks},
  note    = {Under review},
  year    = {2026}
}
```

---

## License and acknowledgements

This work builds on [Hypatia](https://github.com/snkas/hypatia) (ETH Zurich),
distributed under the MIT License; see `LICENSE`. Population data is from the
EU Copernicus GHS-POP R2023A product.

## Contact

Muhammad Sajid Sarwar — Postdoctoral Fellow, School of Engineering,
UBC Okanagan — muhammadsajid.sarwar@ubc.ca
