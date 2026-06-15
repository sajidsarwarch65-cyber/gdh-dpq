# GDH-DPQ: Gateway-Aware Dual-Price Routing for LEO Satellite Networks

Built on top of Hypatia (https://github.com/snkas/hypatia).

## Overview
GDH-DPQ uses Lagrangian relaxation with dual-price decomposition to jointly optimize ISL routing and gateway assignment under realistic traffic and capacity constraints. Implemented as a centralised SDN controller pushing OpenFlow flow rules to satellite nodes dynamically.

## Key Results
| Metric | Best Baseline | GDH-DPQ | Gain |
|--------|--------------|---------|------|
| PLR @ scale 1.0 | 1.78% (GPSR) | 0.02% | 89x lower |
| Primary flow (ns-3) | 6,668 Mbit (DSP) | 9,125 Mbit | +36.9% |
| Total network (ns-3) | 387 Gbit (DBPR) | 508 Gbit | +31.3% |
| Jain fairness (ns-3) | 0.740 (GPSR) | 0.830 | +12.2% |
| Median RTT (ns-3) | 155 ms (DBPR) | 119 ms | -23.5% |

## Key Files
- `satgenpy/satgen/dynamic_state/fstate_gdh_dpq.py`  GDH-DPQ core algorithm
- `satgenpy/satgen/dynamic_state/algorithm_baselines_over_isls.py`  5 baselines
- `ns3-sat-sim/simulator/src/openflow/model/gdh-dpq-controller.h/.cc`  SDN controller
- `paper/satellite_networks_state/main_starlink_550.py`  generate routing tables
- `paper/ns3_experiments/starlink_loaded/generate_loaded_runs.py`  ns-3 setup

## How to Run
See full instructions in the LaTeX report or contact the authors.

## Citation
Muhammad Sajid Sarwar and Jahangir Hossain, "GDH-DPQ: Gateway-Aware Dual-Price Routing for LEO Satellite Networks", 2025.

## Contact
Muhammad Sajid Sarwar, Postdoctoral Fellow, UBC Okanagan
Supervisor: Prof. Jahangir Hossain
