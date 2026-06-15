import os, random
import numpy as np

Ns = 1584  # satellites
Ng = 100   # ground stations
# Node IDs: sats 0..1583, GS 1584..1683

simulation_end_time_ns = 200 * 1000 * 1000 * 1000  # 200s
dynamic_state_update_interval_ns = 1000 * 1000 * 1000  # 1s
dynamic_state = "dynamic_state_1000ms_for_200s"

# Primary flow: Tokyo(GID=0,node=1584) -> London(GID=27,node=1611)
src_primary = Ns + 0    # Tokyo
dst_primary = Ns + 27   # London

# Background flows: random city pairs (excluding Tokyo and London)
random.seed(42)
n_background = 20  # 20 competing flows
all_gids = list(range(Ng))
bg_pairs = []
while len(bg_pairs) < n_background:
    s = random.choice(all_gids)
    d = random.choice(all_gids)
    if s != d and s not in [0,27] and d not in [0,27]:
        bg_pairs.append((Ns+s, Ns+d))

print(f"Primary flow: {src_primary} -> {dst_primary}")
print(f"Background flows: {n_background} pairs")

# Data rate: primary=10Mbps, background=5Mbps each
primary_rate_mbps = 10.0
bg_rate_mbps = 5.0

for algo in ["gdh_dpq", "ca_dsp", "dsp"]:
    run_dir = f"runs/starlink_{algo}_loaded"
    os.makedirs(run_dir+"/logs_ns3", exist_ok=True)

    sat_net = (f"starlink_550_isls_plus_grid_ground_stations_top_100"
               f"_algorithm_{algo}_over_isls")
    sat_net_dir = (f"../../../../satellite_networks_state/gen_data/{sat_net}")
    routes_dir  = sat_net_dir + f"/{dynamic_state}"

    config = f"""simulation_end_time_ns={simulation_end_time_ns}
simulation_seed=123456789
satellite_network_dir="{sat_net_dir}"
satellite_network_routes_dir="{routes_dir}"
dynamic_state_update_interval_ns={dynamic_state_update_interval_ns}
isl_data_rate_megabit_per_s=10.0
gsl_data_rate_megabit_per_s=10.0
isl_max_queue_size_pkts=100
gsl_max_queue_size_pkts=100
enable_isl_utilization_tracking=true
isl_utilization_tracking_interval_ns=1000000000
tcp_socket_type=TcpNewReno
enable_tcp_flow_scheduler=true
tcp_flow_schedule_filename="schedule.csv"
tcp_flow_enable_logging_for_tcp_flow_ids=set(0)
"""
    with open(run_dir+"/config_ns3.properties","w") as f:
        f.write(config)

    # Write schedule: flow 0 = primary, flows 1..N = background
    byte_size_primary = int(primary_rate_mbps*1e6/8 * 10)  # 10 Mbit
    byte_size_bg      = int(bg_rate_mbps*1e6/8 * 10)       # 5 Mbit
    with open(run_dir+"/schedule.csv","w") as f:
        # Primary flow starts at t=0
        f.write(f"0,{src_primary},{dst_primary},{byte_size_primary},0,,\n")
        # Background flows start at t=1s with random offsets
        for i,(s,d) in enumerate(bg_pairs):
            start_ns = random.randint(0, 5) * 1000000000  # 0-5s
            f.write(f"{i+1},{s},{d},{byte_size_bg},{start_ns},,\n")

    print(f"Generated: {run_dir} ({n_background} background flows)")

print("Done")
