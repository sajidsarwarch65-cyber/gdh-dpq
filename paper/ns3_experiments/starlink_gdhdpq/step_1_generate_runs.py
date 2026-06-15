import os
import exputil
try:
    from .run_list import *
except (ImportError, SystemError):
    from run_list import *

local_shell = exputil.LocalShell()

for run in get_tcp_run_list():
    run_dir = "runs/" + run["name"]
    os.makedirs(run_dir + "/logs_ns3", exist_ok=True)

    # Write config
    sat_net_dir = (f"../../../../satellite_networks_state/gen_data/"
                   f"{run['satellite_network']}")
    routes_dir  = sat_net_dir + f"/{run['dynamic_state']}"

    config = f"""simulation_end_time_ns={run['simulation_end_time_ns']}
simulation_seed=123456789
satellite_network_dir="{sat_net_dir}"
satellite_network_routes_dir="{routes_dir}"
dynamic_state_update_interval_ns={run['dynamic_state_update_interval_ns']}
isl_data_rate_megabit_per_s={run['data_rate_megabit_per_s']}
gsl_data_rate_megabit_per_s={run['data_rate_megabit_per_s']}
isl_max_queue_size_pkts={run['queue_size_pkt']}
gsl_max_queue_size_pkts={run['queue_size_pkt']}
enable_isl_utilization_tracking={"true" if run['enable_isl_utilization_tracking'] else "false"}
isl_utilization_tracking_interval_ns={run['isl_utilization_tracking_interval_ns']}
tcp_socket_type={run['tcp_socket_type']}
enable_tcp_flow_scheduler=true
tcp_flow_schedule_filename="schedule.csv"
tcp_flow_enable_logging_for_tcp_flow_ids=set(0)
"""
    with open(run_dir + "/config_ns3.properties", "w") as f:
        f.write(config)

    # Write schedule: one flow from from_id to to_id at t=0
    with open(run_dir + "/schedule.csv", "w") as f:
        f.write(f"0,{run['from_id']},{run['to_id']},{int(run['data_rate_megabit_per_s']*1e6/8)},0,,\n")

    print(f"Generated: {run['name']}")

print("Success: generated runs")
