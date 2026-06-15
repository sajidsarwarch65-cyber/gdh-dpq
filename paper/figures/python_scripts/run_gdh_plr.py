import sys
sys.path.insert(0,'/home/msarwa02/hypatia/satgenpy')
import satgen

output_dir="/home/msarwa02/hypatia/paper/satellite_networks_state/gen_data"
name="starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_gdh_dpq_over_isls"

MAX_GSL_LENGTH_M = 1089686.0  # from Starlink config
MAX_ISL_LENGTH_M = 5016000.0  # from Starlink +Grid config

print("Running GDH-DPQ dynamic state for 200 steps...")
satgen.help_dynamic_state(
    output_dir,
    1,           # num_threads
    name,
    1000,        # time_step_ms
    200,         # duration_s
    MAX_GSL_LENGTH_M,
    MAX_ISL_LENGTH_M,
    "algorithm_gdh_dpq_over_isls",
    True         # verbose - so we can see PLR/maxGW output
)
print("Done")
