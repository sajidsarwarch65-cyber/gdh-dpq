# algorithm_gdh_rl_over_isls.py  RL forwarding driver (mirrors the GDH-DPQ
# driver but calls the RL fstate generator). Same interface as the dual so the
# satgenpy dynamic-state pipeline invokes it identically.
from .fstate_calculation import *
from .fstate_gdh_rl import calculate_fstate_gdh_rl


def algorithm_gdh_rl_over_isls(
        output_dynamic_state_dir, time_since_epoch_ns,
        satellites, ground_stations,
        sat_net_graph_only_satellites_with_isls,
        ground_station_satellites_in_range,
        num_isls_per_sat, sat_neighbor_to_if,
        list_gsl_interfaces_info, prev_output, enable_verbose_logs):
    if enable_verbose_logs:
        print("\nALGORITHM: GDH-RL OVER ISLS")

    if sat_net_graph_only_satellites_with_isls.number_of_nodes() != len(satellites):
        raise ValueError("graph nodes != satellites")

    # bandwidth state (identical to dual driver)
    output_filename = (output_dynamic_state_dir + "/gsl_if_bandwidth_"
                       + str(time_since_epoch_ns) + ".txt")
    with open(output_filename, "w+") as f_out:
        if time_since_epoch_ns == 0:
            for node_id in range(len(satellites)):
                f_out.write("%d,%d,%f\n" % (node_id, num_isls_per_sat[node_id],
                    list_gsl_interfaces_info[node_id]["aggregate_max_bandwidth"]))
            for node_id in range(len(satellites), len(satellites) + len(ground_stations)):
                f_out.write("%d,%d,%f\n" % (node_id, 0,
                    list_gsl_interfaces_info[node_id]["aggregate_max_bandwidth"]))

    prev_fstate = prev_output.get("fstate") if prev_output is not None else None
    gid_to_sat_gsl_if_idx = [0] * len(ground_stations)
    import os
    scale = float(os.environ.get("GDHDPQ_SCALE", "1.0"))

    fstate, new_state = calculate_fstate_gdh_rl(
        output_dynamic_state_dir, time_since_epoch_ns,
        len(satellites), len(ground_stations),
        sat_net_graph_only_satellites_with_isls,
        num_isls_per_sat, gid_to_sat_gsl_if_idx,
        ground_station_satellites_in_range,
        sat_neighbor_to_if, prev_fstate, enable_verbose_logs,
        scale=scale,
    )
    return {"fstate": fstate}
