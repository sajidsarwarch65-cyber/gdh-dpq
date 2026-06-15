# =============================================================
# algorithm_gdh_dpq_over_isls.py
#
# Drop this file into:
#   ~/hypatia/satgenpy/satgen/dynamic_state/algorithm_gdh_dpq_over_isls.py
#
# This is a drop-in replacement for algorithm_free_one_only_over_isls.py
# It calls GDH-DPQ instead of Floyd-Warshall shortest path.
# =============================================================

from .fstate_calculation import *
from .fstate_gdh_dpq import calculate_fstate_gdh_dpq


def algorithm_gdh_dpq_over_isls(
        output_dynamic_state_dir,
        time_since_epoch_ns,
        satellites,
        ground_stations,
        sat_net_graph_only_satellites_with_isls,
        ground_station_satellites_in_range,
        num_isls_per_sat,
        sat_neighbor_to_if,
        list_gsl_interfaces_info,
        prev_output,
        enable_verbose_logs
):
    """
    GDH-DPQ OVER INTER-SATELLITE LINKS ALGORITHM

    Same interface as algorithm_free_one_only_over_isls but replaces
    Floyd-Warshall shortest-path with a two-stage Lagrangian dual
    routing algorithm that is:
      - capacity-aware  (gateway + ISL load constraints via λ and ν)
      - queue-aware     (M/D/1 delay feedback from previous time step)
      - stability-aware (handover penalty for short-lifetime ISL links)
    """

    if enable_verbose_logs:
        print("\nALGORITHM: GDH-DPQ OVER ISLS")

    # Validate graph
    if sat_net_graph_only_satellites_with_isls.number_of_nodes() != len(satellites):
        raise ValueError("Number of nodes in graph does not match number of satellites")
    for sid in range(len(satellites)):
        for n in sat_net_graph_only_satellites_with_isls.neighbors(sid):
            if n >= len(satellites):
                raise ValueError("Graph cannot contain satellite-to-ground-station links")

    # ── Bandwidth state (identical to baseline) ───────────────
    output_filename = (output_dynamic_state_dir
                       + "/gsl_if_bandwidth_"
                       + str(time_since_epoch_ns) + ".txt")
    if enable_verbose_logs:
        print("  > Writing interface bandwidth state to: " + output_filename)

    with open(output_filename, "w+") as f_out:
        if time_since_epoch_ns == 0:
            for node_id in range(len(satellites)):
                f_out.write("%d,%d,%f\n" % (
                    node_id,
                    num_isls_per_sat[node_id],
                    list_gsl_interfaces_info[node_id]["aggregate_max_bandwidth"]
                ))
            for node_id in range(len(satellites), len(satellites) + len(ground_stations)):
                f_out.write("%d,%d,%f\n" % (
                    node_id,
                    0,
                    list_gsl_interfaces_info[node_id]["aggregate_max_bandwidth"]
                ))

    # ── Forwarding state via GDH-DPQ ─────────────────────────
    prev_fstate      = None
    prev_gdhdpq_state = None

    if prev_output is not None:
        prev_fstate       = prev_output.get("fstate")
        prev_gdhdpq_state = prev_output.get("gdhdpq_state")

    gid_to_sat_gsl_if_idx = [0] * len(ground_stations)

    fstate, new_gdhdpq_state = calculate_fstate_gdh_dpq(
        output_dynamic_state_dir,
        time_since_epoch_ns,
        len(satellites),
        len(ground_stations),
        sat_net_graph_only_satellites_with_isls,
        num_isls_per_sat,
        gid_to_sat_gsl_if_idx,
        ground_station_satellites_in_range,
        sat_neighbor_to_if,
        prev_fstate,
        enable_verbose_logs,
        gdhdpq_state=prev_gdhdpq_state,
    )

    if enable_verbose_logs:
        print("")

    return {
        "fstate":       fstate,
        "gdhdpq_state": new_gdhdpq_state,
    }
