dynamic_state_update_interval_ms = 1000
simulation_end_time_s = 6
isl_utilization_tracking_interval_ns = 1 * 1000 * 1000 * 1000

dynamic_state_update_interval_ns = dynamic_state_update_interval_ms * 1000 * 1000
simulation_end_time_ns = simulation_end_time_s * 1000 * 1000 * 1000
dynamic_state = ("dynamic_state_" + str(dynamic_state_update_interval_ms)
                 + "ms_for_" + str(simulation_end_time_s) + "s")

Ns = 1584

# Correct city GIDs:
# Tokyo=0(1584), London=27(1611), Delhi=1(1585), New York=32(1616)
# Paris=24(1608), Moscow=21(1605), Mumbai=3(1587), Beijing=4(1588)
chosen_pairs = [
    ("starlink_gdhdpq",     Ns+0,  Ns+27, "TcpNewReno", "gdh_dpq"),    # Tokyo->London
    ("starlink_gdhdpq",     Ns+0,  Ns+24, "TcpNewReno", "gdh_dpq"),    # Tokyo->Paris
    ("starlink_gdhdpq",     Ns+1,  Ns+32, "TcpNewReno", "gdh_dpq"),    # Delhi->NY
    ("starlink_gdhrl",      Ns+0,  Ns+27, "TcpNewReno", "gdh_rl"),     # Tokyo->London (RL)
    ("starlink_gdhrl",      Ns+0,  Ns+24, "TcpNewReno", "gdh_rl"),     # Tokyo->Paris  (RL)
    ("starlink_gdhrl",      Ns+1,  Ns+32, "TcpNewReno", "gdh_rl"),     # Delhi->NY     (RL)
    ("starlink_cadsp",      Ns+0,  Ns+27, "TcpNewReno", "ca_dsp"),     # Tokyo->London
    ("starlink_dsp",        Ns+0,  Ns+27, "TcpNewReno", "dsp"),        # Tokyo->London
    ("starlink_gpsr",       Ns+0,  Ns+27, "TcpNewReno", "gpsr"),       # Tokyo->London
    ("starlink_directdrop", Ns+0,  Ns+27, "TcpNewReno", "direct_drop"),# Tokyo->London
]

def get_tcp_run_list():
    run_list = []
    for p in chosen_pairs:
        algo = p[4]
        sat_net = (f"starlink_550_isls_plus_grid_ground_stations_top_100"
                   f"_algorithm_{algo}_over_isls")
        run_list.append({
            "name": f"{p[0]}_{p[1]}_to_{p[2]}_with_{p[3]}_at_10_Mbps",
            "satellite_network": sat_net,
            "dynamic_state": dynamic_state,
            "dynamic_state_update_interval_ns": dynamic_state_update_interval_ns,
            "simulation_end_time_ns": simulation_end_time_ns,
            "data_rate_megabit_per_s": 10.0,
            "queue_size_pkt": 100,
            "enable_isl_utilization_tracking": True,
            "isl_utilization_tracking_interval_ns": isl_utilization_tracking_interval_ns,
            "from_id": p[1],
            "to_id": p[2],
            "tcp_socket_type": p[3],
        })
    return run_list
