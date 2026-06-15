import exputil, time
try:
    from .run_list import *
except (ImportError, SystemError):
    from run_list import *

local_shell = exputil.LocalShell()

commands = []
for run in get_tcp_run_list():
    run_dir = "runs/" + run["name"]
    cmd = (f"cd ../../../ns3-sat-sim/simulator; "
           f"./waf --run=\"main_satnet "
           f"--run_dir='../../paper/ns3_experiments/starlink_gdhdpq/{run_dir}'\" "
           f"2>&1 | tee '{run_dir}/logs_ns3/console.txt'")
    commands.append(cmd)

print(f"Running {len(commands)} ns-3 simulations...")
for i, cmd in enumerate(commands):
    print(f"Starting {i+1}/{len(commands)}")
    local_shell.perfect_exec(cmd)
print("Finished.")
