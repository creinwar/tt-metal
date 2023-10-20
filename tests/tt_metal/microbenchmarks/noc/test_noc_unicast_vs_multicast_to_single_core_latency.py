import os
import sys
sys.path.append(f"{os.environ['PYTHONPATH']}/tt_metal/tools/profiler")

from loguru import logger
import pytest
from process_device_log import import_log_run_stats
import plot_setup


def test_noc_unicast_vs_multicast_to_single_core_latency():
    os.system(f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/noc/test_noc_unicast_vs_multicast_to_single_core_latency")
    setup = plot_setup.default_setup()
    setup.timerAnalysis = {
        "LATENCY": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "timerID": 5},
            "end": {"risc": "BRISC", "timerID": 6},
        },
    }
    setup.deviceInputLog = "unicast_to_single_core_microbenchmark/profile_log_device.csv"
    stats = import_log_run_stats(setup)
    core = [key for key in stats["devices"][0]["cores"].keys() if key != "DEVICE"][0]
    unicast_cycles = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"]["LATENCY"]["stats"]["First"]

    setup.deviceInputLog = "multicast_to_single_core_microbenchmark/profile_log_device.csv"
    stats = import_log_run_stats(setup)
    multicast_cycles = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"]["LATENCY"]["stats"]["First"]

    unicast_cycles_lower_bound = 330
    unicast_cycles_upper_bound = 400
    unicast_cycles_within_bounds = unicast_cycles_lower_bound <= unicast_cycles <= unicast_cycles_upper_bound

    multicast_cycles_lower_bound = 450
    multicast_cycles_upper_bound = 600
    multicast_cycles_within_bounds = multicast_cycles_lower_bound <= multicast_cycles <= multicast_cycles_upper_bound

    if not unicast_cycles_within_bounds:
        logger.warning(f"Unicast cycles not within bounds. Received {unicast_cycles}, was expecting between {unicast_cycles_lower_bound} and {unicast_cycles_upper_bound}")
    else:
        logger.info(f"Unicast cycles within bounds. Received {unicast_cycles}")

    if not multicast_cycles_within_bounds:
        logger.warning(f"Multicast cycles not within bounds. Received {multicast_cycles}, was expecting between {multicast_cycles_lower_bound} and {multicast_cycles_upper_bound}")
    else:
        logger.info(f"Multicast cycles within bounds. Received {multicast_cycles}")

    assert unicast_cycles_within_bounds and multicast_cycles_within_bounds
