# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import random
import os
import numpy as np
from functools import partial
from itertools import chain
from operator import contains, eq, getitem
from pathlib import Path
import json
import copy
import multiprocess
import signal
import time
import psutil

from loguru import logger

from tests.scripts.common import run_process_and_get_result


@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)

    yield


@pytest.fixture(scope="function")
def function_level_defaults(reset_seeds):
    yield


@pytest.fixture(scope="session")
def model_location_generator():
    def model_location_generator_(model_version, model_subdir=""):
        model_folder = Path("tt_dnn-models") / model_subdir
        internal_weka_path = Path("/mnt/MLPerf") / model_folder / model_version
        has_internal_weka = internal_weka_path.exists()
        internal_cache_path = Path("/opt/tt-metal-models") / model_folder / model_version
        has_internal_cache = internal_cache_path.exists()
        if has_internal_weka:
            return internal_weka_path
        elif has_internal_cache:
            return internal_cache_path
        else:
            return model_version

    return model_location_generator_


@pytest.fixture(scope="session")
def get_tt_cache_path():
    def get_tt_cache_path_(model_version, model_subdir="", default_dir=""):
        model_folder = Path("tt_dnn-models/tt") / model_subdir
        internal_weka_path = Path("/mnt/MLPerf") / model_folder / model_version
        has_internal_weka = internal_weka_path.exists()
        internal_cache_path = Path("/opt/tt-metal-models") / model_folder / model_version
        has_internal_cache = internal_cache_path.exists()
        if has_internal_weka:
            return internal_weka_path
        elif has_internal_cache:
            return internal_cache_path
        else:
            default_path = Path(default_dir) / model_folder / model_version
            default_path.mkdir(parents=True, exist_ok=True)
            return default_path

    return get_tt_cache_path_


@pytest.fixture(scope="function")
def device_params(request):
    return getattr(request, "param", {})


@pytest.fixture(scope="function")
def device(request, device_params):
    import tt_lib as ttl

    device_id = request.config.getoption("device_id")
    request.node.pci_ids = [ttl.device.GetPCIeDeviceID(device_id)]

    num_devices = ttl.device.GetNumPCIeDevices()
    assert device_id < num_devices, "CreateDevice not supported for non-mmio device"
    device = ttl.device.CreateDevice(device_id=device_id, **device_params)
    ttl.device.SetDefaultDevice(device)

    yield device

    ttl.device.DumpDeviceProfiler(device)

    ttl.device.Synchronize(device)
    ttl.device.CloseDevice(device)


@pytest.fixture(scope="function")
def pcie_devices(request, device_params):
    import tt_lib as ttl

    num_devices = ttl.device.GetNumPCIeDevices()
    device_ids = [i for i in range(num_devices)]
    request.node.pci_ids = device_ids

    # Get only physical devices
    devices = ttl.device.CreateDevices(device_ids, **device_params)

    yield [devices[i] for i in range(num_devices)]

    for device in devices.values():
        ttl.device.DumpDeviceProfiler(device)

    ttl.device.CloseDevices(devices)


@pytest.fixture(scope="function")
def all_devices(request, device_params):
    import tt_lib as ttl

    num_devices = ttl.device.GetNumAvailableDevices()
    device_ids = [i for i in range(num_devices)]
    request.node.pci_ids = [ttl.device.GetPCIeDeviceID(i) for i in device_ids]

    # Get only physical devices
    devices = ttl.device.CreateDevices(device_ids, **device_params)

    yield [devices[i] for i in range(num_devices)]

    for device in devices.values():
        ttl.device.DumpDeviceProfiler(device)

    ttl.device.CloseDevices(devices)


@pytest.fixture(scope="function")
def device_mesh(request, silicon_arch_name, silicon_arch_wormhole_b0, device_params):
    import ttnn
    import tt_lib as ttl

    device_ids = ttnn.get_device_ids()

    try:
        param = request.param
    except (ValueError, AttributeError):
        param = len(device_ids)  # Default to using all available devices

    if isinstance(param, tuple):
        grid_dims = param
        assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
        device_grid = ttnn.DeviceGrid(*grid_dims)
        num_devices_requested = grid_dims[0] * grid_dims[1]
        assert num_devices_requested <= len(device_ids), "Requested more devices than available."
    else:
        num_devices_requested = min(param, len(device_ids))
        device_grid = ttnn.DeviceGrid(1, num_devices_requested)

    request.node.pci_ids = [ttl.device.GetPCIeDeviceID(i) for i in device_ids[:num_devices_requested]]

    device_mesh = ttnn.open_device_mesh(device_grid, device_ids[:num_devices_requested], **device_params)

    logger.debug(f"multidevice with {device_mesh.get_num_devices()} devices is created")
    yield device_mesh

    for device in device_mesh.get_devices():
        ttl.device.DumpDeviceProfiler(device)

    ttnn.close_device_mesh(device_mesh)
    del device_mesh


@pytest.fixture(scope="function")
def pcie_device_mesh(request, silicon_arch_name, silicon_arch_wormhole_b0, device_params):
    import ttnn
    import tt_lib as ttl

    device_ids = ttnn.get_pcie_device_ids()
    try:
        num_pcie_devices_requested = min(request.param, len(device_ids))
    except (ValueError, AttributeError):
        num_pcie_devices_requested = len(device_ids)

    request.node.pci_ids = device_ids[:num_pcie_devices_requested]

    device_mesh = ttnn.open_device_mesh(
        ttnn.DeviceGrid(1, num_pcie_devices_requested), device_ids[:num_pcie_devices_requested], **device_params
    )

    logger.debug(f"multidevice with {device_mesh.get_num_devices()} devices is created")
    yield device_mesh

    for device in device_mesh.get_devices():
        ttl.device.DumpDeviceProfiler(device)

    ttnn.close_device_mesh(device_mesh)
    del device_mesh


@pytest.fixture(scope="function")
def t3k_device_mesh(request, silicon_arch_name, silicon_arch_wormhole_b0, device_params):
    import ttnn
    import tt_lib as ttl

    if ttnn.get_num_devices() < 8:
        pytest.skip()
    device_ids = [0, 4, 5, 1, 2, 6, 7, 3]
    try:
        num_devices_requested = min(request.param, len(device_ids))
    except (ValueError, AttributeError):
        num_devices_requested = len(device_ids)

    request.node.pci_ids = [ttl.device.GetPCIeDeviceID(i) for i in device_ids[:num_devices_requested]]

    device_mesh = ttnn.open_device_mesh(
        ttnn.DeviceGrid(1, num_devices_requested), device_ids[:num_devices_requested], **device_params
    )

    logger.debug(f"multidevice with {device_mesh.get_num_devices()} devices is created")
    yield device_mesh

    for device in device_mesh.get_devices():
        ttl.device.DumpDeviceProfiler(device)

    ttnn.close_device_mesh(device_mesh)
    del device_mesh


@pytest.fixture()
def clear_compile_cache():
    yield
    import tt_lib as ttl

    ttl.device.DisablePersistentKernelCache()


@pytest.fixture(autouse=True)
def reset_default_device():
    import tt_lib as ttl

    device = ttl.device.GetDefaultDevice()
    yield
    ttl.device.SetDefaultDevice(device)


@pytest.fixture(scope="function")
def use_program_cache(request):
    import tt_lib as ttl

    if "device" in request.fixturenames:
        dev = request.getfixturevalue("device")
        dev.enable_program_cache()
    elif "all_devices" in request.fixturenames:
        devices = request.getfixturevalue("all_devices")
        for dev in devices:
            dev.enable_program_cache()
    elif "pcie_devices" in request.fixturenames:
        devices = request.getfixturevalue("pcie_devices")
        for dev in devices:
            dev.enable_program_cache()
    elif "device_mesh" in request.fixturenames:
        mesh = request.getfixturevalue("device_mesh")
        for device_id in mesh.get_device_ids():
            mesh.get_device(device_id).enable_program_cache()
    elif "t3k_device_mesh" in request.fixturenames:
        mesh = request.getfixturevalue("t3k_device_mesh")
        for device_id in mesh.get_device_ids():
            mesh.get_device(device_id).enable_program_cache()
    elif "pcie_device_mesh" in request.fixturenames:
        mesh = request.getfixturevalue("pcie_device_mesh")
        for device_id in mesh.get_device_ids():
            mesh.get_device(device_id).enable_program_cache()
    else:
        logger.warning("No device fixture found to apply program cache to: PROGRAM CACHE DISABLED")
    yield


@pytest.fixture(scope="function")
def tracy_profile():
    from tracy import Profiler

    profiler = Profiler()

    profiler.enable()
    yield
    profiler.disable()


###############################
# Modifying pytest hooks
###############################
ALL_ARCHS = set(
    [
        "grayskull",
        "wormhole_b0",
    ]
)


def pytest_addoption(parser):
    parser.addoption(
        "--tt-arch",
        choices=[*ALL_ARCHS],
        default=os.environ.get("ARCH_NAME", "grayskull"),
        help="Target arch, ex. grayskull, wormhole_b0",
    )
    parser.addoption(
        "--pipeline-type",
        default="",
        help="Only `models_device_performance_bare_metal` should run `pytest_runtest_teardown`",
    )
    parser.addoption(
        "--device-id",
        type=int,
        default=0,
        help="Target device id",
    )
    parser.addoption(
        "--input-method",
        action="store",
        choices=["json", "cli"],
        default=None,
        help="Choose input method: 1) json or 2) cli",
    )
    parser.addoption(
        "--input-path",
        action="store",
        default="",
        help="Path to json file with inputs",
    )
    parser.addoption("--cli-input", action="store", default=None, help="Enter prompt if --input-method=cli")
    parser.addoption(
        "--metal-timeout",
        action="store",
        default=None,
        help="Enable process timeout",
    )


@pytest.fixture
def input_path(request):
    return request.config.getoption("--input-path")


def pytest_generate_tests(metafunc):
    """
    This is not a standard docstring.

    We will explain the non-standard fixtures that pytest_generate_tests is
    creating here.

    silicon_arch_name and silicon_arch_<ARCH_NAME>
    ----------------------------------------------

    This is how tests should be requesting accelerator architecture names.
    Tests which aim to run on silicon should request a silicon_arch_name
    fixture. Just that single fixture will parametrize the test to run on the
    provided architecture name from the command line through the --tt-arch
    option. The value of the fixture will be the string value of the
    architecture name. For example,

    @pytest.mark.post_commit
    def test_model_silicon(silicon_arch_name):
        # silicon_arch_name will be one of grayskull, wormhole_b0 etc.
        run_model_on_silicon(silicon_arch_name)
        ...

    If you want to restrict a test to only a specific architecture, you can
    provide an additional fixture in the form of silicon_arch_<ARCH_NAME>. This
    will limit the range of possible values for silicon_arch_name to only be
    ARCH_NAME.

    @pytest.mark.post_commit
    def test_model_silicon_grayskull_only(
        silicon_arch_name,
        silicon_arch_grayskull,
    ):
        # silicon_arch_name can only be grayskull or empty
        run_model_on_silicon(silicon_arch_name)
        ...

    If --tt-arch specifies an architecture that's not ARCH_NAME, the test will
    be skipped. We ensure skipping by providing an empty list parametrization
    for silicon_arch_name, and with the empty_parameter_set_mark config option
    for pytest, will skip any tests with an empty list parametrization.

    Note that you must provide silicon_arch_name as a fixture if you want to
    use the silicon_arch_<ARCH_NAME> fixture.

    Note that if tests want to use the ARCH value from the API, tests should
    create their own separate fixture which will convert the string value
    provided from silicon_arch_name into ARCH. We keep it as strings here
    because these fixtures will be used in tests which do not have access to
    any Python APIs.
    """

    tt_arch = metafunc.config.getoption("--tt-arch")

    silicon_arch_specific_fixture_name_to_avail_archs = {
        "silicon_arch_grayskull": set(
            [
                "grayskull",
            ]
        ),
        "silicon_arch_wormhole_b0": set(
            [
                "wormhole_b0",
            ]
        ),
    }

    check_uses_silicon_arch_specific_fixture = partial(contains, silicon_arch_specific_fixture_name_to_avail_archs)
    test_requested_silicon_arch_fixtures = tuple(
        filter(check_uses_silicon_arch_specific_fixture, metafunc.fixturenames)
    )
    is_test_requesting_specific_silicon_archs = len(test_requested_silicon_arch_fixtures) > 0
    get_archs_for_silicon_arch_specific_fixture = partial(getitem, silicon_arch_specific_fixture_name_to_avail_archs)
    test_requested_silicon_archs = ALL_ARCHS.intersection(
        *map(
            get_archs_for_silicon_arch_specific_fixture,
            test_requested_silicon_arch_fixtures,
        )
    )

    available_archs = test_requested_silicon_archs if is_test_requesting_specific_silicon_archs else ALL_ARCHS
    matches_user_requested_silicon_arch = partial(eq, tt_arch)
    available_archs = tuple(filter(matches_user_requested_silicon_arch, available_archs))

    uses_silicon_arch = "silicon_arch_name" in metafunc.fixturenames

    # sanity
    if is_test_requesting_specific_silicon_archs and not uses_silicon_arch:
        raise Exception(
            f"{metafunc.function} requesting a specific silicon target, but doesn't use silicon_arch_name fixture"
        )

    if uses_silicon_arch:
        metafunc.parametrize("silicon_arch_name", available_archs)
        for test_requested_silicon_arch_fixture in test_requested_silicon_arch_fixtures:
            # The values of these arch-specific fixtures should not be used in
            # the test function, so use any parameters, like [True]
            metafunc.parametrize(test_requested_silicon_arch_fixture, [True])

    input_method = metafunc.config.getoption("--input-method")
    if input_method == "json":
        json_path = metafunc.config.getoption("--input-path")
        if not json_path:
            raise ValueError("Please provide a valid JSON path using --input-path option.")
        with open(json_path, "r") as f:
            data = json.load(f)
        metafunc.parametrize("user_input", [data])
    elif input_method == "cli":
        cli_input = metafunc.config.getoption("--cli-input")
        if not cli_input:
            raise ValueError("Please provide input using --cli-input option.")
        metafunc.parametrize("user_input", [[cli_input]])


# Report stashing to get outcomes etc
phase_report_key = pytest.StashKey()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # store test results for each phase of a call, which can
    # be "setup", "call", "teardown"
    item.stash.setdefault(phase_report_key, {})[rep.when] = rep


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    metal_timeout_enabled = item.config.getoption("--metal-timeout")
    using_xdist = int(os.getenv("PYTEST_XDIST_WORKER_COUNT", "0"))

    if metal_timeout_enabled is not None or using_xdist:
        report = item.stash[phase_report_key]
        test_failed = report.get("call", None) and report["call"].failed
        if test_failed:
            logger.info(f"In custom teardown, open device ids: {set(item.pci_ids)}")
            reset_tensix(set(item.pci_ids))


# This is overriding the timer setup hook from pytest-timeout
# If --metal-timeout is passed, we define a new timeout method that spawns a timer process
# At timeout, the process kills it's parent (the test process) and then itself
@pytest.hookimpl(tryfirst=True)
def pytest_timeout_set_timer(item, settings):
    metal_timeout_enabled = item.config.getoption("--metal-timeout")
    using_xdist = int(os.getenv("PYTEST_XDIST_WORKER_COUNT", "0"))

    if metal_timeout_enabled is not None or using_xdist:
        parent_pid = os.getpid()
        logger.info(f"Metal timeout {settings.timeout} seconds {parent_pid} for {item.nodeid}")

        def get_parent_status():
            try:
                parent = psutil.Process(parent_pid)
            except:
                return "already dead"
            return parent.status()

        def run_timer(settings):
            logger.info(f"Timer started for {item.nodeid}")
            dead_status = ["zombie", "dead", "already dead"]
            timeout = settings.timeout
            parent_status = "running"
            while parent_status not in dead_status and timeout > 0:
                time.sleep(5)
                timeout -= 5
                parent_status = get_parent_status()
            if parent_status != "already dead":
                logger.info(f"Timing out test case")
                os.kill(parent_pid, signal.SIGKILL)
            logger.info(f"Killing timer")
            os._exit(1)

        def cancel():
            logger.info(f"Cancelling timer")
            metal_timer.terminate()

        metal_timer = multiprocess.Process(target=run_timer, args=(settings,), daemon=True)
        item.cancel_timeout = cancel
        metal_timer.start()
    return True


# This is a hook used in pytest-xdist to handle when a worker crashes out
# In our case, combined with pytest-timeout thread method, the worker will crash out for a hang and
# then it should get cleaned up by the controller through this fixture
@pytest.hookimpl(tryfirst=True)
def pytest_handlecrashitem(crashitem, report, sched):
    reset_tensix()


def reset_tensix(tt_open_devices=None):
    metal_env = copy.deepcopy(os.environ)
    arch = metal_env.get("ARCH_NAME")
    if arch != "grayskull" and arch != "wormhole_b0":
        raise Exception(f"Unrecognized arch for tensix-reset: {arch}")

    if tt_open_devices is None:
        logger.info(f"Running reset with reset script: /opt/tt_metal_infra/scripts/ci/{arch}/reset.sh")
        smi_reset_result = run_process_and_get_result(f"/opt/tt_metal_infra/scripts/ci/{arch}/reset.sh")
    else:
        tt_open_devices_str = ",".join([str(i) for i in tt_open_devices])
        check_smi_metal = run_process_and_get_result("tt-smi-metal -h")
        logger.info(f"Running reset for pci devices: {tt_open_devices_str}")
        if check_smi_metal.returncode > 0:
            logger.info(f"Test failed - resetting {arch} with tt-smi")
            smi_reset_result = run_process_and_get_result(f"tt-smi -r {tt_open_devices_str}")
        else:
            smi_reset_result = run_process_and_get_result(f"tt-smi-metal -r {tt_open_devices_str}")
    logger.info(f"tt-smi reset status: {smi_reset_result.returncode}")


@pytest.hookimpl(tryfirst=True)
def pytest_xdist_auto_num_workers(config):
    return 1
