"""GPU device probe and selection (spec 13, decisions V3-F2 and V4-6)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from spells.gpu import (
    BackendDevice,
    GpuDevice,
    GpuSelection,
    engine_env,
    list_devices,
    parse_backend_info,
    parse_list_devices,
    probe_backend_info,
    probe_raw_indices,
    select_device,
)
from spells.platform.stub import StubProcesses

from .fake_platform import fake_platform


class HiddenWindowProcesses(StubProcesses):
    def __init__(self):
        super().__init__("linux")

    def hidden_process_kwargs(self):
        return {"creationflags": 0x08000000}


LLAMA = Path("C:/engines/vulkan/llama-server.exe")
WHISPER = Path("C:/engines/vulkan/whisper-server.exe")
RTX = "NVIDIA GeForce RTX 5060 Laptop GPU"
AMD = "AMD Radeon(TM) 610M"

RTX_LINE = (
    f"ggml_vulkan: 0 = {RTX} (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 "
    "| shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2"
)
AMD_LINE = (
    f"ggml_vulkan: 0 = {AMD} (AMD proprietary driver) | uma: 1 | fp16: 1 | bf16: 0 | fp4: 0 "
    "| warp size: 32 | shared memory: 32768 | int dot: 1 | matrix cores: none"
)


def backend_output(line: str) -> str:
    return (
        f"ggml_vulkan: Found 1 Vulkan devices:\n{line}\n"
        "load_backend: loaded RPC backend from C:\\engines\\vulkan\\ggml-rpc.dll\n"
        f"ggml_vulkan: Found 1 Vulkan devices:\n{line}\n"
        "load_backend: loaded Vulkan backend from C:\\engines\\vulkan\\ggml-vulkan.dll\n"
        "load_backend: loaded CPU backend from C:\\engines\\vulkan\\ggml-cpu-zen4.dll\n"
        "whisper_init_from_file_with_params_no_state: loading model from 'C:\\engines\\x.bin'\n"
        "whisper_init_from_file_with_params_no_state: failed to open 'C:\\engines\\x.bin'\n"
        "error: failed to initialize whisper context\n"
    )


RTX_BACKEND = backend_output(RTX_LINE)
AMD_BACKEND = backend_output(AMD_LINE)
BOTH_BACKEND = (
    "ggml_vulkan: Found 2 Vulkan devices:\n"
    f"{RTX_LINE}\n"
    f"{AMD_LINE.replace('ggml_vulkan: 0 =', 'ggml_vulkan: 1 =')}\n"
)
RTX_DISCRETE = BackendDevice(name=RTX, vendor="NVIDIA", uma=False)
AMD_INTEGRATED = BackendDevice(name=AMD, vendor="AMD proprietary driver", uma=True)

TWO_DEVICES = (
    "Available devices:\n"
    f"  Vulkan0: {RTX} (7810 MiB, 7042 MiB free)\n"
    f"  Vulkan1: {AMD} (48688 MiB, 46253 MiB free)\n"
)
ONE_RTX = f"Available devices:\n  Vulkan0: {RTX} (7810 MiB, 7042 MiB free)\n"
ONE_AMD = f"Available devices:\n  Vulkan0: {AMD} (48688 MiB, 46253 MiB free)\n"
NONE = "Available devices:\n  (none)\n"
INVALID = (
    "ggml_vulkan: Invalid device index 2 in GGML_VK_VISIBLE_DEVICES.\n"
    "ggml_vulkan: Invalid device index 2 in GGML_VK_VISIBLE_DEVICES.\n"
    "Available devices:\n"
    "  (none)\n"
)



def test_parse_two_devices_from_real_output():
    devices = parse_list_devices(TWO_DEVICES)
    assert devices == [
        GpuDevice(raw_index=None, list_name="Vulkan0", name=RTX, memory_mb=7810, free_mb=7042),
        GpuDevice(raw_index=None, list_name="Vulkan1", name=AMD, memory_mb=48688, free_mb=46253),
    ]


def test_parse_none_and_empty():
    assert parse_list_devices(NONE) == []
    assert parse_list_devices("") == []
    assert parse_list_devices("Available devices:\n") == []


def test_parse_tolerates_junk_blank_lines_and_windows_line_endings():
    text = (
        "\r\n"
        "load_backend: loaded Vulkan backend from ggml-vulkan.dll\r\n"
        "ggml_vulkan: Invalid device index 5 in GGML_VK_VISIBLE_DEVICES.\r\n"
        "\r\n"
        "Available devices:\r\n"
        "\r\n"
        f"  Vulkan0: {RTX} (7810 MiB, 7042 MiB free)\r\n"
        "  this is not a device line\r\n"
        "  CUDA0: some card (1 MiB, 1 MiB free)\r\n"
    )
    devices = parse_list_devices(text)
    assert [d.list_name for d in devices] == ["Vulkan0"]
    assert devices[0].name == RTX


def test_list_devices_runs_hidden_with_timeout_and_merges_stderr(use_platform):
    use_platform(fake_platform(processes=HiddenWindowProcesses()))
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "Available devices:\n", ONE_RTX)

    devices = list_devices(LLAMA, env={"X": "1"}, runner=runner)
    assert [d.name for d in devices] == [RTX]
    args, kwargs = calls[0]
    assert args == [str(LLAMA), "--list-devices"]
    assert kwargs["timeout"] == 20
    assert kwargs["creationflags"] & 0x08000000
    assert kwargs["env"] == {"X": "1"}
    assert kwargs["cwd"] == str(LLAMA.parent)


def test_list_devices_returns_empty_on_timeout_or_launch_error():
    def timing_out(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 20)

    def missing(args, **kwargs):
        raise OSError("not found")

    assert list_devices(LLAMA, runner=timing_out) == []
    assert list_devices(LLAMA, runner=missing) == []



def make_runner(per_index: dict[str | None, str], backend: dict[str, str] | None = None):
    """A fake llama-server whose listing depends on GGML_VK_VISIBLE_DEVICES, plus a fake
    whisper-server whose backend probe output (stderr) depends on it as well."""
    calls: list[str | None] = []
    whisper_calls: list[tuple[str | None, list[str], dict]] = []

    def runner(args, **kwargs):
        env = kwargs.get("env") or {}
        index = env.get("GGML_VK_VISIBLE_DEVICES")
        if Path(args[0]).name == "whisper-server.exe":
            whisper_calls.append((index, list(args), kwargs))
            stderr = (backend or {}).get(index, "error: failed to initialize whisper context\n")
            return subprocess.CompletedProcess(args, 3, "", stderr)
        calls.append(index)
        return subprocess.CompletedProcess(args, 0, per_index.get(index, INVALID), "")

    runner.calls = calls
    runner.whisper_calls = whisper_calls
    return runner


def test_probe_maps_raw_indices_and_stops_at_first_empty_run():
    runner = make_runner({None: TWO_DEVICES, "0": ONE_RTX, "1": ONE_AMD})
    devices = probe_raw_indices(LLAMA, env={}, runner=runner)
    assert [(d.raw_index, d.name) for d in devices] == [(0, RTX), (1, AMD)]
    assert devices[0].list_name == "Vulkan0"
    assert runner.calls == [None, "0", "1", "2"]


def test_probe_skips_a_run_that_does_not_filter_to_one_device():
    runner = make_runner({None: TWO_DEVICES, "0": TWO_DEVICES, "1": ONE_AMD})
    devices = probe_raw_indices(LLAMA, env={}, runner=runner)
    assert [(d.raw_index, d.name) for d in devices] == [(1, AMD)]


def test_probe_strips_an_inherited_visible_devices_setting_for_the_unfiltered_run():
    seen_env = []

    def runner(args, **kwargs):
        seen_env.append(dict(kwargs.get("env") or {}))
        return subprocess.CompletedProcess(args, 0, NONE, "")

    probe_raw_indices(LLAMA, env={"GGML_VK_VISIBLE_DEVICES": "1", "KEEP": "y"}, runner=runner)
    assert seen_env == [{"KEEP": "y"}]



def test_parse_backend_info_on_this_machines_two_devices():
    assert parse_backend_info(BOTH_BACKEND) == [RTX_DISCRETE, AMD_INTEGRATED]


def test_parse_backend_info_dedupes_the_repeated_block_and_ignores_the_rest():
    assert parse_backend_info(RTX_BACKEND) == [RTX_DISCRETE]
    assert parse_backend_info(AMD_BACKEND) == [AMD_INTEGRATED]
    assert parse_backend_info("error: failed to initialize whisper context\n") == []
    assert parse_backend_info("") == []


def test_probe_backend_info_is_none_on_failure_or_ambiguity():
    def timing_out(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 20)

    def missing(args, **kwargs):
        raise OSError("not found")

    assert probe_backend_info(WHISPER, 0, env={}, runner=timing_out) is None
    assert probe_backend_info(WHISPER, 0, env={}, runner=missing) is None
    assert probe_backend_info(WHISPER, 0, env={}, runner=make_runner({}, backend={})) is None
    two = make_runner({}, backend={"0": BOTH_BACKEND})
    assert probe_backend_info(WHISPER, 0, env={}, runner=two) is None



def two_gpu_runner(backend: dict[str, str] | None = None):
    return make_runner({None: TWO_DEVICES, "0": ONE_RTX, "1": ONE_AMD}, backend=backend)


REAL_BACKEND = {"0": RTX_BACKEND, "1": AMD_BACKEND}


def test_select_most_memory_without_backend_information():
    selection = select_device(LLAMA, env={}, runner=two_gpu_runner())
    assert isinstance(selection, GpuSelection)
    assert (selection.raw_index, selection.name, selection.memory_mb) == (1, AMD, 48688)
    assert [d.raw_index for d in selection.devices] == [0, 1]
    assert [d.uma for d in selection.devices] == [None, None]


def test_select_prefers_the_discrete_gpu_over_the_integrated_one_with_more_memory():
    runner = two_gpu_runner(REAL_BACKEND)
    selection = select_device(LLAMA, env={}, runner=runner, whisper_server=WHISPER)
    assert (selection.raw_index, selection.name, selection.memory_mb) == (0, RTX, 7810)
    assert [(d.raw_index, d.uma, d.vendor) for d in selection.devices] == [
        (0, False, "NVIDIA"),
        (1, True, "AMD proprietary driver"),
    ]
    assert [index for index, _, _ in runner.whisper_calls] == ["0", "1"]


def test_select_override_still_wins_over_the_class_rule():
    runner = two_gpu_runner(REAL_BACKEND)
    by_index = select_device(LLAMA, override=1, env={}, runner=runner, whisper_server=WHISPER)
    assert (by_index.raw_index, by_index.name) == (1, AMD)
    by_name = select_device(LLAMA, override=AMD, env={}, runner=runner, whisper_server=WHISPER)
    assert (by_name.raw_index, by_name.name) == (1, AMD)
    stale = select_device(LLAMA, cached=(1, RTX), env={}, runner=runner, whisper_server=WHISPER)
    assert (stale.raw_index, stale.name) == (0, RTX)


def test_a_cached_integrated_gpu_gives_way_to_a_discrete_one_that_appeared_later():
    runner = two_gpu_runner(REAL_BACKEND)

    selection = select_device(LLAMA, cached=(1, AMD), env={}, runner=runner, whisper_server=WHISPER)

    assert (selection.raw_index, selection.name) == (0, RTX)


def test_a_cached_gpu_is_kept_when_no_device_is_in_a_better_class():
    runner = make_runner({None: TWO_DEVICES, "0": ONE_RTX, "1": ONE_AMD},
                         backend={"0": AMD_BACKEND.replace(AMD, RTX), "1": AMD_BACKEND})

    selection = select_device(LLAMA, cached=(0, RTX), env={}, runner=runner, whisper_server=WHISPER)

    assert (selection.raw_index, selection.name) == (0, RTX)


def test_select_falls_back_to_most_memory_when_the_backend_probe_fails():
    runner = two_gpu_runner(backend={})
    selection = select_device(LLAMA, env={}, runner=runner, whisper_server=WHISPER)
    assert (selection.raw_index, selection.name) == (1, AMD)
    assert [d.uma for d in selection.devices] == [None, None]


def test_select_ranks_unknown_above_integrated():
    runner = two_gpu_runner(backend={"1": AMD_BACKEND})
    selection = select_device(LLAMA, env={}, runner=runner, whisper_server=WHISPER)
    assert (selection.raw_index, selection.name) == (0, RTX)
    assert [d.uma for d in selection.devices] == [None, True]


def test_select_unknown_override_falls_back_to_the_rule():
    selection = select_device(LLAMA, override=7, env={}, runner=two_gpu_runner())
    assert selection.raw_index == 1
    selection = select_device(LLAMA, override="Ghost GPU", env={}, runner=two_gpu_runner())
    assert selection.raw_index == 1


def test_select_drops_cached_choice_when_the_name_at_that_index_changed():
    runner = make_runner({None: TWO_DEVICES, "0": ONE_AMD, "1": ONE_RTX})
    selection = select_device(LLAMA, cached=(0, RTX), env={}, runner=runner)
    assert (selection.raw_index, selection.name) == (0, AMD)


def test_select_drops_cached_index_that_no_longer_exists():
    runner = make_runner({None: ONE_RTX, "0": ONE_RTX})
    selection = select_device(LLAMA, cached=(1, AMD), env={}, runner=runner)
    assert (selection.raw_index, selection.name) == (0, RTX)


def test_select_none_when_no_vulkan_device():
    selection = select_device(LLAMA, env={}, runner=make_runner({None: NONE}))
    assert selection == GpuSelection(raw_index=None, name="", memory_mb=0, devices=())


def test_engine_env_sets_the_raw_index():
    selection = GpuSelection(raw_index=1, name=AMD, memory_mb=48688, devices=())
    env = engine_env(selection, base={"PATH": "p", "GGML_VK_VISIBLE_DEVICES": "0"})
    assert env == {"PATH": "p", "GGML_VK_VISIBLE_DEVICES": "1"}


def test_engine_env_removes_the_variable_without_a_gpu():
    selection = GpuSelection(raw_index=None, name="", memory_mb=0, devices=())
    env = engine_env(selection, base={"PATH": "p", "GGML_VK_VISIBLE_DEVICES": "0"})
    assert env == {"PATH": "p"}

