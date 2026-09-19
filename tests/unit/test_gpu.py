"""GPU device probe and selection (spec 13, decisions V3-F2 and V4-6)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spells.gpu import (
    PROBE_MODEL_NAME,
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

LLAMA = Path("C:/engines/vulkan/llama-server.exe")
WHISPER = Path("C:/engines/vulkan/whisper-server.exe")
RTX = "NVIDIA GeForce RTX 5060 Laptop GPU"
AMD = "AMD Radeon(TM) 610M"

# Verbatim stderr of the Vulkan whisper-server 1.9.4 on the reference laptop when started
# with GGML_VK_VISIBLE_DEVICES=0 (RTX) and =1 (Radeon 610M) and a model path that does not
# exist; the device block prints once per ggml backend library that loads.
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


# parse_list_devices -----------------------------------------------------------


def test_parse_two_devices_from_real_output():
    devices = parse_list_devices(TWO_DEVICES)
    assert devices == [
        GpuDevice(raw_index=None, list_name="Vulkan0", name=RTX, memory_mb=7810, free_mb=7042),
        GpuDevice(raw_index=None, list_name="Vulkan1", name=AMD, memory_mb=48688, free_mb=46253),
    ]


def test_parse_one_device():
    devices = parse_list_devices(ONE_AMD)
    assert [d.name for d in devices] == [AMD]
    assert devices[0].memory_mb == 48688


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


def test_parse_keeps_parentheses_inside_device_names():
    devices = parse_list_devices(ONE_AMD)
    assert devices[0].name == "AMD Radeon(TM) 610M"


# list_devices -----------------------------------------------------------------


def test_list_devices_runs_hidden_with_timeout_and_merges_stderr():
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "Available devices:\n", ONE_RTX)

    devices = list_devices(LLAMA, env={"X": "1"}, runner=runner)
    assert [d.name for d in devices] == [RTX]
    args, kwargs = calls[0]
    assert args == [str(LLAMA), "--list-devices"]
    assert kwargs["timeout"] == 20
    assert kwargs["creationflags"] & 0x08000000  # CREATE_NO_WINDOW
    assert kwargs["env"] == {"X": "1"}
    assert kwargs["cwd"] == str(LLAMA.parent)


def test_list_devices_returns_empty_on_timeout_or_launch_error():
    def timing_out(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 20)

    def missing(args, **kwargs):
        raise OSError("not found")

    assert list_devices(LLAMA, runner=timing_out) == []
    assert list_devices(LLAMA, runner=missing) == []


# probe_raw_indices --------------------------------------------------------------


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
    # Unfiltered first, then 0, 1, and the empty run at 2 ends the probe before N + 2.
    assert runner.calls == [None, "0", "1", "2"]


def test_probe_reversed_raw_order_is_reported_as_seen():
    runner = make_runner({None: TWO_DEVICES, "0": ONE_AMD, "1": ONE_RTX})
    devices = probe_raw_indices(LLAMA, env={}, runner=runner)
    assert [(d.raw_index, d.name) for d in devices] == [(0, AMD), (1, RTX)]


def test_probe_with_no_devices_makes_no_filtered_runs():
    runner = make_runner({None: NONE})
    assert probe_raw_indices(LLAMA, env={}, runner=runner) == []
    assert runner.calls == [None]


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


# parse_backend_info and probe_backend_info (decision B3-59) -----------------------


def test_parse_backend_info_on_this_machines_two_devices():
    assert parse_backend_info(BOTH_BACKEND) == [RTX_DISCRETE, AMD_INTEGRATED]


def test_parse_backend_info_dedupes_the_repeated_block_and_ignores_the_rest():
    assert parse_backend_info(RTX_BACKEND) == [RTX_DISCRETE]
    assert parse_backend_info(AMD_BACKEND) == [AMD_INTEGRATED]
    assert parse_backend_info("error: failed to initialize whisper context\n") == []
    assert parse_backend_info("") == []


def test_parse_backend_info_keeps_parentheses_in_names_and_reads_the_vendor():
    line = "ggml_vulkan: 0 = Intel(R) Arc(TM) A770 Graphics (Intel Corporation) | uma: 0 | x"
    assert parse_backend_info(line) == [
        BackendDevice(name="Intel(R) Arc(TM) A770 Graphics", vendor="Intel Corporation", uma=False)
    ]


def test_probe_backend_info_runs_whisper_server_hidden_with_a_missing_model():
    runner = make_runner({}, backend={"1": AMD_BACKEND})
    info = probe_backend_info(WHISPER, 1, env={"KEEP": "y"}, runner=runner)
    assert info == AMD_INTEGRATED
    index, args, kwargs = runner.whisper_calls[0]
    assert index == "1"
    assert args[0] == str(WHISPER)
    assert args[args.index("-m") + 1] == str(WHISPER.parent / PROBE_MODEL_NAME)
    assert not Path(args[args.index("-m") + 1]).exists()
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert kwargs["env"] == {"KEEP": "y", "GGML_VK_VISIBLE_DEVICES": "1"}
    assert kwargs["timeout"] == 20
    assert kwargs["creationflags"] & 0x08000000
    assert kwargs["cwd"] == str(WHISPER.parent)


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


# select_device -------------------------------------------------------------------


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


def test_select_leaves_uma_unknown_when_the_backend_names_another_device():
    swapped = {"0": AMD_BACKEND, "1": RTX_BACKEND}
    selection = select_device(LLAMA, env={}, runner=two_gpu_runner(swapped), whisper_server=WHISPER)
    assert [d.uma for d in selection.devices] == [None, None]
    assert selection.raw_index == 1


def test_select_picks_most_memory_within_the_discrete_class():
    small = "Available devices:\n  Vulkan0: Small Card (4000 MiB, 1 MiB free)\n"
    big = "Available devices:\n  Vulkan0: Big Card (16000 MiB, 1 MiB free)\n"
    both = (
        "Available devices:\n"
        "  Vulkan0: Small Card (4000 MiB, 1 MiB free)\n"
        "  Vulkan1: Big Card (16000 MiB, 1 MiB free)\n"
    )
    backend = {
        "0": "ggml_vulkan: 0 = Small Card (NVIDIA) | uma: 0 | fp16: 1\n",
        "1": "ggml_vulkan: 0 = Big Card (NVIDIA) | uma: 0 | fp16: 1\n",
    }
    runner = make_runner({None: both, "0": small, "1": big}, backend=backend)
    selection = select_device(LLAMA, env={}, runner=runner, whisper_server=WHISPER)
    assert (selection.raw_index, selection.name) == (1, "Big Card")


def test_select_with_only_integrated_gpus_still_picks_one():
    runner = two_gpu_runner(backend={"0": AMD_BACKEND.replace(AMD, RTX), "1": AMD_BACKEND})
    selection = select_device(LLAMA, env={}, runner=runner, whisper_server=WHISPER)
    assert [d.uma for d in selection.devices] == [True, True]
    assert (selection.raw_index, selection.name) == (1, AMD)


def test_select_override_by_raw_index_wins():
    selection = select_device(LLAMA, override=0, env={}, runner=two_gpu_runner())
    assert (selection.raw_index, selection.name) == (0, RTX)


def test_select_override_by_name_wins():
    selection = select_device(LLAMA, override=RTX, env={}, runner=two_gpu_runner())
    assert (selection.raw_index, selection.name) == (0, RTX)


def test_select_unknown_override_falls_back_to_the_rule():
    selection = select_device(LLAMA, override=7, env={}, runner=two_gpu_runner())
    assert selection.raw_index == 1
    selection = select_device(LLAMA, override="Ghost GPU", env={}, runner=two_gpu_runner())
    assert selection.raw_index == 1


def test_select_keeps_cached_choice_when_the_probe_still_matches():
    selection = select_device(LLAMA, cached=(0, RTX), env={}, runner=two_gpu_runner())
    assert (selection.raw_index, selection.name) == (0, RTX)


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


def test_select_prefers_the_lower_index_on_a_memory_tie():
    same = f"Available devices:\n  Vulkan0: {RTX} (8000 MiB, 1 MiB free)\n"
    other = "Available devices:\n  Vulkan0: Twin (8000 MiB, 1 MiB free)\n"
    both = (
        "Available devices:\n"
        f"  Vulkan0: {RTX} (8000 MiB, 1 MiB free)\n"
        "  Vulkan1: Twin (8000 MiB, 1 MiB free)\n"
    )
    runner = make_runner({None: both, "0": same, "1": other})
    assert select_device(LLAMA, env={}, runner=runner).raw_index == 0


def test_selection_and_device_are_frozen():
    selection = select_device(LLAMA, env={}, runner=two_gpu_runner(REAL_BACKEND))
    with pytest.raises(AttributeError):
        selection.raw_index = 0
    with pytest.raises(AttributeError):
        selection.devices[0].name = "x"


# engine_env -----------------------------------------------------------------------


def test_engine_env_sets_the_raw_index():
    selection = GpuSelection(raw_index=1, name=AMD, memory_mb=48688, devices=())
    env = engine_env(selection, base={"PATH": "p", "GGML_VK_VISIBLE_DEVICES": "0"})
    assert env == {"PATH": "p", "GGML_VK_VISIBLE_DEVICES": "1"}


def test_engine_env_removes_the_variable_without_a_gpu():
    selection = GpuSelection(raw_index=None, name="", memory_mb=0, devices=())
    env = engine_env(selection, base={"PATH": "p", "GGML_VK_VISIBLE_DEVICES": "0"})
    assert env == {"PATH": "p"}


def test_engine_env_copies_the_base_and_defaults_to_os_environ(monkeypatch):
    base = {"A": "1"}
    selection = GpuSelection(raw_index=0, name=RTX, memory_mb=7810, devices=())
    env = engine_env(selection, base=base)
    assert env == {"A": "1", "GGML_VK_VISIBLE_DEVICES": "0"}
    assert base == {"A": "1"}
    monkeypatch.setenv("SPELLS_TEST_MARKER", "yes")
    assert engine_env(selection)["SPELLS_TEST_MARKER"] == "yes"
    assert engine_env(selection)["GGML_VK_VISIBLE_DEVICES"] == "0"
