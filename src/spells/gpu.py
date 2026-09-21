"""Vulkan GPU probe and device selection, shared by the app and the benchmark (spec 13, 18).

Decisions V3-F2, V4-6 and B3-59: `llama-server --list-devices` shows the filtered device
list while GGML_VK_VISIBLE_DEVICES takes raw physical indices, so the probe runs the listing
once unfiltered to learn how many devices exist and then once per raw index to map each
index to the device it exposes. Integrated GPUs report shared system RAM as their memory,
so the selection prefers a discrete GPU (ggml's `uma: 0`) over an integrated one and picks
the most memory only within the same class. An override or a still-valid cached choice
wins over the rule. Stdlib only, no Win32.

The uma flag is not part of `--list-devices` output (verified on llama.cpp b10997, stdout and
stderr, at every verbosity), and llama-server prints it at no log level either. The Vulkan
whisper-server does print the backend's device line at startup, before it even opens the
model, so probe_backend_info() runs it with a model path that does not exist: about 0.3 s
and exit code 3 per raw index, nothing loaded onto any GPU.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)

LIST_TIMEOUT_S = 20
VISIBLE_DEVICES_ENV = "GGML_VK_VISIBLE_DEVICES"
CREATE_NO_WINDOW = 0x08000000
# Extra raw indices tried beyond the unfiltered count; the first empty run ends the probe.
PROBE_SLACK = 2
# Model path handed to whisper-server for the backend probe; it must never exist.
PROBE_MODEL_NAME = "spells-gpu-probe-no-such-model.bin"

# "  Vulkan0: NVIDIA GeForce RTX 5060 Laptop GPU (7810 MiB, 7042 MiB free)" (common/arg.cpp)
_DEVICE_LINE = re.compile(
    r"^\s*(?P<list_name>Vulkan\d+):\s+(?P<name>.+?)\s+"
    r"\((?P<total>\d+)\s+MiB,\s+(?P<free>\d+)\s+MiB free\)\s*$",
    re.IGNORECASE,
)
# "ggml_vulkan: 0 = AMD Radeon(TM) 610M (AMD proprietary driver) | uma: 1 | fp16: 1 | ..."
# The name may contain parentheses; the vendor is the last group before " | uma:".
_BACKEND_LINE = re.compile(
    r"^ggml_vulkan: \d+ = (?P<name>.*) \((?P<vendor>[^()]*)\) \| uma: (?P<uma>[01])\b"
)


@dataclass(frozen=True)
class GpuDevice:
    """One device as `--list-devices` reports it; raw_index is None until the probe maps it.

    uma is True for an integrated GPU (unified memory, its memory_mb is shared system RAM),
    False for a discrete one, None when the backend probe could not tell.
    """

    raw_index: int | None
    list_name: str
    name: str
    memory_mb: int
    free_mb: int
    uma: bool | None = None
    vendor: str = ""


@dataclass(frozen=True)
class BackendDevice:
    """One device as the ggml Vulkan backend describes it at startup."""

    name: str
    vendor: str
    uma: bool


@dataclass(frozen=True)
class GpuSelection:
    """The chosen device (raw_index None means no Vulkan GPU) plus every device the probe saw."""

    raw_index: int | None
    name: str
    memory_mb: int
    devices: tuple[GpuDevice, ...]


NO_GPU = GpuSelection(raw_index=None, name="", memory_mb=0, devices=())


def parse_list_devices(output: str) -> list[GpuDevice]:
    """Devices from `llama-server --list-devices` output; headers, "(none)" and junk are ignored."""
    devices = []
    for line in output.splitlines():
        match = _DEVICE_LINE.match(line)
        if match is None:
            continue
        devices.append(
            GpuDevice(
                raw_index=None,
                list_name=match.group("list_name"),
                name=match.group("name"),
                memory_mb=int(match.group("total")),
                free_mb=int(match.group("free")),
            )
        )
    return devices


def parse_backend_info(output: str) -> list[BackendDevice]:
    """Devices from the ggml Vulkan backend's startup lines, in order, without repeats.

    The `ggml_vulkan: <i> = <name> (<vendor>) | uma: <0|1> | ...` block is printed once per
    ggml backend library that loads, so the same device appears more than once.
    """
    devices: list[BackendDevice] = []
    for line in output.splitlines():
        match = _BACKEND_LINE.match(line.strip())
        if match is None:
            continue
        device = BackendDevice(
            name=match.group("name").strip(),
            vendor=match.group("vendor").strip(),
            uma=match.group("uma") == "1",
        )
        if device not in devices:
            devices.append(device)
    return devices


def _run_hidden(args: list[str], env: dict[str, str] | None, cwd: Path, runner) -> str | None:
    """Run a probe hidden with the 20 s timeout; stdout plus stderr, None when it could not run."""
    try:
        result = runner(
            args,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=LIST_TIMEOUT_S,
            env=env,
            cwd=str(cwd),
            creationflags=CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        logger.warning("%s did not finish within %s s", " ".join(args[:2]), LIST_TIMEOUT_S)
        return None
    except OSError as exc:
        logger.warning("%s could not run: %s", " ".join(args[:2]), exc)
        return None
    return (result.stdout or "") + "\n" + (result.stderr or "")


def list_devices(
    llama_server: Path,
    env: dict[str, str] | None = None,
    runner=subprocess.run,
) -> list[GpuDevice]:
    """Run `llama-server --list-devices` hidden with a 20 s timeout and parse stdout plus stderr.

    A binary that cannot start or does not answer in time counts as "no devices": the caller
    then launches the CPU builds, which is the safe outcome.
    """
    llama_server = Path(llama_server)
    output = _run_hidden([str(llama_server), "--list-devices"], env, llama_server.parent, runner)
    return parse_list_devices(output) if output is not None else []


def probe_backend_info(
    whisper_server: Path,
    raw_index: int,
    env: dict[str, str] | None = None,
    runner=subprocess.run,
) -> BackendDevice | None:
    """The ggml Vulkan backend's description of raw device raw_index, or None when unknown.

    Runs the Vulkan whisper-server with GGML_VK_VISIBLE_DEVICES set and a model path that does
    not exist: the backend prints its device line before the model open fails. None when the
    run fails or does not describe exactly one device.
    """
    whisper_server = Path(whisper_server)
    probe_env = dict(os.environ if env is None else env)
    probe_env[VISIBLE_DEVICES_ENV] = str(raw_index)
    args = [
        str(whisper_server),
        "-m",
        str(whisper_server.parent / PROBE_MODEL_NAME),
        "--host",
        "127.0.0.1",
        "--port",
        "1",
    ]
    output = _run_hidden(args, probe_env, whisper_server.parent, runner)
    if output is None:
        return None
    devices = parse_backend_info(output)
    if len(devices) != 1:
        logger.warning(
            "backend probe for raw index %d described %d devices instead of one",
            raw_index,
            len(devices),
        )
        return None
    return devices[0]


def probe_raw_indices(
    llama_server: Path,
    env: dict[str, str] | None = None,
    runner=subprocess.run,
) -> list[GpuDevice]:
    """Map raw GGML_VK_VISIBLE_DEVICES indices to devices (decision V3-F2).

    The unfiltered listing gives the count N. Then each raw index 0 .. N+1 is listed alone; a
    run that reports exactly one device maps that index to it, and the first run that reports
    nothing ends the probe (an out-of-range index reports "(none)").
    """
    base = dict(os.environ if env is None else env)
    base.pop(VISIBLE_DEVICES_ENV, None)
    unfiltered = list_devices(llama_server, base, runner)
    if not unfiltered:
        return []
    mapped = []
    for raw_index in range(len(unfiltered) + PROBE_SLACK):
        filtered_env = dict(base)
        filtered_env[VISIBLE_DEVICES_ENV] = str(raw_index)
        seen = list_devices(llama_server, filtered_env, runner)
        if not seen:
            break
        if len(seen) != 1:
            logger.warning(
                "%s=%d listed %d devices instead of one; index skipped",
                VISIBLE_DEVICES_ENV,
                raw_index,
                len(seen),
            )
            continue
        mapped.append(replace(seen[0], raw_index=raw_index))
    return mapped


def _with_backend_info(device: GpuDevice, info: BackendDevice | None) -> GpuDevice:
    if info is None:
        return device
    if info.name != device.name:
        logger.warning(
            "raw index %s is %r for llama-server but %r for the Vulkan backend; uma unknown",
            device.raw_index,
            device.name,
            info.name,
        )
        return device
    return replace(device, uma=info.uma, vendor=info.vendor)


def _class_rank(device: GpuDevice) -> int:
    """Discrete first, then unknown, then integrated (decision B3-59)."""
    if device.uma is False:
        return 0
    if device.uma is None:
        return 1
    return 2


def select_device(
    llama_server: Path,
    override: int | str | None = None,
    cached: tuple[int, str] | None = None,
    env: dict[str, str] | None = None,
    runner=subprocess.run,
    whisper_server: Path | None = None,
) -> GpuSelection:
    """Pick the device for both engines.

    An override (a raw index or a device name from the Diagnostics dropdown) wins when the probe
    still lists it. Otherwise a cached (raw_index, name) pair from settings is kept while the
    device at that index still has that name and no device is in a better class, so a discrete
    GPU fitted after the first start takes over from a cached integrated one. Otherwise the
    rule of decision B3-59: a discrete
    GPU beats an integrated one (the uma flag learned through whisper_server; without it every
    device counts as unknown), and within a class the most reported memory wins, the lower raw
    index on a tie. No device at all gives raw_index None.
    """
    devices = probe_raw_indices(llama_server, env, runner)
    if not devices:
        return NO_GPU
    if whisper_server is not None:
        devices = [
            _with_backend_info(
                device, probe_backend_info(whisper_server, device.raw_index or 0, env, runner)
            )
            for device in devices
        ]
    chosen = None
    if override is not None:
        chosen = _find(devices, override)
        if chosen is None:
            logger.warning("GPU override %r is not among the probed devices; ignored", override)
    if chosen is None and cached is not None:
        candidate = _find(devices, cached[0])
        best_class = min(_class_rank(device) for device in devices)
        if (
            candidate is not None
            and candidate.name == cached[1]
            and _class_rank(candidate) == best_class
        ):
            chosen = candidate
    if chosen is None:
        chosen = min(devices, key=lambda d: (_class_rank(d), -d.memory_mb, d.raw_index or 0))
    return GpuSelection(
        raw_index=chosen.raw_index,
        name=chosen.name,
        memory_mb=chosen.memory_mb,
        devices=tuple(devices),
    )


def choose_device(devices: Sequence[GpuDevice], override: int | None = None) -> GpuSelection:
    """Choose from the startup probe without subprocesses or a cached manual preference."""
    if not devices:
        return NO_GPU
    chosen = _find(devices, override) if override is not None else None
    if chosen is None:
        chosen = min(devices, key=lambda d: (_class_rank(d), -d.memory_mb, d.raw_index or 0))
    return GpuSelection(chosen.raw_index, chosen.name, chosen.memory_mb, tuple(devices))


def _find(devices: Sequence[GpuDevice], key: int | str) -> GpuDevice | None:
    for device in devices:
        if isinstance(key, str):
            if device.name == key:
                return device
        elif device.raw_index == key:
            return device
    return None


def engine_env(selection: GpuSelection, base: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of base (os.environ by default) with GGML_VK_VISIBLE_DEVICES set or removed."""
    env = dict(os.environ if base is None else base)
    if selection.raw_index is None:
        env.pop(VISIBLE_DEVICES_ENV, None)
    else:
        env[VISIBLE_DEVICES_ENV] = str(selection.raw_index)
    return env
