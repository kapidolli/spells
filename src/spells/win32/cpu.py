from __future__ import annotations

import ctypes
import logging
import struct
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import dataclass

from spells.models import CpuPlan

logger = logging.getLogger(__name__)

RELATION_PROCESSOR_CORE = 0
LTP_PC_SMT = 0x1
ERROR_INSUFFICIENT_BUFFER = 122
MAX_ENGINE_THREADS = 8
RECORD_HEADER = struct.Struct("<II")
PROCESSOR_OFFSET = RECORD_HEADER.size
GROUP_COUNT_OFFSET = PROCESSOR_OFFSET + 22
GROUP_MASKS_OFFSET = PROCESSOR_OFFSET + 24

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetLogicalProcessorInformationEx.argtypes = (
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(wintypes.DWORD),
)
_kernel32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL


@dataclass(frozen=True)
class ProcessorCore:
    efficiency_class: int
    smt: bool
    group: int
    mask: int


def group_affinity_size(pointer_size: int) -> int:
    return pointer_size + 8


def parse_cores(buffer: bytes, pointer_size: int = 8) -> tuple[ProcessorCore, ...]:
    mask_format = "<Q" if pointer_size == 8 else "<I"
    affinity_size = group_affinity_size(pointer_size)
    cores: list[ProcessorCore] = []
    offset = 0
    while offset + RECORD_HEADER.size <= len(buffer):
        relationship, size = RECORD_HEADER.unpack_from(buffer, offset)
        if size <= 0 or offset + size > len(buffer):
            break
        if relationship == RELATION_PROCESSOR_CORE and size >= GROUP_MASKS_OFFSET:
            flags = buffer[offset + PROCESSOR_OFFSET]
            efficiency = buffer[offset + PROCESSOR_OFFSET + 1]
            (group_count,) = struct.unpack_from("<H", buffer, offset + GROUP_COUNT_OFFSET)
            for index in range(group_count):
                start = offset + GROUP_MASKS_OFFSET + index * affinity_size
                if start + affinity_size > offset + size:
                    break
                (mask,) = struct.unpack_from(mask_format, buffer, start)
                (group,) = struct.unpack_from("<H", buffer, start + pointer_size)
                cores.append(
                    ProcessorCore(
                        efficiency_class=efficiency,
                        smt=bool(flags & LTP_PC_SMT),
                        group=group,
                        mask=mask,
                    )
                )
        offset += size
    return tuple(cores)


def is_hybrid(cores: Sequence[ProcessorCore]) -> bool:
    return len({core.efficiency_class for core in cores}) > 1


def performance_cores(cores: Sequence[ProcessorCore]) -> tuple[ProcessorCore, ...]:
    if not cores:
        return ()
    best = max(core.efficiency_class for core in cores)
    fastest = [core for core in cores if core.efficiency_class == best and core.mask]
    if not fastest:
        return ()
    group = fastest[0].group
    return tuple(core for core in fastest if core.group == group)


def plan_for(cores: Sequence[ProcessorCore]) -> CpuPlan:
    if not is_hybrid(cores):
        return CpuPlan()
    chosen = performance_cores(cores)
    if not chosen:
        return CpuPlan()
    mask = 0
    for core in chosen:
        mask |= core.mask & -core.mask
    return CpuPlan(threads=min(len(chosen), MAX_ENGINE_THREADS), affinity_mask=mask)


def read_topology() -> bytes:
    length = wintypes.DWORD(0)
    _kernel32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, None, ctypes.byref(length))
    error = ctypes.get_last_error()
    if error != ERROR_INSUFFICIENT_BUFFER or length.value == 0:
        raise ctypes.WinError(error)
    buffer = ctypes.create_string_buffer(length.value)
    if not _kernel32.GetLogicalProcessorInformationEx(
        RELATION_PROCESSOR_CORE, buffer, ctypes.byref(length)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.raw[: length.value]


def detect_cpu_plan() -> CpuPlan:
    try:
        cores = parse_cores(read_topology(), ctypes.sizeof(ctypes.c_void_p))
    except OSError as exc:
        logger.warning("could not read the processor topology: %s", exc)
        return CpuPlan()
    plan = plan_for(cores)
    if plan.affinity_mask is not None:
        logger.info(
            "hybrid processor: engines pinned to %d performance cores (mask %s)",
            plan.threads,
            plan.cpu_mask_hex,
        )
    return plan
