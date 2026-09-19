from __future__ import annotations

import struct

import pytest

from spells.models import CpuPlan
from spells.win32 import cpu
from spells.win32.cpu import ProcessorCore, parse_cores, plan_for

RELATION_CACHE = 2


def core_record(efficiency: int, mask: int, *, smt: bool = False, group: int = 0,
                pointer_size: int = 8) -> bytes:
    mask_format = "<Q" if pointer_size == 8 else "<I"
    processor = bytes([cpu.LTP_PC_SMT if smt else 0, efficiency]) + bytes(20)
    processor += struct.pack("<H", 1)
    processor += struct.pack(mask_format, mask) + struct.pack("<H", group) + bytes(6)
    size = 8 + len(processor)
    return struct.pack("<II", cpu.RELATION_PROCESSOR_CORE, size) + processor


def other_record(size: int = 48) -> bytes:
    return struct.pack("<II", RELATION_CACHE, size) + bytes(size - 8)


def i7_14700k() -> bytes:
    records = [core_record(1, 0b11 << (2 * index), smt=True) for index in range(8)]
    records += [core_record(0, 1 << (16 + index)) for index in range(12)]
    return b"".join(records)


def test_a_performance_core_record_parses():
    (core,) = parse_cores(core_record(1, 0b1100, smt=True, group=0))
    assert core == ProcessorCore(efficiency_class=1, smt=True, group=0, mask=0b1100)


def test_the_i7_14700k_topology_pins_one_logical_processor_per_performance_core():
    cores = parse_cores(i7_14700k())
    assert len(cores) == 20
    assert cpu.is_hybrid(cores)
    assert len(cpu.performance_cores(cores)) == 8
    plan = plan_for(cores)
    assert plan == CpuPlan(threads=8, affinity_mask=0x5555)
    assert plan.cpu_mask_hex == "5555"


def test_records_of_other_relationships_are_skipped():
    buffer = other_record() + core_record(1, 0b11, smt=True) + other_record(32)
    assert [core.mask for core in parse_cores(buffer)] == [0b11]


def test_a_truncated_buffer_stops_the_parse_without_raising():
    buffer = i7_14700k()
    assert len(parse_cores(buffer[:-5])) == 19
    assert parse_cores(b"") == ()
    assert parse_cores(struct.pack("<II", 0, 0) + bytes(40)) == ()


def test_a_32_bit_layout_parses():
    buffer = core_record(1, 0b11, smt=True, pointer_size=4) + core_record(0, 0b100, pointer_size=4)
    cores = parse_cores(buffer, pointer_size=4)
    assert [(core.efficiency_class, core.mask) for core in cores] == [(1, 0b11), (0, 0b100)]


def test_a_uniform_processor_is_not_pinned():
    cores = parse_cores(b"".join(core_record(0, 0b11 << (2 * i), smt=True) for i in range(8)))
    assert not cpu.is_hybrid(cores)
    assert plan_for(cores) == CpuPlan()
    assert CpuPlan().cpu_mask_hex is None


def test_threads_are_capped_at_eight():
    records = [core_record(2, 1 << index) for index in range(12)]
    records += [core_record(0, 1 << (12 + index)) for index in range(4)]
    plan = plan_for(parse_cores(b"".join(records)))
    assert plan.threads == 8
    assert plan.affinity_mask == 0xFFF


def test_only_the_group_of_the_first_performance_core_is_used():
    records = [core_record(1, 0b1, group=0), core_record(1, 0b1, group=1), core_record(0, 0b10)]
    plan = plan_for(parse_cores(b"".join(records)))
    assert plan == CpuPlan(threads=1, affinity_mask=0b1)


def test_detection_failure_means_no_pinning(monkeypatch, caplog):
    def broken() -> bytes:
        raise OSError(5, "denied")

    monkeypatch.setattr(cpu, "read_topology", broken)
    assert cpu.detect_cpu_plan() == CpuPlan()
    assert "processor topology" in caplog.text


def test_detection_uses_the_parsed_topology(monkeypatch):
    monkeypatch.setattr(cpu, "read_topology", i7_14700k)
    assert cpu.detect_cpu_plan() == CpuPlan(threads=8, affinity_mask=0x5555)


@pytest.mark.parametrize("pointer_size", [4, 8])
def test_the_group_affinity_size_follows_the_pointer_size(pointer_size):
    assert cpu.group_affinity_size(pointer_size) == pointer_size + 8
