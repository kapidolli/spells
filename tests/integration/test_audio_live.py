"""Live microphone checks: need a real input device (run with -m integration)."""

from __future__ import annotations

import time

import pytest

from spells.audio import Recorder, list_input_devices

pytestmark = pytest.mark.integration


def test_list_input_devices_has_a_default():
    devices = list_input_devices()
    assert devices
    assert sum(1 for d in devices if d.is_default) == 1
    for d in devices:
        print(f"[{d.index}] {d.name}{' (default)' if d.is_default else ''}")


def test_record_one_second_from_default_device():
    levels: list[float] = []
    first_level_at: list[float] = []

    def on_level(level: float) -> None:
        if not first_level_at:
            first_level_at.append(time.perf_counter())
        levels.append(level)

    rec = Recorder(on_level=on_level)
    t0 = time.perf_counter()
    rec.start()
    time.sleep(1.0)
    pcm = rec.stop()
    stopped_at = time.perf_counter()

    expected = 16000 * 2
    assert expected * 0.8 <= len(pcm) <= expected * 1.2
    assert rec.open_latency_ms is not None
    assert rec.open_latency_ms > 0
    assert levels
    assert all(0.0 <= v <= 1.0 for v in levels)
    first_data_ms = (first_level_at[0] - t0) * 1000 if first_level_at else float("nan")
    print(
        f"open_latency_ms={rec.open_latency_ms:.1f} first_data_ms={first_data_ms:.1f} "
        f"bytes={len(pcm)} level_updates={len(levels)} max_level={max(levels):.3f} "
        f"total_ms={(stopped_at - t0) * 1000:.1f}"
    )
