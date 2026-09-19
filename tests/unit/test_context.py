"""Unit tests for spells.context: the press-time target snapshot.

Spec 6 step 1 and 11 step 1; batch 2 decision V1-1 (no elevation in TargetContext).
The Win32 layer is faked, so nothing here touches a real window.
"""

from __future__ import annotations

import inspect
import time

import pytest

from spells import context
from spells.models import TargetContext

HWND = 0x00110022


class FakeWindow:
    """The three spells.win32.window functions context uses, recorded and scriptable."""

    def __init__(self, hwnd=HWND, process="notepad.exe", title="Untitled - Notepad", error_on=()):
        self.hwnd = hwnd
        self.process = process
        self.title = title
        self.error_on = set(error_on)
        self.calls: list[tuple] = []

    def _maybe_fail(self, name: str) -> None:
        if name in self.error_on:
            raise OSError(5, f"{name} failed")

    def foreground_hwnd(self):
        self.calls.append(("foreground_hwnd",))
        self._maybe_fail("foreground_hwnd")
        return self.hwnd

    def window_process_name(self, hwnd):
        self.calls.append(("window_process_name", hwnd))
        self._maybe_fail("window_process_name")
        return self.process

    def window_title(self, hwnd):
        self.calls.append(("window_title", hwnd))
        self._maybe_fail("window_title")
        return self.title


class FakeClock:
    """Returns the given values in turn, then repeats the last one."""

    def __init__(self, *values: float):
        self.values = list(values) or [0.0]
        self.calls = 0

    def __call__(self) -> float:
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


def test_capture_reads_the_foreground_window():
    window = FakeWindow()
    ctx = context.capture(window=window, clock=FakeClock(12.5))
    assert ctx == TargetContext(
        hwnd=HWND, process="notepad.exe", title="Untitled - Notepad", captured_at=12.5
    )


def test_capture_asks_about_the_foreground_window_only():
    window = FakeWindow()
    context.capture(window=window, clock=FakeClock(0.0))
    assert window.calls == [
        ("foreground_hwnd",),
        ("window_process_name", HWND),
        ("window_title", HWND),
    ]


def test_capture_stamps_the_clock_exactly_once():
    clock = FakeClock(3.0, 99.0)
    ctx = context.capture(window=FakeWindow(), clock=clock)
    assert clock.calls == 1
    assert ctx.captured_at == 3.0


def test_capture_keeps_an_unknown_process_empty():
    ctx = context.capture(window=FakeWindow(process="", title=""), clock=FakeClock(1.0))
    assert ctx.process == ""
    assert ctx.title == ""
    assert ctx.hwnd == HWND


def test_capture_handles_no_foreground_window():
    ctx = context.capture(window=FakeWindow(hwnd=0, process="", title=""), clock=FakeClock(1.0))
    assert ctx.hwnd == 0


@pytest.mark.parametrize("failing", ["foreground_hwnd", "window_process_name", "window_title"])
def test_capture_returns_an_empty_context_when_win32_fails(failing):
    ctx = context.capture(window=FakeWindow(error_on=[failing]), clock=FakeClock(7.25))
    assert ctx == TargetContext(hwnd=0, process="", title="", captured_at=7.25)


def test_capture_does_not_retry_a_failing_lookup():
    window = FakeWindow(error_on=["window_title"])
    context.capture(window=window, clock=FakeClock(0.0))
    assert [call[0] for call in window.calls] == [
        "foreground_hwnd",
        "window_process_name",
        "window_title",
    ]


def test_capture_survives_a_backend_that_is_not_a_window_module():
    ctx = context.capture(window=object(), clock=FakeClock(2.0))
    assert ctx == TargetContext(hwnd=0, process="", title="", captured_at=2.0)


def test_capture_defaults_to_the_win32_window_module():
    from spells.win32 import window as win32_window

    default = inspect.signature(context.capture).parameters["window"].default
    assert default is win32_window


def test_capture_defaults_to_a_high_resolution_clock():
    # B3-23: time.monotonic is GetTickCount64 on Windows and steps in 15.6 ms.
    default = inspect.signature(context.capture).parameters["clock"].default
    assert default is time.perf_counter
    assert time.get_clock_info("perf_counter").resolution < 0.001
