"""Unit tests for spells.context: the press-time target snapshot.

Spec 6 step 1 and 11 step 1; batch 2 decision V1-1 (no elevation in TargetContext).
The platform's focus is faked, so nothing here touches a real window.
"""

from __future__ import annotations

import inspect
import time

import pytest

from spells import context
from spells.models import TargetContext

from .fake_platform import fake_platform

HWND = 0x00110022


class FakeFocus:
    """The three Focus methods context uses, recorded and scriptable."""

    def __init__(self, window=HWND, process="notepad.exe", title="Untitled - Notepad", error_on=()):
        self.window = window
        self.process = process
        self.title_text = title
        self.error_on = set(error_on)
        self.calls: list[tuple] = []

    def _maybe_fail(self, name: str) -> None:
        if name in self.error_on:
            raise OSError(5, f"{name} failed")

    def foreground(self):
        self.calls.append(("foreground",))
        self._maybe_fail("foreground")
        return self.window

    def app_name(self, window):
        self.calls.append(("app_name", window))
        self._maybe_fail("app_name")
        return self.process

    def title(self, window):
        self.calls.append(("title", window))
        self._maybe_fail("title")
        return self.title_text


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
    focus = FakeFocus()
    ctx = context.capture(focus=focus, clock=FakeClock(12.5))
    assert ctx == TargetContext(
        window=HWND, process="notepad.exe", title="Untitled - Notepad", captured_at=12.5
    )


def test_capture_asks_about_the_foreground_window_only():
    focus = FakeFocus()
    context.capture(focus=focus, clock=FakeClock(0.0))
    assert focus.calls == [
        ("foreground",),
        ("app_name", HWND),
        ("title", HWND),
    ]


def test_capture_stamps_the_clock_exactly_once():
    clock = FakeClock(3.0, 99.0)
    ctx = context.capture(focus=FakeFocus(), clock=clock)
    assert clock.calls == 1
    assert ctx.captured_at == 3.0


def test_capture_keeps_an_unknown_process_empty():
    ctx = context.capture(focus=FakeFocus(process="", title=""), clock=FakeClock(1.0))
    assert ctx.process == ""
    assert ctx.title == ""
    assert ctx.window == HWND


def test_capture_handles_no_foreground_window():
    ctx = context.capture(focus=FakeFocus(window=0, process="", title=""), clock=FakeClock(1.0))
    assert ctx.window == 0


@pytest.mark.parametrize("failing", ["foreground", "app_name", "title"])
def test_capture_returns_an_empty_context_when_win32_fails(failing):
    ctx = context.capture(focus=FakeFocus(error_on=[failing]), clock=FakeClock(7.25))
    assert ctx == TargetContext(window=0, process="", title="", captured_at=7.25)


def test_capture_does_not_retry_a_failing_lookup():
    focus = FakeFocus(error_on=["title"])
    context.capture(focus=focus, clock=FakeClock(0.0))
    assert [call[0] for call in focus.calls] == [
        "foreground",
        "app_name",
        "title",
    ]


def test_capture_survives_a_backend_that_is_not_a_window_module():
    ctx = context.capture(focus=object(), clock=FakeClock(2.0))
    assert ctx == TargetContext(window=0, process="", title="", captured_at=2.0)


def test_capture_defaults_to_the_platform_focus(use_platform):
    focus = FakeFocus()
    use_platform(fake_platform(focus=focus))
    assert inspect.signature(context.capture).parameters["focus"].default is None
    context.capture(clock=FakeClock(0.0))
    assert focus.calls[0] == ("foreground",)


def test_capture_defaults_to_a_high_resolution_clock():
    # B3-23: time.monotonic is GetTickCount64 on Windows and steps in 15.6 ms.
    default = inspect.signature(context.capture).parameters["clock"].default
    assert default is time.perf_counter
    assert time.get_clock_info("perf_counter").resolution < 0.001
