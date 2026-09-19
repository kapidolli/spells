"""Unit tests for spells.inject with fake Win32 backends.

Spec 11 (all steps), 12 (delivery budget), 16 (focus moved or target elevated); batch 2
decisions V1-1 and V2-3. No real clipboard, window or keystroke is touched here.
"""

from __future__ import annotations

import inspect
import time

import pytest

from spells.inject import (
    DeliveryReport,
    InjectBackends,
    deliver,
    ends_with_whitespace,
)
from spells.models import DeliveryMethod, DeliveryOutcome, DeliveryResult, TargetContext

TARGET_HWND = 0x00AA00BB
OTHER_HWND = 0x00CC00DD
SEQUENCE = 77
TEXT = "Grüße aus Spells \U0001f600"
SNAPSHOT = object()


class FakeWindow:
    def __init__(self, log, foreground=TARGET_HWND, elevated=False, error_on=()):
        self.log = log
        self.foreground = foreground
        self.elevated = elevated
        self.error_on = set(error_on)

    def _maybe_fail(self, name):
        if name in self.error_on:
            raise OSError(5, f"{name} failed")

    def foreground_hwnd(self):
        self.log.append(("foreground_hwnd",))
        self._maybe_fail("foreground_hwnd")
        return self.foreground

    def is_elevated_window(self, hwnd):
        self.log.append(("is_elevated_window", hwnd))
        self._maybe_fail("is_elevated_window")
        return self.elevated


class FakeInput:
    """The win32.input calls delivery and live insertion make.

    `typed` and `backspaced` record the live-insertion keyword arguments as well, which the
    livetext tests read; the shared log keeps the shape the delivery tests assert on.
    """

    def __init__(self, log, error_on=(), clipboard=None):
        self.log = log
        self.error_on = set(error_on)
        self.typed: list[tuple[str, dict]] = []
        self.backspaced: list[tuple[int, dict]] = []
        self.clipboard = clipboard

    def _maybe_fail(self, name):
        if name in self.error_on:
            raise RuntimeError(f"{name} failed")

    def release_held_modifiers(self):
        self.log.append(("release_held_modifiers",))
        self._maybe_fail("release_held_modifiers")
        return []

    def send_ctrl_v(self):
        self.log.append(("send_ctrl_v",))
        self._maybe_fail("send_ctrl_v")

    def send_ctrl_c(self):
        """Edit mode copies the selection with this (spec 8.5); the fake app answers it."""
        self.log.append(("send_ctrl_c",))
        self._maybe_fail("send_ctrl_c")
        if self.clipboard is not None:
            self.clipboard.copy()

    def type_unicode(self, text, batch_size=None, **kwargs):
        self.log.append(("type_unicode", text))
        self.typed.append((text, dict(kwargs)))
        self._maybe_fail("type_unicode")

    def send_backspaces(self, count, batch_size=None, **kwargs):
        self.log.append(("send_backspaces", count))
        self.backspaced.append((int(count), dict(kwargs)))
        self._maybe_fail("send_backspaces")


class FakeClipboard:
    """The clipboard plus the target app behind it.

    `selection` is what the window would put on the clipboard for a Ctrl+C; None means
    nothing is selected, so the sequence number never moves, which is exactly what edit
    mode reads as "nothing selected" (spec 8.5).
    """

    def __init__(self, log, sequence=SEQUENCE, restore_result=True, error_on=(), selection=None):
        self.log = log
        self.sequence = sequence
        self.restore_result = restore_result
        self.error_on = set(error_on)
        self.selection = selection
        self.text = ""
        self.copies = 0

    def _maybe_fail(self, name):
        if name in self.error_on:
            raise OSError(1418, f"{name} failed")

    def snapshot(self):
        self.log.append(("snapshot",))
        self._maybe_fail("snapshot")
        return SNAPSHOT

    def set_text(self, text):
        self.log.append(("set_text", text))
        self._maybe_fail("set_text")
        return self.sequence

    def restore(self, snap, expected_sequence):
        self.log.append(("restore", snap, expected_sequence))
        self._maybe_fail("restore")
        return self.restore_result

    def sequence_number(self):
        self.log.append(("sequence_number",))
        self._maybe_fail("sequence_number")
        return self.sequence

    def get_text(self):
        self.log.append(("get_text",))
        self._maybe_fail("get_text")
        return self.text

    def copy(self):
        """What the focused app does on Ctrl+C when it has a selection."""
        if self.selection is None:
            return
        self.copies += 1
        self.text = self.selection
        self.sequence += 1


class FakeClock:
    """Returns the given values in turn, then repeats the last one."""

    def __init__(self, *values: float):
        self.values = list(values) or [0.0]
        self.calls = 0

    def __call__(self) -> float:
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


def make(window_error=(), input_error=(), clipboard_error=(), selection=None, **window_kwargs):
    """A shared call log plus backends whose calls all land in it, in order."""
    log: list[tuple] = []
    clipboard = FakeClipboard(log, error_on=clipboard_error, selection=selection)
    backends = InjectBackends(
        window=FakeWindow(log, error_on=window_error, **window_kwargs),
        input=FakeInput(log, error_on=input_error, clipboard=clipboard),
        clipboard=clipboard,
    )
    return log, backends


def make_sleeper(log):
    def sleeper(seconds):
        log.append(("sleep", seconds))

    return sleeper


def ctx(hwnd=TARGET_HWND):
    return TargetContext(hwnd=hwnd, process="notepad.exe", title="Untitled", captured_at=1.0)


def run(
    method=DeliveryMethod.PASTE,
    clock=None,
    backends=None,
    log=None,
    target_hwnd=TARGET_HWND,
    **kwargs,
):
    return deliver(
        TEXT,
        ctx(target_hwnd),
        method,
        backends=backends,
        sleeper=make_sleeper(log),
        clock=clock or FakeClock(0.0),
        **kwargs,
    )


# target checks (spec 11 step 1)


def test_focus_changed_copies_the_text_once_and_never_pastes():
    log, backends = make(foreground=OTHER_HWND)
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.COPIED_FOCUS_CHANGED
    assert log == [
        ("foreground_hwnd",),
        ("is_elevated_window", OTHER_HWND),
        ("set_text", TEXT),
    ]


def test_focus_changed_detail_names_both_hwnds():
    log, backends = make(foreground=OTHER_HWND)
    report = run(backends=backends, log=log)
    assert str(TARGET_HWND) in report.result.detail
    assert str(OTHER_HWND) in report.result.detail


def test_focus_changed_report_has_no_timing_and_no_restore():
    log, backends = make(foreground=OTHER_HWND)
    report = run(backends=backends, log=log)
    assert report.elapsed_ms is None
    assert report.clipboard_restored is None


def test_elevated_new_foreground_copies_with_copied_elevated():
    log, backends = make(foreground=OTHER_HWND, elevated=True)
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.COPIED_ELEVATED
    assert log == [
        ("foreground_hwnd",),
        ("is_elevated_window", OTHER_HWND),
        ("set_text", TEXT),
    ]


def test_elevated_detail_says_elevated():
    log, backends = make(foreground=OTHER_HWND, elevated=True)
    report = run(backends=backends, log=log)
    assert "elevated" in report.result.detail


def test_elevation_is_checked_even_when_the_target_kept_focus():
    log, backends = make(elevated=True)
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.COPIED_ELEVATED
    assert log == [
        ("foreground_hwnd",),
        ("is_elevated_window", TARGET_HWND),
        ("set_text", TEXT),
    ]


def test_elevation_wins_over_a_focus_change():
    log, backends = make(foreground=OTHER_HWND, elevated=True)
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.COPIED_ELEVATED


def test_a_raising_elevation_check_still_pastes():
    log, backends = make(window_error=["is_elevated_window"])
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.PASTED
    assert ("send_ctrl_v",) in log


def test_a_raising_elevation_check_does_not_hide_a_focus_change():
    log, backends = make(foreground=OTHER_HWND, window_error=["is_elevated_window"])
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.COPIED_FOCUS_CHANGED


@pytest.mark.parametrize(
    ("foreground", "target_hwnd"),
    [(0, 0), (0, TARGET_HWND), (TARGET_HWND, 0)],
)
def test_a_missing_hwnd_copies_instead_of_pasting(foreground, target_hwnd):
    log, backends = make(foreground=foreground, elevated=True)
    report = run(backends=backends, log=log, target_hwnd=target_hwnd)
    assert report.result.outcome is DeliveryOutcome.COPIED_FOCUS_CHANGED
    assert str(foreground) in report.result.detail
    assert str(target_hwnd) in report.result.detail
    # hwnd 0 owns no process, so asking whether it is elevated would answer yes.
    assert log == [("foreground_hwnd",), ("set_text", TEXT)]


# paste (spec 11 step 2)


def test_paste_call_order_and_restore_arguments():
    log, backends = make()
    report = run(backends=backends, log=log, clock=FakeClock(1.0, 1.02))
    assert log == [
        ("foreground_hwnd",),
        ("is_elevated_window", TARGET_HWND),
        ("snapshot",),
        ("set_text", TEXT),
        ("release_held_modifiers",),
        ("send_ctrl_v",),
        ("sleep", 0.3),
        ("restore", SNAPSHOT, SEQUENCE),
    ]
    assert report.result.outcome is DeliveryOutcome.PASTED
    assert report.clipboard_restored is True
    assert report.elapsed_ms == pytest.approx(20.0)


def test_paste_uses_the_given_restore_delay():
    log, backends = make()
    run(backends=backends, log=log, restore_delay_s=1.25)
    assert ("sleep", 1.25) in log


def test_paste_restore_refused_is_still_pasted_with_a_note():
    log = []
    backends = InjectBackends(
        window=FakeWindow(log),
        input=FakeInput(log),
        clipboard=FakeClipboard(log, restore_result=False),
    )
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.PASTED
    assert report.clipboard_restored is False
    assert "not restored" in report.result.detail


def test_paste_restore_raising_is_still_pasted():
    log, backends = make(clipboard_error=["restore"])
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.PASTED
    assert report.clipboard_restored is False
    assert "OSError" in report.result.detail


def test_paste_records_elapsed_before_the_restore_delay():
    log, backends = make()
    report = run(backends=backends, log=log, clock=FakeClock(0.0, 0.05, 9.0))
    assert report.elapsed_ms == pytest.approx(50.0)


def test_snapshot_failure_fails_without_touching_the_clipboard_or_keyboard():
    log, backends = make(clipboard_error=["snapshot"])
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.FAILED
    assert "OSError" in report.result.detail
    assert log == [("foreground_hwnd",), ("is_elevated_window", TARGET_HWND), ("snapshot",)]


def test_set_text_failure_fails_and_sends_no_ctrl_v():
    log, backends = make(clipboard_error=["set_text"])
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.FAILED
    assert [entry[0] for entry in log] == [
        "foreground_hwnd",
        "is_elevated_window",
        "snapshot",
        "set_text",
    ]
    assert report.elapsed_ms is None


# type (spec 11 step 3, decision V2-3)


def test_type_releases_modifiers_before_typing():
    log, backends = make()
    report = run(method=DeliveryMethod.TYPE, backends=backends, log=log, clock=FakeClock(2.0, 2.01))
    assert log == [
        ("foreground_hwnd",),
        ("is_elevated_window", TARGET_HWND),
        ("release_held_modifiers",),
        ("type_unicode", TEXT),
    ]
    assert report.result.outcome is DeliveryOutcome.TYPED
    assert report.elapsed_ms == pytest.approx(10.0)
    assert report.clipboard_restored is None


# failures (spec 11 step 4)


def test_send_ctrl_v_raising_gives_failed_and_never_escapes():
    log, backends = make(input_error=["send_ctrl_v"])
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.FAILED
    assert "RuntimeError" in report.result.detail


def test_a_failure_after_set_text_says_the_text_is_on_the_clipboard():
    for failing in ("release_held_modifiers", "send_ctrl_v"):
        log, backends = make(input_error=[failing])
        report = run(backends=backends, log=log)
        assert report.result.outcome is DeliveryOutcome.FAILED
        assert report.result.detail.endswith("; the text is on the clipboard")
        assert ("restore", SNAPSHOT, SEQUENCE) not in log


def test_type_unicode_raising_gives_failed():
    log, backends = make(input_error=["type_unicode"])
    report = run(method=DeliveryMethod.TYPE, backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.FAILED
    assert "RuntimeError" in report.result.detail


def test_a_failing_window_backend_gives_failed():
    log, backends = make(window_error=["foreground_hwnd"])
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.FAILED
    assert "OSError" in report.result.detail


def test_a_failing_copy_fallback_gives_failed():
    log, backends = make(foreground=OTHER_HWND, clipboard_error=["set_text"])
    report = run(backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.FAILED


def test_an_unknown_method_fails_without_delivering():
    log, backends = make()
    report = run(method="shout", backends=backends, log=log)
    assert report.result.outcome is DeliveryOutcome.FAILED
    assert log == [("foreground_hwnd",), ("is_elevated_window", TARGET_HWND)]


# helpers and defaults


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", False),
        ("word", False),
        ("word ", True),
        ("word\n", True),
        ("word\t", True),
        ("word ", True),
        (" ", True),
        ("\U0001f600", False),
    ],
)
def test_ends_with_whitespace(text, expected):
    assert ends_with_whitespace(text) is expected


def test_default_backends_are_the_win32_modules():
    from spells.win32 import clipboard as win32_clipboard
    from spells.win32 import input as win32_input
    from spells.win32 import window as win32_window

    backends = InjectBackends()
    assert backends.window is win32_window
    assert backends.input is win32_input
    assert backends.clipboard is win32_clipboard


def test_deliver_defaults_to_a_high_resolution_clock():
    # B3-23: time.monotonic is GetTickCount64 on Windows and steps in 15.6 ms, which
    # cannot measure a delivery against the 100 ms budget of spec 12.
    parameters = inspect.signature(deliver).parameters
    assert parameters["clock"].default is time.perf_counter
    assert parameters["sleeper"].default is time.sleep
    assert parameters["restore_delay_s"].default == 0.3


def test_delivery_report_defaults_and_outcome_shortcut():
    report = DeliveryReport(DeliveryResult(DeliveryOutcome.TYPED))
    assert report.elapsed_ms is None
    assert report.clipboard_restored is None
    assert report.outcome is DeliveryOutcome.TYPED
