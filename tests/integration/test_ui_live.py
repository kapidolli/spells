"""Live test for the pill on the real desktop (spec 14.2, mockup window rules).

Shows the pill in every state for 300 ms and checks that the foreground window never
changed (WS_EX_NOACTIVATE holds), that the window is a tool window absent from the
taskbar, and that it sits on the target window's monitor. Skips when the workstation is
locked. Run with:

    .venv/Scripts/python.exe -m pytest tests/integration/test_ui_live.py -m integration -q
"""

from __future__ import annotations

import time

import pytest

from spells.pipeline import (
    CAP_WARNING_TEXT,
    STARTING_ENGINES_TEXT,
    Notice,
    PillState,
    PipelineEvent,
    TrayState,
)
from spells.win32 import foreground_hwnd, window_process_name
from spells.win32.window import (
    WS_EX_APPWINDOW,
    WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW,
    monitor_handle_for_window,
    window_extended_style,
)

pytestmark = pytest.mark.integration

LOCK_SCREEN_PROCESSES = {"lockapp.exe", "logonui.exe"}


def locked_desktop_reason() -> str | None:
    process = window_process_name(foreground_hwnd()).lower()
    if process in LOCK_SCREEN_PROCESSES or not foreground_hwnd():
        return f"{process or 'nothing'} holds the foreground, so the workstation is locked"
    return None


_LOCKED = locked_desktop_reason()
if _LOCKED:
    pytest.skip(_LOCKED, allow_module_level=True)

from PySide6 import QtWidgets

from spells.ui.pill import Pill

TARGET = foreground_hwnd()
STATES = [
    PipelineEvent(pill=PillState.RECORDING, tray=TrayState.RECORDING, level=0.6, target_hwnd=TARGET),
    PipelineEvent(pill=PillState.RECORDING, tray=TrayState.RECORDING, level=0.6, busy=True),
    PipelineEvent(pill=PillState.LATCHED, tray=TrayState.RECORDING, level=0.3),
    PipelineEvent(pill=PillState.PROCESSING, tray=TrayState.PROCESSING),
    PipelineEvent(pill=PillState.STARTING_ENGINES, tray=TrayState.PROCESSING, text=STARTING_ENGINES_TEXT),
    PipelineEvent(pill=PillState.RECORDING, tray=TrayState.RECORDING, text=CAP_WARNING_TEXT),
    PipelineEvent(pill=PillState.IDLE, tray=TrayState.READY, notice=Notice.ERROR, notice_text="Mic is busy"),
    PipelineEvent(pill=PillState.IDLE, tray=TrayState.READY, notice=Notice.COPIED, notice_text="Copied"),
]


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(["spells-live"])


def pump(app: QtWidgets.QApplication, seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.01)


def test_pill_never_steals_focus_and_stays_off_the_taskbar(app):
    before = foreground_hwnd()
    assert before, "no foreground window to protect"
    pill = Pill()
    try:
        for ev in STATES:
            pill.apply(ev)
            pump(app, 0.3)
            assert pill.isVisible()
            assert foreground_hwnd() == before, f"foreground changed while showing {ev.pill}"
            style = window_extended_style(int(pill.winId()))
            assert style & WS_EX_NOACTIVATE, "WS_EX_NOACTIVATE missing"
            assert style & WS_EX_TOOLWINDOW, "not a tool window"
            assert not style & WS_EX_APPWINDOW, "would appear on the taskbar"
            assert int(pill.winId()) != foreground_hwnd()
            assert monitor_handle_for_window(int(pill.winId())) == monitor_handle_for_window(before), (
                "the pill must sit on the target window's monitor"
            )
        pill.apply(PipelineEvent(pill=PillState.IDLE, tray=TrayState.READY))
        pump(app, 0.4)
        assert not pill.isVisible()
    finally:
        pill.close()
        pump(app, 0.05)
