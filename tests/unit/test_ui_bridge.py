"""spells.ui.bridge: every foreign-thread callback lands on the Qt thread.

The pipeline, the engine supervisor, the hotkey thread and the settings store call back on
their own threads (spec 5.1, 5.2, B3-8); the bridge only emits queued signals there and the
handlers run on the thread that owns the QApplication.
"""

from __future__ import annotations

import threading
import time

import pytest
from PySide6 import QtCore

from spells.models import EngineState
from spells.ui.bridge import UiBridge

from .test_ui_support import WHISPER, event, flush, qt_app


@pytest.fixture(scope="module")
def app():
    return qt_app()


class Receiver(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple] = []

    def on_anything(self, *args) -> None:
        self.seen.append((threading.get_ident(), QtCore.QThread.currentThread(), args))


def run_on_thread(fn) -> float:
    """Call fn on a fresh thread; returns how long the call itself blocked, in ms."""
    elapsed: list[float] = []

    def body() -> None:
        started = time.perf_counter()
        fn()
        elapsed.append((time.perf_counter() - started) * 1000.0)

    thread = threading.Thread(target=body)
    thread.start()
    thread.join(5.0)
    assert not thread.is_alive()
    return elapsed[0]


def test_pipeline_event_is_delivered_on_the_qt_thread(app):
    bridge = UiBridge()
    receiver = Receiver()
    bridge.connect_pipeline(receiver.on_anything)
    ev = event()
    run_on_thread(lambda: bridge.on_pipeline_event(ev))
    assert receiver.seen == []
    flush(app)
    ident, qthread, args = receiver.seen[0]
    assert ident == threading.get_ident()
    assert qthread is app.thread()
    assert args == (ev,)


def test_engine_status_is_delivered_on_the_qt_thread(app):
    bridge = UiBridge()
    receiver = Receiver()
    bridge.connect_engine_status(receiver.on_anything)
    run_on_thread(lambda: bridge.on_engine_status(WHISPER, EngineState.READY, "ok"))
    flush(app)
    ident, _qthread, args = receiver.seen[0]
    assert ident == threading.get_ident()
    assert args == (WHISPER, EngineState.READY, "ok")


def test_hotkey_error_is_delivered_on_the_qt_thread(app):
    bridge = UiBridge()
    receiver = Receiver()
    bridge.connect_hotkey_error(receiver.on_anything)
    run_on_thread(lambda: bridge.on_hotkey_error("SetWindowsHookEx failed"))
    flush(app)
    ident, _qthread, args = receiver.seen[0]
    assert ident == threading.get_ident()
    assert args == ("SetWindowsHookEx failed",)


def test_settings_change_is_delivered_on_the_qt_thread(app, tmp_path):
    from spells.config import default_settings

    bridge = UiBridge()
    receiver = Receiver()
    bridge.connect_settings(receiver.on_anything)
    settings = default_settings()
    run_on_thread(lambda: bridge.on_settings(settings))
    flush(app)
    ident, _qthread, args = receiver.seen[0]
    assert ident == threading.get_ident()
    assert args[0] is settings


def test_a_callback_never_blocks_the_foreign_thread(app):
    bridge = UiBridge()
    receiver = Receiver()
    bridge.connect_pipeline(receiver.on_anything)
    blocked_ms = run_on_thread(lambda: [bridge.on_pipeline_event(event()) for _ in range(200)])
    assert blocked_ms < 500.0
    flush(app)
    assert len(receiver.seen) == 200


def test_a_call_from_the_qt_thread_is_still_queued_not_reentrant(app):
    bridge = UiBridge()
    receiver = Receiver()
    bridge.connect_pipeline(receiver.on_anything)
    bridge.on_pipeline_event(event())
    assert receiver.seen == []
    flush(app)
    assert len(receiver.seen) == 1


def test_plain_callables_receive_on_the_qt_thread_too(app):
    bridge = UiBridge()
    seen: list[int] = []
    bridge.connect_hotkey_error(lambda message: seen.append(threading.get_ident()))
    run_on_thread(lambda: bridge.on_hotkey_error("x"))
    flush(app)
    assert seen == [threading.get_ident()]
