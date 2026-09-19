"""The Qt side of the thread boundary (spec 5.1, 5.2 ui row, B3-8).

Pipeline.on_event, EngineSupervisor.on_status, HotkeyThread.on_error and the ConfigStore
subscriber all run on foreign threads. Each of the callbacks here only emits a signal and
returns; every connection made through the connect_* helpers is queued, so the handlers run
on the thread that owns the QApplication, later, never inside the caller. Nothing here
blocks, and nothing here calls back into the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6 import QtCore

QUEUED = QtCore.Qt.ConnectionType.QueuedConnection


class UiBridge(QtCore.QObject):
    """Create it on the Qt thread before the threads that will call it start."""

    pipeline_event = QtCore.Signal(object)
    engine_status = QtCore.Signal(object, object, str)
    hotkey_error = QtCore.Signal(str)
    settings_changed = QtCore.Signal(object)

    # Callbacks handed to the collaborators (any thread) ---------------------------------------

    def on_pipeline_event(self, event: Any) -> None:
        self.pipeline_event.emit(event)

    def on_engine_status(self, engine: Any, state: Any, reason: str) -> None:
        self.engine_status.emit(engine, state, str(reason))

    def on_hotkey_error(self, message: str) -> None:
        self.hotkey_error.emit(str(message))

    def on_settings(self, settings: Any) -> None:
        self.settings_changed.emit(settings)

    # Queued connections for the Qt-side handlers --------------------------------------------------

    def connect_pipeline(self, slot: Callable[[Any], None]) -> None:
        self.pipeline_event.connect(slot, QUEUED)

    def connect_engine_status(self, slot: Callable[[Any, Any, str], None]) -> None:
        self.engine_status.connect(slot, QUEUED)

    def connect_hotkey_error(self, slot: Callable[[str], None]) -> None:
        self.hotkey_error.connect(slot, QUEUED)

    def connect_settings(self, slot: Callable[[Any], None]) -> None:
        self.settings_changed.connect(slot, QUEUED)


__all__ = ["UiBridge"]
