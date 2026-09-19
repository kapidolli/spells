"""Shared fixtures and fakes for the ui unit tests (no tests of its own).

Every test_ui_* module imports this first: it forces the offscreen Qt platform before
PySide6 loads, hands out the one QApplication of the process, and provides fakes for the
pipeline, the engine supervisor, the hotkey thread and the history store so no thread,
engine or Win32 call is involved.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import dataclass, field
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from spells.audio import CaptureInfo
from spells.config import ConfigStore
from spells.gpu import GpuDevice, GpuSelection
from spells.history import AudioPolicy, HistoryEntry
from spells.hotkey import HotkeyStats
from spells.models import Engine, EngineState, StageTimings
from spells.pipeline import Notice, PillState, PipelineEvent, TrayState
from spells.quality import CheckResult, QualitySignals

WHISPER = Engine.WHISPER
LLAMA = Engine.LLAMA


def qt_app() -> QtWidgets.QApplication:
    """The process-wide QApplication, created on first use."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["spells-tests"])
    return app


def flush(app: QtWidgets.QApplication, rounds: int = 3) -> None:
    """Deliver queued signals and deferred deletes."""
    for _ in range(rounds):
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def event(
    pill: PillState = PillState.IDLE,
    tray: TrayState = TrayState.READY,
    **changes,
) -> PipelineEvent:
    return PipelineEvent(pill=pill, tray=tray, **changes)


class FakePipeline:
    """The Pipeline slice the ui consumes."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.retry_result = True
        self.timings: list[StageTimings] = []
        self.captures: list[CaptureInfo] = []
        self.guards: dict[str, int] = {}
        self.checks: list[tuple[str, str]] = []
        self.check_result = CheckResult("GOOD", "reads correctly")

    def check_transcript(self, text: str, language: str = "") -> CheckResult:
        self.checks.append((text, language))
        if isinstance(self.check_result, BaseException):
            raise self.check_result
        return self.check_result

    def retry_last(self) -> bool:
        self.calls.append("retry_last")
        return self.retry_result

    @property
    def retry_available(self) -> bool:
        return self.retry_result

    def recent_timings(self) -> list[StageTimings]:
        return list(self.timings)

    def recent_captures(self) -> list[CaptureInfo]:
        return list(self.captures)

    def guard_counts(self) -> dict[str, int]:
        return dict(self.guards)


class FakeEngines:
    """The EngineSupervisor slice the ui consumes, scripted per test."""

    def __init__(self) -> None:
        self.states = {WHISPER: EngineState.STARTING, LLAMA: EngineState.STARTING}
        self.reasons = {WHISPER: "ok", LLAMA: "ok"}
        self.variants = {WHISPER: "vulkan", LLAMA: "vulkan"}
        self.verified: dict[Engine, bool | None] = {WHISPER: True, LLAMA: None}
        self.calls: list[tuple] = []

    def status(self, engine: Engine) -> EngineState:
        return self.states[engine]

    def reason(self, engine: Engine) -> str:
        return self.reasons[engine]

    def variant(self, engine: Engine) -> str:
        return self.variants[engine]

    def gpu_verified(self, engine: Engine) -> bool | None:
        return self.verified[engine]

    def restart(self, engine: Engine) -> None:
        self.calls.append(("restart", engine))

    def set_gpu(self, selection: GpuSelection) -> None:
        self.calls.append(("set_gpu", selection))


class FakeHotkey:
    """The HotkeyThread slice the ui consumes: update_chords, the chords property, stats."""

    def __init__(self, chords: tuple = ()) -> None:
        self._chords = tuple(chords)
        self.pushed: list[tuple] = []
        self.stats = HotkeyStats()

    @property
    def chords(self) -> tuple:
        return self._chords

    def update_chords(self, chords) -> None:
        self._chords = tuple(chords)
        self.pushed.append(self._chords)


class FakeHistory:
    def __init__(self, entries: list[HistoryEntry] | None = None) -> None:
        self.entries = list(entries or [])
        self.calls: list[tuple] = []
        self.retention = "100"
        self.recordings: dict[str, Path] = {}

    def recent(self, limit: int = 100) -> list[HistoryEntry]:
        self.calls.append(("recent", limit))
        return list(self.entries[:limit])

    def search(self, query: str, limit: int = 100) -> list[HistoryEntry]:
        self.calls.append(("search", query, limit))
        needle = query.casefold()
        return [
            e
            for e in self.entries
            if needle in e.raw_text.casefold() or needle in e.cleaned_text.casefold()
        ][:limit]

    def clear(self) -> None:
        self.calls.append(("clear",))
        self.entries = []

    def recent_timings(self, limit: int = 20) -> list[StageTimings]:
        return [e.timings for e in self.entries[:limit]]

    def set_retention(self, retention: str) -> None:
        self.calls.append(("set_retention", retention))
        self.retention = retention

    def delete(self, entry_id: int) -> bool:
        self.calls.append(("delete", entry_id))
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.id != entry_id]
        return len(self.entries) != before

    def set_check(self, entry_id: int, verdict: str, reason: str, checked_at: float = 0.0) -> bool:
        self.calls.append(("set_check", entry_id, verdict, reason))
        for entry in self.entries:
            if entry.id == entry_id:
                entry.check_verdict = verdict
                entry.check_reason = reason
                entry.checked_at = checked_at or 1_700_000_500.0
                return True
        return False

    def recording_path(self, entry: HistoryEntry) -> Path | None:
        path = self.recordings.get(entry.audio_file)
        return path if path is not None and path.is_file() else None

    def recordings_usage(self) -> tuple[int, int]:
        paths = [p for p in self.recordings.values() if p.is_file()]
        return len(paths), sum(p.stat().st_size for p in paths)

    def clear_recordings(self) -> int:
        self.calls.append(("clear_recordings",))
        removed = 0
        for path in self.recordings.values():
            if path.is_file():
                path.unlink()
                removed += 1
        self.recordings = {}
        for entry in self.entries:
            entry.audio_file = ""
        return removed

    def prune_recordings(self, audio: AudioPolicy) -> int:
        self.calls.append(("prune_recordings", audio))
        return 0


def history_entry(
    index: int,
    raw: str = "raw text",
    cleaned: str = "Cleaned text.",
    **overrides,
) -> HistoryEntry:
    fields = {
        "id": index,
        "created_at": 1_700_000_000.0 + index,
        "raw_text": raw,
        "cleaned_text": cleaned,
        "delivered_text": cleaned,
        "app_process": "notepad.exe",
        "app_title": "Untitled",
        "language": "en",
        "used_llm": True,
        "cleanup_reason": "ok",
        "outcome": "pasted",
        "timings": StageTimings(press_to_pill_ms=40.0, mic_open_ms=30.0, delivery_ms=5.0),
        "signals": QualitySignals(
            avg_logprob=-0.2,
            source="whisper-server",
            audio_s=4.0,
            word_count=8,
            words_per_minute=120.0,
            filler_count=1,
        ),
        "quality_label": "good",
        "quality_reason": "Recognition confidence and speaking rate both look normal.",
    }
    fields.update(overrides)
    return HistoryEntry(**fields)


DEVICES = (
    GpuDevice(raw_index=0, list_name="Vulkan0", name="Radeon 610M", memory_mb=8000, free_mb=7000, uma=True),
    GpuDevice(raw_index=1, list_name="Vulkan1", name="RTX 5060", memory_mb=8151, free_mb=8000, uma=False),
)
SELECTION = GpuSelection(raw_index=1, name="RTX 5060", memory_mb=8151, devices=DEVICES)


@dataclass
class Messages:
    """Collects what the dialogs would have shown in a QMessageBox."""

    shown: list[tuple[str, str, str]] = field(default_factory=list)

    def __call__(self, kind: str, title: str, text: str) -> None:
        self.shown.append((kind, title, text))

    @property
    def texts(self) -> list[str]:
        return [text for _kind, _title, text in self.shown]


def make_config(tmp_path: Path) -> ConfigStore:
    return ConfigStore(tmp_path / "settings.json")


__all__ = [
    "DEVICES",
    "LLAMA",
    "SELECTION",
    "WHISPER",
    "FakeEngines",
    "FakeHistory",
    "FakeHotkey",
    "FakePipeline",
    "Messages",
    "Notice",
    "PillState",
    "PipelineEvent",
    "TrayState",
    "event",
    "flush",
    "history_entry",
    "make_config",
    "qt_app",
]
