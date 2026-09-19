"""Dictation history: a small SQLite store of recent dictations, and its export.

Spec 5.2 (history row), 6 step 9, 8.4 (the quality fields), 14.4 (History page),
15 (location and retention), 17 (local, limitable, clearable). Stores text, app,
language, delivery outcome, the stage timings and the quality signals of each
dictation.

Audio. Spec 17 promises that audio is never written to disk, and that stands
unless the user turns "Keep the recordings" on. With it on, add() writes the
dictation's PCM as a 16 kHz mono WAV named after its row into a recordings folder
beside the database, the row points at the file, deleting a row deletes its file,
clearing the history clears the folder, and a retention of its own (a number of
recordings and a number of megabytes, oldest deleted first) runs after each
dictation. The recordings never leave the machine and never enter a diagnostics
bundle.

created_at is wall-clock seconds as returned by time.time(): age pruning
("7d", "30d") compares it against time.time(), so any other clock would
expire rows at the wrong moment. add() fills it in when it is 0 or None.

Thread safety: one connection guarded by a re-entrant lock, so the pipeline
thread can add entries while the settings window queries them.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import sqlite3
import threading
import time
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, Self

from spells.models import StageTimings
from spells.quality import QualitySignals, RowStats, signals_from_dict, signals_to_dict

log = logging.getLogger(__name__)

SCHEMA_VERSION = 3

Retention = Literal["100", "7d", "30d", "off"]

RETENTIONS: tuple[Retention, ...] = ("100", "7d", "30d", "off")
_COUNT_LIMIT = 100
_RETENTION_DAYS: dict[str, int] = {"7d": 7, "30d": 30}
_DAY_S = 86_400.0
_TIMING_FIELDS = {f.name for f in dataclasses.fields(StageTimings)}

RECORDINGS_DIR_NAME = "recordings"
RECORDING_SAMPLE_RATE = 16_000
RECORDING_SUFFIX = ".wav"
MEGABYTE = 1024 * 1024
DEFAULT_AUDIO_KEEP_COUNT = 200
DEFAULT_AUDIO_KEEP_MB = 1000

EXPORT_FORMATS = ("json", "csv", "markdown")
EXPORT_SUFFIXES = {"json": ".json", "csv": ".csv", "markdown": ".md"}
EXPORT_LABELS = {
    "json": "JSON, every field",
    "csv": "CSV, every field",
    "markdown": "Markdown, a transcript log to read",
}

OUTCOME_NO_AUDIO = "no_audio"
OUTCOME_NOT_WRITTEN = "not_written"

MODE_DICTATE = "dictate"
MODE_COMPOSE = "compose"
MODE_EDIT = "edit"
WRITING_MODES = (MODE_COMPOSE, MODE_EDIT)

GATE_REASONS = (
    "disabled",
    "profile_off",
    "language_unscored",
    "short_clean",
    "clean_text",
    "engine_not_ready",
    "cpu_fallback",
    "compose",
)


@dataclass
class HistoryEntry:
    """One dictation as shown in the History page (spec 6 step 9, 8.4)."""

    id: int | None
    created_at: float
    raw_text: str
    cleaned_text: str
    delivered_text: str
    app_process: str
    app_title: str
    language: str
    used_llm: bool
    cleanup_reason: str
    outcome: str
    timings: StageTimings
    asr_engine: str = ""
    cleanup_engine: str = ""
    audio_file: str = ""
    signals: QualitySignals = dataclasses.field(default_factory=QualitySignals)
    quality_label: str = ""
    quality_reason: str = ""
    check_verdict: str = ""
    check_reason: str = ""
    checked_at: float = 0.0
    mode: str = MODE_DICTATE
    instruction: str = ""
    selection_chars: int = 0

    @property
    def wrote(self) -> bool:
        """True for a compose or edit row, where the delivered text is written, not dictated."""
        return self.mode in WRITING_MODES


@dataclass(frozen=True)
class AudioPolicy:
    """Whether to keep a dictation's audio, and how much of it to keep (spec 15).

    max_mb counts mebibytes, which is what Windows shows as MB.
    """

    keep: bool = False
    max_files: int = DEFAULT_AUDIO_KEEP_COUNT
    max_mb: int = DEFAULT_AUDIO_KEEP_MB

    @property
    def max_bytes(self) -> int:
        return max(0, self.max_mb) * MEGABYTE


def default_history_path() -> Path:
    """%LOCALAPPDATA%\\Spells\\history.db (spec 15)."""
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Local"
    return root / "Spells" / "history.db"


def recordings_dir_for(history_path: Path) -> Path:
    """The recordings folder beside a history database (spec 15)."""
    return Path(history_path).parent / RECORDINGS_DIR_NAME


def default_recordings_dir() -> Path:
    """%LOCALAPPDATA%\\Spells\\recordings\\ (spec 15)."""
    return recordings_dir_for(default_history_path())


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    raw_text TEXT NOT NULL,
    cleaned_text TEXT NOT NULL,
    delivered_text TEXT NOT NULL,
    app_process TEXT NOT NULL,
    app_title TEXT NOT NULL,
    language TEXT NOT NULL,
    used_llm INTEGER NOT NULL,
    cleanup_reason TEXT NOT NULL,
    outcome TEXT NOT NULL,
    timings TEXT NOT NULL,
    asr_engine TEXT NOT NULL DEFAULT '',
    cleanup_engine TEXT NOT NULL DEFAULT '',
    audio_file TEXT NOT NULL DEFAULT '',
    signals TEXT NOT NULL DEFAULT '',
    quality_label TEXT NOT NULL DEFAULT '',
    quality_reason TEXT NOT NULL DEFAULT '',
    check_verdict TEXT NOT NULL DEFAULT '',
    check_reason TEXT NOT NULL DEFAULT '',
    checked_at REAL NOT NULL DEFAULT 0,
    mode TEXT NOT NULL DEFAULT 'dictate',
    instruction TEXT NOT NULL DEFAULT '',
    selection_chars INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS entries_created_at ON entries (created_at DESC, id DESC);
"""

_ADDED_IN_2: tuple[tuple[str, str], ...] = (
    ("asr_engine", "asr_engine TEXT NOT NULL DEFAULT ''"),
    ("cleanup_engine", "cleanup_engine TEXT NOT NULL DEFAULT ''"),
    ("audio_file", "audio_file TEXT NOT NULL DEFAULT ''"),
    ("signals", "signals TEXT NOT NULL DEFAULT ''"),
    ("quality_label", "quality_label TEXT NOT NULL DEFAULT ''"),
    ("quality_reason", "quality_reason TEXT NOT NULL DEFAULT ''"),
    ("check_verdict", "check_verdict TEXT NOT NULL DEFAULT ''"),
    ("check_reason", "check_reason TEXT NOT NULL DEFAULT ''"),
    ("checked_at", "checked_at REAL NOT NULL DEFAULT 0"),
)

_ADDED_IN_3: tuple[tuple[str, str], ...] = (
    ("mode", "mode TEXT NOT NULL DEFAULT 'dictate'"),
    ("instruction", "instruction TEXT NOT NULL DEFAULT ''"),
    ("selection_chars", "selection_chars INTEGER NOT NULL DEFAULT 0"),
)

_COLUMNS = (
    "id",
    "created_at",
    "raw_text",
    "cleaned_text",
    "delivered_text",
    "app_process",
    "app_title",
    "language",
    "used_llm",
    "cleanup_reason",
    "outcome",
    "timings",
    "asr_engine",
    "cleanup_engine",
    "audio_file",
    "signals",
    "quality_label",
    "quality_reason",
    "check_verdict",
    "check_reason",
    "checked_at",
    "mode",
    "instruction",
    "selection_chars",
)
_SELECT = "SELECT " + ", ".join(_COLUMNS) + " FROM entries"
_INSERT = (
    "INSERT INTO entries (" + ", ".join(_COLUMNS[1:]) + ") VALUES ("
    + ", ".join("?" for _ in _COLUMNS[1:]) + ")"
)
_NEWEST_FIRST = " ORDER BY created_at DESC, id DESC LIMIT ?"
_SEARCH_WHERE = (
    " WHERE spells_match(?, raw_text, cleaned_text, delivered_text, app_process, app_title)"
)


def _match(needle: str, *fields: str | None) -> int:
    """SQL function: case-folded substring test over several columns."""
    return int(any(needle in (field or "").casefold() for field in fields))


def _timings_to_json(timings: StageTimings) -> str:
    return json.dumps(dataclasses.asdict(timings))


def _timings_from_json(text: str) -> StageTimings:
    """Rebuild StageTimings, ignoring keys this version does not know."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return StageTimings()
    if not isinstance(data, dict):
        return StageTimings()
    kwargs = {key: value for key, value in data.items() if key in _TIMING_FIELDS}
    extra = kwargs.get("extra")
    kwargs["extra"] = dict(extra) if isinstance(extra, dict) else {}
    return StageTimings(**kwargs)


def _signals_to_json(signals: QualitySignals) -> str:
    return json.dumps(signals_to_dict(signals))


def _signals_from_json(text: str) -> QualitySignals:
    if not text:
        return QualitySignals()
    try:
        return signals_from_dict(json.loads(text))
    except (TypeError, ValueError):
        return QualitySignals()


def _row_to_entry(row: tuple) -> HistoryEntry:
    (
        rowid,
        created_at,
        raw_text,
        cleaned_text,
        delivered_text,
        app_process,
        app_title,
        language,
        used_llm,
        cleanup_reason,
        outcome,
        timings,
        asr_engine,
        cleanup_engine,
        audio_file,
        signals,
        quality_label,
        quality_reason,
        check_verdict,
        check_reason,
        checked_at,
        mode,
        instruction,
        selection_chars,
    ) = row
    return HistoryEntry(
        id=rowid,
        created_at=created_at,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        delivered_text=delivered_text,
        app_process=app_process,
        app_title=app_title,
        language=language,
        used_llm=bool(used_llm),
        cleanup_reason=cleanup_reason,
        outcome=outcome,
        timings=_timings_from_json(timings),
        asr_engine=asr_engine or "",
        cleanup_engine=cleanup_engine or "",
        audio_file=audio_file or "",
        signals=_signals_from_json(signals or ""),
        quality_label=quality_label or "",
        quality_reason=quality_reason or "",
        check_verdict=check_verdict or "",
        check_reason=check_reason or "",
        checked_at=float(checked_at or 0.0),
        mode=mode or MODE_DICTATE,
        instruction=instruction or "",
        selection_chars=int(selection_chars or 0),
    )


def _check_retention(retention: str) -> Retention:
    if retention not in RETENTIONS:
        raise ValueError(f"unknown retention {retention!r}, expected one of {RETENTIONS}")
    return retention  # type: ignore[return-value]


def recording_name(entry_id: int) -> str:
    """The WAV file name of a history row."""
    return f"{int(entry_id):06d}{RECORDING_SUFFIX}"


def _write_wav(path: Path, pcm16: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm16)


def wav_duration_s(path: Path) -> float | None:
    """The length of a WAV file, or None when it cannot be read."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (OSError, wave.Error):
        return None


class HistoryStore:
    """SQLite-backed history with the retention policy of spec 15.

    With path None there is no database: add() returns -1 and every query
    returns []. Retention "off" stops storing (add() returns -1) and makes
    prune() a no-op, but never deletes what is already there and reads keep
    working; clear() is the user's explicit way to delete rows.

    Reads prune first, so an app that sat idle past a retention age does not
    show expired rows.

    The recordings folder defaults to a folder beside the database. It is created
    on the first recording, not at startup, so a user who never turns recordings
    on never gets the folder.
    """

    def __init__(
        self,
        path: Path | None,
        retention: Retention = "100",
        *,
        recordings_dir: Path | None = None,
    ) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        if recordings_dir is not None:
            self.recordings_dir: Path | None = Path(recordings_dir)
        elif self.path is not None:
            self.recordings_dir = recordings_dir_for(self.path)
        else:
            self.recordings_dir = None
        self._retention = _check_retention(retention)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        _migrate(conn)
        conn.create_function("spells_match", 6, _match, deterministic=True)
        self._conn = conn
        self.prune()

    # Lifecycle

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # State

    @property
    def retention(self) -> Retention:
        return self._retention

    @property
    def enabled(self) -> bool:
        """True when new dictations are being recorded."""
        return self._conn is not None and self._retention != "off"

    def set_retention(self, retention: Retention) -> None:
        """Change the policy and apply it right away (spec 14.4: no restart)."""
        with self._lock:
            self._retention = _check_retention(retention)
            if self._conn is not None:
                self._prune_locked()

    # Writes

    def add(
        self,
        entry: HistoryEntry,
        pcm16: bytes | None = None,
        sample_rate: int = RECORDING_SAMPLE_RATE,
        audio: AudioPolicy | None = None,
    ) -> int:
        """Insert one dictation and apply the retention. Returns the rowid, or -1 when disabled.

        The PCM is written only when the caller passes it and audio.keep is true, so the
        promise of spec 17 holds by construction: without the setting no audio reaches this
        function. A recording that cannot be written is logged and never fails the dictation.
        """
        with self._lock:
            if not self.enabled:
                return -1
            assert self._conn is not None
            created_at = entry.created_at if entry.created_at else time.time()
            cursor = self._conn.execute(
                _INSERT,
                (
                    created_at,
                    entry.raw_text,
                    entry.cleaned_text,
                    entry.delivered_text,
                    entry.app_process,
                    entry.app_title,
                    entry.language,
                    int(entry.used_llm),
                    entry.cleanup_reason,
                    entry.outcome,
                    _timings_to_json(entry.timings),
                    entry.asr_engine,
                    entry.cleanup_engine,
                    "",
                    _signals_to_json(entry.signals),
                    entry.quality_label,
                    entry.quality_reason,
                    entry.check_verdict,
                    entry.check_reason,
                    entry.checked_at,
                    entry.mode or MODE_DICTATE,
                    entry.instruction,
                    int(entry.selection_chars),
                ),
            )
            rowid = cursor.lastrowid
            row_id = int(rowid) if rowid is not None else -1
            if row_id > 0 and pcm16 and audio is not None and audio.keep:
                self._store_recording_locked(row_id, pcm16, sample_rate, audio)
            self._prune_locked()
            return row_id

    def set_check(self, entry_id: int, verdict: str, reason: str, checked_at: float = 0.0) -> bool:
        """Store one on-demand check result on a row (spec 8.4)."""
        with self._lock:
            if self._conn is None:
                return False
            cursor = self._conn.execute(
                "UPDATE entries SET check_verdict = ?, check_reason = ?, checked_at = ? WHERE id = ?",
                (verdict, reason, checked_at or time.time(), int(entry_id)),
            )
            return bool(cursor.rowcount)

    def delete(self, entry_id: int) -> bool:
        """Delete one row and its recording."""
        with self._lock:
            if self._conn is None:
                return False
            row = self._conn.execute(
                "SELECT audio_file FROM entries WHERE id = ?", (int(entry_id),)
            ).fetchone()
            cursor = self._conn.execute("DELETE FROM entries WHERE id = ?", (int(entry_id),))
            if row and row[0]:
                self._remove_files([str(row[0])])
            return bool(cursor.rowcount)

    def clear(self) -> None:
        """Delete every entry and its recording, and compact the file (spec 17)."""
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute("DELETE FROM entries")
            self._compact_locked()
            self.clear_recordings()

    def prune(self) -> None:
        """Apply the retention: newest 100, or younger than 7 or 30 days. No-op under "off"."""
        with self._lock:
            if self._conn is None:
                return
            self._prune_locked()

    def recording_path(self, entry: HistoryEntry | str) -> Path | None:
        """The WAV of a row, or None when there is none or the file is gone."""
        name = entry if isinstance(entry, str) else entry.audio_file
        if not name or self.recordings_dir is None:
            return None
        path = self.recordings_dir / name
        try:
            return path if path.is_file() else None
        except OSError:
            return None

    def recordings_usage(self) -> tuple[int, int]:
        """(files, bytes) in the recordings folder."""
        files = self._recording_files()
        return len(files), sum(size for _name, size, _mtime in files)

    def clear_recordings(self) -> int:
        """Delete every recording and forget them on the rows. Returns the count deleted."""
        with self._lock:
            names = [name for name, _size, _mtime in self._recording_files()]
            removed = self._remove_files(names)
            if self._conn is not None:
                self._conn.execute("UPDATE entries SET audio_file = '' WHERE audio_file <> ''")
            return removed

    def prune_recordings(self, audio: AudioPolicy) -> int:
        """Keep the newest max_files recordings within max_mb, oldest deleted first."""
        with self._lock:
            return self._prune_recordings_locked(audio)

    # Reads (each prunes first so expired rows never surface)

    def recent(self, limit: int = 100) -> list[HistoryEntry]:
        with self._lock:
            if self._conn is None:
                return []
            self._prune_locked()
            rows = self._conn.execute(_SELECT + _NEWEST_FIRST, (limit,)).fetchall()
        return [_row_to_entry(row) for row in rows]

    def search(self, query: str, limit: int = 100) -> list[HistoryEntry]:
        """Case-insensitive substring search over the texts and the app fields, newest first."""
        with self._lock:
            if self._conn is None:
                return []
            self._prune_locked()
            rows = self._conn.execute(
                _SELECT + _SEARCH_WHERE + _NEWEST_FIRST, (query.casefold(), limit)
            ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def recent_timings(self, limit: int = 20) -> list[StageTimings]:
        """Timings of the newest dictations, for the rolling medians (decisions V1-4, V1-13)."""
        with self._lock:
            if self._conn is None:
                return []
            self._prune_locked()
            rows = self._conn.execute(
                "SELECT timings FROM entries" + _NEWEST_FIRST, (limit,)
            ).fetchall()
        return [_timings_from_json(row[0]) for row in rows]

    # Internals (call with the lock held and a connection open)

    def _prune_locked(self) -> None:
        assert self._conn is not None
        retention = self._retention
        if retention == "off":
            return
        if retention == "100":
            doomed = self._conn.execute(
                "SELECT audio_file FROM entries WHERE audio_file <> '' AND id NOT IN "
                "(SELECT id FROM entries ORDER BY created_at DESC, id DESC LIMIT ?)",
                (_COUNT_LIMIT,),
            ).fetchall()
            self._conn.execute(
                "DELETE FROM entries WHERE id NOT IN "
                "(SELECT id FROM entries ORDER BY created_at DESC, id DESC LIMIT ?)",
                (_COUNT_LIMIT,),
            )
        else:
            cutoff = time.time() - _RETENTION_DAYS[retention] * _DAY_S
            doomed = self._conn.execute(
                "SELECT audio_file FROM entries WHERE audio_file <> '' AND created_at < ?",
                (cutoff,),
            ).fetchall()
            self._conn.execute("DELETE FROM entries WHERE created_at < ?", (cutoff,))
        if doomed:
            self._remove_files([str(row[0]) for row in doomed])

    def _store_recording_locked(
        self, row_id: int, pcm16: bytes, sample_rate: int, audio: AudioPolicy
    ) -> None:
        assert self._conn is not None
        if self.recordings_dir is None:
            return
        name = recording_name(row_id)
        try:
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            _write_wav(self.recordings_dir / name, pcm16, sample_rate)
        except (OSError, wave.Error):
            log.warning("could not write the recording of dictation %d", row_id, exc_info=True)
            return
        self._conn.execute("UPDATE entries SET audio_file = ? WHERE id = ?", (name, row_id))
        self._prune_recordings_locked(audio)

    def _recording_files(self) -> list[tuple[str, int, float]]:
        """(name, bytes, modified) of every WAV in the folder, newest first."""
        if self.recordings_dir is None:
            return []
        found: list[tuple[str, int, float]] = []
        try:
            with os.scandir(self.recordings_dir) as entries:
                for item in entries:
                    if not item.name.lower().endswith(RECORDING_SUFFIX):
                        continue
                    try:
                        info = item.stat()
                    except OSError:
                        continue
                    if item.is_file():
                        found.append((item.name, info.st_size, info.st_mtime))
        except (FileNotFoundError, NotADirectoryError):
            return []
        except OSError:
            log.warning("could not list %s", self.recordings_dir, exc_info=True)
            return []
        found.sort(key=lambda row: (row[2], row[0]), reverse=True)
        return found

    def _prune_recordings_locked(self, audio: AudioPolicy) -> int:
        files = self._recording_files()
        max_files = max(0, audio.max_files)
        max_bytes = audio.max_bytes
        doomed: list[str] = []
        kept_bytes = 0
        for index, (name, size, _mtime) in enumerate(files):
            over_count = index >= max_files
            over_bytes = kept_bytes + size > max_bytes
            if over_count or over_bytes:
                doomed.append(name)
            else:
                kept_bytes += size
        if not doomed:
            return 0
        removed = self._remove_files(doomed)
        if self._conn is not None:
            placeholders = ", ".join("?" for _ in doomed)
            self._conn.execute(
                f"UPDATE entries SET audio_file = '' WHERE audio_file IN ({placeholders})",
                doomed,
            )
        return removed

    def _remove_files(self, names: Iterable[str]) -> int:
        if self.recordings_dir is None:
            return 0
        removed = 0
        for name in names:
            if not name:
                continue
            try:
                (self.recordings_dir / name).unlink()
                removed += 1
            except FileNotFoundError:
                continue
            except OSError:
                log.warning("could not delete the recording %s", name, exc_info=True)
        return removed

    def _compact_locked(self) -> None:
        """Best effort: fold the WAL back and vacuum so deleted text leaves the file."""
        assert self._conn is not None
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.execute("VACUUM")
        except sqlite3.OperationalError:
            pass


def _migrate(conn: sqlite3.Connection) -> int:
    """Bring an existing database forward to SCHEMA_VERSION without losing rows (spec 15).

    Version 1 is the released shape. Its rows keep every value they have and gain the
    columns of version 2 with empty defaults, so an upgraded history reads exactly as it
    did, with the quality fields blank until the next dictation fills them. Version 3 adds
    the writing columns of spec 8.5 the same way: every existing row reads as the dictation
    it was, because the mode defaults to "dictate". A database written by a newer version is
    left alone: every read names its columns, so the extra ones a future version adds are
    simply not selected.
    """
    stored = _stored_version(conn)
    if stored > SCHEMA_VERSION:
        log.warning(
            "the history database is version %d, newer than this app (%d); reading it as it is",
            stored,
            SCHEMA_VERSION,
        )
        return stored
    existing = {row[1] for row in conn.execute("PRAGMA table_info(entries)")}
    for name, ddl in _ADDED_IN_2 + _ADDED_IN_3:
        if name not in existing:
            conn.execute(f"ALTER TABLE entries ADD COLUMN {ddl}")
    if stored != SCHEMA_VERSION:
        log.info("history database migrated from version %d to %d", stored, SCHEMA_VERSION)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    return SCHEMA_VERSION


def _stored_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.DatabaseError:
        return 1
    if not row:
        return 1
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 1


def split_cleanup_reason(entry: HistoryEntry) -> tuple[str, str]:
    """(gate reason, guard rejection) of a row; each is empty when it does not apply.

    One stored reason carries both cases (spec 8.1 and 8.3), so the export separates them:
    a gate reason says cleanup never ran, a guard rejection says it ran and its output was
    refused. "ok" is neither.
    """
    reason = entry.cleanup_reason or ""
    if not reason or reason == "ok":
        return "", ""
    if reason in GATE_REASONS:
        return reason, ""
    return "", reason


def row_stats(entry: HistoryEntry) -> RowStats:
    """One row's contribution to the History page summary (spec 14.4)."""
    signals = entry.signals
    changed = (entry.cleaned_text != entry.raw_text) if entry.used_llm else None
    return RowStats(
        words_per_minute=signals.words_per_minute,
        word_count=signals.word_count,
        filler_count=signals.filler_count,
        cleanup_changed=changed,
    )


def local_time(created_at: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))


def export_file_name(fmt: str, when: float | None = None) -> str:
    """spells-history-2026-09-17.json and its siblings."""
    stamp = time.strftime("%Y-%m-%d", time.localtime(when if when is not None else time.time()))
    return f"spells-history-{stamp}{EXPORT_SUFFIXES.get(fmt, '.txt')}"


def entry_to_dict(entry: HistoryEntry) -> dict:
    """Everything the export carries for one dictation (spec 14.4)."""
    gate_reason, guard_rejection = split_cleanup_reason(entry)
    signals = entry.signals
    timings = entry.timings
    return {
        "time": local_time(entry.created_at),
        "created_at": entry.created_at,
        "mode": entry.mode or MODE_DICTATE,
        "instruction": entry.instruction,
        "selection_chars": entry.selection_chars,
        "language": entry.language,
        "app_process": entry.app_process,
        "app_title": entry.app_title,
        "raw_text": entry.raw_text,
        "cleaned_text": entry.cleaned_text,
        "delivered_text": entry.delivered_text,
        "outcome": entry.outcome,
        "cleanup_ran": entry.used_llm,
        "gate_reason": gate_reason,
        "guard_rejection": guard_rejection,
        "asr_engine": entry.asr_engine,
        "cleanup_engine": entry.cleanup_engine,
        "recording": entry.audio_file,
        "quality": {
            "label": entry.quality_label,
            "reason": entry.quality_reason,
            "check_verdict": entry.check_verdict,
            "check_reason": entry.check_reason,
            "checked_at": entry.checked_at or None,
            "avg_logprob": signals.avg_logprob,
            "compression_ratio": signals.compression_ratio,
            "no_speech_prob": signals.no_speech_prob,
            "temperature": signals.temperature,
            "segments": signals.segments,
            "source": signals.source,
            "audio_s": signals.audio_s,
            "word_count": signals.word_count,
            "words_per_minute": signals.words_per_minute,
            "filler_count": signals.filler_count,
            "correction_count": signals.correction_count,
            "dropped_blocks": signals.dropped_blocks,
        },
        "timings": {
            "press_to_pill_ms": timings.press_to_pill_ms,
            "mic_open_ms": timings.mic_open_ms,
            "release_to_transcript_ms": timings.release_to_transcript_ms,
            "release_to_cleaned_ms": timings.release_to_cleaned_ms,
            "delivery_ms": timings.delivery_ms,
            **timings.extra,
        },
    }


CSV_COLUMNS = (
    "time",
    "mode",
    "instruction",
    "selection_chars",
    "language",
    "app_process",
    "app_title",
    "raw_text",
    "cleaned_text",
    "delivered_text",
    "outcome",
    "cleanup_ran",
    "gate_reason",
    "guard_rejection",
    "asr_engine",
    "cleanup_engine",
    "recording",
    "quality_label",
    "quality_reason",
    "check_verdict",
    "check_reason",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
    "temperature",
    "audio_s",
    "word_count",
    "words_per_minute",
    "filler_count",
    "correction_count",
    "dropped_blocks",
    "press_to_pill_ms",
    "mic_open_ms",
    "release_to_transcript_ms",
    "release_to_cleaned_ms",
    "delivery_ms",
)


def _csv_row(data: dict) -> list:
    quality = data["quality"]
    timings = data["timings"]
    flat = {**data, **{key: quality.get(key) for key in quality}, **timings}
    flat["quality_label"] = quality["label"]
    flat["quality_reason"] = quality["reason"]
    return [_csv_value(flat.get(column)) for column in CSV_COLUMNS]


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value


def write_json(entries: Iterable[HistoryEntry], out: IO[str]) -> int:
    """Stream a JSON array, one dictation at a time (spec 14.4)."""
    out.write("[\n")
    count = 0
    for entry in entries:
        if count:
            out.write(",\n")
        out.write(json.dumps(entry_to_dict(entry), ensure_ascii=False, indent=2))
        count += 1
    out.write("\n]\n" if count else "]\n")
    return count


def write_csv(entries: Iterable[HistoryEntry], out: IO[str]) -> int:
    writer = csv.writer(out)
    writer.writerow(CSV_COLUMNS)
    count = 0
    for entry in entries:
        writer.writerow(_csv_row(entry_to_dict(entry)))
        count += 1
    return count


def write_markdown(entries: Iterable[HistoryEntry], out: IO[str]) -> int:
    """A transcript log to read back: the time, the app and the delivered text."""
    out.write("# Spells transcript log\n\n")
    count = 0
    for entry in entries:
        app = entry.app_process or "unknown app"
        out.write(f"## {local_time(entry.created_at)}  {app}\n\n")
        if entry.wrote:
            said = (entry.instruction or entry.raw_text).strip()
            label = "Edited" if entry.mode == MODE_EDIT else "Wrote"
            out.write(f"{label} from: {said}\n\n" if said else f"{label}:\n\n")
        text = (entry.delivered_text or entry.cleaned_text or entry.raw_text).strip()
        out.write((text if text else "(nothing was delivered)") + "\n\n")
        count += 1
    if not count:
        out.write("Nothing to show.\n")
    return count


_WRITERS = {"json": write_json, "csv": write_csv, "markdown": write_markdown}


def export_entries(entries: Iterable[HistoryEntry], out: IO[str], fmt: str) -> int:
    writer = _WRITERS.get(fmt)
    if writer is None:
        raise ValueError(f"unknown export format {fmt!r}, expected one of {EXPORT_FORMATS}")
    return writer(entries, out)


def export_to_path(entries: Iterable[HistoryEntry], path: Path, fmt: str) -> int:
    """Write the export straight to a file, without building it in memory first."""
    newline = "" if fmt == "csv" else "\n"
    with open(path, "w", encoding="utf-8", newline=newline) as out:
        return export_entries(entries, out, fmt)



__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_AUDIO_KEEP_COUNT",
    "DEFAULT_AUDIO_KEEP_MB",
    "EXPORT_FORMATS",
    "EXPORT_LABELS",
    "EXPORT_SUFFIXES",
    "MODE_COMPOSE",
    "MODE_DICTATE",
    "MODE_EDIT",
    "OUTCOME_NOT_WRITTEN",
    "OUTCOME_NO_AUDIO",
    "RECORDINGS_DIR_NAME",
    "RECORDING_SAMPLE_RATE",
    "RETENTIONS",
    "SCHEMA_VERSION",
    "AudioPolicy",
    "HistoryEntry",
    "HistoryStore",
    "Retention",
    "default_history_path",
    "default_recordings_dir",
    "entry_to_dict",
    "export_entries",
    "export_file_name",
    "export_to_path",
    "local_time",
    "recording_name",
    "recordings_dir_for",
    "row_stats",
    "split_cleanup_reason",
    "wav_duration_s",
    "write_csv",
    "write_json",
    "write_markdown",
]
