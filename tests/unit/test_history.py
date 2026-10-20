"""Unit tests for spells.history: the SQLite dictation history store."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from spells import history as history_module
from spells.history import (
    CSV_COLUMNS,
    AudioPolicy,
    HistoryEntry,
    HistoryStore,
    entry_to_dict,
    export_to_path,
    row_stats,
    split_cleanup_reason,
    write_csv,
    write_json,
    write_markdown,
)
from spells.models import StageTimings
from spells.quality import QualitySignals

BASE = 1_700_000_000.0
DAY = 86_400.0


def make_entry(n: int, **overrides) -> HistoryEntry:
    fields = {
        "id": None,
        "created_at": BASE + n,
        "raw_text": f"raw text {n}",
        "cleaned_text": f"cleaned text {n}",
        "delivered_text": f"delivered text {n}",
        "app_process": "notepad.exe",
        "app_title": "Untitled - Notepad",
        "language": "en",
        "used_llm": n % 2 == 0,
        "cleanup_reason": "ok",
        "outcome": "pasted",
        "timings": StageTimings(press_to_pill_ms=float(n), mic_open_ms=10.0 + n),
    }
    fields.update(overrides)
    return HistoryEntry(**fields)


def silence(seconds: float, rate: int = 16_000) -> bytes:
    return bytes(2 * int(rate * seconds))


def row_count(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]


@pytest.fixture
def store(tmp_path: Path):
    with HistoryStore(tmp_path / "history.db") as s:
        yield s


def test_add_returns_rowid_and_recent_is_newest_first(store):
    ids = [store.add(make_entry(n)) for n in range(5)]
    assert ids == [1, 2, 3, 4, 5]
    got = store.recent()
    assert [e.raw_text for e in got] == [f"raw text {n}" for n in (4, 3, 2, 1, 0)]
    assert [e.id for e in got] == [5, 4, 3, 2, 1]
    assert got[0].created_at == BASE + 4


def test_recent_orders_by_created_at_not_by_insertion(store):
    store.add(make_entry(1, created_at=BASE + 100))
    store.add(make_entry(2, created_at=BASE + 50))
    assert [e.raw_text for e in store.recent()] == ["raw text 1", "raw text 2"]


def test_add_round_trips_every_field(store):
    entry = make_entry(
        7,
        created_at=BASE + 7.5,
        used_llm=True,
        cleanup_reason="timeout",
        outcome="copied_focus_changed",
        language="de",
        app_process="Code.exe",
        app_title="main.py - Visual Studio Code",
    )
    rowid = store.add(entry)
    (got,) = store.recent()
    assert got.id == rowid
    assert dataclasses.replace(got, id=None) == entry
    assert isinstance(got.used_llm, bool)


@pytest.mark.parametrize(
    "field",
    ["raw_text", "app_title"],
)
def test_search_matches_each_field_case_insensitively(store, field):
    store.add(make_entry(1))
    hit = store.add(make_entry(2, **{field: "xx Needle-Value yy"}))
    assert [e.id for e in store.search("NEEDLE-VALUE")] == [hit]
    assert [e.id for e in store.search("needle-value")] == [hit]
    assert [e.id for e in store.search("dle-Val")] == [hit]


def test_search_folds_non_ascii_case(store):
    hit = store.add(make_entry(1, raw_text="Über den Wolken"))
    store.add(make_entry(2, raw_text="nothing here"))
    assert [e.id for e in store.search("über")] == [hit]
    assert [e.id for e in store.search("ÜBER DEN")] == [hit]


def test_search_treats_like_wildcards_literally(store):
    store.add(make_entry(1, raw_text="one hundred percent"))
    hit = store.add(make_entry(2, raw_text="100% sure"))
    assert [e.id for e in store.search("%")] == [hit]
    assert store.search("_") == []


def test_clear_removes_everything_and_store_stays_usable(store):
    for n in range(3):
        store.add(make_entry(n))
    store.clear()
    assert store.recent() == []
    assert store.recent_timings() == []
    assert store.add(make_entry(9)) > 0


def test_none_path_keeps_nothing():
    with HistoryStore(None) as s:
        assert s.add(make_entry(1)) == -1
        assert s.recent() == []
        assert s.search("raw") == []
        assert s.recent_timings() == []
        s.clear()
        s.prune()
        s.set_retention("7d")
        assert s.add(make_entry(2)) == -1


def test_off_retention_stops_storing_but_reads_keep_working(tmp_path):
    path = tmp_path / "history.db"
    with HistoryStore(path) as s:
        kept = s.add(make_entry(1))
    with HistoryStore(path, retention="off") as s:
        assert s.add(make_entry(2)) == -1
        assert [e.id for e in s.recent()] == [kept]
        assert [e.id for e in s.search("raw")] == [kept]
        assert len(s.recent_timings()) == 1
        s.prune()
        assert [e.id for e in s.recent()] == [kept]
    assert row_count(path) == 1


def test_clear_is_the_explicit_way_to_delete_under_off(tmp_path):
    path = tmp_path / "history.db"
    with HistoryStore(path) as s:
        s.add(make_entry(1))
        s.set_retention("off")
        s.clear()
        assert s.recent() == []
    assert row_count(path) == 0


@pytest.mark.parametrize("unset", [0, None])
def test_created_at_unset_is_filled_with_wall_clock_and_survives_7d(tmp_path, unset):
    before = time.time()
    with HistoryStore(tmp_path / "history.db", retention="7d") as s:
        rowid = s.add(make_entry(1, created_at=unset))
        (got,) = s.recent()
    assert got.id == rowid
    assert before <= got.created_at <= time.time()


@pytest.mark.parametrize(
    "read",
    [
        lambda s: s.recent(),
        lambda s: s.search("raw"),
    ],
    ids=["recent", "search"],
)
def test_reads_prune_rows_past_their_age(tmp_path, read):
    path = tmp_path / "history.db"
    with HistoryStore(path, retention="7d") as s:
        s.add(make_entry(1, created_at=time.time()))
        assert len(read(s)) == 1
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE entries SET created_at = ?", (time.time() - 8 * DAY,))
        assert read(s) == []
    assert row_count(path) == 0


def test_prune_by_count_keeps_newest_100(store):
    for n in range(105):
        store.add(make_entry(n))
    got = store.recent(limit=1000)
    assert len(got) == 100
    assert got[0].raw_text == "raw text 104"
    assert got[-1].raw_text == "raw text 5"


@pytest.mark.parametrize("retention,days", [("7d", 7)])
def test_prune_by_age(tmp_path, retention, days):
    now = time.time()
    with HistoryStore(tmp_path / "history.db", retention=retention) as s:
        old = s.add(make_entry(1, created_at=now - (days + 1) * DAY))
        inside = s.add(make_entry(2, created_at=now - (days - 1) * DAY))
        fresh = s.add(make_entry(3, created_at=now))
        ids = [e.id for e in s.recent()]
    assert old not in ids
    assert inside in ids
    assert fresh in ids


def test_prune_runs_at_open(tmp_path):
    path = tmp_path / "history.db"
    now = time.time()
    with HistoryStore(path, retention="30d") as s:
        for n in range(120):
            s.add(make_entry(n, created_at=now - n))
    assert row_count(path) == 120
    with HistoryStore(path, retention="100") as s:
        assert len(s.recent(limit=1000)) == 100


def test_recent_timings_newest_first_default_20(store):
    for n in range(25):
        store.add(make_entry(n, timings=StageTimings(mic_open_ms=float(n))))
    timings = store.recent_timings()
    assert len(timings) == 20
    assert all(isinstance(t, StageTimings) for t in timings)
    assert timings[0].mic_open_ms == 24.0
    assert timings[-1].mic_open_ms == 5.0
    assert len(store.recent_timings(limit=5)) == 5


def test_schema_version_and_wal(tmp_path):
    path = tmp_path / "history.db"
    with HistoryStore(path):
        pass
    with sqlite3.connect(path) as conn:
        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        assert version == ("4",)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_a_row_holds_a_file_name_and_never_the_audio_itself(tmp_path):
    with HistoryStore(tmp_path / "history.db") as store:
        store.add(make_entry(1), pcm16=silence(1.0), audio=AudioPolicy(keep=True))
    blob = (tmp_path / "history.db").read_bytes()
    assert b"RIFF" not in blob
    assert b"000001.wav" in blob


def test_timings_json_round_trip_including_extra(store):
    timings = StageTimings(
        press_to_pill_ms=12.5,
        mic_open_ms=None,
        release_to_transcript_ms=640.0,
        release_to_cleaned_ms=1200.25,
        delivery_ms=35.0,
        extra={"asr_queue_ms": 5.0, "guard_ms": 0.5},
    )
    store.add(make_entry(1, timings=timings))
    (got,) = store.recent()
    assert got.timings == timings
    assert got.timings.extra == {"asr_queue_ms": 5.0, "guard_ms": 0.5}
    with sqlite3.connect(store.path) as conn:
        raw = conn.execute("SELECT timings FROM entries").fetchone()[0]
    assert json.loads(raw)["extra"] == {"asr_queue_ms": 5.0, "guard_ms": 0.5}
    assert json.loads(raw)["mic_open_ms"] is None


def test_timings_json_tolerates_unknown_and_missing_keys(store):
    store.add(make_entry(1))
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE entries SET timings = ?",
            (json.dumps({"mic_open_ms": 3.0, "future_field": 1}),),
        )
    (got,) = store.recent()
    assert got.timings == StageTimings(mic_open_ms=3.0)


def test_concurrent_adds_from_threads(store):
    def worker(offset: int) -> None:
        for n in range(25):
            store.add(make_entry(offset + n))

    threads = [threading.Thread(target=worker, args=(i * 100,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.recent(limit=1000)) == 100



V1_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE entries (
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
    timings TEXT NOT NULL
);
CREATE INDEX entries_created_at ON entries (created_at DESC, id DESC);
"""

VERSION_QUERY = "SELECT value FROM meta WHERE key = 'schema_version'"


def write_v1_database(path: Path, rows: int = 3) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(V1_SCHEMA)
    conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '1')")
    for n in range(rows):
        conn.execute(
            "INSERT INTO entries (created_at, raw_text, cleaned_text, delivered_text,"
            " app_process, app_title, language, used_llm, cleanup_reason, outcome, timings)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                BASE + n,
                f"old raw {n}",
                f"old cleaned {n}",
                f"old delivered {n}",
                "slack.exe",
                "Slack",
                "de",
                n % 2,
                "ok",
                "pasted",
                json.dumps({"press_to_pill_ms": 11.0, "extra": {"audio_s": 4.0}}),
            ),
        )
    conn.commit()
    conn.close()


def test_an_old_database_migrates_in_place_without_losing_rows(tmp_path):
    path = tmp_path / "history.db"
    write_v1_database(path, rows=4)
    with HistoryStore(path) as store:
        entries = store.recent()
        assert [e.raw_text for e in entries] == [f"old raw {n}" for n in (3, 2, 1, 0)]
        assert entries[0].app_process == "slack.exe"
        assert entries[0].language == "de"
        assert entries[0].timings.press_to_pill_ms == 11.0
        assert entries[0].timings.extra == {"audio_s": 4.0}
    with sqlite3.connect(path) as conn:
        assert conn.execute(VERSION_QUERY).fetchone() == ("4",)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(entries)")}
        uploaded = conn.execute("SELECT DISTINCT uploaded_at FROM entries").fetchall()
    assert {"audio_file", "signals", "quality_label", "check_verdict", "uploaded_at"} <= columns
    assert uploaded == [(0.0,)]
    assert row_count(path) == 4


def test_a_database_without_a_version_row_is_treated_as_version_one(tmp_path):
    path = tmp_path / "history.db"
    conn = sqlite3.connect(path)
    conn.executescript(V1_SCHEMA)
    conn.commit()
    conn.close()
    with HistoryStore(path) as store:
        assert store.add(make_entry(1)) == 1
        assert len(store.recent()) == 1


def test_a_newer_database_is_read_as_it_is(tmp_path):
    path = tmp_path / "history.db"
    with HistoryStore(path) as store:
        store.add(make_entry(1))
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        conn.execute("ALTER TABLE entries ADD COLUMN future_column TEXT")
    with HistoryStore(path) as store:
        assert len(store.recent()) == 1
    with sqlite3.connect(path) as conn:
        assert conn.execute(VERSION_QUERY).fetchone() == ("99",)



def test_quality_fields_round_trip(store):
    signals = QualitySignals(
        avg_logprob=-0.42,
        compression_ratio=1.8,
        no_speech_prob=0.02,
        temperature=0.0,
        segments=3,
        source="whisper-server",
        audio_s=6.0,
        word_count=13,
        words_per_minute=130.0,
        filler_count=2,
        correction_count=1,
    )
    store.add(
        make_entry(
            1,
            signals=signals,
            quality_label="uncertain",
            quality_reason="Low recognition confidence.",
            asr_engine="whisper",
            cleanup_engine="llama",
        )
    )
    got = store.recent()[0]
    assert got.signals == signals
    assert got.quality_label == "uncertain"
    assert got.asr_engine == "whisper"
    assert got.cleanup_engine == "llama"


def test_set_check_stores_a_verdict(store):
    row_id = store.add(make_entry(1))
    assert store.set_check(row_id, "GARBLED", "the words do not fit", checked_at=BASE)
    got = store.recent()[0]
    assert got.check_verdict == "GARBLED"
    assert got.check_reason == "the words do not fit"
    assert got.checked_at == BASE


def test_row_stats_counts_a_cleanup_change_only_where_cleanup_ran():
    changed = make_entry(1, used_llm=True, raw_text="um hello", cleaned_text="Hello")
    same = make_entry(2, used_llm=True, raw_text="Hello", cleaned_text="Hello")
    skipped = make_entry(3, used_llm=False, raw_text="Hello", cleaned_text="Hello")
    assert row_stats(changed).cleanup_changed is True
    assert row_stats(same).cleanup_changed is False
    assert row_stats(skipped).cleanup_changed is None


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("ok", ("", "")),
        ("short_clean", ("short_clean", "")),
        ("length_ratio", ("", "length_ratio")),
        ("timeout", ("", "timeout")),
    ],
)
def test_split_cleanup_reason(reason, expected):
    assert split_cleanup_reason(make_entry(1, cleanup_reason=reason)) == expected



def test_no_recording_is_written_without_the_setting(tmp_path):
    with HistoryStore(tmp_path / "history.db") as store:
        store.add(make_entry(1), pcm16=silence(1.0))
        store.add(make_entry(2), pcm16=silence(1.0), audio=AudioPolicy(keep=False))
        assert not (tmp_path / "recordings").exists()
        assert store.recent()[0].audio_file == ""


def test_a_recording_is_written_and_named_after_the_row(tmp_path):
    with HistoryStore(tmp_path / "history.db") as store:
        row_id = store.add(make_entry(1), pcm16=silence(2.0), audio=AudioPolicy(keep=True))
        entry = store.recent()[0]
        assert entry.audio_file == f"{row_id:06d}.wav"
        path = store.recording_path(entry)
        assert path is not None and path.is_file()
        assert path.parent == tmp_path / "recordings"
        assert history_module.wav_duration_s(path) == pytest.approx(2.0)


def test_wav_duration_of_a_file_that_is_not_a_wav(tmp_path):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wav")
    assert history_module.wav_duration_s(broken) is None
    assert history_module.wav_duration_s(tmp_path / "missing.wav") is None


def test_deleting_a_row_deletes_its_recording(tmp_path):
    with HistoryStore(tmp_path / "history.db") as store:
        row_id = store.add(make_entry(1), pcm16=silence(0.5), audio=AudioPolicy(keep=True))
        path = store.recording_path(store.recent()[0])
        assert store.delete(row_id) is True
        assert store.recent() == []
        assert not path.exists()


def test_clearing_the_history_clears_the_folder(tmp_path):
    with HistoryStore(tmp_path / "history.db") as store:
        for n in range(3):
            store.add(make_entry(n), pcm16=silence(0.5), audio=AudioPolicy(keep=True))
        assert len(list((tmp_path / "recordings").iterdir())) == 3
        store.clear()
        assert list((tmp_path / "recordings").iterdir()) == []
        assert store.recent() == []


def test_retention_deletes_the_recordings_of_the_rows_it_drops(tmp_path):
    with HistoryStore(tmp_path / "history.db", "7d") as store:
        store.add(
            make_entry(1, created_at=time.time() - 30 * DAY),
            pcm16=silence(0.5),
            audio=AudioPolicy(keep=True),
        )
        recent = store.add(
            make_entry(2, created_at=time.time()), pcm16=silence(0.5), audio=AudioPolicy(keep=True)
        )
        names = sorted(p.name for p in (tmp_path / "recordings").iterdir())
        assert names == [f"{recent:06d}.wav"]


def test_the_count_limit_prunes_the_oldest_recordings_first(tmp_path):
    policy = AudioPolicy(keep=True, max_files=3, max_mb=1000)
    with HistoryStore(tmp_path / "history.db") as store:
        ids = [store.add(make_entry(n), pcm16=silence(0.2), audio=policy) for n in range(6)]
        names = sorted(p.name for p in (tmp_path / "recordings").iterdir())
        assert names == sorted(f"{i:06d}.wav" for i in ids[-3:])
        kept = [e.audio_file for e in store.recent()]
        assert kept[:3] == [f"{i:06d}.wav" for i in reversed(ids[-3:])]
        assert kept[3] == ""


def test_clear_recordings_keeps_the_rows(tmp_path):
    with HistoryStore(tmp_path / "history.db") as store:
        for n in range(3):
            store.add(make_entry(n), pcm16=silence(0.2), audio=AudioPolicy(keep=True))
        assert store.clear_recordings() == 3
        assert len(store.recent()) == 3
        assert all(entry.audio_file == "" for entry in store.recent())
        assert store.recordings_usage() == (0, 0)


def test_a_write_failure_never_fails_the_dictation(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    with HistoryStore(tmp_path / "history.db") as store:
        monkeypatch.setattr(history_module, "_write_wav", boom)
        row_id = store.add(make_entry(1), pcm16=silence(0.5), audio=AudioPolicy(keep=True))
        assert row_id == 1
        assert store.recent()[0].audio_file == ""



def rich_entry(n: int = 1) -> HistoryEntry:
    return make_entry(
        n,
        raw_text="um I went to the market",
        cleaned_text="I went to the market",
        delivered_text="I went to the market",
        language="en",
        used_llm=True,
        cleanup_reason="ok",
        outcome="pasted",
        asr_engine="whisper",
        cleanup_engine="llama",
        quality_label="good",
        quality_reason="Recognition confidence and speaking rate both look normal.",
        check_verdict="GOOD",
        check_reason="reads correctly",
        signals=QualitySignals(
            avg_logprob=-0.2,
            compression_ratio=1.4,
            no_speech_prob=0.01,
            temperature=0.0,
            segments=1,
            source="whisper-server",
            audio_s=3.0,
            word_count=5,
            words_per_minute=100.0,
            filler_count=1,
            correction_count=0,
        ),
    )


def test_json_export_holds_every_field(tmp_path):
    target = tmp_path / "out.json"
    assert export_to_path([rich_entry()], target, "json") == 1
    data = json.loads(target.read_text(encoding="utf-8"))
    assert len(data) == 1
    row = data[0]
    assert row["language"] == "en"
    assert row["app_process"] == "notepad.exe"
    assert row["app_title"] == "Untitled - Notepad"
    assert row["raw_text"].startswith("um I went")
    assert row["delivered_text"] == "I went to the market"
    assert row["outcome"] == "pasted"
    assert row["gate_reason"] == ""
    assert row["guard_rejection"] == ""
    assert row["asr_engine"] == "whisper"
    assert row["cleanup_engine"] == "llama"
    assert row["quality"]["label"] == "good"
    assert row["quality"]["avg_logprob"] == -0.2
    assert row["quality"]["words_per_minute"] == 100.0
    assert row["quality"]["filler_count"] == 1
    assert row["quality"]["check_verdict"] == "GOOD"
    assert row["timings"]["press_to_pill_ms"] == 1.0
    assert "time" in row


def test_json_export_separates_the_gate_reason_from_a_guard_rejection(tmp_path):
    target = tmp_path / "out.json"
    entries = [
        make_entry(1, cleanup_reason="short_clean"),
        make_entry(2, cleanup_reason="length_ratio"),
    ]
    export_to_path(entries, target, "json")
    rows = json.loads(target.read_text(encoding="utf-8"))
    assert rows[0]["gate_reason"] == "short_clean"
    assert rows[0]["guard_rejection"] == ""
    assert rows[1]["gate_reason"] == ""
    assert rows[1]["guard_rejection"] == "length_ratio"


def test_csv_export_has_a_header_and_one_row_each(tmp_path):
    target = tmp_path / "out.csv"
    assert export_to_path([rich_entry(1), rich_entry(2)], target, "csv") == 2
    with open(target, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["language"] == "en"
    assert rows[0]["quality_label"] == "good"
    assert rows[0]["words_per_minute"] == "100.0"
    assert rows[0]["cleanup_ran"] == "yes"
    assert rows[0]["app_title"] == "Untitled - Notepad"


def test_an_unknown_export_format_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown export format"):
        export_to_path([], tmp_path / "out.txt", "pdf")


class Sink:
    def __init__(self) -> None:
        self.writes = 0
        self.text = ""

    def write(self, chunk: str) -> int:
        self.writes += 1
        self.text += chunk
        return len(chunk)


def test_the_export_writers_stream_one_entry_at_a_time():
    def entries():
        for n in range(50):
            yield rich_entry(n)

    for writer in (write_json, write_csv, write_markdown):
        sink = Sink()
        assert writer(entries(), sink) == 50
        assert sink.writes >= 50



def compose_entry(n: int, **overrides) -> HistoryEntry:
    fields = {
        "raw_text": "write an email to Marta asking for the September invoice",
        "cleaned_text": "",
        "delivered_text": "Hi Marta, could you send me the September invoice?",
        "cleanup_reason": "compose",
        "mode": "compose",
        "instruction": "write an email to Marta asking for the September invoice",
        "selection_chars": 0,
    }
    fields.update(overrides)
    return make_entry(n, **fields)


def test_a_compose_row_keeps_its_mode_its_instruction_and_its_text(store):
    store.add(compose_entry(1))
    entry = store.recent()[0]
    assert entry.mode == "compose"
    assert entry.wrote is True
    assert entry.instruction.startswith("write an email")
    assert entry.delivered_text.startswith("Hi Marta")
    assert entry.cleaned_text == ""


def test_a_writing_row_that_delivered_nothing_still_names_why(store):
    store.add(compose_entry(3, delivered_text="", outcome="not_written", cleanup_reason="refusal"))
    entry = store.recent()[0]
    assert entry.outcome == "not_written"
    assert entry.delivered_text == ""
    assert split_cleanup_reason(entry) == ("", "refusal")


def test_the_export_carries_the_mode_the_instruction_and_the_selection_length(store):
    store.add(compose_entry(5, mode="edit", instruction="make this shorter", selection_chars=99))
    data = entry_to_dict(store.recent()[0])
    assert data["mode"] == "edit"
    assert data["instruction"] == "make this shorter"
    assert data["selection_chars"] == 99
    assert "mode" in CSV_COLUMNS and "instruction" in CSV_COLUMNS


def test_the_markdown_log_says_what_a_written_row_was_written_from(store):
    store.add(compose_entry(6))
    out = io.StringIO()
    write_markdown(store.recent(), out)
    text = out.getvalue()
    assert "Wrote from: write an email to Marta" in text
    assert "Hi Marta" in text


def test_a_version_one_database_reads_every_row_as_a_dictation(tmp_path):
    path = tmp_path / "history.db"
    write_v1_database(path, rows=2)
    with HistoryStore(path) as store:
        entries = store.recent()
        assert [entry.mode for entry in entries] == ["dictate", "dictate"]
        assert all(entry.instruction == "" for entry in entries)
        assert all(entry.wrote is False for entry in entries)
        store.add(compose_entry(8))
        assert store.recent()[0].mode == "compose"



def young_entry(n: int, **overrides) -> HistoryEntry:
    return make_entry(n, created_at=time.time() - 3600 + n, **overrides)


def uploaded_at(path: Path) -> dict[int, float]:
    with sqlite3.connect(path) as conn:
        return dict(conn.execute("SELECT id, uploaded_at FROM entries").fetchall())


def test_the_install_id_is_made_once_and_survives_a_clear_and_a_restart(tmp_path):
    path = tmp_path / "history.db"
    with HistoryStore(path) as store:
        first = store.install_id()
        assert len(first) == 36
        assert store.install_id() == first
        store.add(make_entry(1))
        store.clear()
        assert store.install_id() == first
    with HistoryStore(path) as store:
        assert store.install_id() == first
    with HistoryStore(tmp_path / "other.db") as other:
        assert other.install_id() != first


def test_new_rows_are_pending_oldest_first_until_marked(store):
    store.add(make_entry(3, created_at=BASE + 30))
    store.add(make_entry(1, created_at=BASE + 10))
    store.add(make_entry(2, created_at=BASE + 20))
    assert store.pending_upload_ids() == [2, 3, 1]
    assert store.pending_upload_count() == 3
    assert store.mark_uploaded([2, 3], BASE + 100) == 2
    assert store.pending_upload_ids() == [1]
    assert uploaded_at(store.path)[2] == BASE + 100


def test_a_check_that_finishes_after_the_upload_makes_the_row_pending_again(store):
    store.add(make_entry(1))
    store.add(make_entry(2))
    store.set_check(1, "GOOD", "fine", checked_at=BASE + 50)
    store.mark_uploaded([1, 2], BASE + 100)
    assert store.pending_upload_ids() == []
    store.set_check(2, "POOR", "garbled", checked_at=BASE + 200)
    assert store.pending_upload_ids() == [2]


def test_a_row_is_marked_only_when_its_check_is_still_the_one_that_was_sent(store):
    store.add(make_entry(1))
    store.add(make_entry(2))
    store.add(make_entry(3))
    store.set_check(3, "GOOD", "fine", checked_at=BASE + 10)
    sent = {1: 0.0, 2: 0.0, 3: BASE + 10}
    store.set_check(1, "POOR", "garbled", checked_at=BASE + 50)
    assert store.mark_uploaded([1, 2, 3], BASE + 100, checked_at=sent) == 2
    marks = uploaded_at(store.path)
    assert marks[1] == 0.0
    assert marks[2] == marks[3] == BASE + 100
    assert store.pending_upload_ids() == [1]
    store.set_check(2, "POOR", "garbled", checked_at=BASE + 150)
    assert store.mark_uploaded([2], BASE + 200, checked_at={2: BASE + 120}) == 0
    assert uploaded_at(store.path)[2] == BASE + 100
    assert store.pending_upload_ids() == [1, 2]


def test_rows_from_skipped_apps_are_marked_and_never_pending(store):
    store.add(make_entry(1, app_process="KeePassXC.exe"))
    store.add(make_entry(2, app_process="notepad.exe"))
    store.add(make_entry(3, app_process="keepassxc.exe"))
    assert store.mark_upload_skipped([" keepassxc.EXE ", ""]) == 2
    assert store.pending_upload_ids() == [2]
    store.mark_uploaded([1, 2, 3], BASE + 100)
    marks = uploaded_at(store.path)
    assert marks[1] == -1.0 and marks[3] == -1.0
    assert marks[2] == BASE + 100
    assert store.mark_upload_skipped([]) == 0


def test_a_new_address_resets_what_was_sent_but_not_what_was_skipped(store):
    for n in range(3):
        store.add(make_entry(n, app_process="secret.exe" if n == 2 else "notepad.exe"))
    store.mark_upload_skipped(["secret.exe"])
    store.mark_uploaded([1, 2], BASE + 100)
    assert store.pending_upload_ids() == []
    assert store.reset_uploaded() == 2
    assert store.pending_upload_ids() == [1, 2]
    assert uploaded_at(store.path)[3] == -1.0


def test_the_hold_keeps_unsent_rows_past_the_count_limit(tmp_path):
    with HistoryStore(tmp_path / "history.db", upload_hold=True) as store:
        for n in range(120):
            store.add(young_entry(n))
        assert row_count(store.path) == 120
        assert store.upload_lost() == 0
        store.mark_uploaded(range(1, 61), time.time())
        store.prune()
        assert row_count(store.path) == 100
        assert store.upload_lost() == 0


def test_the_hold_ends_after_thirty_days_and_counts_the_unsent_rows_it_loses(tmp_path):
    now = time.time()
    with HistoryStore(tmp_path / "history.db", upload_hold=True) as store:
        store.add(make_entry(1, created_at=now - 40 * DAY))
        store.add(make_entry(2, created_at=now - 35 * DAY))
        store.add(make_entry(3, created_at=now - 20 * DAY))
        store.add(make_entry(4, created_at=now - 10 * DAY))
        store.mark_uploaded([2], now - 34 * DAY)
        store.set_retention("7d")
        assert [entry.id for entry in store.recent()] == [4, 3]
        assert store.upload_lost() == 1
        store.settle_upload_lost(5)
        assert store.upload_lost() == 0


def test_the_hold_keeps_the_recordings_of_unsent_rows(tmp_path):
    policy = AudioPolicy(keep=True, max_files=1, max_mb=1000)
    with HistoryStore(tmp_path / "history.db", upload_hold=True, hold_recordings=True) as store:
        for n in range(3):
            store.add(young_entry(n), pcm16=silence(0.2), audio=policy)
        assert store.recordings_usage()[0] == 3
        store.mark_uploaded([1, 2, 3], time.time())
        store.prune_recordings(policy)
        assert store.recordings_usage()[0] == 1
        assert [bool(entry.audio_file) for entry in store.recent()] == [True, False, False]


def test_a_version_three_database_gains_uploaded_at_with_every_row_unsent(tmp_path):
    path = tmp_path / "history.db"
    write_v1_database(path, rows=2)
    with HistoryStore(path):
        pass
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
        conn.execute("ALTER TABLE entries DROP COLUMN uploaded_at")
    with HistoryStore(path) as store:
        assert store.pending_upload_ids() == [1, 2]
    with sqlite3.connect(path) as conn:
        assert conn.execute(VERSION_QUERY).fetchone() == ("4",)


def test_rows_from_skipped_apps_are_neither_held_nor_waiting_nor_lost(tmp_path):
    now = time.time()
    path = tmp_path / "history.db"
    with HistoryStore(path, "7d", upload_hold=True, upload_skip_apps=["KeePassXC.exe"]) as store:
        store.add(make_entry(1, created_at=now - 10 * DAY, app_process="keepassxc.exe"))
        store.add(make_entry(2, created_at=now - 10 * DAY))
        store.add(make_entry(3, created_at=now - 3600, app_process=" KEEPASSXC.exe "))
        store.add(make_entry(4, created_at=now - 3600))
        assert [entry.id for entry in store.recent()] == [4, 3, 2]
        assert store.pending_upload_ids() == [2, 4]
        assert store.pending_upload_count() == 2
        assert store.upload_lost() == 0
        assert uploaded_at(path)[3] == 0.0


def test_the_hold_does_not_keep_the_recordings_of_skipped_apps(tmp_path):
    policy = AudioPolicy(keep=True, max_files=1, max_mb=1000)
    with HistoryStore(
        tmp_path / "history.db",
        upload_hold=True,
        hold_recordings=True,
        upload_skip_apps=["secret.exe"],
    ) as store:
        store.add(young_entry(1, app_process="secret.exe"), pcm16=silence(0.2), audio=policy)
        store.add(young_entry(2, app_process="secret.exe"), pcm16=silence(0.2), audio=policy)
        store.add(young_entry(3), pcm16=silence(0.2), audio=policy)
        assert store.recordings_usage()[0] == 1
        assert [bool(entry.audio_file) for entry in store.recent()] == [True, False, False]


def test_reset_uploaded_recordings_marks_only_sent_rows_that_kept_a_recording(tmp_path):
    policy = AudioPolicy(keep=True, max_files=10, max_mb=100)
    with HistoryStore(tmp_path / "history.db", upload_hold=True) as store:
        store.add(young_entry(0), pcm16=silence(0.2), audio=policy)
        store.add(young_entry(1))
        store.add(young_entry(2), pcm16=silence(0.2), audio=policy)
        store.mark_uploaded([1, 2], time.time())
        assert store.pending_upload_ids() == [3]
        assert store.reset_uploaded_recordings() == 1
        assert store.pending_upload_ids() == [1, 3]
