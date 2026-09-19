"""The History page: the export, the quality badge, the speaking stats, the recordings
and the on-demand check (spec 8.4, 14.4, 15, 17).

Nothing here opens an audio device or an engine: the player is a stand-in with the same
signals, the check runs through a fake pipeline, and the batch executor is inline so a
submitted batch is finished by the time submit() returns.
"""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6 import QtCore

from spells import quality
from spells.history import AudioPolicy
from spells.quality import CheckResult, QualitySignals
from spells.ui.checks import CheckJob, CheckRunner
from spells.ui.recordings import (
    PLAY_GLYPH,
    RECORDINGS_NOTE,
    STOP_GLYPH,
    PlayCell,
    WavPlayer,
    format_duration,
)
from spells.ui.settings import (
    CHECK_BATCH,
    EMPTY_STAT,
    QUALITY_COLUMN,
    SOUND_COLUMN,
    HistoryTab,
    check_line,
    quality_badge,
    speaking_stats,
    usage_text,
)

from .test_ui_support import (
    FakeHistory,
    FakePipeline,
    Messages,
    history_entry,
    make_config,
    qt_app,
)


@pytest.fixture(scope="module")
def app():
    return qt_app()


class FakePlayer(QtCore.QObject):
    """The WavPlayer slice the page uses, with the same three signals."""

    started = QtCore.Signal(object)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.played: list[Path] = []
        self.stops = 0
        self.result = True
        self._playing = False

    @property
    def playing(self) -> bool:
        return self._playing

    def play(self, path: Path) -> bool:
        self.played.append(path)
        self._playing = bool(self.result)
        return self.result

    def stop(self) -> None:
        self.stops += 1
        self._playing = False

    def finish(self) -> None:
        self._playing = False
        self.finished.emit(None)


def inline(work) -> None:
    work()


def write_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(2 * int(rate * seconds)))
    return path


def make_page(tmp_path, *, entries=None, history=None, pipeline=None, target=None, player=None):
    config = make_config(tmp_path)
    messages = Messages()
    store = history if history is not None else FakeHistory(entries or [])
    pipe = pipeline if pipeline is not None else FakePipeline()
    asked: list[tuple[Path, str]] = []
    revealed: list[Path] = []

    def export_dialog(default: Path, fmt: str):
        asked.append((default, fmt))
        if callable(target):
            return target(default, fmt)
        return target

    page = HistoryTab(
        config=config,
        history=store,
        notify=messages,
        confirm=lambda _title, _text: True,
        pipeline=pipe,
        export_dialog=export_dialog,
        player=player if player is not None else FakePlayer(),
        reveal=revealed.append,
        executor=inline,
    )
    page.refresh()
    return SimpleNamespace(
        page=page,
        config=config,
        messages=messages,
        history=store,
        pipeline=pipe,
        asked=asked,
        revealed=revealed,
    )


def with_recording(tmp_path, entry, name: str = "000001.wav", seconds: float = 2.0):
    store = FakeHistory([entry])
    entry.audio_file = name
    folder = tmp_path / "recordings"
    folder.mkdir(exist_ok=True)
    store.recordings[name] = write_wav(folder / name, seconds)
    return store


# The list, the badge and the summary


def test_the_table_has_a_quality_and_a_sound_column(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)])
    headers = [ui.page.table.horizontalHeaderItem(i).text() for i in range(ui.page.table.columnCount())]
    assert headers == ["When", "App", "Language", "Quality", "Sound", "Text"]


def test_each_row_shows_its_label_as_a_badge_with_the_reason_on_hover(app, tmp_path):
    entry = history_entry(1, quality_label="uncertain", quality_reason="Low recognition confidence.")
    ui = make_page(tmp_path, entries=[entry])
    badge = ui.page.table.cellWidget(0, QUALITY_COLUMN)
    assert badge is not None
    assert badge.text() == "Uncertain"
    assert badge.property("badge") == "caution"
    assert badge.toolTip() == "Low recognition confidence."


def test_a_row_without_a_label_shows_no_badge(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1, quality_label="", quality_reason="")])
    assert ui.page.table.cellWidget(0, QUALITY_COLUMN) is None


@pytest.mark.parametrize(
    "label,expected",
    [("good", ("Good", "ok")), ("uncertain", ("Uncertain", "caution")), ("poor", ("Poor", "critical")), ("", ("", "muted"))],
)
def test_quality_badge_mapping(label, expected):
    assert quality_badge(label) == expected


def test_the_summary_is_computed_over_the_rows_on_screen(app, tmp_path):
    fast = history_entry(1, signals=QualitySignals(words_per_minute=180.0, word_count=100, filler_count=2))
    slow = history_entry(2, signals=QualitySignals(words_per_minute=100.0, word_count=100, filler_count=0))
    ui = make_page(tmp_path, entries=[fast, slow])
    assert ui.page.rate_stat.value.text() == "140"
    assert ui.page.filler_stat.value.text() == "1.0"


def test_the_summary_counts_how_often_cleanup_changed_the_text(app, tmp_path):
    changed = history_entry(1, raw_text="um hello", cleaned_text="Hello.", used_llm=True)
    same = history_entry(2, raw_text="Hello.", cleaned_text="Hello.", used_llm=True)
    ui = make_page(tmp_path, entries=[changed, same])
    assert ui.page.cleanup_stat.value.text() == "50%"


def test_the_summary_says_so_when_there_is_nothing_to_show(app, tmp_path):
    ui = make_page(tmp_path)
    assert ui.page.rate_stat.value.text() == EMPTY_STAT
    assert ui.page.cleanup_stat.value.text() == EMPTY_STAT


def test_the_summary_follows_the_search(app, tmp_path):
    fast = history_entry(1, raw="alpha", signals=QualitySignals(words_per_minute=300.0, word_count=10))
    slow = history_entry(2, raw="beta", signals=QualitySignals(words_per_minute=100.0, word_count=10))
    ui = make_page(tmp_path, entries=[fast, slow])
    ui.page.search.setText("beta")
    ui.page.run_search()
    assert ui.page.table.rowCount() == 1
    assert ui.page.rate_stat.value.text() == "100"


# The speaking stats of one row


def test_the_selected_row_shows_how_it_was_spoken(app, tmp_path):
    entry = history_entry(
        1,
        signals=QualitySignals(audio_s=6.0, word_count=12, words_per_minute=120.0, filler_count=2, correction_count=1),
        quality_reason="Recognition confidence and speaking rate both look normal.",
    )
    ui = make_page(tmp_path, entries=[entry])
    ui.page.table.selectRow(0)
    text = ui.page.stats_line.text()
    assert "6.0 s" in text
    assert "120 words per minute" in text
    assert "2 fillers" in text
    assert "1 self-correction" in text
    assert text.endswith("look normal.")


def test_one_filler_is_singular():
    entry = history_entry(1, signals=QualitySignals(filler_count=1), quality_reason="")
    assert speaking_stats(entry).endswith("1 filler")


def test_dropped_audio_is_named_in_the_stats():
    entry = history_entry(1, signals=QualitySignals(filler_count=0, dropped_blocks=3), quality_reason="")
    assert "some audio was dropped" in speaking_stats(entry)


def test_the_stats_line_is_empty_without_a_selection(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)])
    ui.page.table.clearSelection()
    ui.page._show_selected()
    assert ui.page.stats_line.text() == ""


# Export


def test_export_writes_json_and_says_where_it_went(app, tmp_path):
    target = tmp_path / "out.json"
    ui = make_page(tmp_path, entries=[history_entry(1), history_entry(2)], target=target)
    assert ui.page.export("json") == target
    rows = json.loads(target.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["quality"]["label"] == "good"
    assert str(target) in ui.messages.texts[-1]
    assert "Saved 2 dictations" in ui.messages.texts[-1]


def test_export_offers_the_three_formats(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)])
    actions = [action.data() for action in ui.page.export_menu.actions()]
    assert actions == ["json", "csv", "markdown"]


def test_export_suggests_a_plain_file_name(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)], target=None)
    ui.page.export("json")
    default, fmt = ui.asked[-1]
    assert default.name.startswith("spells-history-")
    assert default.name.endswith(".json")
    assert fmt == "json"


def test_export_writes_csv(app, tmp_path):
    target = tmp_path / "out.csv"
    ui = make_page(tmp_path, entries=[history_entry(1)], target=target)
    ui.page.export("csv")
    with open(target, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["app_title"] == "Untitled"
    assert rows[0]["quality_label"] == "good"


def test_export_writes_a_markdown_log(app, tmp_path):
    target = tmp_path / "out.md"
    ui = make_page(tmp_path, entries=[history_entry(1, cleaned="Hello there.")], target=target)
    ui.page.export("markdown")
    text = target.read_text(encoding="utf-8")
    assert "# Spells transcript log" in text
    assert "Hello there." in text


def test_export_exports_what_the_filter_shows(app, tmp_path):
    target = tmp_path / "out.json"
    ui = make_page(tmp_path, entries=[history_entry(1, raw="alpha"), history_entry(2, raw="beta")], target=target)
    ui.page.search.setText("beta")
    ui.page.run_search()
    assert ui.page.export("json") == target
    rows = json.loads(target.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["raw_text"] == "beta"


def test_a_cancelled_dialog_writes_nothing(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)], target=None)
    assert ui.page.export("json") is None
    assert ui.messages.shown == []


def test_a_failed_export_is_reported_and_not_raised(app, tmp_path):
    target = tmp_path / "missing" / "out.json"
    ui = make_page(tmp_path, entries=[history_entry(1)], target=target)
    assert ui.page.export("json") is None
    assert "Could not write" in ui.messages.texts[-1]


# Recordings


def test_a_row_with_a_recording_gets_a_play_control(app, tmp_path):
    entry = history_entry(1)
    store = with_recording(tmp_path, entry)
    ui = make_page(tmp_path, history=store)
    cell = ui.page.table.cellWidget(0, SOUND_COLUMN)
    assert isinstance(cell, PlayCell)
    assert cell.length.text() == "0:02"


def test_a_row_without_a_recording_gets_none(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)])
    assert ui.page.table.cellWidget(0, SOUND_COLUMN) is None


def test_playing_and_stopping_a_row(app, tmp_path):
    entry = history_entry(1)
    store = with_recording(tmp_path, entry)
    player = FakePlayer()
    ui = make_page(tmp_path, history=store, player=player)
    cell = ui.page.table.cellWidget(0, SOUND_COLUMN)
    cell.button.click()
    assert player.played == [store.recordings["000001.wav"]]
    assert cell.button.text() in (STOP_GLYPH, "x")
    cell.button.click()
    assert player.stops >= 1
    assert cell.button.text() in (PLAY_GLYPH, "x")


def test_playback_ending_resets_the_button(app, tmp_path):
    entry = history_entry(1)
    store = with_recording(tmp_path, entry)
    player = FakePlayer()
    ui = make_page(tmp_path, history=store, player=player)
    cell = ui.page.table.cellWidget(0, SOUND_COLUMN)
    cell.button.click()
    player.finish()
    assert cell.button.toolTip() == "Play this recording"


def test_a_missing_file_is_reported_instead_of_played(app, tmp_path):
    entry = history_entry(1)
    store = with_recording(tmp_path, entry)
    store.recordings["000001.wav"].unlink()
    ui = make_page(tmp_path, history=store)
    ui.page._toggle_play(0)
    assert ui.page.table.cellWidget(0, SOUND_COLUMN) is None


def test_show_in_folder_reveals_the_file(app, tmp_path):
    entry = history_entry(1)
    store = with_recording(tmp_path, entry)
    ui = make_page(tmp_path, history=store)
    ui.page.table.selectRow(0)
    assert ui.page.reveal_button.isEnabled()
    ui.page.reveal_button.click()
    assert ui.revealed == [store.recordings["000001.wav"]]


def test_show_in_folder_is_off_without_a_recording(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)])
    ui.page.table.selectRow(0)
    assert ui.page.reveal_button.isEnabled() is False


def test_keep_recordings_is_off_by_default_and_the_limits_follow(app, tmp_path):
    ui = make_page(tmp_path)
    assert ui.page.keep_audio.isChecked() is False
    assert ui.page.keep_count.isEnabled() is False
    ui.page.keep_audio.setChecked(True)
    assert ui.config.settings.history.keep_audio is True
    assert ui.page.keep_count.isEnabled() is True
    assert ui.page.keep_mb.isEnabled() is True


def test_the_limits_reach_the_settings_and_prune(app, tmp_path):
    ui = make_page(tmp_path)
    ui.page.keep_audio.setChecked(True)
    ui.page.keep_count.setValue(25)
    ui.page.keep_mb.setValue(64)
    assert ui.config.settings.history.audio_keep_count == 25
    assert ui.config.settings.history.audio_keep_mb == 64
    pruned = [call for call in ui.history.calls if call[0] == "prune_recordings"]
    assert pruned and pruned[-1][1] == AudioPolicy(True, 25, 64)


def test_delete_all_recordings_keeps_the_dictations(app, tmp_path):
    entry = history_entry(1)
    store = with_recording(tmp_path, entry)
    ui = make_page(tmp_path, history=store)
    assert ui.page.delete_audio_button.isEnabled()
    ui.page.delete_audio_button.click()
    assert ("clear_recordings",) in store.calls
    assert len(store.entries) == 1
    assert store.recordings_usage() == (0, 0)


def test_the_page_says_recordings_stay_on_this_computer(app, tmp_path):
    ui = make_page(tmp_path)
    labels = [w.text() for w in ui.page.findChildren(type(ui.page.stats_line)) if w.text()]
    assert any(RECORDINGS_NOTE == text for text in labels)


def test_usage_text():
    assert usage_text(0, 0) == "No recordings saved yet."
    assert usage_text(1, 1024 * 1024) == "1 recording, 1 MB."
    assert usage_text(12, 5 * 1024 * 1024) == "12 recordings, 5 MB."


def test_deleting_a_row_goes_through_the_store(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1), history_entry(2)])
    ui.page.table.selectRow(0)
    ui.page.delete_button.click()
    assert ("delete", 1) in ui.history.calls
    assert ui.page.table.rowCount() == 1


# The on-demand check


def test_checking_one_row_stores_the_verdict(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1, raw="krbl mmm what")])
    ui.pipeline.check_result = CheckResult("GARBLED", "the words do not fit")
    ui.page.table.selectRow(0)
    assert ui.page.check_selected() is True
    assert ui.pipeline.checks == [("krbl mmm what", "en")]
    assert ("set_check", 1, "GARBLED", "the words do not fit") in ui.history.calls
    assert "Looks garbled" in ui.page.check_line.text()


def test_checking_the_last_twenty_stops_at_twenty(app, tmp_path):
    entries = [history_entry(n) for n in range(30)]
    ui = make_page(tmp_path, entries=entries)
    assert ui.page.check_recent() is True
    assert len(ui.pipeline.checks) == CHECK_BATCH


def test_a_row_with_no_transcript_is_not_checked(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1, raw="   ")])
    ui.page.table.selectRow(0)
    assert ui.page.check_selected() is False
    assert ui.pipeline.checks == []


def test_a_check_that_cannot_run_says_so_politely(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)])
    ui.pipeline.check_result = CheckResult("", "", "unavailable")
    ui.page.table.selectRow(0)
    ui.page.check_selected()
    assert ui.history.calls.count(("set_check", 1, "", "")) == 0
    assert "not serving" in ui.messages.texts[-1]


def test_an_answer_in_the_wrong_shape_is_not_stored(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)])
    ui.pipeline.check_result = CheckResult("", "", "unreadable")
    ui.page.table.selectRow(0)
    ui.page.check_selected()
    assert not any(call[0] == "set_check" for call in ui.history.calls)


def test_a_batch_stops_when_the_engine_goes_away(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(n) for n in range(5)])
    ui.pipeline.check_result = CheckResult("", "", "unavailable")
    ui.page.check_recent()
    assert len(ui.pipeline.checks) == 1


def test_the_check_never_blocks_the_page(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)])
    ui.page.table.selectRow(0)
    ui.page.check_selected()
    assert ui.page.checker.busy is False
    assert ui.page.check_recent_button.isEnabled()


def test_check_line_reads_plainly():
    entry = history_entry(1, check_verdict="GOOD", check_reason="reads correctly", checked_at=1_700_000_500.0)
    line = check_line(entry)
    assert "Reads correctly" in line
    assert "reads correctly" in line
    assert check_line(history_entry(2)) == ""


# WavPlayer and its helpers


class FakeStream:
    def __init__(self, callback, sample_rate: int) -> None:
        self.callback = callback
        self.sample_rate = sample_rate
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1

    def pump(self, frames: int) -> bytes:
        buffer = bytearray(frames * 2)
        self.callback(memoryview(buffer), frames)
        return bytes(buffer)


def make_player(app):
    streams: list[FakeStream] = []

    def factory(callback, sample_rate):
        stream = FakeStream(callback, sample_rate)
        streams.append(stream)
        return stream

    return WavPlayer(factory), streams


def test_the_player_reads_the_wav_and_opens_a_stream(app, tmp_path):
    player, streams = make_player(app)
    path = write_wav(tmp_path / "one.wav", 0.5)
    assert player.play(path) is True
    assert player.playing is True
    assert streams[0].sample_rate == 16000
    assert streams[0].started == 1
    player.stop()


def test_the_player_feeds_the_frames_and_then_silence(app, tmp_path):
    player, streams = make_player(app)
    player.play(write_wav(tmp_path / "one.wav", 0.01))
    frames = int(16000 * 0.01)
    assert len(streams[0].pump(frames)) == frames * 2
    streams[0].pump(frames)
    assert player.playing is False
    player.stop()


def test_stopping_closes_the_stream(app, tmp_path):
    player, streams = make_player(app)
    player.play(write_wav(tmp_path / "one.wav", 0.05))
    player.stop()
    assert streams[0].stopped == 1
    assert streams[0].closed == 1
    assert player.playing is False
    assert player.path is None


def test_playing_another_file_replaces_the_first(app, tmp_path):
    player, streams = make_player(app)
    player.play(write_wav(tmp_path / "one.wav", 0.05))
    player.play(write_wav(tmp_path / "two.wav", 0.05))
    assert len(streams) == 2
    assert streams[0].stopped == 1
    player.stop()


def test_a_file_that_is_not_a_wav_is_reported(app, tmp_path):
    player, streams = make_player(app)
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wav")
    failures: list[str] = []
    player.failed.connect(failures.append)
    assert player.play(broken) is False
    assert streams == []
    assert failures


def test_a_stream_that_will_not_open_is_reported(app, tmp_path):
    def factory(_callback, _rate):
        raise RuntimeError("no output device")

    player = WavPlayer(factory)
    failures: list[str] = []
    player.failed.connect(failures.append)
    assert player.play(write_wav(tmp_path / "one.wav", 0.05)) is False
    assert player.playing is False
    assert "no output device" in failures[0]


def test_the_poll_timer_reports_the_end_on_the_qt_thread(app, tmp_path):
    player, streams = make_player(app)
    ended: list = []
    player.finished.connect(lambda path: ended.append(path))
    player.play(write_wav(tmp_path / "one.wav", 0.01))
    streams[0].pump(int(16000 * 0.02))
    player._poll()
    assert ended
    assert player.playing is False


@pytest.mark.parametrize(
    "seconds,expected", [(0.0, "0:00"), (2.4, "0:02"), (65.0, "1:05"), (None, ""), (-1.0, "")]
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_play_cell_swaps_its_glyph(app):
    cell = PlayCell(0, 3.0)
    assert cell.length.text() == "0:03"
    cell.set_playing(True)
    assert cell.button.toolTip() == "Stop"
    cell.set_playing(False)
    assert cell.button.toolTip() == "Play this recording"


# CheckRunner


def test_the_runner_walks_the_batch_in_order(app):
    seen: list[tuple[int, object]] = []
    runner = CheckRunner(lambda text, language: CheckResult("GOOD", text), inline)
    runner.checked.connect(lambda entry_id, result: seen.append((entry_id, result.reason)))
    assert runner.submit([CheckJob(1, "one", "en"), CheckJob(2, "two", "de")]) is True
    assert seen == [(1, "one"), (2, "two")]


def test_the_runner_refuses_an_empty_batch(app):
    runner = CheckRunner(lambda text, language: CheckResult("GOOD", ""), inline)
    assert runner.submit([]) is False


def test_the_runner_reports_what_it_stored_and_skipped(app):
    results = iter([CheckResult("GOOD", "fine"), CheckResult("", "", "unreadable")])
    runner = CheckRunner(lambda text, language: next(results), inline)
    done: list[tuple[int, int]] = []
    runner.done.connect(lambda stored, skipped: done.append((stored, skipped)))
    runner.submit([CheckJob(1, "one"), CheckJob(2, "two")])
    assert done == [(1, 1)]


def test_the_runner_survives_a_raising_check(app):
    def boom(_text, _language):
        raise RuntimeError("engine gone")

    runner = CheckRunner(boom, inline)
    failures: list[str] = []
    runner.failed.connect(failures.append)
    runner.submit([CheckJob(1, "one")])
    assert failures == ["engine gone"]
    assert runner.busy is False


def test_a_cancelled_runner_stops_at_the_next_row(app):
    runner = CheckRunner(lambda text, language: CheckResult("GOOD", ""), lambda work: None)
    runner.submit([CheckJob(1, "one"), CheckJob(2, "two")])
    runner.cancel()
    assert runner.busy is True


def test_the_prompt_the_page_would_send_never_asks_for_a_rewrite():
    system, _user = quality.check_messages("hello", "English")
    assert "never rewrite" in system


# Written rows read as written (spec 8.5) ------------------------------------------------------


def written_entry(index: int = 1, **overrides):
    fields = {
        "raw": "write an email to Marta asking for the September invoice",
        "cleaned": "",
        "delivered_text": "Hi Marta, could you send me the September invoice?",
        "mode": "compose",
        "instruction": "write an email to Marta asking for the September invoice",
        "cleanup_reason": "compose",
    }
    fields.update(overrides)
    return history_entry(index, **fields)


def test_a_written_row_shows_the_text_the_model_wrote(app, tmp_path):
    ui = make_page(tmp_path, entries=[written_entry()])
    ui.page.table.selectRow(0)
    assert ui.page.table.item(0, 5).text().startswith("Hi Marta")


def test_a_written_row_labels_both_panels_for_writing(app, tmp_path):
    ui = make_page(tmp_path, entries=[written_entry()])
    ui.page.table.selectRow(0)
    assert ui.page.raw_panel.title_label.text() == "What you asked for"
    assert ui.page.cleaned_panel.title_label.text() == "What was written"
    assert ui.page.raw.toPlainText().startswith("write an email")
    assert ui.page.cleaned.toPlainText().startswith("Hi Marta")


def test_a_dictated_row_keeps_the_dictation_labels(app, tmp_path):
    ui = make_page(tmp_path, entries=[history_entry(1)])
    ui.page.table.selectRow(0)
    assert ui.page.raw_panel.title_label.text() == "What you said"
    assert ui.page.cleaned_panel.title_label.text() == "Cleaned text"
    assert ui.page.cleaned.toPlainText() == "Cleaned text."


def test_selecting_a_written_row_after_a_dictated_one_relabels_the_panels(app, tmp_path):
    ui = make_page(tmp_path, entries=[written_entry(1), history_entry(2)])
    ui.page.table.selectRow(1)
    assert ui.page.raw_panel.title_label.text() == "What you said"
    ui.page.table.selectRow(0)
    assert ui.page.raw_panel.title_label.text() == "What you asked for"
