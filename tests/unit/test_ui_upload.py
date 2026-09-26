from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest
from PySide6 import QtWidgets

from spells import upload
from spells.config import ConfigStore, UploadSettings
from spells.history import AudioPolicy, HistoryStore
from spells.ui.uploading import (
    PAUSED_TEXT,
    TEST_OK_TEXT,
    UploadCoordinator,
    UploadView,
    capture,
    direct_runner,
    status_text,
)
from spells.ui.uploadpage import (
    HOLD_NOTE,
    KEEP_AUDIO_QUESTION,
    UNENCRYPTED_HINT,
    URL_HINT,
    UploadPage,
    parse_apps,
)
from spells.upload import Response
from spells.uploadtoken import TOKEN_FILE, TokenStore

from .test_ui_support import Messages, flush, history_entry, qt_app
from .test_upload import FakeTransport, make_entry

URL = "https://example.com/spells"
TOKEN = "a-very-secret-token-0123456789abcdef"


@pytest.fixture(scope="module")
def app():
    return qt_app()


class Clock:
    def __init__(self, now: float | None = None) -> None:
        self.now = time.time() if now is None else now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ThreadRunner:
    def __init__(self) -> None:
        self.jobs: list[tuple[threading.Thread, object, dict]] = []

    def __call__(self, work, done) -> None:
        box: dict = {}
        thread = threading.Thread(target=lambda: box.update(result=capture(work)), daemon=True)
        thread.start()
        self.jobs.append((thread, done, box))

    def finish(self) -> None:
        while self.jobs:
            thread, done, box = self.jobs.pop(0)
            thread.join(10)
            done(box["result"])


class Gate:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, _payload):
        self.entered.set()
        self.release.wait(10)
        return 200


class World:
    def __init__(self, tmp_path, runner=direct_runner, **settings) -> None:
        self.config = ConfigStore(tmp_path / "settings.json")
        if settings:
            self.config.update(lambda s: replace(s, upload=replace(s.upload, **settings)))
        self.history = HistoryStore(tmp_path / "history.db")
        self.tokens = TokenStore(tmp_path / TOKEN_FILE)
        self.transport = FakeTransport()
        self.clock = Clock()
        self.dictating = False
        self.coordinator = UploadCoordinator(
            config=self.config,
            history=self.history,
            tokens=self.tokens,
            busy=lambda: self.dictating,
            transport=self.transport,
            runner=runner,
            clock=self.clock,
            version="0.6.0",
        )

    @property
    def upload(self) -> UploadSettings:
        return self.config.settings.upload

    def change(self, **fields) -> None:
        self.config.update(lambda s: replace(s, upload=replace(s.upload, **fields)))
        self.coordinator.apply_settings(self.config.settings)

    def add(self, count: int = 1) -> None:
        for n in range(count):
            self.history.add(make_entry(n))


@pytest.fixture
def world(tmp_path):
    made = World(tmp_path, enabled=True, url=URL)
    yield made
    made.history.close()


# the coordinator


def test_the_tick_does_nothing_while_uploading_is_off(app, tmp_path):
    world = World(tmp_path, url=URL)
    world.add()
    world.coordinator.tick()
    assert world.transport.sent == []
    assert world.coordinator.view.waiting == 0
    world.history.close()


def test_the_first_tick_uploads_and_marks_what_it_sent(world):
    world.add(3)
    world.coordinator.tick()
    assert len(world.transport.sent) == 1
    assert world.upload.last_success == world.clock.now
    assert world.upload.last_attempt == world.clock.now
    assert world.upload.last_sent == 3
    assert world.upload.last_error == ""
    assert world.history.pending_upload_ids() == []
    assert world.coordinator.view.working == ""


def test_the_tick_waits_while_a_dictation_is_in_flight(world):
    world.dictating = True
    world.coordinator.tick()
    assert world.transport.sent == []
    world.dictating = False
    world.coordinator.tick()
    assert len(world.transport.sent) == 1


def test_a_daily_upload_runs_once_a_day_and_not_sooner(world):
    world.coordinator.tick()
    world.clock.advance(upload.DAY_S - 60)
    world.coordinator.tick()
    assert len(world.transport.sent) == 1
    world.clock.advance(120)
    world.coordinator.tick()
    assert len(world.transport.sent) == 2


def test_a_weekly_upload_waits_a_week(world):
    world.change(schedule="weekly")
    world.coordinator.tick()
    world.clock.advance(6 * upload.DAY_S)
    world.coordinator.tick()
    assert len(world.transport.sent) == 1
    world.clock.advance(upload.DAY_S)
    world.coordinator.tick()
    assert len(world.transport.sent) == 2


def test_manual_never_runs_on_the_tick_but_upload_now_does(world):
    world.change(schedule="manual")
    world.add()
    world.coordinator.tick()
    assert world.transport.sent == []
    world.coordinator.upload_now()
    assert world.upload.last_sent == 1


def test_upload_now_asks_for_the_switch_first(app, tmp_path):
    world = World(tmp_path, url=URL)
    world.coordinator.upload_now()
    assert world.transport.sent == []
    assert world.coordinator.view.message
    world.history.close()


def test_a_failure_is_retried_on_the_next_tick(world):
    world.add()
    world.transport.answers = [503]
    world.coordinator.tick()
    assert "503" in world.upload.last_error
    assert world.upload.last_success == 0.0
    world.clock.advance(3600)
    world.coordinator.tick()
    assert world.upload.last_error == ""
    assert world.upload.last_sent == 1


def test_a_refused_token_pauses_the_tick_until_the_token_changes(world):
    world.add()
    world.transport.answers = [401]
    world.coordinator.tick()
    assert world.upload.last_error == upload.TOKEN_REFUSED
    world.clock.advance(3 * upload.DAY_S)
    world.coordinator.tick()
    assert len(world.transport.sent) == 1
    world.coordinator.set_token(TOKEN)
    assert world.upload.last_error == ""
    world.coordinator.tick()
    assert len(world.transport.sent) == 2
    assert world.transport.sent[1].headers["Authorization"] == f"Bearer {TOKEN}"


def test_upload_now_runs_even_while_paused(world):
    world.transport.answers = [403]
    world.coordinator.tick()
    world.coordinator.upload_now()
    assert len(world.transport.sent) == 2
    assert world.upload.last_error == ""


def test_a_new_address_resends_everything_and_unpauses(world):
    world.add(2)
    world.coordinator.tick()
    world.transport.answers = [401]
    world.clock.advance(upload.DAY_S)
    world.coordinator.tick()
    assert world.upload.last_error == upload.TOKEN_REFUSED
    world.change(url="https://example.org/other")
    assert world.upload.last_error == ""
    assert world.upload.last_success == 0.0
    assert world.history.pending_upload_ids() == [1, 2]
    world.coordinator.tick()
    assert world.transport.sent[-1].url == "https://example.org/other"
    assert world.transport.ids(len(world.transport.sent) - 1) == [1, 2]


OTHER_URL = "https://example.org/other"


def blocked_run(tmp_path, count: int = 250):
    runner = ThreadRunner()
    world = World(tmp_path, runner=runner, enabled=True, url=URL)
    world.add(count)
    gate = Gate()
    world.transport.answers = [gate]
    world.coordinator.upload_now()
    assert gate.entered.wait(10)
    return world, runner, gate


def test_a_new_address_during_a_run_stops_it_and_leaves_everything_for_the_new_one(
    app, tmp_path
):
    world, runner, gate = blocked_run(tmp_path)
    world.change(url=OTHER_URL)
    gate.release.set()
    runner.finish()
    assert [sent.url for sent in world.transport.sent] == [URL]
    assert world.upload.last_success == 0.0
    assert world.upload.last_sent == 0
    assert world.upload.last_error == ""
    assert len(world.history.pending_upload_ids()) == 250
    assert world.coordinator.view.working == ""
    assert world.coordinator.view.waiting == 250
    world.coordinator.tick()
    runner.finish()
    later = world.transport.sent[1:]
    assert {sent.url for sent in later} == {OTHER_URL}
    assert sum(len(sent.payload["entries"]) for sent in later) == 250
    assert world.upload.last_sent == 250
    assert world.history.pending_upload_ids() == []
    world.history.close()


def test_switching_uploading_off_during_a_run_stops_it_before_the_next_batch(app, tmp_path):
    world, runner, gate = blocked_run(tmp_path)
    world.change(enabled=False)
    gate.release.set()
    runner.finish()
    assert [sent.url for sent in world.transport.sent] == [URL]
    assert world.upload.last_success == 0.0
    assert world.coordinator.view.working == ""
    world.history.close()


def test_the_history_holds_unsent_rows_only_while_uploading_is_on(world):
    assert world.history.upload_hold is True
    world.change(enabled=False)
    assert world.history.upload_hold is False
    world.change(enabled=True)
    assert world.history.upload_hold is True


def add_with_recording(world, n: int = 0) -> None:
    policy = AudioPolicy(keep=True, max_files=10, max_mb=100)
    world.history.add(make_entry(n), pcm16=bytes(3200), audio=policy)


def test_switching_the_recordings_on_sends_the_kept_ones_again(world):
    add_with_recording(world)
    world.add()
    world.coordinator.tick()
    assert world.history.pending_upload_ids() == []
    assert world.transport.sent[-1].payload["entries"][0]["audio"] is None
    world.change(include_audio=True)
    assert world.history.pending_upload_ids() == [1]
    assert world.coordinator.view.waiting == 1
    world.coordinator.upload_now()
    assert world.transport.ids(len(world.transport.sent) - 1) == [1]
    assert world.transport.sent[-1].payload["entries"][0]["audio"]["format"] == "wav"
    assert world.history.pending_upload_ids() == []


def test_switching_the_recordings_on_during_an_upload_sends_them_after_it(tmp_path):
    runner = ThreadRunner()
    world = World(tmp_path, runner=runner, enabled=True, url=URL)
    add_with_recording(world)
    gate = Gate()
    world.transport.answers.append(gate)
    world.coordinator.upload_now()
    assert gate.entered.wait(10)
    world.change(include_audio=True)
    gate.release.set()
    runner.finish()
    assert world.history.pending_upload_ids() == [1]
    world.history.close()


def test_switching_the_recordings_off_sends_nothing_again(world):
    add_with_recording(world)
    world.change(include_audio=True)
    world.coordinator.tick()
    assert world.history.pending_upload_ids() == []
    world.change(include_audio=False)
    assert world.history.pending_upload_ids() == []


def test_recordings_are_held_only_while_they_are_uploaded(world):
    assert world.history.hold_recordings is world.upload.include_audio
    world.change(include_audio=True)
    assert world.history.hold_recordings is True
    world.change(include_audio=False)
    assert world.history.hold_recordings is False
    world.change(include_audio=True, enabled=False)
    assert world.history.hold_recordings is False


def test_skipped_apps_are_not_counted_as_waiting(world):
    world.history.add(make_entry(1, app_process="KeePassXC.exe"))
    world.history.add(make_entry(2))
    world.change(skip_apps=["keepassxc.exe"])
    assert world.coordinator.view.waiting == 1


def test_the_history_learns_the_skipped_apps_with_the_hold(app, tmp_path):
    world = World(tmp_path, enabled=True, url=URL, skip_apps=["KeePassXC.exe"])
    assert world.history.upload_skip_apps == ("keepassxc.exe",)
    world.history.add(make_entry(1, app_process="keepassxc.exe"))
    assert world.history.pending_upload_count() == 0
    world.change(skip_apps=["signal.exe"])
    assert world.history.upload_skip_apps == ("signal.exe",)
    world.change(enabled=False)
    assert world.history.upload_skip_apps == ("signal.exe",)
    world.history.close()


def test_a_run_that_left_an_entry_out_says_so(world, monkeypatch):
    world.add(2)
    real = upload.wire_entry

    def broken(entry, *args, **kwargs):
        if entry.id == 1:
            raise RuntimeError("cannot build")
        return real(entry, *args, **kwargs)

    monkeypatch.setattr(upload, "wire_entry", broken)
    world.coordinator.upload_now()
    assert world.upload.last_sent == 1
    assert world.upload.last_error == ""
    assert world.coordinator.view.message == "Left out 1 dictation Spells could not read."
    assert world.coordinator.view.waiting == 1


def test_the_connection_test_reports_and_changes_nothing(world):
    world.add()
    world.coordinator.test_connection()
    assert world.coordinator.view.message == TEST_OK_TEXT
    assert world.transport.sent[0].payload["entries"] == []
    assert world.upload.last_attempt == 0.0
    assert world.history.pending_upload_ids() == [1]
    world.transport.answers = [401]
    world.coordinator.test_connection()
    assert world.coordinator.view.message == upload.TOKEN_REFUSED
    assert world.upload.last_error == ""


def test_the_token_never_lands_in_the_settings_file(world, tmp_path):
    world.coordinator.set_token(TOKEN)
    world.add()
    world.coordinator.upload_now()
    text = (tmp_path / "settings.json").read_text(encoding="utf-8")
    assert TOKEN not in text
    assert TOKEN.encode() not in (tmp_path / TOKEN_FILE).read_bytes()
    assert world.coordinator.token() == TOKEN
    assert world.transport.sent[0].headers["Authorization"] == f"Bearer {TOKEN}"


def test_a_crashed_job_becomes_the_last_error_without_the_token(world, tmp_path):
    world.coordinator.set_token(TOKEN)

    def broken(*_args):
        raise RuntimeError(f"boom {TOKEN}")

    world.history.install_id = broken
    world.coordinator.upload_now()
    assert "boom" in world.upload.last_error
    assert TOKEN not in world.upload.last_error
    assert world.coordinator.view.working == ""
    world.coordinator.test_connection()
    assert "boom" in world.coordinator.view.message
    assert TOKEN not in world.coordinator.view.message
    assert TOKEN not in (tmp_path / "settings.json").read_text(encoding="utf-8")


def test_a_token_that_cannot_be_sent_is_refused_and_the_old_one_kept(world):
    assert world.coordinator.set_token(TOKEN) == ""
    assert world.coordinator.set_token("part1\npart2") == upload.TOKEN_PROBLEM
    assert world.tokens.read() == TOKEN


def test_errors_never_carry_the_token_into_the_settings_file(world, tmp_path):
    world.coordinator.set_token(TOKEN)
    world.add()
    echoed = Response(400, json.dumps({"error": f"unknown bearer {TOKEN}"}).encode())
    world.transport.answers = [echoed]
    world.coordinator.upload_now()
    assert "unknown bearer" in world.upload.last_error
    assert TOKEN not in world.upload.last_error
    world.transport.answers = [ValueError(f"Invalid header value b'Bearer {TOKEN}'")]
    world.coordinator.upload_now()
    assert world.upload.last_error == upload.SEND_FAILED
    world.transport.answers = [echoed]
    world.coordinator.test_connection()
    assert "unknown bearer" in world.coordinator.view.message
    assert TOKEN not in world.coordinator.view.message
    assert TOKEN not in world.coordinator.status()
    assert TOKEN not in (tmp_path / "settings.json").read_text(encoding="utf-8")


def test_the_real_runner_delivers_on_the_qt_thread(app, tmp_path):
    config = ConfigStore(tmp_path / "settings.json")
    config.update(lambda s: replace(s, upload=replace(s.upload, enabled=True, url=URL)))
    transport = FakeTransport()
    with HistoryStore(tmp_path / "history.db") as history:
        history.add(make_entry(1))
        coordinator = UploadCoordinator(
            config=config, history=history, tokens=TokenStore(None), transport=transport
        )
        coordinator.upload_now()
        assert coordinator.view.working == "upload"
        deadline = time.time() + 10
        while coordinator.view.working and time.time() < deadline:
            flush(app, 1)
        assert coordinator.view.working == ""
        assert config.settings.upload.last_sent == 1
        coordinator.shutdown()


def test_shutdown_stops_the_tick(world):
    world.coordinator.start_timer()
    world.coordinator.shutdown()
    world.coordinator.tick()
    assert world.transport.sent == []


# the status line


def test_the_status_line_says_what_happened_and_what_waits():
    now = 1_800_000_000.0
    fresh = UploadSettings(enabled=True)
    assert status_text(fresh, UploadView(waiting=4), now) == "Nothing uploaded yet. 4 waiting."
    done = replace(fresh, last_success=now - 60, last_sent=1)
    assert status_text(done, UploadView(), now) == (
        "Last upload today, 1 dictation sent. 0 waiting."
    )
    older = replace(fresh, last_success=now - 3 * upload.DAY_S, last_sent=12)
    assert "3 days ago, 12 dictations sent" in status_text(older, UploadView(), now)
    refused = replace(fresh, last_error=upload.TOKEN_REFUSED)
    assert status_text(refused, UploadView(), now).endswith(PAUSED_TEXT)
    assert status_text(fresh, UploadView(working="upload"), now) == "Uploading..."
    assert status_text(fresh, UploadView(working="test"), now) == "Testing the connection..."
    off = UploadSettings()
    assert "waiting" not in status_text(off, UploadView(message=TEST_OK_TEXT), now)
    assert status_text(off, UploadView(message=TEST_OK_TEXT), now).endswith(TEST_OK_TEXT)


# the page


def make_page(tmp_path, *, attach=True, confirm=True, **settings):
    world = World(tmp_path, **settings)
    messages = Messages()
    asked: list[tuple[str, str]] = []

    def answer(title: str, text: str) -> bool:
        asked.append((title, text))
        return confirm

    page = UploadPage(config=world.config, notify=messages, confirm=answer, clock=world.clock)
    if attach:
        page.attach(world.coordinator)
    return page, world, messages, asked


def finish(edit: QtWidgets.QLineEdit, text: str) -> None:
    edit.setText(text)
    edit.editingFinished.emit()


def test_the_page_shows_the_settings(app, tmp_path):
    page, world, *_ = make_page(
        tmp_path,
        enabled=True,
        url="http://example.com/in",
        schedule="weekly",
        include_audio=True,
        skip_apps=["keepassxc.exe", "Signal.exe"],
        device_name="Study",
    )
    assert page.enabled.isChecked()
    assert page.url.text() == "http://example.com/in"
    assert page.schedule.currentData() == "weekly"
    assert page.include_audio.isChecked()
    assert page.skip_apps.text() == "keepassxc.exe, Signal.exe"
    assert page.device_name.text() == "Study"
    assert page.url_row.description_label.text() == UNENCRYPTED_HINT
    assert page.hold_note.text() == HOLD_NOTE
    assert "30 days" in HOLD_NOTE
    page.close()
    world.history.close()


def test_the_page_follows_a_change_made_elsewhere(app, tmp_path):
    page, world, *_ = make_page(tmp_path)
    world.config.update(
        lambda s: replace(s, upload=replace(s.upload, enabled=True, schedule="manual"))
    )
    page.apply_settings(world.config.settings)
    assert page.enabled.isChecked()
    assert page.schedule.currentData() == "manual"
    assert world.history.upload_hold is True
    page.close()
    world.history.close()


def test_switching_uploading_on_writes_the_setting_and_holds_the_history(app, tmp_path):
    page, world, *_ = make_page(tmp_path)
    assert world.history.upload_hold is False
    page.enabled.click()
    assert world.upload.enabled is True
    assert world.history.upload_hold is True
    assert world.transport.sent == []
    page.close()
    world.history.close()


def test_an_http_address_says_it_is_sent_unencrypted(app, tmp_path):
    page, world, *_ = make_page(tmp_path)
    assert page.url_row.description_label.text() == URL_HINT
    page.url.setText("http://example.com/in")
    assert page.url_row.description_label.text() == UNENCRYPTED_HINT
    page.url.setText("https://example.com/in")
    assert page.url_row.description_label.text() == URL_HINT
    page.close()
    world.history.close()


def test_an_address_is_stored_trimmed_and_a_bad_one_is_refused(app, tmp_path):
    page, world, messages, _ = make_page(tmp_path)
    finish(page.url, "  https://example.com/spells  ")
    assert world.upload.url == "https://example.com/spells"
    finish(page.url, "ftp://example.com/spells")
    assert world.upload.url == "https://example.com/spells"
    assert page.url.text() == "https://example.com/spells"
    assert messages.texts and "https://" in messages.texts[-1]
    finish(page.url, "")
    assert world.upload.url == ""
    page.close()
    world.history.close()


def test_the_schedule_and_the_lists_are_written(app, tmp_path):
    page, world, *_ = make_page(tmp_path)
    page.schedule.setCurrentIndex(page.schedule.findData("weekly"))
    assert world.upload.schedule == "weekly"
    finish(page.skip_apps, " keepassxc.exe, ,KeePassXC.EXE, signal.exe ")
    assert world.upload.skip_apps == ["keepassxc.exe", "signal.exe"]
    finish(page.device_name, "  Study  ")
    assert world.upload.device_name == "Study"
    page.close()
    world.history.close()


def test_parse_apps_drops_blanks_and_repeats():
    assert parse_apps("a.exe, b.exe,,A.EXE , ") == ["a.exe", "b.exe"]
    assert parse_apps("") == []


def test_including_the_recordings_offers_to_keep_them(app, tmp_path):
    page, world, _messages, asked = make_page(tmp_path)
    page.include_audio.click()
    assert world.upload.include_audio is True
    assert asked == [("Include the recordings", KEEP_AUDIO_QUESTION)]
    assert world.config.settings.history.keep_audio is True
    page.include_audio.click()
    assert world.upload.include_audio is False
    page.include_audio.click()
    assert len(asked) == 1
    page.close()
    world.history.close()


def test_declining_to_keep_recordings_leaves_them_off(app, tmp_path):
    page, world, _messages, asked = make_page(tmp_path, confirm=False)
    page.include_audio.click()
    assert world.upload.include_audio is True
    assert len(asked) == 1
    assert world.config.settings.history.keep_audio is False
    page.close()
    world.history.close()


def test_the_token_field_saves_to_the_token_file_only(app, tmp_path):
    page, world, *_ = make_page(tmp_path, enabled=True, url=URL)
    assert page.token.echoMode() == QtWidgets.QLineEdit.EchoMode.Password
    finish(page.token, TOKEN)
    assert world.tokens.read() == TOKEN
    raw = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert TOKEN not in json.dumps(raw)
    second = UploadPage(
        config=world.config, notify=Messages(), confirm=lambda *_: True, clock=world.clock
    )
    second.attach(world.coordinator)
    assert second.token.text() == TOKEN
    finish(page.token, "")
    assert not world.tokens.exists()
    page.close()
    second.close()
    world.history.close()


def test_the_page_refuses_a_token_it_cannot_send_and_says_why(app, tmp_path):
    page, world, messages, _ = make_page(tmp_path, enabled=True, url=URL)
    finish(page.token, TOKEN)
    finish(page.token, "part1\npart2")
    assert messages.texts == [upload.TOKEN_PROBLEM]
    assert world.tokens.read() == TOKEN
    assert page.token.text() == TOKEN
    assert "part2" not in (tmp_path / "settings.json").read_text(encoding="utf-8")
    page.close()
    world.history.close()


def test_the_buttons_need_an_address_and_the_switch(app, tmp_path):
    page, world, *_ = make_page(tmp_path)
    assert not page.test_button.isEnabled()
    assert not page.upload_button.isEnabled()
    finish(page.url, URL)
    assert page.test_button.isEnabled()
    assert not page.upload_button.isEnabled()
    page.enabled.click()
    assert page.upload_button.isEnabled()
    page.close()
    world.history.close()


def test_the_buttons_run_the_coordinator_and_the_status_follows(app, tmp_path):
    page, world, *_ = make_page(tmp_path, enabled=True, url=URL)
    world.history.add(history_entry(1))
    world.coordinator.refresh()
    assert "1 waiting." in page.status_line
    page.test_button.click()
    assert page.status_line.endswith(TEST_OK_TEXT)
    page.upload_button.click()
    assert "1 dictation sent" in page.status_line
    assert "0 waiting." in page.status_line
    page.close()
    world.history.close()


def test_an_unattached_page_says_so_and_offers_nothing(app, tmp_path):
    page, world, *_ = make_page(tmp_path, attach=False, enabled=True, url=URL)
    assert not page.token.isEnabled()
    assert not page.test_button.isEnabled()
    assert not page.upload_button.isEnabled()
    assert "not available" in page.status_line
    page.close()
    world.history.close()
