from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
import time
import urllib.error
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from spells import upload
from spells.history import AudioPolicy, HistoryEntry, HistoryStore
from spells.models import StageTimings
from spells.quality import QualitySignals
from spells.upload import (
    DAY_S,
    WEEK_S,
    Item,
    Response,
    Uploader,
    batches,
    classify,
    encode_body,
    envelope,
    upload_due,
    url_problem,
    wire_entry,
)

NOW = 1_800_000_000.0
URL = "https://example.com/spells"


def make_entry(n: int, **overrides) -> HistoryEntry:
    fields = {
        "id": None,
        "created_at": time.time() - 3600 + n,
        "raw_text": f"raw {n}",
        "cleaned_text": f"Cleaned {n}.",
        "delivered_text": f"Cleaned {n}.",
        "app_process": "notepad.exe",
        "app_title": "Untitled",
        "language": "en",
        "used_llm": True,
        "cleanup_reason": "ok",
        "outcome": "pasted",
        "timings": StageTimings(release_to_transcript_ms=812.0, extra={"audio_s": 6.2}),
        "signals": QualitySignals(audio_s=6.2, word_count=17, words_per_minute=164.5),
    }
    fields.update(overrides)
    return HistoryEntry(**fields)


@dataclass
class Sent:
    url: str
    payload: dict
    headers: dict
    timeout: float
    size: int


class FakeTransport:
    def __init__(self, *answers, default=200) -> None:
        self.answers = list(answers)
        self.default = default
        self.sent: list[Sent] = []

    def __call__(self, url, body, headers, timeout):
        payload = json.loads(body.decode("utf-8"))
        self.sent.append(Sent(url, payload, dict(headers), timeout, len(body)))
        answer = self.answers.pop(0) if self.answers else self.default
        if callable(answer):
            answer = answer(payload)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, int):
            return Response(answer)
        return answer

    def ids(self, index: int) -> list[int]:
        return [entry["id"] for entry in self.sent[index].payload["entries"]]


@pytest.fixture
def store(tmp_path: Path):
    with HistoryStore(tmp_path / "history.db", upload_hold=True) as history:
        yield history


def uploader(store, transport, **kwargs) -> Uploader:
    kwargs.setdefault("url", URL)
    kwargs.setdefault("device_name", "DESKTOP-1")
    kwargs.setdefault("version", "0.6.0")
    return Uploader(history=store, transport=transport, **kwargs)


# when an upload is due


@pytest.mark.parametrize(
    ("schedule", "last_success", "last_attempt", "now", "due"),
    [
        ("manual", 0.0, 0.0, NOW, False),
        ("manual", NOW - 30 * DAY_S, NOW - 30 * DAY_S, NOW, False),
        ("daily", 0.0, 0.0, NOW, True),
        ("daily", NOW - DAY_S + 60, NOW - DAY_S + 60, NOW, False),
        ("daily", NOW - DAY_S, NOW - DAY_S, NOW, True),
        ("weekly", NOW - 6 * DAY_S, NOW - 6 * DAY_S, NOW, False),
        ("weekly", NOW - WEEK_S, NOW - WEEK_S, NOW, True),
        ("daily", NOW + 3600, NOW + 3600, NOW, True),
        ("weekly", NOW - 3600, NOW + 60, NOW, True),
        ("daily", NOW - 3600, NOW - 60, NOW, True),
        ("weekly", NOW - DAY_S, NOW - 2 * DAY_S, NOW, False),
        ("hourly", 0.0, 0.0, NOW, False),
    ],
)
def test_upload_due(schedule, last_success, last_attempt, now, due):
    assert upload_due(schedule, last_success, last_attempt, now) is due


# addresses


@pytest.mark.parametrize(
    "url", ["https://example.com/spells", "http://example.com:8080/in", "HTTPS://Example.com"]
)
def test_http_and_https_addresses_are_accepted(url):
    assert url_problem(url) == ""


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "ftp://example.com/x",
        "file:///C:/spells",
        "example.com/spells",
        "https:///nohost",
        "https://example.com/a b",
        "https://example.com:99999/x",
    ],
)
def test_other_addresses_are_refused_with_a_reason(url):
    assert url_problem(url)


def test_only_http_counts_as_unencrypted():
    assert upload.is_unencrypted("http://example.com/x")
    assert upload.is_unencrypted(" HTTP://example.com/x")
    assert not upload.is_unencrypted("https://example.com/x")
    assert not upload.is_unencrypted("")


# what one entry looks like on the wire


def test_iso_time_is_local_with_offset_and_milliseconds():
    text = upload.iso_time(NOW + 0.345)
    parsed = datetime.fromisoformat(text)
    assert parsed.utcoffset() is not None
    assert text[19:23] == ".345"
    assert parsed.timestamp() == pytest.approx(NOW + 0.345)


def test_a_wire_entry_carries_every_field_of_the_protocol():
    entry = make_entry(
        1,
        id=123,
        mode="compose",
        instruction="write a note",
        selection_chars=4,
        asr_engine="qwen3-asr",
        cleanup_engine="gemma",
        check_verdict="GOOD",
        check_reason="fine",
        checked_at=NOW,
        quality_label="good",
    )
    wire = wire_entry(entry, "install-1")
    assert wire["key"] == "install-1:123"
    assert wire["id"] == 123
    assert wire["mode"] == "compose"
    assert wire["app"] == {"process": "notepad.exe", "title": "Untitled"}
    assert wire["text"] == {"raw": "raw 1", "cleaned": "Cleaned 1.", "delivered": "Cleaned 1."}
    assert wire["instruction"] == "write a note"
    assert wire["selectionChars"] == 4
    assert wire["usedLlm"] is True
    assert wire["engines"] == {"asr": "qwen3-asr", "cleanup": "gemma"}
    assert wire["timings"]["release_to_transcript_ms"] == 812.0
    assert wire["timings"]["extra"] == {"audio_s": 6.2}
    assert wire["signals"]["words_per_minute"] == 164.5
    assert wire["quality"]["checkVerdict"] == "GOOD"
    assert wire["quality"]["checkedAt"] == upload.iso_time(NOW)
    assert wire["audio"] is None
    assert "audioSkipped" not in wire
    assert set(wire) == {
        "key", "id", "createdAt", "mode", "language", "app", "text", "instruction",
        "selectionChars", "usedLlm", "cleanupReason", "outcome", "engines", "timings",
        "signals", "quality", "audio",
    }


def test_an_unchecked_entry_has_no_check_time_and_numbers_stay_json():
    entry = make_entry(1, id=1, timings=StageTimings(delivery_ms=float("nan")))
    wire = wire_entry(entry, "x")
    assert wire["quality"]["checkedAt"] is None
    assert wire["timings"]["delivery_ms"] is None
    json.loads(Item.of(wire).data)


def test_read_audio_describes_the_wav_and_carries_it_whole(tmp_path):
    with HistoryStore(tmp_path / "history.db") as history:
        history.add(make_entry(1), pcm16=bytes(3200), audio=AudioPolicy(keep=True))
        path = history.recording_path(history.recent()[0])
    raw = path.read_bytes()
    audio = upload.read_audio(path)
    assert audio["format"] == "wav"
    assert (audio["sampleRate"], audio["channels"], audio["sampleWidth"]) == (16000, 1, 2)
    assert audio["seconds"] == 0.1
    assert audio["bytes"] == len(raw)
    assert audio["sha256"] == hashlib.sha256(raw).hexdigest()
    assert base64.b64decode(audio["data"]) == raw
    assert upload.read_audio(tmp_path / "missing.wav") is None
    assert upload.read_audio(None) is None


NOISE = b"\x01\x02" * 1600


def bad_chunk_size(raw: bytes) -> bytes:
    return raw[:16] + (60).to_bytes(4, "little") + raw[20:]


@pytest.mark.parametrize(
    "spoil",
    [bad_chunk_size, lambda raw: raw[:30], lambda raw: b"RIFF" + b"\xff" * 40, lambda raw: b""],
)
def test_a_broken_recording_reads_as_no_audio_and_logs_only_its_name(tmp_path, caplog, spoil):
    with HistoryStore(tmp_path / "history.db") as history:
        history.add(make_entry(1), pcm16=NOISE, audio=AudioPolicy(keep=True))
        path = history.recording_path(history.recent()[0])
    path.write_bytes(spoil(path.read_bytes()))
    with caplog.at_level(logging.WARNING, logger="spells.upload"):
        assert upload.read_audio(path) is None
    assert path.name in caplog.text
    assert str(path.parent) not in caplog.text


def test_an_unreadable_recording_file_logs_only_its_name(tmp_path, caplog):
    folder = tmp_path / "private-folder"
    folder.mkdir()
    with caplog.at_level(logging.WARNING, logger="spells.upload"):
        assert upload.read_audio(folder) is None
    assert "private-folder" in caplog.text
    assert str(tmp_path) not in caplog.text


def test_the_body_is_the_envelope_with_the_entries_last():
    head = envelope(install_id="abc", device_name="PC", lost=2, sent_at="now", version="0.6.0")
    items = [Item.of(wire_entry(make_entry(n, id=n), "abc")) for n in (1, 2)]
    body = json.loads(encode_body(head, items))
    assert body["format"] == "spells-upload/1"
    assert body["app"] == {"name": "Spells", "version": "0.6.0"}
    assert body["device"] == {"installId": "abc", "name": "PC"}
    assert body["lost"] == 2
    assert [entry["key"] for entry in body["entries"]] == ["abc:1", "abc:2"]
    assert json.loads(encode_body(head, []))["entries"] == []


# batches


def items(count: int, size: int = 10) -> list[Item]:
    return [Item(n, {"id": n}, b"x" * size) for n in range(1, count + 1)]


def test_a_batch_holds_at_most_two_hundred_entries():
    got = list(batches(items(450)))
    assert [len(batch) for batch in got] == [200, 200, 50]
    assert [item.id for batch in got for item in batch] == list(range(1, 451))


def test_a_batch_stays_under_the_byte_limit():
    got = list(batches(items(10, size=100), max_bytes=350, overhead=20))
    assert [len(batch) for batch in got] == [3, 3, 3, 1]
    for batch in got:
        assert 20 + sum(item.size for item in batch) + len(batch) - 1 <= 350


def test_an_entry_larger_than_the_limit_goes_alone():
    parts = [*items(2, size=10), Item(3, {"id": 3}, b"y" * 1000), *items(1, size=10)]
    got = list(batches(parts, max_bytes=100))
    assert [[item.id for item in batch] for batch in got] == [[1, 2], [3], [1]]


def test_no_items_make_no_batches():
    assert list(batches([])) == []


# answers


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (200, upload.OK),
        (202, upload.OK),
        (204, upload.OK),
        (413, upload.TOO_LARGE),
        (401, upload.REFUSED),
        (403, upload.REFUSED),
        (400, upload.FAILED),
        (429, upload.FAILED),
        (500, upload.FAILED),
        (503, upload.FAILED),
        (302, upload.FAILED),
        (404, upload.FAILED),
    ],
)
def test_classify(status, kind):
    assert classify(status).kind == kind


def test_a_refusal_names_the_token():
    assert classify(401).message == upload.TOKEN_REFUSED


def test_a_bad_request_shows_the_first_two_hundred_characters_of_the_error():
    body = json.dumps({"error": "entry abc:1 has no createdAt\n" + "x" * 500}).encode()
    message = classify(400, body).message
    assert "entry abc:1 has no createdAt" in message
    assert "\n" not in message
    assert message.count("x") == 200 - len("entry abc:1 has no createdAt ")
    assert classify(400, b"not json").message == "The server did not accept the upload (400)."
    assert "(400)" in classify(400, b'{"error": 5}').message


# a run against a fake server


def test_a_run_sends_everything_oldest_first_and_marks_it(store):
    for n in range(3):
        store.add(make_entry(n))
    transport = FakeTransport()
    result = uploader(store, transport, token="secret-token").run()
    assert result.ok and result.sent == 3
    assert len(transport.sent) == 1
    request = transport.sent[0]
    assert request.url == URL
    assert request.timeout == 120.0
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert request.headers["Content-Type"] == "application/json; charset=utf-8"
    assert request.headers["User-Agent"] == "Spells/0.6.0"
    payload = request.payload
    install = store.install_id()
    assert payload["device"] == {"installId": install, "name": "DESKTOP-1"}
    assert payload["lost"] == 0
    assert [entry["key"] for entry in payload["entries"]] == [f"{install}:{n}" for n in (1, 2, 3)]
    assert store.pending_upload_ids() == []


def test_a_check_that_finishes_while_its_batch_is_in_flight_is_sent_again(store):
    for n in range(3):
        store.add(make_entry(n))

    def check_then_accept(_payload):
        store.set_check(1, "POOR", "garbled")
        return 200

    transport = FakeTransport(check_then_accept)
    result = uploader(store, transport).run()
    assert result.ok
    assert transport.sent[0].payload["entries"][0]["quality"]["checkVerdict"] == ""
    assert store.pending_upload_ids() == [1]
    uploader(store, transport).run()
    assert transport.ids(1) == [1]
    assert transport.sent[1].payload["entries"][0]["quality"]["checkVerdict"] == "POOR"
    assert store.pending_upload_ids() == []


def test_without_a_token_there_is_no_authorization_header(store):
    store.add(make_entry(1))
    transport = FakeTransport()
    uploader(store, transport, token="  ").run()
    assert "Authorization" not in transport.sent[0].headers


def test_a_run_with_nothing_new_sends_an_empty_batch(store):
    transport = FakeTransport()
    result = uploader(store, transport).run()
    assert result.ok and result.sent == 0
    assert transport.sent[0].payload["entries"] == []


def test_skipped_apps_are_never_sent(store):
    store.add(make_entry(1, app_process="KeePassXC.exe"))
    store.add(make_entry(2))
    transport = FakeTransport()
    result = uploader(store, transport, skip_apps=["keepassxc.exe"]).run()
    assert result.sent == 1
    assert transport.ids(0) == [2]
    store.reset_uploaded()
    uploader(store, transport, skip_apps=[]).run()
    assert transport.ids(1) == [2]


def test_the_batches_follow_the_count_limit(store):
    for n in range(5):
        store.add(make_entry(n))
    transport = FakeTransport()
    result = uploader(store, transport, max_entries=2).run()
    assert result.sent == 5
    assert [transport.ids(i) for i in range(3)] == [[1, 2], [3, 4], [5]]


def test_a_too_large_batch_is_split_in_half_until_it_fits(store):
    for n in range(6):
        store.add(make_entry(n))
    transport = FakeTransport(default=lambda payload: 413 if len(payload["entries"]) > 2 else 200)
    result = uploader(store, transport).run()
    assert result.ok and result.sent == 6
    sizes = [len(request.payload["entries"]) for request in transport.sent]
    assert sizes == [6, 3, 1, 2, 3, 1, 2]
    assert store.pending_upload_ids() == []


def test_a_lone_entry_that_is_too_large_goes_once_more_without_its_audio(store):
    policy = AudioPolicy(keep=True)
    store.add(make_entry(1), pcm16=bytes(3200), audio=policy)
    store.add(make_entry(2))

    def answer(payload):
        has_audio = any(entry["audio"] for entry in payload["entries"])
        return 413 if has_audio else 200

    transport = FakeTransport(default=answer)
    result = uploader(store, transport, include_audio=True).run()
    assert result.ok and result.sent == 2 and result.moved_on == 0
    retried = transport.sent[2].payload["entries"]
    assert [entry["id"] for entry in retried] == [1]
    assert retried[0]["audio"] is None
    assert retried[0]["audioSkipped"] == "too large"
    assert store.pending_upload_ids() == []


def test_an_entry_still_too_large_without_audio_is_left_and_the_run_moves_on(store):
    store.add(make_entry(1, raw_text="huge"))
    store.add(make_entry(2))

    def answer(payload):
        return 413 if any(e["text"]["raw"] == "huge" for e in payload["entries"]) else 200

    transport = FakeTransport(default=answer)
    result = uploader(store, transport).run()
    assert result.ok and result.sent == 1 and result.moved_on == 1
    assert store.pending_upload_ids() == [1]


def test_audio_travels_only_when_asked(store):
    store.add(make_entry(1), pcm16=bytes(3200), audio=AudioPolicy(keep=True))
    raw = store.recording_path(store.recent()[0]).read_bytes()
    without = FakeTransport()
    uploader(store, without).run()
    assert without.sent[0].payload["entries"][0]["audio"] is None
    store.reset_uploaded()
    carrying = FakeTransport()
    uploader(store, carrying, include_audio=True).run()
    audio = carrying.sent[0].payload["entries"][0]["audio"]
    assert audio["sha256"] == hashlib.sha256(raw).hexdigest()
    assert base64.b64decode(audio["data"]) == raw


def test_a_broken_recording_goes_without_audio_and_the_run_goes_on(store):
    policy = AudioPolicy(keep=True)
    store.add(make_entry(1), pcm16=NOISE, audio=policy)
    store.add(make_entry(2), pcm16=NOISE, audio=policy)
    path = store.recording_path(store.entries_by_ids([1])[0])
    path.write_bytes(bad_chunk_size(path.read_bytes()))
    transport = FakeTransport()
    result = uploader(store, transport, include_audio=True).run()
    assert result.ok and result.sent == 2
    entries = transport.sent[0].payload["entries"]
    assert entries[0]["audio"] is None
    assert entries[1]["audio"]["format"] == "wav"
    assert store.pending_upload_ids() == []


def test_an_entry_that_cannot_be_prepared_is_left_out_and_the_run_goes_on(store, monkeypatch):
    for n in range(3):
        store.add(make_entry(n))
    real = upload.wire_entry

    def broken(entry, *args, **kwargs):
        if entry.id == 1:
            raise RuntimeError("cannot build")
        return real(entry, *args, **kwargs)

    monkeypatch.setattr(upload, "wire_entry", broken)
    transport = FakeTransport()
    result = uploader(store, transport).run()
    assert result.ok and result.sent == 2 and result.unreadable == 1
    assert transport.ids(0) == [2, 3]
    assert store.pending_upload_ids() == [1]
    again = uploader(store, transport).run()
    assert again.ok and again.sent == 0 and again.unreadable == 1
    assert transport.sent[1].payload["entries"] == []


def test_a_refused_token_stops_the_run_and_marks_nothing(store):
    store.add(make_entry(1))
    transport = FakeTransport(401)
    result = uploader(store, transport).run()
    assert result.refused and result.error == upload.TOKEN_REFUSED
    assert store.pending_upload_ids() == [1]


def test_a_server_error_stops_the_run_after_the_batches_that_went_through(store):
    for n in range(3):
        store.add(make_entry(n))
    transport = FakeTransport(200, 503)
    result = uploader(store, transport, max_entries=1).run()
    assert result.sent == 1
    assert "503" in result.error and not result.refused
    assert len(transport.sent) == 2
    assert store.pending_upload_ids() == [2, 3]


def test_a_bad_request_stops_with_the_servers_words(store):
    store.add(make_entry(1))
    transport = FakeTransport(Response(400, b'{"error": "createdAt is not a date"}'))
    result = uploader(store, transport).run()
    assert result.error == "The server did not accept the upload: createdAt is not a date"


@pytest.mark.parametrize(
    ("exc", "text"),
    [
        (urllib.error.URLError(ConnectionRefusedError(10061, "refused")), "could not reach"),
        (TimeoutError("timed out"), "two minutes"),
        (urllib.error.URLError(TimeoutError("timed out")), "two minutes"),
    ],
)
def test_a_network_failure_stops_the_run(store, exc, text):
    store.add(make_entry(1))
    result = uploader(store, FakeTransport(exc)).run()
    assert text in result.error
    assert store.pending_upload_ids() == [1]


def test_lost_rows_are_reported_and_settled_after_the_server_accepts(tmp_path):
    with HistoryStore(tmp_path / "history.db", "7d", upload_hold=True) as store:
        store.add(make_entry(1, created_at=time.time() - 60 * DAY_S))
        store.add(make_entry(2))
        assert store.upload_lost() == 1
        transport = FakeTransport()
        uploader(store, transport, max_entries=1).run()
        assert transport.sent[0].payload["lost"] == 1
        assert store.upload_lost() == 0
        uploader(store, transport).run()
        assert transport.sent[1].payload["lost"] == 0


def test_lost_rows_stay_counted_when_the_server_fails(tmp_path):
    with HistoryStore(tmp_path / "history.db", "7d", upload_hold=True) as store:
        store.add(make_entry(1, created_at=time.time() - 60 * DAY_S))
        uploader(store, FakeTransport(500)).run()
        assert store.upload_lost() == 1


def test_a_bad_address_never_reaches_the_transport(store):
    store.add(make_entry(1))
    transport = FakeTransport()
    result = uploader(store, transport, url="ftp://example.com").run()
    assert result.error and transport.sent == []
    assert uploader(store, transport, url="").test().error


def test_a_cancelled_run_stops_before_the_next_batch(store):
    for n in range(3):
        store.add(make_entry(n))
    transport = FakeTransport()

    def stop() -> bool:
        return len(transport.sent) >= 1

    result = uploader(store, transport, max_entries=1, cancelled=stop).run()
    assert result.cancelled and not result.ok
    assert result.sent == 1


def test_a_cancelled_run_sends_no_half_of_a_split_batch(store):
    for n in range(4):
        store.add(make_entry(n))
    transport = FakeTransport(413)

    def stop() -> bool:
        return len(transport.sent) >= 1

    result = uploader(store, transport, cancelled=stop).run()
    assert result.cancelled
    assert len(transport.sent) == 1
    assert store.pending_upload_ids() == [1, 2, 3, 4]


def test_the_connection_test_sends_an_empty_batch_and_changes_nothing(tmp_path):
    with HistoryStore(tmp_path / "history.db", "7d", upload_hold=True) as store:
        store.add(make_entry(1, created_at=time.time() - 60 * DAY_S))
        store.add(make_entry(2))
        transport = FakeTransport()
        result = uploader(store, transport, token="t").test()
        assert result.ok
        payload = transport.sent[0].payload
        assert payload["entries"] == [] and payload["lost"] == 0
        assert store.pending_upload_ids() == [2]
        assert store.upload_lost() == 1
        refused = uploader(store, FakeTransport(403)).test()
        assert refused.refused and refused.error == upload.TOKEN_REFUSED


SECRET ="tok-0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize("token", ["", SECRET, "abc-DEF_123.~+/=", "!#$%&'*^`|"])
def test_bearer_token_characters_are_accepted(token):
    assert upload.token_problem(token) == ""


@pytest.mark.parametrize(
    "token",
    ["part1\npart2", "part1\rpart2", "two words", "tab\there", "s3cr€et", "nul\x00", "del\x7f"],
)
def test_a_token_with_other_characters_is_refused(token):
    assert upload.token_problem(token) == upload.TOKEN_PROBLEM


def test_a_token_that_cannot_be_sent_never_reaches_the_transport(store):
    store.add(make_entry(1))
    transport = FakeTransport()
    assert uploader(store, transport, token="part1\npart2").run().error == upload.TOKEN_PROBLEM
    assert uploader(store, transport, token="part1\npart2").test().error == upload.TOKEN_PROBLEM
    assert transport.sent == []
    assert store.pending_upload_ids() == [1]


@pytest.mark.parametrize(
    "exc",
    [
        ValueError(f"Invalid header value b'Bearer {SECRET}'"),
        UnicodeEncodeError("latin-1", SECRET, 0, 1, "ordinal not in range(256)"),
        RuntimeError(f"unexpected {SECRET}"),
    ],
)
def test_an_unexpected_transport_failure_shows_a_fixed_message(store, exc):
    store.add(make_entry(1))
    result = uploader(store, FakeTransport(exc), token=SECRET).run()
    assert result.error == upload.SEND_FAILED
    assert store.pending_upload_ids() == [1]
    tested = uploader(store, FakeTransport(exc), token=SECRET).test()
    assert tested.error == upload.SEND_FAILED


def test_the_token_is_removed_from_the_servers_words(store):
    store.add(make_entry(1))
    body = json.dumps({"error": f"no such bearer {SECRET}, not {SECRET}"}).encode()
    result = uploader(store, FakeTransport(Response(400, body)), token=SECRET).run()
    assert SECRET not in result.error
    assert "no such bearer" in result.error
    tested = uploader(store, FakeTransport(Response(400, body)), token=SECRET).test()
    assert SECRET not in tested.error


def test_a_token_cut_by_the_error_limit_is_removed_before_the_cut(store):
    store.add(make_entry(1))
    body = json.dumps({"error": "x" * 180 + SECRET}).encode()
    result = uploader(store, FakeTransport(Response(400, body)), token=SECRET).run()
    assert SECRET[:20] not in result.error


def test_the_token_is_removed_from_a_network_failure(store):
    store.add(make_entry(1))
    exc = urllib.error.URLError(f"no route for {SECRET}")
    result = uploader(store, FakeTransport(exc), token=SECRET).run()
    assert "could not reach" in result.error
    assert SECRET not in result.error


def test_the_device_name_defaults_to_the_computer_name(store, monkeypatch):
    monkeypatch.setenv("COMPUTERNAME", "STUDY-PC")
    transport = FakeTransport()
    uploader(store, transport, device_name="").run()
    assert transport.sent[0].payload["device"]["name"] == "STUDY-PC"


# the real transport, against a server on this computer


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.server.bodies.append(self.rfile.read(length))
        if self.path == "/moved":
            self.send_response(302)
            self.send_header("Location", "http://example.com/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        reply = b'{"error": "nope"}' if self.path == "/bad" else b'{"ok": true}'
        self.send_response(400 if self.path == "/bad" else 200)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.bodies = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def test_the_http_transport_posts_and_never_follows_a_redirect(server, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    ok = upload.http_transport(base + "/in", b'{"entries":[]}', headers, 5.0)
    assert ok.status == 200 and ok.body == b'{"ok": true}'
    bad = upload.http_transport(base + "/bad", b"{}", headers, 5.0)
    assert bad.status == 400 and b"nope" in bad.body
    moved = upload.http_transport(base + "/moved", b"{}", headers, 5.0)
    assert moved.status == 302
    assert server.bodies == [b'{"entries":[]}', b"{}", b"{}"]
