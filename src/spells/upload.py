from __future__ import annotations

import base64
import dataclasses
import hashlib
import http.client
import io
import json
import logging
import math
import os
import platform
import re
import time
import urllib.error
import urllib.request
import wave
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from spells import __version__
from spells.history import MODE_DICTATE, HistoryEntry
from spells.quality import signals_to_dict

log = logging.getLogger(__name__)

FORMAT = "spells-upload/1"
APP_NAME = "Spells"
USER_AGENT = f"Spells/{__version__}"

SCHEDULE_MANUAL = "manual"
SCHEDULE_DAILY = "daily"
SCHEDULE_WEEKLY = "weekly"
SCHEDULES = (SCHEDULE_MANUAL, SCHEDULE_DAILY, SCHEDULE_WEEKLY)

MAX_ENTRIES = 200
MAX_BATCH_BYTES = 8 * 1024 * 1024
TIMEOUT_S = 120.0
MAX_REPLY_BYTES = 64 * 1024
MAX_ERROR_CHARS = 200
ENVELOPE_SLACK = 256

DAY_S = 24 * 60 * 60
WEEK_S = 7 * DAY_S

ALLOWED_SCHEMES = ("https", "http")

TOKEN_REFUSED = "The server refused the token."
TOKEN_PROBLEM = (
    "The token can only hold letters, digits and punctuation, with no spaces or line breaks."
)
SEND_FAILED = "Spells could not send the upload."
REDACTED = "[token]"
AUDIO_TOO_LARGE = "too large"

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_TOKEN_RE = re.compile(r"[\x21-\x7e]*")


def url_problem(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return "Type the address of your server first."
    if any(char.isspace() for char in text):
        return "The address cannot contain spaces."
    try:
        parts = urlsplit(text)
        _ = parts.port
    except ValueError:
        return "That is not an address Spells can read."
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        return "The address has to start with https:// or http://."
    if not parts.hostname:
        return "The address names no server."
    return ""


def token_problem(token: str) -> str:
    if _TOKEN_RE.fullmatch(token or ""):
        return ""
    return TOKEN_PROBLEM


def redact(text: str, *secrets: str) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return text


def is_unencrypted(url: str) -> bool:
    return urlsplit((url or "").strip()).scheme.lower() == "http"


def upload_due(schedule: str, last_success: float, last_attempt: float, now: float) -> bool:
    if schedule not in (SCHEDULE_DAILY, SCHEDULE_WEEKLY):
        return False
    if last_success <= 0:
        return True
    if last_success > now or last_attempt > now:
        return True
    if last_attempt > last_success:
        return True
    period = DAY_S if schedule == SCHEDULE_DAILY else WEEK_S
    return now - last_success >= period


def iso_time(stamp: float) -> str:
    return datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="milliseconds")


def computer_name() -> str:
    return os.environ.get("COMPUTERNAME") or platform.node() or "Windows PC"


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def read_audio(path: Path | None) -> dict | None:
    if path is None:
        return None
    name = Path(path).name
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        log.warning("could not read the recording %s (%s)", name, exc.__class__.__name__)
        return None
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.getnframes()
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        log.warning(
            "the recording %s is not a readable WAV file (%s)", name, exc.__class__.__name__
        )
        return None
    return {
        "format": "wav",
        "sampleRate": rate,
        "channels": channels,
        "sampleWidth": width,
        "seconds": round(frames / rate, 3) if rate else 0.0,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "data": base64.b64encode(data).decode("ascii"),
    }


def wire_entry(entry: HistoryEntry, install_id: str, audio: dict | None = None) -> dict:
    return {
        "key": f"{install_id}:{entry.id}",
        "id": entry.id,
        "createdAt": iso_time(entry.created_at),
        "mode": entry.mode or MODE_DICTATE,
        "language": entry.language,
        "app": {"process": entry.app_process, "title": entry.app_title},
        "text": {
            "raw": entry.raw_text,
            "cleaned": entry.cleaned_text,
            "delivered": entry.delivered_text,
        },
        "instruction": entry.instruction,
        "selectionChars": entry.selection_chars,
        "usedLlm": entry.used_llm,
        "cleanupReason": entry.cleanup_reason,
        "outcome": entry.outcome,
        "engines": {"asr": entry.asr_engine, "cleanup": entry.cleanup_engine},
        "timings": _finite(dataclasses.asdict(entry.timings)),
        "signals": _finite(signals_to_dict(entry.signals)),
        "quality": {
            "label": entry.quality_label,
            "reason": entry.quality_reason,
            "checkVerdict": entry.check_verdict,
            "checkReason": entry.check_reason,
            "checkedAt": iso_time(entry.checked_at) if entry.checked_at else None,
        },
        "audio": audio,
    }


def encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class Item:
    id: int
    wire: dict
    data: bytes
    checked_at: float = 0.0

    @classmethod
    def of(cls, wire: dict, checked_at: float = 0.0) -> Item:
        return cls(int(wire["id"]), wire, encode(wire), float(checked_at))

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def has_audio(self) -> bool:
        return self.wire.get("audio") is not None

    def without_audio(self) -> Item:
        return Item.of(
            {**self.wire, "audio": None, "audioSkipped": AUDIO_TOO_LARGE}, self.checked_at
        )


def envelope(
    *, install_id: str, device_name: str, lost: int, sent_at: str, version: str = __version__
) -> dict:
    return {
        "format": FORMAT,
        "sentAt": sent_at,
        "app": {"name": APP_NAME, "version": version},
        "device": {"installId": install_id, "name": device_name},
        "lost": int(lost),
        "entries": [],
    }


def encode_body(head: dict, items: Sequence[Item]) -> bytes:
    prefix = encode({**head, "entries": []})
    return prefix[:-2] + b",".join(item.data for item in items) + b"]}"


def batches(
    items: Iterable[Item],
    *,
    max_entries: int = MAX_ENTRIES,
    max_bytes: int = MAX_BATCH_BYTES,
    overhead: int = 0,
) -> Iterator[list[Item]]:
    current: list[Item] = []
    size = overhead
    for item in items:
        extra = item.size + (1 if current else 0)
        if current and (len(current) >= max_entries or size + extra > max_bytes):
            yield current
            current = []
            size = overhead
            extra = item.size
        current.append(item)
        size += extra
    if current:
        yield current


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes = b""


Transport = Callable[[str, bytes, dict[str, str], float], Response]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> Response:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as reply:
            return Response(int(reply.status), reply.read(MAX_REPLY_BYTES))
    except urllib.error.HTTPError as exc:
        try:
            data = exc.read(MAX_REPLY_BYTES) or b""
        except (OSError, http.client.HTTPException):
            data = b""
        finally:
            exc.close()
        return Response(int(exc.code), data)


OK = "ok"
TOO_LARGE = "too_large"
REFUSED = "refused"
FAILED = "failed"


@dataclass(frozen=True)
class Outcome:
    kind: str
    message: str = ""


def server_error(body: bytes, secret: str = "") -> str:
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, AttributeError):
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, str):
        return ""
    return _CONTROL_RE.sub(" ", redact(error, secret)).strip()[:MAX_ERROR_CHARS]


def classify(status: int, body: bytes = b"", secret: str = "") -> Outcome:
    if 200 <= status < 300:
        return Outcome(OK)
    if status == 413:
        return Outcome(TOO_LARGE, "The server found the upload too large (413).")
    if status in (401, 403):
        return Outcome(REFUSED, TOKEN_REFUSED)
    if status == 400:
        error = server_error(body, secret)
        if error:
            return Outcome(FAILED, f"The server did not accept the upload: {error}")
        return Outcome(FAILED, "The server did not accept the upload (400).")
    if status == 429:
        return Outcome(FAILED, "The server asked Spells to wait (429).")
    if status >= 500:
        return Outcome(FAILED, f"The server had a problem ({status}).")
    if 300 <= status < 400:
        return Outcome(
            FAILED, f"The server answered with a redirect ({status}). Use the address it names."
        )
    return Outcome(FAILED, f"The server answered {status}.")


def network_problem(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None) or exc
    if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
        return "The server did not answer within two minutes."
    text = str(reason) or reason.__class__.__name__
    return f"Spells could not reach the server ({text})."


@dataclass(frozen=True)
class RunResult:
    sent: int = 0
    error: str = ""
    refused: bool = False
    moved_on: int = 0
    cancelled: bool = False
    unreadable: int = 0

    @property
    def ok(self) -> bool:
        return not self.error and not self.cancelled


class _Stop(Exception):
    def __init__(self, outcome: Outcome) -> None:
        super().__init__(outcome.message)
        self.outcome = outcome


class _Cancelled(Exception):
    pass


class Uploader:
    def __init__(
        self,
        *,
        history: Any,
        url: str,
        token: str = "",
        include_audio: bool = False,
        skip_apps: Sequence[str] = (),
        device_name: str = "",
        transport: Transport = http_transport,
        clock: Callable[[], float] = time.time,
        version: str = __version__,
        cancelled: Callable[[], bool] | None = None,
        max_entries: int = MAX_ENTRIES,
        max_bytes: int = MAX_BATCH_BYTES,
        timeout_s: float = TIMEOUT_S,
    ) -> None:
        self._history = history
        self._url = (url or "").strip()
        self._token = (token or "").strip()
        self._include_audio = bool(include_audio)
        self._skip_apps = tuple(skip_apps)
        self._device_name = (device_name or "").strip() or computer_name()
        self._transport = transport
        self._clock = clock
        self._version = version
        self._cancelled = cancelled or (lambda: False)
        self._max_entries = max(1, int(max_entries))
        self._max_bytes = int(max_bytes)
        self._timeout = float(timeout_s)
        self._install_id = ""
        self._sent = 0
        self._moved_on = 0
        self._unreadable = 0

    def test(self) -> RunResult:
        problem = url_problem(self._url) or token_problem(self._token)
        if problem:
            return RunResult(error=problem)
        self._install_id = self._history.install_id()
        outcome = self._post([], record=False)
        if outcome.kind == OK:
            return RunResult()
        return RunResult(error=self._redact(outcome.message), refused=outcome.kind == REFUSED)

    def run(self) -> RunResult:
        problem = url_problem(self._url) or token_problem(self._token)
        if problem:
            return RunResult(error=problem)
        history = self._history
        if self._skip_apps:
            history.mark_upload_skipped(self._skip_apps)
        self._install_id = history.install_id()
        self._sent = 0
        self._moved_on = 0
        self._unreadable = 0
        ids = history.pending_upload_ids()
        try:
            posted = False
            for batch in batches(
                self._items(ids),
                max_entries=self._max_entries,
                max_bytes=self._max_bytes,
                overhead=self._overhead(),
            ):
                self._send(batch)
                posted = True
            if not posted:
                self._send([])
        except _Cancelled:
            return self._result(cancelled=True)
        except _Stop as stop:
            return self._result(stop.outcome)
        return self._result()

    def _result(self, outcome: Outcome | None = None, *, cancelled: bool = False) -> RunResult:
        if outcome is None:
            return RunResult(
                sent=self._sent,
                moved_on=self._moved_on,
                cancelled=cancelled,
                unreadable=self._unreadable,
            )
        return RunResult(
            sent=self._sent,
            error=self._redact(outcome.message),
            refused=outcome.kind == REFUSED,
            moved_on=self._moved_on,
            unreadable=self._unreadable,
        )

    def _redact(self, text: str) -> str:
        return redact(text, self._token)

    def _items(self, ids: Sequence[int]) -> Iterator[Item]:
        for start in range(0, len(ids), self._max_entries):
            for entry in self._history.entries_by_ids(ids[start : start + self._max_entries]):
                item = self._item(entry)
                if item is not None:
                    yield item

    def _item(self, entry: HistoryEntry) -> Item | None:
        try:
            audio = None
            if self._include_audio and entry.audio_file:
                audio = read_audio(self._history.recording_path(entry))
            return Item.of(wire_entry(entry, self._install_id, audio), entry.checked_at)
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            self._unreadable += 1
            log.warning(
                "dictation %s could not be prepared for the upload (%s)",
                entry.id,
                exc.__class__.__name__,
            )
            return None

    def _overhead(self) -> int:
        head = self._head(self._history.upload_lost())
        return len(encode(head)) + ENVELOPE_SLACK

    def _head(self, lost: int) -> dict:
        return envelope(
            install_id=self._install_id,
            device_name=self._device_name,
            lost=lost,
            sent_at=iso_time(self._clock()),
            version=self._version,
        )

    def _send(self, batch: list[Item]) -> None:
        outcome = self._post(batch)
        if outcome.kind == OK:
            return
        if outcome.kind != TOO_LARGE:
            raise _Stop(outcome)
        if not batch:
            raise _Stop(outcome)
        if len(batch) > 1:
            half = len(batch) // 2
            self._send(batch[:half])
            self._send(batch[half:])
            return
        item = batch[0]
        if item.has_audio:
            retry = self._post([item.without_audio()])
            if retry.kind == OK:
                return
            if retry.kind != TOO_LARGE:
                raise _Stop(retry)
        self._moved_on += 1
        log.warning("dictation %d is too large for the server even without audio", item.id)

    def _post(self, items: Sequence[Item], *, record: bool = True) -> Outcome:
        if record and self._cancelled():
            raise _Cancelled
        history = self._history
        reported = history.upload_lost() if record else 0
        body = encode_body(self._head(reported), items)
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": f"Spells/{self._version}",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            response = self._transport(self._url, body, headers, self._timeout)
        except (OSError, http.client.HTTPException) as exc:
            log.info("the upload did not reach the server: %s", exc.__class__.__name__)
            return Outcome(FAILED, self._redact(network_problem(exc)))
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            log.info("the upload could not be sent: %s", exc.__class__.__name__)
            return Outcome(FAILED, SEND_FAILED)
        outcome = classify(int(response.status), response.body, self._token)
        if outcome.kind == OK and record:
            if items:
                history.mark_uploaded(
                    [item.id for item in items],
                    self._clock(),
                    checked_at={item.id: item.checked_at for item in items},
                )
                self._sent += len(items)
            if reported:
                history.settle_upload_lost(reported)
        elif outcome.kind != OK:
            log.info("the upload server answered %d", response.status)
        return outcome


__all__ = [
    "ALLOWED_SCHEMES",
    "AUDIO_TOO_LARGE",
    "FORMAT",
    "MAX_BATCH_BYTES",
    "MAX_ENTRIES",
    "REDACTED",
    "SCHEDULES",
    "SCHEDULE_DAILY",
    "SCHEDULE_MANUAL",
    "SCHEDULE_WEEKLY",
    "SEND_FAILED",
    "TIMEOUT_S",
    "TOKEN_PROBLEM",
    "TOKEN_REFUSED",
    "Item",
    "Outcome",
    "Response",
    "RunResult",
    "Transport",
    "Uploader",
    "batches",
    "classify",
    "computer_name",
    "encode",
    "encode_body",
    "envelope",
    "http_transport",
    "is_unencrypted",
    "iso_time",
    "network_problem",
    "read_audio",
    "redact",
    "server_error",
    "token_problem",
    "upload_due",
    "url_problem",
    "wire_entry",
]
