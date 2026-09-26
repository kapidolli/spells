from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from PySide6 import QtCore

from spells import __version__, upload
from spells.config import ConfigStore, Settings, UploadSettings
from spells.upload import RunResult, Transport, Uploader
from spells.uploadtoken import TokenStore

log = logging.getLogger(__name__)

TICK_MS = 60 * 60 * 1000
FIRST_TICK_MS = 90 * 1000

WORK_UPLOAD = "upload"
WORK_TEST = "test"

PAUSED_TEXT = "Automatic uploads wait until you change the address or the token."
TEST_OK_TEXT = "The server accepted the test."
OFF_TEXT = "Turn on uploading first."

Runner = Callable[[Callable[[], Any], Callable[[Any], None]], None]


@dataclass(frozen=True)
class UploadView:
    working: str = ""
    message: str = ""
    waiting: int = 0


def paused(settings: UploadSettings) -> bool:
    return settings.last_error == upload.TOKEN_REFUSED


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _ago(stamp: float, now: float) -> str:
    days = int((now - stamp) // upload.DAY_S)
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    return time.strftime("on %d %B %Y", time.localtime(stamp))


def status_text(settings: UploadSettings, view: UploadView, now: float) -> str:
    if view.working == WORK_UPLOAD:
        return "Uploading..."
    if view.working == WORK_TEST:
        return "Testing the connection..."
    parts: list[str] = []
    if settings.last_success > 0:
        parts.append(
            f"Last upload {_ago(settings.last_success, now)}, "
            f"{_plural(settings.last_sent, 'dictation')} sent."
        )
    else:
        parts.append("Nothing uploaded yet.")
    if settings.enabled:
        parts.append(f"{view.waiting} waiting.")
    if settings.last_error:
        parts.append(settings.last_error)
        if paused(settings):
            parts.append(PAUSED_TEXT)
    if view.message:
        parts.append(view.message)
    return " ".join(parts)


def capture(work: Callable[[], Any]) -> Any:
    try:
        return work()
    except Exception as exc:
        log.exception("an upload job failed")
        return exc


def direct_runner(work: Callable[[], Any], done: Callable[[Any], None]) -> None:
    done(capture(work))


class _Relay(QtCore.QObject):
    done = QtCore.Signal(object, object)


class UploadCoordinator(QtCore.QObject):
    changed = QtCore.Signal(object)

    def __init__(
        self,
        *,
        config: ConfigStore,
        history: Any,
        tokens: TokenStore,
        busy: Callable[[], bool] | None = None,
        transport: Transport | None = None,
        runner: Runner | None = None,
        clock: Callable[[], float] = time.time,
        version: str = __version__,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._history = history
        self._tokens = tokens
        self._busy = busy or (lambda: False)
        self._transport = transport or upload.http_transport
        self._runner = runner or self._thread_runner
        self._clock = clock
        self._version = version
        self._cancel = threading.Event()
        self._timer: QtCore.QTimer | None = None
        self._relay = _Relay(self)
        self._relay.done.connect(self._deliver, QtCore.Qt.ConnectionType.QueuedConnection)
        current = config.settings.upload
        self._url = current.url
        self._enabled = current.enabled
        self._include_audio = current.include_audio
        self._hold(current)
        self._view = UploadView(waiting=self._count_waiting(current))

    @property
    def view(self) -> UploadView:
        return self._view

    @property
    def settings(self) -> UploadSettings:
        return self._config.settings.upload

    def status(self) -> str:
        return status_text(self.settings, self._view, self._clock())

    def apply_settings(self, _settings: Settings | None = None) -> None:
        self._follow()
        self.refresh()

    def refresh(self) -> None:
        self._set(waiting=self._count_waiting(self.settings))

    def token(self) -> str:
        return self._tokens.read()

    def has_token(self) -> bool:
        return self._tokens.exists()

    def set_token(self, token: str) -> None:
        value = (token or "").strip()
        if value == self._tokens.read():
            return
        self._tokens.write(value)
        if paused(self.settings):
            self._store(lambda current: replace(current, last_error=""))
        self._set(message="")

    def start_timer(self) -> None:
        if self._timer is not None:
            return
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self.tick)
        self._timer.start()
        QtCore.QTimer.singleShot(FIRST_TICK_MS, self.tick)

    def stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def shutdown(self) -> None:
        self.stop_timer()
        self._cancel.set()

    def tick(self) -> None:
        if self._view.working or self._cancel.is_set():
            return
        if self._busy():
            return
        current = self.settings
        if not current.enabled or not current.url or paused(current):
            return
        if not upload.upload_due(
            current.schedule, current.last_success, current.last_attempt, self._clock()
        ):
            return
        self._start_upload()

    def upload_now(self) -> None:
        if self._view.working:
            return
        if not self.settings.enabled:
            self._set(message=OFF_TEXT)
            return
        self._start_upload()

    def test_connection(self) -> None:
        if self._view.working:
            return
        uploader = self._uploader(self.settings)
        self._set(working=WORK_TEST, message="")
        self._run(uploader.test, self._tested)

    def _start_upload(self) -> None:
        current = self.settings
        now = self._clock()
        self._store(lambda settings: replace(settings, last_attempt=now))
        uploader = self._uploader(current)
        self._set(working=WORK_UPLOAD, message="")
        self._run(uploader.run, self._uploaded)

    def _uploader(self, current: UploadSettings) -> Uploader:
        return Uploader(
            history=self._history,
            url=current.url,
            token=self._tokens.read(),
            include_audio=current.include_audio,
            skip_apps=current.skip_apps,
            device_name=current.device_name,
            transport=self._transport,
            clock=self._clock,
            version=self._version,
            cancelled=self._cancel.is_set,
        )

    def _uploaded(self, result: Any) -> None:
        now = self._clock()
        message = ""
        if isinstance(result, BaseException):
            reason = str(result) or result.__class__.__name__
            error = f"The upload stopped ({reason})."
            self._store(lambda current: replace(current, last_error=error))
        elif result.ok:
            self._store(
                lambda current: replace(
                    current, last_success=now, last_sent=result.sent, last_error=""
                )
            )
        elif not result.cancelled:
            self._store(lambda current: replace(current, last_error=result.error))
        if isinstance(result, RunResult) and result.moved_on:
            message = (
                f"{_plural(result.moved_on, 'dictation')} too large for the server, "
                "even without the recording."
            )
        self._set(working="", message=message, waiting=self._count_waiting(self.settings))

    def _tested(self, result: Any) -> None:
        if isinstance(result, BaseException):
            message = f"The test stopped ({str(result) or result.__class__.__name__})."
        elif result.ok:
            message = TEST_OK_TEXT
        else:
            message = result.error
        self._set(working="", message=message)

    def _follow(self) -> None:
        current = self.settings
        if (current.enabled, current.include_audio) != (self._enabled, self._include_audio):
            self._enabled = current.enabled
            self._include_audio = current.include_audio
            self._hold(current)
        if current.url != self._url:
            self._url = current.url
            self._call("reset_uploaded")
            self._store(
                lambda settings: replace(
                    settings, last_success=0.0, last_attempt=0.0, last_sent=0, last_error=""
                )
            )
            self._set(message="")

    def _hold(self, current: UploadSettings) -> None:
        self._call("set_upload_hold", current.enabled, current.enabled and current.include_audio)

    def _count_waiting(self, current: UploadSettings) -> int:
        if not current.enabled:
            return 0
        if current.skip_apps:
            self._call("mark_upload_skipped", current.skip_apps)
        count = self._call("pending_upload_count")
        return int(count) if isinstance(count, int) else 0

    def _call(self, name: str, *args: Any) -> Any:
        method = getattr(self._history, name, None)
        if method is None:
            return None
        try:
            return method(*args)
        except Exception:
            log.exception("the history could not answer %s", name)
            return None

    def _run(self, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        try:
            self._runner(work, done)
        except Exception as exc:
            log.exception("an upload job could not be started")
            done(exc)

    def _thread_runner(self, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        relay = self._relay

        def target() -> None:
            result = capture(work)
            try:
                relay.done.emit(done, result)
            except RuntimeError:
                log.debug("the upload finished after the window closed")

        threading.Thread(target=target, name="spells-upload", daemon=True).start()

    def _deliver(self, done: Callable[[Any], None], result: Any) -> None:
        done(result)

    def _set(self, **fields: Any) -> None:
        self._view = replace(self._view, **fields)
        self.changed.emit(self._view)

    def _store(self, mutate: Callable[[UploadSettings], UploadSettings]) -> None:
        try:
            self._config.update(lambda settings: replace(settings, upload=mutate(settings.upload)))
        except Exception:
            log.exception("the upload settings could not be written")


__all__ = [
    "FIRST_TICK_MS",
    "OFF_TEXT",
    "PAUSED_TEXT",
    "TEST_OK_TEXT",
    "TICK_MS",
    "WORK_TEST",
    "WORK_UPLOAD",
    "UploadCoordinator",
    "UploadView",
    "capture",
    "direct_runner",
    "paused",
    "status_text",
]
