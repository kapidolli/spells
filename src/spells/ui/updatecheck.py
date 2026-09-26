"""The Qt side of the update check (spec 19.7): the timer, the threads and the state.

`spells.updates` holds every rule and every byte of network work and knows nothing about
Qt. This module is the part that has to live near the event loop: it runs those functions on
a worker thread, turns their results into one `Phase` the About page renders, remembers in
the settings file when a check last ran and which version the tray last mentioned, and owns
the hourly tick that the weekly check rides on.

Nothing here starts a check on its own beyond that tick, and the tick does nothing unless
the user switched the weekly check on. A build with no update address (the usual case for a
development checkout) reports `Phase.UNCONFIGURED` and never touches the network at all.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from PySide6 import QtCore

from spells import __version__, updates
from spells.config import ConfigStore, Settings, UpdateSettings
from spells.updates import Release, UpdateCancelled, UpdateError

log = logging.getLogger(__name__)

TICK_MS = 60 * 60 * 1000
FIRST_TICK_MS = 90 * 1000

DOWNLOAD_DIR_PREFIX = "spells-update-"


class Phase(Enum):
    """What the About page is showing right now."""

    UNCONFIGURED = "unconfigured"
    IDLE = "idle"
    CHECKING = "checking"
    UP_TO_DATE = "up_to_date"
    AVAILABLE = "available"
    BLOCKED = "blocked"
    DOWNLOADING = "downloading"
    READY = "ready"
    INSTALLING = "installing"
    ERROR = "error"


@dataclass(frozen=True)
class UpdateView:
    """One snapshot for the About page; every field is already in the user's words."""

    phase: Phase = Phase.IDLE
    release: Release | None = None
    message: str = ""
    last_check: float = 0.0
    done_bytes: int = 0
    total_bytes: int = 0
    weekly_check: bool = False


def capture(work: Callable[[], Any]) -> Any:
    """The result of the work, or the exception it raised, which is also logged."""
    try:
        return work()
    except Exception as exc:
        log.exception("an update job failed")
        return exc


def direct_runner(work: Callable[[], Any], done: Callable[[Any], None]) -> None:
    """Run the work on this thread and hand the result, or the exception, straight back.

    The tests use this so a check or a download is one call with no thread and no event
    loop; the coordinator's own runner behaves the same way, one thread later.
    """
    done(capture(work))


class _Job(QtCore.QThread):
    """One piece of work on a thread of its own; the result comes back on the Qt thread."""

    done = QtCore.Signal(object)

    def __init__(self, work: Callable[[], Any], parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._work = work

    def run(self) -> None:
        self.done.emit(capture(self._work))


class UpdateCoordinator(QtCore.QObject):
    """Checks for a newer Spells, downloads it when asked, and starts it.

    Every path here is started by the user, except the hourly tick, which does nothing
    unless `updates.weekly_check` is on and a week has gone by. A check that fails is a
    line on the About page and nothing more: it never raises, never retries in a loop and
    never touches a dictation.
    """

    changed = QtCore.Signal(object)
    update_found = QtCore.Signal(str, str)
    _progress = QtCore.Signal(int, int)

    def __init__(
        self,
        *,
        config: ConfigStore,
        source_url: str = "",
        busy: Callable[[], bool] | None = None,
        on_quit: Callable[[], None] | None = None,
        fetch: Callable[..., Release] = updates.fetch_release,
        download: Callable[..., Path] = updates.download_installer,
        launch: Callable[..., None] = updates.launch_installer,
        runner: Callable[[Callable[[], Any], Callable[[Any], None]], None] | None = None,
        clock: Callable[[], float] = time.time,
        temp_dir: Path | None = None,
        current_version: str = __version__,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._source = source_url or ""
        self._busy = busy or (lambda: False)
        self._on_quit = on_quit
        self._fetch = fetch
        self._download = download
        self._launch = launch
        self._runner = runner or self._thread_runner
        self._clock = clock
        self._temp_dir = temp_dir
        self._version = current_version
        self._cancel = threading.Event()
        self._job: _Job | None = None
        self._installer: Path | None = None
        self._timer: QtCore.QTimer | None = None
        phase = Phase.IDLE if self._source else Phase.UNCONFIGURED
        settings = config.settings.updates
        self._view = UpdateView(
            phase=phase,
            message=self._idle_message(settings),
            last_check=settings.last_check,
            weekly_check=settings.weekly_check,
        )
        self._progress.connect(self._on_progress)

    # What the page reads ------------------------------------------------------------------

    @property
    def configured(self) -> bool:
        """Whether this build was given an update address when it was built."""
        return bool(self._source)

    @property
    def view(self) -> UpdateView:
        return self._view

    def apply_settings(self, settings: Settings) -> None:
        """Follow a settings change that came from anywhere, including another window."""
        current = settings.updates
        if current.weekly_check == self._view.weekly_check:
            return
        self._set(weekly_check=current.weekly_check)

    def set_weekly_check(self, enabled: bool) -> None:
        """The About page's switch. Turning it on never starts a check by itself."""
        value = bool(enabled)
        self._store(lambda current: replace(current, weekly_check=value))
        self._set(weekly_check=value)

    # Checking -----------------------------------------------------------------------------

    def start_timer(self) -> None:
        """The hourly tick the weekly check rides on. Safe to call more than once."""
        if self._timer is not None or not self._source:
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

    def tick(self) -> None:
        """One weekly-check opportunity. Does nothing unless the user asked for it."""
        if not self._source or self._view.phase in _WORKING:
            return
        if self._busy():
            return
        settings = self._config.settings.updates
        if not updates.check_due(
            enabled=settings.weekly_check, now=self._clock(), last_check=settings.last_check
        ):
            return
        self.check(manual=False)

    def check(self, manual: bool = True) -> None:
        """Ask the server what the newest version is. Always allowed when the user asks."""
        if not self._source:
            self._set(phase=Phase.UNCONFIGURED, message=UNCONFIGURED_TEXT)
            return
        if self._view.phase in _WORKING:
            return
        self._set(phase=Phase.CHECKING, message="Checking for updates...")
        source = self._source
        self._run(lambda: self._fetch(source), lambda result: self._checked(result, manual))

    def _checked(self, result: Any, manual: bool) -> None:
        now = self._clock()
        self._store(lambda current: replace(current, last_check=now))
        if isinstance(result, BaseException):
            reason = str(result) or result.__class__.__name__
            log.info("The update check did not finish: %s", reason)
            self._set(phase=Phase.ERROR, message=reason, last_check=now)
            return
        decision = updates.decide(result, self._version)
        if not decision.available:
            self._set(
                phase=Phase.UP_TO_DATE, release=None, message=decision.reason, last_check=now
            )
            return
        release = decision.release
        assert release is not None
        phase = Phase.BLOCKED if decision.blocked else Phase.AVAILABLE
        self._set(phase=phase, release=release, message=decision.reason, last_check=now)
        if not decision.blocked and not manual:
            self._offer(release, now)

    def _offer(self, release: Release, now: float) -> None:
        """One balloon for this version, at most one a day, and never over a dictation."""
        settings = self._config.settings.updates
        if not updates.balloon_due(
            now=now,
            last_at=settings.last_offer_at,
            version=release.version,
            last_version=settings.last_offer_version,
        ):
            return
        if self._busy():
            return
        self._store(
            lambda current: replace(
                current, last_offer_version=release.version, last_offer_at=now
            )
        )
        self.update_found.emit(release.version, updates.date_text(release.released))

    # Downloading and installing -----------------------------------------------------------

    def start_download(self) -> None:
        """Fetch the installer the check found. The only other network call the check makes."""
        release = self._view.release
        if release is None or self._view.phase not in (Phase.AVAILABLE, Phase.ERROR):
            return
        self._cancel.clear()
        self._installer = None
        self._set(
            phase=Phase.DOWNLOADING,
            message=f"Downloading {updates.size_text(release.size_bytes)}...",
            done_bytes=0,
            total_bytes=release.size_bytes,
        )
        directory = Path(self._temp_dir) if self._temp_dir else _download_dir()

        def work() -> Path:
            return self._download(
                release,
                directory,
                progress=self._progress.emit,
                cancel=self._cancel.is_set,
            )

        self._run(work, self._downloaded)

    def cancel_download(self) -> None:
        """Stop the download. The partial file is deleted by the worker before it returns."""
        if self._view.phase is not Phase.DOWNLOADING:
            return
        self._cancel.set()
        self._set(message="Stopping the download...")

    def _downloaded(self, result: Any) -> None:
        if isinstance(result, UpdateCancelled):
            self._set(
                phase=Phase.AVAILABLE, message="The download was cancelled.", done_bytes=0
            )
            return
        if isinstance(result, BaseException):
            reason = str(result) or result.__class__.__name__
            log.warning("The update download failed: %s", reason)
            self._set(phase=Phase.ERROR, message=reason, done_bytes=0)
            return
        self._installer = Path(result)
        self._set(
            phase=Phase.READY,
            message=(
                "The download was checked against its hash. "
                "Spells will close while it installs."
            ),
        )

    def install(self) -> None:
        """Start the verified installer and quit, so Setup can replace the files."""
        if self._view.phase is not Phase.READY or self._installer is None:
            return
        release = self._view.release
        if release is None or not updates.verify_file(self._installer, release):
            self._set(
                phase=Phase.ERROR,
                message=(
                    "The downloaded installer no longer matches its checksum, "
                    "so Spells will not run it."
                ),
            )
            return
        self._set(phase=Phase.INSTALLING, message="Starting the installer. Spells will close.")
        try:
            self._launch(self._installer)
        except UpdateError as exc:
            log.warning("The installer could not be started: %s", exc)
            self._set(phase=Phase.ERROR, message=str(exc))
            return
        if self._on_quit is not None:
            self._on_quit()

    # Internals ----------------------------------------------------------------------------

    def _run(self, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        try:
            self._runner(work, done)
        except Exception as exc:
            log.exception("an update job could not be started")
            done(exc)

    def _thread_runner(self, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        job = _Job(work, self)
        job.done.connect(done)
        job.finished.connect(job.deleteLater)
        self._job = job
        job.start()

    def _on_progress(self, done_bytes: int, total_bytes: int) -> None:
        if self._view.phase is not Phase.DOWNLOADING:
            return
        self._set(done_bytes=int(done_bytes), total_bytes=int(total_bytes))

    def _set(self, **fields: Any) -> None:
        self._view = replace(self._view, **fields)
        self.changed.emit(self._view)

    def _store(self, mutate: Callable[[UpdateSettings], UpdateSettings]) -> None:
        try:
            self._config.update(
                lambda settings: replace(settings, updates=mutate(settings.updates))
            )
        except Exception:
            log.exception("the update settings could not be written")

    def _idle_message(self, settings: UpdateSettings) -> str:
        if not self._source:
            return UNCONFIGURED_TEXT
        if settings.last_check <= 0:
            return "Spells has not checked for updates yet."
        return f"Last checked {_when(settings.last_check, self._clock())}."


UNCONFIGURED_TEXT = (
    "This build of Spells was made without an update address, so it cannot check."
)

_WORKING = (Phase.CHECKING, Phase.DOWNLOADING, Phase.INSTALLING)


def _download_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix=DOWNLOAD_DIR_PREFIX))


def _when(stamp: float, now: float) -> str:
    """"today", "yesterday" or a date, which is as precise as this ever needs to be."""
    if stamp <= 0:
        return "never"
    days = int((now - stamp) // updates.DAY_S)
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    return time.strftime("%d %B %Y", time.localtime(stamp))


__all__ = [
    "FIRST_TICK_MS",
    "TICK_MS",
    "UNCONFIGURED_TEXT",
    "Phase",
    "UpdateCoordinator",
    "UpdateView",
    "capture",
    "direct_runner",
]
