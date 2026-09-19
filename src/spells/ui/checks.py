"""Running the on-demand transcript check off the Qt thread (spec 8.4).

The check is one cleanup-model request per row, so a batch of twenty would freeze the
settings window for as long as the model takes. CheckRunner hands the whole batch to one
worker, which walks it in order and reports each row as it lands; the UI stays live, the
cleanup engine sees one request at a time, and a batch can be abandoned when the page
closes.

Signals are emitted from the worker thread and arrive on the Qt thread through Qt's
queued connections, which is the same rule the rest of the app follows (spec 14).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from PySide6 import QtCore

from spells.quality import CheckResult

log = logging.getLogger(__name__)

Check = Callable[[str, str], CheckResult]


@dataclass(frozen=True)
class CheckJob:
    """One row to check: its history id, its raw transcript and its language code."""

    entry_id: int
    text: str
    language: str = ""


def default_executor(work: Callable[[], None]) -> None:
    QtCore.QThreadPool.globalInstance().start(_Runnable(work))


class _Runnable(QtCore.QRunnable):
    def __init__(self, work: Callable[[], None]) -> None:
        super().__init__()
        self._work = work

    def run(self) -> None:
        self._work()


class CheckRunner(QtCore.QObject):
    """Runs a batch of checks on a worker thread, one row at a time.

    submit() refuses a second batch while one is running and returns False, so a double
    click cannot queue the model twice. cancel() stops the batch at the next row; a
    request already in flight finishes and is dropped.
    """

    checked = QtCore.Signal(int, object)
    done = QtCore.Signal(int, int)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        check: Check,
        executor: Callable[[Callable[[], None]], None] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._check = check
        self._executor = executor or default_executor
        self._busy = False
        self._cancelled = False

    @property
    def busy(self) -> bool:
        return self._busy

    def cancel(self) -> None:
        self._cancelled = True

    def submit(self, jobs: Sequence[CheckJob]) -> bool:
        if self._busy or not jobs:
            return False
        self._busy = True
        self._cancelled = False
        batch = list(jobs)
        self._executor(lambda: self._run(batch))
        return True

    def _run(self, jobs: list[CheckJob]) -> None:
        stored = 0
        skipped = 0
        try:
            for job in jobs:
                if self._cancelled:
                    break
                result = self._one(job)
                if result is None:
                    skipped += 1
                    continue
                if result.ok:
                    stored += 1
                    self.checked.emit(job.entry_id, result)
                else:
                    skipped += 1
                    if result.status == "unavailable":
                        break
        finally:
            self._busy = False
            self.done.emit(stored, skipped)

    def _one(self, job: CheckJob) -> Any:
        try:
            return self._check(job.text, job.language)
        except Exception as exc:
            log.exception("the transcript check raised")
            self.failed.emit(str(exc))
            return None


__all__ = ["Check", "CheckJob", "CheckRunner", "default_executor"]
