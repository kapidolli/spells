"""Snapshot of the target window at press time.

Spec 5.2 (context row), 6 step 1 and 11 step 1; batch 2 decision V1-1: elevation is not
part of the snapshot, because the hotkey cannot fire while an elevated window holds the
foreground, and `inject` re-checks the current foreground window at delivery time.

capture() runs on the recording controller thread immediately after the hotkey press
(batch 3 decision B3-27), so it stays cheap: three queries to the platform's focus, no
sleeps and no retries beyond what the platform layer already does. It never raises either.
A failure yields an empty context, whose window 0 `inject` treats as "no target window"
and answers with the clipboard fallback, outcome copied_focus_changed (spec 11 step 1).

The clock is time.perf_counter, not time.monotonic: monotonic is GetTickCount64 here and
steps in 15.6 ms (batch 3 decision B3-23), and captured_at shares its epoch with the
hotkey thread and the pipeline's stage timers, which already use perf_counter.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from spells import platform
from spells.models import TargetContext

logger = logging.getLogger(__name__)

EMPTY_PROCESS = ""
EMPTY_TITLE = ""


def capture(
    *,
    focus: Any = None,
    clock: Callable[[], float] = time.perf_counter,
) -> TargetContext:
    """The foreground window's token, app name and title, stamped with the clock.

    `focus` is the platform's Focus (`foreground`, `app_name`, `title`), and
    `platform.current().focus` when it is None; tests pass a fake. The app name is the exe
    basename on Windows and is empty when the process cannot be opened.
    """
    captured_at = clock()
    try:
        focus = focus or platform.current().focus
        window = int(focus.foreground())
        process = focus.app_name(window)
        title = focus.title(window)
    except Exception:
        logger.debug("target capture failed, using an empty context", exc_info=True)
        return TargetContext(
            window=0, process=EMPTY_PROCESS, title=EMPTY_TITLE, captured_at=captured_at
        )
    return TargetContext(window=window, process=process, title=title, captured_at=captured_at)
