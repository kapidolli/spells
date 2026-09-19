"""Snapshot of the target window at press time.

Spec 5.2 (context row), 6 step 1 and 11 step 1; batch 2 decision V1-1: elevation is not
part of the snapshot, because the hotkey cannot fire while an elevated window holds the
foreground, and `inject` re-checks the current foreground window at delivery time.

capture() runs on the recording controller thread immediately after the hotkey press
(batch 3 decision B3-27), so it stays cheap: three Win32 queries, no sleeps and no retries
beyond what the win32 layer already does. It never raises either. A failure yields an
empty context, whose hwnd 0 `inject` treats as "no target window" and answers with the
clipboard fallback, outcome copied_focus_changed (spec 11 step 1).

The clock is time.perf_counter, not time.monotonic: monotonic is GetTickCount64 here and
steps in 15.6 ms (batch 3 decision B3-23), and captured_at shares its epoch with the
hotkey thread and the pipeline's stage timers, which already use perf_counter.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from spells.models import TargetContext
from spells.win32 import window as win32_window

logger = logging.getLogger(__name__)

EMPTY_PROCESS = ""
EMPTY_TITLE = ""


def capture(
    *,
    window: Any = win32_window,
    clock: Callable[[], float] = time.perf_counter,
) -> TargetContext:
    """The foreground window's handle, process name and title, stamped with the clock.

    `window` is the spells.win32.window contract (`foreground_hwnd`, `window_process_name`,
    `window_title`); tests pass a fake. The process name is the exe basename and is empty
    when the process cannot be opened.
    """
    captured_at = clock()
    try:
        hwnd = int(window.foreground_hwnd())
        process = window.window_process_name(hwnd)
        title = window.window_title(hwnd)
    except Exception:
        logger.debug("target capture failed, using an empty context", exc_info=True)
        return TargetContext(
            hwnd=0, process=EMPTY_PROCESS, title=EMPTY_TITLE, captured_at=captured_at
        )
    return TargetContext(hwnd=hwnd, process=process, title=title, captured_at=captured_at)
