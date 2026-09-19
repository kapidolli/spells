"""Delivery: target checks at delivery time, then paste or type the text.

Spec 5.2 (inject row), 6 step 8, 11 (all steps), 12 (100 ms delivery budget) and 16
(focus moved or target elevated). Batch 2 decisions V1-1 and E12 (elevation is re-checked
on the CURRENT foreground window at delivery time, not captured at press), V2-3 (release
held modifiers before typing as well as before Ctrl+V), V2-4 and batch 3 B3-32 (which
clipboard formats survive a snapshot), V2-12 and batch 3 B3-31 (typing batches and line
breaks), V2-16 (a delayed-render change does not bump the sequence number, accepted) and
E11 (our text carries the clipboard-history exclusion format).

Every Win32 detail lives in spells.win32; this module only sequences those calls, times
them and turns the outcome into a DeliveryResult. deliver() never raises: the pipeline
records whatever comes back and the user is told through the pill or a notification.

The clock is time.perf_counter, not time.monotonic: monotonic is GetTickCount64 here and
steps in 15.6 ms (batch 3 decision B3-23), which cannot measure a delivery against the
100 ms budget, and its epoch differs from the perf_counter one the rest of the app uses.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from spells.models import (
    DeliveryMethod,
    DeliveryOutcome,
    DeliveryResult,
    TargetContext,
)
from spells.win32 import clipboard as win32_clipboard
from spells.win32 import input as win32_input
from spells.win32 import window as win32_window

logger = logging.getLogger(__name__)

# Spec 11 step 2.4: wait this long before putting the user's clipboard back, so the target
# app has processed the Ctrl+V it was sent.
RESTORE_DELAY_S = 0.3


@dataclass(frozen=True)
class InjectBackends:
    """The Win32 modules delivery talks to. Tests pass fakes with the same functions."""

    window: Any = win32_window
    input: Any = win32_input
    clipboard: Any = win32_clipboard


DEFAULT_BACKENDS = InjectBackends()


@dataclass(frozen=True)
class DeliveryReport:
    """What deliver() did: the stored result plus the numbers the pipeline records.

    `elapsed_ms` covers the target checks plus the injection itself, measured before the
    restore delay (spec 12 budgets post-process plus inject at 100 ms), and is None when
    nothing was injected. `clipboard_restored` is None for every method but paste.
    """

    result: DeliveryResult
    elapsed_ms: float | None = None
    clipboard_restored: bool | None = None

    @property
    def outcome(self) -> DeliveryOutcome:
        return self.result.outcome


def ends_with_whitespace(text: str) -> bool:
    """Whether delivered text ended in whitespace, for LastDelivery (spec 10.4)."""
    return bool(text) and text[-1].isspace()


def deliver(
    text: str,
    ctx: TargetContext,
    method: DeliveryMethod,
    *,
    backends: InjectBackends | None = None,
    restore_delay_s: float = RESTORE_DELAY_S,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> DeliveryReport:
    """Put `text` into the window captured at press time, or fall back to the clipboard.

    `ctx` is the context captured on the hotkey press. `sleeper` and `clock` are injected
    so tests can drive the restore delay and the timing without waiting.
    """
    if backends is None:
        backends = DEFAULT_BACKENDS
    started = clock()
    try:
        fallback = _target_check(text, ctx, backends)
        if fallback is not None:
            return fallback
        if method == DeliveryMethod.PASTE:
            return _paste(text, backends, restore_delay_s, sleeper, clock, started)
        if method == DeliveryMethod.TYPE:
            return _type(text, backends, clock, started)
        return _failed(f"unknown delivery method: {method!r}")
    except Exception as exc:
        logger.warning("delivery failed", exc_info=True)
        return _failed(repr(exc))


def target_holds_focus(ctx: TargetContext, backends: InjectBackends) -> str:
    """"" when the captured window still owns the foreground, else why it does not.

    The same three questions _target_check asks, in the same order, without touching the
    clipboard: live insertion (spec 6, B5-57) asks them before every burst it sends and a
    non-empty answer stops it at once.
    """
    current = int(backends.window.foreground_hwnd())
    if not current or not ctx.hwnd:
        return "no target window"
    if _is_elevated(backends, current):
        return "foreground window is elevated"
    if current != ctx.hwnd:
        return "focus moved"
    return ""


def _target_check(text: str, ctx: TargetContext, backends: InjectBackends) -> DeliveryReport | None:
    """Spec 11 step 1: inject only into the window that was focused at press time.

    Returns None when that window still holds the foreground, otherwise the copy fallback:
    the text goes on the clipboard as it is (no snapshot and no restore, because we are
    not taking the clipboard back off the user) and the outcome says why.

    Order: a missing hwnd first, then elevation, then the focus comparison. hwnd 0 (the
    capture failed, or no window holds the foreground) owns no process, so the elevation
    query would answer "yes" about nothing; there is simply no window to inject into and
    the clipboard is the only safe place for the text. Otherwise elevation is checked on
    the CURRENT foreground window (V1-1, E12) and wins over a plain focus change, and it
    is checked even when the target kept focus.
    """
    current = int(backends.window.foreground_hwnd())
    if not current or not ctx.hwnd:
        outcome = DeliveryOutcome.COPIED_FOCUS_CHANGED
        reason = "no target window"
    elif _is_elevated(backends, current):
        outcome = DeliveryOutcome.COPIED_ELEVATED
        reason = "foreground window is elevated"
    elif current != ctx.hwnd:
        outcome = DeliveryOutcome.COPIED_FOCUS_CHANGED
        reason = "focus moved"
    else:
        return None
    backends.clipboard.set_text(text)
    detail = f"{reason}: captured hwnd {ctx.hwnd}, foreground hwnd {current}"
    logger.info("copied instead of injecting: %s", detail)
    return DeliveryReport(DeliveryResult(outcome, detail))


def _is_elevated(backends: InjectBackends, hwnd: int) -> bool:
    """Whether the foreground window runs above us; a broken query counts as not elevated.

    win32.window.is_elevated_window already answers True for a window it cannot inspect,
    so an exception here is the query itself failing, not a verdict, and it must not turn
    every delivery into a failure.
    """
    try:
        return bool(backends.window.is_elevated_window(hwnd))
    except Exception:
        logger.warning("elevation check failed for hwnd %s, treating it as not elevated", hwnd)
        logger.debug("elevation check traceback", exc_info=True)
        return False


def _paste(
    text: str,
    backends: InjectBackends,
    restore_delay_s: float,
    sleeper: Callable[[float], None],
    clock: Callable[[], float],
    started: float,
) -> DeliveryReport:
    """Spec 11 step 2: snapshot, set, Ctrl+V, then put the snapshot back if nothing moved."""
    clipboard = backends.clipboard
    try:
        snap = clipboard.snapshot()
        sequence = clipboard.set_text(text)
    except Exception as exc:
        # Nothing has been injected yet, so this is a clean failure.
        logger.warning("clipboard could not be prepared for the paste", exc_info=True)
        return _failed(repr(exc))
    try:
        backends.input.release_held_modifiers()
        backends.input.send_ctrl_v()
    except Exception as exc:
        # Our text is on the clipboard and the user's copy is not back yet, which is
        # exactly where the copy fallback would have left things: say so, so the pipeline
        # can still tell the user to press Ctrl+V (spec 11 step 1, spec 16).
        logger.warning("Ctrl+V could not be sent", exc_info=True)
        return _failed(f"{exc!r}; the text is on the clipboard")
    elapsed_ms = _elapsed_ms(clock, started)
    sleeper(restore_delay_s)
    restored, note = _restore(clipboard, snap, sequence)
    return DeliveryReport(DeliveryResult(DeliveryOutcome.PASTED, note), elapsed_ms, restored)


def _restore(clipboard: Any, snap: Any, sequence: int) -> tuple[bool, str]:
    """Put the user's clipboard back; a refusal or a failure is a note, not an error."""
    try:
        restored = bool(clipboard.restore(snap, sequence))
    except Exception as exc:
        logger.warning("clipboard could not be restored after the paste", exc_info=True)
        return False, f"clipboard not restored: {exc!r}"
    if restored:
        return True, "clipboard restored"
    return False, "clipboard not restored: it changed after the paste"


def _type(
    text: str,
    backends: InjectBackends,
    clock: Callable[[], float],
    started: float,
) -> DeliveryReport:
    """Spec 11 step 3 with decision V2-3: release held modifiers, then type the text."""
    backends.input.release_held_modifiers()
    backends.input.type_unicode(text)
    return DeliveryReport(
        DeliveryResult(DeliveryOutcome.TYPED, ""), _elapsed_ms(clock, started), None
    )


def _elapsed_ms(clock: Callable[[], float], started: float) -> float:
    return (clock() - started) * 1000.0


def _failed(detail: str) -> DeliveryReport:
    return DeliveryReport(DeliveryResult(DeliveryOutcome.FAILED, detail))
