"""Live insertion of partial transcripts into the window being dictated into.

Spec 6 (partial passes), 11 (delivery) and decision B5-57. The pure half is the
reconciliation: given what Spells typed and what it now wants the field to hold, how many
Backspace taps and which tail of text get it there. The other half is one session per
dictation, which owns the only thing Spells may ever take back: the characters it typed
itself, counted exactly.

Three rules hold everywhere here. The clipboard is never touched, because a dictation must
leave the user's clipboard exactly as it found it. The foreground window is checked before
every burst, with the same three questions inject.deliver asks, and a session that finds
another window stops for good and removes nothing. And a backspace count is never larger
than the number of characters this session typed, whatever the reconciliation asks for.
"""

from __future__ import annotations

import logging
import threading

from spells.inject import DEFAULT_BACKENDS, InjectBackends, target_holds_focus
from spells.models import TargetContext

log = logging.getLogger(__name__)

PLAIN_PROFILES = frozenset({"Chat", "Email and docs", "Default"})
TAG = 0x5554544D


def common_prefix_len(left: str, right: str) -> int:
    """How many leading characters the two strings share."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def reconcile(inserted: str, target: str) -> tuple[int, str]:
    """(backspaces, tail) that turn `inserted` into `target`.

    The longest common prefix stays, everything after it is deleted, and the rest of the
    target is typed. A partial that only grows costs no backspace at all; one that rewrites
    an earlier word costs every character from that word onward.
    """
    prefix = common_prefix_len(inserted, target)
    return len(inserted) - prefix, target[prefix:]


def live_typing_allowed(profile_name: str, *, enabled: bool, everywhere: bool) -> bool:
    """Whether live insertion may run for this profile (spec 14.4, General)."""
    if not enabled:
        return False
    return bool(everywhere) or profile_name in PLAIN_PROFILES


class LiveSession:
    """The characters Spells typed into one window during one dictation.

    `apply` is the only mutator and takes a lock for the whole burst, so the partial thread
    and the processing worker can never interleave two bursts into the same field.
    """

    def __init__(
        self,
        ctx: TargetContext,
        backends: InjectBackends | None = None,
        *,
        batch_size: int | None = None,
    ) -> None:
        self.ctx = ctx
        self._backends = backends if backends is not None else DEFAULT_BACKENDS
        self._batch_size = batch_size
        self._lock = threading.RLock()
        self._inserted = ""
        self._stopped = False
        self._stop_reason = ""

    @property
    def inserted(self) -> str:
        with self._lock:
            return self._inserted

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    @property
    def stop_reason(self) -> str:
        with self._lock:
            return self._stop_reason

    @property
    def typed_anything(self) -> bool:
        with self._lock:
            return bool(self._inserted)

    @property
    def active(self) -> bool:
        with self._lock:
            return not self._stopped

    def stop(self, reason: str = "") -> None:
        """Stop inserting and removing. Nothing already typed is taken back."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._stop_reason = reason

    def apply(self, text: str) -> bool:
        """Make the field hold `text` instead of what this session typed.

        Returns True when the field now holds it, False when the session stopped, which
        happens when another window holds the foreground or an input call failed. A stopped
        session leaves every character it typed where it is.
        """
        with self._lock:
            if self._stopped:
                return False
            backspaces, tail = reconcile(self._inserted, text)
            backspaces = min(backspaces, len(self._inserted))
            if not backspaces and not tail:
                return True
            if backspaces and not self._erase(backspaces):
                return False
            return not (tail and not self._type(tail))

    def _erase(self, count: int) -> bool:
        reason = self._focus_reason()
        if reason:
            return self._halt(reason)
        try:
            self._send_backspaces(count)
        except Exception:
            log.exception("live insertion: the backspaces were refused")
            return self._halt("backspace refused")
        self._inserted = self._inserted[: len(self._inserted) - count]
        return True

    def _type(self, tail: str) -> bool:
        reason = self._focus_reason()
        if reason:
            return self._halt(reason)
        try:
            self._type_text(tail)
        except Exception:
            log.exception("live insertion: the text could not be typed")
            return self._halt("typing refused")
        self._inserted += tail
        return True

    def _send_backspaces(self, count: int) -> None:
        kwargs = {"extra_info": TAG, "release_modifiers": True}
        if self._batch_size is not None:
            self._backends.input.send_backspaces(count, self._batch_size, **kwargs)
        else:
            self._backends.input.send_backspaces(count, **kwargs)

    def _type_text(self, tail: str) -> None:
        kwargs = {"extra_info": TAG, "release_modifiers": True}
        if self._batch_size is not None:
            self._backends.input.type_unicode(tail, self._batch_size, **kwargs)
        else:
            self._backends.input.type_unicode(tail, **kwargs)

    def _focus_reason(self) -> str:
        try:
            return target_holds_focus(self.ctx, self._backends)
        except Exception:
            log.warning("live insertion: the foreground check failed", exc_info=True)
            return "foreground check failed"

    def _foreground_title(self) -> str:
        window = self._backends.window
        title = getattr(window, "window_title", None)
        if title is None:
            return ""
        try:
            return str(title(int(window.foreground_hwnd())))[:80]
        except (OSError, TypeError, ValueError):
            return ""

    def _halt(self, reason: str) -> bool:
        self._stopped = True
        self._stop_reason = reason
        detail = reason
        if reason == "focus moved":
            detail = f"{reason} to {self._foreground_title()!r}"
        log.info(
            "live insertion stopped after %d characters: %s", len(self._inserted), detail
        )
        return False


__all__ = [
    "PLAIN_PROFILES",
    "TAG",
    "LiveSession",
    "common_prefix_len",
    "live_typing_allowed",
    "reconcile",
]
