"""Hotkey chord state machine and the hotkey thread (spec 5.1, 6 steps 1 to 3, 13).

Batch 2 decisions that shape this module: V1-6 (the state machine runs on the
hotkey thread and the hook callback consults it synchronously, never blocking),
V1-8 (signal set, tap window measured from release, the press that ends a latch
is consumed, Esc cancels during the tap window too, mask injection on the Win
key-up), V2-8 (per-language chords share the state machine) and V3-F3 (the probe
is sent from a timer and checked by a second timer; a missed check reinstalls).

Batch 3 decisions: B3-22 (every dictation carries an integer id, the callbacks
carry it, and `end_recording(id)` replaces `set_recording(bool)`), B3-23 (timing
uses time.perf_counter), B3-25 (the probe is recognized by vk, the injected flag
and the private tag together) and B3-26 (a failed hook install is counted,
reported through `on_error` and retried every tick; the thread never exits).

The Win32 side is the spells.win32.hook contract. It is passed in as a backend
object so unit tests run against tests/unit/fake_hook.py; the real module is
imported only when no backend is given.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from spells.models import Chord
from spells.vk import GENERIC_OF, SPECIFIC_OF, VK_LWIN, VK_RWIN

log = logging.getLogger(__name__)

# Private dwExtraInfo tags. PROBE_TAG marks the liveness probe (spec 13); MASK_TAG
# marks every key Spells itself injects for the target app, which is the Start menu
# mask key and the synthetic key-ups of B3-11. The probe and the mask key share
# VK 0xE8, so the tag is what tells them apart in the hook callback.
PROBE_TAG = 0x55545452
MASK_TAG = 0x5554544D
MASK_VK = 0xE8
VK_PROBE = 0xE8
VK_ESCAPE = 0x1B
LLKHF_INJECTED = 0x10

# The chord keys themselves come from spells.vk, shared with config's chord
# validation so the two cannot disagree about what counts as a modifier.
WIN_VKS = frozenset({VK_LWIN, VK_RWIN})
# Every modifier the hook can report: the three generic codes, their six
# side-specific codes, and the two Win keys, which have no generic code.
MODIFIER_VKS = frozenset(SPECIFIC_OF) | frozenset(GENERIC_OF) | WIN_VKS

MASK_INJECT = ((MASK_VK, True, MASK_TAG), (MASK_VK, False, MASK_TAG))

# Timer ids used with the backend's message loop.
TIMER_TICK = 1
TIMER_PROBE = 2
TIMER_PROBE_CHECK = 3
TICK_MS = 50
SWITCH_INTERVAL_S = 0.001


class KeyEventLike(Protocol):
    """The spells.win32.hook.KeyEvent shape, as seen by the state machine."""

    vk: int
    scan: int
    flags: int
    extra_info: int
    keydown: bool
    time_ms: int


@dataclass(frozen=True)
class HotkeyCallbacks:
    """The hotkey signals of V1-8, carrying the dictation id of B3-22.

    Each is invoked on the hotkey thread inside the hook procedure, so each must
    only enqueue and return (B3-27). The id is assigned when `pressed` fires and
    identifies that dictation for the rest of its life.
    """

    pressed: Callable[[Chord, int], None]
    released: Callable[[Chord, int], None]
    latched: Callable[[Chord, int], None]
    discarded: Callable[[int], None]
    cancelled: Callable[[int], None]


@dataclass(frozen=True)
class Decision:
    """What the hook callback does with the current event.

    inject lists (vk, keydown, extra_info) keys to send before the current event
    is passed on; swallow says whether the event itself is dropped.
    """

    swallow: bool
    inject: tuple[tuple[int, bool, int], ...] = ()


class ChordState(Enum):
    IDLE = "idle"
    HELD = "held"
    TAP_WINDOW = "tap_window"
    LATCHED = "latched"


class ChordStateMachine:
    """Hold, tap window, double-tap latch, Esc cancel and Start menu masking (spec 6).

    Transitions (callbacks in brackets):

    - IDLE, chord completed by a fresh key-down: [pressed] -> HELD
    - HELD, any chord key up before tap_max_s: -> TAP_WINDOW (silent)
    - HELD, any chord key up at or after tap_max_s: [released] -> IDLE
    - TAP_WINDOW, same chord completed within tap_window_s of the release: [latched] -> LATCHED
    - TAP_WINDOW, window expired (on_tick or the next key event): [discarded] -> IDLE
    - TAP_WINDOW, a different chord completed: [discarded], [pressed other] -> HELD
    - LATCHED, any chord completed: [released active] -> IDLE; that press is consumed
    - HELD, TAP_WINDOW or LATCHED, Esc key-down: [cancelled] -> IDLE
    - IDLE with set_writing(id), Esc key-down: [cancelled id], writing cleared (spec 8.5:
      the recording is long over while the writing model works, so Esc has to reach it
      through the id the pipeline armed rather than through the chord state)
    - any state, end_recording(current id): -> IDLE, no callback (the pipeline ended
      the recording itself, for example at the 10-minute cap)

    Timing uses the injected clock, not KeyEvent.time_ms: the callback runs promptly
    (spec 5.1) and on_tick has no event to read a time from. The default clock is
    time.perf_counter, because GetTickCount-based clocks step in 15.6 ms (B3-23).
    """

    def __init__(
        self,
        chords: Iterable[Chord],
        callbacks: HotkeyCallbacks,
        now: Callable[[], float] = time.perf_counter,
        tap_max_s: float = 0.25,
        tap_window_s: float = 0.4,
        on_exception: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._chords: tuple[Chord, ...] = tuple(chords)
        self._callbacks = callbacks
        self._now = now
        # Integer milliseconds: the spec states these limits in ms, the hook is
        # ms-granular, and float subtraction noise must not move a boundary.
        self._tap_max_ms = round(tap_max_s * 1000)
        self._tap_window_ms = round(tap_window_s * 1000)
        self._on_exception = on_exception
        self._state = ChordState.IDLE
        self._active: Chord | None = None
        self._dictation_id: int | None = None
        self._last_id = 0
        self._press_ms = 0
        self._release_ms = 0
        self._down: set[int] = set()
        self._swallowed: set[int] = set()
        self._released: set[int] = set()
        self._mask_pending = False
        self._probe_seen = False
        self._writing_id: int | None = None
        self.callback_exceptions = 0

    # Public state ----------------------------------------------------------

    @property
    def state(self) -> ChordState:
        return self._state

    @property
    def active_chord(self) -> Chord | None:
        return self._active

    @property
    def dictation_id(self) -> int | None:
        """The id of the dictation in flight, or None when there is none (B3-22)."""
        return self._dictation_id

    @property
    def chords(self) -> tuple[Chord, ...]:
        return self._chords

    @property
    def recording(self) -> bool:
        """True while a chord is held, tapped, or latched."""
        return self._state is not ChordState.IDLE

    @property
    def probe_seen(self) -> bool:
        return self._probe_seen

    def consume_probe(self) -> bool:
        seen, self._probe_seen = self._probe_seen, False
        return seen

    def set_chords(self, chords: Iterable[Chord]) -> None:
        """Swap the chord set. An active dictation keeps its chord until it ends."""
        self._chords = tuple(chords)

    def set_writing(self, dictation_id: int | None) -> None:
        """Name the composition the writing model is working on, or None when it is done.

        Recording is over by then, so the Esc rule of spec 6 would no longer see anything to
        cancel. While an id is set, an Esc key-down is swallowed and reported through
        `cancelled` with that id, which is what stops the request and delivers nothing
        (spec 8.5). One plain assignment, so any thread may make it.
        """
        self._writing_id = dictation_id

    @property
    def writing_id(self) -> int | None:
        return self._writing_id

    def end_recording(self, dictation_id: int) -> None:
        """End the dictation with this id, in whatever state it is (B3-22).

        The pipeline calls this when it ended the recording itself (the 10-minute
        cap of V1-8, an engine failure), so no callback fires: the pipeline already
        knows. A stale id is ignored, so a late end of a finished dictation can
        never kill the one that replaced it. Chord keys that are still physically
        down stay swallowed until their own key-up, and a pending Start menu mask
        is still injected on the Win key-up.

        Call it on the hotkey thread; HotkeyThread.end_recording queues it there.
        """
        if self._dictation_id is None or dictation_id != self._dictation_id:
            return
        self._to_idle()

    def resync(self, is_key_down: Callable[[int], bool]) -> None:
        """Rebuild the physical key state after a hook (re)install.

        Key-ups missed while the hook was dead would otherwise leave keys stuck
        down and let a partial chord fire. No callbacks are emitted.
        """
        candidates = set(self._down)
        for chord in self._chords:
            for key in chord.keys:
                candidates.update(SPECIFIC_OF.get(key, (key,)))
        self._down = {vk for vk in candidates if is_key_down(vk)}
        self._swallowed &= self._down
        self._released &= self._down

    # Event handling ------------------------------------------------------------

    def on_tick(self) -> None:
        """Timer entry point: evaluates tap-window expiry."""
        self._expire_tap_window()

    def on_key(self, event: KeyEventLike) -> Decision:
        vk = event.vk
        extra_info = event.extra_info
        # B3-25: all three of vk, the injected flag and the private tag must match,
        # so no key a real keyboard can produce is ever taken for our own probe.
        if vk == VK_PROBE and extra_info == PROBE_TAG and event.flags & LLKHF_INJECTED:
            self._probe_seen = True
            return Decision(swallow=True)
        if extra_info == MASK_TAG:
            if not event.keydown:
                self._note_synthetic_release(vk)
            return Decision(swallow=False)
        self._expire_tap_window()
        if event.keydown:
            return self._on_keydown(vk)
        return self._on_keyup(vk)

    def _note_synthetic_release(self, vk: int) -> None:
        if vk not in MODIFIER_VKS:
            return
        family = _family(vk)
        held = family & self._down
        self._released |= held

    def _on_keydown(self, vk: int) -> Decision:
        fresh = vk not in self._down
        self._down.add(vk)
        if not fresh:
            if vk in self._released:
                return Decision(swallow=True)
            return Decision(swallow=vk in self._swallowed)
        self._released.discard(vk)
        if vk == VK_ESCAPE:
            if self.recording:
                self._cancel()
                self._swallowed.add(vk)
                return Decision(swallow=True)
            if self._writing_id is not None:
                self._cancel_writing()
                self._swallowed.add(vk)
                return Decision(swallow=True)
            return Decision(swallow=False)
        chord = self._completed_chord(vk)
        if chord is None:
            return Decision(swallow=False)
        self._on_chord_completed(chord)
        inject = self._swallow_chord_keys(chord, vk)
        if any(key in WIN_VKS for key in chord.keys):
            self._mask_pending = True
        return Decision(swallow=vk in self._swallowed, inject=inject)

    def _swallow_chord_keys(self, chord: Chord, completing_vk: int) -> tuple[tuple, ...]:
        """Swallow every non-modifier chord key that is down, to its own key-up.

        Not only the key that completed the chord: one pressed before the modifiers
        would otherwise keep feeding auto-repeats to the target app for the whole
        dictation. That key's own key-down already reached the app, though, so
        swallowing the rest of its life would leave a stuck key there (B3-11).
        Each such key therefore gets a synthetic key-up first, injected with our own
        tag so the callback passes it straight through. The completing key needs
        none: its key-down is the one being swallowed right now.
        """
        inject: list[tuple[int, bool, int]] = []
        for key in chord.keys:
            if key in MODIFIER_VKS or key not in self._down:
                continue
            if key != completing_vk and key not in self._swallowed:
                inject.append((key, False, MASK_TAG))
            self._swallowed.add(key)
        return tuple(inject)

    def _on_keyup(self, vk: int) -> Decision:
        self._down.discard(vk)
        released = vk in self._released
        self._released.discard(vk)
        swallow = released or vk in self._swallowed
        self._swallowed.discard(vk)
        inject: tuple[tuple[int, bool, int], ...] = ()
        if vk in WIN_VKS and self._mask_pending:
            # A chord swallowed the keys pressed during this Win hold, so Windows
            # would treat the key-up as a lone Win tap and open the Start menu.
            self._mask_pending = False
            if not released:
                inject = MASK_INJECT
        if (
            self._state is ChordState.HELD
            and self._active is not None
            and _chord_has_key(self._active, vk)
        ):
            if self._now_ms() - self._press_ms < self._tap_max_ms:
                self._state = ChordState.TAP_WINDOW
                self._release_ms = self._now_ms()
            else:
                self._release()
        return Decision(swallow=swallow, inject=inject)

    def _on_chord_completed(self, chord: Chord) -> None:
        state = self._state
        if state is ChordState.IDLE:
            self._start(chord)
        elif state is ChordState.TAP_WINDOW:
            if chord == self._active:
                self._state = ChordState.LATCHED
                self._emit(self._callbacks.latched, chord, self._dictation_id)
            else:
                self._discard()
                self._start(chord)
        elif state is ChordState.LATCHED:
            # This press only ends the latch; it is consumed, its release is ignored.
            self._release()
        # HELD: another chord while one is held is ignored.

    def _expire_tap_window(self) -> None:
        if (
            self._state is ChordState.TAP_WINDOW
            and self._now_ms() - self._release_ms >= self._tap_window_ms
        ):
            self._discard()

    def _start(self, chord: Chord) -> None:
        self._last_id += 1
        self._dictation_id = self._last_id
        self._active = chord
        self._press_ms = self._now_ms()
        self._state = ChordState.HELD
        self._emit(self._callbacks.pressed, chord, self._dictation_id)

    def _release(self) -> None:
        chord, dictation_id = self._active, self._dictation_id
        self._to_idle()
        self._emit(self._callbacks.released, chord, dictation_id)

    def _discard(self) -> None:
        dictation_id = self._dictation_id
        self._to_idle()
        self._emit(self._callbacks.discarded, dictation_id)

    def _cancel(self) -> None:
        dictation_id = self._dictation_id
        self._to_idle()
        self._emit(self._callbacks.cancelled, dictation_id)

    def _cancel_writing(self) -> None:
        dictation_id, self._writing_id = self._writing_id, None
        self._emit(self._callbacks.cancelled, dictation_id)

    def _to_idle(self) -> None:
        self._state = ChordState.IDLE
        self._active = None
        self._dictation_id = None

    def _emit(self, callback: Callable[..., None], *args: object) -> None:
        try:
            callback(*args)
        except Exception as exc:
            self.callback_exceptions += 1
            log.exception("hotkey callback %r failed", callback)
            if self._on_exception is not None:
                try:
                    self._on_exception(exc)
                except Exception:
                    log.exception("hotkey on_exception handler failed")

    def _now_ms(self) -> int:
        return round(self._now() * 1000)

    def _completed_chord(self, vk: int) -> Chord | None:
        """The chord completed by a fresh key-down of vk; the longest wins if several.

        Two chords of equal length can complete on the same event without breaking
        the config rule that forbids subsets (Ctrl+D and Alt+D, with both modifiers
        held, complete together on D). The tie-break is configuration order: the
        first such chord in the configured sequence wins, so the outcome is stable
        and the user can predict it from the order shown in settings.
        """
        best: Chord | None = None
        for chord in self._chords:
            if not _chord_has_key(chord, vk):
                continue
            if not all(self._is_down(key) for key in chord.keys):
                continue
            if best is None or len(chord.keys) > len(best.keys):
                best = chord
        return best

    def _is_down(self, key: int) -> bool:
        if key in self._down:
            return True
        return any(GENERIC_OF.get(vk) == key for vk in self._down)


def _family(vk: int) -> frozenset[int]:
    generic = GENERIC_OF.get(vk, vk)
    return frozenset({vk, generic, *SPECIFIC_OF.get(generic, ())})


def _chord_has_key(chord: Chord, vk: int) -> bool:
    if vk in chord.keys:
        return True
    generic = GENERIC_OF.get(vk)
    return generic is not None and generic in chord.keys


@dataclass
class HotkeyStats:
    probes_sent: int = 0
    probes_missed: int = 0
    reinstalls: int = 0
    hook_exceptions: int = 0
    install_failures: int = 0
    last_error: str = ""


class HotkeyThread(threading.Thread):
    """Owns the keyboard hook, its message loop, and the liveness probe (spec 5.1, 13).

    Everything the hook callback and the timers do runs on this thread and never
    blocks (V1-6, V3-F3); end_recording, update_chords, request_reinstall and stop
    may be called from any thread.

    A hook install that Windows refuses never kills the thread (B3-26): it is
    counted, passed to `on_error` for a tray warning, and retried on every 50 ms
    tick until it succeeds.
    """

    def __init__(
        self,
        chords: Iterable[Chord],
        callbacks: HotkeyCallbacks,
        hook_backend=None,
        probe_interval_s: float = 15.0,
        probe_check_s: float = 0.1,
        now: Callable[[], float] = time.perf_counter,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(name="spells-hotkey", daemon=True)
        if hook_backend is None:
            # Production path only: tests always pass a fake backend.
            from spells.win32 import hook as hook_backend
        self._backend = hook_backend
        self._probe_interval_ms = max(1, round(probe_interval_s * 1000))
        self._probe_check_ms = max(1, round(probe_check_s * 1000))
        self._on_error = on_error
        self.stats = HotkeyStats()
        self._sm = ChordStateMachine(chords, callbacks, now, on_exception=self._count_exception)
        self._stop_event = threading.Event()
        self._reinstall_requested = threading.Event()
        self._lock = threading.Lock()
        self._pending_ends: list[int] = []
        self._handle: int | None = None
        self._installed_once = False

    # Cross-thread API -------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._sm.recording

    @property
    def chords(self) -> tuple[Chord, ...]:
        """The chord set the hook currently matches (a tuple swapped whole, so any thread may read)."""
        return self._sm.chords

    def end_recording(self, dictation_id: int) -> None:
        """End that dictation (B3-22). Safe from any thread; applied at the next tick.

        Because it is applied up to one tick late, a callback for that dictation can
        still arrive in the meantime: a key-up crossing the request fires `released`
        with the id the pipeline has already ended. The pipeline therefore ignores
        every callback whose id is not the one it currently considers in flight,
        which is exactly the rule the state machine applies to this call.

        After stop() the request is dropped: no tick will ever run to apply it, and
        a caller that keeps ending dictations must not grow the queue forever.
        """
        if self._stop_event.is_set():
            return
        with self._lock:
            self._pending_ends.append(dictation_id)

    def update_chords(self, chords: Iterable[Chord]) -> None:
        self._sm.set_chords(chords)

    def set_writing(self, dictation_id: int | None) -> None:
        """Arm or disarm the Esc cancel for a composition (spec 8.5). Safe from any thread."""
        self._sm.set_writing(dictation_id)

    @property
    def writing_id(self) -> int | None:
        return self._sm.writing_id

    def request_reinstall(self) -> None:
        """Reinstall the hook on the next tick (resume from sleep, session unlock)."""
        self._reinstall_requested.set()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)

    # Thread body -----------------------------------------------------------------

    def run(self) -> None:
        sys.setswitchinterval(SWITCH_INTERVAL_S)
        try:
            self._backend.set_current_thread_priority_highest()
        except Exception:
            log.exception("could not raise the hotkey thread priority")
        # A refused install is not fatal: the message loop starts anyway and the tick
        # retries until the hook takes, so the thread outlives a bad moment (B3-26).
        self._install()
        timers = {TIMER_TICK: TICK_MS, TIMER_PROBE: self._probe_interval_ms}
        try:
            self._backend.run_message_loop(self._stop_event, timers, self._on_timer)
        finally:
            self._uninstall()

    def hook_callback(self, event: KeyEventLike) -> bool:
        """The installed hook callback. Never raises; True swallows the event."""
        try:
            decision = self._sm.on_key(event)
        except Exception:
            self.stats.hook_exceptions += 1
            log.exception("hook callback failed")
            return False
        for vk, keydown, extra_info in decision.inject:
            try:
                self._backend.send_key(vk, keydown, extra_info)
            except Exception:
                self.stats.hook_exceptions += 1
                log.exception("mask injection failed")
        return bool(decision.swallow)

    def _on_timer(self, timer_id: int) -> None:
        try:
            if timer_id == TIMER_TICK:
                if self._reinstall_requested.is_set():
                    self._reinstall_requested.clear()
                    self._reinstall()
                elif self._handle is None:
                    # An install Windows refused, retried until it takes (B3-26).
                    self._install()
                self._apply_pending_ends()
                self._sm.on_tick()
            elif timer_id == TIMER_PROBE:
                self._send_probe()
            elif timer_id == TIMER_PROBE_CHECK:
                self._check_probe()
        except Exception:
            self.stats.hook_exceptions += 1
            log.exception("hotkey timer %d failed", timer_id)

    def _send_probe(self) -> None:
        if self._sm.recording:
            return
        self._sm.consume_probe()  # only this probe may satisfy the check
        self._backend.send_key(VK_PROBE, True, PROBE_TAG)
        self._backend.send_key(VK_PROBE, False, PROBE_TAG)
        self.stats.probes_sent += 1
        self._backend.set_timer(TIMER_PROBE_CHECK, self._probe_check_ms)

    def _check_probe(self) -> None:
        self._backend.kill_timer(TIMER_PROBE_CHECK)
        if self._sm.consume_probe():
            return
        self.stats.probes_missed += 1
        log.warning("keyboard hook probe missed; reinstalling the hook")
        self._reinstall()

    def _apply_pending_ends(self) -> None:
        with self._lock:
            pending, self._pending_ends = self._pending_ends, []
        for dictation_id in pending:
            self._sm.end_recording(dictation_id)

    def _install(self) -> bool:
        """Install the hook. A refusal is counted and reported, never raised (B3-26)."""
        try:
            self._handle = self._backend.install_keyboard_hook(self.hook_callback)
        except Exception as exc:
            self._handle = None
            self.stats.install_failures += 1
            message = f"keyboard hook install failed: {exc}"
            log.exception("keyboard hook install failed")
            if message != self.stats.last_error:
                # The retry runs every 50 ms, so report one tray warning per distinct
                # error rather than twenty a second for as long as the hook is refused.
                self._report_error(message)
            self.stats.last_error = message
            return False
        if self._installed_once:
            self.stats.reinstalls += 1
        self._installed_once = True
        if self.stats.last_error:
            # The hook is back after a refusal: an empty message clears the tray warning.
            self.stats.last_error = ""
            self._report_error("")
        self._sm.resync(self._backend.is_key_down)
        return True

    def _report_error(self, message: str) -> None:
        if self._on_error is None:
            return
        try:
            self._on_error(message)
        except Exception:
            log.exception("hotkey on_error handler failed")

    def _uninstall(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            self._backend.uninstall_keyboard_hook(handle)
        except Exception:
            # Expected when Windows already removed the hook.
            log.debug("uninstall of hook %r failed", handle, exc_info=True)

    def _reinstall(self) -> None:
        """Replace the hook. A failure leaves the retry to the next tick (B3-26)."""
        self._uninstall()
        self._install()

    def _count_exception(self, _exc: BaseException) -> None:
        self.stats.hook_exceptions += 1
