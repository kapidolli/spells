"""In-process fake of the spells.win32.hook contract.

One FakeHook instance stands in for the spells.win32.hook module: it exposes the
same constants, the KeyEvent type, and every function of the contract, and adds
test helpers to feed key events, fire timers synchronously, and inspect what the
code under test sent or registered.

Behaviour that mirrors Windows:

- install_keyboard_hook returns a fresh handle each time and records it; the
  most recently installed callback is the live one.
- send_key records the call and, like SendInput, queues an injected KeyEvent
  (LLKHF_INJECTED set, extra_info carried through) that is delivered to the live
  callback after the current command returns, never re-entrantly. When no hook
  is installed the event is recorded as lost, which is what happens when Windows
  has removed the hook.
- drop_hook simulates Windows silently removing the hook: the callback is
  forgotten without an uninstall record, and uninstalling the stale handle
  raises OSError like a failed UnhookWindowsHookEx.
- run_message_loop registers the periodic timers and then blocks until the stop
  event is set; timers fire only when a test calls fire_timer, so every timer
  handler runs synchronously on the test thread.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

LLKHF_INJECTED = 0x10
VK_PROBE = 0xE8


@dataclass(frozen=True)
class KeyEvent:
    vk: int
    scan: int
    flags: int
    extra_info: int
    keydown: bool
    time_ms: int


class FakeClock:
    """A manually advanced monotonic clock in seconds."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def ms(self) -> int:
        return int(self.t * 1000)


class FakeHook:
    """Fake backend implementing the spells.win32.hook contract in-process."""

    LLKHF_INJECTED = LLKHF_INJECTED
    VK_PROBE = VK_PROBE
    KeyEvent = KeyEvent

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock if clock is not None else FakeClock()
        self.hooks: dict[int, Callable[[KeyEvent], bool]] = {}
        self.install_log: list[int] = []
        self.uninstall_log: list[int] = []
        self.sent: list[tuple[int, bool, int]] = []
        self.timers: dict[int, int] = {}
        self.timer_log: list[tuple[str, int, int | None]] = []
        # Ordered record of everything that crossed the fake: ("sent", vk, keydown,
        # extra_info), ("event", vk, keydown, swallowed) and ("lost", vk, keydown,
        # extra_info) for injected keys that found no hook installed.
        self.trace: list[tuple] = []
        self.keys_down: set[int] = set()
        self.priority_calls: list[int] = []
        # While positive, install_keyboard_hook raises (a failing SetWindowsHookEx)
        # with install_error as the message; a test changes it between attempts to
        # simulate the reason changing.
        self.install_failures_left = 0
        self.install_error = "SetWindowsHookEx failed: simulated"
        self.poll_ms: int | None = None
        self.installed = threading.Event()
        self.loop_started = threading.Event()
        self.loop_exited = threading.Event()
        self._on_timer: Callable[[int], None] | None = None
        self._pending: list[KeyEvent] = []
        self._next_handle = 1

    # Contract -------------------------------------------------------------

    def install_keyboard_hook(self, callback: Callable[[KeyEvent], bool]) -> int:
        if self.install_failures_left > 0:
            self.install_failures_left -= 1
            raise OSError(self.install_error)
        handle = self._next_handle
        self._next_handle += 1
        self.hooks[handle] = callback
        self.install_log.append(handle)
        self.installed.set()
        return handle

    def uninstall_keyboard_hook(self, handle: int) -> None:
        if handle not in self.hooks:
            raise OSError(f"UnhookWindowsHookEx failed: unknown hook handle {handle}")
        del self.hooks[handle]
        self.uninstall_log.append(handle)

    def run_message_loop(
        self,
        stop: threading.Event,
        timers: dict[int, int],
        on_timer: Callable[[int], None],
        poll_ms: int = 50,
    ) -> None:
        for timer_id, interval_ms in timers.items():
            self.set_timer(timer_id, interval_ms)
        self._on_timer = on_timer
        self.poll_ms = poll_ms
        self.loop_started.set()
        try:
            while not stop.wait(poll_ms / 1000):
                pass
        finally:
            self._on_timer = None
            self.loop_exited.set()

    def set_timer(self, id: int, interval_ms: int) -> None:
        self.timers[id] = interval_ms
        self.timer_log.append(("set", id, interval_ms))

    def kill_timer(self, id: int) -> None:
        self.timers.pop(id, None)
        self.timer_log.append(("kill", id, None))

    def send_key(self, vk: int, keydown: bool, extra_info: int = 0) -> None:
        self.sent.append((vk, keydown, extra_info))
        self.trace.append(("sent", vk, keydown, extra_info))
        self._pending.append(KeyEvent(vk, 0, LLKHF_INJECTED, extra_info, keydown, self.clock.ms()))

    def is_key_down(self, vk: int) -> bool:
        return vk in self.keys_down

    def set_current_thread_priority_highest(self) -> None:
        self.priority_calls.append(threading.get_ident())

    # Test helpers ---------------------------------------------------------

    @property
    def callback(self) -> Callable[[KeyEvent], bool] | None:
        """The live (most recently installed) hook callback, or None."""
        if not self.hooks:
            return None
        return self.hooks[max(self.hooks)]

    @property
    def loop_running(self) -> bool:
        return self._on_timer is not None

    def drop_hook(self) -> None:
        """Simulate Windows removing the hook without telling anyone."""
        self.hooks.clear()

    def event(
        self,
        vk: int,
        keydown: bool,
        *,
        injected: bool = False,
        extra_info: int = 0,
        scan: int = 0,
    ) -> KeyEvent:
        flags = LLKHF_INJECTED if injected else 0
        return KeyEvent(vk, scan, flags, extra_info, keydown, self.clock.ms())

    def feed(self, event: KeyEvent) -> bool:
        """Deliver one event to the live hook callback and return its swallow decision."""
        callback = self.callback
        if callback is None:
            raise RuntimeError("no keyboard hook installed")
        swallowed = callback(event)
        self.trace.append(("event", event.vk, event.keydown, swallowed))
        self._deliver_pending()
        return swallowed

    def press(self, vk: int, **kwargs) -> bool:
        self.keys_down.add(vk)
        return self.feed(self.event(vk, True, **kwargs))

    def release(self, vk: int, **kwargs) -> bool:
        self.keys_down.discard(vk)
        return self.feed(self.event(vk, False, **kwargs))

    def fire_timer(self, timer_id: int) -> None:
        """Fire a registered timer synchronously, then deliver any injected keys."""
        if self._on_timer is None:
            raise RuntimeError("message loop is not running")
        if timer_id not in self.timers:
            raise ValueError(f"timer {timer_id} is not registered")
        self._on_timer(timer_id)
        self._deliver_pending()

    def _deliver_pending(self) -> None:
        while self._pending:
            event = self._pending.pop(0)
            callback = self.callback
            if callback is None:
                self.trace.append(("lost", event.vk, event.keydown, event.extra_info))
                continue
            swallowed = callback(event)
            self.trace.append(("event", event.vk, event.keydown, swallowed))
