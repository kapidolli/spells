"""The KeyHook backend for Linux: read every keyboard through evdev (spells.platform.base).

Wayland has no global keyboard hook, so the hotkey thread reads /dev/input/event* directly,
which works the same on KDE, GNOME and X11. Reading those devices needs the Spells udev rule
(it grants the logged-in user access to keyboards, as desktops already do for uinput).

Differences from the Windows hook, all accepted:

- The events are observed, not intercepted, so a swallow decision cannot keep a key from the
  focused app. Ctrl+Super reaches it as a harmless chord, and Esc still reaches it after it
  cancels a dictation.
- There is no Start menu to mask and no hook timeout to probe for, so mask_keys is empty and
  supports_probe is false.
- Keyboards come and go (USB, Bluetooth, resume), so the device list is rescanned every few
  seconds and a device that fails a read is dropped until it shows up again.
"""

from __future__ import annotations

import errno
import fcntl
import glob
import logging
import os
import select
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from spells.linux import keycodes
from spells.linux.uinput import DEVICE_NAME, VirtualKeyboard

log = logging.getLogger(__name__)

DEVICE_GLOB = "/dev/input/event*"
EV_KEY = 1
KEY_MAX = 0x2FF
KEY_BYTES = KEY_MAX // 8 + 1
EVENT = struct.Struct("qqHHi")
READ_EVENTS = 64
RESCAN_S = 3.0
TYPING_KEYS = (keycodes.KEY_A, keycodes.KEY_Z, keycodes.KEY_SPACE)

IOC_READ = 2


def _ioc_read(nr: int, size: int) -> int:
    return (IOC_READ << 30) | (size << 16) | (ord("E") << 8) | nr


def eviocgbit(event_type: int, size: int) -> int:
    return _ioc_read(0x20 + event_type, size)


def eviocgkey(size: int) -> int:
    return _ioc_read(0x18, size)


def eviocgname(size: int) -> int:
    return _ioc_read(0x06, size)


@dataclass(frozen=True)
class KeyEvent:
    """One key event in the shape of spells.win32.hook.KeyEvent."""

    vk: int
    scan: int
    flags: int
    extra_info: int
    keydown: bool
    time_ms: int


def has_bit(bits: bytes, index: int) -> bool:
    byte = index // 8
    return byte < len(bits) and bool(bits[byte] & (1 << (index % 8)))


def is_keyboard(key_bits: bytes) -> bool:
    return all(has_bit(key_bits, code) for code in TYPING_KEYS)


def parse_events(data: bytes) -> list[tuple[int, int, int, int]]:
    """(seconds, microseconds, code, value) of every EV_KEY event in a read buffer."""
    events = []
    usable = len(data) - len(data) % EVENT.size
    for offset in range(0, usable, EVENT.size):
        sec, usec, kind, code, value = EVENT.unpack_from(data, offset)
        if kind == EV_KEY:
            events.append((sec, usec, code, value))
    return events


def event_for(sec: int, usec: int, code: int, value: int) -> KeyEvent:
    return KeyEvent(
        vk=keycodes.vk_for_code(code),
        scan=code,
        flags=0,
        extra_info=0,
        keydown=value != 0,
        time_ms=(sec * 1000 + usec // 1000) & 0xFFFFFFFF,
    )


def _read_name(fd: int) -> str:
    buffer = bytearray(256)
    try:
        fcntl.ioctl(fd, eviocgname(len(buffer)), buffer)
    except OSError:
        return ""
    return bytes(buffer).split(b"\0", 1)[0].decode("utf-8", "replace")


def _key_bits(fd: int) -> bytes:
    buffer = bytearray(KEY_BYTES)
    fcntl.ioctl(fd, eviocgbit(EV_KEY, len(buffer)), buffer)
    return bytes(buffer)


class NoKeyboardAccess(PermissionError):
    pass


class EvdevHook:
    """KeyHook over evdev. Everything but send_key and is_key_down runs on the hotkey thread."""

    mask_keys: tuple[tuple[int, bool, int], ...] = ()
    supports_probe = False

    def __init__(
        self,
        keyboard: VirtualKeyboard,
        *,
        device_glob: str = DEVICE_GLOB,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._keyboard = keyboard
        self._glob = device_glob
        self._clock = clock
        self._devices: dict[str, int] = {}
        self._denied: set[str] = set()
        self._ignored: set[str] = set()
        self._callback: Callable[[Any], bool] | None = None
        self._timers: dict[int, tuple[int, float]] = {}
        self._lock = threading.Lock()
        self._next_rescan = 0.0
        self._handle = 0

    def install_keyboard_hook(self, callback: Callable[[Any], bool]) -> int:
        self._scan()
        if not self._devices:
            if self._denied:
                raise NoKeyboardAccess(
                    "Spells cannot read the keyboard. Install the Spells udev rule "
                    "(build/linux/install.sh does it) and sign in again."
                )
            raise NoKeyboardAccess("Spells found no keyboard to listen to.")
        self._callback = callback
        self._handle += 1
        return self._handle

    def uninstall_keyboard_hook(self, handle: int) -> None:
        self._callback = None
        with self._lock:
            for fd in self._devices.values():
                os.close(fd)
            self._devices.clear()
        self._ignored.clear()
        self._denied.clear()

    def send_key(self, vk: int, keydown: bool, extra_info: int = 0) -> None:
        for code in keycodes.codes_for_vk(vk)[:1]:
            self._keyboard.key(code, keydown)

    def is_key_down(self, vk: int) -> bool:
        codes = keycodes.codes_for_vk(vk)
        if not codes:
            return False
        with self._lock:
            fds = list(self._devices.values())
        for fd in fds:
            buffer = bytearray(KEY_BYTES)
            try:
                fcntl.ioctl(fd, eviocgkey(len(buffer)), buffer)
            except OSError:
                continue
            if any(has_bit(bytes(buffer), code) for code in codes):
                return True
        return False

    def set_timer(self, timer_id: int, interval_ms: int) -> None:
        self._timers[timer_id] = (interval_ms, self._clock() + interval_ms / 1000)

    def kill_timer(self, timer_id: int) -> None:
        self._timers.pop(timer_id, None)

    def set_current_thread_priority_highest(self) -> None:
        return None

    def chord_available(self, modifiers: int, vk: int) -> bool:
        return True

    def run_message_loop(
        self,
        stop: threading.Event,
        timers: dict[int, int],
        on_timer: Callable[[int], None] | None,
        poll_ms: int = 50,
    ) -> None:
        for timer_id, interval in timers.items():
            self.set_timer(timer_id, interval)
        while not stop.is_set():
            now = self._clock()
            if self._callback is not None and now >= self._next_rescan:
                self._scan()
            with self._lock:
                by_fd = {fd: path for path, fd in self._devices.items()}
            wait_s = poll_ms / 1000
            if self._timers:
                due = min(deadline for _, deadline in self._timers.values())
                wait_s = max(0.0, min(wait_s, due - now))
            if by_fd:
                try:
                    ready, _, _ = select.select(list(by_fd), [], [], wait_s)
                except (OSError, ValueError):
                    ready = []
                for fd in ready:
                    self._read(fd, by_fd[fd])
            else:
                stop.wait(wait_s)
            self._fire_timers(on_timer)

    def _fire_timers(self, on_timer: Callable[[int], None] | None) -> None:
        now = self._clock()
        for timer_id, (interval, deadline) in list(self._timers.items()):
            if now < deadline:
                continue
            if timer_id in self._timers:
                self._timers[timer_id] = (interval, now + interval / 1000)
            if on_timer is not None:
                on_timer(timer_id)

    def _read(self, fd: int, path: str) -> None:
        try:
            data = os.read(fd, EVENT.size * READ_EVENTS)
        except BlockingIOError:
            return
        except OSError as exc:
            if exc.errno not in (errno.ENODEV, errno.EIO, errno.EBADF):
                log.warning("Reading %s failed: %s", path, exc)
            self._drop(path)
            return
        callback = self._callback
        if callback is None:
            return
        for sec, usec, code, value in parse_events(data):
            callback(event_for(sec, usec, code, value))

    def _drop(self, path: str) -> None:
        with self._lock:
            fd = self._devices.pop(path, None)
        if fd is not None:
            log.info("Keyboard %s went away", path)
            try:
                os.close(fd)
            except OSError:
                pass

    def _scan(self) -> None:
        self._next_rescan = self._clock() + RESCAN_S
        present = set(glob.glob(self._glob))
        for path in list(self._devices):
            if path not in present:
                self._drop(path)
        self._ignored &= present
        self._denied &= present
        for path in sorted(present - set(self._devices) - self._ignored):
            self._open(path)

    def _open(self, path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError:
            if path not in self._denied:
                self._denied.add(path)
            return
        except OSError:
            return
        try:
            bits = _key_bits(fd)
        except OSError:
            os.close(fd)
            return
        name = _read_name(fd)
        if not is_keyboard(bits) or name == DEVICE_NAME:
            os.close(fd)
            self._ignored.add(path)
            return
        self._denied.discard(path)
        with self._lock:
            self._devices[path] = fd
        log.info("Listening to keyboard %s (%s)", path, name)
