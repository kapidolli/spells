"""A virtual keyboard on /dev/uinput, for the paste and copy shortcuts and Backspace.

Wayland gives apps no way to send keys to other apps, but a uinput device is a keyboard like
any other to the compositor. Desktops grant the logged-in user access to /dev/uinput through
the udev uaccess tag (Steam's controller rules do it on most systems; the Spells udev rule
does it otherwise). The compositor needs a moment to pick a new device up, so the device is
created once, on first use, and kept until the app exits.
"""

from __future__ import annotations

import fcntl
import logging
import os
import struct
import threading
import time

from spells.linux import keycodes

log = logging.getLogger(__name__)

DEVICE_PATH = "/dev/uinput"
DEVICE_NAME = "Spells virtual keyboard"

EV_SYN = 0
EV_KEY = 1
SYN_REPORT = 0
BUS_VIRTUAL = 0x06

UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502
UI_SET_EVBIT = 0x40045564
UI_SET_KEYBIT = 0x40045565
UI_DEV_SETUP = 0x405C5503

EVENT = struct.Struct("qqHHi")
SETUP = struct.Struct("HHHH80sI")

SETTLE_S = 0.25
KEY_GAP_S = 0.004


class VirtualKeyboard:
    def __init__(self, path: str = DEVICE_PATH) -> None:
        self._path = path
        self._fd: int | None = None
        self._lock = threading.Lock()

    def _open(self) -> int:
        if self._fd is not None:
            return self._fd
        fd = os.open(self._path, os.O_WRONLY | os.O_NONBLOCK)
        try:
            fcntl.ioctl(fd, UI_SET_EVBIT, EV_KEY)
            for code in sorted(set(keycodes.CODE_TO_VK)):
                fcntl.ioctl(fd, UI_SET_KEYBIT, code)
            setup = SETUP.pack(BUS_VIRTUAL, 0x5350, 0x4C53, 1, DEVICE_NAME.encode(), 0)
            fcntl.ioctl(fd, UI_DEV_SETUP, setup)
            fcntl.ioctl(fd, UI_DEV_CREATE)
        except OSError:
            os.close(fd)
            raise
        self._fd = fd
        log.info("Virtual keyboard created on %s", self._path)
        time.sleep(SETTLE_S)
        return fd

    def prepare(self) -> bool:
        """Create the device now so the first paste does not wait for it; False when it cannot."""
        try:
            with self._lock:
                self._open()
        except OSError as exc:
            log.warning("The virtual keyboard is unavailable: %s", exc)
            return False
        return True

    def _emit(self, fd: int, code: int, value: int) -> None:
        os.write(fd, EVENT.pack(0, 0, EV_KEY, code, value))
        os.write(fd, EVENT.pack(0, 0, EV_SYN, SYN_REPORT, 0))

    def key(self, code: int, down: bool) -> None:
        with self._lock:
            self._emit(self._open(), code, 1 if down else 0)

    def tap(self, *codes: int, repeat: int = 1) -> None:
        """Press the codes in order and release them in reverse, `repeat` times."""
        with self._lock:
            fd = self._open()
            for _ in range(repeat):
                for code in codes:
                    self._emit(fd, code, 1)
                    time.sleep(KEY_GAP_S)
                for code in reversed(codes):
                    self._emit(fd, code, 0)
                    time.sleep(KEY_GAP_S)

    def close(self) -> None:
        with self._lock:
            fd, self._fd = self._fd, None
            if fd is None:
                return
            try:
                fcntl.ioctl(fd, UI_DEV_DESTROY)
            except OSError:
                log.debug("uinput destroy failed", exc_info=True)
            os.close(fd)
