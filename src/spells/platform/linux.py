"""The Linux platform: evdev hotkeys, uinput shortcuts, wl-clipboard and KWin focus.

Built for KDE Plasma on Wayland first. The hotkey and the paste work on any desktop that
grants the Spells udev rule; app profiles, window titles and the "focus moved" check need
KWin. Text is always delivered through the clipboard: "type" pastes too, because a virtual
keyboard can only send the keys of the current layout, and terminals get Ctrl+Shift+V.

Qt runs through XWayland when the session is Wayland, because Wayland lets no app place its
own window or keep it from taking focus, which the pill needs. SPELLS_QT_PLATFORM overrides
that choice.
"""

from __future__ import annotations

import atexit
import functools
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from spells.hotkey import HotkeyThread
from spells.linux import clipboard as linux_clipboard
from spells.linux import desktop, keycodes
from spells.linux import instance as linux_instance
from spells.linux import process as linux_process
from spells.linux.evdev import EvdevHook
from spells.linux.kwin import FocusTracker
from spells.linux.uinput import VirtualKeyboard
from spells.models import CpuPlan
from spells.platform.base import Capabilities, MessageHandlers, Platform
from spells.platform.keys_linux import LinuxKeys
from spells.platform.stub import StubAppearance, StubUpdater

log = logging.getLogger(__name__)

QT_PLATFORM_ENV = "SPELLS_QT_PLATFORM"
TERMINALS = frozenset(
    {
        "org.kde.konsole",
        "konsole",
        "kitty",
        "alacritty",
        "org.wezfurlong.wezterm",
        "foot",
        "footclient",
        "com.mitchellh.ghostty",
        "org.gnome.terminal",
        "gnome-terminal-server",
        "org.gnome.ptyxis",
        "xterm",
        "tilix",
        "terminator",
        "org.kde.yakuake",
    }
)
TYPE_RESTORE_DELAY_S = 0.3
PASTE_SETTLE_S = 0.02


def is_terminal(app: str) -> bool:
    return app.casefold() in TERMINALS


class LinuxFocus:
    def __init__(self, tracker: FocusTracker) -> None:
        self._tracker = tracker

    def foreground(self) -> int:
        return self._tracker.foreground()

    def app_name(self, window: int) -> str:
        return self._tracker.info(window).app

    def title(self, window: int) -> str:
        return self._tracker.info(window).title

    def is_elevated(self, window: int) -> bool:
        return False


class LinuxKeyboard:
    def __init__(self, keyboard: VirtualKeyboard, focus: LinuxFocus, clipboard: Any) -> None:
        self._keyboard = keyboard
        self._focus = focus
        self._clipboard = clipboard

    def _in_terminal(self) -> bool:
        return is_terminal(self._focus.app_name(self._focus.foreground()))

    def type_unicode(
        self, text: str, *, extra_info: int = 0, release_modifiers: bool = False
    ) -> None:
        snap = self._clipboard.snapshot()
        sequence = self._clipboard.set_text(text)
        self.send_paste()
        timer = threading.Timer(
            TYPE_RESTORE_DELAY_S, self._restore, args=(snap, sequence)
        )
        timer.daemon = True
        timer.start()

    def _restore(self, snap: Any, sequence: int) -> None:
        try:
            self._clipboard.restore(snap, sequence)
        except Exception:
            log.warning("The clipboard could not be restored after typing", exc_info=True)

    def send_backspaces(
        self, count: int, *, extra_info: int = 0, release_modifiers: bool = False
    ) -> None:
        if count > 0:
            self._keyboard.tap(keycodes.KEY_BACKSPACE, repeat=count)

    def send_paste(self) -> None:
        time.sleep(PASTE_SETTLE_S)
        if self._in_terminal():
            self._keyboard.tap(keycodes.KEY_LEFTCTRL, keycodes.KEY_LEFTSHIFT, keycodes.KEY_V)
        else:
            self._keyboard.tap(keycodes.KEY_LEFTCTRL, keycodes.KEY_V)

    def send_copy(self) -> None:
        if self._in_terminal():
            self._keyboard.tap(keycodes.KEY_LEFTCTRL, keycodes.KEY_LEFTSHIFT, keycodes.KEY_C)
        else:
            self._keyboard.tap(keycodes.KEY_LEFTCTRL, keycodes.KEY_C)

    def release_held_modifiers(self) -> list[int]:
        return []


class LinuxOverlay:
    def __init__(self, tracker: FocusTracker) -> None:
        self._tracker = tracker

    def monitor_rect_for(self, window: int) -> tuple[int, int, int, int]:
        output = self._tracker.info(window).output
        if output is not None:
            return output
        from PySide6 import QtGui

        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return (0, 0, 0, 0)
        rect = screen.geometry()
        return (rect.left(), rect.top(), rect.right() + 1, rect.bottom() + 1)

    def set_no_activate(self, native_id: int) -> None:
        return None


class LinuxInstance:
    def acquire(self, name: str) -> bool:
        return linux_instance.acquire(name)

    def release(self, name: str) -> None:
        linux_instance.release(name)

    def signal_running(self, name: str, command: str) -> bool:
        return linux_instance.signal_running(name, command)

    def find_running(self, name: str) -> int:
        return linux_instance.find_running(name)

    def message_window(self, name: str, handlers: MessageHandlers) -> Any:
        return linux_instance.SocketWindow(name, handlers)


class LinuxProcesses:
    def create_job(self) -> Any:
        return linux_process.EngineJob()

    def spawn_hidden(
        self,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        stdout_path: Path,
        stderr_path: Path,
        cwd: Path | None = None,
        job: Any = None,
        affinity_mask: int | None = None,
    ) -> Any:
        return linux_process.spawn_hidden(
            args,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=cwd,
            job=job,
            affinity_mask=affinity_mask,
        )

    def cpu_plan(self) -> CpuPlan:
        return linux_process.detect_cpu_plan()

    def hidden_process_kwargs(self) -> dict[str, Any]:
        return {}


class LinuxAutostart:
    def apply(self, enabled: bool, command: str) -> bool:
        return desktop.apply_autostart(enabled, command)

    def current_command(self) -> str:
        return desktop.autostart_command()


class LinuxSecrets:
    def protect(self, data: bytes, description: str = "") -> bytes:
        return desktop.protect(data, description)

    def unprotect(self, blob: bytes) -> bytes:
        return desktop.unprotect(blob)


class LinuxShell:
    def open_path(self, path: str) -> None:
        desktop.open_path(path)

    def reveal(self, path: Path) -> None:
        desktop.reveal(path)

    def has_settings(self, kind: str) -> bool:
        return desktop.has_settings(kind)

    def open_settings(self, kind: str) -> bool:
        return desktop.open_settings(kind)

    def microphone_blocked(self) -> bool | None:
        return None

    def audio_sources(self) -> tuple[str, tuple[str, ...]] | None:
        return desktop.audio_sources()


class LinuxAppearance(StubAppearance):
    def apps_dark(self) -> bool | None:
        from PySide6 import QtCore, QtGui

        if QtGui.QGuiApplication.instance() is None:
            return None
        scheme = QtGui.QGuiApplication.styleHints().colorScheme()
        if scheme == QtCore.Qt.ColorScheme.Unknown:
            return None
        return scheme == QtCore.Qt.ColorScheme.Dark


def choose_qt_platform(env: dict[str, str] | Any = os.environ) -> None:
    """Run Qt through XWayland on Wayland unless the user or SPELLS_QT_PLATFORM says otherwise."""
    wanted = env.get(QT_PLATFORM_ENV)
    if wanted:
        env["QT_QPA_PLATFORM"] = wanted
    elif env.get("WAYLAND_DISPLAY") and env.get("DISPLAY") and "QT_QPA_PLATFORM" not in env:
        env["QT_QPA_PLATFORM"] = "xcb"


def capabilities(*, keys_out: bool, focus: bool, clipboard: bool) -> Capabilities:
    return Capabilities(
        hold_to_talk=True,
        esc_cancel=True,
        live_typing=False,
        app_profiles=focus,
        window_titles=focus,
        edit_hotkey=keys_out and clipboard,
        clipboard_restore=clipboard,
        autostart=True,
        self_update=False,
        app_records_chords=True,
        elevation_check=False,
    )


def build() -> Platform:
    choose_qt_platform()
    keyboard = VirtualKeyboard()
    tracker = FocusTracker()
    tracker.start()
    atexit.register(tracker.stop)
    focus = LinuxFocus(tracker)
    has_clipboard = linux_clipboard.available()
    if not has_clipboard:
        log.warning("wl-copy and wl-paste are missing; install wl-clipboard to paste dictations")
    keys_out = keyboard.prepare()
    atexit.register(keyboard.close)
    hook = EvdevHook(keyboard)
    return Platform(
        name="linux",
        capabilities=capabilities(keys_out=keys_out, focus=tracker.available, clipboard=has_clipboard),
        key_hook=hook,
        hotkeys=functools.partial(HotkeyThread, hook_backend=hook),
        focus=focus,
        keyboard=LinuxKeyboard(keyboard, focus, linux_clipboard),
        clipboard=linux_clipboard,
        overlay=LinuxOverlay(tracker),
        instance=LinuxInstance(),
        processes=LinuxProcesses(),
        autostart=LinuxAutostart(),
        secrets=LinuxSecrets(),
        shell=LinuxShell(),
        appearance=LinuxAppearance(),
        keys=LinuxKeys(),
        updater=StubUpdater("linux"),
    )
