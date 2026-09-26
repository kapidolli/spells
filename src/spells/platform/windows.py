from __future__ import annotations

import functools
from dataclasses import replace
from pathlib import Path
from typing import Any

from spells import updates
from spells.hotkey import HotkeyThread
from spells.models import CpuPlan
from spells.platform import stub
from spells.platform.base import Capabilities, MessageHandlers, Platform
from spells.platform.keys_windows import WindowsKeys
from spells.win32 import autostart as win32_autostart
from spells.win32 import clipboard as win32_clipboard
from spells.win32 import cpu as win32_cpu
from spells.win32 import dpapi as win32_dpapi
from spells.win32 import hook as win32_hook
from spells.win32 import input as win32_input
from spells.win32 import instance as win32_instance
from spells.win32 import msgwindow as win32_msgwindow
from spells.win32 import process as win32_process
from spells.win32 import shell as win32_shell
from spells.win32 import window as win32_window


class WindowsFocus:
    def foreground(self) -> int:
        return win32_window.foreground_hwnd()

    def app_name(self, window: int) -> str:
        return win32_window.window_process_name(window)

    def title(self, window: int) -> str:
        return win32_window.window_title(window)

    def is_elevated(self, window: int) -> bool:
        return win32_window.is_elevated_window(window)


class WindowsKeyboard:
    def type_unicode(
        self,
        text: str,
        batch_size: int = win32_input.DEFAULT_BATCH_SIZE,
        *,
        extra_info: int = 0,
        release_modifiers: bool = False,
    ) -> None:
        win32_input.type_unicode(
            text, batch_size, extra_info=extra_info, release_modifiers=release_modifiers
        )

    def send_backspaces(
        self,
        count: int,
        batch_size: int = win32_input.DEFAULT_BATCH_SIZE,
        *,
        extra_info: int = 0,
        release_modifiers: bool = False,
    ) -> None:
        win32_input.send_backspaces(
            count, batch_size, extra_info=extra_info, release_modifiers=release_modifiers
        )

    def send_paste(self) -> None:
        win32_input.send_ctrl_v()

    def send_copy(self) -> None:
        win32_input.send_ctrl_c()

    def release_held_modifiers(self) -> list[int]:
        return win32_input.release_held_modifiers()


class WindowsProcesses:
    def create_job(self) -> Any:
        return win32_process.JobObject()

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
        return win32_process.spawn_hidden(
            args,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=cwd,
            job=job,
            affinity_mask=affinity_mask,
        )

    def cpu_plan(self) -> CpuPlan:
        return win32_cpu.detect_cpu_plan()

    def hidden_process_kwargs(self) -> dict[str, Any]:
        return {"creationflags": 0x08000000}


class WindowsInstance:
    def acquire(self, name: str) -> bool:
        return win32_instance.acquire_single_instance(name)

    def release(self, name: str) -> None:
        win32_instance.release_single_instance(name)

    def signal_running(self, name: str, command: str) -> bool:
        return win32_instance.signal_running_instance(name, command)

    def find_running(self, name: str) -> int:
        return win32_instance.find_message_window(name)

    def message_window(self, name: str, handlers: MessageHandlers) -> Any:
        return win32_msgwindow.MessageWindow(name, handlers)


class WindowsAutostart:
    def apply(self, enabled: bool, command: str) -> bool:
        return win32_autostart.apply(enabled, command)

    def current_command(self) -> str:
        return win32_autostart.current_command()


class WindowsShell:
    def open_path(self, path: str) -> None:
        win32_shell.open_path(path)

    def reveal(self, path: Path) -> None:
        win32_shell.reveal(path)

    def has_settings(self, kind: str) -> bool:
        return kind in win32_shell.SETTINGS_URIS

    def open_settings(self, kind: str) -> bool:
        return win32_shell.open_settings(kind)

    def microphone_blocked(self) -> bool | None:
        return win32_shell.microphone_privacy_denied()


class WindowsSecrets:
    def protect(self, data: bytes, description: str = "") -> bytes:
        return win32_dpapi.protect(data, description)

    def unprotect(self, blob: bytes) -> bytes:
        return win32_dpapi.unprotect(blob)


class WindowsUpdater:
    @property
    def can_apply(self) -> bool:
        return True

    def apply(self, installer_path: Path) -> None:
        updates.launch_installer(installer_path)


def build() -> Platform:
    return replace(
        stub.build("windows"),
        capabilities=Capabilities.everything(),
        key_hook=win32_hook,
        hotkeys=functools.partial(HotkeyThread, hook_backend=win32_hook),
        focus=WindowsFocus(),
        keyboard=WindowsKeyboard(),
        clipboard=win32_clipboard,
        keys=WindowsKeys(),
        processes=WindowsProcesses(),
        instance=WindowsInstance(),
        autostart=WindowsAutostart(),
        updater=WindowsUpdater(),
        shell=WindowsShell(),
        secrets=WindowsSecrets(),
    )
