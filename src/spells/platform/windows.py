from __future__ import annotations

import functools
from dataclasses import replace
from pathlib import Path
from typing import Any

from spells.hotkey import HotkeyThread
from spells.models import CpuPlan
from spells.platform import stub
from spells.platform.base import Capabilities, Platform
from spells.platform.keys_windows import WindowsKeys
from spells.win32 import clipboard as win32_clipboard
from spells.win32 import cpu as win32_cpu
from spells.win32 import hook as win32_hook
from spells.win32 import input as win32_input
from spells.win32 import process as win32_process
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
    )
