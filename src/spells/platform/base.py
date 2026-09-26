from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

from spells.models import Chord, CpuPlan

SETTINGS_SOUND = "sound"
SETTINGS_MICROPHONE_PRIVACY = "microphone_privacy"


class PlatformUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Capabilities:
    hold_to_talk: bool = False
    esc_cancel: bool = False
    live_typing: bool = False
    app_profiles: bool = False
    window_titles: bool = False
    edit_hotkey: bool = False
    clipboard_restore: bool = False
    autostart: bool = False
    self_update: bool = False
    app_records_chords: bool = False
    elevation_check: bool = False

    @classmethod
    def everything(cls) -> Capabilities:
        return cls(**{item.name: True for item in fields(cls)})

    def missing(self) -> tuple[str, ...]:
        return tuple(item.name for item in fields(self) if not getattr(self, item.name))


@dataclass
class MessageHandlers:
    """Callbacks invoked on the window's thread. Any of them may be None."""

    on_copydata: Callable[[str], None] | None = None
    on_query_end_session: Callable[[int], bool] | None = None
    on_end_session: Callable[[int], None] | None = None
    on_resume: Callable[[], None] | None = None
    on_session_unlock: Callable[[], None] | None = None


@dataclass(frozen=True)
class PlatformPaths:
    settings_path: Path
    history_path: Path
    log_dir: Path


class KeyHook(Protocol):
    mask_keys: tuple[tuple[int, bool, int], ...]
    supports_probe: bool

    def install_keyboard_hook(self, callback: Callable[[Any], bool]) -> int: ...

    def uninstall_keyboard_hook(self, handle: int) -> None: ...

    def send_key(self, vk: int, keydown: bool, extra_info: int = 0) -> None: ...

    def is_key_down(self, vk: int) -> bool: ...

    def run_message_loop(
        self,
        stop: threading.Event,
        timers: dict[int, int],
        on_timer: Callable[[int], None] | None,
        poll_ms: int = 50,
    ) -> None: ...

    def set_timer(self, timer_id: int, interval_ms: int) -> None: ...

    def kill_timer(self, timer_id: int) -> None: ...

    def set_current_thread_priority_highest(self) -> None: ...

    def chord_available(self, modifiers: int, vk: int) -> bool: ...


class HotkeyService(Protocol):
    @property
    def recording(self) -> bool: ...

    @property
    def chords(self) -> tuple[Chord, ...]: ...

    @property
    def writing_id(self) -> int | None: ...

    def start(self) -> None: ...

    def stop(self, timeout: float = 1.0) -> None: ...

    def update_chords(self, chords: Iterable[Chord]) -> None: ...

    def end_recording(self, dictation_id: int) -> None: ...

    def set_writing(self, dictation_id: int | None) -> None: ...

    def request_reinstall(self) -> None: ...


HotkeyFactory = Callable[..., HotkeyService]


class Focus(Protocol):
    def foreground(self) -> int: ...

    def app_name(self, window: int) -> str: ...

    def title(self, window: int) -> str: ...

    def is_elevated(self, window: int) -> bool: ...


class KeyboardOut(Protocol):
    def type_unicode(
        self, text: str, *, extra_info: int = 0, release_modifiers: bool = False
    ) -> None: ...

    def send_backspaces(
        self, count: int, *, extra_info: int = 0, release_modifiers: bool = False
    ) -> None: ...

    def send_paste(self) -> None: ...

    def send_copy(self) -> None: ...

    def release_held_modifiers(self) -> list[int]: ...


class Clipboard(Protocol):
    def snapshot(self) -> Any: ...

    def set_text(self, text: str) -> int: ...

    def get_text(self) -> str | None: ...

    def sequence_number(self) -> int: ...

    def restore(self, snapshot: Any, expected_sequence: int) -> bool: ...


class Overlay(Protocol):
    def monitor_rect_for(self, window: int) -> tuple[int, int, int, int]: ...

    def set_no_activate(self, native_id: int) -> None: ...


class Instance(Protocol):
    def acquire(self, name: str) -> bool: ...

    def release(self, name: str) -> None: ...

    def signal_running(self, name: str, command: str) -> bool: ...

    def find_running(self, name: str) -> int: ...

    def message_window(self, name: str, handlers: MessageHandlers) -> Any: ...


class Processes(Protocol):
    def create_job(self) -> Any: ...

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
    ) -> Any: ...

    def cpu_plan(self) -> CpuPlan: ...

    def hidden_process_kwargs(self) -> dict[str, Any]: ...


class Autostart(Protocol):
    def apply(self, enabled: bool, command: str) -> bool: ...

    def current_command(self) -> str: ...


class Secrets(Protocol):
    def protect(self, data: bytes, description: str = "") -> bytes: ...

    def unprotect(self, blob: bytes) -> bytes: ...


class Shell(Protocol):
    def open_path(self, path: str) -> None: ...

    def reveal(self, path: Path) -> None: ...

    def has_settings(self, kind: str) -> bool: ...

    def open_settings(self, kind: str) -> bool: ...

    def microphone_blocked(self) -> bool | None: ...


class Appearance(Protocol):
    def apps_dark(self) -> bool | None: ...

    def taskbar_light(self) -> bool: ...

    def accent_palette(self) -> tuple[str, ...] | None: ...

    def accent_colour(self) -> str | None: ...

    def animations_enabled(self) -> bool: ...

    def style_title_bar(self, native_id: int, *, dark: bool, caption: str, text: str) -> bool: ...

    def round_corners(self, native_id: int) -> bool: ...


class KeyNames(Protocol):
    def display_name(self, vk: int) -> str: ...

    def vk_from_event(self, event: Any) -> int | None: ...


class Updater(Protocol):
    @property
    def can_apply(self) -> bool: ...

    def apply(self, installer_path: Path) -> None: ...


@dataclass(frozen=True)
class Platform:
    name: str
    capabilities: Capabilities
    key_hook: KeyHook | None
    hotkeys: HotkeyFactory
    focus: Focus
    keyboard: KeyboardOut
    clipboard: Clipboard
    overlay: Overlay
    instance: Instance
    processes: Processes
    autostart: Autostart
    secrets: Secrets
    shell: Shell
    appearance: Appearance
    keys: KeyNames
    updater: Updater
