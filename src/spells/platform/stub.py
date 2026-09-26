from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from spells.models import Chord, CpuPlan
from spells.platform.base import (
    Capabilities,
    MessageHandlers,
    Platform,
    PlatformUnavailable,
)


def _refuse(system: str, job: str) -> PlatformUnavailable:
    return PlatformUnavailable(f"{job} is not available on {system} yet")


class StubHotkeys:
    def __init__(
        self,
        chords: Iterable[Chord],
        callbacks: Any,
        *,
        on_error: Callable[[str], None] | None = None,
        **_: Any,
    ) -> None:
        self._chords = tuple(chords)
        self._writing: int | None = None

    @property
    def recording(self) -> bool:
        return False

    @property
    def chords(self) -> tuple[Chord, ...]:
        return self._chords

    @property
    def writing_id(self) -> int | None:
        return self._writing

    def start(self) -> None:
        return None

    def stop(self, timeout: float = 1.0) -> None:
        return None

    def update_chords(self, chords: Iterable[Chord]) -> None:
        self._chords = tuple(chords)

    def end_recording(self, dictation_id: int) -> None:
        return None

    def set_writing(self, dictation_id: int | None) -> None:
        self._writing = dictation_id

    def request_reinstall(self) -> None:
        return None


class StubFocus:
    def foreground(self) -> int:
        return 0

    def app_name(self, window: int) -> str:
        return ""

    def title(self, window: int) -> str:
        return ""

    def is_elevated(self, window: int) -> bool:
        return False


class StubKeyboard:
    def __init__(self, system: str) -> None:
        self._system = system

    def type_unicode(self, text: str, *, extra_info: int = 0, release_modifiers: bool = False) -> None:
        raise _refuse(self._system, "Typing into other apps")

    def send_backspaces(self, count: int, *, extra_info: int = 0, release_modifiers: bool = False) -> None:
        raise _refuse(self._system, "Typing into other apps")

    def send_paste(self) -> None:
        raise _refuse(self._system, "Pasting into other apps")

    def send_copy(self) -> None:
        raise _refuse(self._system, "Copying from other apps")

    def release_held_modifiers(self) -> list[int]:
        return []


class StubClipboard:
    def __init__(self, system: str) -> None:
        self._system = system

    def snapshot(self) -> Any:
        raise _refuse(self._system, "The clipboard")

    def set_text(self, text: str) -> int:
        raise _refuse(self._system, "The clipboard")

    def get_text(self) -> str | None:
        raise _refuse(self._system, "The clipboard")

    def sequence_number(self) -> int:
        raise _refuse(self._system, "The clipboard")

    def restore(self, snapshot: Any, expected_sequence: int) -> bool:
        raise _refuse(self._system, "The clipboard")


class StubOverlay:
    def __init__(self, system: str) -> None:
        self._system = system

    def monitor_rect_for(self, window: int) -> tuple[int, int, int, int]:
        raise _refuse(self._system, "Placing the pill")

    def set_no_activate(self, native_id: int) -> None:
        return None


class NullMessageWindow:
    def destroy(self) -> None:
        return None

    def run(self, stop: Any) -> None:
        return None


class StubInstance:
    def acquire(self, name: str) -> bool:
        return True

    def release(self, name: str) -> None:
        return None

    def signal_running(self, name: str, command: str) -> bool:
        return False

    def find_running(self, name: str) -> int:
        return 0

    def message_window(self, name: str, handlers: MessageHandlers) -> NullMessageWindow:
        return NullMessageWindow()


class StubProcesses:
    def __init__(self, system: str) -> None:
        self._system = system

    def create_job(self) -> Any:
        return None

    def spawn_hidden(self, args: list[str], **kwargs: Any) -> Any:
        raise _refuse(self._system, "Starting the speech engines")

    def cpu_plan(self) -> CpuPlan:
        return CpuPlan()

    def hidden_process_kwargs(self) -> dict[str, Any]:
        return {}


class StubAutostart:
    def apply(self, enabled: bool, command: str) -> bool:
        return False

    def current_command(self) -> str:
        return ""


class StubSecrets:
    def __init__(self, system: str) -> None:
        self._system = system

    def protect(self, data: bytes, description: str = "") -> bytes:
        raise _refuse(self._system, "Protected storage")

    def unprotect(self, blob: bytes) -> bytes:
        raise _refuse(self._system, "Protected storage")


class StubShell:
    def open_path(self, path: str) -> None:
        from PySide6 import QtCore, QtGui

        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def reveal(self, path: Path) -> None:
        self.open_path(str(Path(path).parent))

    def has_settings(self, kind: str) -> bool:
        return False

    def open_settings(self, kind: str) -> bool:
        return False

    def microphone_blocked(self) -> bool | None:
        return None


class StubAppearance:
    def apps_dark(self) -> bool | None:
        return None

    def taskbar_light(self) -> bool:
        return False

    def accent_palette(self) -> tuple[str, ...] | None:
        return None

    def accent_colour(self) -> str | None:
        from PySide6 import QtGui

        if QtGui.QGuiApplication.instance() is None:
            return None
        colour = QtGui.QGuiApplication.palette().color(QtGui.QPalette.ColorRole.Accent)
        return colour.name().upper() if colour.isValid() else None

    def animations_enabled(self) -> bool:
        return True

    def style_title_bar(self, native_id: int, *, dark: bool, caption: str, text: str) -> bool:
        return False

    def round_corners(self, native_id: int) -> bool:
        return False


class StubUpdater:
    def __init__(self, system: str) -> None:
        self._system = system

    @property
    def can_apply(self) -> bool:
        return False

    def apply(self, installer_path: Path) -> None:
        raise _refuse(self._system, "Installing updates")


def _keys(name: str) -> Any:
    if name == "macos":
        from spells.platform.keys_macos import MacKeys

        return MacKeys()
    from spells.platform.keys_linux import LinuxKeys

    return LinuxKeys()


def build(name: str) -> Platform:
    return Platform(
        name=name,
        capabilities=Capabilities(),
        key_hook=None,
        hotkeys=StubHotkeys,
        focus=StubFocus(),
        keyboard=StubKeyboard(name),
        clipboard=StubClipboard(name),
        overlay=StubOverlay(name),
        instance=StubInstance(),
        processes=StubProcesses(name),
        autostart=StubAutostart(),
        secrets=StubSecrets(name),
        shell=StubShell(),
        appearance=StubAppearance(),
        keys=_keys(name),
        updater=StubUpdater(name),
    )
