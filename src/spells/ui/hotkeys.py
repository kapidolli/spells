"""Recording hotkeys: the capture dialog, the recorder widget and the RegisterHotKey probe.

The capture reads one chord from real key events and folds the side-specific modifiers to
their generic codes. A chord with a non-modifier key is probed through RegisterHotKey: a
taken chord is refused, any other probe error is a warning that lets the chord through
(B3-34). The equal, subset and superset rule stays with config's validation, whose
SettingsError names both chords and is shown as is.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spells import vk
from spells.config import ConfigStore, Settings, SettingsError
from spells.models import Chord
from spells.ui import style, theme
from spells.ui.tray import chords_of
from spells.ui.widgets import KeycapRow, make_button, make_label

log = logging.getLogger(__name__)

Notify = Callable[[str, str, str], None]

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_FLAGS: dict[int, int] = {
    vk.VK_CONTROL: MOD_CONTROL,
    vk.VK_MENU: MOD_ALT,
    vk.VK_SHIFT: MOD_SHIFT,
    vk.VK_LWIN: MOD_WIN,
    vk.VK_RWIN: MOD_WIN,
}
VK_ESCAPE = 0x1B


def _default_probe(modifiers: int, key: int) -> bool:
    from spells.win32.hook import register_hotkey_probe

    return register_hotkey_probe(modifiers, key)


def _default_notify(kind: str, title: str, text: str) -> None:
    from spells.ui.welcome import default_notify

    default_notify(kind, title, text)


def fold_keys(keys: Any) -> tuple[int, ...]:
    """Generic modifier codes, duplicates removed, order kept."""
    folded: list[int] = []
    for key in keys:
        code = vk.generic_modifier(int(key))
        if code not in folded:
            folded.append(code)
    return tuple(folded)


def probe_arguments(keys: Any) -> tuple[int, int] | None:
    """(modifier flags, key) for RegisterHotKey, or None when the chord cannot be probed.

    A chord is probeable when it holds exactly one non-modifier key (spec 14.4).
    """
    folded = fold_keys(keys)
    others = [key for key in folded if key not in MOD_FLAGS]
    if len(others) != 1:
        return None
    flags = 0
    for key in folded:
        flags |= MOD_FLAGS.get(key, 0)
    return flags, others[0]


def chord_is_free(keys: Any, probe: Callable[[int, int], bool] | None, notify: Notify) -> bool:
    """False when another app owns the chord; a probe failure is a warning and passes."""
    args = probe_arguments(keys)
    if args is None:
        return True
    label = theme.chord_label(keys)
    try:
        free = bool((probe or _default_probe)(*args))
    except Exception as exc:
        log.warning("RegisterHotKey probe failed for %s", label, exc_info=True)
        notify(
            "warning",
            "Hotkey check failed",
            f"Could not check whether {label} is free ({exc}). The chord was saved anyway.",
        )
        return True
    if not free:
        notify(
            "warning",
            "Hotkey already in use",
            f"{label} is already registered by another app or by Windows. Choose a different chord.",
        )
    return free


class ChordCaptureDialog(QtWidgets.QDialog):
    """Reads one chord from real key events: the keys down when the first key is released.

    While it is open the hook runs with no chords, so the chord being pressed cannot start a
    dictation. On show it snapshots the hook's current chords (`HotkeyThread.chords`, so a
    paused tray stays paused) and restores exactly that snapshot when it closes; the
    `active_chords` given at construction are the fallback for a hotkey without that
    property. Esc cancels.
    """

    def __init__(
        self,
        *,
        hotkey: Any,
        active_chords: tuple[Chord, ...],
        title: str = "Press your new hotkey",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._hotkey = hotkey
        self._active = tuple(active_chords)
        self.chord: tuple[int, ...] | None = None
        self._down: list[int] = []
        self._suspended = False
        self.setWindowTitle("Record a hotkey")
        self.setModal(True)
        self.setFixedWidth(440)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(0)
        heading = make_label(title, "subtitle", parent=self)
        layout.addWidget(heading)
        layout.addSpacing(6)
        hint = make_label(
            "Hold the keys of the new hotkey, then release them. Modifiers alone are allowed. Esc cancels.",
            "body",
            "secondary",
            wrap=True,
            parent=self,
        )
        layout.addWidget(hint)
        layout.addSpacing(16)
        well = QtWidgets.QFrame(self)
        well.setProperty("role", "well")
        well.setMinimumHeight(96)
        well_layout = QtWidgets.QHBoxLayout(well)
        well_layout.setContentsMargins(16, 16, 16, 16)
        self.preview = KeycapRow((), large=True, placeholder="Waiting for keys", parent=well)
        well_layout.addStretch(1)
        well_layout.addWidget(self.preview)
        well_layout.addStretch(1)
        layout.addWidget(well)
        layout.addSpacing(20)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel = make_button("Cancel", parent=self)
        cancel.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        style.apply_window_chrome(self)
        if not self._suspended:
            self._suspended = True
            current = getattr(self._hotkey, "chords", None)
            if current is not None:
                self._active = tuple(current)
            self._push(())
        self.grabKeyboard()
        self.setFocus()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self.releaseKeyboard()
        if self._suspended:
            self._suspended = False
            self._push(self._active)
        super().hideEvent(event)

    def _push(self, chords: tuple[Chord, ...]) -> None:
        try:
            self._hotkey.update_chords(chords)
        except Exception:
            log.exception("hotkey.update_chords failed during capture")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        code = int(event.nativeVirtualKey())
        if code == VK_ESCAPE or (code == 0 and event.key() == QtCore.Qt.Key.Key_Escape):
            self.reject()
            return
        if event.isAutoRepeat() or code == 0:
            return
        generic = vk.generic_modifier(code)
        if generic not in self._down:
            self._down.append(generic)
        self.preview.set_keys(theme.chord_keys(self._down))

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat() or not self._down:
            return
        self.chord = tuple(self._down)
        self.accept()


class HotkeyRecorder(QtWidgets.QWidget):
    """The chord as keycaps and a Change button; accept_chord() is the whole rule."""

    def __init__(
        self,
        *,
        config: ConfigStore,
        hotkey: Any,
        probe: Callable[[int, int], bool] | None,
        notify: Notify | None,
        chord_of: Callable[[Settings], Chord | None],
        with_chord: Callable[[Settings, tuple[int, ...]], Settings],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._hotkey = hotkey
        self._probe = probe or _default_probe
        self._notify = notify or _default_notify
        self._chord_of = chord_of
        self._with_chord = with_chord
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.label = KeycapRow((), parent=self)
        layout.addWidget(self.label, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.change_button = make_button("Change", parent=self)
        self.change_button.clicked.connect(self.start_capture)
        layout.addWidget(self.change_button)
        self.apply_settings(config.settings)

    def apply_settings(self, settings: Settings) -> None:
        chord = self._chord_of(settings)
        self.label.set_keys(theme.chord_keys(chord) if chord is not None else ())

    def start_capture(self) -> None:
        dialog = ChordCaptureDialog(hotkey=self._hotkey, active_chords=chords_of(self._config.settings), parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.chord:
            self.accept_chord(dialog.chord)

    def probe_ok(self, keys: tuple[int, ...]) -> bool:
        """False when another app owns the chord; a probe failure is a warning and passes."""
        return chord_is_free(keys, self._probe, self._notify)

    def accept_chord(self, keys: Any) -> bool:
        folded = fold_keys(keys)
        if not folded:
            return False
        if not self.probe_ok(folded):
            return False
        try:
            self._config.update(lambda settings: self._with_chord(settings, folded))
        except SettingsError as exc:
            self._notify("warning", "Hotkey refused", str(exc))
            return False
        self.apply_settings(self._config.settings)
        return True


__all__ = [
    "MOD_FLAGS",
    "ChordCaptureDialog",
    "HotkeyRecorder",
    "chord_is_free",
    "fold_keys",
    "probe_arguments",
]
