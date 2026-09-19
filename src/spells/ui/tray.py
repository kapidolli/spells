"""The tray icon and its menu (spec 14.1).

The seven states are composed from the engine status per engine, the hotkey install error,
the GPU probe result and the pipeline's own tray state; `compose_tray_state` is pure so the
composition is testable without a tray. Notifications go through QSystemTrayIcon.showMessage;
a Windows balloon has no buttons, so when the pipeline names an `ms-settings:` URI the text
invites a click and the click launches it. The update balloon of spec 19.7 uses the same
mechanism with the sentinel `OPEN_ABOUT`, which is never launched: it opens the About page
in this process, where the change list and the Install button are.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spells.config import ConfigStore, Settings, SettingsError
from spells.models import Chord, Engine, EngineState
from spells.pipeline import Notice, PillState, PipelineEvent, TrayState
from spells.ui import style, theme
from spells.ui.icons import TrayIconState, tray_icon

log = logging.getLogger(__name__)

NOTIFICATION_MS = 8000
OPEN_SETTINGS_HINT = "Open settings: click this message."
OPEN_ABOUT_HINT = "See what changed: click this message."
OPEN_ABOUT = "spells:about"

ENGINE_NAMES = {Engine.WHISPER: "Speech engine", Engine.LLAMA: "Cleanup engine"}
STATE_WORDS = {
    EngineState.STARTING: "starting",
    EngineState.READY: "ready",
    EngineState.RESTARTING: "restarting",
    EngineState.PAUSED: "paused",
    EngineState.CPU_FALLBACK: "running on CPU",
    EngineState.UNLOADED: "unloaded",
    EngineState.FAILED: "failed",
}
REASON_TEXT = {
    "ok": "",
    "no_vulkan_gpu": "no Vulkan GPU",
    "oom": "out of GPU memory",
    "crash": "crashed",
    "dll_not_found": "a DLL is missing",
    "idle": "unloaded after idle",
    "no_model": "no cleanup model for your languages",
    "cpu_selected": "chosen for this computer",
}
# llama FAILED with reason no_model: nothing was launched and dictation runs without cleanup,
# so it is a warning with this line, never an error (spec 14.1, 16).
NO_CLEANUP_MODEL = "Cleanup is off: no cleanup model for your languages"
HEADLINES = {
    TrayIconState.STARTING: "Starting engines",
    TrayIconState.READY: "Ready",
    TrayIconState.RECORDING: "Recording",
    TrayIconState.PROCESSING: "Processing",
    TrayIconState.WARNING: "Warning",
    TrayIconState.ERROR: "Error",
    TrayIconState.IDLE: "Idle, engines unloaded",
}
WARNING_STATES = frozenset({EngineState.RESTARTING, EngineState.PAUSED, EngineState.CPU_FALLBACK})
MIC_SETTINGS_PREFIX = "ms-settings:"


@dataclass(frozen=True)
class TrayInputs:
    """Everything the tray state depends on."""

    engines: Mapping[Engine, tuple[EngineState, str]]
    pipeline: TrayState = TrayState.READY
    hook_error: str | None = None
    no_vulkan_gpu: bool = False
    mic_error: str | None = None
    paused: bool = False
    extra_lines: tuple[str, ...] = field(default_factory=tuple)


def _missing_cleanup_model(engine: Engine, state: EngineState, reason: str) -> bool:
    return engine is Engine.LLAMA and state is EngineState.FAILED and reason == "no_model"


def reason_text(engine: Engine, reason: str) -> str:
    """The readable reason; no_model names the model the engine lacks."""
    if reason == "no_model" and engine is Engine.WHISPER:
        return "no speech model installed"
    return REASON_TEXT.get(reason, reason)


def _cpu_by_choice(state: EngineState, reason: str) -> bool:
    return state is EngineState.CPU_FALLBACK and reason == "cpu_selected"


def _engine_line(engine: Engine, state: EngineState, reason: str) -> str:
    if _missing_cleanup_model(engine, state, reason):
        return NO_CLEANUP_MODEL
    text = f"{ENGINE_NAMES[engine]} {STATE_WORDS[state]}"
    detail = reason_text(engine, reason)
    if detail and not (state is EngineState.UNLOADED and reason == "idle"):
        text += f" ({detail})"
    return text


def compose_tray_state(inputs: TrayInputs) -> tuple[TrayIconState, str]:
    """The tray state of spec 14.1 and its tooltip, from the inputs."""
    no_model = any(
        _missing_cleanup_model(engine, state, reason)
        for engine, (state, reason) in inputs.engines.items()
    )
    # The states that decide Error, Warning, Starting and Idle; a missing cleanup model only warns.
    states = {
        engine: state
        for engine, (state, reason) in inputs.engines.items()
        if not _missing_cleanup_model(engine, state, reason) and not _cpu_by_choice(state, reason)
    }
    lines: list[str] = []
    for engine, (state, reason) in inputs.engines.items():
        if state is not EngineState.READY:
            lines.append(_engine_line(engine, state, reason))
    if inputs.no_vulkan_gpu:
        lines.append("No Vulkan GPU: the engines run on the CPU")
    if inputs.hook_error:
        lines.append(f"Hotkey unavailable: {inputs.hook_error}")
    if inputs.mic_error:
        lines.append(f"Microphone: {inputs.mic_error}")
    lines.extend(inputs.extra_lines)

    if EngineState.FAILED in states.values() or inputs.mic_error:
        state = TrayIconState.ERROR
    elif inputs.pipeline is TrayState.RECORDING:
        state = TrayIconState.RECORDING
    elif inputs.pipeline is TrayState.PROCESSING:
        state = TrayIconState.PROCESSING
    elif (
        inputs.hook_error
        or inputs.no_vulkan_gpu
        or inputs.extra_lines
        or no_model
        or any(s in WARNING_STATES for s in states.values())
    ):
        state = TrayIconState.WARNING
    elif EngineState.STARTING in states.values():
        state = TrayIconState.STARTING
    elif EngineState.UNLOADED in states.values():
        state = TrayIconState.IDLE
    else:
        state = TrayIconState.READY

    tooltip = f"{theme.APP_NAME}: {HEADLINES[state]}"
    if inputs.paused:
        tooltip += "\nDictation paused"
    if lines:
        tooltip += "\n" + "\n".join(lines)
    return state, tooltip


def chords_of(settings: Settings) -> tuple[Chord, ...]:
    """The chord set in the order the hook matches it (spec 6, 8.5).

    The main chord, the per-language chords, then the compose and edit chords, which are
    only there once the user has recorded them. Two chords of equal length completing on
    the same key event are decided by this order, and config's conflict check walks the
    same one.
    """
    general = settings.general
    writing = (general.compose_chord, general.edit_chord)
    return (
        general.main_chord,
        *general.language_chords,
        *(chord for chord in writing if chord is not None),
    )


class Tray(QtCore.QObject):
    """QSystemTrayIcon plus the menu; every slot here runs on the Qt thread."""

    quit_requested = QtCore.Signal()
    open_requested = QtCore.Signal(str)

    def __init__(
        self,
        *,
        config: ConfigStore,
        pipeline: Any,
        hotkey: Any,
        engines: Any,
        no_vulkan_gpu: bool = False,
        icon: QtWidgets.QSystemTrayIcon | None = None,
        launcher: Callable[[str], Any] | None = None,
        light_taskbar: bool | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._pipeline = pipeline
        self._hotkey = hotkey
        self._launcher = launcher or os.startfile
        self._light = theme.taskbar_is_light() if light_taskbar is None else light_taskbar
        self._icons: dict[TrayIconState, QtGui.QIcon] = {}
        self._engines: dict[Engine, tuple[EngineState, str]] = {}
        for engine in Engine:
            try:
                self._engines[engine] = (engines.status(engine), engines.reason(engine))
            except Exception:
                log.exception("could not read the initial status of %s", engine.value)
                self._engines[engine] = (EngineState.STARTING, "ok")
        self._no_vulkan_gpu = no_vulkan_gpu
        self._extra_warning: str | None = None
        self._pipeline_state = TrayState.READY
        self._hook_error: str | None = self._initial_hook_error(hotkey)
        self._mic_error: str | None = None
        self._paused = False
        self._pending_action: str | None = None
        self.state = TrayIconState.READY
        self.tooltip = ""

        self.icon = icon if icon is not None else QtWidgets.QSystemTrayIcon(self)
        self.icon.activated.connect(self._on_activated)
        self.icon.messageClicked.connect(self._on_message_clicked)

        self.menu = QtWidgets.QMenu()
        self.menu.setFont(style.font("body"))
        self.language_menu = self.menu.addMenu("Language")
        self.language_menu.setFont(style.font("body"))
        self._language_group = QtGui.QActionGroup(self)
        self._language_group.setExclusive(True)
        self.language_actions: dict[str, QtGui.QAction] = {}
        self.cleanup_action = self.menu.addAction("Cleanup")
        self.cleanup_action.setCheckable(True)
        self.cleanup_action.triggered.connect(self._on_cleanup)
        self.menu.addSeparator()
        self.pause_action = self.menu.addAction("Pause dictation")
        self.pause_action.setCheckable(True)
        self.pause_action.triggered.connect(self._on_pause)
        self.retry_action = self.menu.addAction("Retry last dictation")
        self.retry_action.setEnabled(False)
        self.retry_action.triggered.connect(self._on_retry)
        self.menu.addSeparator()
        self.history_action = self.menu.addAction("History")
        self.history_action.triggered.connect(lambda: self.open_requested.emit("history"))
        self.settings_action = self.menu.addAction("Settings")
        self.settings_action.triggered.connect(lambda: self.open_requested.emit("general"))
        self.menu.addSeparator()
        self.quit_action = self.menu.addAction("Quit")
        self.quit_action.triggered.connect(self.quit_requested.emit)
        self.menu.aboutToShow.connect(lambda: style.round_popup(self.menu))
        self.language_menu.aboutToShow.connect(lambda: style.round_popup(self.language_menu))
        self._check_icon = QtGui.QIcon()
        self._blank_icon = QtGui.QIcon()
        self.refresh_menu_icons()
        style.notifier().changed.connect(self._on_theme_changed)
        self.icon.setContextMenu(self.menu)

        self.apply_settings(config.settings)
        self._refresh()

    @staticmethod
    def _initial_hook_error(hotkey: Any) -> str | None:
        stats = getattr(hotkey, "stats", None)
        if stats is not None and getattr(stats, "install_failures", 0) and getattr(stats, "last_error", ""):
            return str(stats.last_error)
        return None

    # Public ------------------------------------------------------------------------------------

    def show(self) -> None:
        self.icon.show()

    def refresh_menu_icons(self) -> None:
        """Glyph icons for the menu in the current text colour."""
        glyphs = (
            (self.language_menu.menuAction(), style.Glyph.GLOBE),
            (self.retry_action, style.Glyph.REFRESH),
            (self.history_action, style.Glyph.HISTORY),
            (self.settings_action, style.Glyph.SETTINGS),
            (self.quit_action, style.Glyph.POWER),
        )
        for action, glyph in glyphs:
            action.setIcon(style.glyph_icon(glyph))
        self._check_icon = style.glyph_icon(style.Glyph.CHECK, style.palette().accent_text)
        blank = QtGui.QPixmap(16, 16)
        blank.fill(QtCore.Qt.GlobalColor.transparent)
        self._blank_icon = QtGui.QIcon(blank)
        self._sync_check_icons()

    def _on_theme_changed(self, _palette: object) -> None:
        self.refresh_menu_icons()

    def _sync_check_icons(self) -> None:
        """Checkable entries show a check mark when on and keep their column when off."""
        for action in (self.cleanup_action, self.pause_action, *self.language_actions.values()):
            action.setIcon(self._check_icon if action.isChecked() else self._blank_icon)

    def hide(self) -> None:
        self.icon.hide()

    @property
    def paused(self) -> bool:
        return self._paused

    def inputs(self) -> TrayInputs:
        return TrayInputs(
            engines=dict(self._engines),
            pipeline=self._pipeline_state,
            hook_error=self._hook_error,
            no_vulkan_gpu=self._no_vulkan_gpu,
            mic_error=self._mic_error,
            paused=self._paused,
            extra_lines=(self._extra_warning,) if self._extra_warning else (),
        )

    def busy(self) -> bool:
        """True while a dictation is being recorded or processed (spec 14.1)."""
        return self._pipeline_state in (TrayState.RECORDING, TrayState.PROCESSING)

    def notify_update(self, version: str, released: str = "") -> bool:
        """One balloon for a new version, and never over a dictation (spec 19.7).

        Returns whether it was shown. How often this is allowed to happen at all is the
        coordinator's decision; the rule the tray keeps for itself is that a dictation in
        flight is never interrupted by it.
        """
        if not version or self.busy():
            return False
        when = f", {released}" if released else ""
        self.notify(f"{theme.APP_NAME} {version} is available{when}.", OPEN_ABOUT)
        return True

    def notify(self, text: str, action: str | None = None, *, warning: bool = False) -> None:
        """A toast; an `ms-settings:` action is launched when the toast is clicked."""
        body = text
        self._pending_action = action if action else None
        if action == OPEN_ABOUT:
            body = f"{text}\n{OPEN_ABOUT_HINT}"
        elif action:
            body = f"{text}\n{OPEN_SETTINGS_HINT}"
        kind = (
            QtWidgets.QSystemTrayIcon.MessageIcon.Warning
            if warning or action
            else QtWidgets.QSystemTrayIcon.MessageIcon.Information
        )
        self.icon.showMessage(theme.APP_NAME, body, kind, NOTIFICATION_MS)

    # Slots (Qt thread) -------------------------------------------------------------------------

    def apply_event(self, event: PipelineEvent) -> None:
        self._pipeline_state = event.tray
        self.retry_action.setEnabled(bool(event.retry_available))
        if event.pill in (PillState.RECORDING, PillState.LATCHED):
            self._mic_error = None
        if (
            event.notice is Notice.ERROR
            and event.notification_action
            and str(event.notification_action).startswith(MIC_SETTINGS_PREFIX)
        ):
            self._mic_error = event.notification or event.notice_text
        if event.notification:
            self.notify(event.notification, event.notification_action, warning=event.notice is Notice.ERROR)
        self._refresh()

    def set_engine_status(self, engine: Engine, state: EngineState, reason: str) -> None:
        self._engines[Engine(engine)] = (EngineState(state), str(reason))
        self._refresh()

    def set_hook_error(self, message: str) -> None:
        self._hook_error = message or None
        self._refresh()

    def set_no_vulkan_gpu(self, value: bool) -> None:
        self._no_vulkan_gpu = bool(value)
        self._refresh()

    def set_extra_warning(self, reason: str | None) -> None:
        """A startup problem the app found itself, such as a missing speech model.

        It shows as the Warning state with the reason in the tooltip, and None clears it. A
        missing cleanup model does not come through here: it reaches the tray from the
        supervisor as llama FAILED with reason no_model, which composes Warning on its own.
        """
        self._extra_warning = str(reason) if reason else None
        self._refresh()

    def apply_settings(self, settings: Settings) -> None:
        self._rebuild_languages(settings)
        self.cleanup_action.setChecked(settings.cleanup.enabled)
        if not self._paused:
            self._push_chords(chords_of(settings))
        self._refresh()

    # Menu handlers -----------------------------------------------------------------------------

    def _rebuild_languages(self, settings: Settings) -> None:
        for action in list(self.language_actions.values()):
            self._language_group.removeAction(action)
            self.language_menu.removeAction(action)
            action.deleteLater()
        self.language_actions = {}
        codes = ["auto", *settings.general.enabled_languages]
        for code in codes:
            text = "Auto" if code == "auto" else theme.language_name(code)
            action = self.language_menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(settings.general.language_mode == code)
            action.triggered.connect(lambda _checked=False, c=code: self._on_language(c))
            self._language_group.addAction(action)
            self.language_actions[code] = action

    def _on_language(self, code: str) -> None:
        def mutate(settings: Settings) -> Settings:
            return replace(settings, general=replace(settings.general, language_mode=code))

        self._update(mutate)

    def _on_cleanup(self, checked: bool) -> None:
        def mutate(settings: Settings) -> Settings:
            return replace(settings, cleanup=replace(settings.cleanup, enabled=bool(checked)))

        self._update(mutate)

    def _on_pause(self, checked: bool) -> None:
        self._paused = bool(checked)
        if self._paused:
            self._push_chords(())
        else:
            self._push_chords(chords_of(self._config.settings))
        self._refresh()

    def _on_retry(self) -> None:
        try:
            self._pipeline.retry_last()
        except Exception:
            log.exception("retry_last failed")

    def _on_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QtWidgets.QSystemTrayIcon.ActivationReason.Trigger,
            QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.open_requested.emit("general")

    def _on_message_clicked(self) -> None:
        action, self._pending_action = self._pending_action, None
        if not action:
            return
        if action == OPEN_ABOUT:
            self.open_requested.emit("about")
            return
        try:
            self._launcher(action)
        except Exception:
            log.exception("could not open %s", action)

    # Internals ---------------------------------------------------------------------------------

    def _update(self, mutator: Callable[[Settings], Settings]) -> None:
        try:
            self._config.update(mutator)
        except SettingsError as exc:
            log.warning("settings change refused: %s", exc)
            self.notify(str(exc), warning=True)
        except Exception:
            log.exception("settings change failed")

    def _push_chords(self, chords: tuple[Chord, ...]) -> None:
        try:
            self._hotkey.update_chords(chords)
        except Exception:
            log.exception("hotkey.update_chords failed")

    def _refresh(self) -> None:
        state, tooltip = compose_tray_state(self.inputs())
        self.state = state
        self.tooltip = tooltip
        icon = self._icons.get(state)
        if icon is None:
            icon = self._icons[state] = tray_icon(state, light_taskbar=self._light)
        self.icon.setIcon(icon)
        self.icon.setToolTip(tooltip)
        self._sync_check_icons()


__all__ = [
    "OPEN_ABOUT",
    "Tray",
    "TrayInputs",
    "chords_of",
    "compose_tray_state",
    "reason_text",
]
