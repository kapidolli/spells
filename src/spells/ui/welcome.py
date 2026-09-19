"""The first-run page (spec 14.3) and the microphone widgets it shares with the General page.

The page is a short tour in three steps: how dictation works (hold the hotkey, speak,
release) with a small animated hint of the pill, the languages the user dictates in, and
the microphone with a live meter, a box to try it, Start with Windows and the elevated
window note. MicPicker lists the input devices of the capture host API, WASAPI where the
machine has one (audio.list_input_devices), and writes general.mic_device through
ConfigStore.update(); a name stored while capture went through MME is matched by prefix,
because MME cuts device names at 31 characters. MicMeter opens its own short Recorder on
the chosen device at the device's own rate and shows the level and the capture path; the
level callback runs on the PortAudio thread and is marshalled through a queued signal. The
meter runs only while the microphone step is on screen.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spells.audio import MicError, names_match
from spells.config import ConfigStore, Settings, SettingsError
from spells.ui import brand, style, theme
from spells.ui.languages import LanguageAdder, LanguageList, set_language_enabled
from spells.ui.pill import meter_level
from spells.ui.recordings import RECORDINGS_NOTE
from spells.ui.widgets import (
    Card,
    ComboBox,
    GlyphLabel,
    InfoBar,
    KeycapRow,
    LevelBar,
    LogoMark,
    ScrollPage,
    SettingRow,
    StepDots,
    ToggleSwitch,
    make_button,
    make_label,
    painter_for,
)

log = logging.getLogger(__name__)

Notify = Callable[[str, str, str], None]

ELEVATED_NOTE = (
    "One limit to know about: the hotkey cannot start a dictation while an elevated "
    "(administrator) window has the focus, because Windows lets no keyboard hook of a normal "
    "app see input there (UIPI). Click a normal window first. If the focus moves to an "
    "elevated window while a dictation is still processing, the text goes to the clipboard "
    "instead and a \"Copied\" notice tells you to press Ctrl+V."
)
TRY_PLACEHOLDER = "Click here, hold {chord} and speak. Release to see the text appear."
WRITING_NOTE = (
    "Two more hotkeys wait for you in Settings: one where what you say is an instruction, "
    "so \"write an email to Marta asking for the September invoice\" inserts the email, and "
    "one that changes text you have selected, such as make this shorter or translate this "
    "into German."
)
STEP_NAMES = ("intro", "languages", "microphone")
STEP_INTRO = 0
STEP_LANGUAGES = 1
STEP_MICROPHONE = 2
HINT_CYCLE_MS = 4600
HINT_FRAME_MS = 33
HINT_TEXT = "See you at ten."
HINT_POSE = (4, 7, 11, 9, 14, 18, 13, 20, 15, 17, 10, 12, 6, 8, 4)


def default_notify(kind: str, title: str, text: str) -> None:
    box = QtWidgets.QMessageBox(None)
    box.setWindowTitle(title)
    box.setText(text)
    icons = {
        "warning": QtWidgets.QMessageBox.Icon.Warning,
        "error": QtWidgets.QMessageBox.Icon.Critical,
    }
    box.setIcon(icons.get(kind, QtWidgets.QMessageBox.Icon.Information))
    box.exec()


def _default_devices() -> list:
    from spells.audio import list_input_devices

    return list_input_devices()


def _default_recorder(**kwargs: Any) -> Any:
    from spells.audio import Recorder

    return Recorder(**kwargs)


def listening_text(recorder: Any) -> str:
    """What the meter says while it runs: the capture path and the rate it opened."""
    host_api = str(getattr(recorder, "host_api", "") or "")
    rate = int(getattr(recorder, "sample_rate", 0) or 0)
    if host_api and rate:
        return f"Listening through {host_api} at {rate / 1000:g} kHz. Speak to see the level move."
    return "Listening. Speak to see the level move."


class MicMeter(QtWidgets.QWidget):
    """A live level bar fed by its own Recorder; start() and stop() bracket the stream."""

    _level_arrived = QtCore.Signal(float)

    def __init__(
        self,
        *,
        recorder_factory: Callable[..., Any] | None = None,
        notify: Notify | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._factory = recorder_factory or _default_recorder
        self._notify = notify or default_notify
        self._recorder: Any = None
        self.level = 0.0
        self._level_arrived.connect(self._apply_level, QtCore.Qt.ConnectionType.QueuedConnection)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.bar = LevelBar(self)
        self.status = make_label("", "caption", "secondary", wrap=True, parent=self)
        self.status.setVisible(False)
        layout.addWidget(self.bar)
        layout.addWidget(self.status)

    @property
    def running(self) -> bool:
        return self._recorder is not None

    def _set_status(self, text: str) -> None:
        self.status.setText(text)
        self.status.setVisible(bool(text))

    def start(self, device: Any) -> None:
        self.stop()
        self._set_status("")
        recorder = self._factory(device=device, sample_rate=None, on_level=self._on_level, keep_open=False)
        try:
            recorder.start()
        except MicError as exc:
            self._set_status(exc.message)
            self._close(recorder)
            return
        except Exception as exc:
            log.exception("microphone test failed to start")
            self._set_status(f"Could not open the microphone: {exc}")
            self._close(recorder)
            return
        self._recorder = recorder
        self._set_status(listening_text(recorder))

    def stop(self) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return
        try:
            recorder.cancel()
        except Exception:
            log.exception("microphone test cancel failed")
        self._close(recorder)
        self.level = 0.0
        self.bar.set_level(0.0)
        self._set_status("")

    @staticmethod
    def _close(recorder: Any) -> None:
        try:
            recorder.close()
        except Exception:
            log.exception("microphone test close failed")

    def _on_level(self, level: float) -> None:
        self._level_arrived.emit(float(level))

    def _apply_level(self, level: float) -> None:
        self.level = meter_level(level)
        self.bar.set_level(self.level)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self.stop()
        super().hideEvent(event)


class MicPicker(ComboBox):
    """"System default" plus each input device; the choice is stored by device name."""

    device_changed = QtCore.Signal(object)

    def __init__(
        self,
        *,
        config: ConfigStore,
        devices: Callable[[], list] | None = None,
        notify: Notify | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._devices = devices or _default_devices
        self._notify = notify or default_notify
        self._names: list[str | None] = [None]
        self.reload()
        self.apply_settings(config.settings)
        self.currentIndexChanged.connect(self._on_index)

    def reload(self) -> None:
        self.blockSignals(True)
        try:
            self.clear()
            default_name = ""
            names: list[str | None] = [None]
            try:
                devices = list(self._devices())
            except MicError as exc:
                self._notify("warning", "Microphones", exc.message)
                devices = []
            except Exception as exc:
                log.exception("microphone enumeration failed")
                self._notify("warning", "Microphones", f"Could not list the microphones: {exc}")
                devices = []
            for device in devices:
                if device.is_default:
                    default_name = device.name
            label = f"System default ({default_name})" if default_name else "System default"
            self.addItem(label)
            for device in devices:
                self.addItem(device.name)
                names.append(device.name)
            self._names = names
        finally:
            self.blockSignals(False)

    def current_device(self) -> str | None:
        index = self.currentIndex()
        return self._names[index] if 0 <= index < len(self._names) else None

    def index_for(self, wanted: str) -> int:
        """The row for a stored name: exact first, then a prefix, else System default.

        Settings written while capture went through MME hold a name cut at 31
        characters, so the exact round misses and the prefix round finds the
        full WASAPI name.
        """
        needle = wanted.strip().casefold()
        for i, name in enumerate(self._names):
            if name is not None and name.strip().casefold() == needle:
                return i
        for i, name in enumerate(self._names):
            if name is not None and names_match(wanted, name):
                return i
        return 0

    def apply_settings(self, settings: Settings) -> None:
        wanted = settings.general.mic_device
        index = 0
        if wanted is not None:
            index = self.index_for(wanted)
        self.blockSignals(True)
        try:
            self.setCurrentIndex(index)
        finally:
            self.blockSignals(False)

    def _on_index(self, index: int) -> None:
        name = self.current_device()

        def mutate(settings: Settings) -> Settings:
            return replace(settings, general=replace(settings.general, mic_device=name))

        try:
            self._config.update(mutate)
        except SettingsError as exc:
            self._notify("warning", "Microphone", str(exc))
            return
        self.device_changed.emit(name)


# The animated hints ---------------------------------------------------------------------------------


class PillHint(QtWidgets.QWidget):
    """A small copy of the pill: listening bars, then the processing ring, then gone."""

    def __init__(self, phase: Callable[[], float], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._phase = phase
        self.setFixedSize(150, 56)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        t = self._phase()
        painter = painter_for(self)
        try:
            opacity = 1.0
            if 3000 <= t < 3240:
                opacity = 1.0 - (t - 3000) / 240
            elif t >= 3240:
                opacity = 0.0
            elif t < 160:
                opacity = t / 160
            if opacity <= 0:
                return
            painter.setOpacity(opacity)
            height = 28.0
            width = 124.0
            body = QtCore.QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
            shadow = QtGui.QColor(theme.PILL_SHADOW)
            for spread, alpha in ((6, 0.05), (3, 0.09)):
                colour = QtGui.QColor(shadow)
                colour.setAlphaF(alpha)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(colour)
                painter.drawRoundedRect(body.adjusted(-spread / 2, 0, spread / 2, spread), height / 2 + spread / 2, height / 2 + spread / 2)
            painter.setBrush(theme.PILL_FILL)
            painter.drawRoundedRect(body, height / 2, height / 2)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(theme.PILL_EDGE, 1.0))
            painter.drawRoundedRect(body.adjusted(0.5, 0.5, -0.5, -0.5), height / 2 - 0.5, height / 2 - 0.5)
            cy = body.center().y()
            if t < 2200:
                bar = 2.4
                gap = 2.4
                total = 15 * bar + 14 * gap
                x = round(body.center().x() - total / 2)
                energy = 0.55 + 0.45 * math.sin(t / 260.0) ** 2
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(theme.PILL_BAR)
                for index, rest in enumerate(HINT_POSE):
                    wobble = 0.7 + 0.3 * math.sin(t / (140 + index * 23) + index * 1.7)
                    h = max(2.4, min(16.0, rest * 0.8 * energy * wobble))
                    painter.drawRoundedRect(QtCore.QRectF(x + index * (bar + gap), cy - h / 2, bar, h), bar / 2, bar / 2)
            else:
                size = 11.0
                font = theme.pill_font(10, 0.1)
                label = "Processing"
                text_width = QtGui.QFontMetricsF(font).horizontalAdvance(label)
                total = size + 6 + text_width
                x = body.center().x() - total / 2
                ring = QtCore.QRectF(x + 0.75, cy - size / 2 + 0.75, size - 1.5, size - 1.5)
                painter.setPen(QtGui.QPen(theme.PILL_SPINNER_TRACK, 1.3))
                painter.drawEllipse(ring)
                painter.setPen(QtGui.QPen(theme.PILL_SPINNER, 1.3, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
                angle = (t - 2200) / 900.0 * 360.0
                painter.drawArc(ring, int((90 - angle) * 16), -270 * 16)
                painter.setFont(font)
                painter.setPen(theme.PILL_TEXT)
                painter.drawText(QtCore.QRectF(x + size + 6, body.top(), text_width + 2, height), int(QtCore.Qt.AlignmentFlag.AlignVCenter), label)
        finally:
            painter.end()


class TypingHint(QtWidgets.QWidget):
    """A text field where the words appear once the pill has finished."""

    def __init__(self, phase: Callable[[], float], *, still: bool = False, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._phase = phase
        self._still = still
        self.setFixedSize(168, 40)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        current = style.palette()
        t = self._phase()
        painter = painter_for(self)
        try:
            field = QtCore.QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
            painter.setPen(QtGui.QPen(current.color("control_border"), 1.0))
            painter.setBrush(current.color("card"))
            painter.drawRoundedRect(field, 6, 6)
            font = style.font("lead")
            painter.setFont(font)
            if self._still:
                count = len(HINT_TEXT)
            elif t < 3100:
                count = 0
            else:
                count = min(len(HINT_TEXT), int((t - 3100) / 45))
            text = HINT_TEXT[:count]
            metrics = QtGui.QFontMetricsF(font)
            painter.setPen(current.color("text"))
            left = 12.0
            painter.drawText(QtCore.QRectF(left, 0, field.width() - left, self.height()), int(QtCore.Qt.AlignmentFlag.AlignVCenter), text)
            caret_on = self._still or int(t / 500) % 2 == 0 or 3100 <= t < 3900
            if caret_on:
                x = left + metrics.horizontalAdvance(text) + 1
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(current.color("accent"))
                painter.drawRect(QtCore.QRectF(x, (self.height() - 18) / 2, 1.5, 18))
        finally:
            painter.end()


class StepCard(QtWidgets.QFrame):
    def __init__(self, number: int, title: str, text: str, art: QtWidgets.QWidget, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 14)
        layout.setSpacing(0)
        stage = QtWidgets.QFrame(self)
        stage.setProperty("role", "art")
        stage.setFixedHeight(104)
        stage_layout = QtWidgets.QHBoxLayout(stage)
        stage_layout.setContentsMargins(8, 8, 8, 8)
        stage_layout.addStretch(1)
        art.setParent(stage)
        stage_layout.addWidget(art, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        stage_layout.addStretch(1)
        layout.addWidget(stage)
        layout.addSpacing(12)
        heading = QtWidgets.QHBoxLayout()
        heading.setContentsMargins(14, 0, 14, 0)
        heading.setSpacing(8)
        badge = make_label(str(number), "caption_strong", parent=self)
        badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(22, 22)
        badge.setProperty("role", "stepnumber")
        heading.addWidget(badge)
        heading.addWidget(make_label(title, "body_strong", parent=self), 1)
        layout.addLayout(heading)
        layout.addSpacing(6)
        body = make_label(text, "caption", "secondary", wrap=True, parent=self)
        body.setContentsMargins(14, 0, 14, 0)
        layout.addWidget(body)
        layout.addStretch(1)


# The page -----------------------------------------------------------------------------------------------


class WelcomePage(QtWidgets.QDialog):
    """First launch only, and reachable from the General page afterwards."""

    def __init__(
        self,
        *,
        config: ConfigStore,
        hotkey: Any,
        devices: Callable[[], list] | None = None,
        recorder_factory: Callable[..., Any] | None = None,
        on_change_hotkey: Callable[[], None] | None = None,
        notify: Notify | None = None,
        reduced_motion: bool | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._hotkey = hotkey
        self._on_change_hotkey = on_change_hotkey
        self._notify = notify or default_notify
        self._reduced_motion = theme.reduced_motion() if reduced_motion is None else bool(reduced_motion)
        self._phase = 1200.0 if self._reduced_motion else 0.0
        self.setObjectName("WelcomePage")
        self.setWindowTitle(f"Welcome to {theme.APP_NAME}")
        current = style.palette()
        self.setWindowIcon(brand.app_icon(current.color("brand_a"), current.color("brand_b")))
        self.resize(840, 700)
        self.setMinimumSize(720, 600)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QtWidgets.QFrame(self)
        body.setObjectName("spellsWelcomeBody")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        self.steps = QtWidgets.QStackedWidget(body)
        body_layout.addWidget(self.steps)
        outer.addWidget(body, 1)

        intro = ScrollPage("", "", self.steps)
        intro.title_label.hide()
        intro.body_layout.setContentsMargins(40, 12, 40, 24)
        intro.body_layout.insertStretch(0, 1)
        hero = QtWidgets.QWidget(intro.body)
        hero_layout = QtWidgets.QHBoxLayout(hero)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(20)
        self.logo = LogoMark(68, hero)
        hero_layout.addWidget(self.logo)
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(4)
        titles.addWidget(make_label(f"Welcome to {theme.APP_NAME}", "hero", parent=hero))
        titles.addWidget(
            make_label("Dictate into any app. Everything runs on this computer; nothing leaves it.", "lead", "secondary", wrap=True, parent=hero)
        )
        hero_layout.addLayout(titles, 1)
        intro.add_widget(hero, spacing_before=0)
        steps_row = QtWidgets.QWidget(intro.body)
        steps_layout = QtWidgets.QHBoxLayout(steps_row)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(12)
        self.hotkey_art = KeycapRow((), large=True)
        self.pill_hint = PillHint(lambda: self._phase)
        self.typing_hint = TypingHint(lambda: self._phase, still=self._reduced_motion)
        steps_layout.addWidget(StepCard(1, "Hold the hotkey", "Click where the text should go, then hold the keys.", self.hotkey_art, steps_row), 1)
        steps_layout.addWidget(StepCard(2, "Speak", "The pill at the bottom of the screen shows that Spells is listening.", self.pill_hint, steps_row), 1)
        steps_layout.addWidget(StepCard(3, "Release", "The cleaned-up text appears at the cursor.", self.typing_hint, steps_row), 1)
        intro.add_widget(steps_row, spacing_before=28)
        hotkey_card = Card(intro.body)
        self.hotkey_label = KeycapRow((), parent=hotkey_card)
        self.change_button = make_button("Change", parent=hotkey_card)
        self.change_button.clicked.connect(self._change_hotkey)
        hotkey_row = SettingRow(
            "Your hotkey",
            "Tap it twice to keep recording hands-free; press it again to finish.",
            self.hotkey_label,
            glyph=style.Glyph.KEYBOARD,
        )
        hotkey_row.add_control(self.change_button)
        hotkey_card.add_row(hotkey_row)
        intro.add_widget(hotkey_card, spacing_before=16)
        writing = QtWidgets.QWidget(intro.body)
        writing_layout = QtWidgets.QHBoxLayout(writing)
        writing_layout.setContentsMargins(4, 0, 4, 0)
        writing_layout.setSpacing(8)
        writing_layout.addWidget(GlyphLabel(style.Glyph.BRUSH, 14, tone="secondary", parent=writing))
        writing_layout.addWidget(
            make_label(WRITING_NOTE, "caption", "secondary", wrap=True, parent=writing), 1
        )
        intro.add_widget(writing, spacing_before=16)
        privacy = QtWidgets.QWidget(intro.body)
        privacy_layout = QtWidgets.QHBoxLayout(privacy)
        privacy_layout.setContentsMargins(4, 0, 4, 0)
        privacy_layout.setSpacing(8)
        privacy_layout.addWidget(GlyphLabel(style.Glyph.SHIELD, 14, tone="secondary", parent=privacy))
        privacy_layout.addWidget(
            make_label("Speech recognition and cleanup run in local engines, even without a network.", "caption", "secondary", parent=privacy),
            1,
        )
        intro.add_widget(privacy, spacing_before=20)
        recordings = QtWidgets.QWidget(intro.body)
        recordings_layout = QtWidgets.QHBoxLayout(recordings)
        recordings_layout.setContentsMargins(4, 0, 4, 0)
        recordings_layout.setSpacing(8)
        recordings_layout.addWidget(GlyphLabel(style.Glyph.WAVEFORM, 14, tone="secondary", parent=recordings))
        recordings_layout.addWidget(
            make_label(RECORDINGS_NOTE, "caption", "secondary", wrap=True, parent=recordings), 1
        )
        intro.add_widget(recordings, spacing_before=6)
        intro.finish()
        self.steps.addWidget(intro)

        languages = ScrollPage(
            "Which languages do you speak?",
            "Spells listens for these and picks its models for them. You can change them later in settings.",
            self.steps,
        )
        languages.body_layout.setContentsMargins(40, 32, 40, 24)
        self.language_list = LanguageList(show_hotkeys=False, parent=languages.body)
        self.language_list.remove_language.connect(lambda code: self._set_language(code, False))
        languages.add_section("Your languages", self.language_list)
        adder_card = Card(languages.body)
        self.language_adder = LanguageAdder(adder_card)
        self.language_adder.add_language.connect(lambda code: self._set_language(code, True))
        adder_card.add_widget(self.language_adder)
        languages.add_section("Add a language", adder_card)
        languages.finish()
        self.steps.addWidget(languages)

        microphone = ScrollPage("Check your microphone", "Pick the microphone Spells should use, then try a dictation.", self.steps)
        microphone.body_layout.setContentsMargins(40, 32, 40, 24)
        mic_card = Card(microphone.body)
        self.mic_picker = MicPicker(config=config, devices=devices, notify=self._notify, parent=mic_card)
        self.mic_picker.setMinimumWidth(260)
        self.mic_picker.device_changed.connect(self._restart_meter)
        self.meter = MicMeter(recorder_factory=recorder_factory, notify=self._notify, parent=mic_card)
        mic_card.add_row(SettingRow("Microphone", "Speak to see the level move.", self.mic_picker, glyph=style.Glyph.MICROPHONE, below=self.meter))
        microphone.add_widget(mic_card, spacing_before=0)
        try_card = Card(microphone.body, margins=(16, 14, 16, 16))
        try_card.body.setSpacing(8)
        try_card.add_widget(make_label("Try it", "body_strong", parent=try_card))
        self.try_box = QtWidgets.QPlainTextEdit(try_card)
        self.try_box.setFont(style.font("lead"))
        self.try_box.setFixedHeight(96)
        try_card.add_widget(self.try_box)
        microphone.add_widget(try_card, spacing_before=12)
        startup = Card(microphone.body)
        self.autostart = ToggleSwitch(startup)
        self.autostart.setAccessibleName("Start with Windows")
        self.autostart.toggled.connect(self._on_autostart)
        startup.add_row(SettingRow("Start with Windows", "Spells waits in the tray, ready for your hotkey.", self.autostart))
        microphone.add_widget(startup, spacing_before=12)
        self.note = InfoBar(ELEVATED_NOTE, "info", parent=microphone.body)
        microphone.add_widget(self.note, spacing_before=12)
        microphone.finish()
        self.steps.addWidget(microphone)

        footer = QtWidgets.QFrame(self)
        footer.setObjectName("spellsFooter")
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(28, 14, 24, 14)
        footer_layout.setSpacing(8)
        self.dots = StepDots(len(STEP_NAMES), footer)
        footer_layout.addWidget(self.dots)
        footer_layout.addStretch(1)
        self.back_button = make_button("Back", parent=footer)
        self.back_button.setMinimumWidth(96)
        self.back_button.clicked.connect(lambda: self.set_step(self.step() - 1))
        footer_layout.addWidget(self.back_button)
        self.next_button = make_button("Next", "primary", parent=footer)
        self.next_button.setMinimumWidth(96)
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self._next)
        footer_layout.addWidget(self.next_button)
        outer.addWidget(footer)

        self._clock = QtCore.QTimer(self)
        self._clock.setInterval(HINT_FRAME_MS)
        self._clock.timeout.connect(self._tick)

        self.set_step(STEP_INTRO)
        self.apply_settings(config.settings)

    # Steps ---------------------------------------------------------------------------------------

    def step(self) -> int:
        return self.steps.currentIndex()

    def set_step(self, index: int) -> None:
        index = max(0, min(len(STEP_NAMES) - 1, int(index)))
        self.steps.setCurrentIndex(index)
        self.dots.set_current(index)
        self.back_button.setVisible(index > 0)
        self.next_button.setText("Finish" if index == len(STEP_NAMES) - 1 else "Next")
        self._sync_activity()

    def _next(self) -> None:
        if self.step() >= len(STEP_NAMES) - 1:
            self.close()
            return
        self.set_step(self.step() + 1)

    def _sync_activity(self) -> None:
        visible = self.isVisible()
        on_intro = self.step() == STEP_INTRO
        if visible and on_intro and not self._reduced_motion:
            if not self._clock.isActive():
                self._clock.start()
        else:
            self._clock.stop()
        if visible and self.step() == STEP_MICROPHONE:
            if not self.meter.running:
                self.meter.start(self.mic_picker.current_device())
        else:
            self.meter.stop()

    def _tick(self) -> None:
        self._phase = (self._phase + HINT_FRAME_MS) % HINT_CYCLE_MS
        self.pill_hint.update()
        self.typing_hint.update()

    # Settings -------------------------------------------------------------------------------------

    def apply_settings(self, settings: Settings) -> None:
        keys = theme.chord_keys(settings.general.main_chord)
        self.hotkey_label.set_keys(keys)
        self.hotkey_art.set_keys(keys)
        self.try_box.setPlaceholderText(TRY_PLACEHOLDER.format(chord=theme.chord_label(settings.general.main_chord)))
        self.autostart.blockSignals(True)
        try:
            self.autostart.setChecked(settings.general.autostart)
        finally:
            self.autostart.blockSignals(False)
        self.mic_picker.apply_settings(settings)
        self.language_list.apply_settings(settings)
        self.language_adder.apply_settings(settings)

    def _set_language(self, code: str, enabled: bool) -> None:
        set_language_enabled(self._config, self._notify, code, enabled)
        self.apply_settings(self._config.settings)

    def _change_hotkey(self) -> None:
        if self._on_change_hotkey is not None:
            self._on_change_hotkey()

    def _on_autostart(self, checked: bool) -> None:
        def mutate(settings: Settings) -> Settings:
            return replace(settings, general=replace(settings.general, autostart=bool(checked)))

        try:
            self._config.update(mutate)
        except SettingsError as exc:
            self._notify("warning", "Start with Windows", str(exc))

    def _restart_meter(self, device: Any) -> None:
        if self.isVisible() and self.step() == STEP_MICROPHONE:
            self.meter.start(device)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        style.apply_window_chrome(self)
        self._sync_activity()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self._clock.stop()
        self.meter.stop()
        super().hideEvent(event)


__all__ = [
    "ELEVATED_NOTE",
    "STEP_MICROPHONE",
    "STEP_NAMES",
    "MicMeter",
    "MicPicker",
    "WelcomePage",
    "default_notify",
]
