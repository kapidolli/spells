"""The settings window (spec 14.4): a navigation rail and one page per section.

Every change goes through ConfigStore.update() with a mutator that builds a new frozen
Settings; the widgets never hold state of their own. Refreshes from a foreign update arrive
through the bridge (apply_settings), and every refresh path blocks the widget signals so a
refresh never writes back.

The pages are General (hotkey, microphone, behaviour), Languages (languages.py), Cleanup,
Apps, Vocabulary, History, Diagnostics (diagnostics.py) and About (about.py). The Apps page
edits rules that pick a built-in profile and a delivery method; a rule has no tone of its
own (B3-7). Tones are edited on the Cleanup page per profile name.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spells import modelcatalog
from spells.config import (
    RETENTION_CHOICES,
    ConfigStore,
    ProfileRule,
    Replacement,
    Settings,
    SettingsError,
    Snippet,
    TermEntry,
    Vocabulary,
)
from spells.history import (
    EXPORT_FORMATS,
    EXPORT_LABELS,
    AudioPolicy,
    export_file_name,
    export_to_path,
    row_stats,
    wav_duration_s,
)
from spells.models import Chord, ChordMode, DeliveryMethod
from spells.profiles import BUILTIN_PROFILES
from spells.quality import CheckResult, check_label, summarize
from spells.ui import brand, style, theme
from spells.ui.about import AboutPage
from spells.ui.checks import CheckJob, CheckRunner
from spells.ui.diagnostics import DiagnosticsTab
from spells.ui.hotkeys import ChordCaptureDialog, HotkeyRecorder, fold_keys, probe_arguments
from spells.ui.languages import LanguagesPage, hardware_for
from spells.ui.recordings import (
    KEEP_AUDIO_HINT,
    KEEP_AUDIO_TITLE,
    RECORDINGS_NOTE,
    PlayCell,
    WavPlayer,
    default_reveal,
)
from spells.ui.welcome import MicMeter, MicPicker, Notify, default_notify
from spells.ui.widgets import (
    Badge,
    Card,
    CellHost,
    Chip,
    ComboBox,
    FlowLayout,
    GlyphLabel,
    InfoBar,
    NavRail,
    ScrollPage,
    SegmentedControl,
    SettingRow,
    SpinBox,
    ToggleSwitch,
    make_button,
    make_label,
    set_prop,
)

log = logging.getLogger(__name__)

RETENTION_LABELS = {
    "100": "Last 100",
    "7d": "7 days",
    "30d": "30 days",
    "off": "Off",
}
QUALITY_COLUMN = 3
SOUND_COLUMN = 4
CHECK_BATCH = 20
EMPTY_STAT = "No data"
SAID_TITLE = "What you said"
CLEANED_TITLE = "Cleaned text"
INSTRUCTION_TITLE = "What you asked for"
WRITTEN_TITLE = "What was written"
WRITING_TONE_HINT = (
    "Composing uses the tone of the app you write into, the same tones as cleanup below."
)
NO_WRITING_MODEL = (
    "No installed model writes text, so the compose and edit hotkeys deliver nothing."
)
CHECK_UNAVAILABLE = (
    "The cleanup model is not serving right now, so nothing was checked. Try again once the "
    "engines are ready."
)
QUALITY_BADGES = {
    "good": ("Good", "ok"),
    "uncertain": ("Uncertain", "caution"),
    "poor": ("Poor", "critical"),
}
EXPORT_FILTERS = {
    "json": "JSON files (*.json)",
    "csv": "CSV files (*.csv)",
    "markdown": "Markdown files (*.md)",
}
DELIVERY_LABELS = {DeliveryMethod.PASTE: "Paste", DeliveryMethod.TYPE: "Type"}
PAGES: tuple[tuple[str, str, str, bool], ...] = (
    ("general", "General", style.Glyph.SETTINGS, False),
    ("languages", "Languages", style.Glyph.GLOBE, False),
    ("cleanup", "Writing", style.Glyph.BRUSH, False),
    ("apps", "Apps", style.Glyph.APPS, False),
    ("vocabulary", "Vocabulary", style.Glyph.BOOK, False),
    ("history", "History", style.Glyph.HISTORY, False),
    ("diagnostics", "Diagnostics", style.Glyph.PULSE, True),
    ("about", "About", style.Glyph.INFO, True),
)
TAB_NAMES = tuple(key for key, _label, _glyph, _bottom in PAGES)
LIST_SAVE_DELAY_MS = 600
PICK_DELAY_MS = 3000

Confirm = Callable[[str, str], bool]


MIC_DEVICE_HINT = "Spells records from this microphone."
MIC_DEVICE_HINT_PATH = "Spells records from this microphone through {host_api}, at its own sample rate."


def mic_device_hint(host_api: Callable[[], str] | None = None) -> str:
    """The Input device row's description, naming the host API capture goes through."""
    reader = host_api or _default_host_api
    try:
        name = str(reader() or "")
    except Exception:
        log.exception("could not read the capture host API")
        name = ""
    return MIC_DEVICE_HINT_PATH.format(host_api=name) if name else MIC_DEVICE_HINT


def _default_host_api() -> str:
    from spells.audio import capture_host_api

    return capture_host_api()


def default_confirm(title: str, text: str) -> bool:
    answer = QtWidgets.QMessageBox.question(
        None,
        title,
        text,
        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        QtWidgets.QMessageBox.StandardButton.No,
    )
    return answer == QtWidgets.QMessageBox.StandardButton.Yes


def _default_window_picker() -> tuple[str, str]:
    from spells.win32.window import foreground_hwnd, window_process_name, window_title

    hwnd = foreground_hwnd()
    return window_process_name(hwnd), window_title(hwnd)


def _split_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


class _Blocked:
    """Block a widget's signals for the duration of a refresh."""

    def __init__(self, *widgets: QtCore.QObject) -> None:
        self._widgets = widgets

    def __enter__(self) -> None:
        for widget in self._widgets:
            widget.blockSignals(True)

    def __exit__(self, *exc: object) -> None:
        for widget in self._widgets:
            widget.blockSignals(False)


def row_text(entry: Any) -> str:
    """The text a history row shows: what cleanup produced, or what the model wrote."""
    if getattr(entry, "wrote", False):
        return entry.delivered_text
    return entry.cleaned_text


def _table(columns: list[str], parent: QtWidgets.QWidget, *, embedded: bool = True) -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget(0, len(columns), parent)
    table.setHorizontalHeaderLabels(columns)
    header = table.horizontalHeader()
    header.setStretchLastSection(True)
    header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
    header.setHighlightSections(False)
    header.setFont(style.font("caption_strong"))
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(40)
    table.setShowGrid(False)
    table.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setWordWrap(False)
    table.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
    table.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    table.setMouseTracking(True)
    if embedded:
        table.setProperty("role", "embedded")
    return table


def _set_rows(table: QtWidgets.QTableWidget, rows: list[list[str]]) -> None:
    table.setRowCount(0)
    for cells in rows:
        row = table.rowCount()
        table.insertRow(row)
        for column, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)
            item.setToolTip(text)
            table.setItem(row, column, item)


def _fit_table(table: QtWidgets.QTableWidget, minimum_rows: int = 1, maximum_rows: int = 8) -> None:
    rows = max(minimum_rows, min(maximum_rows, table.rowCount()))
    height = table.horizontalHeader().sizeHint().height() + rows * table.verticalHeader().defaultSectionSize() + 4
    table.setFixedHeight(height)


class _EmptyState(QtWidgets.QWidget):
    def __init__(self, glyph: str, title: str, text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 28, 16, 28)
        layout.setSpacing(6)
        icon = GlyphLabel(glyph, 28, tone="tertiary", parent=self)
        layout.addWidget(icon, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(4)
        heading = make_label(title, "body_strong", parent=self)
        heading.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(heading)
        body = make_label(text, "caption", "secondary", wrap=True, parent=self)
        body.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(body)


def _input_row(parent: QtWidgets.QWidget, *widgets: QtWidgets.QWidget, stretch: tuple[int, ...] = ()) -> QtWidgets.QWidget:
    row = QtWidgets.QWidget(parent)
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, stretch[index] if index < len(stretch) else 0)
    return row


def _line_edit(placeholder: str, parent: QtWidgets.QWidget) -> QtWidgets.QLineEdit:
    edit = QtWidgets.QLineEdit(parent)
    edit.setFont(style.font("body"))
    edit.setPlaceholderText(placeholder)
    return edit


# General ----------------------------------------------------------------------------------------------


class GeneralTab(ScrollPage):
    welcome_requested = QtCore.Signal()

    def __init__(
        self,
        *,
        config: ConfigStore,
        hotkey: Any,
        probe: Callable[[int, int], bool] | None,
        notify: Notify,
        devices: Callable[[], list] | None,
        recorder_factory: Callable[..., Any] | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("General", "Your hotkey, your microphone and how Spells behaves.", parent)
        self._config = config
        self._hotkey = hotkey
        self._notify = notify
        self._loading = False

        if config.notice:
            self.notice = InfoBar(config.notice, "caution", parent=self.body)
            self.add_widget(self.notice, spacing_before=0)
            self.body_layout.addSpacing(16)

        dictation = Card(self.body)
        self.main_recorder = HotkeyRecorder(
            config=config,
            hotkey=hotkey,
            probe=probe,
            notify=notify,
            chord_of=lambda s: s.general.main_chord,
            with_chord=lambda s, keys: replace(s, general=replace(s.general, main_chord=Chord(keys=keys))),
            parent=dictation,
        )
        dictation.add_row(
            SettingRow(
                "Dictation hotkey",
                "Hold it and speak, release to insert the text. Tap it twice to keep recording hands-free, then press it again to finish.",
                self.main_recorder,
                glyph=style.Glyph.KEYBOARD,
            )
        )
        self.add_section("Dictation", dictation)

        writing = Card(self.body)
        self.compose_recorder = HotkeyRecorder(
            config=config,
            hotkey=hotkey,
            probe=probe,
            notify=notify,
            chord_of=lambda s: s.general.compose_chord,
            with_chord=lambda s, keys: replace(
                s,
                general=replace(
                    s.general, compose_chord=Chord(keys=keys, mode=ChordMode.COMPOSE)
                ),
            ),
            parent=writing,
        )
        writing.add_row(
            SettingRow(
                "Write hotkey",
                "Hold it and say what to write, such as an email to somebody. The finished text appears where the cursor is.",
                self.compose_recorder,
                glyph=style.Glyph.KEYBOARD,
            )
        )
        self.edit_recorder = HotkeyRecorder(
            config=config,
            hotkey=hotkey,
            probe=probe,
            notify=notify,
            chord_of=lambda s: s.general.edit_chord,
            with_chord=lambda s, keys: replace(
                s,
                general=replace(s.general, edit_chord=Chord(keys=keys, mode=ChordMode.EDIT)),
            ),
            parent=writing,
        )
        writing.add_row(
            SettingRow(
                "Edit hotkey",
                "Select some text, hold it and say what to change, such as make this shorter. The selection is replaced. With nothing selected it writes instead.",
                self.edit_recorder,
                glyph=style.Glyph.KEYBOARD,
            )
        )
        self.add_section(
            "Writing",
            writing,
            description="Both are empty until you record them, and neither can share a chord with the others.",
        )

        mic = Card(self.body)
        self.mic_picker = MicPicker(config=config, devices=devices, notify=notify, parent=mic)
        self.mic_picker.setMinimumWidth(300)
        self.mic_picker.setMaximumWidth(380)
        mic.add_row(SettingRow("Input device", mic_device_hint(), self.mic_picker, glyph=style.Glyph.MICROPHONE))
        self.meter = MicMeter(recorder_factory=recorder_factory, notify=notify, parent=mic)
        self.test_button = make_button("Test", parent=mic)
        self.test_button.setCheckable(True)
        self.test_button.setMinimumWidth(96)
        self.test_button.toggled.connect(self._toggle_test)
        mic.add_row(SettingRow("Test your microphone", "Speak and watch the level move.", self.test_button, below=self.meter))
        self.mic_picker.device_changed.connect(self._device_changed)
        self.add_section("Microphone", mic)

        behaviour = Card(self.body)
        self.autostart = ToggleSwitch(behaviour)
        self.autostart.setAccessibleName("Start with Windows")
        self.autostart.toggled.connect(lambda v: self._set_general(autostart=bool(v)))
        behaviour.add_row(SettingRow("Start with Windows", "Spells starts in the background when you sign in.", self.autostart))
        self.keep_mic_warm = ToggleSwitch(behaviour)
        self.keep_mic_warm.setAccessibleName("Keep the microphone warm")
        self.keep_mic_warm.toggled.connect(lambda v: self._set_general(keep_mic_warm=bool(v)))
        behaviour.add_row(
            SettingRow("Keep the microphone warm", "Recording starts faster; the microphone stays open while Spells runs.", self.keep_mic_warm)
        )
        self.sounds = ToggleSwitch(behaviour)
        self.sounds.setAccessibleName("Start and stop sounds")
        self.sounds.toggled.connect(lambda v: self._set_general(sounds=bool(v)))
        behaviour.add_row(SettingRow("Start and stop sounds", "A short sound when recording starts and stops.", self.sounds))
        self.live_text = ToggleSwitch(behaviour)
        self.live_text.setAccessibleName("Type the words while I speak")
        self.live_text.toggled.connect(lambda v: self._set_general(live_text=bool(v)))
        behaviour.add_row(
            SettingRow(
                "Type the words while I speak",
                "A draft appears in the box as you talk and corrects itself; the finished "
                "text replaces it when you let go. Needs a speech model fast enough on "
                "this computer to keep up, and stays off otherwise.",
                self.live_text,
            )
        )
        self.live_text_everywhere = ToggleSwitch(behaviour)
        self.live_text_everywhere.setAccessibleName("Also in code editors and terminals")
        self.live_text_everywhere.toggled.connect(
            lambda v: self._set_general(live_text_everywhere=bool(v))
        )
        behaviour.add_row(
            SettingRow(
                "Also in code editors and terminals",
                "Off by default. Autocomplete and automatic indenting fight the draft, so "
                "what you see while speaking can come out wrong in those windows.",
                self.live_text_everywhere,
            )
        )
        self.idle_unload = SpinBox(behaviour)
        self.idle_unload.setRange(0, 24 * 60)
        self.idle_unload.setSuffix(" min")
        self.idle_unload.setSpecialValueText("Never")
        self.idle_unload.setMinimumWidth(120)
        self.idle_unload.editingFinished.connect(self._idle_unload_edited)
        behaviour.add_row(
            SettingRow("Free GPU memory when idle", "Unload the engines after this many idle minutes. They load again when you dictate.", self.idle_unload)
        )
        self.add_section("Behaviour", behaviour)

        tour = Card(self.body)
        self.welcome_button = make_button("Show welcome", parent=tour)
        self.welcome_button.clicked.connect(self.welcome_requested.emit)
        tour.add_row(SettingRow("Welcome tour", "See how dictation works and check your microphone again.", self.welcome_button, glyph=style.Glyph.INFO))
        self.add_section("Help", tour)
        self.finish()

        self.apply_settings(config.settings)

    def apply_settings(self, settings: Settings) -> None:
        self._loading = True
        try:
            self.main_recorder.apply_settings(settings)
            self.compose_recorder.apply_settings(settings)
            self.edit_recorder.apply_settings(settings)
            self.mic_picker.apply_settings(settings)
            with _Blocked(
                self.autostart,
                self.keep_mic_warm,
                self.sounds,
                self.live_text,
                self.live_text_everywhere,
                self.idle_unload,
            ):
                self.autostart.setChecked(settings.general.autostart)
                self.keep_mic_warm.setChecked(settings.general.keep_mic_warm)
                self.sounds.setChecked(settings.general.sounds)
                self.live_text.setChecked(settings.general.live_text)
                self.live_text_everywhere.setChecked(settings.general.live_text_everywhere)
                self.live_text_everywhere.setEnabled(settings.general.live_text)
                self.idle_unload.setValue(settings.general.idle_unload_minutes)
        finally:
            self._loading = False

    def _update(self, mutator: Callable[[Settings], Settings], title: str) -> bool:
        try:
            settings = self._config.update(mutator)
        except SettingsError as exc:
            self._notify("warning", title, str(exc))
            self.apply_settings(self._config.settings)
            return False
        self.apply_settings(settings)
        return True

    def _set_general(self, **changes: Any) -> None:
        if self._loading:
            return
        self._update(lambda s: replace(s, general=replace(s.general, **changes)), "Settings")

    def _idle_unload_edited(self) -> None:
        value = int(self.idle_unload.value())
        if value != self._config.settings.general.idle_unload_minutes:
            self._set_general(idle_unload_minutes=value)

    def _toggle_test(self, on: bool) -> None:
        self.test_button.setText("Stop" if on else "Test")
        if on:
            self.meter.start(self.mic_picker.current_device())
        else:
            self.meter.stop()

    def _device_changed(self, device: Any) -> None:
        if self.test_button.isChecked():
            self.meter.start(device)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self.test_button.setChecked(False)
        super().hideEvent(event)


# Cleanup ----------------------------------------------------------------------------------------------


class WritingModelCard(Card):
    """Which model writes, why it was chosen, and where composing takes its tone from.

    The choice is the catalog's compose choice for the enabled languages and this hardware
    (spec 13, B5-44); it carries its own plain reason, so this card only shows it.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._key: tuple | None = None
        self.choice: Any = None
        self.model_row: SettingRow | None = None

    def refresh(self, languages: Sequence[str], hardware: Any, *, force: bool = False) -> None:
        key = (tuple(languages), hardware)
        if key == self._key and not force:
            return
        self._key = key
        self.clear()
        self.model_row = None
        self.choice = None
        notes: list[str] = []
        try:
            selection = modelcatalog.select_models(list(languages), hardware)
        except Exception as exc:
            log.exception("select_models failed for the writing card")
            self.add_row(
                SettingRow(
                    "The writing model could not be chosen",
                    str(exc),
                    glyph=style.Glyph.WARNING,
                    badge_glyph=True,
                )
            )
            return
        self.choice = selection.compose
        if self.choice is not None:
            row = SettingRow(
                self.choice.display_name,
                self.choice.reason,
                glyph=style.Glyph.BRUSH,
                badge_glyph=True,
            )
            row.title_label.setFont(style.font("body_strong"))
            installed = bool(getattr(self.choice, "installed", True))
            row.add_control(
                Badge(
                    "Installed" if installed else "Not installed",
                    "ok" if installed else "caution",
                )
            )
            self.model_row = row
            self.add_row(row)
        else:
            row = SettingRow(
                "No writing model",
                NO_WRITING_MODEL,
                glyph=style.Glyph.BRUSH,
                badge_glyph=True,
            )
            row.add_control(Badge("Off", "muted"))
            self.model_row = row
            self.add_row(row)
            notes = [note for note in selection.notes if "writ" in note or "compose" in note]
        self.add_row(SettingRow("Tone", WRITING_TONE_HINT))
        for note in notes:
            label = make_label(note, "caption", "secondary", wrap=True)
            holder = QtWidgets.QWidget()
            holder_layout = QtWidgets.QHBoxLayout(holder)
            holder_layout.setContentsMargins(68, 10, 16, 12)
            holder_layout.addWidget(label)
            self.add_row(holder)


class CleanupTab(ScrollPage):
    def __init__(
        self,
        *,
        config: ConfigStore,
        notify: Notify,
        gpu_selection: Any = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(
            "Writing",
            "A local language model tidies each transcript, and writes the text your write and edit hotkeys ask for.",
            parent,
        )
        self._config = config
        self._notify = notify
        self._gpu = gpu_selection
        self._loading = False
        settings = config.settings
        self.original_fillers = {code: list(words) for code, words in settings.cleanup.fillers.items()}

        self.writing = WritingModelCard(self.body)
        self.add_section(
            "Writing model",
            self.writing,
            description="What the write and edit hotkeys use. Speed does not decide it: composing is not on the dictation path.",
        )

        main = Card(self.body)
        self.enabled = ToggleSwitch(main)
        self.enabled.setAccessibleName("Clean up transcripts")
        self.enabled.toggled.connect(self._on_enabled)
        main.add_row(SettingRow("Clean up transcripts", "Uses the local cleanup model. Turn it off to insert exactly what was recognised.", self.enabled, glyph=style.Glyph.BRUSH))
        self.timeout = SpinBox(main)
        self.timeout.setRange(200, 30000)
        self.timeout.setSingleStep(100)
        self.timeout.setSuffix(" ms")
        self.timeout.setMinimumWidth(120)
        self.timeout.editingFinished.connect(self._on_timeout)
        main.add_row(SettingRow("Timeout", "Base limit before using the raw transcript. On the processor, longer text gets 5 to 15 seconds unless this limit is higher.", self.timeout))
        self.add_section("Cleanup", main)

        tones = Card(self.body)
        self.tone_edits: dict[str, QtWidgets.QLineEdit] = {}
        for name in settings.cleanup.tones:
            edit = _line_edit("Describe the tone", tones)
            edit.setMinimumWidth(280)
            edit.editingFinished.connect(lambda n=name: self._tone_edited(n))
            tones.add_row(SettingRow(name, "", edit, wide_control=True))
            self.tone_edits[name] = edit
        self.add_section("Tone per profile", tones, description="The only place a tone is edited. App rules on the Apps page pick the profile.")

        lists = Card(self.body)
        columns = QtWidgets.QWidget(lists)
        columns_layout = QtWidgets.QHBoxLayout(columns)
        columns_layout.setContentsMargins(16, 14, 16, 16)
        columns_layout.setSpacing(16)
        fillers_col = QtWidgets.QVBoxLayout()
        fillers_col.setSpacing(6)
        fillers_col.addWidget(make_label("Fillers", "body_strong", parent=columns))
        fillers_col.addWidget(make_label("Words dropped from transcripts.", "caption", "secondary", parent=columns))
        self.fillers = QtWidgets.QPlainTextEdit(columns)
        self.fillers.setFont(style.font("body"))
        self.fillers.setMinimumHeight(170)
        fillers_col.addWidget(self.fillers)
        corrections_col = QtWidgets.QVBoxLayout()
        corrections_col.setSpacing(6)
        corrections_col.addWidget(make_label("Self-correction cues", "body_strong", parent=columns))
        corrections_col.addWidget(make_label("Phrases that mark a correction, such as \"I mean\".", "caption", "secondary", parent=columns))
        self.corrections = QtWidgets.QPlainTextEdit(columns)
        self.corrections.setFont(style.font("body"))
        self.corrections.setMinimumHeight(170)
        corrections_col.addWidget(self.corrections)
        columns_layout.addLayout(fillers_col)
        columns_layout.addLayout(corrections_col)
        lists.add_widget(columns)
        header = self.add_section("Fillers and self-corrections", lists, description="One per line, per language. Saved as you type.")
        self.language = ComboBox(header)
        self.language.setMinimumWidth(180)
        self.language.currentIndexChanged.connect(self._language_changed)
        header.add_trailing(self.language)
        self.finish()

        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(LIST_SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self.save_lists)
        self.fillers.textChanged.connect(self._lists_edited)
        self.corrections.textChanged.connect(self._lists_edited)

        self.apply_settings(settings)

    @property
    def gpu_selection(self) -> Any:
        return self._gpu() if callable(self._gpu) else self._gpu

    def refresh_models(self, *, force: bool = False) -> None:
        """Re-run the catalog choice for the enabled languages and this hardware."""
        languages = self._config.settings.general.enabled_languages
        self.writing.refresh(languages, hardware_for(self.gpu_selection), force=force)

    def apply_settings(self, settings: Settings) -> None:
        self._loading = True
        try:
            self.refresh_models()
            with _Blocked(self.enabled, self.timeout):
                self.enabled.setChecked(settings.cleanup.enabled)
                self.timeout.setValue(settings.cleanup.timeout_ms)
            for name, edit in self.tone_edits.items():
                if not edit.hasFocus():
                    edit.setText(settings.cleanup.tones.get(name, ""))
                    edit.setCursorPosition(0)
            current = self.language.currentData()
            with _Blocked(self.language):
                self.language.clear()
                for code in settings.general.enabled_languages:
                    self.language.addItem(theme.language_name(code), code)
                index = self.language.findData(current) if current else 0
                self.language.setCurrentIndex(max(0, index))
            self._load_lists(settings)
        finally:
            self._loading = False

    def _load_lists(self, settings: Settings) -> None:
        code = self.language.currentData()
        was_loading = self._loading
        self._loading = True
        try:
            with _Blocked(self.fillers, self.corrections):
                self.fillers.setPlainText("\n".join(settings.cleanup.fillers.get(code, [])) if code else "")
                self.corrections.setPlainText("\n".join(settings.cleanup.corrections.get(code, [])) if code else "")
        finally:
            self._loading = was_loading
        self._save_timer.stop()

    def _update(self, mutator: Callable[[Settings], Settings], title: str) -> None:
        try:
            settings = self._config.update(mutator)
        except SettingsError as exc:
            self._notify("warning", title, str(exc))
            self.apply_settings(self._config.settings)
            return
        self.apply_settings(settings)

    def _on_enabled(self, checked: bool) -> None:
        if self._loading:
            return
        self._update(lambda s: replace(s, cleanup=replace(s.cleanup, enabled=bool(checked))), "Cleanup")

    def _on_timeout(self) -> None:
        if self._loading:
            return
        value = int(self.timeout.value())
        if value != self._config.settings.cleanup.timeout_ms:
            self._update(lambda s: replace(s, cleanup=replace(s.cleanup, timeout_ms=value)), "Cleanup")

    def _tone_edited(self, name: str) -> None:
        self.set_tone(name, self.tone_edits[name].text())

    def set_tone(self, name: str, tone: str) -> None:
        if self._config.settings.cleanup.tones.get(name) == tone:
            return

        def mutate(s: Settings) -> Settings:
            tones = dict(s.cleanup.tones)
            tones[name] = tone
            return replace(s, cleanup=replace(s.cleanup, tones=tones))

        self._update(mutate, "Tone")

    def _language_changed(self, _index: int) -> None:
        if self._loading:
            return
        self._load_lists(self._config.settings)

    def _lists_edited(self) -> None:
        if self._loading:
            return
        self._save_timer.start()

    def save_lists(self) -> None:
        """Write the visible filler and correction lists for the current language."""
        self._save_timer.stop()
        code = self.language.currentData()
        if not code:
            return
        fillers = _lines(self.fillers.toPlainText())
        corrections = _lines(self.corrections.toPlainText())
        settings = self._config.settings
        if fillers == settings.cleanup.fillers.get(code, []) and corrections == settings.cleanup.corrections.get(code, []):
            return

        def mutate(s: Settings) -> Settings:
            new_fillers = {k: list(v) for k, v in s.cleanup.fillers.items()}
            new_corrections = {k: list(v) for k, v in s.cleanup.corrections.items()}
            new_fillers[code] = fillers
            new_corrections[code] = corrections
            return replace(s, cleanup=replace(s.cleanup, fillers=new_fillers, corrections=new_corrections))

        self._update(mutate, "Cleanup lists")


# Apps ------------------------------------------------------------------------------------------------------


class RuleEditor(QtWidgets.QDialog):
    """One Apps rule: a name, process names, title substrings, a profile, a delivery method."""

    def __init__(
        self, *, profile_names: list[str], rule: ProfileRule | None = None, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("App rule")
        self.setMinimumWidth(520)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(0)
        layout.addWidget(make_label("Edit rule" if rule is not None else "New rule", "subtitle", parent=self))
        layout.addSpacing(6)
        layout.addWidget(
            make_label(
                "A rule matches by process name, by a window title substring, or both. With both set, both must match.",
                "body",
                "secondary",
                wrap=True,
                parent=self,
            )
        )
        layout.addSpacing(18)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(12)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.name = _line_edit("Slack", self)
        form.addRow(make_label("Name", parent=self), self.name)
        self.process = _line_edit("slack.exe, teams.exe", self)
        form.addRow(make_label("Process names", parent=self), self.process)
        self.title = _line_edit("a part of the window title", self)
        form.addRow(make_label("Title contains", parent=self), self.title)
        self.profile = ComboBox(self)
        for profile_name in profile_names:
            self.profile.addItem(profile_name)
        form.addRow(make_label("Profile", parent=self), self.profile)
        self.delivery = SegmentedControl(self)
        for method in DeliveryMethod:
            self.delivery.addItem(DELIVERY_LABELS.get(method, method.value.capitalize()), method)
        form.addRow(make_label("Delivery", parent=self), self.delivery)
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addWidget(make_label("Separate several names with commas.", "caption", "tertiary", parent=self))
        layout.addSpacing(22)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        self.save_button = make_button("Save", "primary", parent=self)
        self.save_button.setMinimumWidth(96)
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.accept)
        cancel = make_button("Cancel", parent=self)
        cancel.setMinimumWidth(96)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(self.save_button)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        if rule is not None:
            self.name.setText(rule.name)
            self.process.setText(", ".join(rule.match_process))
            self.title.setText(", ".join(rule.match_title))
            self.profile.setCurrentText(rule.profile.name)
            self.delivery.setCurrentIndex(self.delivery.findData(rule.profile.delivery))

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        style.apply_window_chrome(self)

    def rule(self) -> ProfileRule:
        profile_name = self.profile.currentText()
        base = BUILTIN_PROFILES.get(profile_name, BUILTIN_PROFILES["Default"])
        method = self.delivery.currentData() or DeliveryMethod.PASTE
        profile = replace(base, delivery=DeliveryMethod(method))
        return ProfileRule(
            name=self.name.text().strip(),
            match_process=_split_list(self.process.text()),
            match_title=_split_list(self.title.text()),
            profile=profile,
        )


class AppsTab(ScrollPage):
    def __init__(
        self,
        *,
        config: ConfigStore,
        notify: Notify,
        window_picker: Callable[[], tuple[str, str]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(
            "Apps",
            "Rules pick a profile and a delivery method per app, by process name, by a window title substring, or both. Built-in rules apply when no rule here matches.",
            parent,
        )
        self._config = config
        self._notify = notify
        self.window_picker = window_picker or _default_window_picker
        card = Card(self.body)
        self.table = _table(["Name", "Process", "Title contains", "Profile", "Delivery"], card)
        self.table.doubleClicked.connect(lambda _index: self._edit_selected())
        self.table.itemSelectionChanged.connect(self._selection_changed)
        card.add_widget(self.table)
        self.empty = _EmptyState(style.Glyph.APPS, "No rules yet", "Add a rule, or click a running window to fill one in.", card)
        card.add_widget(self.empty)
        actions = QtWidgets.QWidget(card)
        actions_layout = QtWidgets.QHBoxLayout(actions)
        actions_layout.setContentsMargins(12, 8, 12, 10)
        actions_layout.setSpacing(4)
        self.edit_button = make_button("Edit", "subtle", glyph=style.Glyph.EDIT, parent=actions)
        self.edit_button.clicked.connect(self._edit_selected)
        self.remove_button = make_button("Remove", "subtle", glyph=style.Glyph.DELETE, parent=actions)
        self.remove_button.clicked.connect(self._remove_selected)
        actions_layout.addWidget(self.edit_button)
        actions_layout.addWidget(self.remove_button)
        actions_layout.addStretch(1)
        self.actions = actions
        card.add_widget(actions)
        header = self.add_section("Your rules", card)
        self.pick_button = make_button("Add by clicking a window", glyph=style.Glyph.WINDOW, parent=header)
        self.pick_button.clicked.connect(self._start_pick)
        header.add_trailing(self.pick_button)
        self.add_button = make_button("Add rule", "primary", glyph=style.Glyph.ADD, parent=header)
        self.add_button.clicked.connect(self._add)
        header.add_trailing(self.add_button)
        self.finish()
        self.apply_settings(config.settings)

    def rules(self) -> list[ProfileRule]:
        return list(self._config.settings.profiles)

    def apply_settings(self, settings: Settings) -> None:
        _set_rows(
            self.table,
            [
                [
                    rule.name,
                    ", ".join(rule.match_process),
                    ", ".join(rule.match_title),
                    rule.profile.name,
                    DELIVERY_LABELS.get(rule.profile.delivery, rule.profile.delivery.value),
                ]
                for rule in settings.profiles
            ],
        )
        has_rules = bool(settings.profiles)
        self.table.setVisible(has_rules)
        self.actions.setVisible(has_rules)
        self.empty.setVisible(not has_rules)
        _fit_table(self.table, maximum_rows=10)
        header = self.table.horizontalHeader()
        for column, width in enumerate((150, 170, 170, 130)):
            header.resizeSection(column, width)
        self._selection_changed()

    def _selection_changed(self) -> None:
        selected = self.table.currentRow() >= 0 and bool(self.table.selectedItems())
        self.edit_button.setEnabled(selected)
        self.remove_button.setEnabled(selected)

    def apply_rule(self, index: int | None, rule: ProfileRule) -> None:
        def mutate(s: Settings) -> Settings:
            rules = list(s.profiles)
            if index is None or not 0 <= index < len(rules):
                rules.append(rule)
            else:
                rules[index] = rule
            return replace(s, profiles=rules)

        self._update(mutate)

    def remove_rule(self, index: int) -> None:
        def mutate(s: Settings) -> Settings:
            rules = list(s.profiles)
            if 0 <= index < len(rules):
                del rules[index]
            return replace(s, profiles=rules)

        self._update(mutate)

    def editor_for_picked_window(self) -> RuleEditor:
        """A rule editor prefilled from the foreground window's process and title."""
        process, title = "", ""
        try:
            process, title = self.window_picker()
        except Exception:
            log.exception("window pick failed")
        editor = RuleEditor(profile_names=list(BUILTIN_PROFILES), parent=self)
        editor.process.setText(process)
        editor.title.setText(title)
        editor.name.setText(process.rsplit(".", 1)[0] if process else "")
        return editor

    def _update(self, mutator: Callable[[Settings], Settings]) -> None:
        try:
            settings = self._config.update(mutator)
        except SettingsError as exc:
            self._notify("warning", "App rules", str(exc))
            return
        self.apply_settings(settings)

    def _add(self) -> None:
        editor = RuleEditor(profile_names=list(BUILTIN_PROFILES), parent=self)
        if editor.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.apply_rule(None, editor.rule())

    def _edit_selected(self) -> None:
        row = self.table.currentRow()
        rules = self.rules()
        if not 0 <= row < len(rules):
            return
        editor = RuleEditor(profile_names=list(BUILTIN_PROFILES), rule=rules[row], parent=self)
        if editor.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.apply_rule(row, editor.rule())

    def _remove_selected(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.remove_rule(row)

    def _start_pick(self) -> None:
        self.pick_button.setEnabled(False)
        self.pick_button.setText(f"Click the app window within {PICK_DELAY_MS // 1000} s")
        QtCore.QTimer.singleShot(PICK_DELAY_MS, self._finish_pick)

    def _finish_pick(self) -> None:
        self.pick_button.setEnabled(True)
        self.pick_button.setText("Add by clicking a window")
        editor = self.editor_for_picked_window()
        self.activateWindow()
        if editor.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.apply_rule(None, editor.rule())


# Vocabulary -------------------------------------------------------------------------------------------


class VocabularyTab(ScrollPage):
    def __init__(
        self,
        *,
        config: ConfigStore,
        notify: Notify,
        file_dialog: Callable[[bool], Path | None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("Vocabulary", "Teach Spells your words, fix recurring mistakes and expand short triggers into text.", parent)
        self._config = config
        self._notify = notify
        self._file_dialog = file_dialog or self._default_file_dialog
        self.import_button = make_button("Import", glyph=style.Glyph.DOWNLOAD, parent=self.body)
        self.import_button.clicked.connect(self._import)
        self.add_header_action(self.import_button)
        self.export_button = make_button("Export", glyph=style.Glyph.UPLOAD, parent=self.body)
        self.export_button.clicked.connect(self._export)
        self.add_header_action(self.export_button)

        terms = Card(self.body)
        self.terms_area = QtWidgets.QWidget(terms)
        self.terms_flow = FlowLayout(self.terms_area, spacing=8)
        self.terms_flow.setContentsMargins(16, 14, 16, 4)
        terms.add_widget(self.terms_area)
        self.terms_empty = make_label("No terms yet. Add names, products and jargon Spells should recognise.", "caption", "secondary", wrap=True, parent=terms)
        self.terms_empty.setContentsMargins(16, 14, 16, 0)
        terms.add_widget(self.terms_empty)
        self.term_input = _line_edit("Add a word or a name", terms)
        self.term_input.returnPressed.connect(self._add_term_from_input)
        add_term = make_button("Add", glyph=style.Glyph.ADD, parent=terms)
        add_term.clicked.connect(self._add_term_from_input)
        terms.add_widget(_input_row(terms, self.term_input, add_term, stretch=(1, 0)))
        self.term_chips: list[Chip] = []
        self.add_section("Terms", terms, description="Recognition is biased toward these words.")

        replacements = Card(self.body)
        self.replacements = _table(["Find", "Replace with"], replacements)
        replacements.add_widget(self.replacements)
        self.find_input = _line_edit("Find", replacements)
        self.replace_input = _line_edit("Replace with", replacements)
        self.find_input.returnPressed.connect(self._add_replacement_from_input)
        self.replace_input.returnPressed.connect(self._add_replacement_from_input)
        add_replacement = make_button("Add", glyph=style.Glyph.ADD, parent=replacements)
        add_replacement.clicked.connect(self._add_replacement_from_input)
        remove_replacement = make_button("Remove", "subtle", glyph=style.Glyph.DELETE, parent=replacements)
        remove_replacement.clicked.connect(lambda: self.remove_replacement(self.replacements.currentRow()))
        replacements.add_widget(
            _input_row(replacements, self.find_input, self.replace_input, add_replacement, remove_replacement, stretch=(1, 1, 0, 0))
        )
        self.add_section("Replacements", replacements, description="Applied to every transcript after recognition.")

        snippets = Card(self.body)
        self.snippets = _table(["Trigger", "Text"], snippets)
        snippets.add_widget(self.snippets)
        self.trigger_input = _line_edit("Trigger", snippets)
        self.snippet_input = _line_edit("Text to insert", snippets)
        self.snippet_input.returnPressed.connect(self._add_snippet_from_input)
        add_snippet = make_button("Add", glyph=style.Glyph.ADD, parent=snippets)
        add_snippet.clicked.connect(self._add_snippet_from_input)
        remove_snippet = make_button("Remove", "subtle", glyph=style.Glyph.DELETE, parent=snippets)
        remove_snippet.clicked.connect(lambda: self.remove_snippet(self.snippets.currentRow()))
        snippets.add_widget(
            _input_row(snippets, self.trigger_input, self.snippet_input, add_snippet, remove_snippet, stretch=(1, 2, 0, 0))
        )
        self.add_section("Snippets", snippets, description="Say the trigger on its own to insert the text.")
        self.finish()

        self.apply_settings(config.settings)

    def apply_settings(self, settings: Settings) -> None:
        vocabulary = settings.vocabulary
        self.terms_flow.clear()
        self.term_chips = []
        for index, term in enumerate(vocabulary.terms):
            chip = Chip(term.text, remove_tooltip=f"Remove {term.text}", parent=self.terms_area)
            chip.remove_clicked.connect(lambda i=index: self.remove_term(i))
            self.terms_flow.addWidget(chip)
            chip.show()
            self.term_chips.append(chip)
        self.terms_area.setVisible(bool(vocabulary.terms))
        self.terms_empty.setVisible(not vocabulary.terms)
        self.terms_area.updateGeometry()
        _set_rows(self.replacements, [[entry.find, entry.replace] for entry in vocabulary.replacements])
        _fit_table(self.replacements)
        self.replacements.horizontalHeader().resizeSection(0, 240)
        _set_rows(self.snippets, [[snippet.trigger, snippet.text] for snippet in vocabulary.snippets])
        _fit_table(self.snippets)
        self.snippets.horizontalHeader().resizeSection(0, 200)

    def term_texts(self) -> list[str]:
        return [chip.text() for chip in self.term_chips]

    def _update(self, mutate_vocabulary: Callable[[Vocabulary], Vocabulary]) -> None:
        try:
            settings = self._config.update(lambda s: replace(s, vocabulary=mutate_vocabulary(s.vocabulary)))
        except SettingsError as exc:
            self._notify("warning", "Vocabulary", str(exc))
            return
        self.apply_settings(settings)

    def add_term(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        now = time.time()

        def mutate(v: Vocabulary) -> Vocabulary:
            terms = [t for t in v.terms if t.text != text]
            terms.append(TermEntry(text=text, edited_at=now))
            return replace(v, terms=terms)

        self._update(mutate)

    def remove_term(self, index: int) -> None:
        self._update(lambda v: replace(v, terms=[t for i, t in enumerate(v.terms) if i != index]))

    def add_replacement(self, find: str, replace_with: str) -> None:
        find = find.strip()
        if not find:
            return

        def mutate(v: Vocabulary) -> Vocabulary:
            entries = [r for r in v.replacements if r.find != find]
            entries.append(Replacement(find=find, replace=replace_with))
            return replace(v, replacements=entries)

        self._update(mutate)

    def remove_replacement(self, index: int) -> None:
        self._update(lambda v: replace(v, replacements=[r for i, r in enumerate(v.replacements) if i != index]))

    def add_snippet(self, trigger: str, text: str) -> None:
        trigger = trigger.strip()
        if not trigger:
            return

        def mutate(v: Vocabulary) -> Vocabulary:
            entries = [s for s in v.snippets if s.trigger != trigger]
            entries.append(Snippet(trigger=trigger, text=text))
            return replace(v, snippets=entries)

        self._update(mutate)

    def remove_snippet(self, index: int) -> None:
        self._update(lambda v: replace(v, snippets=[s for i, s in enumerate(v.snippets) if i != index]))

    def export_to(self, path: Path) -> None:
        vocabulary = self._config.settings.vocabulary
        data = {
            "terms": [{"text": t.text, "edited_at": t.edited_at} for t in vocabulary.terms],
            "replacements": [{"find": r.find, "replace": r.replace} for r in vocabulary.replacements],
            "snippets": [{"trigger": s.trigger, "text": s.text} for s in vocabulary.snippets],
        }
        try:
            Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            self._notify("warning", "Export", f"Could not write {path}: {exc}")

    def import_from(self, path: Path) -> None:
        """Merge a file written by export_to; existing entries with the same key are replaced."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            terms = [TermEntry(text=str(t["text"]), edited_at=float(t.get("edited_at", 0.0))) for t in data.get("terms", [])]
            replacements = [Replacement(find=str(r["find"]), replace=str(r["replace"])) for r in data.get("replacements", [])]
            snippets = [Snippet(trigger=str(s["trigger"]), text=str(s["text"])) for s in data.get("snippets", [])]
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            self._notify("warning", "Import", f"{path} is not a vocabulary export: {exc}")
            return

        def mutate(v: Vocabulary) -> Vocabulary:
            new_terms = [t for t in v.terms if t.text not in {i.text for i in terms}] + terms
            new_replacements = [r for r in v.replacements if r.find not in {i.find for i in replacements}] + replacements
            new_snippets = [s for s in v.snippets if s.trigger not in {i.trigger for i in snippets}] + snippets
            return Vocabulary(terms=new_terms, replacements=new_replacements, snippets=new_snippets)

        self._update(mutate)

    def _add_term_from_input(self) -> None:
        self.add_term(self.term_input.text())
        self.term_input.clear()

    def _add_replacement_from_input(self) -> None:
        self.add_replacement(self.find_input.text(), self.replace_input.text())
        self.find_input.clear()
        self.replace_input.clear()

    def _add_snippet_from_input(self) -> None:
        self.add_snippet(self.trigger_input.text(), self.snippet_input.text())
        self.trigger_input.clear()
        self.snippet_input.clear()

    def _default_file_dialog(self, saving: bool) -> Path | None:
        if saving:
            chosen, _f = QtWidgets.QFileDialog.getSaveFileName(self, "Export vocabulary", "vocabulary.json", "JSON (*.json)")
        else:
            chosen, _f = QtWidgets.QFileDialog.getOpenFileName(self, "Import vocabulary", "", "JSON (*.json)")
        return Path(chosen) if chosen else None

    def _import(self) -> None:
        path = self._file_dialog(False)
        if path is not None:
            self.import_from(path)

    def _export(self) -> None:
        path = self._file_dialog(True)
        if path is not None:
            self.export_to(path)




def quality_badge(label: str) -> tuple[str, str]:
    """The badge text and kind of a stored quality label."""
    return QUALITY_BADGES.get(label or "", ("", "muted"))


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def speaking_stats(entry: Any) -> str:
    """How the dictation was spoken, plus the reason behind its label (spec 14.4)."""
    signals = getattr(entry, "signals", None)
    if signals is None:
        return ""
    parts: list[str] = []
    if signals.audio_s:
        parts.append(f"{signals.audio_s:.1f} s")
    if signals.words_per_minute:
        parts.append(f"{signals.words_per_minute:.0f} words per minute")
    parts.append(_plural(signals.filler_count, "filler"))
    if signals.correction_count:
        parts.append(_plural(signals.correction_count, "self-correction"))
    if signals.dropped_blocks:
        parts.append("some audio was dropped")
    line = ", ".join(parts)
    reason = getattr(entry, "quality_reason", "")
    return f"{line}. {reason}" if reason else line


def check_line(entry: Any) -> str:
    """What the on-demand check said about this row, or nothing when it never ran."""
    verdict = getattr(entry, "check_verdict", "")
    if not verdict:
        return ""
    label = check_label(verdict) or verdict
    when = getattr(entry, "checked_at", 0.0)
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(when)) if when else ""
    head = f"Checked {stamp}: {label}" if stamp else f"Checked: {label}"
    reason = getattr(entry, "check_reason", "")
    return f"{head}. {reason}" if reason else head


def usage_text(files: int, used_bytes: int) -> str:
    if not files:
        return "No recordings saved yet."
    return f"{_plural(files, 'recording')}, {used_bytes / (1024 * 1024):.0f} MB."


# History ------------------------------------------------------------------------------------------------


class _TextPanel(Card):
    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, margins=(16, 12, 12, 12))
        self.body.setSpacing(6)
        self.title_label = make_label(title, "caption_strong", "secondary", parent=self)
        self.add_widget(self.title_label)
        self.edit = QtWidgets.QPlainTextEdit(self)
        self.edit.setProperty("role", "embedded")
        self.edit.setReadOnly(True)
        self.edit.setFont(style.font("body"))
        self.edit.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.edit.setPlaceholderText("Select a dictation above.")
        self.add_widget(self.edit, 1)


class _Stat(QtWidgets.QWidget):
    """One figure of the History summary: a big value over a plain caption."""

    def __init__(self, caption: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.value = make_label(EMPTY_STAT, "stat", parent=self)
        layout.addWidget(self.value)
        layout.addWidget(make_label(caption, "caption", "secondary", parent=self))

    def set_value(self, text: str) -> None:
        empty = not text
        self.value.setText(text or EMPTY_STAT)
        self.value.setFont(style.font("body" if empty else "stat"))
        set_prop(self.value, "tone", "tertiary" if empty else None)


def _default_export_dialog(default: Path, fmt: str) -> Path | None:
    chosen, _filter = QtWidgets.QFileDialog.getSaveFileName(
        None, "Export history", str(default), EXPORT_FILTERS[fmt]
    )
    return Path(chosen) if chosen else None


class HistoryTab(ScrollPage):
    """The History page: what was said, how it came out, and the recordings (spec 14.4).

    The table shows one row per dictation with a quality badge and, when the recording was
    kept, a play control. The summary above it is computed over the rows on screen, so a
    search narrows the figures with the list. Nothing here touches an engine except the
    on-demand check, which runs on a worker thread through CheckRunner and is skipped
    politely when the cleanup engine is not serving.
    """

    def __init__(
        self,
        *,
        config: ConfigStore,
        history: Any,
        notify: Notify,
        confirm: Confirm | None = None,
        pipeline: Any = None,
        export_dialog: Callable[[Path, str], Path | None] | None = None,
        player: Any = None,
        reveal: Callable[[Path], None] | None = None,
        executor: Callable[[Callable[[], None]], None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(
            "History",
            "Your recent dictations, stored only on this computer.",
            parent,
        )
        self._config = config
        self._history = history
        self._notify = notify
        self._confirm = confirm or default_confirm
        self._pipeline = pipeline
        self._export_dialog = export_dialog or _default_export_dialog
        self._reveal = reveal or default_reveal
        self._entries: list[Any] = []
        self._play_cells: dict[int, PlayCell] = {}
        self._playing_row: int | None = None
        self.player = player if player is not None else WavPlayer(parent=self)
        self.player.finished.connect(self._playback_finished)
        self.player.failed.connect(lambda text: self._notify("warning", "Recording", text))
        self.checker = CheckRunner(self._check_transcript, executor, parent=self)
        self.checker.checked.connect(self._store_check)
        self.checker.done.connect(self._checks_done)
        self.checker.failed.connect(lambda text: self._notify("warning", "Check", text))

        toolbar = QtWidgets.QWidget(self.body)
        top = QtWidgets.QHBoxLayout(toolbar)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)
        self.search = _line_edit("Search what you said and the cleaned text", toolbar)
        self.search.setClearButtonEnabled(True)
        self.search.addAction(style.glyph_icon(style.Glyph.SEARCH, style.palette().text3), QtWidgets.QLineEdit.ActionPosition.LeadingPosition)
        self.search.returnPressed.connect(self.run_search)
        self.search.textChanged.connect(self._search_edited)
        top.addWidget(self.search, 1)
        top.addWidget(make_label("Keep", "body", "secondary", parent=toolbar))
        self.retention = SegmentedControl(toolbar)
        for code in RETENTION_CHOICES:
            self.retention.addItem(RETENTION_LABELS[code], code)
        self.retention.setToolTip("Off stops storing new dictations and keeps the ones already stored.")
        self.retention.currentIndexChanged.connect(self._on_retention)
        top.addWidget(self.retention)
        self.export_button = make_button("Export", glyph=style.Glyph.UPLOAD, parent=toolbar)
        self.export_menu = QtWidgets.QMenu(self.export_button)
        for fmt in EXPORT_FORMATS:
            action = self.export_menu.addAction(EXPORT_LABELS[fmt])
            action.setData(fmt)
            action.triggered.connect(lambda _checked=False, code=fmt: self.export(code))
        self.export_button.setMenu(self.export_menu)
        self.export_button.setToolTip("Save what this list shows, with the filter you typed.")
        top.addWidget(self.export_button)
        self.add_widget(toolbar, spacing_before=0)
        self._search_timer = QtCore.QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self.run_search)

        summary = Card(self.body)
        summary_row = QtWidgets.QWidget(summary)
        summary_layout = QtWidgets.QHBoxLayout(summary_row)
        summary_layout.setContentsMargins(16, 12, 16, 12)
        summary_layout.setSpacing(32)
        self.rate_stat = _Stat("Median words per minute", summary_row)
        self.filler_stat = _Stat("Fillers per 100 words", summary_row)
        self.cleanup_stat = _Stat("Cleanup changed the text", summary_row)
        for stat in (self.rate_stat, self.filler_stat, self.cleanup_stat):
            summary_layout.addWidget(stat)
        summary_layout.addStretch(1)
        summary.add_widget(summary_row)
        self.add_widget(summary, spacing_before=16)

        card = Card(self.body)
        self.table = _table(["When", "App", "Language", "Quality", "Sound", "Text"], card)
        self.table.itemSelectionChanged.connect(self._show_selected)
        header = self.table.horizontalHeader()
        for column, width in enumerate((150, 140, 100, 110, 100)):
            header.resizeSection(column, width)
        self.table.setMinimumHeight(260)
        card.add_widget(self.table)
        self.empty = _EmptyState(style.Glyph.HISTORY, "Nothing here yet", "Dictations appear here after you use your hotkey.", card)
        card.add_widget(self.empty)
        self.add_widget(card, spacing_before=16)

        details = QtWidgets.QWidget(self.body)
        details_layout = QtWidgets.QHBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(12)
        raw_panel = _TextPanel(SAID_TITLE, details)
        cleaned_panel = _TextPanel(CLEANED_TITLE, details)
        self.raw_panel = raw_panel
        self.cleaned_panel = cleaned_panel
        self.raw = raw_panel.edit
        self.cleaned = cleaned_panel.edit
        raw_panel.setMinimumHeight(140)
        cleaned_panel.setMinimumHeight(140)
        details_layout.addWidget(raw_panel)
        details_layout.addWidget(cleaned_panel)
        self.add_widget(details, spacing_before=12)

        self.stats_line = make_label("", "caption", "secondary", wrap=True, parent=self.body)
        self.add_widget(self.stats_line, spacing_before=8)
        self.check_line = make_label("", "caption", "secondary", wrap=True, parent=self.body)
        self.add_widget(self.check_line, spacing_before=2)

        buttons = QtWidgets.QWidget(self.body)
        buttons_layout = QtWidgets.QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)
        self.copy_button = make_button("Copy", glyph=style.Glyph.COPY, parent=buttons)
        self.copy_button.clicked.connect(self._copy)
        self.check_button = make_button("Check this transcript", glyph=style.Glyph.CHECK, parent=buttons)
        self.check_button.setToolTip("Ask the cleanup model whether this reads like real speech.")
        self.check_button.clicked.connect(self.check_selected)
        self.check_recent_button = make_button(f"Check the last {CHECK_BATCH}", parent=buttons)
        self.check_recent_button.clicked.connect(self.check_recent)
        self.reveal_button = make_button("Show in folder", glyph=style.Glyph.FOLDER, parent=buttons)
        self.reveal_button.clicked.connect(self._reveal_selected)
        self.delete_button = make_button("Delete", glyph=style.Glyph.DELETE, parent=buttons)
        self.delete_button.clicked.connect(self._delete_selected)
        self.clear_button = make_button("Clear history", "danger", glyph=style.Glyph.DELETE, parent=buttons)
        self.clear_button.clicked.connect(self._clear)
        for button in (self.copy_button, self.check_button, self.check_recent_button, self.reveal_button, self.delete_button):
            buttons_layout.addWidget(button)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self.clear_button)
        self.add_widget(buttons, spacing_before=12)

        recordings = Card(self.body)
        self.keep_audio = ToggleSwitch(parent=recordings)
        self.keep_audio.toggled.connect(self._on_keep_audio)
        recordings.add_row(
            SettingRow(KEEP_AUDIO_TITLE, KEEP_AUDIO_HINT, self.keep_audio, glyph=style.Glyph.WAVEFORM)
        )
        limits = QtWidgets.QWidget(recordings)
        limits_layout = QtWidgets.QHBoxLayout(limits)
        limits_layout.setContentsMargins(0, 0, 0, 0)
        limits_layout.setSpacing(8)
        self.keep_count = SpinBox(limits)
        self.keep_count.setRange(1, 10_000)
        self.keep_count.setSuffix(" recordings")
        self.keep_count.valueChanged.connect(self._on_keep_count)
        self.keep_mb = SpinBox(limits)
        self.keep_mb.setRange(1, 100_000)
        self.keep_mb.setSuffix(" MB")
        self.keep_mb.valueChanged.connect(self._on_keep_mb)
        limits_layout.addWidget(self.keep_count)
        limits_layout.addWidget(self.keep_mb)
        recordings.add_row(
            SettingRow(
                "How much to keep",
                "The oldest recordings are deleted first once either limit is reached.",
                limits,
            )
        )
        self.delete_audio_button = make_button("Delete all recordings", "danger", glyph=style.Glyph.DELETE, parent=recordings)
        self.delete_audio_button.clicked.connect(self._delete_recordings)
        self.usage_row = SettingRow("Recordings on disk", "", self.delete_audio_button)
        recordings.add_row(self.usage_row)
        note = QtWidgets.QWidget(recordings)
        note_layout = QtWidgets.QHBoxLayout(note)
        note_layout.setContentsMargins(16, 0, 16, 12)
        note_layout.setSpacing(8)
        note_layout.addWidget(GlyphLabel(style.Glyph.SHIELD, 14, tone="secondary", parent=note))
        note_layout.addWidget(make_label(RECORDINGS_NOTE, "caption", "secondary", wrap=True, parent=note), 1)
        recordings.add_widget(note)
        self.add_section("Recordings", recordings)
        self.finish()

        self.apply_settings(config.settings)

    # Settings

    def apply_settings(self, settings: Settings) -> None:
        history = settings.history
        with _Blocked(self.retention):
            index = self.retention.findData(history.retention)
            self.retention.setCurrentIndex(max(0, index))
        with _Blocked(self.keep_audio):
            self.keep_audio.setChecked(history.keep_audio)
        with _Blocked(self.keep_count):
            self.keep_count.setValue(history.audio_keep_count)
        with _Blocked(self.keep_mb):
            self.keep_mb.setValue(history.audio_keep_mb)
        self.keep_count.setEnabled(history.keep_audio)
        self.keep_mb.setEnabled(history.keep_audio)
        self._refresh_usage()

    # The list

    def refresh(self) -> None:
        self.run_search()

    def run_search(self) -> None:
        self._search_timer.stop()
        self._stop_playback()
        query = self.search.text().strip()
        try:
            entries = self._history.search(query) if query else self._history.recent()
        except Exception as exc:
            log.exception("history query failed")
            self._notify("warning", "History", f"Could not read the history: {exc}")
            entries = []
        self._entries = list(entries)
        rows = []
        for entry in self._entries:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.created_at))
            rows.append([
                when,
                entry.app_process,
                theme.language_name(entry.language) if entry.language else "",
                "",
                "",
                row_text(entry).replace("\n", " ")[:160],
            ])
        _set_rows(self.table, rows)
        self._fill_cells()
        self._update_summary()
        has_rows = bool(self._entries) or bool(query)
        self.table.setVisible(has_rows)
        self.empty.setVisible(not has_rows)
        self.raw.setPlainText("")
        self.cleaned.setPlainText("")
        self._show_selected()

    def _fill_cells(self) -> None:
        self._play_cells = {}
        for row, entry in enumerate(self._entries):
            label, kind = quality_badge(entry.quality_label)
            if label:
                badge = Badge(label, kind, self.table)
                badge.setToolTip(entry.quality_reason or "")
                self.table.setCellWidget(row, QUALITY_COLUMN, CellHost(badge, self.table))
            path = self._recording_path(entry)
            if path is None:
                continue
            cell = PlayCell(row, wav_duration_s(path), self.table)
            cell.toggled.connect(self._toggle_play)
            self._play_cells[row] = cell
            self.table.setCellWidget(row, SOUND_COLUMN, cell)

    def _update_summary(self) -> None:
        summary = summarize(row_stats(entry) for entry in self._entries)
        self.rate_stat.set_value(
            "" if summary.median_words_per_minute is None else f"{summary.median_words_per_minute:.0f}"
        )
        self.filler_stat.set_value("" if summary.filler_rate is None else f"{summary.filler_rate:.1f}")
        self.cleanup_stat.set_value(
            "" if summary.cleanup_changed_rate is None else f"{summary.cleanup_changed_rate * 100:.0f}%"
        )

    def _search_edited(self, _text: str) -> None:
        self._search_timer.start()

    def _selected_entry(self) -> Any | None:
        row = self.table.currentRow()
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None

    def _show_selected(self) -> None:
        entry = self._selected_entry()
        wrote = bool(entry is not None and getattr(entry, "wrote", False))
        self.raw_panel.title_label.setText(INSTRUCTION_TITLE if wrote else SAID_TITLE)
        self.cleaned_panel.title_label.setText(WRITTEN_TITLE if wrote else CLEANED_TITLE)
        self.raw.setPlainText(entry.raw_text if entry else "")
        self.cleaned.setPlainText(row_text(entry) if entry else "")
        self.stats_line.setText(speaking_stats(entry) if entry else "")
        self.check_line.setText(check_line(entry) if entry else "")
        has_entry = entry is not None
        self.copy_button.setEnabled(has_entry)
        self.delete_button.setEnabled(has_entry)
        self.check_button.setEnabled(has_entry and bool(entry.raw_text.strip()))
        self.reveal_button.setEnabled(has_entry and self._recording_path(entry) is not None)

    def _copy(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        QtWidgets.QApplication.clipboard().setText(entry.delivered_text or entry.cleaned_text)

    def _clear(self) -> None:
        if not self._confirm("Clear history", "Delete every stored dictation and its recording? This cannot be undone."):
            return
        self._stop_playback()
        try:
            self._history.clear()
        except Exception as exc:
            log.exception("history clear failed")
            self._notify("warning", "History", f"Could not clear the history: {exc}")
        self.run_search()

    def _delete_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None or entry.id is None:
            return
        if not self._confirm("Delete this dictation", "Delete this dictation and its recording?"):
            return
        self._stop_playback()
        delete = getattr(self._history, "delete", None)
        if delete is None:
            return
        try:
            delete(entry.id)
        except Exception as exc:
            log.exception("history delete failed")
            self._notify("warning", "History", f"Could not delete the dictation: {exc}")
        self.run_search()

    def _on_retention(self, index: int) -> None:
        code = self.retention.itemData(index)
        if not code:
            return
        try:
            self._config.update(lambda s: replace(s, history=replace(s.history, retention=code)))
        except SettingsError as exc:
            self._notify("warning", "History", str(exc))
            return
        set_retention = getattr(self._history, "set_retention", None)
        if set_retention is not None:
            try:
                set_retention(code)
            except Exception:
                log.exception("history.set_retention failed")
        self.run_search()

    # Export

    def export(self, fmt: str) -> Path | None:
        """Save what the list is showing, in one of the three formats (spec 14.4)."""
        default = Path.home() / "Desktop" / export_file_name(fmt)
        target = self._export_dialog(default, fmt)
        if target is None:
            return None
        target = Path(target)
        try:
            written = export_to_path(list(self._entries), target, fmt)
        except (OSError, ValueError) as exc:
            log.exception("history export failed")
            self._notify("warning", "Export", f"Could not write {target}: {exc}")
            return None
        self._notify(
            "info",
            "Export",
            f"Saved {written} dictation{'' if written == 1 else 's'} to {target}.",
        )
        return target

    # Recordings

    def _recording_path(self, entry: Any) -> Path | None:
        getter = getattr(self._history, "recording_path", None)
        if getter is None or not getattr(entry, "audio_file", ""):
            return None
        try:
            return getter(entry)
        except Exception:
            log.exception("could not resolve the recording of a history row")
            return None

    def _toggle_play(self, row: int) -> None:
        if self._playing_row == row:
            self._stop_playback()
            return
        self._stop_playback()
        if not (0 <= row < len(self._entries)):
            return
        path = self._recording_path(self._entries[row])
        if path is None:
            self._notify("warning", "Recording", "That recording is no longer on disk.")
            self.run_search()
            return
        if self.player.play(path):
            self._playing_row = row
            cell = self._play_cells.get(row)
            if cell is not None:
                cell.set_playing(True)

    def _stop_playback(self) -> None:
        if self.player.playing:
            self.player.stop()
        self._mark_stopped()

    def _playback_finished(self, _path: Any = None) -> None:
        self._mark_stopped()

    def _mark_stopped(self) -> None:
        row = self._playing_row
        self._playing_row = None
        cell = self._play_cells.get(row) if row is not None else None
        if cell is not None:
            cell.set_playing(False)

    def _reveal_selected(self) -> None:
        entry = self._selected_entry()
        path = self._recording_path(entry) if entry is not None else None
        if path is None:
            return
        try:
            self._reveal(path)
        except Exception as exc:
            log.exception("could not reveal the recording")
            self._notify("warning", "Recording", f"Could not open the folder: {exc}")

    def _on_keep_audio(self, value: bool) -> None:
        try:
            self._config.update(lambda s: replace(s, history=replace(s.history, keep_audio=bool(value))))
        except SettingsError as exc:
            self._notify("warning", "Recordings", str(exc))
            return
        self.keep_count.setEnabled(bool(value))
        self.keep_mb.setEnabled(bool(value))
        self._refresh_usage()

    def _on_keep_count(self, value: int) -> None:
        self._update_audio_limits(audio_keep_count=int(value))

    def _on_keep_mb(self, value: int) -> None:
        self._update_audio_limits(audio_keep_mb=int(value))

    def _update_audio_limits(self, **changes: int) -> None:
        try:
            self._config.update(lambda s: replace(s, history=replace(s.history, **changes)))
        except SettingsError as exc:
            self._notify("warning", "Recordings", str(exc))
            return
        prune = getattr(self._history, "prune_recordings", None)
        history = self._config.settings.history
        if prune is not None and history.keep_audio:
            try:
                prune(AudioPolicy(True, history.audio_keep_count, history.audio_keep_mb))
            except Exception:
                log.exception("pruning the recordings failed")
        self._refresh_usage()
        self.run_search()

    def _delete_recordings(self) -> None:
        if not self._confirm("Delete all recordings", "Delete every recording? The dictations themselves stay."):
            return
        self._stop_playback()
        clear = getattr(self._history, "clear_recordings", None)
        if clear is not None:
            try:
                clear()
            except Exception as exc:
                log.exception("clearing the recordings failed")
                self._notify("warning", "Recordings", f"Could not delete the recordings: {exc}")
        self._refresh_usage()
        self.run_search()

    def _refresh_usage(self) -> None:
        usage = getattr(self._history, "recordings_usage", None)
        files, used = (0, 0)
        if usage is not None:
            try:
                files, used = usage()
            except Exception:
                log.exception("could not measure the recordings folder")
        self.usage_row.set_description(usage_text(files, used))
        self.delete_audio_button.setEnabled(files > 0)

    # The on-demand check

    def _check_transcript(self, text: str, language: str) -> CheckResult:
        checker = getattr(self._pipeline, "check_transcript", None)
        if checker is None:
            return CheckResult("", "", "unavailable")
        return checker(text, language)

    def check_selected(self) -> bool:
        entry = self._selected_entry()
        if entry is None or entry.id is None or not entry.raw_text.strip():
            return False
        return self._start([CheckJob(entry.id, entry.raw_text, entry.language)])

    def check_recent(self) -> bool:
        jobs = [
            CheckJob(entry.id, entry.raw_text, entry.language)
            for entry in self._entries[:CHECK_BATCH]
            if entry.id is not None and entry.raw_text.strip()
        ]
        return self._start(jobs)

    def _start(self, jobs: list[CheckJob]) -> bool:
        if not jobs:
            return False
        if self.checker.busy:
            self._notify("info", "Check", "A check is already running.")
            return False
        self.check_button.setEnabled(False)
        self.check_recent_button.setEnabled(False)
        if not self.checker.submit(jobs):
            self._checks_done(0, 0)
            return False
        return True

    def _store_check(self, entry_id: int, result: CheckResult) -> None:
        setter = getattr(self._history, "set_check", None)
        if setter is not None:
            try:
                setter(entry_id, result.verdict, result.reason)
            except Exception:
                log.exception("could not store a check result")
        for entry in self._entries:
            if entry.id == entry_id:
                entry.check_verdict = result.verdict
                entry.check_reason = result.reason
        selected = self._selected_entry()
        if selected is not None and selected.id == entry_id:
            self.check_line.setText(check_line(selected))

    def _checks_done(self, stored: int, skipped: int) -> None:
        self.check_recent_button.setEnabled(True)
        self._show_selected()
        if stored:
            self._notify("info", "Check", f"Checked {stored} transcript{'' if stored == 1 else 's'}.")
        elif skipped:
            self._notify("info", "Check", CHECK_UNAVAILABLE)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.run_search()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self._stop_playback()
        super().hideEvent(event)


# The window ------------------------------------------------------------------------------------------------


class SettingsDialog(QtWidgets.QDialog):
    welcome_requested = QtCore.Signal()

    def __init__(
        self,
        *,
        config: ConfigStore,
        engines: Any,
        pipeline: Any,
        hotkey: Any,
        history: Any,
        gpu_selection: Any,
        log_dir: Path,
        devices: Callable[[], list] | None = None,
        probe: Callable[[int, int], bool] | None = None,
        notify: Notify | None = None,
        confirm: Confirm | None = None,
        recorder_factory: Callable[..., Any] | None = None,
        window_picker: Callable[[], tuple[str, str]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        notify = notify or default_notify
        self.setObjectName("SettingsDialog")
        self.setWindowTitle(f"{theme.APP_NAME} settings")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        current = style.palette()
        self.setWindowIcon(brand.app_icon(current.color("brand_a"), current.color("brand_b")))
        self.resize(1080, 760)
        self.setMinimumSize(880, 560)
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.nav = NavRail(self)
        outer.addWidget(self.nav)
        layer = QtWidgets.QFrame(self)
        layer.setObjectName("spellsLayer")
        layer_layout = QtWidgets.QVBoxLayout(layer)
        layer_layout.setContentsMargins(1, 1, 0, 0)
        layer_layout.setSpacing(0)
        self.pages = QtWidgets.QStackedWidget(layer)
        layer_layout.addWidget(self.pages)
        outer.addWidget(layer, 1)
        self.tabs = self.pages

        self.general = GeneralTab(
            config=config,
            hotkey=hotkey,
            probe=probe,
            notify=notify,
            devices=devices,
            recorder_factory=recorder_factory,
            parent=self.pages,
        )
        self.general.welcome_requested.connect(self.welcome_requested.emit)
        self.diagnostics = DiagnosticsTab(
            config=config,
            engines=engines,
            pipeline=pipeline,
            hotkey=hotkey,
            gpu_selection=gpu_selection,
            log_dir=log_dir,
            notify=notify,
            parent=self.pages,
        )
        self.languages = LanguagesPage(
            config=config,
            hotkey=hotkey,
            notify=notify,
            probe=probe,
            gpu_selection=lambda: self.diagnostics.selection,
            parent=self.pages,
        )
        self.diagnostics.gpu_changed.connect(lambda: self.languages.refresh_models(force=True))
        self.cleanup = CleanupTab(
            config=config,
            notify=notify,
            gpu_selection=lambda: self.diagnostics.selection,
            parent=self.pages,
        )
        self.apps = AppsTab(config=config, notify=notify, window_picker=window_picker, parent=self.pages)
        self.vocabulary = VocabularyTab(config=config, notify=notify, parent=self.pages)
        self.history = HistoryTab(
            config=config,
            history=history,
            notify=notify,
            confirm=confirm,
            pipeline=pipeline,
            parent=self.pages,
        )
        self.about = AboutPage(parent=self.pages)
        self._tabs_by_name: dict[str, QtWidgets.QWidget] = {
            "general": self.general,
            "languages": self.languages,
            "cleanup": self.cleanup,
            "apps": self.apps,
            "vocabulary": self.vocabulary,
            "history": self.history,
            "diagnostics": self.diagnostics,
            "about": self.about,
        }
        for key, label, glyph, bottom in PAGES:
            self.pages.addWidget(self._tabs_by_name[key])
            self.nav.add_item(key, label, glyph, bottom=bottom)
        self.nav.current_changed.connect(self.show_tab)
        self.show_tab("general")
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setInterval(2000)
        self._refresh_timer.timeout.connect(self._periodic_refresh)
        style.notifier().changed.connect(self._theme_changed)

    def page_names(self) -> list[str]:
        return list(self._tabs_by_name)

    def page_labels(self) -> list[str]:
        return self.nav.labels()

    def current_page(self) -> str:
        widget = self.pages.currentWidget()
        for name, page in self._tabs_by_name.items():
            if page is widget:
                return name
        return "general"

    def show_tab(self, name: str) -> None:
        if name not in self._tabs_by_name:
            name = "general"
        widget = self._tabs_by_name[name]
        self.pages.setCurrentWidget(widget)
        self.nav.set_current(name)
        if widget is self.history:
            self.history.run_search()
        elif widget is self.diagnostics:
            self.diagnostics.refresh()

    def apply_settings(self, settings: Settings) -> None:
        """Refresh every page from a settings snapshot (Qt thread; the bridge delivers it)."""
        for page in self._tabs_by_name.values():
            apply = getattr(page, "apply_settings", None)
            if apply is not None:
                apply(settings)

    def on_engine_status(self, _engine: Any, _state: Any, _reason: str) -> None:
        if self.isVisible():
            self.diagnostics.refresh()

    def _periodic_refresh(self) -> None:
        if self.isVisible() and self.pages.currentWidget() is self.diagnostics:
            self.diagnostics.refresh()

    def _theme_changed(self, _palette: Any) -> None:
        current = style.palette()
        self.setWindowIcon(brand.app_icon(current.color("brand_a"), current.color("brand_b")))
        if self.isVisible():
            style.apply_window_chrome(self)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        style.apply_window_chrome(self)
        self._refresh_timer.start()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self._refresh_timer.stop()
        super().hideEvent(event)


__all__ = [
    "BUILTIN_PROFILES",
    "PAGES",
    "TAB_NAMES",
    "AppsTab",
    "ChordCaptureDialog",
    "CleanupTab",
    "GeneralTab",
    "HistoryTab",
    "HotkeyRecorder",
    "RuleEditor",
    "SettingsDialog",
    "VocabularyTab",
    "fold_keys",
    "probe_arguments",
]
