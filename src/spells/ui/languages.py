"""The language picker and the models chosen for the languages.

The user keeps the languages they dictate in as chips (general.enabled_languages), each with
its per-language hotkeys (general.language_chords), and adds more from a searchable list of
every Whisper language with English, German and Albanian first. The models card asks
spells.modelcatalog.select_models for the speech model per language and the cleanup model
and shows each choice with its reason; it refreshes whenever the language set changes.
Every write goes through ConfigStore.update().
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spells import calibrate, modelcatalog
from spells.config import ConfigStore, Settings, SettingsError
from spells.gpu import GpuSelection
from spells.models import Chord
from spells.ui import style, theme
from spells.ui.hotkeys import ChordCaptureDialog, chord_is_free, fold_keys
from spells.ui.tray import chords_of
from spells.ui.widgets import (
    Badge,
    Card,
    Chip,
    FlowLayout,
    IconButton,
    KeycapRow,
    ScrollPage,
    SettingRow,
    StatusDot,
    make_button,
    make_label,
)

log = logging.getLogger(__name__)

Notify = Callable[[str, str, str], None]
Hardware = modelcatalog.Hardware

POPULAR_LANGUAGES = ("en", "de", "sq")
RESULT_ROW_HEIGHT = 36
VISIBLE_RESULTS = 6


def hardware_for(selection: GpuSelection | None) -> Hardware:
    """GPU when the app's GPU selection has a device the catalog counts as a GPU, else CPU."""
    rule = getattr(modelcatalog, "hardware_from_gpu", None)
    if rule is not None:
        return rule(selection)
    if selection is None or selection.raw_index is None:
        return Hardware.CPU
    return Hardware.GPU


def hardware_text(
    hardware: Hardware, selection: GpuSelection | None, *, measured: bool = False
) -> str:
    if measured and hardware is Hardware.GPU and selection is not None and selection.name:
        return f"For your graphics, {selection.name}, measured faster"
    if measured:
        return "For your processor, measured faster than the graphics"
    if hardware is Hardware.GPU and selection is not None and selection.name:
        return f"For your GPU, {selection.name}"
    if hardware is Hardware.GPU:
        return "For your GPU"
    return "For your processor"


def set_language_enabled(config: ConfigStore, notify: Notify, code: str, enabled: bool) -> bool:
    """Add or remove one language; the last language stays, and the tray's mode stays valid."""
    settings = config.settings
    languages = list(settings.general.enabled_languages)
    if enabled and code not in languages:
        languages.append(code)
    elif not enabled and code in languages:
        if len(languages) == 1:
            notify("warning", "Languages", "Keep at least one language enabled.")
            return False
        languages.remove(code)
    else:
        return False
    mode = settings.general.language_mode
    if mode != "auto" and mode not in languages:
        mode = "auto"

    def mutate(s: Settings) -> Settings:
        return replace(s, general=replace(s.general, enabled_languages=languages, language_mode=mode))

    try:
        config.update(mutate)
    except SettingsError as exc:
        notify("warning", "Languages", str(exc))
        return False
    return True


# The enabled languages ------------------------------------------------------------------------


class LanguageRow(QtWidgets.QWidget):
    """A language chip and, on the settings page, its per-language hotkeys."""

    remove_language = QtCore.Signal(str)
    add_hotkey = QtCore.Signal(str)
    remove_hotkey = QtCore.Signal(int)

    def __init__(
        self,
        code: str,
        chords: Sequence[tuple[int, Chord]],
        *,
        enabled: bool,
        show_hotkeys: bool,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.code = code
        name = theme.language_name(code)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)
        self.chip = Chip(
            name,
            code=code,
            removable=enabled,
            muted=not enabled,
            remove_tooltip=f"Remove {name}",
            parent=self,
        )
        self.chip.remove_clicked.connect(lambda: self.remove_language.emit(code))
        layout.addWidget(self.chip)
        self.keycaps: list[KeycapRow] = []
        if not enabled:
            note = make_label("Not one of your languages", "caption", "tertiary", parent=self)
            layout.addWidget(note)
        layout.addStretch(1)
        if not show_hotkeys:
            self.setMinimumHeight(52)
            return
        if chords and enabled:
            hint = make_label(f"Hold for {name} only", "caption", "secondary", parent=self)
            layout.addWidget(hint)
            layout.addSpacing(4)
        for index, chord in chords:
            keys = KeycapRow(theme.chord_keys(chord), parent=self)
            keys.setToolTip(f"Dictates one utterance in {name}")
            layout.addWidget(keys)
            self.keycaps.append(keys)
            remove = IconButton(style.Glyph.CLOSE, f"Remove the hotkey {theme.chord_label(chord)}", pixels=10, parent=self)
            remove.clicked.connect(lambda _checked=False, i=index: self.remove_hotkey.emit(i))
            layout.addWidget(remove)
        if enabled:
            self.add_button = make_button("Add hotkey", "subtle", glyph=style.Glyph.ADD, parent=self)
            self.add_button.setToolTip(f"A hotkey that dictates in {name} without changing the language mode")
            self.add_button.clicked.connect(lambda: self.add_hotkey.emit(code))
            layout.addWidget(self.add_button)
        self.setMinimumHeight(56)

    def hotkey_labels(self) -> list[str]:
        return [keys.text() for keys in self.keycaps]


class LanguageList(Card):
    """One row per enabled language, then a row per language that only has a hotkey left."""

    remove_language = QtCore.Signal(str)
    add_hotkey = QtCore.Signal(str)
    remove_hotkey = QtCore.Signal(int)

    def __init__(self, *, show_hotkeys: bool = True, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._show_hotkeys = show_hotkeys
        self._key: tuple | None = None
        self.language_rows: dict[str, LanguageRow] = {}

    def apply_settings(self, settings: Settings) -> None:
        general = settings.general
        chords = list(enumerate(general.language_chords))
        key = (tuple(general.enabled_languages), tuple((i, c.keys, c.language) for i, c in chords) if self._show_hotkeys else ())
        if key == self._key:
            return
        self._key = key
        self.clear()
        self.language_rows = {}
        codes = list(general.enabled_languages)
        orphans = [c.language for _i, c in chords if c.language and c.language not in codes] if self._show_hotkeys else []
        for code in [*codes, *dict.fromkeys(orphans)]:
            own = [(i, c) for i, c in chords if c.language == code]
            row = LanguageRow(code, own, enabled=code in codes, show_hotkeys=self._show_hotkeys, parent=self)
            row.remove_language.connect(self.remove_language.emit)
            row.add_hotkey.connect(self.add_hotkey.emit)
            row.remove_hotkey.connect(self.remove_hotkey.emit)
            self.add_row(row)
            self.language_rows[code] = row

    def codes(self) -> list[str]:
        return list(self.language_rows)


# Adding a language --------------------------------------------------------------------------------


class _ResultDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, enabled: Callable[[], set[str]], parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._enabled = enabled

    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> QtCore.QSize:
        return QtCore.QSize(option.rect.width(), RESULT_ROW_HEIGHT)

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> None:
        current = style.palette()
        code = index.data(QtCore.Qt.ItemDataRole.UserRole)
        name = index.data(QtCore.Qt.ItemDataRole.DisplayRole)
        added = code in self._enabled()
        rect = QtCore.QRectF(option.rect).adjusted(0, 1, 0, -1)
        painter.save()
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            state = option.state
            hovered = bool(state & QtWidgets.QStyle.StateFlag.State_MouseOver)
            selected = bool(state & QtWidgets.QStyle.StateFlag.State_Selected)
            if hovered or selected:
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(current.color("nav_selected" if selected else "nav_hover"))
                painter.drawRoundedRect(rect, 6, 6)
            body = style.font("body")
            painter.setFont(body)
            painter.setPen(current.color("text"))
            name_width = QtGui.QFontMetricsF(body).horizontalAdvance(name)
            painter.drawText(rect.adjusted(12, 0, 0, 0), int(QtCore.Qt.AlignmentFlag.AlignVCenter), name)
            painter.setFont(style.font("caption"))
            painter.setPen(current.color("text3"))
            painter.drawText(rect.adjusted(12 + name_width + 8, 0, 0, 0), int(QtCore.Qt.AlignmentFlag.AlignVCenter), code)
            painter.setFont(style.font("caption_strong"))
            painter.setPen(current.color("success" if added else "accent_text"))
            painter.drawText(
                rect.adjusted(0, 0, -12, 0),
                int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight),
                "Added" if added else "Add",
            )
        finally:
            painter.restore()


class LanguageAdder(QtWidgets.QWidget):
    """Quick picks for English, German and Albanian, then a search over every Whisper language."""

    add_language = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._enabled: set[str] = set()
        self._choices = theme.language_choices()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 10)
        layout.setSpacing(10)
        self.suggestions = QtWidgets.QWidget(self)
        suggestion_layout = QtWidgets.QHBoxLayout(self.suggestions)
        suggestion_layout.setContentsMargins(0, 0, 0, 0)
        suggestion_layout.setSpacing(8)
        suggestion_layout.addWidget(make_label("Suggested", "caption", "secondary", parent=self.suggestions))
        self.suggestion_buttons: dict[str, QtWidgets.QPushButton] = {}
        for code in POPULAR_LANGUAGES:
            button = make_button(theme.language_name(code), "chipadd", glyph=style.Glyph.ADD, parent=self.suggestions)
            button.clicked.connect(lambda _checked=False, c=code: self.add_language.emit(c))
            suggestion_layout.addWidget(button)
            self.suggestion_buttons[code] = button
        suggestion_layout.addStretch(1)
        layout.addWidget(self.suggestions)
        self.search = QtWidgets.QLineEdit(self)
        self.search.setFont(style.font("body"))
        self.search.setPlaceholderText(f"Search {len(self._choices)} languages")
        self.search.setClearButtonEnabled(True)
        self.search.addAction(style.glyph_icon(style.Glyph.SEARCH, style.palette().text3), QtWidgets.QLineEdit.ActionPosition.LeadingPosition)
        self.search.textChanged.connect(self.set_query)
        self.search.returnPressed.connect(self._add_first)
        layout.addWidget(self.search)
        self.results = QtWidgets.QListWidget(self)
        self.results.setProperty("role", "embedded")
        self.results.setItemDelegate(_ResultDelegate(lambda: self._enabled, self.results))
        self.results.setMouseTracking(True)
        self.results.setUniformItemSizes(True)
        self.results.setFixedHeight(RESULT_ROW_HEIGHT * VISIBLE_RESULTS + 4)
        self.results.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results.itemClicked.connect(self._item_clicked)
        self.results.itemActivated.connect(self._item_clicked)
        layout.addWidget(self.results)
        self.set_query("")

    def apply_settings(self, settings: Settings) -> None:
        self._enabled = set(settings.general.enabled_languages)
        pending = [code for code in POPULAR_LANGUAGES if code not in self._enabled]
        for code, button in self.suggestion_buttons.items():
            button.setVisible(code not in self._enabled)
        self.suggestions.setVisible(bool(pending))
        self.results.viewport().update()

    def matches(self, query: str) -> list[tuple[str, str]]:
        needle = query.strip().casefold()
        choices = self._choices
        if not needle:
            popular = [(c, n) for c, n in choices if c in POPULAR_LANGUAGES]
            popular.sort(key=lambda item: POPULAR_LANGUAGES.index(item[0]))
            return popular + [(c, n) for c, n in choices if c not in POPULAR_LANGUAGES]
        starts = [(c, n) for c, n in choices if n.casefold().startswith(needle) or c == needle]
        contains = [(c, n) for c, n in choices if needle in n.casefold() and (c, n) not in starts]
        return starts + contains

    def set_query(self, text: str) -> None:
        self.results.clear()
        for code, name in self.matches(text):
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, code)
            item.setToolTip(f"{name} ({code})")
            self.results.addItem(item)

    def visible_codes(self) -> list[str]:
        return [self.results.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.results.count())]

    def _item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        code = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if code and code not in self._enabled:
            self.add_language.emit(code)

    def _add_first(self) -> None:
        for code in self.visible_codes():
            if code not in self._enabled:
                self.add_language.emit(code)
                self.search.clear()
                return


# Models -----------------------------------------------------------------------------------------------


class ModelsCard(Card):
    """The speech model per language and the cleanup model select_models picks, with reasons."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._key: tuple | None = None
        self.selection: Any = None
        self.model_rows: list[SettingRow] = []
        self.error: str = ""

    def refresh(
        self,
        languages: Sequence[str],
        hardware: Hardware,
        *,
        force: bool = False,
        catalog: Sequence[Any] | None = None,
    ) -> None:
        key = (tuple(languages), hardware, None if catalog is None else id(catalog))
        if key == self._key and not force:
            return
        self._key = key
        self.clear()
        self.model_rows = []
        self.error = ""
        try:
            selection = modelcatalog.select_models(list(languages), hardware, None, catalog)
        except Exception as exc:
            log.exception("select_models failed")
            self.selection = None
            self.error = str(exc)
            self._add(SettingRow("Models could not be chosen", str(exc), glyph=style.Glyph.WARNING, badge_glyph=True))
            return
        self.selection = selection
        for choice in selection.asr:
            self._add(self._choice_row(choice, "Speech recognition", style.Glyph.MICROPHONE))
        uncovered = [code for code in languages if selection.asr_for(code) is None]
        if uncovered:
            names = ", ".join(theme.language_name(code) for code in uncovered)
            row = SettingRow(f"No speech model for {names}", "Dictation in these languages is not available yet.", glyph=style.Glyph.WARNING, badge_glyph=True)
            row.add_control(Badge("Missing", "caution"))
            self._add(row)
        if selection.cleanup is not None:
            self._add(self._choice_row(selection.cleanup, "Cleanup", style.Glyph.BRUSH))
        else:
            row = SettingRow(
                "No cleanup model",
                "Transcripts are delivered as recognised, without the cleanup pass.",
                glyph=style.Glyph.BRUSH,
                badge_glyph=True,
            )
            row.add_control(Badge("Off", "muted"))
            self._add(row)
        for note in selection.notes:
            label = make_label(note, "caption", "secondary", wrap=True)
            holder = QtWidgets.QWidget()
            holder_layout = QtWidgets.QHBoxLayout(holder)
            holder_layout.setContentsMargins(68, 10, 16, 12)
            holder_layout.addWidget(label)
            self.add_row(holder)

    def _add(self, row: SettingRow) -> None:
        self.add_row(row)
        self.model_rows.append(row)

    def _choice_row(self, choice: Any, kind: str, glyph: str) -> SettingRow:
        below = QtWidgets.QWidget()
        below_layout = QtWidgets.QVBoxLayout(below)
        below_layout.setContentsMargins(0, 0, 0, 0)
        below_layout.setSpacing(6)
        tags = QtWidgets.QWidget(below)
        flow = FlowLayout(tags, spacing=4)
        flow.addWidget(make_label(f"{kind} for", "caption", "secondary", parent=tags))
        for code in choice.languages:
            tag = make_label(theme.language_name(code), "caption", parent=tags)
            tag.setProperty("role", "tag")
            tag.setFixedHeight(20)
            flow.addWidget(tag)
        below_layout.addWidget(tags)
        reason = make_label(choice.reason, "caption", "tertiary", wrap=True, parent=below)
        below_layout.addWidget(reason)
        row = SettingRow(choice.display_name, "", glyph=glyph, badge_glyph=True, below=below)
        row.title_label.setFont(style.font("body_strong"))
        row.reason_label = reason
        installed = bool(getattr(choice, "installed", True))
        row.add_control(Badge("Installed" if installed else "Not installed", "ok" if installed else "caution"))
        if row.glyph_label is not None:
            row.layout().setAlignment(row.glyph_label, QtCore.Qt.AlignmentFlag.AlignTop)
        return row

    def names(self) -> list[str]:
        return [row.title_label.text() for row in self.model_rows]


# The page -------------------------------------------------------------------------------------------------


class LanguagesPage(ScrollPage):
    def __init__(
        self,
        *,
        config: ConfigStore,
        hotkey: Any,
        notify: Notify,
        probe: Callable[[int, int], bool] | None = None,
        gpu_selection: Callable[[], GpuSelection | None] | GpuSelection | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(
            "Languages",
            "Pick the languages you dictate in. Spells recognises which one you speak and chooses the models for them.",
            parent,
        )
        self._config = config
        self._hotkey = hotkey
        self._notify = notify
        self._probe = probe
        self._gpu = gpu_selection
        self.language_list = LanguageList(show_hotkeys=True, parent=self.body)
        self.language_list.remove_language.connect(lambda code: self.set_language_enabled(code, False))
        self.language_list.add_hotkey.connect(self._add_chord_interactive)
        self.language_list.remove_hotkey.connect(self.remove_language_chord)
        self.add_section(
            "Your languages",
            self.language_list,
            description="A per-language hotkey dictates one utterance in that language without changing the language mode.",
        )
        adder_card = Card(self.body)
        self.adder = LanguageAdder(adder_card)
        self.adder.add_language.connect(lambda code: self.set_language_enabled(code, True))
        adder_card.add_widget(self.adder)
        self.add_section("Add a language", adder_card)
        self.models = ModelsCard(self.body)
        header = self.add_section("Models for your languages", self.models)
        self.hardware_badge = Badge("", "neutral", header)
        header.add_trailing(self.hardware_badge)
        self.finish()
        self.apply_settings(config.settings)

    @property
    def selection(self) -> GpuSelection | None:
        source = self._gpu
        return source() if callable(source) else source

    def hardware(self) -> Hardware:
        return hardware_for(self.selection)

    def apply_settings(self, settings: Settings) -> None:
        self.language_list.apply_settings(settings)
        self.adder.apply_settings(settings)
        self.refresh_models(settings)

    def refresh_models(self, settings: Settings | None = None, *, force: bool = False) -> None:
        settings = settings or self._config.settings
        languages = list(settings.general.enabled_languages)
        measured = self._measured_view(languages)
        if measured is None:
            hardware = self.hardware()
            self.hardware_badge.set_badge("neutral", hardware_text(hardware, self.selection))
            self.models.refresh(languages, hardware, force=force)
            return
        hardware, catalog = measured
        self.hardware_badge.set_badge(
            "neutral", hardware_text(hardware, self.selection, measured=True)
        )
        self.models.refresh(languages, hardware, force=True, catalog=catalog)

    def _measured_view(self, languages: Sequence[str]) -> tuple[Hardware, Any] | None:
        selection = self.selection
        if selection is None or not calibrate.needs_measuring(selection):
            return None
        try:
            results = calibrate.load(calibrate.store_path(self._config.path), selection.name)
            if not results:
                return None
            catalog = modelcatalog.load_catalog()
            everything = frozenset(model.id for model in catalog)
            hardware, _, measured = calibrate.choose_hardware(
                selection, languages, everything, catalog, results
            )
        except Exception:
            log.exception("the speed measurements could not be shown")
            return None
        return hardware, measured

    def set_language_enabled(self, code: str, enabled: bool) -> None:
        set_language_enabled(self._config, self._notify, code, enabled)
        self.apply_settings(self._config.settings)

    def add_language_chord(self, code: str, keys: Any) -> bool:
        folded = fold_keys(keys)
        if not folded or not code:
            return False
        if not chord_is_free(folded, self._probe, self._notify):
            return False
        chord = Chord(keys=folded, language=code)

        def mutate(s: Settings) -> Settings:
            return replace(s, general=replace(s.general, language_chords=[*s.general.language_chords, chord]))

        try:
            self._config.update(mutate)
        except SettingsError as exc:
            self._notify("warning", "Hotkey refused", str(exc))
            self.apply_settings(self._config.settings)
            return False
        self.apply_settings(self._config.settings)
        return True

    def remove_language_chord(self, index: int) -> None:
        def mutate(s: Settings) -> Settings:
            chords = list(s.general.language_chords)
            if 0 <= index < len(chords):
                del chords[index]
            return replace(s, general=replace(s.general, language_chords=chords))

        try:
            self._config.update(mutate)
        except SettingsError as exc:
            self._notify("warning", "Hotkeys", str(exc))
        self.apply_settings(self._config.settings)

    def _add_chord_interactive(self, code: str) -> None:
        settings = self._config.settings
        dialog = ChordCaptureDialog(
            hotkey=self._hotkey,
            active_chords=chords_of(settings),
            title=f"Press a hotkey for {theme.language_name(code)}",
            parent=self,
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.chord:
            self.add_language_chord(code, dialog.chord)


class StatusLine(QtWidgets.QWidget):
    """A dot and a sentence, used by the welcome page to summarise the model choice."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.dot = StatusDot("ok", self)
        layout.addWidget(self.dot)
        self.label = make_label("", "caption", "secondary", wrap=True, parent=self)
        layout.addWidget(self.label, 1)

    def set_text(self, level: str, text: str) -> None:
        self.dot.set_level(level)
        self.label.setText(text)


__all__ = [
    "POPULAR_LANGUAGES",
    "LanguageAdder",
    "LanguageList",
    "LanguageRow",
    "LanguagesPage",
    "ModelsCard",
    "hardware_for",
    "set_language_enabled",
]
