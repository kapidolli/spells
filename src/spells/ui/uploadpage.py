from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spells import upload
from spells.config import ConfigStore, Settings, SettingsError, UploadSettings
from spells.ui import style
from spells.ui.uploading import UploadView, status_text
from spells.ui.welcome import Notify
from spells.ui.widgets import (
    Card,
    GlyphLabel,
    ScrollPage,
    SegmentedControl,
    SettingRow,
    ToggleSwitch,
    make_button,
    make_label,
)

log = logging.getLogger(__name__)

TITLE = "Upload"
SUBTITLE = "Send your dictations to a server you run. Nothing is sent until you set it up here."
ENABLE_TITLE = "Upload my dictations to a server"
ENABLE_HINT = (
    "Off by default. With it on, Spells sends each dictation to the address below: what you "
    "said, the cleaned and delivered text, the app and its window title, the language and "
    "the timings."
)
URL_TITLE = "Server address"
URL_HINT = "Spells sends to this address exactly as you type it."
UNENCRYPTED_HINT = "Sent unencrypted, so anyone on the network in between can read it."
URL_PLACEHOLDER = "https://example.com/spells"
TOKEN_TITLE = "Token"
TOKEN_HINT = (
    "Sent as a bearer token. Kept encrypted for your Windows account, never in the "
    "settings file."
)
SCHEDULE_TITLE = "When to upload"
SCHEDULE_HINT = (
    "Daily and weekly run in the background, never during a dictation. Manual waits for "
    "Upload now."
)
SCHEDULE_LABELS = {
    upload.SCHEDULE_MANUAL: "Manual",
    upload.SCHEDULE_DAILY: "Daily",
    upload.SCHEDULE_WEEKLY: "Weekly",
}
AUDIO_TITLE = "Include the recordings"
AUDIO_HINT = (
    "Also sends the recordings Spells keeps, as WAV files. Spells keeps them only with Keep "
    "the recordings on the History page."
)
KEEP_AUDIO_QUESTION = (
    "Spells can only send the recordings it keeps. Turn on Keep the recordings on the "
    "History page too?"
)
SKIP_TITLE = "Never upload from these apps"
SKIP_HINT = (
    "Process names separated by commas, such as keepassxc.exe. What you dictate into them "
    "stays on this computer."
)
SKIP_PLACEHOLDER = "keepassxc.exe, 1password.exe"
DEVICE_TITLE = "Name of this computer"
DEVICE_HINT = "How the server tells your computers apart. Empty uses the name Windows gives it."
HOLD_NOTE = (
    "While uploading is on, dictations not sent yet are kept for up to 30 days, even past "
    "your history limit, and so are their recordings when you include them."
)
STATUS_TITLE = "Status"
TEST_BUTTON = "Test connection"
UPLOAD_BUTTON = "Upload now"
UNATTACHED_TEXT = "Uploading is not available in this window."


def parse_apps(text: str) -> list[str]:
    apps: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        name = part.strip()
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            apps.append(name)
    return apps


@contextmanager
def _blocked(*widgets: QtCore.QObject) -> Iterator[None]:
    previous = [widget.blockSignals(True) for widget in widgets]
    try:
        yield
    finally:
        for widget, state in zip(widgets, previous, strict=True):
            widget.blockSignals(state)


def _line_edit(placeholder: str, parent: QtWidgets.QWidget) -> QtWidgets.QLineEdit:
    edit = QtWidgets.QLineEdit(parent)
    edit.setFont(style.font("body"))
    edit.setPlaceholderText(placeholder)
    edit.setMinimumWidth(300)
    return edit


class UploadPage(ScrollPage):
    def __init__(
        self,
        *,
        config: ConfigStore,
        notify: Notify,
        confirm: Callable[[str, str], bool],
        clock: Callable[[], float] = time.time,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(TITLE, SUBTITLE, parent)
        self._config = config
        self._notify = notify
        self._confirm = confirm
        self._clock = clock
        self._coordinator: Any = None
        self._view = UploadView()
        self._loading = False

        server = Card(self.body)
        self.enabled = ToggleSwitch(server)
        self.enabled.setAccessibleName(ENABLE_TITLE)
        self.enabled.toggled.connect(self._on_enabled)
        server.add_row(
            SettingRow(ENABLE_TITLE, ENABLE_HINT, self.enabled, glyph=style.Glyph.UPLOAD)
        )
        self.url = _line_edit(URL_PLACEHOLDER, server)
        self.url.setAccessibleName(URL_TITLE)
        self.url.editingFinished.connect(self._url_edited)
        self.url.textChanged.connect(self._url_typed)
        self.url_row = SettingRow(URL_TITLE, URL_HINT, self.url, wide_control=True)
        server.add_row(self.url_row)
        self.token = _line_edit("", server)
        self.token.setAccessibleName(TOKEN_TITLE)
        self.token.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.token.editingFinished.connect(self._token_edited)
        server.add_row(SettingRow(TOKEN_TITLE, TOKEN_HINT, self.token, wide_control=True))
        self.schedule = SegmentedControl(server)
        for code in upload.SCHEDULES:
            self.schedule.addItem(SCHEDULE_LABELS[code], code)
        self.schedule.currentIndexChanged.connect(self._on_schedule)
        server.add_row(SettingRow(SCHEDULE_TITLE, SCHEDULE_HINT, self.schedule))
        self.add_section("Server", server)

        sent = Card(self.body)
        self.include_audio = ToggleSwitch(sent)
        self.include_audio.setAccessibleName(AUDIO_TITLE)
        self.include_audio.toggled.connect(self._on_include_audio)
        sent.add_row(
            SettingRow(AUDIO_TITLE, AUDIO_HINT, self.include_audio, glyph=style.Glyph.WAVEFORM)
        )
        self.skip_apps = _line_edit(SKIP_PLACEHOLDER, sent)
        self.skip_apps.setAccessibleName(SKIP_TITLE)
        self.skip_apps.editingFinished.connect(self._skip_edited)
        sent.add_row(SettingRow(SKIP_TITLE, SKIP_HINT, self.skip_apps, wide_control=True))
        self.device_name = _line_edit(upload.computer_name(), sent)
        self.device_name.setAccessibleName(DEVICE_TITLE)
        self.device_name.editingFinished.connect(self._device_edited)
        sent.add_row(SettingRow(DEVICE_TITLE, DEVICE_HINT, self.device_name, wide_control=True))
        note = QtWidgets.QWidget(sent)
        note_layout = QtWidgets.QHBoxLayout(note)
        note_layout.setContentsMargins(16, 0, 16, 12)
        note_layout.setSpacing(8)
        note_layout.addWidget(GlyphLabel(style.Glyph.SHIELD, 14, tone="secondary", parent=note))
        self.hold_note = make_label(HOLD_NOTE, "caption", "secondary", wrap=True, parent=note)
        note_layout.addWidget(self.hold_note, 1)
        sent.add_widget(note)
        self.add_section("What is sent", sent)

        status = Card(self.body)
        self.test_button = make_button(TEST_BUTTON, glyph=style.Glyph.CHECK, parent=status)
        self.test_button.clicked.connect(self._test)
        self.upload_button = make_button(
            UPLOAD_BUTTON, "primary", glyph=style.Glyph.UPLOAD, parent=status
        )
        self.upload_button.clicked.connect(self._upload_now)
        buttons = QtWidgets.QWidget(status)
        buttons_layout = QtWidgets.QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)
        buttons_layout.addWidget(self.test_button)
        buttons_layout.addWidget(self.upload_button)
        buttons_layout.addStretch(1)
        self.status_row = SettingRow(
            STATUS_TITLE, UNATTACHED_TEXT, glyph=style.Glyph.INFO, below=buttons
        )
        status.add_row(self.status_row)
        self.add_section("Status", status)
        self.finish()

        self.apply_settings(config.settings)

    def attach(self, coordinator: Any) -> None:
        self._coordinator = coordinator
        coordinator.changed.connect(self._on_view)
        self._view = coordinator.view
        with _blocked(self.token):
            self.token.setText(coordinator.token())
            self.token.setCursorPosition(0)
        self._render()

    @property
    def status_line(self) -> str:
        return self.status_row.description_label.text()

    def apply_settings(self, settings: Settings) -> None:
        if self._coordinator is not None:
            self._coordinator.apply_settings(settings)
        current = self._config.settings.upload
        self._loading = True
        try:
            with _blocked(self.enabled, self.schedule, self.include_audio):
                self.enabled.setChecked(current.enabled)
                self.schedule.setCurrentIndex(max(0, self.schedule.findData(current.schedule)))
                self.include_audio.setChecked(current.include_audio)
            for edit, text in (
                (self.url, current.url),
                (self.skip_apps, ", ".join(current.skip_apps)),
                (self.device_name, current.device_name),
            ):
                if not edit.hasFocus() and edit.text() != text:
                    edit.setText(text)
                    edit.setCursorPosition(0)
        finally:
            self._loading = False
        self._render()

    def _on_view(self, view: UploadView) -> None:
        self._view = view
        self._render()

    def _render(self) -> None:
        current = self._config.settings.upload
        coordinator = self._coordinator
        address = bool(current.url) and not upload.url_problem(current.url)
        working = bool(self._view.working)
        if coordinator is None:
            self.status_row.set_description(UNATTACHED_TEXT)
        else:
            self.status_row.set_description(status_text(current, self._view, self._clock()))
        self.token.setEnabled(coordinator is not None)
        self.test_button.setEnabled(coordinator is not None and address and not working)
        self.upload_button.setEnabled(
            coordinator is not None and address and current.enabled and not working
        )
        self._url_typed(self.url.text())

    def _url_typed(self, text: str) -> None:
        unencrypted = upload.is_unencrypted(text) and not upload.url_problem(text)
        self.url_row.set_description(UNENCRYPTED_HINT if unencrypted else URL_HINT)

    def _update(self, mutate: Callable[[UploadSettings], UploadSettings]) -> bool:
        try:
            settings = self._config.update(
                lambda current: replace(current, upload=mutate(current.upload))
            )
        except SettingsError as exc:
            self._notify("warning", TITLE, str(exc))
            self.apply_settings(self._config.settings)
            return False
        self.apply_settings(settings)
        return True

    def _on_enabled(self, checked: bool) -> None:
        if self._loading:
            return
        self._update(lambda current: replace(current, enabled=bool(checked)))

    def _url_edited(self) -> None:
        text = self.url.text().strip()
        stored = self._config.settings.upload.url
        if text == stored:
            return
        problem = upload.url_problem(text) if text else ""
        if problem:
            self._notify("warning", TITLE, problem)
            with _blocked(self.url):
                self.url.setText(stored)
            self._url_typed(stored)
            return
        self._update(lambda current: replace(current, url=text))

    def _token_edited(self) -> None:
        coordinator = self._coordinator
        if coordinator is None:
            return
        try:
            coordinator.set_token(self.token.text())
        except OSError as exc:
            log.warning("the upload token could not be saved: %s", exc.__class__.__name__)
            self._notify("warning", TITLE, "The token could not be saved on this computer.")

    def _on_schedule(self, index: int) -> None:
        code = self.schedule.itemData(index)
        if self._loading or not code or code == self._config.settings.upload.schedule:
            return
        self._update(lambda current: replace(current, schedule=code))

    def _on_include_audio(self, checked: bool) -> None:
        if self._loading:
            return
        if not self._update(lambda current: replace(current, include_audio=bool(checked))):
            return
        if not checked or self._config.settings.history.keep_audio:
            return
        if not self._confirm(AUDIO_TITLE, KEEP_AUDIO_QUESTION):
            return
        try:
            settings = self._config.update(
                lambda current: replace(current, history=replace(current.history, keep_audio=True))
            )
        except SettingsError as exc:
            self._notify("warning", TITLE, str(exc))
            return
        self.apply_settings(settings)

    def _skip_edited(self) -> None:
        apps = parse_apps(self.skip_apps.text())
        if apps == self._config.settings.upload.skip_apps:
            return
        self._update(lambda current: replace(current, skip_apps=apps))

    def _device_edited(self) -> None:
        name = self.device_name.text().strip()
        if name == self._config.settings.upload.device_name:
            return
        self._update(lambda current: replace(current, device_name=name))

    def _test(self) -> None:
        if self._coordinator is not None:
            self._coordinator.test_connection()

    def _upload_now(self) -> None:
        if self._coordinator is not None:
            self._coordinator.upload_now()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._coordinator is not None:
            self._coordinator.refresh()


__all__ = [
    "AUDIO_TITLE",
    "HOLD_NOTE",
    "KEEP_AUDIO_QUESTION",
    "TITLE",
    "UNENCRYPTED_HINT",
    "URL_HINT",
    "UploadPage",
    "parse_apps",
]
