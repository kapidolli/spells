"""The About page: the mark, the version, updates (spec 19.7) and the licences of spec 21.

The update section is the only place in the app that can open a connection, and only
because somebody pressed a button or switched the weekly check on. The page itself holds no
policy: `spells.ui.updatecheck.UpdateCoordinator` decides everything and hands this page one
`UpdateView` at a time, which is why the page works unattached (a settings dialog built
without a coordinator simply shows the version and the licences).
"""

from __future__ import annotations

import re
from typing import Any

from PySide6 import QtCore, QtWidgets

from spells import __version__, updates
from spells.ui import style, theme
from spells.ui.diagnostics import LICENSE_COMPONENTS, license_text
from spells.ui.updatecheck import UNCONFIGURED_TEXT, Phase, UpdateView
from spells.ui.widgets import (
    Card,
    LogoMark,
    ScrollPage,
    SettingRow,
    ToggleSwitch,
    make_button,
    make_label,
)

LIST_MARKER = re.compile(r"(\d+|[a-zA-Z])[.)]\s")

CHECK_BUTTON = "Check for updates"
WEEKLY_TITLE = "Check every week"
WEEKLY_DESCRIPTION = (
    "Off by default. With it on, Spells asks the update server once a week, never during a "
    "dictation, and tells you in the tray when there is something new. Nothing else about "
    "the app changes: no audio, no text and no history ever leaves this computer."
)
INSTALL_BUTTON = "Install this version"
CANCEL_BUTTON = "Cancel"
NO_CHANGES = "The version file lists no changes for this release."


def reflow(text: str) -> str:
    """Join hard-wrapped lines into paragraphs so the narrow pane wraps them once."""
    lines = text.splitlines()
    result: list[str] = []
    for line in lines:
        previous = result[-1] if result else ""
        joinable = (
            previous
            and line.strip()
            and len(previous) >= 50
            and not line.startswith((" ", "\t", "-", "*", "=", "("))
            and not previous.startswith("===")
            and not LIST_MARKER.match(line)
        )
        if joinable:
            result[-1] = previous.rstrip() + " " + line.strip()
        else:
            result.append(line)
    return "\n".join(result)


def release_headline(view: UpdateView) -> str:
    """The one line that names the version, its date and its size."""
    release = view.release
    if release is None:
        return ""
    parts = [f"{theme.APP_NAME} {release.version}"]
    when = updates.date_text(release.released)
    if when:
        parts.append(when)
    parts.append(updates.size_text(release.size_bytes))
    return ", ".join(parts)


class AboutPage(ScrollPage):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("About", "", parent)
        self.title_label.hide()
        self._coordinator: Any = None

        hero = QtWidgets.QWidget(self.body)
        hero_layout = QtWidgets.QHBoxLayout(hero)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(20)
        self.logo = LogoMark(72, hero)
        hero_layout.addWidget(self.logo)
        names = QtWidgets.QVBoxLayout()
        names.setSpacing(2)
        names.addStretch(1)
        names.addWidget(make_label(theme.APP_NAME, "hero", parent=hero))
        self.version_label = make_label(f"Version {__version__}", "body", "secondary", parent=hero)
        names.addWidget(self.version_label)
        names.addWidget(make_label("Private dictation for Windows.", "body", "secondary", parent=hero))
        names.addStretch(1)
        hero_layout.addLayout(names, 1)
        self.body_layout.insertWidget(0, hero)
        self.body_layout.insertSpacing(1, 8)

        self._build_updates()

        privacy = Card(self.body)
        privacy.add_row(
            SettingRow(
                "Runs on this computer",
                "Speech recognition and cleanup run in local engines. Nothing you say is sent anywhere.",
                glyph=style.Glyph.SHIELD,
            )
        )
        privacy.add_row(
            SettingRow(
                "History stays here",
                "Dictations are stored locally. Change how long they are kept, or clear them, on the History page.",
                glyph=style.Glyph.HISTORY,
            )
        )
        self.add_section("Privacy", privacy)

        licenses_card = Card(self.body)
        panes = QtWidgets.QWidget(licenses_card)
        panes_layout = QtWidgets.QHBoxLayout(panes)
        panes_layout.setContentsMargins(8, 8, 12, 12)
        panes_layout.setSpacing(12)
        self.licenses = QtWidgets.QListWidget(panes)
        self.licenses.setProperty("role", "embedded")
        self.licenses.setFont(style.font("body"))
        self.licenses.setFixedWidth(300)
        self.licenses.setWordWrap(True)
        self.licenses.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for component, license_name, _names in LICENSE_COMPONENTS:
            item = QtWidgets.QListWidgetItem(f"{component}\n{license_name}")
            item.setToolTip(f"{component}: {license_name}")
            self.licenses.addItem(item)
        self.licenses.currentRowChanged.connect(self._show_license)
        panes_layout.addWidget(self.licenses)
        self.license_text = QtWidgets.QPlainTextEdit(panes)
        self.license_text.setProperty("role", "mono")
        self.license_text.setReadOnly(True)
        self.license_text.setFont(style.font("caption"))
        self.license_text.setPlaceholderText("Select a component to read its licence.")
        panes_layout.addWidget(self.license_text, 1)
        panes.setMinimumHeight(420)
        licenses_card.add_widget(panes)
        self.add_section("Third-party licenses", licenses_card, description="The components Spells is built from and the terms they ship under.")
        self.finish()
        self.apply_update_view(UpdateView(phase=Phase.UNCONFIGURED, message=UNCONFIGURED_TEXT))

    # Updates ------------------------------------------------------------------------------

    def _build_updates(self) -> None:
        card = Card(self.body)
        self.check_button = make_button(CHECK_BUTTON, "primary", glyph=style.Glyph.REFRESH)
        self.check_button.clicked.connect(self._on_check)
        self.version_row = SettingRow(
            f"{theme.APP_NAME} {__version__}",
            "",
            self.check_button,
            glyph=style.Glyph.INFO,
        )
        card.add_row(self.version_row)
        self.weekly_switch = ToggleSwitch()
        self.weekly_switch.toggled.connect(self._on_weekly)
        card.add_row(
            SettingRow(
                WEEKLY_TITLE,
                WEEKLY_DESCRIPTION,
                self.weekly_switch,
                glyph=style.Glyph.GLOBE,
            )
        )
        self.update_card = card
        self.add_section(
            "Updates",
            card,
            description=(
                "Spells never looks for an update on its own unless you switch the weekly "
                "check on, and an update downloads the installer only, never the models."
            ),
        )

        panel = QtWidgets.QFrame(self.body)
        panel.setProperty("role", "card")
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 14, 16, 16)
        panel_layout.setSpacing(8)
        self.release_label = make_label("", "body_strong", wrap=True, parent=panel)
        panel_layout.addWidget(self.release_label)
        self.changes_label = make_label("", "body", "secondary", wrap=True, parent=panel)
        panel_layout.addWidget(self.changes_label)
        self.progress = QtWidgets.QProgressBar(panel)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        panel_layout.addWidget(self.progress)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(8)
        self.install_button = make_button(INSTALL_BUTTON, "primary", glyph=style.Glyph.DOWNLOAD)
        self.install_button.clicked.connect(self._on_install)
        buttons.addWidget(self.install_button)
        self.cancel_button = make_button(CANCEL_BUTTON, glyph=style.Glyph.CLOSE)
        self.cancel_button.clicked.connect(self._on_cancel)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        panel_layout.addLayout(buttons)
        self.release_panel = panel
        self.add_widget(panel, spacing_before=8)
        panel.setVisible(False)

    def attach(self, coordinator: Any) -> None:
        """Wire the page to the coordinator that owns the checking, the download and the quit."""
        self._coordinator = coordinator
        coordinator.changed.connect(self.apply_update_view)
        self.apply_update_view(coordinator.view)

    def apply_update_view(self, view: UpdateView) -> None:
        """Render one snapshot. Every decision was taken before this was called."""
        phase = view.phase
        configured = phase is not Phase.UNCONFIGURED
        working = phase in (Phase.CHECKING, Phase.DOWNLOADING, Phase.INSTALLING)
        self.check_button.setEnabled(configured and not working)
        self.weekly_switch.setEnabled(configured)
        blocked = self.weekly_switch.blockSignals(True)
        self.weekly_switch.setChecked(bool(view.weekly_check))
        self.weekly_switch.blockSignals(blocked)
        self.version_row.set_description(view.message)

        release = view.release
        show_panel = release is not None and phase in (
            Phase.AVAILABLE,
            Phase.BLOCKED,
            Phase.DOWNLOADING,
            Phase.READY,
            Phase.INSTALLING,
        )
        self.release_panel.setVisible(bool(show_panel))
        if not show_panel:
            return
        assert release is not None
        self.release_label.setText(release_headline(view))
        self.changes_label.setText(
            "\n".join(f"- {line}" for line in release.changes) if release.changes else NO_CHANGES
        )
        downloading = phase is Phase.DOWNLOADING
        self.progress.setVisible(downloading)
        if downloading:
            self.progress.setMaximum(max(1, view.total_bytes))
            self.progress.setValue(min(view.done_bytes, max(1, view.total_bytes)))
            self.progress.setFormat(
                f"{updates.size_text(view.done_bytes)} of "
                f"{updates.size_text(view.total_bytes)}"
            )
        self.install_button.setVisible(phase in (Phase.AVAILABLE, Phase.READY))
        self.install_button.setEnabled(phase in (Phase.AVAILABLE, Phase.READY))
        self.install_button.setText(
            INSTALL_BUTTON if phase is Phase.AVAILABLE else "Close Spells and install"
        )
        self.cancel_button.setVisible(downloading)

    def apply_settings(self, settings: object) -> None:
        coordinator = self._coordinator
        if coordinator is not None:
            coordinator.apply_settings(settings)

    def _on_check(self) -> None:
        if self._coordinator is not None:
            self._coordinator.check(manual=True)

    def _on_weekly(self, checked: bool) -> None:
        if self._coordinator is not None:
            self._coordinator.set_weekly_check(bool(checked))

    def _on_install(self) -> None:
        coordinator = self._coordinator
        if coordinator is None:
            return
        if coordinator.view.phase is Phase.READY:
            coordinator.install()
        else:
            coordinator.start_download()

    def _on_cancel(self) -> None:
        if self._coordinator is not None:
            self._coordinator.cancel_download()

    # Licences -----------------------------------------------------------------------------

    def showEvent(self, event: object) -> None:
        super().showEvent(event)
        if self.licenses.currentRow() < 0 and self.licenses.count():
            self.licenses.setCurrentRow(0)

    def _show_license(self, row: int) -> None:
        if 0 <= row < len(LICENSE_COMPONENTS):
            component, license_name, names = LICENSE_COMPONENTS[row]
            self.license_text.setPlainText(reflow(license_text(component, license_name, names)))


__all__ = ["AboutPage", "release_headline"]
