"""spells.ui.about: the version, the privacy summary and the third-party licences of spec 21."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PySide6 import QtCore

from spells.ui.about import AboutPage, reflow
from spells.ui.diagnostics import LICENSE_COMPONENTS
from spells.ui.updatecheck import Phase, UpdateView
from spells.updates import Release

from .test_ui_support import qt_app

REPO_LICENSES = Path(__file__).resolve().parents[2] / "build" / "licenses"
POINTER = "The full text ships in the licenses folder"


@pytest.fixture(scope="module")
def app():
    return qt_app()


def test_licences_list_and_text(app):
    page = AboutPage()
    assert page.licenses.count() == len(LICENSE_COMPONENTS)
    page.licenses.setCurrentRow(0)
    assert LICENSE_COMPONENTS[0][1] in page.license_text.toPlainText()
    page.close()


def test_reflow_joins_hard_wrapped_lines_but_keeps_structure():
    text = (
        "whisper.cpp, llama.cpp, ggml\n"
        "License: MIT\n"
        "\n"
        "=== whisper_cpp v1 (MIT) ===\n"
        "\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files.\n"
        "\n"
        "1. First clause that is long enough to be a wrapped line of legal text here\n"
        "2. Second clause\n"
        "  indented continuation\n"
    )
    result = reflow(text).splitlines()
    assert result[0] == "whisper.cpp, llama.cpp, ggml"
    assert result[1] == "License: MIT"
    assert result[3] == "=== whisper_cpp v1 (MIT) ==="
    assert result[5].endswith("obtaining a copy of this software and associated documentation files.")
    assert "2. Second clause" in result
    assert "  indented continuation" in result



class FakeCoordinator(QtCore.QObject):
    """The whole coordinator surface the page uses, with the calls recorded."""

    changed = QtCore.Signal(object)

    def __init__(self, view: UpdateView) -> None:
        super().__init__()
        self.view = view
        self.calls: list[str] = []

    def push(self, **fields) -> None:
        self.view = replace(self.view, **fields)
        self.changed.emit(self.view)

    def check(self, manual: bool = True) -> None:
        self.calls.append(f"check:{manual}")

    def set_weekly_check(self, enabled: bool) -> None:
        self.calls.append(f"weekly:{bool(enabled)}")

    def start_download(self) -> None:
        self.calls.append("download")

    def cancel_download(self) -> None:
        self.calls.append("cancel")

    def install(self) -> None:
        self.calls.append("install")

    def apply_settings(self, settings) -> None:
        self.calls.append("settings")


def a_release(**overrides) -> Release:
    fields = {
        "version": "0.3.0",
        "released": "2026-10-01",
        "url": "https://spells.example.com/Spells-Online-Setup-0.3.0.exe",
        "size_bytes": 97_296_166,
        "sha256": "a" * 64,
        "changes": ("A faster Albanian.", "The meter moves with your voice."),
    }
    fields.update(overrides)
    return Release(**fields)


def attached(view: UpdateView) -> tuple[AboutPage, FakeCoordinator]:
    page = AboutPage()
    coordinator = FakeCoordinator(view)
    page.attach(coordinator)
    return page, coordinator


def test_the_check_button_always_asks_for_a_check(app):
    page, coordinator = attached(UpdateView(phase=Phase.IDLE, message="Not checked yet."))

    assert page.check_button.isEnabled()
    page.check_button.click()

    assert coordinator.calls == ["check:True"]
    page.close()


def test_flipping_the_switch_asks_the_coordinator_to_store_it(app):
    page, coordinator = attached(UpdateView(phase=Phase.IDLE))

    page.weekly_switch.click()

    assert coordinator.calls == ["weekly:True"]
    page.close()


def test_an_available_update_shows_its_version_date_size_and_changes(app):
    release = a_release()
    page, coordinator = attached(UpdateView(phase=Phase.IDLE))

    coordinator.push(phase=Phase.AVAILABLE, release=release)

    assert not page.release_panel.isHidden()
    headline = page.release_label.text()
    assert "Spells 0.3.0" in headline
    assert "1 October 2026" in headline
    assert "97.3 MB" in headline
    for line in release.changes:
        assert line in page.changes_label.text()
    assert not page.install_button.isHidden()
    assert page.progress.isHidden()
    page.close()


def test_the_install_button_starts_the_download(app):
    page, coordinator = attached(UpdateView(phase=Phase.IDLE))
    coordinator.push(phase=Phase.AVAILABLE, release=a_release())

    page.install_button.click()

    assert coordinator.calls == ["download"]
    page.close()


def test_the_download_shows_progress_and_a_cancel_button(app):
    page, coordinator = attached(UpdateView(phase=Phase.IDLE))
    coordinator.push(phase=Phase.AVAILABLE, release=a_release())

    coordinator.push(phase=Phase.DOWNLOADING, done_bytes=48_648_083, total_bytes=97_296_166)

    assert not page.progress.isHidden()
    assert page.progress.value() == 48_648_083
    assert page.progress.maximum() == 97_296_166
    assert "97.3 MB" in page.progress.format()
    assert not page.cancel_button.isHidden()
    assert page.install_button.isHidden()
    assert not page.check_button.isEnabled()

    page.cancel_button.click()
    assert coordinator.calls == ["cancel"]
    page.close()


def test_a_verified_download_says_spells_will_close_and_installs_on_the_button(app):
    page, coordinator = attached(UpdateView(phase=Phase.IDLE))
    coordinator.push(phase=Phase.AVAILABLE, release=a_release())

    coordinator.push(phase=Phase.READY, message="Spells will close while it installs.")

    assert "close" in page.version_row.description_label.text()
    assert not page.install_button.isHidden()
    assert "install" in page.install_button.text().lower()
    assert page.cancel_button.isHidden()

    page.install_button.click()
    assert coordinator.calls == ["install"]
    page.close()


def test_a_blocked_update_offers_no_install(app):
    page, coordinator = attached(UpdateView(phase=Phase.IDLE))

    coordinator.push(
        phase=Phase.BLOCKED,
        release=a_release(minimum_version="0.9.0"),
        message="Install 0.9.0 first, then check again.",
    )

    assert page.install_button.isHidden()
    assert "0.9.0" in page.version_row.description_label.text()
    page.close()


def test_a_failed_check_leaves_the_button_usable_again(app):
    page, coordinator = attached(UpdateView(phase=Phase.CHECKING, message="Checking..."))

    assert not page.check_button.isEnabled()
    coordinator.push(phase=Phase.ERROR, message="Spells could not reach the update server.")

    assert page.check_button.isEnabled()
    assert "could not reach" in page.version_row.description_label.text()
    assert page.release_panel.isHidden()
    page.close()

