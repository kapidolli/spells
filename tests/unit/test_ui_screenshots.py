"""spells.ui.screenshots: the renderer builds windows from fakes and writes PNGs."""

from __future__ import annotations

import pytest
from PySide6 import QtGui

from spells.ui import screenshots

from .test_ui_support import qt_app


@pytest.fixture(scope="module")
def app():
    return qt_app()


def test_render_writes_the_selected_shots(app, tmp_path):
    saved = screenshots.render(tmp_path, themes=("light",), accent="brand", only="pill-recording-light")
    assert [path.name for path in saved] == ["pill-recording-light.png"]
    image = QtGui.QImage(str(saved[0]))
    assert not image.isNull() and image.width() > 180


def test_render_builds_a_settings_page_and_the_welcome_tour(app, tmp_path):
    saved = screenshots.render(tmp_path, themes=("dark",), accent="brand", only="settings-languages-dark")
    assert [path.name for path in saved] == ["settings-languages-dark.png"]
    saved = screenshots.render(tmp_path, themes=("dark",), accent="brand", only="welcome-1")
    assert [path.name for path in saved] == ["welcome-1-intro-dark.png"]


def test_main_parses_the_options(app, tmp_path, capsys):
    assert screenshots.main([str(tmp_path), "--themes", "light", "--accent", "brand", "--only", "tray-icons"]) == 0
    assert (tmp_path / "tray-icons-light.png").is_file()
    assert "1 screenshots" in capsys.readouterr().out
