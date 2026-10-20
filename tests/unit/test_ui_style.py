"""spells.ui.style: the palette per theme and accent, fonts, the stylesheet and the install."""

from __future__ import annotations

import dataclasses

import pytest
from PySide6 import QtGui, QtWidgets

from spells import platform
from spells.platform.keys_windows import WindowsKeys
from spells.ui import style, theme
from spells.ui.style import BRAND_SHADES, AccentShades, build_palette

from .test_ui_support import flush, qt_app

MAGENTA = bytes.fromhex("FF97FA00FF35EE00C800B3009A0089008C007C006B005E004C00420000CC6A00")
EM_DASH = "\u2014"


@pytest.fixture(scope="module")
def app():
    return qt_app()


def test_the_explorer_accent_palette_is_read_lightest_first():
    shades = style.shades_from_palette_bytes(MAGENTA)
    assert shades == AccentShades("#FF97FA", "#FF35EE", "#C800B3", "#9A0089", "#8C007C", "#6B005E", "#4C0042")
    assert style.shades_from_palette_bytes(b"\x00" * 8) is None
    assert style.shades_from_palette_bytes("text") is None


def test_one_colour_gives_seven_ordered_shades(app):
    shades = style.shades_from_colour(QtGui.QColor("#0078D4"))
    lightness = [QtGui.QColor(getattr(shades, f.name)).lightnessF() for f in dataclasses.fields(shades)]
    assert lightness == sorted(lightness, reverse=True)
    assert shades.base == "#0078D4"


def test_the_stylesheet_is_fully_substituted():
    for dark in (False, True):
        sheet = style.stylesheet(build_palette(dark))
        assert "$" not in sheet
        assert EM_DASH not in sheet
        assert "QFrame[role=\"card\"]" in sheet


def test_install_applies_the_palette_style_and_sheet(app):
    for leftover in app.topLevelWidgets():
        leftover.deleteLater()
    flush(app)
    changed: list[object] = []
    style.notifier().changed.connect(changed.append)
    installed = style.install(app, dark=True, accent=BRAND_SHADES, watch=False)
    assert style.palette() == installed and installed.dark
    assert "QFrame" in app.styleSheet()
    assert isinstance(app.style(), (style.SpellsStyle, QtWidgets.QStyle))
    assert app.palette().color(QtGui.QPalette.ColorRole.Window).name().upper() == installed.base
    assert changed and changed[-1] == installed
    sheet = app.styleSheet()
    again = style.install(app, dark=True, accent=BRAND_SHADES, watch=False)
    assert again == installed and app.styleSheet() is not None and app.styleSheet() == sheet


def test_chord_keys_orders_modifiers_first(use_platform):
    use_platform(dataclasses.replace(platform.current(), keys=WindowsKeys()))
    assert theme.chord_keys((0x44, 0x5B, 0x11)) == ["Ctrl", "Win", "D"]
    assert theme.chord_label((0x6B, 0x11)) == "Ctrl+Num +"
    assert theme.chord_keys((0x6B, 0x11)) == ["Ctrl", "Num +"]
