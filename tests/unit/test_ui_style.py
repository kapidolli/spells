"""spells.ui.style: the palette per theme and accent, fonts, the stylesheet and the install."""

from __future__ import annotations

import dataclasses

import pytest
from PySide6 import QtGui, QtWidgets

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


def test_light_and_dark_palettes_use_the_winui_accent_shades():
    shades = style.shades_from_palette_bytes(MAGENTA)
    light = build_palette(False, shades)
    dark = build_palette(True, shades)
    assert light.accent == "#8C007C" and light.on_accent == "#FFFFFF"
    assert dark.accent == "#FF35EE" and dark.on_accent == "#000000"
    assert light.accent_text == "#6B005E" and dark.accent_text == "#FF97FA"
    assert light.base != dark.base and light.text != dark.text
    brand = build_palette(False)
    assert brand.accent == BRAND_SHADES.dark1
    assert brand.brand_a == "#5B3CF0" and brand.brand_b == "#16B4E8"


def test_every_palette_entry_is_a_valid_colour():
    for dark in (False, True):
        palette = build_palette(dark, BRAND_SHADES)
        for field in dataclasses.fields(palette):
            if field.name == "dark":
                continue
            assert QtGui.QColor(getattr(palette, field.name)).isValid(), field.name


def test_text_contrast_is_readable_on_surfaces():
    def luminance(hex_colour: str) -> float:
        colour = QtGui.QColor(hex_colour)
        channels = []
        for value in (colour.redF(), colour.greenF(), colour.blueF()):
            channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    def contrast(a: str, b: str) -> float:
        high, low = sorted((luminance(a), luminance(b)), reverse=True)
        return (high + 0.05) / (low + 0.05)

    for dark in (False, True):
        palette = build_palette(dark)
        assert contrast(palette.text, palette.card) >= 12
        assert contrast(palette.text2, palette.card) >= 4.5
        assert contrast(palette.on_accent, palette.accent) >= 4.5


def test_the_stylesheet_is_fully_substituted():
    for dark in (False, True):
        sheet = style.stylesheet(build_palette(dark))
        assert "$" not in sheet
        assert EM_DASH not in sheet
        assert "QFrame[role=\"card\"]" in sheet


def test_fonts_resolve_by_role(app):
    title = style.font("title")
    body = style.font("body")
    assert title.pixelSize() == 28 and title.weight() == QtGui.QFont.Weight.DemiBold
    assert body.pixelSize() == 14 and body.families()[0] == "Segoe UI Variable Text"
    assert style.font("caption").pixelSize() == 12
    with pytest.raises(KeyError):
        style.font("nope")


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


def test_glyph_icons_and_chrome_are_safe_offscreen(app):
    icon = style.glyph_icon(style.Glyph.SETTINGS, gap=6)
    assert not icon.isNull()
    widget = QtWidgets.QWidget()
    assert style.apply_window_chrome(widget) is False
    assert style.round_popup(widget) is False
    widget.close()
    assert style.colorref("#102030") == 0x302010


def test_mix_blends_two_colours():
    colour = style.mix("#FFFFFF", "#000000", 0.5)
    assert 126 <= colour.red() <= 129
    assert style.mix("#FF0000", "#0000FF", 1.0).name() == "#ff0000"


def test_chord_keys_orders_modifiers_first():
    assert theme.chord_keys((0x44, 0x5B, 0x11)) == ["Ctrl", "Win", "D"]
    assert theme.chord_label((0x6B, 0x11)) == "Ctrl+Num +"
    assert theme.chord_keys((0x6B, 0x11)) == ["Ctrl", "Num +"]
