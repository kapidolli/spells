"""Design tokens, type, the application stylesheet and the Windows window chrome.

The palette follows the Windows app theme (AppsUseLightTheme) and uses the brand indigo of the
logo as its accent, over a fixed set of cool neutral surfaces. install() can follow the Windows
accent colour instead (the Explorer AccentPalette: Dark 1 drives the controls in light mode,
Light 2 in dark mode, as WinUI does). Custom-painted widgets read `palette()` at paint time, so a theme
change only needs `install()` again and a repaint.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass
from string import Template
from typing import ClassVar

from PySide6 import QtCore, QtGui, QtWidgets

log = logging.getLogger(__name__)

SPACE_XS = 4
SPACE_S = 8
SPACE_M = 12
SPACE_L = 16
SPACE_XL = 24
SPACE_XXL = 32

RADIUS_SMALL = 4
RADIUS_CONTROL = 6
RADIUS_WINDOW = 8
RADIUS_CARD = 10
RADIUS_KEYCAP = 5
RADIUS_CHIP = 16

CONTROL_HEIGHT = 32
NAV_WIDTH = 256
NAV_ITEM_HEIGHT = 36
ROW_MIN_HEIGHT = 64

TEXT_FAMILIES = ("Segoe UI Variable Text", "Segoe UI")
TEXT_SEMIBOLD_FAMILIES = ("Segoe UI Variable Text Semibold", "Segoe UI Semibold", "Segoe UI")
SMALL_FAMILIES = ("Segoe UI Variable Small", "Segoe UI")
SMALL_SEMIBOLD_FAMILIES = ("Segoe UI Variable Small Semibold", "Segoe UI Semibold", "Segoe UI")
DISPLAY_SEMIBOLD_FAMILIES = ("Segoe UI Variable Display Semibold", "Segoe UI Semibold", "Segoe UI")
MONO_FAMILIES = ("Cascadia Mono", "Consolas", "Courier New")
ICON_FAMILIES = ("Segoe Fluent Icons", "Segoe MDL2 Assets")

FONT_ROLES: dict[str, tuple[tuple[str, ...], int, int]] = {
    "caption": (SMALL_FAMILIES, 12, 400),
    "caption_strong": (SMALL_SEMIBOLD_FAMILIES, 12, 600),
    "body": (TEXT_FAMILIES, 14, 400),
    "body_strong": (TEXT_SEMIBOLD_FAMILIES, 14, 600),
    "lead": (TEXT_FAMILIES, 15, 400),
    "brand": (DISPLAY_SEMIBOLD_FAMILIES, 18, 600),
    "subtitle": (DISPLAY_SEMIBOLD_FAMILIES, 20, 600),
    "stat": (DISPLAY_SEMIBOLD_FAMILIES, 22, 600),
    "title": (DISPLAY_SEMIBOLD_FAMILIES, 28, 600),
    "hero": (DISPLAY_SEMIBOLD_FAMILIES, 32, 600),
    "mono": (MONO_FAMILIES, 12, 400),
}


class Glyph:
    """Segoe Fluent Icons code points (the same in Segoe MDL2 Assets on Windows 10)."""

    SETTINGS = chr(0xE713)
    GLOBE = chr(0xE774)
    MICROPHONE = chr(0xE720)
    KEYBOARD = chr(0xE765)
    BRUSH = chr(0xE771)
    APPS = chr(0xECAA)
    BOOK = chr(0xE82D)
    HISTORY = chr(0xE81C)
    PULSE = chr(0xE9D9)
    INFO = chr(0xE946)
    DELETE = chr(0xE74D)
    CLOSE = chr(0xE711)
    ADD = chr(0xE710)
    SEARCH = chr(0xE721)
    COPY = chr(0xE8C8)
    FOLDER = chr(0xE8B7)
    SAVE = chr(0xE74E)
    REFRESH = chr(0xE72C)
    PAUSE = chr(0xE769)
    POWER = chr(0xE7E8)
    WARNING = chr(0xE7BA)
    CHECK = chr(0xE73E)
    CHIP = chr(0xE950)
    WINDOW = chr(0xE737)
    EDIT = chr(0xE70F)
    UPLOAD = chr(0xE898)
    DOWNLOAD = chr(0xE896)
    SHIELD = chr(0xEA18)
    WAVEFORM = chr(0xE9E9)
    LOCK = chr(0xE72E)


@dataclass(frozen=True)
class AccentShades:
    """The seven shades of a Windows accent, lightest first."""

    light3: str
    light2: str
    light1: str
    base: str
    dark1: str
    dark2: str
    dark3: str


BRAND_SHADES = AccentShades("#D6CCFF", "#B3A1FF", "#8A70F7", "#5B3CF0", "#5234E0", "#4128BF", "#2E1B91")

_LIGHT = {
    "base": "#F3F3F7",
    "layer": "#F9F9FB",
    "layer_border": "#E6E6EC",
    "card": "#FFFFFF",
    "card_border": "#E5E5EC",
    "card_hover": "#F7F7FA",
    "control": "#FFFFFF",
    "control_hover": "#F6F6F9",
    "control_pressed": "#F0F0F4",
    "control_border": "#DCDCE3",
    "control_bottom": "#C8C8D1",
    "nav_hover": "#EAEAF0",
    "nav_selected": "#E6E6EE",
    "divider": "#EDEDF2",
    "text": "#1A1A22",
    "text2": "#5C5C6B",
    "text3": "#8B8B99",
    "disabled": "#B0B0BC",
    "success": "#0F7B0F",
    "success_bg": "#E6F4E6",
    "caution": "#9D5D00",
    "caution_bg": "#FFF4D6",
    "critical": "#C42B1C",
    "critical_bg": "#FDE7E9",
    "keycap": "#FFFFFF",
    "keycap_border": "#D4D4DD",
    "keycap_bottom": "#BBBBC6",
    "chip": "#FFFFFF",
    "chip_border": "#D9D9E1",
    "track_off_border": "#8B8B99",
    "scrollbar": "#C4C4CE",
    "brand_a": "#5B3CF0",
    "brand_b": "#16B4E8",
}

_DARK = {
    "base": "#1C1C22",
    "layer": "#232329",
    "layer_border": "#2E2E35",
    "card": "#2B2B32",
    "card_border": "#35353D",
    "card_hover": "#303038",
    "control": "#35353D",
    "control_hover": "#3B3B44",
    "control_pressed": "#303038",
    "control_border": "#43434C",
    "control_bottom": "#393941",
    "nav_hover": "#2A2A31",
    "nav_selected": "#2F2F37",
    "divider": "#34343C",
    "text": "#F3F3F7",
    "text2": "#B8B8C4",
    "text3": "#8C8C99",
    "disabled": "#5C5C66",
    "success": "#6CCB5F",
    "success_bg": "#233526",
    "caution": "#F2B84B",
    "caution_bg": "#3A3120",
    "critical": "#FF99A4",
    "critical_bg": "#402529",
    "keycap": "#3A3A43",
    "keycap_border": "#4B4B55",
    "keycap_bottom": "#26262C",
    "chip": "#303038",
    "chip_border": "#41414A",
    "track_off_border": "#9C9CA8",
    "scrollbar": "#5A5A64",
    "brand_a": "#5B3CF0",
    "brand_b": "#16B4E8",
}


@dataclass(frozen=True)
class Palette:
    """Every colour the windows use, as #RRGGBB strings."""

    dark: bool
    base: str
    layer: str
    layer_border: str
    card: str
    card_border: str
    card_hover: str
    control: str
    control_hover: str
    control_pressed: str
    control_border: str
    control_bottom: str
    nav_hover: str
    nav_selected: str
    divider: str
    text: str
    text2: str
    text3: str
    disabled: str
    success: str
    success_bg: str
    caution: str
    caution_bg: str
    critical: str
    critical_bg: str
    keycap: str
    keycap_border: str
    keycap_bottom: str
    chip: str
    chip_border: str
    track_off_border: str
    scrollbar: str
    brand_a: str
    brand_b: str
    accent: str
    accent_hover: str
    accent_pressed: str
    on_accent: str
    accent_text: str
    accent_soft: str

    def color(self, name: str) -> QtGui.QColor:
        return QtGui.QColor(getattr(self, name))


def mix(first: str | QtGui.QColor, second: str | QtGui.QColor, amount: float) -> QtGui.QColor:
    """`amount` of the first colour over the rest of the second."""
    a = QtGui.QColor(first)
    b = QtGui.QColor(second)
    t = min(1.0, max(0.0, float(amount)))
    return QtGui.QColor.fromRgbF(
        a.redF() * t + b.redF() * (1 - t),
        a.greenF() * t + b.greenF() * (1 - t),
        a.blueF() * t + b.blueF() * (1 - t),
        a.alphaF() * t + b.alphaF() * (1 - t),
    )


def _hex(colour: QtGui.QColor) -> str:
    return colour.name(QtGui.QColor.NameFormat.HexRgb).upper()


def build_palette(dark: bool, shades: AccentShades | None = None) -> Palette:
    """The palette for a theme and an accent; None uses the brand indigo."""
    surfaces = dict(_DARK if dark else _LIGHT)
    accent_shades = shades or BRAND_SHADES
    if dark:
        accent = accent_shades.light2
        accent_text = accent_shades.light3
        on_accent = "#000000" if QtGui.QColor(accent).lightnessF() > 0.55 else "#FFFFFF"
        soft = mix(accent_shades.light1, surfaces["card"], 0.22)
    else:
        accent = accent_shades.dark1
        accent_text = accent_shades.dark2
        on_accent = "#FFFFFF" if QtGui.QColor(accent).lightnessF() < 0.62 else "#000000"
        soft = mix(accent_shades.base, surfaces["card"], 0.10)
    return Palette(
        dark=dark,
        accent=_hex(QtGui.QColor(accent)),
        accent_hover=_hex(mix(accent, surfaces["card"], 0.9)),
        accent_pressed=_hex(mix(accent, surfaces["card"], 0.8)),
        on_accent=on_accent,
        accent_text=_hex(QtGui.QColor(accent_text)),
        accent_soft=_hex(soft),
        **surfaces,
    )


def shades_from_palette_bytes(data: bytes) -> AccentShades | None:
    """The Explorer AccentPalette value: eight RGBA quads, lightest first; the eighth is unused."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 28:
        return None
    colours = [f"#{data[i]:02X}{data[i + 1]:02X}{data[i + 2]:02X}" for i in range(0, 28, 4)]
    return AccentShades(*colours)


def shades_from_colour(colour: QtGui.QColor) -> AccentShades:
    """Seven shades around one colour, for when only the DWM accent colour is readable."""
    base = QtGui.QColor(colour)
    hue, saturation, lightness, _alpha = base.getHslF()
    hue = max(0.0, hue)

    def shade(delta: float) -> str:
        value = min(0.95, max(0.08, lightness + delta))
        return _hex(QtGui.QColor.fromHslF(hue, saturation, value))

    return AccentShades(shade(0.36), shade(0.24), shade(0.12), _hex(base), shade(-0.06), shade(-0.14), shade(-0.22))


def _read_registry(path: str, name: str) -> object | None:
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            value, _kind = winreg.QueryValueEx(key, name)
            return value
    except OSError:
        return None


def system_accent() -> AccentShades | None:
    """The Windows accent shades, or None when they cannot be read."""
    data = _read_registry(r"Software\Microsoft\Windows\CurrentVersion\Explorer\Accent", "AccentPalette")
    shades = shades_from_palette_bytes(data) if data is not None else None
    if shades is not None:
        return shades
    abgr = _read_registry(r"Software\Microsoft\Windows\DWM", "AccentColor")
    if isinstance(abgr, int):
        red, green, blue = abgr & 0xFF, (abgr >> 8) & 0xFF, (abgr >> 16) & 0xFF
        return shades_from_colour(QtGui.QColor(red, green, blue))
    return None


def apps_use_dark_theme() -> bool:
    """Whether Windows apps use the dark theme (AppsUseLightTheme is 0)."""
    value = _read_registry(r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize", "AppsUseLightTheme")
    if isinstance(value, int):
        return value == 0
    app = QtGui.QGuiApplication.instance()
    if app is not None:
        try:
            return app.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
        except Exception:
            log.debug("the Qt colour scheme is unavailable", exc_info=True)
            return False
    return False


# Fonts ------------------------------------------------------------------------------------------


def font(role: str = "body") -> QtGui.QFont:
    families, pixels, weight = FONT_ROLES[role]
    result = QtGui.QFont()
    result.setFamilies(list(families))
    result.setPixelSize(pixels)
    result.setWeight(QtGui.QFont.Weight(weight))
    result.setHintingPreference(QtGui.QFont.HintingPreference.PreferNoHinting)
    return result


def icon_font(pixels: int = 16) -> QtGui.QFont:
    result = QtGui.QFont()
    result.setFamilies(list(ICON_FAMILIES))
    result.setPixelSize(pixels)
    return result


def has_icon_font() -> bool:
    families = set(QtGui.QFontDatabase.families())
    return any(name in families for name in ICON_FAMILIES)


def glyph_pixmap(glyph: str, pixels: int, colour: QtGui.QColor | str, device_ratio: float = 1.0) -> QtGui.QPixmap:
    size = max(1, round(pixels * device_ratio))
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    if has_icon_font():
        painter = QtGui.QPainter(pixmap)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
            painter.setFont(icon_font(size))
            painter.setPen(QtGui.QColor(colour))
            painter.drawText(QtCore.QRect(0, 0, size, size), int(QtCore.Qt.AlignmentFlag.AlignCenter), glyph)
        finally:
            painter.end()
    pixmap.setDevicePixelRatio(device_ratio)
    return pixmap


def _with_gap(pixmap: QtGui.QPixmap, gap: int) -> QtGui.QPixmap:
    if gap <= 0:
        return pixmap
    ratio = pixmap.devicePixelRatio()
    wide = QtGui.QPixmap(pixmap.width() + round(gap * ratio), pixmap.height())
    wide.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(wide)
    try:
        painter.drawPixmap(0, 0, pixmap)
    finally:
        painter.end()
    wide.setDevicePixelRatio(ratio)
    return wide


def glyph_icon(
    glyph: str,
    colour: QtGui.QColor | str | None = None,
    disabled: QtGui.QColor | str | None = None,
    *,
    gap: int = 0,
) -> QtGui.QIcon:
    """A glyph as an icon; `gap` adds transparent space on the right, before a button's text."""
    current = palette()
    normal = QtGui.QColor(colour) if colour is not None else current.color("text")
    muted = QtGui.QColor(disabled) if disabled is not None else current.color("disabled")
    icon = QtGui.QIcon()
    for pixels in (14, 16, 20, 24, 32):
        scaled_gap = round(gap * pixels / 14)
        icon.addPixmap(_with_gap(glyph_pixmap(glyph, pixels, normal), scaled_gap), QtGui.QIcon.Mode.Normal)
        icon.addPixmap(_with_gap(glyph_pixmap(glyph, pixels, normal), scaled_gap), QtGui.QIcon.Mode.Active)
        icon.addPixmap(_with_gap(glyph_pixmap(glyph, pixels, muted), scaled_gap), QtGui.QIcon.Mode.Disabled)
    return icon


# The stylesheet -------------------------------------------------------------------------------------

_QSS = Template(
    """
QWidget { color: $text; }
QDialog { background: $base; }
QFrame#spellsLayer { background: $layer; border: none; border-left: 1px solid $layer_border;
    border-top: 1px solid $layer_border; border-top-left-radius: 8px; }
QFrame#spellsRail { background: transparent; border: none; }
QFrame#spellsFooter { background: $base; border: none; border-top: 1px solid $divider; }
QFrame#spellsWelcomeBody { border: none; background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
    stop: 0 $accent_soft, stop: 0.32 $layer, stop: 1 $layer); }
QWidget#scrollBody { background: transparent; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget#qt_scrollarea_viewport { background: transparent; }

QFrame[role="card"] { background: $card; border: 1px solid $card_border; border-radius: 10px; }
QFrame[role="well"] { background: $base; border: 1px solid $divider; border-radius: 8px; }
QFrame[role="art"] { background: $base; border: none; border-bottom: 1px solid $divider;
    border-top-left-radius: 9px; border-top-right-radius: 9px; }
QFrame[role="divider"] { background: $divider; border: none; min-height: 1px; max-height: 1px; }
QFrame[role="vdivider"] { background: $divider; border: none; min-width: 1px; max-width: 1px; }
QFrame[role="chip"] { background: $chip; border: 1px solid $chip_border; border-radius: 16px; }
QFrame[role="chip"][muted="true"] { background: transparent; border: 1px dashed $chip_border; }
QFrame[infobar="info"] { background: $accent_soft; border: none; border-radius: 8px; }
QFrame[infobar="caution"] { background: $caution_bg; border: none; border-radius: 8px; }
QFrame[infobar="critical"] { background: $critical_bg; border: none; border-radius: 8px; }
QFrame[infobar="success"] { background: $success_bg; border: none; border-radius: 8px; }
QFrame[infobar="caution"][flush="true"], QFrame[infobar="info"][flush="true"] { border-radius: 0px;
    border-bottom-left-radius: 9px; border-bottom-right-radius: 9px; }

QLabel { background: transparent; }
QLabel[tone="secondary"] { color: $text2; }
QLabel[tone="tertiary"] { color: $text3; }
QLabel[tone="accent"] { color: $accent_text; }
QLabel[tone="success"] { color: $success; }
QLabel[tone="caution"] { color: $caution; }
QLabel[tone="critical"] { color: $critical; }
QLabel[tone="disabled"] { color: $disabled; }
QLabel[badge="ok"] { background: $success_bg; color: $success; border-radius: 12px; padding: 0px 10px; }
QLabel[badge="caution"] { background: $caution_bg; color: $caution; border-radius: 12px; padding: 0px 10px; }
QLabel[badge="critical"] { background: $critical_bg; color: $critical; border-radius: 12px; padding: 0px 10px; }
QLabel[badge="neutral"] { background: $accent_soft; color: $accent_text; border-radius: 12px; padding: 0px 10px; }
QLabel[badge="muted"] { background: $nav_selected; color: $text2; border-radius: 12px; padding: 0px 10px; }
QLabel[role="code"] { background: $accent_soft; color: $accent_text; border-radius: 12px; padding: 0px 7px; }
QLabel[role="tag"] { background: $nav_selected; color: $text2; border-radius: 10px; padding: 0px 8px; }
QLabel[role="glyphbadge"] { background: $accent_soft; color: $accent_text; border-radius: 8px; }
QLabel[role="stepnumber"] { background: $accent; color: $on_accent; border-radius: 11px; }

QPushButton { background: $control; color: $text; border: 1px solid $control_border;
    border-bottom-color: $control_bottom; border-radius: 6px; padding: 5px 14px; min-height: 20px; }
QPushButton:hover { background: $control_hover; }
QPushButton:pressed { background: $control_pressed; color: $text2; border-bottom-color: $control_border; }
QPushButton:disabled { background: $control; color: $disabled; border-color: $control_border; }
QPushButton:checked { background: $accent_soft; color: $accent_text; border-color: $accent_soft; }
QPushButton[kind="primary"] { background: $accent; color: $on_accent; border: 1px solid $accent; }
QPushButton[kind="primary"]:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton[kind="primary"]:pressed { background: $accent_pressed; border-color: $accent_pressed; color: $on_accent; }
QPushButton[kind="primary"]:disabled { background: $nav_selected; border-color: $nav_selected; color: $disabled; }
QPushButton[kind="subtle"] { background: transparent; border: 1px solid transparent; color: $accent_text; padding: 5px 10px; }
QPushButton[kind="subtle"]:hover { background: $nav_hover; }
QPushButton[kind="subtle"]:pressed { background: $nav_selected; }
QPushButton[kind="subtle"]:disabled { color: $disabled; background: transparent; }
QPushButton[kind="danger"] { color: $critical; }
QPushButton[kind="chipadd"] { background: $chip; border: 1px solid $chip_border; border-radius: 16px;
    padding: 5px 14px 5px 10px; }
QPushButton[kind="chipadd"]:hover { background: $nav_hover; }
QToolButton { background: transparent; border: none; border-radius: 6px; color: $text2; padding: 4px; }
QToolButton:hover { background: $nav_hover; color: $text; }
QToolButton:pressed { background: $nav_selected; }
QToolButton[role="chipclose"] { border-radius: 10px; padding: 0px; }

QLineEdit, QPlainTextEdit, QSpinBox, QComboBox { background: $control; color: $text;
    border: 1px solid $control_border; border-bottom-color: $control_bottom; border-radius: 6px;
    selection-background-color: $accent; selection-color: $on_accent; }
QLineEdit { padding: 5px 10px; min-height: 20px; }
QSpinBox { padding: 5px 8px 5px 10px; min-height: 20px; }
QComboBox { padding: 5px 10px 5px 12px; min-height: 20px; }
QPlainTextEdit { padding: 4px 4px; }
QLineEdit:hover, QSpinBox:hover, QComboBox:hover, QPlainTextEdit:hover { background: $control_hover; }
QLineEdit:focus, QSpinBox:focus, QPlainTextEdit:focus { background: $card; border-bottom: 2px solid $accent; }
QLineEdit:focus, QSpinBox:focus { padding-bottom: 4px; }
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled, QPlainTextEdit:disabled { color: $disabled; }
QPlainTextEdit[role="embedded"], QLineEdit[role="embedded"] { background: transparent; border: none; padding: 0px; }
QPlainTextEdit[role="embedded"]:focus { border: none; }
QPlainTextEdit[role="mono"] { background: $base; border: 1px solid $divider; border-radius: 8px; padding: 8px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 26px; border: none; }
QComboBox::down-arrow { width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: $card; color: $text; border: 1px solid $card_border; padding: 4px;
    outline: none; selection-background-color: $nav_selected; selection-color: $text; }
QComboBox QAbstractItemView::item { min-height: 30px; padding: 0px 8px; border-radius: 4px; }
QSpinBox::up-button, QSpinBox::down-button { subcontrol-origin: border; width: 22px; border: none; background: transparent; }
QSpinBox::up-button { subcontrol-position: top right; margin: 3px 3px 0px 0px; }
QSpinBox::down-button { subcontrol-position: bottom right; margin: 0px 3px 3px 0px; }
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: $nav_hover; border-radius: 4px; }
QSpinBox::up-arrow, QSpinBox::down-arrow { width: 8px; height: 8px; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 0px; }
QScrollBar::handle:vertical { background: $scrollbar; min-height: 40px; border-radius: 3px; margin: 2px 3px 2px 3px; }
QScrollBar::handle:vertical:hover { background: $text3; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0px; }
QScrollBar::handle:horizontal { background: $scrollbar; min-width: 40px; border-radius: 3px; margin: 3px 2px 3px 2px; }
QScrollBar::handle:horizontal:hover { background: $text3; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QTableView, QListView { background: $card; color: $text; border: 1px solid $card_border; border-radius: 10px;
    outline: none; gridline-color: transparent; selection-background-color: $accent_soft; selection-color: $text; }
QTableView[role="embedded"], QListView[role="embedded"] { border: none; border-radius: 0px; background: transparent; }
QTableView::item { padding: 0px 12px; border: none; border-bottom: 1px solid $divider; }
QTableView::item:hover { background: $card_hover; }
QTableView::item:selected { background: $accent_soft; color: $text; }
QListView::item { padding: 8px 10px; border-radius: 6px; }
QListView::item:hover { background: $nav_hover; }
QListView::item:selected { background: $nav_selected; color: $text; }
QHeaderView { background: transparent; border: none; }
QHeaderView::section { background: transparent; color: $text2; border: none; border-bottom: 1px solid $divider;
    padding: 8px 12px 8px 15px; }
QTableCornerButton::section { background: transparent; border: none; }

QMenu { background: $card; color: $text; border: 1px solid $card_border; padding: 4px; }
QMenu::item { padding: 7px 32px 7px 10px; border-radius: 4px; }
QMenu::item:selected { background: $nav_hover; color: $text; }
QMenu::item:disabled { color: $disabled; }
QMenu::separator { height: 1px; background: $divider; margin: 4px 6px; }
QMenu::icon { padding-left: 10px; }
QMenu::indicator { width: 16px; height: 16px; padding-left: 6px; }
QToolTip { background: $card; color: $text; border: 1px solid $card_border; padding: 6px 8px; }
QMessageBox, QInputDialog, QFileDialog { background: $base; }
"""
)


def stylesheet(current: Palette) -> str:
    return _QSS.substitute(asdict(current))


def qt_palette(current: Palette) -> QtGui.QPalette:
    role = QtGui.QPalette.ColorRole
    group = QtGui.QPalette.ColorGroup
    result = QtGui.QPalette()
    colours = {
        role.Window: current.base,
        role.WindowText: current.text,
        role.Base: current.control,
        role.AlternateBase: current.card_hover,
        role.Text: current.text,
        role.Button: current.control,
        role.ButtonText: current.text,
        role.BrightText: current.on_accent,
        role.Highlight: current.accent,
        role.HighlightedText: current.on_accent,
        role.ToolTipBase: current.card,
        role.ToolTipText: current.text,
        role.PlaceholderText: current.text3,
        role.Link: current.accent_text,
        role.Light: current.card,
        role.Midlight: current.card_hover,
        role.Mid: current.divider,
        role.Dark: current.control_bottom,
        role.Shadow: current.control_bottom,
        role.Accent: current.accent,
    }
    for name, value in colours.items():
        result.setColor(name, QtGui.QColor(value))
    for name in (role.WindowText, role.Text, role.ButtonText):
        result.setColor(group.Disabled, name, QtGui.QColor(current.disabled))
    return result


class SpellsStyle(QtWidgets.QProxyStyle):
    """Fusion with Windows 11 chevrons for combo and spin arrows and no dotted focus frames."""

    _ARROWS: ClassVar[dict[QtWidgets.QStyle.PrimitiveElement, str]] = {
        QtWidgets.QStyle.PrimitiveElement.PE_IndicatorArrowDown: "down",
        QtWidgets.QStyle.PrimitiveElement.PE_IndicatorArrowUp: "up",
        QtWidgets.QStyle.PrimitiveElement.PE_IndicatorSpinDown: "down",
        QtWidgets.QStyle.PrimitiveElement.PE_IndicatorSpinUp: "up",
        QtWidgets.QStyle.PrimitiveElement.PE_IndicatorArrowRight: "right",
    }

    def drawPrimitive(self, element, option, painter, widget=None) -> None:
        direction = self._ARROWS.get(element)
        if direction is not None:
            enabled = bool(option.state & QtWidgets.QStyle.StateFlag.State_Enabled)
            colour = palette().color("text2" if enabled else "disabled")
            draw_chevron(painter, QtCore.QRectF(option.rect), colour, direction)
            return
        if element == QtWidgets.QStyle.PrimitiveElement.PE_FrameFocusRect:
            return
        super().drawPrimitive(element, option, painter, widget)


def draw_chevron(
    painter: QtGui.QPainter, rect: QtCore.QRectF, colour: QtGui.QColor, direction: str = "down", size: float = 10.0
) -> None:
    size = min(rect.width(), rect.height(), size)
    half = size / 2.0
    centre = rect.center()
    painter.save()
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(colour, 1.3, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        path = QtGui.QPainterPath()
        if direction == "down":
            path.moveTo(centre.x() - half, centre.y() - half / 2)
            path.lineTo(centre.x(), centre.y() + half / 2)
            path.lineTo(centre.x() + half, centre.y() - half / 2)
        elif direction == "up":
            path.moveTo(centre.x() - half, centre.y() + half / 2)
            path.lineTo(centre.x(), centre.y() - half / 2)
            path.lineTo(centre.x() + half, centre.y() + half / 2)
        else:
            path.moveTo(centre.x() - half / 2, centre.y() - half)
            path.lineTo(centre.x() + half / 2, centre.y())
            path.lineTo(centre.x() - half / 2, centre.y() + half)
        painter.drawPath(path)
    finally:
        painter.restore()


# Installation ------------------------------------------------------------------------------------


class ThemeNotifier(QtCore.QObject):
    changed = QtCore.Signal(object)


_state: dict[str, object] = {"palette": None, "notifier": None, "style": None, "watching": False}


def palette() -> Palette:
    """The installed palette, or the light brand palette before install()."""
    current = _state["palette"]
    if isinstance(current, Palette):
        return current
    return build_palette(False, None)


def notifier() -> ThemeNotifier:
    current = _state["notifier"]
    if not isinstance(current, ThemeNotifier):
        current = ThemeNotifier()
        _state["notifier"] = current
    return current


def install(
    app: QtWidgets.QApplication,
    *,
    dark: bool | None = None,
    accent: AccentShades | None = None,
    follow_system_accent: bool = False,
    watch: bool = True,
) -> Palette:
    """Apply the palette, the proxy style and the stylesheet to the application."""
    is_dark = apps_use_dark_theme() if dark is None else bool(dark)
    shades = accent if accent is not None else (system_accent() if follow_system_accent else None)
    current = build_palette(is_dark, shades)
    _state["palette"] = current
    if not isinstance(_state["style"], SpellsStyle):
        proxy = SpellsStyle("Fusion")
        _state["style"] = proxy
        app.setStyle(proxy)
    qpalette = qt_palette(current)
    if app.palette() != qpalette:
        app.setPalette(qpalette)
    body = font("body")
    if app.font() != body:
        app.setFont(body)
    sheet = stylesheet(current)
    if app.styleSheet() != sheet:
        app.setStyleSheet(sheet)
    if watch and dark is None and not _state["watching"]:
        _state["watching"] = True
        try:
            app.styleHints().colorSchemeChanged.connect(lambda _scheme: _follow_system(app))
        except Exception:
            log.debug("colour scheme changes are not observable", exc_info=True)
    notifier().changed.emit(current)
    for widget in app.topLevelWidgets():
        widget.update()
        if widget.isVisible():
            apply_window_chrome(widget)
    return current


def _follow_system(app: QtWidgets.QApplication) -> None:
    try:
        install(app, watch=False)
    except Exception:
        log.exception("could not follow the Windows theme")


# Window chrome -------------------------------------------------------------------------------------

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36
DWMWCP_ROUND = 2
DWMWCP_ROUNDSMALL = 3


def colorref(colour: str | QtGui.QColor) -> int:
    value = QtGui.QColor(colour)
    return value.red() | (value.green() << 8) | (value.blue() << 16)


def _native_windows() -> bool:
    return sys.platform == "win32" and QtGui.QGuiApplication.platformName() == "windows"


def _set_dwm_attribute(hwnd: int, attribute: int, value: int) -> bool:
    import ctypes
    from ctypes import wintypes

    data = ctypes.c_int(int(value))
    result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
        wintypes.HWND(hwnd), ctypes.c_uint(attribute), ctypes.byref(data), ctypes.sizeof(data)
    )
    return result == 0


def apply_window_chrome(widget: QtWidgets.QWidget) -> bool:
    """Dark title bar and a caption in the window's base colour (Windows 11); False when skipped."""
    if not _native_windows() or not widget.isWindow():
        return False
    flags = widget.windowFlags()
    if flags & QtCore.Qt.WindowType.FramelessWindowHint:
        return False
    current = palette()
    try:
        hwnd = int(widget.winId())
        _set_dwm_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if current.dark else 0)
        _set_dwm_attribute(hwnd, DWMWA_CAPTION_COLOR, colorref(current.base))
        _set_dwm_attribute(hwnd, DWMWA_TEXT_COLOR, colorref(current.text))
    except Exception:
        log.debug("could not colour the title bar", exc_info=True)
        return False
    return True


def round_popup(widget: QtWidgets.QWidget) -> bool:
    """Ask DWM for small rounded corners and its shadow on a popup such as a menu."""
    if not _native_windows():
        return False
    try:
        return _set_dwm_attribute(int(widget.winId()), DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUNDSMALL)
    except Exception:
        log.debug("could not round the popup corners", exc_info=True)
        return False


__all__ = [
    "BRAND_SHADES",
    "AccentShades",
    "Glyph",
    "Palette",
    "SpellsStyle",
    "apply_window_chrome",
    "apps_use_dark_theme",
    "build_palette",
    "font",
    "glyph_icon",
    "glyph_pixmap",
    "install",
    "mix",
    "notifier",
    "palette",
    "round_popup",
    "shades_from_colour",
    "shades_from_palette_bytes",
    "stylesheet",
    "system_accent",
]
