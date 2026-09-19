"""Colours, fonts, key and language names shared by the ui widgets.

The pill colours and type are the numbers block of the approved pill mockup;
everything else here is small, pure and Qt-free where it can be, so the widgets stay thin.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from functools import lru_cache

from PySide6 import QtGui

from spells import vk
from spells.datafiles import data_path
from spells.models import Chord

log = logging.getLogger(__name__)

APP_NAME = "Spells"

# Pill palette (mockup: Colour).
PILL_FILL = QtGui.QColor(0, 0, 0, 217)
PILL_EDGE = QtGui.QColor(255, 255, 255, 26)
PILL_SHADOW = QtGui.QColor(0, 0, 0, 102)
PILL_BAR = QtGui.QColor(255, 255, 255, 255)
PILL_TEXT = QtGui.QColor(255, 255, 255, 235)
PILL_ERROR = QtGui.QColor(0xFF, 0xB4, 0x54, 255)
PILL_SPINNER = QtGui.QColor(255, 255, 255, 230)
PILL_SPINNER_TRACK = QtGui.QColor(255, 255, 255, 46)
PILL_LOCK = QtGui.QColor(255, 255, 255, 242)

# Tray badge colours: the shapes differ per state, colour only helps.
TRAY_RED = QtGui.QColor(0xE5, 0x48, 0x4D)
TRAY_AMBER = QtGui.QColor(0xFF, 0xB4, 0x54)
TRAY_DARK = QtGui.QColor(0x15, 0x21, 0x2A)
TRAY_LIGHT = QtGui.QColor(0xFF, 0xFF, 0xFF)

PILL_FONT_FAMILIES = [
    "Segoe UI Variable Small",
    "Segoe UI Variable Text",
    "Segoe UI Variable",
    "Segoe UI",
]


def pill_font(pixel_size: float, letter_spacing: float) -> QtGui.QFont:
    """The pill's type: Segoe UI Variable Small, 12 px, weight 400, 0.12 px tracking (scaled)."""
    font = QtGui.QFont()
    font.setFamilies(PILL_FONT_FAMILIES)
    font.setPixelSize(max(1, round(pixel_size)))
    font.setWeight(QtGui.QFont.Weight.Normal)
    font.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, letter_spacing)
    return font


# Languages ------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _code_to_name() -> dict[str, str]:
    table = json.loads(data_path("whisper_languages.json").read_text(encoding="utf-8"))
    return {code: name.capitalize() for name, code in table.items()}


def language_name(code: str) -> str:
    """"German" for "de"; an unknown code is shown upper-cased."""
    return _code_to_name().get(code, code.upper())


def language_choices() -> list[tuple[str, str]]:
    """Every whisper language as (code, name), sorted by name."""
    return sorted(_code_to_name().items(), key=lambda item: item[1])


# Keys and chords --------------------------------------------------------------------------------

_MODIFIER_ORDER = (vk.VK_CONTROL, vk.VK_MENU, vk.VK_SHIFT, vk.VK_LWIN, vk.VK_RWIN)

KEY_NAMES: dict[int, str] = {
    vk.VK_SHIFT: "Shift",
    vk.VK_CONTROL: "Ctrl",
    vk.VK_MENU: "Alt",
    vk.VK_LWIN: "Win",
    vk.VK_RWIN: "Right Win",
    vk.VK_LSHIFT: "Left Shift",
    vk.VK_RSHIFT: "Right Shift",
    vk.VK_LCONTROL: "Left Ctrl",
    vk.VK_RCONTROL: "Right Ctrl",
    vk.VK_LMENU: "Left Alt",
    vk.VK_RMENU: "Right Alt",
    0x08: "Backspace",
    0x09: "Tab",
    0x0D: "Enter",
    0x13: "Pause",
    0x14: "Caps Lock",
    0x1B: "Esc",
    0x20: "Space",
    0x21: "Page Up",
    0x22: "Page Down",
    0x23: "End",
    0x24: "Home",
    0x25: "Left",
    0x26: "Up",
    0x27: "Right",
    0x28: "Down",
    0x2C: "Print Screen",
    0x2D: "Insert",
    0x2E: "Delete",
    0x5D: "Menu",
    0x6A: "Num *",
    0x6B: "Num +",
    0x6D: "Num -",
    0x6E: "Num .",
    0x6F: "Num /",
    0x90: "Num Lock",
    0x91: "Scroll Lock",
    0xBA: ";",
    0xBB: "=",
    0xBC: ",",
    0xBD: "-",
    0xBE: ".",
    0xBF: "/",
    0xC0: "`",
    0xDB: "[",
    0xDC: "\\",
    0xDD: "]",
    0xDE: "'",
}
KEY_NAMES.update({code: chr(code) for code in range(0x30, 0x3A)})
KEY_NAMES.update({code: chr(code) for code in range(0x41, 0x5B)})
KEY_NAMES.update({0x60 + n: f"Num {n}" for n in range(10)})
KEY_NAMES.update({0x70 + n: f"F{n + 1}" for n in range(24)})


def key_name(code: int) -> str:
    return KEY_NAMES.get(code, f"VK 0x{code:02X}")


def chord_keys(chord: Chord | Iterable[int]) -> list[str]:
    """The key names of a chord, modifiers first in a fixed order, then the other keys as configured."""
    keys = tuple(chord.keys) if isinstance(chord, Chord) else tuple(chord)
    generic = [vk.generic_modifier(key) for key in keys]
    modifiers = [m for m in _MODIFIER_ORDER if m in generic]
    others = [key for key in generic if key not in _MODIFIER_ORDER]
    return [key_name(key) for key in modifiers + others]


def chord_label(chord: Chord | Iterable[int]) -> str:
    """"Ctrl+Win+D": modifiers first in a fixed order, then the other keys as configured."""
    return "+".join(chord_keys(chord))


# Windows appearance --------------------------------------------------------------------------------


def taskbar_is_light() -> bool:
    """Whether the Windows taskbar uses the light theme (SystemUsesLightTheme); dark when unknown."""
    try:
        import winreg
    except ImportError:
        return False
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            value, _kind = winreg.QueryValueEx(key, "SystemUsesLightTheme")
            return bool(value)
    except OSError:
        return False


def reduced_motion() -> bool:
    """True when Windows animations are switched off (the pill's reduced motion path)."""
    try:
        from spells.win32.window import animations_enabled
    except Exception:
        log.debug("win32 window helpers unavailable", exc_info=True)
        return False
    try:
        return not animations_enabled()
    except Exception:
        log.debug("could not query the animation setting", exc_info=True)
        return False


__all__ = [
    "APP_NAME",
    "KEY_NAMES",
    "PILL_BAR",
    "PILL_EDGE",
    "PILL_ERROR",
    "PILL_FILL",
    "PILL_FONT_FAMILIES",
    "PILL_LOCK",
    "PILL_SHADOW",
    "PILL_SPINNER",
    "PILL_SPINNER_TRACK",
    "PILL_TEXT",
    "TRAY_AMBER",
    "TRAY_DARK",
    "TRAY_LIGHT",
    "TRAY_RED",
    "chord_keys",
    "chord_label",
    "key_name",
    "language_choices",
    "language_name",
    "pill_font",
    "reduced_motion",
    "taskbar_is_light",
]
