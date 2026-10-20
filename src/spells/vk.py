"""Shared virtual-key (VK) constants for hotkey chords.

Values come from the Win32 VK_* table (winuser.h) and are the key vocabulary on
every system, but this module is stdlib only and has no Win32 dependency, so any
module can import it without pulling in ctypes. spells.config uses it for
chord-conflict validation, spells.hotkey for the chord state machine, the key
name tables in spells.platform to name keys and read key events, and
spells.win32.input for modifier release.
"""

from __future__ import annotations

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5

GENERIC_OF: dict[int, int] = {
    VK_LSHIFT: VK_SHIFT,
    VK_RSHIFT: VK_SHIFT,
    VK_LCONTROL: VK_CONTROL,
    VK_RCONTROL: VK_CONTROL,
    VK_LMENU: VK_MENU,
    VK_RMENU: VK_MENU,
}

SPECIFIC_OF: dict[int, tuple[int, int]] = {
    VK_SHIFT: (VK_LSHIFT, VK_RSHIFT),
    VK_CONTROL: (VK_LCONTROL, VK_RCONTROL),
    VK_MENU: (VK_LMENU, VK_RMENU),
}


def generic_modifier(vk: int) -> int:
    """vk folded to its generic modifier code, or vk unchanged if it has none."""
    return GENERIC_OF.get(vk, vk)
