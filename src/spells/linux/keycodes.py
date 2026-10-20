"""Linux input key codes (linux/input-event-codes.h) and their Windows VK equivalents.

The chord state machine speaks VK codes on every system. Keyboard events read from evdev are
turned into the specific VK (Left Ctrl, not Ctrl) the Windows hook reports, and the virtual
keyboard turns a VK back into the key code it sends.
"""

from __future__ import annotations

from spells import vk

KEY_ESC = 1
KEY_BACKSPACE = 14
KEY_LEFTCTRL = 29
KEY_LEFTSHIFT = 42
KEY_A = 30
KEY_C = 46
KEY_V = 47
KEY_Z = 44
KEY_SPACE = 57
KEY_INSERT = 110

UNMAPPED_BASE = 0x10000

_LETTER_ROWS = ((16, "QWERTYUIOP"), (30, "ASDFGHJKL"), (44, "ZXCVBNM"))

CODE_TO_VK: dict[int, int] = {
    1: 0x1B,
    11: 0x30,
    12: 0xBD,
    13: 0xBB,
    14: 0x08,
    15: 0x09,
    26: 0xDB,
    27: 0xDD,
    28: 0x0D,
    29: vk.VK_LCONTROL,
    39: 0xBA,
    40: 0xDE,
    41: 0xC0,
    42: vk.VK_LSHIFT,
    43: 0xDC,
    51: 0xBC,
    52: 0xBE,
    53: 0xBF,
    54: vk.VK_RSHIFT,
    55: 0x6A,
    56: vk.VK_LMENU,
    57: 0x20,
    58: 0x14,
    69: 0x90,
    70: 0x91,
    71: 0x67,
    72: 0x68,
    73: 0x69,
    74: 0x6D,
    75: 0x64,
    76: 0x65,
    77: 0x66,
    78: 0x6B,
    79: 0x61,
    80: 0x62,
    81: 0x63,
    82: 0x60,
    83: 0x6E,
    86: 0xE2,
    87: 0x7A,
    88: 0x7B,
    96: 0x0D,
    97: vk.VK_RCONTROL,
    98: 0x6F,
    99: 0x2C,
    100: vk.VK_RMENU,
    102: 0x24,
    103: 0x26,
    104: 0x21,
    105: 0x25,
    106: 0x27,
    107: 0x23,
    108: 0x28,
    109: 0x22,
    110: 0x2D,
    111: 0x2E,
    113: 0xAD,
    114: 0xAE,
    115: 0xAF,
    119: 0x13,
    125: vk.VK_LWIN,
    126: vk.VK_RWIN,
    127: 0x5D,
    163: 0xB0,
    164: 0xB3,
    165: 0xB1,
    166: 0xB2,
}
CODE_TO_VK.update({code: 0x31 + index for index, code in enumerate(range(2, 11))})
for _start, _letters in _LETTER_ROWS:
    CODE_TO_VK.update({_start + index: ord(letter) for index, letter in enumerate(_letters)})
CODE_TO_VK.update({59 + index: 0x70 + index for index in range(10)})
CODE_TO_VK.update({183 + index: 0x7C + index for index in range(12)})

VK_TO_CODE: dict[int, int] = {value: code for code, value in CODE_TO_VK.items()}
VK_TO_CODE.update(
    {
        vk.VK_CONTROL: KEY_LEFTCTRL,
        vk.VK_SHIFT: KEY_LEFTSHIFT,
        vk.VK_MENU: 56,
        0x0D: 28,
    }
)


def vk_for_code(code: int) -> int:
    """The VK for an evdev key code; keys without one get a code that no chord can contain."""
    return CODE_TO_VK.get(code, UNMAPPED_BASE + code)


def codes_for_vk(key: int) -> tuple[int, ...]:
    """Every evdev key code that reports as this VK, generic modifiers included."""
    if key in vk.SPECIFIC_OF:
        return tuple(VK_TO_CODE[item] for item in vk.SPECIFIC_OF[key])
    if key >= UNMAPPED_BASE:
        return (key - UNMAPPED_BASE,)
    code = VK_TO_CODE.get(key)
    return () if code is None else (code,)
