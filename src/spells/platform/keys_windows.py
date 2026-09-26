from __future__ import annotations

from typing import Any

from spells import vk

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


class WindowsKeys:
    def display_name(self, vk: int) -> str:
        return KEY_NAMES.get(vk, f"VK 0x{vk:02X}")

    def vk_from_event(self, event: Any) -> int | None:
        code = int(event.nativeVirtualKey())
        return code or None
