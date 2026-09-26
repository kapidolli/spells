from __future__ import annotations

from typing import Any

from PySide6 import QtCore

from spells import vk
from spells.platform.keys_windows import WindowsKeys

Key = QtCore.Qt.Key

QT_KEY_TO_VK: dict[int, int] = {
    int(Key.Key_Control): vk.VK_CONTROL,
    int(Key.Key_Alt): vk.VK_MENU,
    int(Key.Key_Shift): vk.VK_SHIFT,
    int(Key.Key_Meta): vk.VK_LWIN,
    int(Key.Key_Super_L): vk.VK_LWIN,
    int(Key.Key_Super_R): vk.VK_RWIN,
    int(Key.Key_Space): 0x20,
    int(Key.Key_Escape): 0x1B,
    int(Key.Key_Tab): 0x09,
    int(Key.Key_Return): 0x0D,
    int(Key.Key_Backspace): 0x08,
}
QT_KEY_TO_VK.update(
    {int(getattr(Key, f"Key_{chr(code)}")): code for code in range(0x41, 0x5B)}
)
QT_KEY_TO_VK.update(
    {int(getattr(Key, f"Key_{digit}")): 0x30 + digit for digit in range(10)}
)
QT_KEY_TO_VK.update(
    {int(getattr(Key, f"Key_F{n}")): 0x70 + n - 1 for n in range(1, 25)}
)

_WIN_NAMES = {vk.VK_LWIN: "Super", vk.VK_RWIN: "Right Super"}


class LinuxKeys:
    def display_name(self, vk: int) -> str:
        return _WIN_NAMES.get(vk) or WindowsKeys().display_name(vk)

    def vk_from_event(self, event: Any) -> int | None:
        return QT_KEY_TO_VK.get(int(event.key()))
