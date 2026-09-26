from __future__ import annotations

from typing import Any

from spells import vk
from spells.platform.keys_linux import QT_KEY_TO_VK
from spells.platform.keys_windows import WindowsKeys

_SYMBOLS = {
    vk.VK_LWIN: "⌘",
    vk.VK_RWIN: "Right ⌘",
    vk.VK_MENU: "⌥",
    vk.VK_LMENU: "Left ⌥",
    vk.VK_RMENU: "Right ⌥",
    vk.VK_CONTROL: "⌃",
    vk.VK_LCONTROL: "Left ⌃",
    vk.VK_RCONTROL: "Right ⌃",
    vk.VK_SHIFT: "⇧",
    vk.VK_LSHIFT: "Left ⇧",
    vk.VK_RSHIFT: "Right ⇧",
}


class MacKeys:
    def display_name(self, vk: int) -> str:
        return _SYMBOLS.get(vk) or WindowsKeys().display_name(vk)

    def vk_from_event(self, event: Any) -> int | None:
        from PySide6 import QtCore

        key = QtCore.Qt.Key
        code = int(event.key())
        if code == int(key.Key_Control):
            code = int(key.Key_Meta)
        elif code == int(key.Key_Meta):
            code = int(key.Key_Control)
        return QT_KEY_TO_VK.get(code)
