import pytest
from PySide6 import QtCore

from spells import vk
from spells.platform.keys_linux import LinuxKeys
from spells.platform.keys_macos import MacKeys
from spells.platform.keys_windows import KEY_NAMES, WindowsKeys

Key = QtCore.Qt.Key


def _event(key, native_vk=0):
    class Event:
        def key(self):
            return int(key)

        def nativeVirtualKey(self):
            return native_vk

    return Event()


def test_windows_names_are_the_existing_table():
    keys = WindowsKeys()
    assert keys.display_name(vk.VK_LWIN) == "Win"
    assert keys.display_name(vk.VK_CONTROL) == "Ctrl"
    assert keys.display_name(0x41) == "A"
    assert keys.display_name(0xE9) == "VK 0xE9"
    assert KEY_NAMES[vk.VK_RWIN] == "Right Win"


def test_windows_reads_the_native_virtual_key():
    assert WindowsKeys().vk_from_event(_event(Key.Key_A, 0x41)) == 0x41
    assert WindowsKeys().vk_from_event(_event(Key.Key_A, 0)) is None


def test_mac_names_use_the_symbols():
    keys = MacKeys()
    assert keys.display_name(vk.VK_LWIN) == "⌘"
    assert keys.display_name(vk.VK_RWIN) == "Right ⌘"
    assert keys.display_name(vk.VK_MENU) == "⌥"
    assert keys.display_name(vk.VK_CONTROL) == "⌃"
    assert keys.display_name(vk.VK_SHIFT) == "⇧"
    assert keys.display_name(0x41) == "A"


def test_mac_swaps_command_and_control():
    keys = MacKeys()
    assert keys.vk_from_event(_event(Key.Key_Control)) == vk.VK_LWIN
    assert keys.vk_from_event(_event(Key.Key_Meta)) == vk.VK_CONTROL


def test_linux_names_the_win_key_super():
    keys = LinuxKeys()
    assert keys.display_name(vk.VK_LWIN) == "Super"
    assert keys.display_name(vk.VK_RWIN) == "Right Super"
    assert keys.display_name(vk.VK_CONTROL) == "Ctrl"


@pytest.mark.parametrize(
    ("qt_key", "code"),
    [
        (Key.Key_Control, vk.VK_CONTROL),
        (Key.Key_Meta, vk.VK_LWIN),
        (Key.Key_Space, 0x20),
        (Key.Key_D, 0x44),
        (Key.Key_7, 0x37),
        (Key.Key_F12, 0x7B),
    ],
)
def test_linux_maps_qt_keys_to_vk(qt_key, code):
    assert LinuxKeys().vk_from_event(_event(qt_key)) == code


def test_unknown_keys_give_none():
    assert LinuxKeys().vk_from_event(_event(Key.Key_MediaPlay)) is None
