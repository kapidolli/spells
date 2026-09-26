from __future__ import annotations

import functools
from dataclasses import replace

from spells.hotkey import HotkeyThread
from spells.platform import stub
from spells.platform.base import Capabilities, Platform
from spells.platform.keys_windows import WindowsKeys
from spells.win32 import hook as win32_hook


def build() -> Platform:
    return replace(
        stub.build("windows"),
        capabilities=Capabilities.everything(),
        key_hook=win32_hook,
        hotkeys=functools.partial(HotkeyThread, hook_backend=win32_hook),
        keys=WindowsKeys(),
    )
