from __future__ import annotations

import sys

from spells.platform.base import Capabilities, Platform, PlatformUnavailable
from spells.platform.paths import system_name

_current: Platform | None = None


def _build() -> Platform:
    name = system_name(sys.platform)
    if name == "windows":
        from spells.platform import windows

        return windows.build()
    from spells.platform import stub

    return stub.build(name)


def current() -> Platform:
    global _current
    if _current is None:
        _current = _build()
    return _current


def swap(new: Platform | None) -> Platform | None:
    global _current
    old, _current = _current, new
    return old


__all__ = ["Capabilities", "Platform", "PlatformUnavailable", "current", "swap"]
