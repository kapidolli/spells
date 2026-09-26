from __future__ import annotations

import sys


def system_name(platform: str = sys.platform) -> str:
    if platform == "win32":
        return "windows"
    if platform == "darwin":
        return "macos"
    return "linux"
