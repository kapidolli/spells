from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

from spells.platform.base import PlatformPaths

APP_DIR = "Spells"
XDG_APP_DIR = "spells"
SETTINGS_FILE = "settings.json"
HISTORY_FILE = "history.db"
LOGS_DIR = "logs"


def system_name(platform: str = sys.platform) -> str:
    if platform == "win32":
        return "windows"
    if platform == "darwin":
        return "macos"
    return "linux"


def windows_paths(env: Mapping[str, str], home: Path) -> PlatformPaths:
    local = env.get("LOCALAPPDATA")
    history = (Path(local) if local else home / "AppData" / "Local") / APP_DIR / HISTORY_FILE
    return PlatformPaths(
        settings_path=Path(env["APPDATA"]) / APP_DIR / SETTINGS_FILE,
        history_path=history,
        log_dir=history.parent / LOGS_DIR,
    )


def _xdg(env: Mapping[str, str], name: str, default: Path) -> Path:
    value = env.get(name, "")
    return Path(value) if value.startswith("/") else default


def linux_paths(env: Mapping[str, str], home: Path) -> PlatformPaths:
    config = _xdg(env, "XDG_CONFIG_HOME", home / ".config")
    data = _xdg(env, "XDG_DATA_HOME", home / ".local" / "share")
    state = _xdg(env, "XDG_STATE_HOME", home / ".local" / "state")
    return PlatformPaths(
        settings_path=config / XDG_APP_DIR / SETTINGS_FILE,
        history_path=data / XDG_APP_DIR / HISTORY_FILE,
        log_dir=state / XDG_APP_DIR / LOGS_DIR,
    )


def macos_paths(env: Mapping[str, str], home: Path) -> PlatformPaths:
    support = home / "Library" / "Application Support" / APP_DIR
    return PlatformPaths(
        settings_path=support / SETTINGS_FILE,
        history_path=support / HISTORY_FILE,
        log_dir=home / "Library" / "Logs" / APP_DIR,
    )


def paths_for(system: str, env: Mapping[str, str], home: Path) -> PlatformPaths:
    builders = {"windows": windows_paths, "linux": linux_paths, "macos": macos_paths}
    return builders[system](env, home)


def current_paths() -> PlatformPaths:
    return paths_for(system_name(), os.environ, Path.home())


def engine_suffix(system: str | None = None) -> str:
    return ".exe" if (system or system_name()) == "windows" else ""
