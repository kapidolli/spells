"""Start with Windows through the HKCU Run key (spec 15).

`HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run`, value `Spells`. Writing there
needs no elevation, survives an in-place upgrade, and the uninstaller removes it. Nothing
here raises: a registry that refuses us is a logged warning and a setting that did not take
effect, never a failed start.
"""

from __future__ import annotations

import logging
import sys
import winreg
from pathlib import Path

log = logging.getLogger(__name__)

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "Spells"
MODULE_ARGS = "-m spells"
PYTHONW = "pythonw.exe"


def current_command(
    *,
    frozen: bool | None = None,
    executable: Path | None = None,
) -> str:
    """The command line the Run value should carry for this installation.

    Frozen: the executable itself. Development: the venv's pythonw.exe (no console window)
    running `-m spells`, falling back to the running interpreter when pythonw is absent.
    """
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    exe = Path(executable) if executable is not None else Path(sys.executable)
    if is_frozen:
        return f'"{exe}"'
    windowless = exe.with_name(PYTHONW)
    if _exists(windowless):
        exe = windowless
    return f'"{exe}" {MODULE_ARGS}'


def current_value(*, key_path: str = RUN_KEY, value_name: str = VALUE_NAME) -> str | None:
    """The Run value as stored, or None when it (or the key) is absent or unreadable."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            value, _kind = winreg.QueryValueEx(key, value_name)
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("Could not read the autostart value in HKCU\\%s", key_path, exc_info=True)
        return None
    return str(value)


def apply(
    enabled: bool,
    command: str,
    *,
    key_path: str = RUN_KEY,
    value_name: str = VALUE_NAME,
) -> bool:
    """Write or delete the Run value. Idempotent; True when the registry now matches."""
    try:
        if enabled:
            return _write(key_path, value_name, command)
        return _delete(key_path, value_name)
    except OSError:
        log.warning(
            "Could not %s the autostart entry in HKCU\\%s",
            "write" if enabled else "remove",
            key_path,
            exc_info=True,
        )
        return False


def _write(key_path: str, value_name: str, command: str) -> bool:
    if current_value(key_path=key_path, value_name=value_name) == command:
        return True
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, command)
    log.info("Autostart enabled: %s", command)
    return True


def _delete(key_path: str, value_name: str) -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, value_name)
    except FileNotFoundError:
        return True
    log.info("Autostart disabled")
    return True


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


__all__ = ["RUN_KEY", "VALUE_NAME", "apply", "current_command", "current_value"]
