from __future__ import annotations

import os
import subprocess
from pathlib import Path

from spells.platform.base import SETTINGS_MICROPHONE_PRIVACY, SETTINGS_SOUND

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

SETTINGS_URIS = {
    SETTINGS_SOUND: "ms-settings:sound",
    SETTINGS_MICROPHONE_PRIVACY: "ms-settings:privacy-microphone",
}

# Where Windows keeps the microphone privacy switches (Settings, Privacy and
# security, Microphone). HKLM: the device-wide switch; HKCU: this user's switch;
# HKCU ...\NonPackaged: "Let desktop apps access your microphone", the one that
# applies to Spells. A denied desktop app still opens the device but receives
# silence, which is why the blocked check looks at the samples.
_CONSENT_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager"
    r"\ConsentStore\microphone"
)


def open_path(path: str) -> None:
    os.startfile(path)


def reveal(path: Path) -> None:
    subprocess.Popen(
        ["explorer.exe", f"/select,{path}"],
        close_fds=True,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    )


def open_settings(kind: str) -> bool:
    uri = SETTINGS_URIS.get(kind)
    if uri is None:
        return False
    os.startfile(uri)
    return True


def microphone_privacy_denied() -> bool | None:
    """True when a Windows microphone privacy switch is set to Deny, None when unreadable."""
    try:
        import winreg
    except ImportError:
        return None
    checks = (
        (winreg.HKEY_LOCAL_MACHINE, _CONSENT_KEY),
        (winreg.HKEY_CURRENT_USER, _CONSENT_KEY),
        (winreg.HKEY_CURRENT_USER, _CONSENT_KEY + r"\NonPackaged"),
    )
    readable = False
    for root, subkey in checks:
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Value")
        except OSError:
            continue
        readable = True
        if str(value).casefold() == "deny":
            return True
    return False if readable else None


__all__ = [
    "SETTINGS_URIS",
    "microphone_privacy_denied",
    "open_path",
    "open_settings",
    "reveal",
]
