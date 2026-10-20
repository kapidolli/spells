"""Freedesktop glue: autostart, the Secret Service, the file manager and system settings."""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

AUTOSTART_NAME = "spells.desktop"
SECRET_LABEL = "Spells"
SECRET_PREFIX = b"secret-service:"
SETTINGS_COMMANDS = {
    "sound": ("systemsettings", "kcm_pulseaudio"),
    "microphone_privacy": ("systemsettings", "kcm_pulseaudio"),
}


def autostart_path() -> Path:
    config = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(config) if config.startswith("/") else Path.home() / ".config"
    return base / "autostart" / AUTOSTART_NAME


def autostart_entry(command: str) -> str:
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Spells\n"
        "Comment=Offline voice dictation\n"
        f"Exec={command}\n"
        "Icon=spells\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )


def apply_autostart(enabled: bool, command: str) -> bool:
    path = autostart_path()
    try:
        if not enabled:
            path.unlink(missing_ok=True)
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(autostart_entry(command), encoding="utf-8")
    except OSError:
        log.exception("Could not update %s", path)
        return False
    return True


def autostart_command(*, frozen: bool | None = None, executable: str | None = None) -> str:
    """The command that starts this installation: the executable, or `python -m spells`."""
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    exe = executable or sys.executable
    return " ".join(desktop_quote(arg) for arg in ([exe] if is_frozen else [exe, "-m", "spells"]))


def desktop_quote(arg: str) -> str:
    """An Exec argument quoted as the Desktop Entry spec asks, not as a shell would."""
    if arg and not any(char in arg for char in ' \t\n"\'\\><~|&;$*?#()`'):
        return arg
    escaped = "".join("\\" + char if char in '"`$\\' else char for char in arg)
    return f'"{escaped}"'


def autostart_exec() -> str:
    """The Exec line of the autostart entry, empty when there is none."""
    try:
        lines = autostart_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        if line.startswith("Exec="):
            return line[len("Exec="):]
    return ""


def _secret_tool(*args: str, data: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    if shutil.which("secret-tool") is None:
        raise OSError("secret-tool is not installed (package libsecret)")
    return subprocess.run(
        ["secret-tool", *args], input=data, capture_output=True, timeout=10, check=True
    )


def protect(data: bytes, description: str = "") -> bytes:
    """Store data in the Secret Service and return a handle; the blob holds no secret."""
    key = secrets.token_hex(8)
    label = f"{SECRET_LABEL}: {description}" if description else SECRET_LABEL
    _secret_tool(
        "store", f"--label={label}", "application", "spells", "key", key,
        data=base64.b64encode(data),
    )
    return SECRET_PREFIX + key.encode("ascii")


def unprotect(blob: bytes) -> bytes:
    if not blob.startswith(SECRET_PREFIX):
        raise ValueError("not a Spells secret handle")
    key = blob[len(SECRET_PREFIX):].decode("ascii")
    result = _secret_tool("lookup", "application", "spells", "key", key)
    return base64.b64decode(result.stdout.strip())


def _detached(args: list[str]) -> bool:
    try:
        subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        log.warning("Could not run %s", shlex.join(args))
        return False
    return True


def open_path(path: str) -> None:
    _detached(["xdg-open", str(path)])


def reveal(path: Path) -> None:
    uri = Path(path).resolve().as_uri()
    shown = subprocess.run(
        [
            "gdbus", "call", "--session",
            "--dest", "org.freedesktop.FileManager1",
            "--object-path", "/org/freedesktop/FileManager1",
            "--method", "org.freedesktop.FileManager1.ShowItems",
            f"['{uri}']", "",
        ],
        capture_output=True,
        timeout=5,
        check=False,
    )
    if shown.returncode != 0:
        open_path(str(Path(path).parent))


def has_settings(kind: str) -> bool:
    command = SETTINGS_COMMANDS.get(kind)
    return command is not None and shutil.which(command[0]) is not None


def open_settings(kind: str) -> bool:
    command = SETTINGS_COMMANDS.get(kind)
    return command is not None and _detached(list(command))


def parse_sources(listing: str, default_name: str) -> tuple[str, tuple[str, ...]]:
    """(default description, input descriptions) from `pactl -f json list sources`."""
    default = ""
    names: list[str] = []
    for source in json.loads(listing):
        description = str(source.get("description") or source.get("name") or "")
        if source.get("monitor_source") or str(source.get("name", "")).endswith(".monitor"):
            continue
        names.append(description)
        if source.get("name") == default_name:
            default = description
    return default, tuple(names)


def audio_sources() -> tuple[str, tuple[str, ...]] | None:
    if shutil.which("pactl") is None:
        return None
    try:
        listing = subprocess.run(
            ["pactl", "-f", "json", "list", "sources"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        default_name = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return parse_sources(listing, default_name)
    except (OSError, subprocess.SubprocessError, ValueError):
        log.warning("Could not list the sound server's inputs", exc_info=True)
        return None
