"""The Wayland clipboard through wl-clipboard (wl-copy and wl-paste).

wl-copy and wl-paste use the data-control protocol, so they reach the clipboard without a
focused window, which an app talking to its own Wayland or X11 connection cannot. Our text is
offered with the sensitive hint, which keeps it out of Klipper's history, as the Windows
build keeps it out of the clipboard history.

The clipboard has no sequence number on Wayland, so `sequence_number` is a fingerprint of the
offered types and the text. A snapshot keeps one representation, the richest of an image,
a file list or text, because wl-copy offers a single type per call.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import zlib
from dataclasses import dataclass

log = logging.getLogger(__name__)

TIMEOUT_S = 2.0
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
TEXT_TYPES = ("text/plain;charset=utf-8", "UTF8_STRING", "text/plain", "STRING", "TEXT")
SNAPSHOT_PREFERENCE = ("image/png", "text/uri-list", *TEXT_TYPES)


@dataclass(frozen=True)
class Snapshot:
    mime: str | None
    data: bytes


def available() -> bool:
    """wl-clipboard is installed and this is a Wayland session; it cannot reach X11."""
    return bool(
        os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy") and shutil.which("wl-paste")
    )


def _paste(*args: str) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            ["wl-paste", *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("wl-paste %s failed: %s", " ".join(args), exc)
        return None


def _copy(data: bytes, *args: str) -> None:
    subprocess.run(
        ["wl-copy", *args],
        input=data,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=TIMEOUT_S,
        check=True,
    )


def offered_types() -> list[str]:
    result = _paste("--list-types")
    if result is None or result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.decode("utf-8", "replace").splitlines() if line.strip()]


def pick_type(types: list[str]) -> str | None:
    for mime in SNAPSHOT_PREFERENCE:
        if mime in types:
            return mime
    return types[0] if types else None


def _text_type(types: list[str]) -> str | None:
    return next((mime for mime in TEXT_TYPES if mime in types), None)


def _read(mime: str) -> bytes | None:
    result = _paste("--no-newline", "--type", mime)
    if result is None or result.returncode != 0:
        return None
    return result.stdout


def fingerprint(types: list[str], text: bytes | None) -> int:
    seed = "\n".join(types).encode("utf-8") + b"\0" + (text or b"")
    return zlib.crc32(seed) & 0x7FFFFFFF


def snapshot() -> Snapshot:
    mime = pick_type(offered_types())
    if mime is None:
        return Snapshot(None, b"")
    data = _read(mime)
    if data is None or len(data) > MAX_SNAPSHOT_BYTES:
        return Snapshot(None, b"")
    return Snapshot(mime, data)


def set_text(text: str) -> int:
    _copy(text.encode("utf-8"), "--sensitive", "--type", "text/plain;charset=utf-8")
    return sequence_number()


def get_text() -> str | None:
    mime = _text_type(offered_types())
    if mime is None:
        return None
    data = _read(mime)
    return None if data is None else data.decode("utf-8", "replace")


def sequence_number() -> int:
    types = offered_types()
    mime = _text_type(types)
    return fingerprint(types, _read(mime) if mime else None)


def restore(snap: Snapshot | None, expected_sequence: int) -> bool:
    if sequence_number() != expected_sequence:
        return False
    if snap is None or snap.mime is None:
        _copy(b"", "--clear")
        return True
    _copy(snap.data, "--type", snap.mime)
    return True
