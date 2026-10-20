"""Single instance on Linux: a lock file plus a Unix socket to the running instance.

The lock (flock on a file in $XDG_RUNTIME_DIR) plays the Windows mutex: the kernel drops it
when the process dies, so a crash never leaves a stale lock. The socket plays the hidden
message window: a second launch connects and writes "open-settings", `spells --quit` writes
"quit", and the running instance hands the text to on_copydata.
"""

from __future__ import annotations

import fcntl
import logging
import os
import socket
import threading
from pathlib import Path
from typing import Any

from spells.platform.base import MessageHandlers

log = logging.getLogger(__name__)

ACCEPT_TIMEOUT_S = 0.25
MAX_COMMAND = 4096

_locks: dict[str, int] = {}
_guard = threading.Lock()


def runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/spells-{os.getuid()}"
    path = Path(base) / "spells"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _safe(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


def socket_path(name: str) -> Path:
    return runtime_dir() / f"{_safe(name)}.sock"


def acquire(name: str) -> bool:
    with _guard:
        if name in _locks:
            return False
        fd = os.open(runtime_dir() / f"{_safe(name)}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        _locks[name] = fd
        return True


def release(name: str) -> None:
    with _guard:
        fd = _locks.pop(name, None)
    if fd is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _connect(name: str) -> socket.socket | None:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2.0)
    try:
        client.connect(str(socket_path(name)))
    except OSError:
        client.close()
        return None
    return client


def find_running(name: str) -> int:
    client = _connect(name)
    if client is None:
        return 0
    client.close()
    return 1


def signal_running(name: str, command: str) -> bool:
    client = _connect(name)
    if client is None:
        return False
    with client:
        try:
            client.sendall(command.encode("utf-8"))
        except OSError:
            return False
    return True


class SocketWindow:
    def __init__(self, name: str, handlers: MessageHandlers) -> None:
        self._path = socket_path(name)
        self._handlers = handlers
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self._path))
        os.chmod(self._path, 0o600)
        self._server.listen(4)
        self._server.settimeout(ACCEPT_TIMEOUT_S)

    def run(self, stop: Any) -> None:
        while not stop.is_set():
            try:
                client, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                if stop.is_set():
                    return
                raise
            with client:
                client.settimeout(1.0)
                try:
                    data = client.recv(MAX_COMMAND)
                except OSError:
                    continue
            handler = self._handlers.on_copydata
            if handler is not None and data:
                try:
                    handler(data.decode("utf-8", "replace"))
                except Exception:
                    log.exception("The single-instance command handler failed")

    def destroy(self) -> None:
        self._server.close()
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass
