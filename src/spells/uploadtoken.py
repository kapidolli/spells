from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

TOKEN_FILE = "upload-token.bin"
DESCRIPTION = "Spells upload token"


def token_path_for(history_path: Path) -> Path:
    return Path(history_path).parent / TOKEN_FILE


def _protect(data: bytes) -> bytes:
    from spells.win32 import dpapi

    return dpapi.protect(data, DESCRIPTION)


def _unprotect(blob: bytes) -> bytes:
    from spells.win32 import dpapi

    return dpapi.unprotect(blob)


class TokenStore:
    def __init__(
        self,
        path: Path | None,
        *,
        protect: Callable[[bytes], bytes] = _protect,
        unprotect: Callable[[bytes], bytes] = _unprotect,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._protect = protect
        self._unprotect = unprotect
        self._memory = ""

    def read(self) -> str:
        if self.path is None:
            return self._memory
        try:
            blob = self.path.read_bytes()
        except FileNotFoundError:
            return ""
        except OSError:
            log.warning("the upload token file could not be read")
            return ""
        if not blob:
            return ""
        try:
            return self._unprotect(blob).decode("utf-8")
        except (OSError, UnicodeDecodeError):
            log.warning("the upload token could not be decrypted for this Windows account")
            return ""

    def exists(self) -> bool:
        if self.path is None:
            return bool(self._memory)
        try:
            return self.path.is_file()
        except OSError:
            return False

    def write(self, token: str) -> None:
        value = (token or "").strip()
        if not value:
            self.delete()
            return
        if self.path is None:
            self._memory = value
            return
        blob = self._protect(value.encode("utf-8"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def delete(self) -> None:
        self._memory = ""
        if self.path is None:
            return
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            log.warning("the upload token file could not be removed")


__all__ = ["TOKEN_FILE", "TokenStore", "token_path_for"]
