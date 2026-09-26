from __future__ import annotations

import ctypes
from ctypes import wintypes

CRYPTPROTECT_UI_FORBIDDEN = 0x1


class DATA_BLOB(ctypes.Structure):
    _fields_ = (
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    )


class DpapiError(OSError):
    pass


_crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_crypt32.CryptProtectData.argtypes = (
    ctypes.POINTER(DATA_BLOB),
    wintypes.LPCWSTR,
    ctypes.POINTER(DATA_BLOB),
    ctypes.c_void_p,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(DATA_BLOB),
)
_crypt32.CryptProtectData.restype = wintypes.BOOL
_crypt32.CryptUnprotectData.argtypes = (
    ctypes.POINTER(DATA_BLOB),
    ctypes.c_void_p,
    ctypes.POINTER(DATA_BLOB),
    ctypes.c_void_p,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(DATA_BLOB),
)
_crypt32.CryptUnprotectData.restype = wintypes.BOOL
_kernel32.LocalFree.argtypes = (wintypes.HLOCAL,)
_kernel32.LocalFree.restype = wintypes.HLOCAL


def _input_blob(data: bytes) -> tuple[DATA_BLOB, ctypes.Array]:
    buffer = ctypes.create_string_buffer(bytes(data), max(1, len(data)))
    blob = DATA_BLOB(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    return blob, buffer


def _take(blob: DATA_BLOB) -> bytes:
    try:
        return ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        _kernel32.LocalFree(ctypes.cast(blob.pbData, ctypes.c_void_p))


def protect(data: bytes, description: str = "") -> bytes:
    source, _buffer = _input_blob(data)
    output = DATA_BLOB()
    ok = _crypt32.CryptProtectData(
        ctypes.byref(source),
        description or None,
        None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output),
    )
    if not ok:
        code = ctypes.get_last_error()
        raise DpapiError(code, f"CryptProtectData failed ({code})")
    return _take(output)


def unprotect(blob: bytes) -> bytes:
    source, _buffer = _input_blob(blob)
    output = DATA_BLOB()
    ok = _crypt32.CryptUnprotectData(
        ctypes.byref(source),
        None,
        None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output),
    )
    if not ok:
        code = ctypes.get_last_error()
        raise DpapiError(code, f"CryptUnprotectData failed ({code})")
    return _take(output)


__all__ = ["CRYPTPROTECT_UI_FORBIDDEN", "DATA_BLOB", "DpapiError", "protect", "unprotect"]
