"""Single instance: named mutex plus the WM_COPYDATA channel to the running instance.

Spec section 15 and 19.5; batch 2 decisions V1-11 and V4-8. A second launch signals
"open-settings"; `Spells.exe --quit` and the uninstaller signal "quit".
"""

import ctypes
import threading
from ctypes import wintypes

from .msgwindow import HWND_MESSAGE, send_copydata

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_user32 = ctypes.WinDLL("user32", use_last_error=True)

ERROR_ALREADY_EXISTS = 183

_kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
_kernel32.CreateMutexW.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
_kernel32.CloseHandle.restype = wintypes.BOOL
_user32.FindWindowExW.argtypes = (wintypes.HWND, wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR)
_user32.FindWindowExW.restype = wintypes.HWND
_user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
_user32.FindWindowW.restype = wintypes.HWND

# Mutex handles kept open for the life of the process (or until released).
_mutexes: dict[str, int] = {}
_mutex_lock = threading.Lock()


def acquire_single_instance(name: str) -> bool:
    """Create the named mutex; False when another process (or an earlier call) owns it."""
    ctypes.set_last_error(0)
    handle = _kernel32.CreateMutexW(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        _kernel32.CloseHandle(handle)
        return False
    with _mutex_lock:
        _mutexes[name] = int(handle)
    return True


def release_single_instance(name: str) -> None:
    """Close a mutex acquired by acquire_single_instance (used by tests and on exit)."""
    with _mutex_lock:
        handle = _mutexes.pop(name, None)
    if handle is not None:
        _kernel32.CloseHandle(handle)


def find_message_window(class_name: str) -> int:
    """The running instance's window by class name, 0 when none is found.

    Looks under HWND_MESSAGE first and then among top-level windows, which is where
    msgwindow.MessageWindow lives.
    """
    hwnd = _user32.FindWindowExW(HWND_MESSAGE, None, class_name, None)
    if not hwnd:
        hwnd = _user32.FindWindowW(class_name, None)
    return int(hwnd or 0)


def signal_running_instance(class_name: str, command: str) -> bool:
    """Send a command ("open-settings", "quit") to the running instance; False when absent."""
    hwnd = find_message_window(class_name)
    if not hwnd:
        return False
    return send_copydata(hwnd, command)
