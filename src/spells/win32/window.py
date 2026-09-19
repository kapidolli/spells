"""Foreground window, process name, integrity level and monitor queries.

Spec sections 11.1 (target checks) and 14.2 (pill placement); batch 2 decision V1-1
(elevation is checked at delivery time on the current foreground window).
"""

import ctypes
import logging
import os
from ctypes import wintypes

from .hook import keyboard_input, send_inputs

logger = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_INTEGRITY_LEVEL = 25  # TOKEN_INFORMATION_CLASS.TokenIntegrityLevel
ERROR_INSUFFICIENT_BUFFER = 122
ERROR_ACCESS_DENIED = 5
MONITOR_DEFAULTTONEAREST = 2
VK_MENU = 0x12
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
WS_EX_NOACTIVATE = 0x08000000
SPI_GETCLIENTAREAANIMATION = 0x1042

SECURITY_MANDATORY_UNTRUSTED_RID = 0x0000
SECURITY_MANDATORY_LOW_RID = 0x1000
SECURITY_MANDATORY_MEDIUM_RID = 0x2000
SECURITY_MANDATORY_HIGH_RID = 0x3000
SECURITY_MANDATORY_SYSTEM_RID = 0x4000


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", SID_AND_ATTRIBUTES)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


_user32.GetForegroundWindow.argtypes = ()
_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
_user32.SetForegroundWindow.restype = wintypes.BOOL
_user32.BringWindowToTop.argtypes = (wintypes.HWND,)
_user32.BringWindowToTop.restype = wintypes.BOOL
_user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
_user32.GetWindowTextLengthW.restype = ctypes.c_int
_user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
_user32.GetWindowTextW.restype = ctypes.c_int
_user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, wintypes.LPDWORD)
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
_user32.AttachThreadInput.restype = wintypes.BOOL
_user32.MonitorFromWindow.argtypes = (wintypes.HWND, wintypes.DWORD)
_user32.MonitorFromWindow.restype = wintypes.HMONITOR
_user32.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.c_void_p)
_user32.GetMonitorInfoW.restype = wintypes.BOOL
_user32.SystemParametersInfoW.argtypes = (wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT)
_user32.SystemParametersInfoW.restype = wintypes.BOOL
_GetWindowLong = getattr(_user32, "GetWindowLongPtrW", None) or _user32.GetWindowLongW
_GetWindowLong.argtypes = (wintypes.HWND, ctypes.c_int)
_GetWindowLong.restype = ctypes.c_ssize_t
_SetWindowLong = getattr(_user32, "SetWindowLongPtrW", None) or _user32.SetWindowLongW
_SetWindowLong.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t)
_SetWindowLong.restype = ctypes.c_ssize_t
_kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.GetCurrentProcess.argtypes = ()
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.GetCurrentThreadId.argtypes = ()
_kernel32.GetCurrentThreadId.restype = wintypes.DWORD
_kernel32.QueryFullProcessImageNameW.argtypes = (
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPWSTR,
    wintypes.PDWORD,
)
_kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
_advapi32.OpenProcessToken.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.PHANDLE)
_advapi32.OpenProcessToken.restype = wintypes.BOOL
_advapi32.GetTokenInformation.argtypes = (
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.PDWORD,
)
_advapi32.GetTokenInformation.restype = wintypes.BOOL
_advapi32.GetSidSubAuthorityCount.argtypes = (ctypes.c_void_p,)
_advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
_advapi32.GetSidSubAuthority.argtypes = (ctypes.c_void_p, wintypes.DWORD)
_advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)


def foreground_hwnd() -> int:
    """The current foreground window, 0 when there is none."""
    return int(_user32.GetForegroundWindow() or 0)


def window_title(hwnd: int) -> str:
    length = _user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    copied = _user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value[:copied] if copied > 0 else ""


def window_pid(hwnd: int) -> int:
    """The process id owning the window, 0 when the window is gone."""
    pid = wintypes.DWORD(0)
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def window_thread_id(hwnd: int) -> int:
    return int(_user32.GetWindowThreadProcessId(hwnd, None))


def process_image_path(pid: int) -> str:
    """Full image path of a process, "" when it cannot be opened."""
    if pid <= 0:
        return ""
    handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not _kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return buffer.value[: size.value]
    finally:
        _kernel32.CloseHandle(handle)


def window_process_name(hwnd: int) -> str:
    """The exe basename of the window's process, "" when access is denied."""
    path = process_image_path(window_pid(hwnd))
    return os.path.basename(path) if path else ""


def _token_integrity_level(process_handle: int) -> int | None:
    token = wintypes.HANDLE()
    if not _advapi32.OpenProcessToken(process_handle, TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        needed = wintypes.DWORD(0)
        _advapi32.GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, None, 0, ctypes.byref(needed))
        if ctypes.get_last_error() != ERROR_INSUFFICIENT_BUFFER or needed.value == 0:
            return None
        buffer = ctypes.create_string_buffer(needed.value)
        if not _advapi32.GetTokenInformation(
            token, TOKEN_INTEGRITY_LEVEL, buffer, needed.value, ctypes.byref(needed)
        ):
            return None
        label = ctypes.cast(buffer, ctypes.POINTER(TOKEN_MANDATORY_LABEL)).contents
        sid = label.Label.Sid
        count = _advapi32.GetSidSubAuthorityCount(sid).contents.value
        if count == 0:
            return None
        return int(_advapi32.GetSidSubAuthority(sid, count - 1).contents.value)
    finally:
        _kernel32.CloseHandle(token)


def process_integrity_level(pid: int) -> int | None:
    """The mandatory integrity RID of a process (0x2000 medium, 0x3000 high), or None
    when the process cannot be opened or its token cannot be queried."""
    if pid <= 0:
        return None
    handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        return _token_integrity_level(handle)
    finally:
        _kernel32.CloseHandle(handle)


def current_integrity_level() -> int:
    level = _token_integrity_level(_kernel32.GetCurrentProcess())
    if level is None:
        raise ctypes.WinError(ctypes.get_last_error())
    return level


def is_higher_integrity(level: int | None, own_level: int) -> bool:
    """Pure comparison: an unknown level counts as higher (access denied is the tell)."""
    return level is None or level > own_level


def is_elevated_window(hwnd: int) -> bool:
    """True when the window's process runs above our integrity level or cannot be inspected."""
    return is_higher_integrity(process_integrity_level(window_pid(hwnd)), current_integrity_level())


def _monitor_info_for_window(hwnd: int) -> MONITORINFO:
    monitor = _user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    if not monitor or not _user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return info


def monitor_work_area_for_window(hwnd: int) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the work area of the monitor nearest the window."""
    work = _monitor_info_for_window(hwnd).rcWork
    return (int(work.left), int(work.top), int(work.right), int(work.bottom))


def monitor_rect_for_window(hwnd: int) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the whole monitor nearest the window, in physical pixels.

    The pill maps the top-left corner to a QScreen with QGuiApplication.screenAt (spec 14.2).
    """
    rect = _monitor_info_for_window(hwnd).rcMonitor
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def monitor_handle_for_window(hwnd: int) -> int:
    """The HMONITOR nearest the window (0 when there is none); two windows on one monitor agree."""
    return int(_user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) or 0)


def window_extended_style(hwnd: int) -> int:
    """The window's WS_EX_* style bits."""
    return int(_GetWindowLong(hwnd, GWL_EXSTYLE)) & 0xFFFFFFFF


def set_window_no_activate(hwnd: int) -> int:
    """Add WS_EX_NOACTIVATE (and WS_EX_TOOLWINDOW) so the window never takes focus (spec 14.2).

    Returns the resulting extended style.
    """
    style = window_extended_style(hwnd)
    wanted = style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
    if wanted != style:
        _SetWindowLong(hwnd, GWL_EXSTYLE, wanted)
    return window_extended_style(hwnd)


def animations_enabled() -> bool:
    """Whether Windows shows animations (Settings > Accessibility > Visual effects).

    False means the user asked for reduced motion; the pill then drops its animations.
    Unknown counts as enabled.
    """
    value = wintypes.BOOL(1)
    if not _user32.SystemParametersInfoW(
        SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(value), 0
    ):
        return True
    return bool(value.value)


def bring_to_foreground(hwnd: int) -> bool:
    """Try to make hwnd the foreground window despite the foreground lock.

    Used by tests to focus their own windows. Sends a synthetic Alt tap first (which
    makes this process the last input source) and falls back to attaching to the current
    foreground thread's input. Returns whether hwnd is the foreground window afterwards.
    """
    if foreground_hwnd() == hwnd:
        return True
    send_inputs([keyboard_input(VK_MENU, True), keyboard_input(VK_MENU, False)])
    _user32.SetForegroundWindow(hwnd)
    if foreground_hwnd() == hwnd:
        return True
    current = foreground_hwnd()
    if current:
        fg_thread = window_thread_id(current)
        own_thread = _kernel32.GetCurrentThreadId()
        if fg_thread and fg_thread != own_thread:
            _user32.AttachThreadInput(fg_thread, own_thread, True)
            try:
                _user32.BringWindowToTop(hwnd)
                _user32.SetForegroundWindow(hwnd)
            finally:
                _user32.AttachThreadInput(fg_thread, own_thread, False)
    return foreground_hwnd() == hwnd
