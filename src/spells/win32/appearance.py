from __future__ import annotations

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36
DWMWCP_ROUND = 2
DWMWCP_ROUNDSMALL = 3

ACCENT_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Accent"
DWM_KEY = r"Software\Microsoft\Windows\DWM"
PERSONALIZE_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"


def _read_registry(path: str, name: str) -> object | None:
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            value, _kind = winreg.QueryValueEx(key, name)
            return value
    except OSError:
        return None


def apps_dark() -> bool | None:
    value = _read_registry(PERSONALIZE_KEY, "AppsUseLightTheme")
    if isinstance(value, int):
        return value == 0
    return None


def taskbar_light() -> bool:
    return bool(_read_registry(PERSONALIZE_KEY, "SystemUsesLightTheme"))


def accent_palette() -> tuple[str, ...] | None:
    data = _read_registry(ACCENT_KEY, "AccentPalette")
    if not isinstance(data, (bytes, bytearray)) or len(data) < 28:
        return None
    return tuple(f"#{data[i]:02X}{data[i + 1]:02X}{data[i + 2]:02X}" for i in range(0, 28, 4))


def accent_colour() -> str | None:
    abgr = _read_registry(DWM_KEY, "AccentColor")
    if not isinstance(abgr, int):
        return None
    red, green, blue = abgr & 0xFF, (abgr >> 8) & 0xFF, (abgr >> 16) & 0xFF
    return f"#{red:02X}{green:02X}{blue:02X}"


def colorref(colour: str) -> int:
    red, green, blue = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return red | (green << 8) | (blue << 16)


def _set_dwm_attribute(hwnd: int, attribute: int, value: int) -> bool:
    import ctypes
    from ctypes import wintypes

    data = ctypes.c_int(int(value))
    result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
        wintypes.HWND(hwnd), ctypes.c_uint(attribute), ctypes.byref(data), ctypes.sizeof(data)
    )
    return result == 0


def style_title_bar(native_id: int, *, dark: bool, caption: str, text: str) -> bool:
    _set_dwm_attribute(native_id, DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if dark else 0)
    _set_dwm_attribute(native_id, DWMWA_CAPTION_COLOR, colorref(caption))
    _set_dwm_attribute(native_id, DWMWA_TEXT_COLOR, colorref(text))
    return True


def round_corners(native_id: int) -> bool:
    return _set_dwm_attribute(native_id, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUNDSMALL)
