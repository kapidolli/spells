from __future__ import annotations

from PySide6 import QtWidgets

UNAVAILABLE = "Not available on this system yet."
NOTES = {
    "hold_to_talk": "Dictation hotkeys are not available on this system yet.",
    "esc_cancel": "Esc cannot cancel a dictation on this system yet.",
    "live_typing": "Live typing is not available on this system yet.",
    "app_profiles": (
        "Spells cannot tell which app is in front on this system yet, so every app uses the "
        "Default profile."
    ),
    "window_titles": "Window titles cannot be read on this system yet.",
    "edit_hotkey": "The edit hotkey is not available on this system yet.",
    "clipboard_restore": "On this system the dictated text stays on the clipboard after it is pasted.",
    "autostart": "Starting with the session is not available on this system yet.",
    "self_update": "Updates are installed from the download page on this system.",
    "app_records_chords": "Change this shortcut in your system's keyboard settings.",
    "elevation_check": "Spells cannot check whether an app runs as administrator on this system.",
}


def note_for(flag: str) -> str:
    return NOTES.get(flag, UNAVAILABLE)


def apply_capability(widget: QtWidgets.QWidget, available: bool, flag: str) -> None:
    if available:
        return
    widget.setEnabled(False)
    widget.setToolTip(note_for(flag))


__all__ = ["NOTES", "UNAVAILABLE", "apply_capability", "note_for"]
