"""Playing back a kept recording, on the History page (spec 14.4, 15, 17).

PySide6-Essentials ships no QtMultimedia, so playback goes through sounddevice, the
same library the recorder uses: a raw 16-bit output stream fed from the WAV's frames by
a callback. The callback only copies bytes and sets a flag; a timer on the Qt thread
notices the flag and emits ``finished``, so nothing Qt-shaped ever runs on the audio
thread.

``RECORDINGS_NOTE`` is the one sentence the welcome page and the History page both show,
so the promise is worded once.
"""

from __future__ import annotations

import logging
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import sounddevice as sd
from PySide6 import QtCore, QtWidgets

from spells.ui import style
from spells.ui.widgets import IconButton, make_label

log = logging.getLogger(__name__)

RECORDINGS_NOTE = (
    "Recordings stay on this computer, in a folder next to your history, and are never "
    "sent anywhere or added to a diagnostics bundle."
)
KEEP_AUDIO_TITLE = "Keep the recordings"
KEEP_AUDIO_HINT = (
    "Off by default. With it on, Spells saves the audio of each dictation so you can "
    "listen back to how you spoke."
)

PLAY_GLYPH = chr(0xE768)
STOP_GLYPH = chr(0xE71A)

POLL_MS = 120
BLOCK_FRAMES = 1024


def format_duration(seconds: float | None) -> str:
    """0:06, or empty when the length cannot be read."""
    if seconds is None or seconds < 0:
        return ""
    whole = round(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def default_stream(callback: Callable, sample_rate: int) -> Any:
    return sd.RawOutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_FRAMES,
        callback=callback,
    )


def default_reveal(path: Path) -> None:
    """Show the file in Explorer with it selected."""
    QtCore.QProcess.startDetached("explorer.exe", [f"/select,{path}"])


class WavPlayer(QtCore.QObject):
    """Plays one 16-bit mono WAV at a time through sounddevice.

    play() replaces whatever was playing. The audio callback copies from the frames read
    at play() time and sets `_done` when they run out; the poll timer turns that into
    stop() plus `finished` on the Qt thread, because a Qt signal must not be emitted from
    PortAudio's callback thread.
    """

    started = QtCore.Signal(object)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        stream_factory: Callable[[Callable, int], Any] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._stream_factory = stream_factory or default_stream
        self._stream: Any = None
        self._frames = b""
        self._offset = 0
        self._done = False
        self._path: Path | None = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def playing(self) -> bool:
        return self._stream is not None and not self._done

    def play(self, path: Path) -> bool:
        self.stop()
        try:
            with wave.open(str(path), "rb") as handle:
                sample_rate = handle.getframerate() or 16000
                self._frames = handle.readframes(handle.getnframes())
        except (OSError, wave.Error) as exc:
            log.warning("could not read the recording %s", path, exc_info=True)
            self.failed.emit(f"Could not read {path.name}: {exc}")
            return False
        if not self._frames:
            self.failed.emit(f"{path.name} holds no audio.")
            return False
        self._offset = 0
        self._done = False
        self._path = path
        try:
            self._stream = self._stream_factory(self._callback, sample_rate)
            self._stream.start()
        except Exception as exc:
            log.warning("could not open the output stream", exc_info=True)
            self._stream = None
            self._path = None
            self.failed.emit(f"Could not play the recording: {exc}")
            return False
        self._timer.start()
        self.started.emit(path)
        return True

    def stop(self) -> None:
        self._timer.stop()
        stream = self._stream
        self._stream = None
        self._done = True
        if stream is not None:
            for method in ("stop", "close"):
                try:
                    getattr(stream, method)()
                except Exception:
                    log.debug("output stream %s() failed", method, exc_info=True)
        self._frames = b""
        self._offset = 0
        self._path = None

    def close(self) -> None:
        self.stop()

    def _callback(self, outdata: Any, frames: int, _time: Any = None, _status: Any = None) -> None:
        wanted = frames * 2
        chunk = self._frames[self._offset : self._offset + wanted]
        self._offset += len(chunk)
        outdata[: len(chunk)] = chunk
        if len(chunk) < wanted:
            outdata[len(chunk) :] = bytes(wanted - len(chunk))
            self._done = True

    def _poll(self) -> None:
        if self._done:
            path = self._path
            self.stop()
            self.finished.emit(path)


class PlayCell(QtWidgets.QWidget):
    """The play and stop control of one History row, with the recording's length."""

    toggled = QtCore.Signal(int)

    def __init__(
        self, row: int, seconds: float | None, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.row = row
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.button = IconButton(PLAY_GLYPH, "Play this recording", size=24, parent=self)
        self.button.clicked.connect(lambda: self.toggled.emit(self.row))
        self.length = make_label(format_duration(seconds), "caption", "secondary", parent=self)
        layout.addWidget(self.button)
        layout.addWidget(self.length)
        layout.addStretch(1)

    def set_playing(self, playing: bool) -> None:
        glyph = STOP_GLYPH if playing else PLAY_GLYPH
        self.button.setText(glyph if style.has_icon_font() else "x")
        self.button.setToolTip("Stop" if playing else "Play this recording")
        self.button.setAccessibleName(self.button.toolTip())


__all__ = [
    "BLOCK_FRAMES",
    "KEEP_AUDIO_HINT",
    "KEEP_AUDIO_TITLE",
    "PLAY_GLYPH",
    "RECORDINGS_NOTE",
    "STOP_GLYPH",
    "PlayCell",
    "WavPlayer",
    "default_reveal",
    "default_stream",
    "format_duration",
]
