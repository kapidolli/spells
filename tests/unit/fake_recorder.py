"""A stand-in for spells.audio.Recorder, fed from a PCM buffer the test sets.

Mirrors the Recorder contract the pipeline relies on (spec 5.2 audio row): start(),
stop() -> pcm16, cancel(), close(), snapshot(), device, keep_open, open_latency_ms,
silent, duration_s, sample_rate and capture_info(). sample_rate None means "the device's own
rate", which is what the app now asks for; the fake answers with NATIVE_SAMPLE_RATE,
the rate WASAPI reports on the owner's machine. start and stop failures are scripted
with MicError; the level callback is driven from the test with feed_level(), which
stands in for the PortAudio thread.
"""

from __future__ import annotations

from collections.abc import Callable

from spells.audio import CaptureInfo, MicError

DEFAULT_OPEN_LATENCY_MS = 37.0
NATIVE_SAMPLE_RATE = 48000


class FakeRecorder:
    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int | None = None,
        on_level: Callable[[float], None] | None = None,
        keep_open: bool = False,
    ) -> None:
        self.device = device
        self.requested_sample_rate = sample_rate
        self.sample_rate = int(sample_rate) if sample_rate else NATIVE_SAMPLE_RATE
        self.on_level = on_level
        self.keep_open = keep_open
        # Two seconds of non-zero samples, long enough for the auto-language path.
        self.pcm: bytes = b"\x01\x00" * (2 * self.sample_rate)
        self.start_error: MicError | None = None
        self.stop_error: MicError | None = None
        self.cold_open_latency_ms = DEFAULT_OPEN_LATENCY_MS
        self.open_latency_ms: float | None = None
        self.error: BaseException | None = None
        self.silent = False
        self.is_recording = False
        self.stream_open = False
        self.closed = False
        self.calls: list[str] = []
        self.host_api = "Windows WASAPI"
        self.device_name = device if isinstance(device, str) else "Fake Microphone"
        self.device_match = "exact"
        self.fallback_notice: str | None = None
        self.channels = 1
        self.dropped_blocks = 0

    @property
    def downmixed(self) -> bool:
        return self.channels > 1

    @property
    def duration_s(self) -> float:
        if not self.is_recording:
            return 0.0
        return len(self.pcm) / (2 * self.sample_rate)

    def capture_info(self) -> CaptureInfo:
        return CaptureInfo(
            device=str(self.device_name or ""),
            host_api=self.host_api,
            sample_rate=self.sample_rate,
            channels=self.channels,
            downmixed=self.downmixed,
            dropped_blocks=self.dropped_blocks,
            match=self.device_match,
        )

    def start(self) -> None:
        self.calls.append("start")
        if self.is_recording:
            raise RuntimeError("Recorder.start() called while already recording")
        if self.start_error is not None:
            raise self.start_error
        self.open_latency_ms = 0.0 if self.stream_open else self.cold_open_latency_ms
        self.stream_open = True
        self.is_recording = True

    def stop(self) -> bytes:
        self.calls.append("stop")
        if not self.is_recording:
            raise RuntimeError("Recorder.stop() called while not recording")
        self.is_recording = False
        if not self.keep_open:
            self.stream_open = False
        if self.stop_error is not None:
            raise self.stop_error
        return self.pcm

    def cancel(self) -> None:
        self.calls.append("cancel")
        self.is_recording = False
        if not self.keep_open:
            self.stream_open = False

    def close(self) -> None:
        self.calls.append("close")
        self.is_recording = False
        self.stream_open = False
        self.closed = True

    def snapshot(self, since_bytes: int = 0) -> tuple[bytes, int]:
        """What the live partial passes read: the audio so far from an offset on."""
        data = self.pcm if self.is_recording else b""
        start = max(0, int(since_bytes))
        start -= start % 2
        if start >= len(data):
            return b"", len(data)
        return data[start:], len(data)

    def feed_level(self, level: float) -> None:
        """Deliver one level update the way the PortAudio callback would."""
        if self.on_level is not None:
            self.on_level(level)
