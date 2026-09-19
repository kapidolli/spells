"""Microphone capture: device listing, native-rate mono int16 recording, level metering.

Spec 5.2 (audio row), 6 steps 1 and 3, 12 (mic-open timing), 14.4 (mic picker,
keep mic warm), 16 (mic missing, busy, blocked). Built on sounddevice, which
wraps PortAudio. Audio is never written to disk (spec 17): the PCM stays in
memory and stop() hands it to the caller.

Devices are listed from the WASAPI host API when the machine has one, and from
PortAudio's default host API (MME on Windows) when it does not. WASAPI reports
the device's shared-mode format (48 kHz here), 3 to 10 ms of latency and the
full device name; MME reports 44.1 kHz, 90 to 180 ms and a name truncated to 31
characters, and it resamples in the driver. Capture therefore runs at the rate
the device offers and the sample rate travels with the recording: the speech
engines resample through miniaudio, which is a better converter than the one
behind MME.

WASAPI shared mode refuses 16 kHz outright (PaErrorCode -9997), so a caller
that asks for a fixed rate is served by the MME entry of the same device. That
is the same automatic fallback a WASAPI device that refuses to open gets.

Device indices are PortAudio's global indices and are only stable within a
process, so settings store the device name; Recorder accepts either.
"""

from __future__ import annotations

import array
import contextlib
import io
import logging
import math
import operator
import threading
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import sounddevice as sd

logger = logging.getLogger(__name__)

MicErrorKind = Literal["missing", "busy", "blocked", "unknown"]
DeviceMatch = Literal["index", "exact", "prefix", "default"]

LEVEL_UPDATES_PER_SECOND = 20
BLOCKED_SILENCE_MS = 500

WASAPI = "Windows WASAPI"
MME = "MME"
FALLBACK_SAMPLE_RATE = 48000

# PaErrorCode values from portaudio.h.
PA_UNANTICIPATED_HOST_ERROR = -9999
PA_INVALID_CHANNEL_COUNT = -9998
PA_INVALID_SAMPLE_RATE = -9997
PA_INVALID_DEVICE = -9996
PA_DEVICE_UNAVAILABLE = -9985

# Host error texts, lower-cased with underscores replaced by spaces. MME says
# "The specified device is already in use"; WASAPI reports AUDCLNT_E_DEVICE_IN_USE
# (0x8889000A) or AUDCLNT_E_DEVICE_INVALIDATED (0x88890004) when a device vanished.
_BUSY_PATTERNS = ("in use", "busy", "already allocated", "0x8889000a")
_MISSING_PATTERNS = (
    "out of range",
    "not found",
    "no driver",
    "baddeviceid",
    "not present",
    "invalidated",
    "disconnected",
    "0x88890004",
)

# Where Windows keeps the microphone privacy switches (Settings, Privacy and
# security, Microphone). HKLM: the device-wide switch; HKCU: this user's switch;
# HKCU ...\NonPackaged: "Let desktop apps access your microphone", the one that
# applies to Spells. A denied desktop app still opens the device but receives
# silence, which is why the blocked check looks at the samples.
_CONSENT_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager"
    r"\ConsentStore\microphone"
)

DEVICE_MOVED_NOTICE = (
    'The microphone "{wanted}" is gone, so Spells is recording from {fallback} instead. '
    "Pick the one you want on the General page."
)


@dataclass(frozen=True)
class AudioDevice:
    """One input device of the capture host API, as the picker and Recorder see it."""

    index: int
    name: str
    is_default: bool
    host_api: str = ""
    sample_rate: int = 0
    channels: int = 0


@dataclass(frozen=True)
class CapturePlan:
    """One way to open a device: a PortAudio index, a rate and a channel count."""

    index: int
    name: str
    host_api: str
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class CaptureInfo:
    """What a recording was captured with, for the history row and Diagnostics."""

    device: str = ""
    host_api: str = ""
    sample_rate: int = 0
    channels: int = 1
    downmixed: bool = False
    dropped_blocks: int = 0
    match: DeviceMatch = "exact"


class MicError(Exception):
    """A microphone problem the pill can name (spec 16 mic row, decision V3-F9).

    kind "missing": no input device at all.
    kind "busy": the device is held by another app or reported unavailable.
    kind "blocked": the stream delivered only zeros for the first 500 ms, which
    is what a desktop app gets when the Windows microphone privacy setting
    denies it (Privacy settings).
    kind "unknown": anything else; the message carries the PortAudio text.

    A stored device name that no longer matches any device is no longer a
    MicError: the recorder falls back to the system default so the dictation
    still happens (spec 16, and the notice of DEVICE_MOVED_NOTICE).
    """

    def __init__(self, kind: MicErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind: MicErrorKind = kind
        self.message = message


def _query() -> tuple[list[Any], list[Any]]:
    try:
        return list(sd.query_devices()), list(sd.query_hostapis())
    except Exception as exc:
        raise MicError("unknown", f"Could not enumerate audio devices: {exc}") from exc


def _api_index(hostapis: Sequence[Any], name: str) -> int | None:
    for index, api in enumerate(hostapis):
        if str(api["name"]).strip().casefold() == name.casefold():
            return index
    return None


def _has_inputs(devices: Sequence[Any], api: Any) -> bool:
    return any(int(devices[i]["max_input_channels"]) > 0 for i in api["devices"])


def _capture_api(devices: Sequence[Any], hostapis: Sequence[Any]) -> int:
    """The host API to list and record from: WASAPI when it has inputs, else the default."""
    wasapi = _api_index(hostapis, WASAPI)
    if wasapi is not None and _has_inputs(devices, hostapis[wasapi]):
        return wasapi
    try:
        fallback = int(getattr(sd.default, "hostapi", 0))
    except (TypeError, ValueError):
        fallback = 0
    if 0 <= fallback < len(hostapis):
        return fallback
    return 0


def _rate_of(info: Any) -> int:
    try:
        return round(float(info.get("default_samplerate") or 0))
    except (TypeError, ValueError):
        return 0


def _api_inputs(devices: Sequence[Any], hostapis: Sequence[Any], api_index: int) -> list[AudioDevice]:
    if not hostapis or not 0 <= api_index < len(hostapis):
        return []
    api = hostapis[api_index]
    api_name = str(api["name"])
    try:
        default_index = int(api["default_input_device"])
    except (TypeError, ValueError, KeyError):
        default_index = -1
    result: list[AudioDevice] = []
    seen: set[str] = set()
    for index in api["devices"]:
        info = devices[index]
        if int(info["max_input_channels"]) <= 0:
            continue
        name = str(info["name"])
        key = name.strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(
            AudioDevice(
                index=index,
                name=name,
                is_default=index == default_index,
                host_api=api_name,
                sample_rate=_rate_of(info),
                channels=int(info["max_input_channels"]),
            )
        )
    return result


def list_input_devices() -> list[AudioDevice]:
    """Input devices of the capture host API, each once, with the system default flagged."""
    devices, hostapis = _query()
    if not hostapis:
        return []
    return _api_inputs(devices, hostapis, _capture_api(devices, hostapis))


def capture_host_api() -> str:
    """The name of the host API device listing and recording use on this machine."""
    devices, hostapis = _query()
    if not hostapis:
        return ""
    return str(hostapis[_capture_api(devices, hostapis)]["name"])


def names_match(stored: str, name: str) -> bool:
    """Exact match, or one name is the start of the other (MME truncates at 31 characters)."""
    left = stored.strip().casefold()
    right = name.strip().casefold()
    if not left or not right:
        return False
    return left == right or left.startswith(right) or right.startswith(left)


def match_device(devices: Sequence[AudioDevice], wanted: str) -> tuple[AudioDevice | None, DeviceMatch]:
    """A stored device name against the current list: exact, then prefix, then the default.

    Settings written before capture moved to WASAPI hold MME's 31-character
    names ("Microphone (HyperX Cloud III Wi"), so an exact match fails and the
    prefix round finds the full name. A name that matches nothing falls back to
    the system default rather than failing the dictation (spec 16).
    """
    needle = wanted.strip().casefold()
    for device in devices:
        if device.name.strip().casefold() == needle:
            return device, "exact"
    for device in devices:
        if names_match(wanted, device.name):
            return device, "prefix"
    return default_device(devices), "default"


def default_device(devices: Sequence[AudioDevice]) -> AudioDevice | None:
    for device in devices:
        if device.is_default:
            return device
    return devices[0] if devices else None


def capture_plans(
    device: int | str | None, sample_rate: int | None
) -> tuple[list[CapturePlan], AudioDevice, DeviceMatch]:
    """The opens to try for one device, best first, with the device that was matched.

    The first plan is the capture host API's entry (WASAPI where there is one)
    at the device's own rate, or at sample_rate when the caller insisted on one.
    The second is the same device's MME entry, which is what serves a WASAPI
    device that refuses to open and a caller that asked for a rate WASAPI's
    shared mode rejects, 16 kHz among them.
    """
    devices, hostapis = _query()
    if not hostapis:
        raise MicError("missing", "No microphone found. Connect one or enable it in Sound settings.")
    primary_api = _capture_api(devices, hostapis)
    primary = _api_inputs(devices, hostapis, primary_api)
    if not primary:
        raise MicError("missing", "No microphone found. Connect one or enable it in Sound settings.")

    match: DeviceMatch = "exact"
    if device is None:
        chosen = default_device(primary)
        match = "default"
    elif isinstance(device, int):
        chosen = next((candidate for candidate in primary if candidate.index == device), None)
        if chosen is None:
            chosen = _device_by_index(devices, hostapis, device)
        if chosen is None:
            raise MicError(
                "missing",
                f"Microphone {device} is not available. Pick another one in the settings.",
            )
        match = "index"
    else:
        chosen, match = match_device(primary, str(device))
    if chosen is None:
        raise MicError("missing", "No microphone found. Connect one or enable it in Sound settings.")

    plans = [_plan(chosen, sample_rate)]
    mme = _mme_twin(devices, hostapis, chosen)
    if mme is not None and mme.index != chosen.index:
        plans.append(_plan(mme, sample_rate))
    return plans, chosen, match


def _plan(device: AudioDevice, sample_rate: int | None) -> CapturePlan:
    rate = int(sample_rate) if sample_rate else (device.sample_rate or FALLBACK_SAMPLE_RATE)
    return CapturePlan(
        index=device.index,
        name=device.name,
        host_api=device.host_api,
        sample_rate=rate,
        channels=max(1, device.channels),
    )


def _device_by_index(
    devices: Sequence[Any], hostapis: Sequence[Any], index: int
) -> AudioDevice | None:
    if not 0 <= index < len(devices):
        return None
    info = devices[index]
    if int(info["max_input_channels"]) <= 0:
        return None
    api = hostapis[int(info["hostapi"])] if 0 <= int(info["hostapi"]) < len(hostapis) else None
    return AudioDevice(
        index=index,
        name=str(info["name"]),
        is_default=False,
        host_api=str(api["name"]) if api is not None else "",
        sample_rate=_rate_of(info),
        channels=int(info["max_input_channels"]),
    )


def _mme_twin(
    devices: Sequence[Any], hostapis: Sequence[Any], chosen: AudioDevice
) -> AudioDevice | None:
    mme = _api_index(hostapis, MME)
    if mme is None:
        return None
    for candidate in _api_inputs(devices, hostapis, mme):
        if names_match(chosen.name, candidate.name):
            return candidate
    return None


def channel_options(plan: CapturePlan) -> tuple[int, ...]:
    """Mono first; the device's own channel count second, for a device that refuses mono."""
    if plan.channels <= 1:
        return (1,)
    return (1, plan.channels)


def downmix_to_mono(pcm16: bytes, channels: int) -> bytes:
    """Average interleaved int16 frames down to one channel; whole frames only."""
    if channels <= 1:
        return pcm16
    stride = 2 * channels
    usable = len(pcm16) - len(pcm16) % stride
    if usable <= 0:
        return b""
    samples = array.array("h")
    samples.frombytes(pcm16[:usable])
    mono = array.array(
        "h",
        (
            sum(samples[i : i + channels]) // channels
            for i in range(0, len(samples), channels)
        ),
    )
    return mono.tobytes()


def rms_level(pcm16: bytes) -> float:
    """RMS of little-endian int16 samples, normalized so full scale is 1.0."""
    usable = len(pcm16) - len(pcm16) % 2
    samples = array.array("h")
    samples.frombytes(pcm16[:usable])
    if not samples:
        return 0.0
    mean_square = sum(map(operator.mul, samples, samples)) / len(samples)
    return min(1.0, math.sqrt(mean_square) / 32768.0)


def pcm16_to_wav(pcm16: bytes, sample_rate: int) -> bytes:
    """Wrap raw mono int16 PCM in a WAV container at its own rate (spec 7.1)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm16)
    return buffer.getvalue()


def is_rate_refusal(exc: BaseException | None) -> bool:
    """True for PortAudio's Invalid sample rate, which WASAPI shared mode answers with.

    The channel count is not what such an open failed on, so there is nothing to
    gain from retrying the same plan with another one: the next plan, the
    device's MME entry, is what can serve that rate.
    """
    if exc is None or not isinstance(exc, sd.PortAudioError):
        return False
    code = exc.args[1] if len(exc.args) > 1 and isinstance(exc.args[1], int) else None
    if code == PA_INVALID_SAMPLE_RATE:
        return True
    return "invalid sample rate" in _error_text(exc)


def classify_open_error(exc: BaseException) -> MicErrorKind:
    """Map a failure while opening a stream to a MicError kind."""
    if isinstance(exc, ValueError):
        # sounddevice's own device lookup: "No input device matching ..."
        return "missing"
    if isinstance(exc, sd.PortAudioError):
        code = exc.args[1] if len(exc.args) > 1 and isinstance(exc.args[1], int) else None
        if code == PA_INVALID_DEVICE:
            return "missing"
        if code == PA_DEVICE_UNAVAILABLE:
            return "busy"
        text = _error_text(exc)
        if any(pattern in text for pattern in _BUSY_PATTERNS):
            return "busy"
        if any(pattern in text for pattern in _MISSING_PATTERNS):
            return "missing"
    return "unknown"


def _error_text(exc: BaseException) -> str:
    parts = [str(exc.args[0])] if exc.args else []
    if len(exc.args) > 2 and isinstance(exc.args[2], tuple) and len(exc.args[2]) == 3:
        parts.append(str(exc.args[2][2]))
    return " ".join(parts).casefold().replace("_", " ")


def _map_open_error(exc: BaseException, label: str) -> MicError:
    kind = classify_open_error(exc)
    if kind == "missing":
        message = (
            f"Microphone {label} is not available. "
            "Check that it is connected and enabled in Sound settings."
        )
    elif kind == "busy":
        message = (
            f"Microphone {label} is in use by another app or unavailable. "
            "Close the other app or pick another microphone in Sound settings."
        )
    else:
        message = f"Could not open microphone {label}: {exc}"
    return MicError(kind, message)


def _microphone_privacy_denied() -> bool | None:
    """True when a Windows microphone privacy switch is set to Deny, None when unreadable."""
    try:
        import winreg
    except ImportError:
        return None
    checks = (
        (winreg.HKEY_LOCAL_MACHINE, _CONSENT_KEY),
        (winreg.HKEY_CURRENT_USER, _CONSENT_KEY),
        (winreg.HKEY_CURRENT_USER, _CONSENT_KEY + r"\NonPackaged"),
    )
    readable = False
    for root, subkey in checks:
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Value")
        except OSError:
            continue
        readable = True
        if str(value).casefold() == "deny":
            return True
    return False if readable else None


def _blocked_message() -> str:
    if _microphone_privacy_denied():
        return (
            "Windows privacy settings block microphone access for desktop apps. "
            "Allow it under Settings, Privacy and security, Microphone."
        )
    return (
        "The microphone delivers only silence. Check the Windows microphone privacy "
        "setting for desktop apps, and that the microphone is not muted."
    )


class Recorder:
    """Captures one dictation from the microphone at the device's own rate.

    start() opens the stream (or reuses the warm one when keep_open), stop()
    returns the PCM, cancel() discards it. on_level runs on the PortAudio
    callback thread about 20 times per second with an RMS level in 0.0 to 1.0
    while recording; keep it cheap and marshal to the UI thread yourself.

    sample_rate None, the default, captures at whatever the device offers
    (48 kHz through WASAPI) and reports it in `sample_rate` once the stream is
    open; every caller reads the rate back from the recorder rather than
    assuming one. Passing an integer insists on that rate, which on Windows
    means the MME entry of the device, because WASAPI shared mode refuses any
    rate but the device's own.

    A device that will not give a mono stream is opened with its own channel
    count and averaged down in the callback, so stop() always returns mono.

    open_latency_ms is the time start() spent until the stream was running
    (near zero when a warm stream is reused); the pipeline stores it in the
    dictation's StageTimings (spec 12, decision V1-13).

    stop() raises MicError("blocked") when the recording lasted at least
    500 ms and every sample was zero. The open itself succeeds in that case,
    so the condition can only be judged from the data.

    An exception inside the PortAudio callback would abort the stream without
    a trace, so the callback catches everything: the first exception is kept
    in `error` for the pipeline to read, level updates stop, and capture goes
    on with whatever the stream still delivers.
    """

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int | None = None,
        on_level: Callable[[float], None] | None = None,
        keep_open: bool = False,
    ) -> None:
        self.device = device
        self.requested_sample_rate = int(sample_rate) if sample_rate else None
        self.sample_rate = self.requested_sample_rate or FALLBACK_SAMPLE_RATE
        self.keep_open = keep_open
        self.open_latency_ms: float | None = None
        self.error: BaseException | None = None
        self.host_api = ""
        self.device_name = ""
        self.device_match: DeviceMatch = "exact"
        self.fallback_notice: str | None = None
        self.channels = 1
        self._on_level = on_level
        self._blocksize = max(1, self.sample_rate // LEVEL_UPDATES_PER_SECOND)
        self._lock = threading.Lock()
        self._stream: Any = None
        self._recording = False
        self._chunks: list[bytes] = []
        self._frames = 0
        self._signal_seen = False
        self._last_open_error: BaseException | None = None
        self.dropped_blocks = 0

    # State

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def duration_s(self) -> float:
        """Seconds of audio captured so far in the current recording (live)."""
        return self._frames / self.sample_rate

    @property
    def downmixed(self) -> bool:
        return self.channels > 1

    @property
    def silent(self) -> bool:
        """True once at least 500 ms arrived and every sample so far was zero."""
        return (
            not self._signal_seen
            and self._frames * 1000 >= BLOCKED_SILENCE_MS * self.sample_rate
        )

    def snapshot(self, since_bytes: int = 0) -> tuple[bytes, int]:
        """The audio captured so far from `since_bytes` on, and the new offset.

        The live partial passes of spec 6 read the recording while it grows, so this copies
        the blocks under the same lock the callback appends them with and never disturbs
        them. The offset it returns is where the next snapshot can carry on, which is how a
        partial keeps its window bounded on a long dictation. Nothing is written to disk.
        """
        with self._lock:
            data = b"".join(self._chunks)
        start = max(0, int(since_bytes))
        start -= start % 2
        if start >= len(data):
            return b"", len(data)
        return data[start:], len(data)

    def capture_info(self) -> CaptureInfo:
        """What the last open settled on, for the history row and Diagnostics."""
        return CaptureInfo(
            device=self.device_name,
            host_api=self.host_api,
            sample_rate=self.sample_rate,
            channels=self.channels,
            downmixed=self.downmixed,
            dropped_blocks=self.dropped_blocks,
            match=self.device_match,
        )

    # Control

    def start(self) -> None:
        """Open the microphone and begin capturing. Raises MicError."""
        started = time.perf_counter()
        with self._lock:
            if self._recording:
                raise RuntimeError("Recorder.start() called while already recording")
        if self._stream is not None and not getattr(self._stream, "active", False):
            # A warm stream that PortAudio stopped on its own: reopen it.
            self._close_stream()
        if self._stream is None:
            self._open_stream()
        with self._lock:
            self._chunks = []
            self._frames = 0
            self._signal_seen = False
            self.dropped_blocks = 0
            self.error = None
            self._recording = True
        self.open_latency_ms = (time.perf_counter() - started) * 1000.0

    def stop(self) -> bytes:
        """End the recording and return its mono int16 PCM. Raises MicError("blocked")."""
        with self._lock:
            if not self._recording:
                raise RuntimeError("Recorder.stop() called while not recording")
        if self.keep_open:
            with self._lock:
                self._recording = False
        else:
            # Pa_StopStream first: full blocks still queued are delivered to the
            # callback before it returns, so the tail of the utterance is kept.
            self._stop_stream()
            with self._lock:
                self._recording = False
            self._close_stream()
        with self._lock:
            data = b"".join(self._chunks)
            self._chunks = []
            blocked = self.silent
        if blocked:
            raise MicError("blocked", _blocked_message())
        return data

    def cancel(self) -> None:
        """Discard the current recording. Never raises MicError."""
        with self._lock:
            self._recording = False
            self._chunks = []
        if not self.keep_open:
            self._close_stream()

    def close(self) -> None:
        """Release the device, warm stream included."""
        with self._lock:
            self._recording = False
            self._chunks = []
        self._close_stream()

    # Internals

    def _open_stream(self) -> None:
        plans, chosen, match = capture_plans(self.device, self.requested_sample_rate)
        label = f'"{chosen.name}"'
        last: BaseException | None = None
        for plan in plans:
            for channels in channel_options(plan):
                stream = self._try_open(plan, channels)
                if stream is None:
                    last = self._last_open_error
                    if is_rate_refusal(last):
                        break
                    continue
                self._stream = stream
                self._apply_match(chosen, match)
                logger.info(
                    "recording from %s through %s at %d Hz, %d channel(s)",
                    plan.name,
                    plan.host_api or "the default host API",
                    plan.sample_rate,
                    channels,
                )
                return
        if last is None:
            raise MicError(
                "missing", f"Microphone {label} offers no way to record from it."
            )
        raise _map_open_error(last, label) from last

    def _try_open(self, plan: CapturePlan, channels: int) -> Any:
        self.sample_rate = plan.sample_rate
        self.channels = channels
        self.host_api = plan.host_api
        self.device_name = plan.name
        self._blocksize = max(1, plan.sample_rate // LEVEL_UPDATES_PER_SECOND)
        stream = None
        try:
            stream = sd.RawInputStream(
                samplerate=plan.sample_rate,
                blocksize=self._blocksize,
                device=plan.index,
                channels=channels,
                dtype="int16",
                latency="low",
                callback=self._callback,
            )
            stream.start()
        except Exception as exc:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
            self._last_open_error = exc
            logger.info(
                "could not open %s through %s at %d Hz, %d channel(s): %s",
                plan.name,
                plan.host_api or "the default host API",
                plan.sample_rate,
                channels,
                exc,
            )
            logger.debug("the input open failed", exc_info=True)
            return None
        return stream

    def _apply_match(self, chosen: AudioDevice, match: DeviceMatch) -> None:
        self.device_match = match
        if match == "default" and isinstance(self.device, str) and self.device.strip():
            self.fallback_notice = DEVICE_MOVED_NOTICE.format(
                wanted=self.device.strip(), fallback=chosen.name
            )
            logger.warning(
                'the stored microphone "%s" matches no device; recording from "%s" instead',
                self.device,
                chosen.name,
            )
        else:
            self.fallback_notice = None

    def _stop_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
        except Exception:
            logger.warning("stopping the input stream failed", exc_info=True)

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        with contextlib.suppress(Exception):
            stream.abort()
        try:
            stream.close()
        except Exception:
            logger.warning("closing the input stream failed", exc_info=True)

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        # Runs on the PortAudio thread: copy, measure, hand off, nothing else.
        # Nothing may escape: an exception here makes PortAudio abort the stream.
        try:
            if status:
                logger.debug("input stream status: %s", status)
                if getattr(status, "input_overflow", False) and self._recording:
                    self.dropped_blocks += 1
            if not self._recording:
                return
            data = bytes(indata)
            channels = self.channels
            if channels > 1:
                data = downmix_to_mono(data, channels)
                frames = len(data) // 2
            level = rms_level(data)
            with self._lock:
                if not self._recording:
                    return
                self._chunks.append(data)
                self._frames += frames
                if level > 0.0:
                    self._signal_seen = True
            if self._on_level is not None and self.error is None:
                self._on_level(level)
        except Exception as exc:
            if self.error is None:
                self.error = exc
                logger.warning("audio callback failed, level updates stopped", exc_info=True)
