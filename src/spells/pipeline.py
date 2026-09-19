"""The dictation pipeline: recording controller and processing worker (spec 5.1, 6, 12, 13).

Two threads own everything between the hotkey callbacks and the delivered text:

- The recording controller consumes the hotkey callbacks through a queue (they run inside
  the Windows hook procedure and only enqueue, B3-27), captures the target window, matches
  the profile, owns the Recorder and the 10-minute cap timer (which posts to the same queue,
  B3-28), and hands completed recordings to the worker. The engines never block it.
- The processing worker drains the FIFO of completed recordings: engine readiness, ASR,
  the cleanup gate and request, post-processing, delivery, history. It alone owns the
  LanguagePolicyState and the last delivery (spec 5.2, unsynchronised by design).

The UI sees one PipelineEvent per change through a thread-safe callback it marshals to the
Qt thread itself; the pipeline never blocks on it and survives its exceptions. Every event
carries the whole steady state (pill, tray, level, badge, retry availability), so the UI can
always render from the latest one, plus an optional transient notice (an error text or the
copied notice) that the UI shows briefly before returning to the steady state.

Clocks: every timestamp here is time.perf_counter (stage timings, captured_at, the
LastDelivery and the `now` passed to postprocess, B3-53); only history.created_at is
wall-clock time.

Live partials (spec 6, decision B5-57) add a third thread, one per dictation, which
transcribes the recording while it grows and types the draft into the window being
dictated into. Its constants:

- PARTIAL_INTERVAL_S 1.2, one tick, about one spoken clause, and long enough that a
  full-window pass on every model the enable rule admits leaves the engine idle between
  passes rather than queuing behind itself.
- PARTIAL_MIN_AUDIO_S 1.5, how much audio the first pass waits for: detection on a very
  short clip is unreliable (spec 7.2 step 1) and the first two words are not worth a redraw.
- PARTIAL_WINDOW_S 30.0, the most audio one pass carries. Past it the draft so far is
  committed and the next pass starts from fresh audio, so the tenth minute of a dictation
  costs exactly what the first one did.
- PARTIAL_MAX_LATENCY_MS 500, the enable rule: the catalog's estimate is for a 15 s
  dictation, so a full 30 s window costs about 1 s of the 1.2 s tick at the bar, and less
  on every model that actually passes it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol

from spells import asr, cleanup, compose, inject, livetext, postprocess, profiles, quality
from spells.asr import (
    LLAMA_ASR,
    WHISPER_SERVER,
    AbortedRequest,
    AbortHandle,
    LanguagePolicyState,
    LlamaAsrClient,
    SpeechEngineUnavailable,
    SpeechRoute,
    WhisperClient,
    WhisperError,
)
from spells.audio import FALLBACK_SAMPLE_RATE, CaptureInfo, MicError
from spells.cleanup import CleanupError, GateInput, LlamaClient
from spells.config import ConfigStore, Settings
from spells.history import (
    OUTCOME_NO_AUDIO,
    OUTCOME_NOT_WRITTEN,
    AudioPolicy,
    HistoryEntry,
    HistoryStore,
)
from spells.hotkey import HotkeyCallbacks
from spells.inject import DeliveryReport, InjectBackends
from spells.models import (
    Chord,
    ChordMode,
    CleanResult,
    ComposeResult,
    DeliveryOutcome,
    DeliveryResult,
    Engine,
    EngineState,
    LangMode,
    LastDelivery,
    Profile,
    StageTimings,
    TargetContext,
    Transcript,
)

log = logging.getLogger(__name__)

CAPTURE_RING = 20
MAX_RECORDING_S = 600.0
WARNING_S = 540.0
# How long a dictation waits for the ASR engine before "Speech engine unavailable" (spec 6
# step 4 gives no bound; the supervisor's own startup timeout is 180 s, so a cold reload of
# large-v3-turbo plus one backoff cycle fits in this). The wait runs in one-second slices
# so stop() can interrupt it.
ENGINE_WAIT_S = 240.0
READY_WAIT_SLICE_S = 1.0
# After the pipeline asks for a whisper restart, the supervisor applies it on its own thread
# within about 100 ms; the retry waits for the state to leave the serving set first.
RESTART_SETTLE_POLLS = 200
RESTART_SETTLE_POLL_S = 0.01
TIMINGS_RING = 20
JOIN_TIMEOUT_S = 5.0
CHECK_TIMEOUT_S = 10.0
MIN_AUDIO_S = 0.1
PARTIAL_INTERVAL_S = 1.2
PARTIAL_MIN_AUDIO_S = 1.5
PARTIAL_WINDOW_S = 30.0
PARTIAL_MAX_LATENCY_MS = 500
PARTIAL_MAX_LATENCY_CANCELLABLE_MS = 2500
TIMEOUT_LATENCY_FACTOR = 4.0
# How long the writing model may run before the request is given up on (spec 8.5). Composing
# is off the dictation latency path, so this is a safety limit and not a target.
COMPOSE_TIMEOUT_S = compose.COMPOSE_TIMEOUT_MS / 1000.0
WRITER_WAIT_S = 120.0

COPIED_NOTICE = "Copied. Press Ctrl+V to paste."
CLIPBOARD_DETAIL = "the text is on the clipboard"
ENGINE_UNAVAILABLE = "Speech engine unavailable"
TRANSCRIPTION_FAILED = "Transcription failed"
DELIVERY_FAILED = "Delivery failed"
DICTATION_FAILED = "Dictation failed"
# The pill strings follow the approved pill mockup;
# the microphone texts the mockup does not list share its short "Mic is busy" form.
STARTING_ENGINES_TEXT = "Starting engines"
CAP_WARNING_TEXT = "One minute left"
MIC_BLOCKED_TEXT = "Mic is blocked"
COPIED_TEXT = "Copied"
NO_AUDIO_TEXT = "No sound from the microphone. Check the device in Settings."
NO_AUDIO_REASON = "No sound reached Spells from the microphone."
AUDIO_LOST_TEXT = "Some audio was lost"
AUDIO_LOST_NOTIFICATION = (
    "Some audio was lost between the microphone and Spells, so words may be missing."
)
# The notification's "Open settings" button targets (spec 16, mic rows).
SOUND_SETTINGS = "ms-settings:sound"
PRIVACY_SETTINGS = "ms-settings:privacy-microphone"
MIC_PILL_TEXT = {
    "missing": "No mic found",
    "busy": "Mic is busy",
    "blocked": MIC_BLOCKED_TEXT,
    "unknown": "Mic error",
}
# CleanResult reasons of spec 8.3 (output guards); the gate reasons of 8.1 are skips. The
# writing guards of 8.5 share the counters: they are the same kind of event, an answer the
# app refused, and Diagnostics shows one list.
GUARD_REASONS = frozenset(
    {"timeout", "error", "finish_length", "empty", "length_ratio", "preamble", "language_switch"}
) | frozenset(compose.GUARD_REASONS)
SERVING = frozenset({EngineState.READY, EngineState.CPU_FALLBACK})
DELIVERED = frozenset({DeliveryOutcome.PASTED, DeliveryOutcome.TYPED})
COPIED = frozenset({DeliveryOutcome.COPIED_FOCUS_CHANGED, DeliveryOutcome.COPIED_ELEVATED})


# Contracts of the collaborators ------------------------------------------------------------


class EngineControl(Protocol):
    """The slice of EngineSupervisor the pipeline consumes (spec 13, consumer side)."""

    @property
    def whisper_url(self) -> str | None: ...

    @property
    def llama_url(self) -> str | None: ...

    @property
    def cleanup_available(self) -> bool: ...

    @property
    def cpu_cleanup_allowed(self) -> bool: ...

    @property
    def cpu_only(self) -> bool: ...

    @property
    def cleanup_languages(self) -> frozenset[str] | None: ...

    def speech_engines(self) -> tuple[Any, ...]: ...

    def url(self, engine: Any) -> str | None: ...

    def status(self, engine: Any) -> EngineState: ...

    def variant(self, engine: Engine) -> str: ...

    def ensure_ready(self) -> None: ...

    def restart(self, engine: Any) -> None: ...

    def wait_ready(self, engine: Any, timeout_s: float) -> bool: ...


class RecorderLike(Protocol):
    """The slice of audio.Recorder the controller uses."""

    device: Any
    keep_open: bool
    open_latency_ms: float | None
    sample_rate: int

    @property
    def silent(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> bytes: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...

    def snapshot(self, since_bytes: int = 0) -> tuple[bytes, int]: ...


RecorderFactory = Callable[..., RecorderLike]
# schedule(delay_s, callback) -> cancel(); the callback runs on some other thread and must
# only enqueue, which is all the pipeline ever does in one.
Scheduler = Callable[[float, Callable[[], None]], Callable[[], None]]


# Events ------------------------------------------------------------------------------------


class PillState(str, Enum):
    """The steady state of the pill (spec 14.2)."""

    IDLE = "idle"
    RECORDING = "recording"
    LATCHED = "latched"
    PROCESSING = "processing"
    STARTING_ENGINES = "starting_engines"
    WRITING = "writing"


class Notice(str, Enum):
    """A transient pill notice shown briefly over the steady state."""

    ERROR = "error"
    COPIED = "copied"


class TrayState(str, Enum):
    """What the pipeline knows about the tray (spec 14.1).

    Starting, Warning, Error and Idle come from the engine status and the hotkey
    `on_error`; the ui composes them with this value.
    """

    READY = "ready"
    RECORDING = "recording"
    PROCESSING = "processing"


@dataclass(frozen=True)
class PipelineEvent:
    """What the UI needs, and nothing more.

    `pill`, `level`, `busy` (a dictation is processing behind the recording: the spinner
    badge of spec 6 step 3) and `text` (the 9-minute warning or the mic-blocked hint) are
    the steady state. `notice` and `notice_text` are transient: the UI shows them for a
    moment and returns to the steady state. `notification` is a toast; when
    `notification_action` is set it is a `ms-settings:` URI for an "Open settings" button.
    `target_hwnd` is the window captured at press time while a recording is active (the pill
    goes on that window's monitor, spec 14.2), None otherwise or when the capture failed.
    `timings` is set once, on the event that completes a dictation that reached ASR, whether
    or not anything was delivered (a dictation that failed carries a notice instead).
    `live_typing` says that this dictation is typing its draft into the target window as it
    goes (spec 6, B5-57), which the pill names so it is never a surprise. In the WRITING
    state `text` is the instruction the writing model is working on (spec 8.5); the pill adds
    the state and the elapsed seconds to it.
    """

    pill: PillState
    tray: TrayState
    level: float = 0.0
    busy: bool = False
    text: str = ""
    live_typing: bool = False
    target_hwnd: int | None = None
    notice: Notice | None = None
    notice_text: str = ""
    notification: str | None = None
    notification_action: str | None = None
    retry_available: bool = False
    dictation_id: int | None = None
    timings: StageTimings | None = None


@dataclass(frozen=True)
class CompletedRecording:
    """One recording handed from the controller to the worker (spec 6 step 3)."""

    dictation_id: int
    pcm: bytes
    ctx: TargetContext
    profile: Profile
    lang_mode: LangMode
    released_at: float
    sample_rate: int
    press_to_pill_ms: float | None = None
    mic_open_ms: float | None = None
    capped: bool = False
    dropped_blocks: int = 0
    capture: CaptureInfo | None = None
    live: Any = None
    mode: ChordMode = ChordMode.DICTATE


# Internals -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Command:
    """One item of the recording controller's queue."""

    kind: str
    dictation_id: int | None = None
    chord: Chord | None = None
    at: float = 0.0
    settings: Settings | None = None


@dataclass
class _Active:
    """The recording in flight, as the controller knows it."""

    dictation_id: int
    chord: Chord | None
    ctx: TargetContext
    profile: Profile
    lang_mode: LangMode
    press_to_pill_ms: float | None
    mic_open_ms: float | None


@dataclass(frozen=True)
class _Job:
    recording: CompletedRecording
    retry: bool = False


class _Failed(Exception):
    """A stage failed in a way that keeps the PCM for "Retry last dictation"."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


def live_capable(choice: Any) -> bool:
    latency = getattr(choice, "latency_ms", None)
    if choice is None or latency is None:
        return False
    if choice.runtime == LLAMA_ASR:
        return latency <= PARTIAL_MAX_LATENCY_CANCELLABLE_MS
    return latency <= PARTIAL_MAX_LATENCY_MS


class _NotServing(Exception):
    """A partial pass found its engine starting, restarting or unloaded."""


class _Live:
    """The live partial loop of one dictation and the draft it typed (spec 6, B5-57).

    One thread per dictation. It snapshots the growing recording every tick, runs it
    through the same language policy the final pass uses (on a copy of the policy state,
    which it never writes back, so the worker keeps owning it), and reconciles the answer
    into the target window through the LiveSession.

    Only one pass is ever in flight: a tick that finds the previous pass still running
    returns at once, and the schedule then skips the ticks that elapsed. Nothing here
    touches the recording controller, waits for an engine, asks for a restart or lets a
    failure reach the dictation: an error is counted, logged once, and the loop goes on.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        dictation_id: int,
        recorder: RecorderLike,
        session: livetext.LiveSession,
        lang_mode: LangMode,
        last_accepted: str,
    ) -> None:
        self.dictation_id = dictation_id
        self.session = session
        self.lang_mode = lang_mode
        self.state = LanguagePolicyState(last_accepted)
        self.text = ""
        self.passes = 0
        self.inserts = 0
        self.skipped = 0
        self.errors = 0
        self.aborted = False
        self.anchor_text = ""
        self.anchor_bytes = 0
        self._pipeline = pipeline
        self._recorder = recorder
        self._stop = threading.Event()
        self._pass_lock = threading.Lock()
        self._abort_lock = threading.Lock()
        self._abort: AbortHandle | None = None
        self._logged_error = False
        self._erase_on_stop = False
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    @property
    def abort_handle(self) -> AbortHandle | None:
        with self._abort_lock:
            return self._abort

    def start(self) -> None:
        thread = threading.Thread(
            target=self._run, name=f"spells-partial-{self.dictation_id}", daemon=True
        )
        self._thread = thread
        thread.start()

    def stop(self, *, erase: bool = False) -> None:
        """End the loop and close any request in flight. Never waits for the thread."""
        self._erase_on_stop = erase or self._erase_on_stop
        self._stop.set()
        with self._abort_lock:
            handle = self._abort
        if handle is not None:
            self.aborted = True
            handle.abort()
        if self._thread is None and self._erase_on_stop:
            self._erase()

    def _run(self) -> None:
        clock = self._pipeline._clock
        next_at = clock() + PARTIAL_INTERVAL_S
        try:
            while not self._stop.is_set():
                wait = next_at - clock()
                if wait > 0 and self._stop.wait(wait):
                    break
                if self._stop.is_set():
                    break
                self.tick()
                now = clock()
                while next_at <= now:
                    next_at += PARTIAL_INTERVAL_S
        except Exception:
            log.exception("the live partial loop of dictation %d ended", self.dictation_id)
        finally:
            if self._erase_on_stop:
                self._erase()

    def tick(self) -> str:
        """One scheduled pass. "ran", "skipped", "waiting", "off", "error" or "aborted"."""
        if self._stop.is_set() or self.session.stopped:
            return "off"
        if not self._pass_lock.acquire(blocking=False):
            self.skipped += 1
            return "skipped"
        try:
            return self._pass()
        finally:
            self._pass_lock.release()

    def _pass(self) -> str:
        pcm, offset = self._snapshot()
        rate = Pipeline._rate_of(self._recorder)
        seconds = len(pcm) / (2.0 * rate)
        if seconds < PARTIAL_MIN_AUDIO_S:
            return "waiting"
        try:
            transcript = self._pipeline._partial_transcribe(self, pcm, rate)
        except AbortedRequest:
            return "aborted"
        except _NotServing:
            return "off"
        except SpeechEngineUnavailable:
            return "off"
        except Exception:
            self.errors += 1
            if not self._logged_error:
                self._logged_error = True
                log.warning(
                    "a live partial pass failed; the dictation is unaffected", exc_info=True
                )
            return "error"
        self.passes += 1
        if self._stop.is_set():
            return "aborted"
        if transcript is None:
            return "empty"
        text = _join_draft(self.anchor_text, transcript.text)
        if seconds >= PARTIAL_WINDOW_S:
            self.anchor_text = text
            self.anchor_bytes = offset
        if text == self.text:
            return "ran"
        self.text = text
        self.inserts += 1
        self.session.apply(text)
        return "ran"

    def _snapshot(self) -> tuple[bytes, int]:
        reader = getattr(self._recorder, "snapshot", None)
        if not callable(reader):
            return b"", self.anchor_bytes
        try:
            pcm, offset = reader(self.anchor_bytes)
        except Exception:
            log.debug("the recording could not be snapshotted", exc_info=True)
            return b"", self.anchor_bytes
        return bytes(pcm), int(offset)

    def set_abort(self, handle: AbortHandle | None) -> None:
        with self._abort_lock:
            self._abort = handle
        if handle is not None and self._stop.is_set():
            handle.abort()

    def _erase(self) -> None:
        """Take back exactly what this session typed (Esc, spec 6 step 2)."""
        if not self.session.typed_anything:
            return
        self.session.apply("")

    def stats(self) -> dict[str, float]:
        extra = {"partials": float(self.passes), "partial_inserts": float(self.inserts)}
        if self.skipped:
            extra["partial_skipped"] = float(self.skipped)
        if self.errors:
            extra["partial_errors"] = float(self.errors)
        if self.aborted:
            extra["partial_aborted"] = 1.0
        return extra


def _join_draft(head: str, tail: str) -> str:
    head = head.rstrip()
    tail = tail.strip()
    if not head:
        return tail
    if not tail:
        return head
    return f"{head} {tail}"


class _UnavailableLlama(LlamaClient):
    """Stands in when the supervisor exposes no llama URL; the gate normally covers this."""

    def __init__(self) -> None:
        super().__init__("http://127.0.0.1:0")

    def chat(self, system: str, user: str, max_tokens: int) -> tuple[str, str]:
        raise CleanupError("cleanup engine unavailable")

    def health(self) -> bool:
        return False


def _timer_scheduler(delay_s: float, callback: Callable[[], None]) -> Callable[[], None]:
    timer = threading.Timer(delay_s, callback)
    timer.daemon = True
    timer.start()
    return timer.cancel


def _default_recorder(**kwargs: Any) -> RecorderLike:
    from spells.audio import Recorder

    return Recorder(**kwargs)


def _default_capture() -> TargetContext:
    from spells import context

    return context.capture()


def _mode_of(chord: Chord | None) -> ChordMode:
    """The mode of the chord that started a dictation; the main chord and None dictate."""
    mode = getattr(chord, "mode", None)
    return mode if isinstance(mode, ChordMode) else ChordMode.DICTATE


def _default_compose(
    base_url: str, timeout_s: float, cancel: compose.CancelHandle | None
) -> compose.ComposeClient:
    return compose.ComposeClient(base_url, timeout_s, cancel)


def _default_llama(base_url: str, timeout_s: float) -> LlamaClient:
    return LlamaClient(base_url, timeout_s=timeout_s)


class Pipeline:
    """The two pipeline threads and the objects the UI consumes.

    Construct it with every collaborator, then call start(). The production wiring (real
    Recorder, context capture, inject backends and clients) is the default when a factory
    is left out; unit tests pass fakes. `hotkey` is a HotkeyThread or any object with
    `end_recording(dictation_id)`, or that callable itself.

    `on_event` runs on the recording controller and on the processing worker thread. The UI
    must connect it to a queued, non-blocking Qt signal and return at once. A handler that
    blocks, or that calls stop() from the Qt thread while the Qt thread waits for it, would
    deadlock: stop() joins the very thread that is waiting for the handler to return.
    """

    def __init__(
        self,
        *,
        config: ConfigStore,
        engines: EngineControl,
        hotkey: Any,
        history: HistoryStore,
        on_event: Callable[[PipelineEvent], None],
        recorder_factory: RecorderFactory | None = None,
        capture: Callable[[], TargetContext] | None = None,
        inject_backends: InjectBackends | None = None,
        whisper_client_factory: Callable[[str], WhisperClient] | None = None,
        llama_client_factory: Callable[[str, float], LlamaClient] | None = None,
        compose_client_factory: Callable[..., Any] | None = None,
        llama_asr_client_factory: Callable[[str], LlamaAsrClient] | None = None,
        scheduler: Scheduler | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
        sample_rate: int | None = None,
        max_recording_s: float = MAX_RECORDING_S,
        warning_s: float = WARNING_S,
        engine_wait_s: float = ENGINE_WAIT_S,
        selection: Any = None,
        partial_threads: bool = True,
    ) -> None:
        self._config = config
        self._engines = engines
        if hasattr(hotkey, "end_recording"):
            self._end_recording_fn = hotkey.end_recording
        elif callable(hotkey):
            self._end_recording_fn = hotkey
        else:
            raise TypeError("hotkey must have end_recording(dictation_id) or be that callable")
        self._set_writing_fn = getattr(hotkey, "set_writing", None)
        self._history = history
        self._on_event = on_event
        self._recorder_factory = recorder_factory or _default_recorder
        self._capture = capture or _default_capture
        self._backends = inject_backends if inject_backends is not None else inject.DEFAULT_BACKENDS
        self._whisper_factory = whisper_client_factory or WhisperClient
        self._llama_factory = llama_client_factory or _default_llama
        self._compose_factory = compose_client_factory or _default_compose
        self._llama_asr_factory = llama_asr_client_factory or LlamaAsrClient
        self._scheduler = scheduler or _timer_scheduler
        self._clock = clock
        self._sleeper = sleeper
        self._sample_rate = sample_rate
        self._max_recording_s = max_recording_s
        self._warning_s = warning_s
        self._engine_wait_s = engine_wait_s
        self._selection = selection
        self._partial_threads = bool(partial_threads)

        self._controller_queue: queue.Queue[_Command | None] = queue.Queue()
        self._worker_queue: queue.Queue[_Job | None] = queue.Queue()
        self._stopping = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._stopped = False

        # Controller-thread state.
        self._recorder: RecorderLike | None = None
        self._active: _Active | None = None
        self._timer_cancels: list[Callable[[], None]] = []
        self._level_value = 0.0  # written on the PortAudio thread
        self._level_queued = False
        self._device_notice_shown = False
        self._live: _Live | None = None

        # Shared UI state, under _state_lock.
        self._state_lock = threading.Lock()
        self._rec_pill: PillState | None = None
        self._rec_text = ""
        self._level = 0.0
        self._target_hwnd: int | None = None
        self._live_typing = False
        self._writing = False
        self._writing_text = ""
        self._compose_cancels: dict[int, compose.CancelHandle] = {}
        self._waiting_engines = False
        self._pending = 0
        self._retry: CompletedRecording | None = None
        self._guard_counts: dict[str, int] = {}
        self._ring: deque[StageTimings] = deque(maxlen=TIMINGS_RING)
        self._captures: deque[CaptureInfo] = deque(maxlen=CAPTURE_RING)

        # Worker-thread state (spec 5.2: unsynchronised, one owner).
        enabled = config.settings.general.enabled_languages
        self._lang_state = LanguagePolicyState(enabled[0] if enabled else "en")
        self._last_delivery: LastDelivery | None = None

        self._unsubscribe = config.subscribe(self._on_settings)

    # Public API ----------------------------------------------------------------------------

    def hotkey_callbacks(self) -> HotkeyCallbacks:
        """The five hotkey callbacks; each only enqueues and returns (spec 5.1)."""
        return HotkeyCallbacks(
            pressed=lambda chord, did: self._enqueue("pressed", did, chord),
            released=lambda chord, did: self._enqueue("released", did, chord),
            latched=lambda chord, did: self._enqueue("latched", did, chord),
            discarded=lambda did: self._enqueue("discarded", did),
            cancelled=lambda did: self._enqueue("cancelled", did),
        )

    def start(self) -> None:
        """Start both threads; the recorder is created (and warmed) on the controller."""
        if self._started or self._stopped:
            return
        self._started = True
        self._controller_queue.put(_Command("settings", settings=self._config.settings))
        self._threads = [
            threading.Thread(target=self._run_controller, name="spells-recording", daemon=True),
            threading.Thread(target=self._run_worker, name="spells-processing", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: float = JOIN_TIMEOUT_S) -> None:
        """Stop both threads, drop pending work and release the microphone. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        self._stopping.set()
        try:
            self._unsubscribe()
        except Exception:
            log.exception("could not unsubscribe from the settings store")
        self._controller_queue.put(None)
        self._worker_queue.put(None)
        me = threading.current_thread()
        for thread in self._threads:
            # An on_event handler that quits runs on one of these threads; never join it.
            if thread is not me and thread.is_alive():
                thread.join(timeout)
        self._threads = []
        self._cancel_timers()
        self._take_live(erase=False)
        self._close_recorder()

    @property
    def running(self) -> bool:
        return self._started and not self._stopped

    def retry_last(self) -> bool:
        """Queue "Retry last dictation" (spec 13). Its result always goes to the clipboard.

        Returns False when no PCM is held. The PCM stays until the attempt completes,
        whatever its outcome, or the next dictation starts.
        """
        with self._state_lock:
            recording = self._retry
            if recording is None:
                return False
            self._pending += 1
        self._worker_queue.put(_Job(recording, retry=True))
        self._emit(recording.dictation_id)
        return True

    @property
    def retry_available(self) -> bool:
        with self._state_lock:
            return self._retry is not None

    def recent_timings(self) -> list[StageTimings]:
        """The last 20 completed dictations' timings, oldest first (spec 6 step 9)."""
        with self._state_lock:
            return list(self._ring)

    def guard_counts(self) -> dict[str, int]:
        """Cleanup guard rejections per CleanResult.reason since start (spec 8.3)."""
        with self._state_lock:
            return dict(self._guard_counts)

    def recent_captures(self) -> list[CaptureInfo]:
        """How the last 20 dictations were captured, oldest first (spec 14.4 Diagnostics)."""
        with self._state_lock:
            return list(self._captures)

    # Test hooks: run the loop bodies on the calling thread when no thread is running.

    def drain_controller(self) -> int:
        """Handle every queued controller command now; returns how many ran."""
        handled = 0
        while True:
            try:
                command = self._controller_queue.get_nowait()
            except queue.Empty:
                return handled
            if command is None:
                self._controller_exit()
                return handled
            self._handle_controller(command)
            handled += 1

    def drain_worker(self) -> int:
        """Process every queued recording now; returns how many ran."""
        handled = 0
        while True:
            try:
                job = self._worker_queue.get_nowait()
            except queue.Empty:
                return handled
            if job is None:
                return handled
            self._handle_job(job)
            handled += 1

    # Enqueueing (hook thread, timer threads, PortAudio thread, settings updaters) ----------

    def _enqueue(self, kind: str, dictation_id: int | None = None, chord: Chord | None = None):
        self._controller_queue.put(_Command(kind, dictation_id, chord, self._clock()))

    def _on_settings(self, settings: Settings) -> None:
        self._controller_queue.put(_Command("settings", settings=settings))

    def _on_level(self, level: float) -> None:
        # PortAudio thread, about 20 times a second: store, and queue at most one event.
        self._level_value = float(level)
        if not self._level_queued:
            self._level_queued = True
            self._controller_queue.put(_Command("level"))

    # Recording controller ----------------------------------------------------------------------

    def _run_controller(self) -> None:
        try:
            while not self._stopping.is_set():
                command = self._controller_queue.get()
                if command is None or self._stopping.is_set():
                    break
                self._handle_controller(command)
        finally:
            self._controller_exit()

    def _controller_exit(self) -> None:
        self._cancel_timers()
        if self._active is not None:
            self._discard(self._active)
        self._take_live(erase=False)
        self._close_recorder()

    def _handle_controller(self, command: _Command) -> None:
        try:
            self._dispatch(command)
        except Exception:
            log.exception("recording controller: %s failed", command.kind)

    def _dispatch(self, command: _Command) -> None:
        kind = command.kind
        if kind == "cancelled" and self._cancel_writing(command.dictation_id):
            return
        if kind == "pressed":
            self._on_pressed(command)
            return
        if kind == "settings":
            self._apply_settings(command.settings)
            return
        if kind == "level":
            self._on_level_command()
            return
        active = self._active
        if active is None or command.dictation_id != active.dictation_id:
            # A callback for a dictation the controller already ended (B3-51), or a timer
            # of one that ended by hotkey: applied up to one tick late, so expected.
            log.debug("ignoring %s for dictation %s", kind, command.dictation_id)
            return
        if kind == "released":
            self._finish(active, command.at, capped=False)
        elif kind == "cap":
            log.info("dictation %d reached the %.0f s cap", active.dictation_id, self._max_recording_s)
            self._finish(active, command.at, capped=True)
        elif kind == "latched":
            with self._state_lock:
                self._rec_pill = PillState.LATCHED
            self._emit(active.dictation_id)
        elif kind == "warn":
            with self._state_lock:
                self._rec_text = CAP_WARNING_TEXT
            self._emit(active.dictation_id)
        elif kind in ("discarded", "cancelled"):
            self._discard(active)
        else:
            log.warning("recording controller: unknown command %r", kind)

    def _on_pressed(self, command: _Command) -> None:
        dictation_id = command.dictation_id
        assert dictation_id is not None
        if self._active is not None:
            log.warning(
                "press for dictation %d while %d is still recording; discarding the old one",
                dictation_id,
                self._active.dictation_id,
            )
            self._discard(self._active)
        with self._state_lock:
            self._retry = None  # the next dictation started: the retry PCM is released
        try:
            # Counts as activity and reloads unloaded engines now, so a reload overlaps the
            # recording (spec 13); the worker's readiness wait is the safety net.
            self._engines.ensure_ready()
        except Exception:
            log.exception("engines.ensure_ready failed")
        settings = self._config.settings
        try:
            ctx = self._capture()
        except Exception:
            log.exception("target capture failed; using an empty context")
            ctx = TargetContext(hwnd=0, process="", title="", captured_at=self._clock())
        profile = profiles.match(ctx, settings)
        lang_mode = self._lang_mode(command.chord, settings)
        try:
            recorder = self._ensure_recorder(settings, warm=False)
            recorder.start()
        except MicError as exc:
            self._mic_error(dictation_id, exc)
            self._end_recording(dictation_id)
            return
        except Exception as exc:
            log.exception("microphone open failed")
            self._mic_error(dictation_id, MicError("unknown", f"Could not open the microphone: {exc}"))
            self._end_recording(dictation_id)
            return
        pill_at = self._clock()
        self._active = _Active(
            dictation_id=dictation_id,
            chord=command.chord,
            ctx=ctx,
            profile=profile,
            lang_mode=lang_mode,
            press_to_pill_ms=(pill_at - command.at) * 1000.0,
            mic_open_ms=recorder.open_latency_ms,
        )
        mode = _mode_of(command.chord)
        live = (
            None
            if mode.writes
            else self._start_live(dictation_id, recorder, ctx, profile, lang_mode, settings)
        )
        with self._state_lock:
            self._rec_pill = PillState.RECORDING
            self._rec_text = ""
            self._level = 0.0
            self._target_hwnd = ctx.hwnd or None
            self._live_typing = live is not None
        self._level_value = 0.0
        self._emit(dictation_id, **self._settle_device(recorder))
        self._arm_timers(dictation_id)

    def _finish(self, active: _Active, released_at: float, *, capped: bool) -> None:
        dictation_id = active.dictation_id
        self._active = None
        self._cancel_timers()
        live = self._take_live(erase=False)
        recorder = self._recorder
        try:
            pcm = recorder.stop() if recorder is not None else b""
        except MicError as exc:
            self._erase_live(live)
            self._clear_recording()
            self._mic_error(dictation_id, exc)
            self._end_recording(dictation_id)
            return
        except Exception:
            log.exception("Recorder.stop failed; dropping dictation %d", dictation_id)
            self._erase_live(live)
            self._clear_recording()
            self._end_recording(dictation_id)
            self._emit(dictation_id)
            return
        self._clear_recording()
        recording = CompletedRecording(
            dictation_id=dictation_id,
            pcm=pcm,
            ctx=active.ctx,
            profile=active.profile,
            lang_mode=active.lang_mode,
            released_at=released_at,
            press_to_pill_ms=active.press_to_pill_ms,
            mic_open_ms=active.mic_open_ms,
            capped=capped,
            dropped_blocks=int(getattr(recorder, "dropped_blocks", 0) or 0),
            sample_rate=self._rate_of(recorder),
            capture=self._capture_of(recorder),
            live=live,
            mode=_mode_of(active.chord),
        )
        if recording.capture is not None:
            with self._state_lock:
                self._captures.append(recording.capture)
        with self._state_lock:
            self._pending += 1
        self._worker_queue.put(_Job(recording))
        # A no-op after a hotkey-driven end; what returns the state machine to idle after
        # the cap (spec 6 step 2).
        self._end_recording(dictation_id)
        self._emit(dictation_id)

    def _discard(self, active: _Active) -> None:
        dictation_id = active.dictation_id
        self._active = None
        self._cancel_timers()
        self._take_live(erase=True)
        recorder = self._recorder
        if recorder is not None:
            try:
                recorder.cancel()
            except Exception:
                log.exception("Recorder.cancel failed")
        self._clear_recording()
        self._end_recording(dictation_id)
        self._emit(dictation_id)

    def _on_level_command(self) -> None:
        self._level_queued = False
        active = self._active
        if active is None:
            return
        recorder = self._recorder
        with self._state_lock:
            self._level = self._level_value
            if recorder is not None and getattr(recorder, "silent", False):
                self._rec_text = MIC_BLOCKED_TEXT
        self._emit(active.dictation_id)

    def _apply_settings(self, settings: Settings | None) -> None:
        if settings is None or self._active is not None:
            return  # a change during a recording is picked up at the next press
        try:
            self._ensure_recorder(settings, warm=True)
        except Exception:
            log.exception("could not apply the microphone settings")

    def _ensure_recorder(self, settings: Settings, *, warm: bool) -> RecorderLike:
        """The recorder for the current mic settings; reopened when they changed (B3-18)."""
        device = settings.general.mic_device
        keep_open = settings.general.keep_mic_warm
        recorder = self._recorder
        if recorder is not None and (recorder.device != device or recorder.keep_open != keep_open):
            log.info("microphone settings changed; reopening the recorder")
            self._close_recorder()
            recorder = None
        if recorder is None:
            recorder = self._recorder_factory(
                device=device,
                sample_rate=self._sample_rate,
                on_level=self._on_level,
                keep_open=keep_open,
            )
            self._recorder = recorder
            if keep_open and warm:
                self._warm(recorder)
        return recorder

    def _settle_device(self, recorder: RecorderLike) -> dict[str, Any]:
        """Keep the stored device name in step with the device the recorder opened.

        A settings file written while capture went through MME holds a name cut
        at 31 characters, so the first recording that matches it by prefix
        rewrites it to the full name WASAPI reports. A name that matches no
        device at all records from the system default instead of failing, and
        says so once per run (spec 16).
        """
        match = str(getattr(recorder, "device_match", "") or "")
        name = str(getattr(recorder, "device_name", "") or "")
        wanted = recorder.device
        if match == "prefix" and name and isinstance(wanted, str) and name != wanted:
            log.info('the stored microphone "%s" is now called "%s"', wanted, name)
            self._rename_device(name)
            recorder.device = name
            return {}
        notice = getattr(recorder, "fallback_notice", None)
        if match == "default" and notice and not self._device_notice_shown:
            self._device_notice_shown = True
            return {"notification": str(notice), "notification_action": SOUND_SETTINGS}
        return {}

    def _rename_device(self, name: str) -> None:
        def mutate(settings: Settings) -> Settings:
            return replace(settings, general=replace(settings.general, mic_device=name))

        try:
            self._config.update(mutate)
        except Exception:
            log.exception("could not store the full microphone name")

    @staticmethod
    def _rate_of(recorder: RecorderLike | None) -> int:
        rate = int(getattr(recorder, "sample_rate", 0) or 0)
        return rate if rate > 0 else FALLBACK_SAMPLE_RATE

    @staticmethod
    def _capture_of(recorder: RecorderLike | None) -> CaptureInfo | None:
        reader = getattr(recorder, "capture_info", None)
        if not callable(reader):
            return None
        try:
            info = reader()
        except Exception:
            log.exception("capture_info failed")
            return None
        return info if isinstance(info, CaptureInfo) else None

    @staticmethod
    def _warm(recorder: RecorderLike) -> None:
        """Open the warm stream now, so the first press finds it open (keep mic warm)."""
        try:
            recorder.start()
            recorder.cancel()
        except MicError as exc:
            log.warning("could not warm the microphone: %s", exc.message)
        except Exception:
            log.exception("could not warm the microphone")

    def _close_recorder(self) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return
        try:
            recorder.close()
        except Exception:
            log.exception("Recorder.close failed")

    def _clear_recording(self) -> None:
        with self._state_lock:
            self._rec_pill = None
            self._rec_text = ""
            self._level = 0.0
            self._target_hwnd = None
            self._live_typing = False

    # Live partials (spec 6, B5-57) ---------------------------------------------------------

    def set_selection(self, selection: Any) -> None:
        """The model selection the enable rule reads; the app calls it when it changes."""
        self._selection = selection

    def live_enabled(self, settings: Settings, profile: Profile, lang_mode: LangMode) -> bool:
        """Whether this dictation may type its draft as it goes (spec 6, B5-57).

        Three questions, all of which must say yes: the setting, the profile, and whether
        every speech model that could serve this dictation is fast enough on this hardware
        for a pass to fit comfortably inside a tick.
        """
        general = settings.general
        if not livetext.live_typing_allowed(
            profile.name,
            enabled=general.live_text,
            everywhere=general.live_text_everywhere,
        ):
            return False
        return self._fast_enough(lang_mode, general.enabled_languages)

    def _fast_enough(self, lang_mode: LangMode, enabled: Sequence[str]) -> bool:
        selection = self._selection
        if selection is None or not getattr(selection, "asr", ()):
            return False
        if lang_mode.code:
            return live_capable(selection.asr_for(lang_mode.code))
        codes = list(dict.fromkeys(enabled))
        if not codes:
            return False
        if len(selection.asr) == 1:
            return live_capable(selection.asr[0])
        return live_capable(selection.asr[asr.first_auto_route(selection.asr, codes)])

    def _integrated_gpu(self) -> bool:
        return bool(getattr(self._selection, "integrated", False))

    def _route_live_capable(self, route: SpeechRoute) -> bool:
        selection = self._selection
        for choice in getattr(selection, "asr", ()) or ():
            if choice.runtime == route.runtime and set(choice.languages) == set(route.languages):
                return live_capable(choice)
        return True

    def _timeout_base_s(self) -> float:
        latencies = [
            choice.latency_ms
            for choice in getattr(self._selection, "asr", ()) or ()
            if choice.runtime == WHISPER_SERVER and choice.latency_ms
        ]
        if not latencies:
            return asr.WHISPER_TIMEOUT_BASE_S
        return max(asr.WHISPER_TIMEOUT_BASE_S, TIMEOUT_LATENCY_FACTOR * max(latencies) / 1000.0)

    def _start_live(
        self,
        dictation_id: int,
        recorder: RecorderLike,
        ctx: TargetContext,
        profile: Profile,
        lang_mode: LangMode,
        settings: Settings,
    ) -> _Live | None:
        self._take_live(erase=False)
        if not ctx.hwnd:
            return None
        try:
            if not self.live_enabled(settings, profile, lang_mode):
                return None
        except Exception:
            log.exception("the live insertion rule could not be evaluated")
            return None
        session = livetext.LiveSession(ctx, self._backends)
        live = _Live(self, dictation_id, recorder, session, lang_mode, self._lang_state.last_accepted)
        self._live = live
        if self._partial_threads:
            live.start()
        return live

    def _take_live(self, *, erase: bool) -> _Live | None:
        """Stop the live loop of the dictation that is ending and hand it to the caller."""
        live, self._live = self._live, None
        if live is None:
            return None
        try:
            live.stop(erase=erase)
        except Exception:
            log.exception("the live partial loop could not be stopped")
        return live

    def _erase_live(self, live: _Live | None) -> None:
        if live is None:
            return
        try:
            live.session.apply("")
        except Exception:
            log.exception("the live draft could not be taken back")

    def run_partial_tick(self) -> str:
        """Run one partial tick on the calling thread (test hook, mirrors the loop body)."""
        live = self._live
        return "off" if live is None else live.tick()

    @property
    def live(self) -> _Live | None:
        return self._live

    def _mic_error(self, dictation_id: int, exc: MicError) -> None:
        kind = getattr(exc, "kind", "unknown")
        self._emit(
            dictation_id,
            notice=Notice.ERROR,
            notice_text=MIC_PILL_TEXT.get(kind, MIC_PILL_TEXT["unknown"]),
            notification=getattr(exc, "message", None) or str(exc),
            notification_action=PRIVACY_SETTINGS if kind == "blocked" else SOUND_SETTINGS,
        )

    def _end_recording(self, dictation_id: int) -> None:
        try:
            self._end_recording_fn(dictation_id)
        except Exception:
            log.exception("hotkey.end_recording(%d) failed", dictation_id)

    def _arm_timers(self, dictation_id: int) -> None:
        self._cancel_timers()
        cancels: list[Callable[[], None]] = []
        if 0 < self._warning_s < self._max_recording_s:
            cancels.append(
                self._scheduler(self._warning_s, lambda: self._enqueue("warn", dictation_id))
            )
        if self._max_recording_s > 0:
            cancels.append(
                self._scheduler(self._max_recording_s, lambda: self._enqueue("cap", dictation_id))
            )
        self._timer_cancels = cancels

    def _cancel_timers(self) -> None:
        cancels, self._timer_cancels = self._timer_cancels, []
        for cancel in cancels:
            try:
                cancel()
            except Exception:
                log.exception("timer cancel failed")

    @staticmethod
    def _lang_mode(chord: Chord | None, settings: Settings) -> LangMode:
        if chord is not None and chord.language:
            return LangMode("forced", chord.language)
        mode = settings.general.language_mode
        if mode == "auto":
            return LangMode("auto")
        return LangMode("locked", mode)

    # Processing worker ----------------------------------------------------------------------

    def _run_worker(self) -> None:
        while not self._stopping.is_set():
            job = self._worker_queue.get()
            if job is None or self._stopping.is_set():
                break
            self._handle_job(job)

    def _handle_job(self, job: _Job) -> None:
        dictation_id = job.recording.dictation_id
        try:
            final = self._process(job)
        except Exception:
            log.exception("processing worker: dictation %d failed", dictation_id)
            final = {"notice": Notice.ERROR, "notice_text": DICTATION_FAILED}
        with self._state_lock:
            self._pending = max(0, self._pending - 1)
            self._waiting_engines = False
        self._emit(dictation_id, **final)

    def _process(self, job: _Job) -> dict[str, Any]:
        """Run one dictation through ASR, cleanup, post-processing, delivery and history.

        Returns the keyword arguments of the completing event.
        """
        recording = job.recording
        dictation_id = recording.dictation_id
        started = self._clock()
        self._emit(dictation_id)
        settings = self._config.settings
        if len(recording.pcm) < self._min_audio_bytes(recording):
            return self._no_audio(recording, job, settings)
        metrics: list[quality.AsrMetrics] = []
        try:
            transcript = self._transcribe(recording, settings, metrics.append)
        except _Failed as exc:
            with self._state_lock:
                # The PCM stays for a manual retry; a failed retry attempt releases it.
                self._retry = None if job.retry else recording
            log.warning("dictation %d: %s", dictation_id, exc.text)
            return {"notice": Notice.ERROR, "notice_text": exc.text}
        if job.retry:
            with self._state_lock:
                self._retry = None
        transcript_at = self._clock()
        reference = started if job.retry else recording.released_at
        if transcript is None:
            log.info("dictation %d: nothing to deliver", dictation_id)
            timings = self._timings_for(
                recording, None, reference, transcript_at, None, None, job.retry
            )
            self._remember_timings(timings, job.retry)
            return {"timings": timings}
        profile = self._with_tone(recording.profile, settings)
        if recording.mode.writes:
            return self._write(job, transcript, profile, settings, reference, transcript_at, metrics)
        result = self._clean(transcript, profile, settings)
        cleaned_at = self._clock()
        text = postprocess.apply(
            result.text, profile, recording.ctx, self._last_delivery, settings.vocabulary, self._clock()
        )
        delivery_started = self._clock()
        if job.retry:
            report = self._copy_to_clipboard(text)
        elif self._typed_live(recording):
            report = self._deliver_live(recording, text, delivery_started)
        else:
            report = inject.deliver(
                text,
                recording.ctx,
                profile.delivery,
                backends=self._backends,
                sleeper=self._sleeper,
                clock=self._clock,
            )
        self._note_delivery(recording, text, report, delivery_started)
        timings = self._timings_for(
            recording, report, reference, transcript_at, cleaned_at, delivery_started, job.retry
        )
        self._remember_timings(timings, job.retry)
        signals = self._signals_for(recording, transcript, metrics, settings)
        self._write_history(
            recording, transcript, result, text, report, timings, signals, settings
        )
        final = self._outcome_ui(report)
        if recording.dropped_blocks:
            log.warning(
                "dictation %d: the sound card dropped %d input blocks; words may be missing",
                dictation_id,
                recording.dropped_blocks,
            )
            if not final.get("notice"):
                final["notice"] = Notice.ERROR
                final["notice_text"] = AUDIO_LOST_TEXT
            if not final.get("notification"):
                final["notification"] = AUDIO_LOST_NOTIFICATION
        final["timings"] = timings
        return final

    # Compose and edit (spec 8.5) ------------------------------------------------------------

    def _write(
        self,
        job: _Job,
        transcript: Transcript,
        profile: Profile,
        settings: Settings,
        reference: float,
        transcript_at: float,
        metrics: list[quality.AsrMetrics],
    ) -> dict[str, Any]:
        """The instruction path: no cleanup, one writing request, then the usual delivery.

        Nothing reaches the cursor until the answer is ready and every guard has passed. A
        rejection delivers nothing at all and says why on the pill, because the only other
        thing on hand is the raw instruction.
        """
        recording = job.recording
        instruction = transcript.text
        selection, note = self._selection_for(recording)
        mode = ChordMode.EDIT if selection else ChordMode.COMPOSE
        result = self._compose(recording.dictation_id, instruction, selection, profile, mode)
        written_at = self._clock()
        if result.reason in GUARD_REASONS:
            with self._state_lock:
                self._guard_counts[result.reason] = self._guard_counts.get(result.reason, 0) + 1
        if not result.ok:
            timings = self._timings_for(
                recording, None, reference, transcript_at, written_at, None, job.retry
            )
            self._remember_timings(timings, job.retry)
            signals = self._signals_for(recording, transcript, metrics, settings)
            self._write_compose_history(
                recording, transcript, result, None, "", timings, signals, settings
            )
            final: dict[str, Any] = {"timings": timings}
            if result.reason != "cancelled":
                final["notice"] = Notice.ERROR
                final["notice_text"] = compose.rejection_text(result.reason)
            return final
        text = postprocess.apply(
            result.text,
            profile,
            recording.ctx,
            None,
            settings.vocabulary,
            self._clock(),
        )
        delivery_started = self._clock()
        if job.retry:
            report = self._copy_to_clipboard(text)
        else:
            report = inject.deliver(
                text,
                recording.ctx,
                profile.delivery,
                backends=self._backends,
                sleeper=self._sleeper,
                clock=self._clock,
            )
        self._note_delivery(recording, text, report, delivery_started)
        timings = self._timings_for(
            recording, report, reference, transcript_at, written_at, delivery_started, job.retry
        )
        self._remember_timings(timings, job.retry)
        signals = self._signals_for(recording, transcript, metrics, settings)
        self._write_compose_history(
            recording, transcript, result, report, text, timings, signals, settings
        )
        final = self._outcome_ui(report)
        if note and not final.get("notice"):
            final["notice"] = Notice.ERROR
            final["notice_text"] = note
        final["timings"] = timings
        return final

    def _selection_for(self, recording: CompletedRecording) -> tuple[str, str]:
        """(selection, pill note) for an edit dictation; two empty strings for a compose one.

        Ctrl+C only goes out while the window captured at press still owns the foreground,
        so an edit whose focus moved never copies out of a stranger's window. Nothing
        selected, a moved focus and a failed copy all read the same from here: compose
        instead, and say so on the pill.
        """
        if recording.mode is not ChordMode.EDIT:
            return "", ""
        held = inject.target_holds_focus(recording.ctx, self._backends)
        if held:
            log.info("edit mode: no selection copied because %s", held)
            return "", compose.NOTHING_SELECTED_TEXT
        picked = compose.read_selection(self._backends, sleeper=self._sleeper, clock=self._clock)
        if not picked.copied or not picked.text.strip():
            return "", compose.NOTHING_SELECTED_TEXT
        return picked.text, ""

    def _compose(
        self,
        dictation_id: int,
        instruction: str,
        selection: str,
        profile: Profile,
        mode: ChordMode,
    ) -> ComposeResult:
        cancel = compose.CancelHandle()
        self._begin_writing(dictation_id, instruction, cancel)
        try:
            url = self._writer_url(cancel)
            client = None if url is None else self._compose_factory(url, COMPOSE_TIMEOUT_S, cancel)
            return compose.compose(
                instruction, client, selection=selection, tone=profile.tone, mode=mode
            )
        finally:
            self._end_writing(dictation_id)

    def _writer_url(self, cancel: compose.CancelHandle) -> str | None:
        engines = self._engines
        writer = getattr(engines, "writer_engine", None)
        if writer is None:
            return engines.llama_url
        url = engines.url(writer)
        if url is not None:
            return url
        engines.ensure_ready()
        deadline = self._clock() + WRITER_WAIT_S
        while url is None and self._clock() < deadline and not cancel.cancelled:
            if self._stopping.is_set():
                break
            engines.wait_ready(writer, READY_WAIT_SLICE_S)
            url = engines.url(writer)
        return url

    def _begin_writing(
        self, dictation_id: int, instruction: str, cancel: compose.CancelHandle
    ) -> None:
        with self._state_lock:
            self._compose_cancels[dictation_id] = cancel
            self._writing = True
            self._writing_text = instruction
        self._arm_escape(dictation_id)
        self._emit(dictation_id)

    def _end_writing(self, dictation_id: int) -> None:
        with self._state_lock:
            self._compose_cancels.pop(dictation_id, None)
            self._writing = bool(self._compose_cancels)
            if not self._writing:
                self._writing_text = ""
        self._arm_escape(None)

    def _arm_escape(self, dictation_id: int | None) -> None:
        """Tell the hotkey layer which composition Esc would cancel (spec 8.5)."""
        if self._set_writing_fn is None:
            return
        try:
            self._set_writing_fn(dictation_id)
        except Exception:
            log.exception("hotkey.set_writing failed")

    def _cancel_writing(self, dictation_id: int | None) -> bool:
        """Close the writing request Esc just cancelled; False when there is none."""
        if dictation_id is None:
            return False
        with self._state_lock:
            cancel = self._compose_cancels.get(dictation_id)
        if cancel is None:
            return False
        log.info("dictation %d: the writing request was cancelled", dictation_id)
        cancel.cancel()
        return True

    def _write_compose_history(
        self,
        recording: CompletedRecording,
        transcript: Transcript,
        result: ComposeResult,
        report: DeliveryReport | None,
        text: str,
        timings: StageTimings,
        signals: quality.QualitySignals,
        settings: Settings,
    ) -> None:
        """One row for a written dictation (spec 8.5).

        The speech signals still describe the instruction, which is a real dictation and is
        scored as one; the written text is not scored, because nothing measured how well the
        model wrote. cleanup_reason carries "compose" when the text was delivered and the
        rejection reason when it was not.
        """
        verdict = quality.assess(signals)
        entry = HistoryEntry(
            id=None,
            created_at=time.time(),
            raw_text=transcript.text,
            cleaned_text="",
            delivered_text=text,
            app_process=recording.ctx.process,
            app_title=recording.ctx.title,
            language=transcript.language,
            used_llm=result.ok,
            cleanup_reason="compose" if result.ok else result.reason,
            outcome=report.outcome.value if report is not None else OUTCOME_NOT_WRITTEN,
            timings=timings,
            asr_engine=transcript.engine or Engine.WHISPER.value,
            cleanup_engine=Engine.LLAMA.value if result.ok else "",
            signals=signals,
            quality_label=verdict.label,
            quality_reason=verdict.reason,
            mode=result.mode.value,
            instruction=result.instruction,
            selection_chars=result.selection_chars,
        )
        self._store_history(entry, recording, settings)

    @staticmethod
    def _min_audio_bytes(recording: CompletedRecording) -> int:
        return int(MIN_AUDIO_S * recording.sample_rate) * 2

    def _no_audio(
        self, recording: CompletedRecording, job: _Job, settings: Settings
    ) -> dict[str, Any]:
        """A recording with no sound in it never reaches the speech engine (B5-42).

        A wireless headset that drops off the air still opens as a device and delivers no
        frames; the empty WAV made whisper-server answer HTTP 400, which reached the user as
        "Transcription failed" with nothing about the real cause. There is nothing to retry
        either, so the PCM is dropped rather than kept for the tray's retry entry, and the
        row is recorded with an outcome of its own instead of as a failed transcription.
        """
        device = settings.general.mic_device
        seconds = len(recording.pcm) / (2.0 * recording.sample_rate)
        log.warning(
            "dictation %d: the microphone (%s) delivered %.0f ms of audio; not transcribing",
            recording.dictation_id,
            device if device not in (None, "") else "default device",
            seconds * 1000.0,
        )
        timings = StageTimings(
            press_to_pill_ms=recording.press_to_pill_ms,
            mic_open_ms=recording.mic_open_ms,
            extra=self._audio_extra(recording, {"no_audio": 1.0}),
        )
        self._remember_timings(timings, job.retry)
        entry = HistoryEntry(
            id=None,
            created_at=time.time(),
            raw_text="",
            cleaned_text="",
            delivered_text="",
            app_process=recording.ctx.process,
            app_title=recording.ctx.title,
            language="",
            used_llm=False,
            cleanup_reason="",
            outcome=OUTCOME_NO_AUDIO,
            timings=timings,
            signals=quality.signals(
                "", audio_s=seconds, dropped_blocks=recording.dropped_blocks
            ),
            quality_label=quality.POOR,
            quality_reason=NO_AUDIO_REASON,
        )
        self._store_history(entry, recording, settings, with_audio=False)
        return {
            "notice": Notice.ERROR,
            "notice_text": NO_AUDIO_TEXT,
            "notification": NO_AUDIO_TEXT,
            "notification_action": SOUND_SETTINGS,
            "timings": timings,
        }

    def _audio_extra(self, recording: CompletedRecording, extra: dict[str, float]) -> dict:
        extra["audio_s"] = len(recording.pcm) / (2.0 * recording.sample_rate)
        if recording.dropped_blocks:
            extra["dropped_blocks"] = float(recording.dropped_blocks)
        return extra

    def _whisper_ready(self, dictation_id: int) -> str | None:
        """The whisper URL once the engine serves, or None when it will not (spec 6 step 4)."""
        return self._serving_url(dictation_id, Engine.WHISPER, lambda: self._engines.whisper_url)

    def _serving_url(
        self, dictation_id: int, engine: Any, url: Callable[[], str | None]
    ) -> str | None:
        engines = self._engines
        status = engines.status(engine)
        if status not in SERVING:
            if status is EngineState.FAILED:
                return None
            with self._state_lock:
                self._waiting_engines = True
            self._emit(dictation_id)
            engines.ensure_ready()
            ready = self._wait_ready(engine)
            with self._state_lock:
                self._waiting_engines = False
            self._emit(dictation_id)
            if not ready:
                return None
        return url()

    def _wait_ready(self, engine: Any) -> bool:
        """engines.wait_ready in one-second slices, so stop() interrupts a pending wait."""
        remaining = self._engine_wait_s
        while remaining > 0 and not self._stopping.is_set():
            slice_s = min(READY_WAIT_SLICE_S, remaining)
            if self._engines.wait_ready(engine, slice_s):
                return True
            remaining -= slice_s
        return False

    def _remember_timings(self, timings: StageTimings, retry: bool) -> None:
        """The Diagnostics ring holds real dictations only: a retry's stages are measured from
        the retry start, not from a release."""
        if retry:
            return
        with self._state_lock:
            self._ring.append(timings)

    def _transcribe(
        self,
        recording: CompletedRecording,
        settings: Settings,
        on_metrics: asr.MetricsSink | None = None,
    ) -> Transcript | None:
        """asr.transcribe with the readiness wait and the single retry of spec 13.

        The retry happens only when the engine was found down (the health check failed and a
        restart was requested, or the supervisor no longer reports it serving). A request
        that failed against a healthy engine fails the dictation at once, with the PCM kept
        for "Retry last dictation" (spec 16).
        """
        routes = self._speech_routes()
        if routes is not None:
            return self._transcribe_routed(recording, settings, routes, on_metrics)
        terms = self._terms(settings)
        enabled = list(settings.general.enabled_languages)
        for attempt in (1, 2):
            url = self._whisper_ready(recording.dictation_id)
            if url is None:
                raise _Failed(ENGINE_UNAVAILABLE)
            client = self._whisper_factory(url)
            try:
                return asr.transcribe(
                    recording.pcm,
                    recording.sample_rate,
                    recording.lang_mode,
                    terms,
                    enabled,
                    self._lang_state,
                    client,
                    on_metrics,
                    timeout_base_s=self._timeout_base_s(),
                )
            except WhisperError as exc:
                log.warning(
                    "dictation %d: whisper request failed on attempt %d: %s",
                    recording.dictation_id,
                    attempt,
                    exc,
                )
                engine_down = self._check_whisper(client)
                if attempt == 1 and engine_down:
                    continue
                break
        raise _Failed(TRANSCRIPTION_FAILED)

    def _partial_transcribe(self, live: _Live, pcm: bytes, rate: int) -> Transcript | None:
        """One partial pass: the final pass's policy, on a copy of its state (B5-57).

        It never waits for an engine, never asks for a restart and never touches
        `self._lang_state`: an engine that is not serving right now ends the pass, and the
        loop tries again at the next tick.
        """
        settings = self._config.settings
        terms = self._terms(settings)
        enabled = list(settings.general.enabled_languages)
        handle = AbortHandle()
        live.set_abort(handle)
        try:
            routes = self._speech_routes()
            if routes is None:
                url = self._serving_now(Engine.WHISPER, lambda: self._engines.whisper_url)
                if url is None:
                    raise _NotServing
                client = asr.AbortableClient(self._whisper_factory(url), handle)
                return asr.transcribe(
                    pcm,
                    rate,
                    live.lang_mode,
                    terms,
                    enabled,
                    live.state,
                    client,
                    timeout_base_s=self._timeout_base_s(),
                )

            def connect(route: SpeechRoute) -> Any:
                if not self._route_live_capable(route):
                    return None
                url = self._serving_now(route.engine, lambda: self._engines.url(route.engine))
                if url is None:
                    return None
                if route.runtime == LLAMA_ASR:
                    inner = self._llama_asr_factory(url)
                else:
                    inner = self._whisper_factory(url)
                return asr.AbortableClient(inner, handle)

            return asr.transcribe_routed(
                pcm,
                rate,
                live.lang_mode,
                terms,
                enabled,
                live.state,
                routes,
                connect,
                timeout_base_s=self._timeout_base_s(),
            )
        finally:
            live.set_abort(None)

    def _serving_now(self, engine: Any, url: Callable[[], str | None]) -> str | None:
        """The engine's URL when it is serving this instant, never a wait (spec 13)."""
        try:
            if self._engines.status(engine) not in SERVING:
                return None
        except Exception:
            log.debug("the engine status could not be read for a partial", exc_info=True)
            return None
        return url()

    def _speech_routes(self) -> tuple[SpeechRoute, ...] | None:
        slots = tuple(self._engines.speech_engines())
        if len(slots) == 1 and slots[0].runtime == WHISPER_SERVER:
            return None
        if not slots:
            return None
        return tuple(SpeechRoute(slot.engine, slot.runtime, slot.languages) for slot in slots)

    def _transcribe_routed(
        self,
        recording: CompletedRecording,
        settings: Settings,
        routes: tuple[SpeechRoute, ...],
        on_metrics: asr.MetricsSink | None = None,
    ) -> Transcript | None:
        terms = self._terms(settings)
        enabled = list(settings.general.enabled_languages)
        dictation_id = recording.dictation_id
        for attempt in (1, 2):
            clients: dict[Any, Any] = {}

            def connect(route: SpeechRoute, clients=clients) -> Any:
                url = self._serving_url(
                    dictation_id, route.engine, lambda: self._engines.url(route.engine)
                )
                if url is None:
                    return None
                if route.runtime == LLAMA_ASR:
                    client = self._llama_asr_factory(url)
                else:
                    client = self._whisper_factory(url)
                clients[route.engine] = client
                return client

            try:
                transcript = asr.transcribe_routed(
                    recording.pcm,
                    recording.sample_rate,
                    recording.lang_mode,
                    terms,
                    enabled,
                    self._lang_state,
                    routes,
                    connect,
                    on_metrics,
                    timeout_base_s=self._timeout_base_s(),
                )
            except SpeechEngineUnavailable as exc:
                log.warning("dictation %d: speech engine %s is unavailable", dictation_id, exc.engine)
                raise _Failed(ENGINE_UNAVAILABLE) from exc
            except WhisperError as exc:
                log.warning(
                    "dictation %d: speech request to %s failed on attempt %d: %s",
                    dictation_id,
                    exc.engine,
                    attempt,
                    exc,
                )
                engine = exc.engine if exc.engine is not None else routes[0].engine
                engine_down = self._check_speech(clients.get(engine), engine)
                if attempt == 1 and engine_down:
                    continue
                break
            if transcript is not None:
                log.info(
                    "dictation %d: transcribed by %s in %s",
                    dictation_id,
                    transcript.engine,
                    transcript.language,
                )
            return transcript
        raise _Failed(TRANSCRIPTION_FAILED)

    def _check_whisper(self, client: WhisperClient) -> bool:
        return self._check_speech(client, Engine.WHISPER)

    def _check_speech(self, client: Any, engine: Any) -> bool:
        """After a failed request: a failed health check asks for a restart (spec 6 step 5).

        Returns True when the engine was found down, which is when a retry after READY is
        warranted (spec 13). When the supervisor already reports a non-serving state it is
        handling the failure itself and a restart request would reset its backoff and
        failure count, so none is sent.
        """
        engines = self._engines
        if engines.status(engine) not in SERVING:
            return True
        try:
            healthy = client.health() if client is not None else False
        except Exception:
            log.debug("speech health check raised; treating the engine as down", exc_info=True)
            healthy = False
        if healthy:
            return False
        log.warning("speech engine health check failed after the request; asking for a restart")
        engines.restart(engine)
        for _ in range(RESTART_SETTLE_POLLS):
            if engines.status(engine) not in SERVING:
                break
            self._sleeper(RESTART_SETTLE_POLL_S)
        return True

    def _clean(self, transcript: Transcript, profile: Profile, settings: Settings) -> CleanResult:
        engines = self._engines
        # A URL exists in READY and CPU_FALLBACK only, so "not ready" and "on CPU" stay
        # distinct reasons (spec 8.1); cleanup_available would fold both into the former.
        url = engines.llama_url
        gate = GateInput(
            cleanup_enabled=settings.cleanup.enabled,
            profile_cleanup=profile.cleanup,
            engine_available=url is not None,
            cpu_fallback=engines.variant(Engine.LLAMA) == "cpu"
            and not engines.cpu_cleanup_allowed,
            cpu_selected=bool(engines.cpu_only) or self._integrated_gpu(),
            cleanup_languages=engines.cleanup_languages,
        )
        if url:
            client: LlamaClient = self._llama_factory(url, settings.cleanup.timeout_ms / 1000.0)
        else:
            client = _UnavailableLlama()
        result = cleanup.clean(
            transcript,
            profile,
            self._terms(settings),
            settings.cleanup.fillers,
            settings.cleanup.corrections,
            list(settings.general.enabled_languages),
            gate,
            client,
        )
        if result.reason in GUARD_REASONS:
            with self._state_lock:
                self._guard_counts[result.reason] = self._guard_counts.get(result.reason, 0) + 1
            log.info("cleanup output rejected: %s", result.reason)
        elif result.reason != "ok":
            log.debug("cleanup skipped: %s", result.reason)
        return result

    @staticmethod
    def _typed_live(recording: CompletedRecording) -> bool:
        live = recording.live
        return live is not None and live.session.typed_anything

    def _deliver_live(
        self, recording: CompletedRecording, text: str, started: float
    ) -> DeliveryReport:
        """Reconcile the draft this dictation typed into the delivered text (B5-57).

        The delivered text always wins: the draft and it share a long prefix in the ordinary
        case, so only the tail is replaced. A session that has stopped, or that cannot
        finish the reconciliation, never injects again: the text goes to the clipboard, so
        nothing is typed into a window Spells did not start in and the draft is never
        duplicated. The characters already typed stay where they are, because taking them
        back would mean typing into whatever window has the foreground now.
        """
        live = recording.live
        session = live.session
        if not session.stopped and session.apply(text):
            elapsed_ms = (self._clock() - started) * 1000.0
            return DeliveryReport(
                DeliveryResult(DeliveryOutcome.TYPED, "live insertion reconciled"), elapsed_ms
            )
        reason = session.stop_reason or "live insertion stopped"
        log.info(
            "dictation %d: %s, so the text goes to the clipboard", recording.dictation_id, reason
        )
        report = self._copy_to_clipboard(text)
        if report.outcome is DeliveryOutcome.FAILED:
            return report
        return DeliveryReport(
            DeliveryResult(DeliveryOutcome.COPIED_FOCUS_CHANGED, f"{reason}: {CLIPBOARD_DETAIL}"),
            report.elapsed_ms,
        )

    def _copy_to_clipboard(self, text: str) -> DeliveryReport:
        """The manual retry never injects: the result goes to the clipboard (spec 13)."""
        started = self._clock()
        try:
            self._backends.clipboard.set_text(text)
        except Exception as exc:
            log.warning("retry: clipboard set failed", exc_info=True)
            return DeliveryReport(DeliveryResult(DeliveryOutcome.FAILED, f"retry copy failed: {exc!r}"))
        elapsed_ms = (self._clock() - started) * 1000.0
        return DeliveryReport(
            DeliveryResult(
                DeliveryOutcome.COPIED_FOCUS_CHANGED, "manual retry: result copied to the clipboard"
            ),
            elapsed_ms,
        )

    def _note_delivery(
        self, recording: CompletedRecording, text: str, report: DeliveryReport, started: float
    ) -> None:
        if report.outcome not in DELIVERED:
            return
        finished_at = started + (report.elapsed_ms or 0.0) / 1000.0
        self._last_delivery = LastDelivery(
            hwnd=recording.ctx.hwnd,
            finished_at=finished_at,
            outcome=report.outcome,
            ended_with_whitespace=inject.ends_with_whitespace(text),
        )

    def _timings_for(
        self,
        recording: CompletedRecording,
        report: DeliveryReport | None,
        reference: float,
        transcript_at: float,
        cleaned_at: float | None,
        delivery_started: float | None,
        retry: bool,
    ) -> StageTimings:
        """Spec 12 stage timings; the stages after an empty transcript stay None."""
        extra = self._audio_extra(recording, {})
        if report is not None and delivery_started is not None:
            extra["release_to_delivered_ms"] = (delivery_started - reference) * 1000.0 + (
                report.elapsed_ms or 0.0
            )
        else:
            extra["nothing_to_deliver"] = 1.0
        if retry:
            extra["retry"] = 1.0
        if recording.capped:
            extra["capped"] = 1.0
        if recording.live is not None:
            extra.update(recording.live.stats())
        return StageTimings(
            press_to_pill_ms=recording.press_to_pill_ms,
            mic_open_ms=recording.mic_open_ms,
            release_to_transcript_ms=(transcript_at - reference) * 1000.0,
            release_to_cleaned_ms=None if cleaned_at is None else (cleaned_at - reference) * 1000.0,
            delivery_ms=None if report is None else report.elapsed_ms,
            extra=extra,
        )

    def _signals_for(
        self,
        recording: CompletedRecording,
        transcript: Transcript,
        metrics: list[quality.AsrMetrics],
        settings: Settings,
    ) -> quality.QualitySignals:
        """The free quality signals of one dictation (spec 8.4).

        The filler and self-correction lists are the ones the cleanup gate uses for that
        language, so the counts shown in History are the same words cleanup looks for.
        """
        language = transcript.language
        return quality.signals(
            transcript.text,
            metrics=metrics[-1] if metrics else None,
            audio_s=len(recording.pcm) / (2.0 * recording.sample_rate),
            fillers=settings.cleanup.fillers.get(language, ()),
            corrections=settings.cleanup.corrections.get(language, ()),
            dropped_blocks=recording.dropped_blocks,
        )

    def _audio_policy(self, settings: Settings) -> AudioPolicy:
        history = settings.history
        return AudioPolicy(
            keep=history.keep_audio,
            max_files=history.audio_keep_count,
            max_mb=history.audio_keep_mb,
        )

    def _write_history(
        self,
        recording: CompletedRecording,
        transcript: Transcript,
        result: CleanResult,
        text: str,
        report: DeliveryReport,
        timings: StageTimings,
        signals: quality.QualitySignals,
        settings: Settings,
    ) -> None:
        verdict = quality.assess(signals)
        entry = HistoryEntry(
            id=None,
            created_at=time.time(),
            raw_text=transcript.text,
            cleaned_text=result.text,
            delivered_text=text,
            app_process=recording.ctx.process,
            app_title=recording.ctx.title,
            language=transcript.language,
            used_llm=result.used_llm,
            cleanup_reason=result.reason,
            outcome=report.outcome.value,
            timings=timings,
            asr_engine=transcript.engine or Engine.WHISPER.value,
            cleanup_engine=Engine.LLAMA.value if result.used_llm else "",
            signals=signals,
            quality_label=verdict.label,
            quality_reason=verdict.reason,
        )
        self._store_history(entry, recording, settings)

    def _store_history(
        self,
        entry: HistoryEntry,
        recording: CompletedRecording,
        settings: Settings,
        *,
        with_audio: bool = True,
    ) -> None:
        """Write the row, and the recording with it when the user asked for one (spec 17).

        The PCM only reaches the store when keep_audio is on, and the call without audio
        stays the single-argument one every history stand-in understands. A dictation that
        held no audio passes with_audio False: there is nothing worth keeping.
        """
        audio = self._audio_policy(settings)
        try:
            if audio.keep and with_audio:
                self._history.add(entry, recording.pcm, recording.sample_rate, audio)
            else:
                self._history.add(entry)
        except Exception:
            log.exception("history write failed; the dictation still counts as done (B3-20)")

    def check_transcript(self, text: str, language: str = "") -> quality.CheckResult:
        """Ask the cleanup model whether a transcript reads like real speech (spec 8.4).

        Never called from a dictation: the History page runs it on a worker thread, on rows
        the user picked. It skips politely when the cleanup engine is not serving, and an
        answer that is not the format the prompt asked for is rejected rather than stored.
        """
        if not text.strip():
            return quality.CheckResult("", "", "empty")
        url = self._engines.llama_url
        if url is None:
            return quality.CheckResult("", "", "unavailable")
        system, user = quality.check_messages(text, asr.language_title(language) if language else "")
        try:
            client = self._llama_factory(url, CHECK_TIMEOUT_S)
            content, _finish = client.chat(system, user, quality.CHECK_MAX_TOKENS)
        except CleanupError as exc:
            log.info("transcript check failed: %s", exc)
            return quality.CheckResult("", "", exc.reason)
        except Exception:
            log.exception("transcript check failed")
            return quality.CheckResult("", "", "error")
        return quality.parse_check(content)

    @staticmethod
    def _outcome_ui(report: DeliveryReport) -> dict[str, Any]:
        outcome = report.outcome
        if outcome in DELIVERED:
            return {}
        if outcome in COPIED or (
            outcome is DeliveryOutcome.FAILED and CLIPBOARD_DETAIL in report.result.detail
        ):
            return {"notice": Notice.COPIED, "notice_text": COPIED_TEXT, "notification": COPIED_NOTICE}
        detail = report.result.detail
        return {
            "notice": Notice.ERROR,
            "notice_text": DELIVERY_FAILED,
            "notification": f"{DELIVERY_FAILED}: {detail}" if detail else DELIVERY_FAILED,
        }

    @staticmethod
    def _with_tone(profile: Profile, settings: Settings) -> Profile:
        tone = settings.cleanup.tones.get(profile.name)
        if tone is None or tone == profile.tone:
            return profile
        return replace(profile, tone=tone)

    @staticmethod
    def _terms(settings: Settings) -> list[str]:
        """Vocabulary terms oldest first, so the newest survive the prompt truncation (7.3)."""
        terms = sorted(settings.vocabulary.terms, key=lambda term: term.edited_at)
        return [term.text for term in terms]

    # Events ----------------------------------------------------------------------------------

    def _emit(
        self,
        dictation_id: int | None,
        *,
        notice: Notice | None = None,
        notice_text: str = "",
        notification: str | None = None,
        notification_action: str | None = None,
        timings: StageTimings | None = None,
    ) -> None:
        with self._state_lock:
            pending = self._pending
            if self._rec_pill is not None:
                pill, text, level = self._rec_pill, self._rec_text, self._level
            elif self._waiting_engines:
                pill, text, level = PillState.STARTING_ENGINES, STARTING_ENGINES_TEXT, 0.0
            elif self._writing:
                pill, text, level = PillState.WRITING, self._writing_text, 0.0
            elif pending > 0:
                pill, text, level = PillState.PROCESSING, "", 0.0
            else:
                pill, text, level = PillState.IDLE, "", 0.0
            if self._rec_pill is not None:
                tray = TrayState.RECORDING
            elif pending > 0:
                tray = TrayState.PROCESSING
            else:
                tray = TrayState.READY
            event = PipelineEvent(
                pill=pill,
                tray=tray,
                level=level,
                busy=pending > 0,
                text=text,
                live_typing=self._live_typing and self._rec_pill is not None,
                notice=notice,
                notice_text=notice_text,
                notification=notification,
                notification_action=notification_action,
                retry_available=self._retry is not None,
                dictation_id=dictation_id,
                timings=timings,
                target_hwnd=self._target_hwnd,
            )
        try:
            self._on_event(event)
        except Exception:
            log.exception("on_event callback failed")


__all__ = [
    "PARTIAL_INTERVAL_S",
    "PARTIAL_MAX_LATENCY_MS",
    "PARTIAL_MIN_AUDIO_S",
    "PARTIAL_WINDOW_S",
    "CompletedRecording",
    "Notice",
    "PillState",
    "Pipeline",
    "PipelineEvent",
    "TrayState",
]

