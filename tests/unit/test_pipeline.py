"""Unit tests for spells.pipeline: the recording controller and the processing worker.

Spec 5.1, 6 (the whole dictation flow), 10.4, 11 (consumer side), 12, 13 (consumer
side), 16 and 20.1 (pipeline row). Everything is faked: the hotkey (callbacks are
invoked directly and end_recording is recorded), the recorder, the engine supervisor,
the whisper and llama clients, the inject backends, the context capture, the
scheduler behind the cap timer and the clock. No thread runs unless a test starts
one; the two loop bodies are driven with drain_controller() and drain_worker().
"""

from __future__ import annotations

import io
import threading
import wave
from dataclasses import replace

import pytest

from spells import quality
from spells.asr import AsrReply, WhisperError
from spells.audio import MicError
from spells.cleanup import CleanupError
from spells.config import ConfigStore, TermEntry
from spells.engines import SpeechSlot
from spells.models import (
    Chord,
    Engine,
    EngineId,
    EngineState,
    StageTimings,
    TargetContext,
)
from spells.pipeline import (
    AUDIO_LOST_NOTIFICATION,
    AUDIO_LOST_TEXT,
    CAP_WARNING_TEXT,
    CHECK_TIMEOUT_S,
    COPIED_NOTICE,
    DELIVERY_FAILED,
    DICTATION_FAILED,
    ENGINE_UNAVAILABLE,
    MIC_BLOCKED_TEXT,
    NO_AUDIO_REASON,
    NO_AUDIO_TEXT,
    PRIVACY_SETTINGS,
    SOUND_SETTINGS,
    TRANSCRIPTION_FAILED,
    Notice,
    PillState,
    Pipeline,
    TrayState,
)

from .fake_clients import ClientFactory, FakeLlamaAsrClient, FakeLlamaClient, FakeWhisperClient
from .fake_hook import FakeClock
from .fake_recorder import NATIVE_SAMPLE_RATE, FakeRecorder
from .test_inject import OTHER_HWND, TARGET_HWND
from .test_inject import make as make_inject_backends

WHISPER = Engine.WHISPER
LLAMA = Engine.LLAMA
MAIN = Chord(keys=(0x11, 0x5B))
DE_CHORD = Chord(keys=(0x11, 0x44), language="de")
LONG_TEXT = (
    "so I think we should meet tomorrow at ten to go through the release plan "
    "and decide who owns the installer work"
)
CLEANED = (
    "I think we should meet tomorrow at ten to go through the release plan "
    "and decide who owns the installer work."
)
OK_RESPONSE = {"text": LONG_TEXT, "language": "english"}


# Fakes local to this file ---------------------------------------------------------


class FakeEngines:
    """The EngineSupervisor slice the pipeline consumes, scripted per test.

    Mirrors the real rules: an engine URL exists in READY and CPU_FALLBACK only,
    cleanup_available is True in READY only, ensure_ready acts only in UNLOADED.
    """

    def __init__(self) -> None:
        self.states = {WHISPER: EngineState.READY, LLAMA: EngineState.READY}
        self.variants = {WHISPER: "vulkan", LLAMA: "vulkan"}
        self.urls: dict[Engine, str | None] = {
            WHISPER: "http://127.0.0.1:9001",
            LLAMA: "http://127.0.0.1:9002",
        }
        self.calls: list[tuple] = []
        # wait_ready answers wait_results in order, then wait_result; True moves to READY.
        self.wait_results: list[bool] = []
        self.wait_result = True
        self.on_wait = None
        self.cpu_cleanup_allowed = False
        self.cpu_only = False
        self.cleanup_languages: frozenset[str] | None = None
        self.slots = (SpeechSlot(EngineId(WHISPER), "whisper-server", ()),)

    def speech_engines(self):
        return self.slots

    def url(self, engine) -> str | None:
        return self._url(WHISPER if engine == EngineId(WHISPER) else engine)

    def _url(self, engine: Engine) -> str | None:
        if self.states[engine] in (EngineState.READY, EngineState.CPU_FALLBACK):
            return self.urls[engine]
        return None

    @property
    def whisper_url(self) -> str | None:
        return self._url(WHISPER)

    @whisper_url.setter
    def whisper_url(self, value: str | None) -> None:
        self.urls[WHISPER] = value

    @property
    def llama_url(self) -> str | None:
        return self._url(LLAMA)

    @llama_url.setter
    def llama_url(self, value: str | None) -> None:
        self.urls[LLAMA] = value

    @property
    def cleanup_available(self) -> bool:
        return self.states[LLAMA] is EngineState.READY

    def status(self, engine: Engine) -> EngineState:
        return self.states[engine]

    def variant(self, engine: Engine) -> str:
        return self.variants[engine]

    def ensure_ready(self) -> None:
        self.calls.append(("ensure_ready",))
        for engine, state in self.states.items():
            if state is EngineState.UNLOADED:
                self.states[engine] = EngineState.STARTING

    def restart(self, engine: Engine) -> None:
        self.calls.append(("restart", engine))
        self.states[engine] = EngineState.STARTING

    def note_activity(self) -> None:
        self.calls.append(("note_activity",))

    def wait_ready(self, engine: Engine, timeout_s: float) -> bool:
        self.calls.append(("wait_ready", engine, timeout_s))
        if self.on_wait is not None:
            self.on_wait()
        result = self.wait_results.pop(0) if self.wait_results else self.wait_result
        if result:
            self.states[engine] = EngineState.READY
        return result


class FakeHotkey:
    def __init__(self) -> None:
        self.ended: list[int] = []

    def end_recording(self, dictation_id: int) -> None:
        self.ended.append(dictation_id)


class FakeHistory:
    def __init__(self, error: Exception | None = None) -> None:
        self.entries: list = []
        self.audio: list = []
        self.rates: list = []
        self.policies: list = []
        self.error = error

    def add(self, entry, pcm16=None, sample_rate=None, audio=None) -> int:
        if self.error is not None:
            raise self.error
        self.entries.append(entry)
        self.audio.append(pcm16)
        self.rates.append(sample_rate)
        self.policies.append(audio)
        return len(self.entries)


class FakeTimer:
    def __init__(self, delay: float, callback) -> None:
        self.delay = delay
        self.callback = callback
        self.cancelled = False
        self.fired = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeScheduler:
    """Records (delay, callback) pairs; fire() runs one the way threading.Timer would."""

    def __init__(self) -> None:
        self.timers: list[FakeTimer] = []

    def __call__(self, delay_s: float, callback):
        timer = FakeTimer(delay_s, callback)
        self.timers.append(timer)
        return timer.cancel

    def pending(self) -> list[FakeTimer]:
        return [t for t in self.timers if not t.cancelled and not t.fired]

    def pending_delays(self) -> list[float]:
        return [t.delay for t in self.pending()]

    def fire(self, delay_s: float) -> None:
        timer = next(t for t in self.pending() if t.delay == delay_s)
        timer.fired = True
        timer.callback()


class Harness:
    def __init__(
        self,
        tmp_path,
        *,
        whisper: FakeWhisperClient | None = None,
        llama: FakeLlamaClient | None = None,
        engines: FakeEngines | None = None,
        history: FakeHistory | None = None,
        on_event=None,
        recorder_setup=None,
        foreground: int = TARGET_HWND,
        elevated: bool = False,
        input_error=(),
        clipboard_error=(),
        llama_factory=None,
        recorder_factory=None,
        settings=None,
        selection=None,
        partial_threads=False,
    ) -> None:
        self.config = ConfigStore(tmp_path / "settings.json")
        if settings is not None:
            self.config.update(settings)
        self.clock = FakeClock()
        self.engines = engines if engines is not None else FakeEngines()
        self.hotkey = FakeHotkey()
        self.history = history if history is not None else FakeHistory()
        self.whisper = whisper if whisper is not None else FakeWhisperClient(responses=[OK_RESPONSE])
        self.llama = llama if llama is not None else FakeLlamaClient(content=CLEANED)
        self.whisper_factory = ClientFactory(self.whisper)
        self.llama_factory = ClientFactory(self.llama)
        self.recorders: list[FakeRecorder] = []
        self.recorder_setup = recorder_setup
        self.scheduler = FakeScheduler()
        self.events = []
        self.sleeps: list[float] = []
        self.inject_log, self.backends = make_inject_backends(
            input_error=input_error,
            clipboard_error=clipboard_error,
            foreground=foreground,
            elevated=elevated,
        )
        self.ctx_hwnd = TARGET_HWND
        self.ctx_process = "notepad.exe"
        self.ctx_title = "Untitled - Notepad"
        self.capture_advance_s = 0.0
        self.pipeline = Pipeline(
            config=self.config,
            engines=self.engines,
            hotkey=self.hotkey,
            history=self.history,
            on_event=on_event if on_event is not None else self.events.append,
            recorder_factory=recorder_factory or self._recorder,
            capture=self._capture,
            inject_backends=self.backends,
            whisper_client_factory=self.whisper_factory,
            llama_client_factory=llama_factory or self.llama_factory,
            scheduler=self.scheduler,
            clock=self.clock.now,
            sleeper=self.sleeps.append,
            selection=selection,
            partial_threads=partial_threads,
        )
        self.callbacks = self.pipeline.hotkey_callbacks()

    # Collaborator hooks

    def _recorder(self, **kwargs) -> FakeRecorder:
        recorder = FakeRecorder(**kwargs)
        if self.recorder_setup is not None:
            self.recorder_setup(recorder)
        self.recorders.append(recorder)
        return recorder

    def _capture(self) -> TargetContext:
        self.clock.advance(self.capture_advance_s)
        return TargetContext(self.ctx_hwnd, self.ctx_process, self.ctx_title, self.clock.now())

    @property
    def recorder(self) -> FakeRecorder:
        return self.recorders[-1]

    @property
    def last(self):
        return self.events[-1]

    # Driving

    def prime(self) -> None:
        """Deliver the settings the controller thread would see at start()."""
        self.config.update(lambda s: s)
        self.pipeline.drain_controller()

    def press(self, dictation_id: int = 1, chord: Chord = MAIN) -> None:
        self.callbacks.pressed(chord, dictation_id)
        self.pipeline.drain_controller()

    def release(self, dictation_id: int = 1, chord: Chord = MAIN, hold_s: float = 1.0) -> None:
        self.clock.advance(hold_s)
        self.callbacks.released(chord, dictation_id)
        self.pipeline.drain_controller()

    def process(self) -> None:
        self.pipeline.drain_worker()

    def dictate(self, dictation_id: int = 1, chord: Chord = MAIN) -> None:
        self.press(dictation_id, chord)
        self.release(dictation_id, chord)
        self.process()

    def delivered_texts(self) -> list[str]:
        return [
            entry[1]
            for entry in self.inject_log
            if entry[0] in ("set_text", "type_unicode")
        ]

    def events_for(self, dictation_id: int):
        return [e for e in self.events if e.dictation_id == dictation_id]


def general(**changes):
    return lambda s: replace(s, general=replace(s.general, **changes))


def cleanup_settings(**changes):
    return lambda s: replace(s, cleanup=replace(s.cleanup, **changes))


def history_settings(**changes):
    return lambda s: replace(s, history=replace(s.history, **changes))


# Recording controller -------------------------------------------------------------------


def test_press_emits_the_recording_event_in_the_same_step(tmp_path):
    h = Harness(tmp_path)
    h.press()
    assert h.recorder.is_recording
    assert h.last.pill is PillState.RECORDING
    assert h.last.tray is TrayState.RECORDING
    assert h.last.dictation_id == 1
    assert h.last.busy is False
    assert h.last.target_hwnd == TARGET_HWND
    assert ("ensure_ready",) in h.engines.calls
    assert ("note_activity",) not in h.engines.calls


def test_press_reloads_unloaded_engines_while_recording(tmp_path):
    engines = FakeEngines()
    engines.states[WHISPER] = EngineState.UNLOADED
    engines.states[LLAMA] = EngineState.UNLOADED
    h = Harness(tmp_path, engines=engines)
    h.press()
    # The reload starts on the press and overlaps the recording (spec 13).
    assert engines.calls == [("ensure_ready",)]
    assert engines.states[WHISPER] is EngineState.STARTING
    assert h.recorder.is_recording
    h.release()
    h.process()
    assert ("wait_ready", WHISPER, 1.0) in engines.calls
    assert h.history.entries[-1].outcome == "pasted"


def test_target_hwnd_follows_the_recording(tmp_path):
    h = Harness(tmp_path)
    h.press()
    assert h.last.target_hwnd == TARGET_HWND
    h.recorder.feed_level(0.3)
    h.pipeline.drain_controller()
    assert h.last.target_hwnd == TARGET_HWND
    h.release()
    assert h.last.target_hwnd is None
    h.process()
    assert all(e.target_hwnd is None for e in h.events if e.pill is not PillState.RECORDING)
    h.ctx_hwnd = 0
    h.press(2)
    assert h.last.pill is PillState.RECORDING
    assert h.last.target_hwnd is None


def test_release_enqueues_the_recording_with_the_chord_language_forced(tmp_path):
    h = Harness(tmp_path)
    h.dictate(chord=DE_CHORD)
    assert [c["language"] for c in h.whisper.calls] == ["de"]


def test_release_uses_the_locked_language_from_settings(tmp_path):
    h = Harness(tmp_path, settings=general(language_mode="sq"))
    h.dictate()
    assert [c["language"] for c in h.whisper.calls] == ["sq"]


def test_release_uses_auto_mode_from_settings(tmp_path):
    h = Harness(tmp_path)
    h.dictate()
    assert [c["language"] for c in h.whisper.calls] == ["auto"]


def test_release_shows_processing_before_the_worker_runs(tmp_path):
    h = Harness(tmp_path)
    h.press()
    h.release()
    assert h.last.pill is PillState.PROCESSING
    assert h.last.tray is TrayState.PROCESSING
    assert not h.recorder.is_recording
    assert h.scheduler.pending() == []


def test_latched_shows_the_lock_and_keeps_recording(tmp_path):
    h = Harness(tmp_path)
    h.press()
    h.callbacks.latched(MAIN, 1)
    h.pipeline.drain_controller()
    assert h.last.pill is PillState.LATCHED
    assert h.recorder.is_recording
    h.release()
    h.process()
    assert len(h.whisper.calls) == 1


def test_tap_discard_drops_the_audio_and_ends_the_recording(tmp_path):
    h = Harness(tmp_path)
    h.press()
    h.callbacks.discarded(1)
    h.pipeline.drain_controller()
    h.process()
    assert h.recorder.calls == ["start", "cancel"]
    assert h.hotkey.ended == [1]
    assert h.last.pill is PillState.IDLE
    assert h.whisper.calls == []


def test_esc_cancel_drops_the_audio_and_ends_the_recording(tmp_path):
    h = Harness(tmp_path)
    h.press()
    h.callbacks.cancelled(1)
    h.pipeline.drain_controller()
    h.process()
    assert h.recorder.calls == ["start", "cancel"]
    assert h.hotkey.ended == [1]
    assert h.scheduler.pending() == []
    assert h.whisper.calls == []


def test_cap_ends_the_recording_through_the_queue(tmp_path):
    h = Harness(tmp_path)
    h.press()
    assert h.scheduler.pending_delays() == [540.0, 600.0]
    h.scheduler.fire(540.0)
    h.pipeline.drain_controller()
    assert h.last.pill is PillState.RECORDING
    assert h.last.text == CAP_WARNING_TEXT
    h.scheduler.fire(600.0)
    # The timer only enqueued: the recorder is still running until the controller acts.
    assert h.recorder.is_recording
    h.pipeline.drain_controller()
    assert not h.recorder.is_recording
    assert h.hotkey.ended == [1]
    assert h.last.pill is PillState.PROCESSING
    h.process()
    assert len(h.whisper.calls) == 1
    assert h.history.entries[-1].outcome == "pasted"


def test_stale_callbacks_are_ignored(tmp_path):
    h = Harness(tmp_path)
    h.press()
    h.scheduler.fire(600.0)
    h.pipeline.drain_controller()
    # The key-up crossed the cap: hotkey still fires released for the ended dictation.
    h.callbacks.released(MAIN, 1)
    h.callbacks.latched(MAIN, 1)
    h.callbacks.cancelled(1)
    h.pipeline.drain_controller()
    assert h.recorder.calls == ["start", "stop"]
    h.process()
    assert len(h.whisper.calls) == 1
    assert h.hotkey.ended == [1]


def test_mic_error_on_start_shows_the_reason_and_ends_the_recording(tmp_path):
    def setup(recorder):
        recorder.start_error = MicError("busy", "Microphone in use by another app.")

    h = Harness(tmp_path, recorder_setup=setup)
    h.press()
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == "Mic is busy"
    assert h.last.notification == "Microphone in use by another app."
    assert h.last.notification_action == SOUND_SETTINGS
    assert h.last.pill is PillState.IDLE
    assert h.hotkey.ended == [1]
    h.release()
    h.process()
    assert h.whisper.calls == []


@pytest.mark.parametrize(
    "kind, text",
    [
        ("missing", "No mic found"),
        ("busy", "Mic is busy"),
        ("blocked", "Mic is blocked"),
        ("unknown", "Mic error"),
    ],
)
def test_every_mic_error_kind_has_its_short_pill_text(tmp_path, kind, text):
    def setup(recorder):
        recorder.start_error = MicError(kind, "The longer message for the balloon.")

    h = Harness(tmp_path, recorder_setup=setup)
    h.press()
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == text
    assert h.last.notification == "The longer message for the balloon."


def test_the_cap_warning_is_the_mockup_string():
    assert CAP_WARNING_TEXT == "One minute left"
    assert MIC_BLOCKED_TEXT == "Mic is blocked"


def test_mic_blocked_on_stop_drops_the_recording(tmp_path):
    def setup(recorder):
        recorder.stop_error = MicError("blocked", "Windows privacy settings block the mic.")

    h = Harness(tmp_path, recorder_setup=setup)
    h.press()
    h.release()
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == MIC_BLOCKED_TEXT
    assert h.last.notification_action == PRIVACY_SETTINGS
    assert h.hotkey.ended == [1]
    h.process()
    assert h.whisper.calls == []
    assert h.last.pill is PillState.IDLE


def test_silent_recorder_shows_the_blocked_hint_mid_recording(tmp_path):
    h = Harness(tmp_path)
    h.press()
    h.recorder.feed_level(0.42)
    h.pipeline.drain_controller()
    assert h.last.pill is PillState.RECORDING
    assert h.last.level == pytest.approx(0.42)
    assert h.last.text == ""
    h.recorder.silent = True
    h.recorder.feed_level(0.0)
    h.pipeline.drain_controller()
    assert h.last.text == MIC_BLOCKED_TEXT


def test_level_updates_are_coalesced_into_one_queued_event(tmp_path):
    h = Harness(tmp_path)
    h.press()
    before = len(h.events)
    for level in (0.1, 0.2, 0.3):
        h.recorder.feed_level(level)
    h.pipeline.drain_controller()
    assert len(h.events) == before + 1
    assert h.last.level == pytest.approx(0.3)


def test_recording_while_a_dictation_processes_shows_the_busy_badge(tmp_path):
    h = Harness(tmp_path)
    h.press(1)
    h.release(1)
    h.press(2)
    assert h.last.pill is PillState.RECORDING
    assert h.last.busy is True
    assert h.last.tray is TrayState.RECORDING
    h.release(2)
    h.process()
    assert len(h.whisper.calls) == 2
    assert h.last.pill is PillState.IDLE
    assert h.last.busy is False


def test_fifo_order_with_two_recordings(tmp_path):
    h = Harness(tmp_path)
    h.press(1)
    h.recorder.pcm = b"\x01\x00" * 40000
    h.release(1)
    h.press(2)
    h.recorder.pcm = b"\x01\x00" * 48000
    h.release(2)
    h.process()
    assert [len(c["wav"]) - 44 for c in h.whisper.calls] == [80000, 96000]
    assert [e.dictation_id for e in h.events if e.timings is not None] == [1, 2]


def test_a_new_press_while_one_is_active_discards_the_old_recording(tmp_path):
    h = Harness(tmp_path)
    h.press(1)
    h.press(2)
    assert h.recorder.calls == ["start", "cancel", "start"]
    assert h.hotkey.ended == [1]
    h.release(2)
    h.process()
    assert len(h.whisper.calls) == 1


def test_controller_survives_a_failing_collaborator(tmp_path):
    failures = [RuntimeError("no audio backend")]

    def factory(**kwargs):
        if failures:
            raise failures.pop()
        return FakeRecorder(**kwargs)

    h = Harness(tmp_path, recorder_factory=factory)
    h.press(1)
    assert h.last.notice is Notice.ERROR
    assert h.hotkey.ended == [1]
    h.press(2)
    assert h.last.pill is PillState.RECORDING
    h.release(2)
    h.process()
    assert len(h.whisper.calls) == 1


# Engine readiness --------------------------------------------------------------------------


def test_engine_not_ready_calls_ensure_ready_and_waits_then_transcribes(tmp_path):
    engines = FakeEngines()
    engines.states[WHISPER] = EngineState.STARTING
    h = Harness(tmp_path, engines=engines)
    h.dictate()
    assert ("ensure_ready",) in engines.calls
    assert ("wait_ready", WHISPER, 1.0) in engines.calls
    assert any(e.pill is PillState.STARTING_ENGINES for e in h.events)
    assert len(h.whisper.calls) == 1
    assert h.history.entries[-1].outcome == "pasted"


def test_wait_ready_timeout_keeps_the_pcm_and_shows_the_error(tmp_path):
    engines = FakeEngines()
    engines.states[WHISPER] = EngineState.UNLOADED
    engines.wait_result = False
    h = Harness(tmp_path, engines=engines)
    h.dictate()
    assert h.whisper.calls == []
    waits = [c for c in engines.calls if c[0] == "wait_ready"]
    assert len(waits) == 240, "240 s in one-second slices"
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == ENGINE_UNAVAILABLE
    assert h.last.retry_available is True
    assert h.pipeline.retry_available is True
    assert h.last.pill is PillState.IDLE


def test_stop_interrupts_a_pending_readiness_wait(tmp_path):
    engines = FakeEngines()
    engines.states[WHISPER] = EngineState.RESTARTING
    engines.wait_result = False
    h = Harness(tmp_path, engines=engines)
    engines.on_wait = h.pipeline.stop
    h.dictate()
    waits = [c for c in engines.calls if c[0] == "wait_ready"]
    assert len(waits) == 1
    assert h.last.notice_text == ENGINE_UNAVAILABLE
    assert not h.pipeline.running


def test_failed_engine_does_not_wait(tmp_path):
    engines = FakeEngines()
    engines.states[WHISPER] = EngineState.FAILED
    h = Harness(tmp_path, engines=engines)
    h.dictate()
    assert not any(c[0] == "wait_ready" for c in engines.calls)
    assert h.last.notice_text == ENGINE_UNAVAILABLE


def test_whisper_error_with_a_dead_engine_restarts_whisper(tmp_path):
    whisper = FakeWhisperClient(responses=[WhisperError("connection refused")], healthy=False)
    h = Harness(tmp_path, whisper=whisper)
    h.dictate()
    assert ("restart", WHISPER) in h.engines.calls
    assert len(whisper.calls) == 2, "one automatic retry after the restart"
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == TRANSCRIPTION_FAILED
    assert h.last.retry_available is True
    assert h.delivered_texts() == []


def test_whisper_error_with_a_healthy_engine_fails_at_once(tmp_path):
    whisper = FakeWhisperClient(responses=[WhisperError("HTTP 500"), OK_RESPONSE], healthy=True)
    h = Harness(tmp_path, whisper=whisper)
    h.dictate()
    assert len(whisper.calls) == 1, "no retry against a healthy engine (spec 16)"
    assert whisper.health_calls == 1
    assert not any(c[0] == "restart" for c in h.engines.calls)
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == TRANSCRIPTION_FAILED
    assert h.last.retry_available is True
    assert h.delivered_texts() == []
    # The kept PCM serves the manual retry.
    assert h.pipeline.retry_last() is True
    h.process()
    assert ("set_text", CLEANED) in h.inject_log
    assert h.last.notice is Notice.COPIED


def test_whisper_error_is_retried_once_after_a_restart_and_can_succeed(tmp_path):
    whisper = FakeWhisperClient(responses=[WhisperError("reset"), OK_RESPONSE], healthy=False)
    h = Harness(tmp_path, whisper=whisper)
    h.dictate()
    assert ("restart", WHISPER) in h.engines.calls
    assert ("wait_ready", WHISPER, 1.0) in h.engines.calls
    assert len(whisper.calls) == 2
    assert whisper.health_calls == 1
    assert h.last.notice is None
    assert h.last.timings is not None
    assert h.delivered_texts() == [CLEANED]


def test_whisper_error_does_not_restart_when_the_supervisor_already_knows(tmp_path):
    engines = FakeEngines()
    whisper = FakeWhisperClient(responses=[WhisperError("reset")], healthy=False)

    def on_call():
        engines.states[WHISPER] = EngineState.RESTARTING

    whisper.on_call = on_call
    h = Harness(tmp_path, whisper=whisper, engines=engines)
    h.dictate()
    assert not any(c[0] == "restart" for c in engines.calls)
    assert whisper.health_calls == 0
    assert len(whisper.calls) == 2, "the supervisor is restarting it: one retry after READY"
    assert h.last.notice_text == TRANSCRIPTION_FAILED


def test_whisper_client_is_built_from_the_current_url_per_request(tmp_path):
    h = Harness(tmp_path)
    h.dictate(1)
    h.engines.whisper_url = "http://127.0.0.1:9100"
    h.dictate(2)
    assert [url for url, _ in h.whisper_factory.calls] == [
        "http://127.0.0.1:9001",
        "http://127.0.0.1:9100",
    ]


def test_transcript_none_delivers_nothing(tmp_path):
    whisper = FakeWhisperClient(responses=[{"text": "", "language": "english"}])
    h = Harness(tmp_path, whisper=whisper)
    h.dictate()
    assert h.llama.calls == []
    assert h.delivered_texts() == []
    assert h.history.entries == []
    assert h.last.pill is PillState.IDLE
    assert h.last.notice is None
    # The dictation still completed: press-to-pill and mic-open feed the Diagnostics medians.
    timings = h.last.timings
    assert timings is not None
    assert h.pipeline.recent_timings() == [timings]
    assert timings.release_to_transcript_ms is not None
    assert timings.release_to_cleaned_ms is None
    assert timings.delivery_ms is None
    assert timings.extra["nothing_to_deliver"] == 1.0


# Cleanup ---------------------------------------------------------------------------------


def test_gate_runs_cleanup_when_every_field_allows_it(tmp_path):
    h = Harness(tmp_path)
    h.dictate()
    assert len(h.llama.calls) == 1
    assert h.history.entries[-1].cleanup_reason == "ok"
    assert h.history.entries[-1].used_llm is True
    assert h.delivered_texts() == [CLEANED]


def test_gate_cleanup_enabled_field(tmp_path):
    h = Harness(tmp_path, settings=cleanup_settings(enabled=False))
    h.dictate()
    assert h.llama.calls == []
    assert h.history.entries[-1].cleanup_reason == "disabled"
    assert h.delivered_texts() == [LONG_TEXT]


def test_gate_profile_cleanup_field(tmp_path):
    h = Harness(tmp_path)
    h.ctx_process = "WindowsTerminal.exe"
    h.dictate()
    assert h.llama.calls == []
    assert h.history.entries[-1].cleanup_reason == "profile_off"
    assert h.history.entries[-1].outcome == "typed"


@pytest.mark.parametrize(
    "state",
    [
        EngineState.STARTING,
        EngineState.RESTARTING,
        EngineState.PAUSED,
        EngineState.UNLOADED,
        EngineState.FAILED,
    ],
)
def test_gate_engine_available_field(tmp_path, state):
    engines = FakeEngines()
    engines.states[LLAMA] = state
    h = Harness(tmp_path, engines=engines)
    h.dictate()
    assert h.llama.calls == []
    assert h.llama_factory.calls == []
    assert h.history.entries[-1].cleanup_reason == "engine_not_ready"


def test_gate_cpu_fallback_field(tmp_path):
    engines = FakeEngines()
    engines.states[LLAMA] = EngineState.CPU_FALLBACK
    engines.variants[LLAMA] = "cpu"
    h = Harness(tmp_path, engines=engines)
    h.dictate()
    assert h.llama.calls == []
    assert h.history.entries[-1].cleanup_reason == "cpu_fallback"


def test_a_cpu_served_llama_cleans_when_the_models_were_chosen_for_the_processor(tmp_path):
    engines = FakeEngines()
    engines.states[LLAMA] = EngineState.CPU_FALLBACK
    engines.variants[LLAMA] = "cpu"
    engines.cpu_cleanup_allowed = True
    h = Harness(tmp_path, engines=engines)
    h.dictate()
    assert len(h.llama.calls) == 1
    assert h.history.entries[-1].cleanup_reason == "ok"


def test_an_unexpected_fallback_still_skips_cleanup_on_a_vulkan_served_choice(tmp_path):
    engines = FakeEngines()
    engines.cpu_cleanup_allowed = True
    h = Harness(tmp_path, engines=engines)
    h.dictate()
    assert h.history.entries[-1].cleanup_reason == "ok"
    engines.states[LLAMA] = EngineState.CPU_FALLBACK
    engines.variants[LLAMA] = "cpu"
    engines.cpu_cleanup_allowed = False
    h.dictate()
    assert h.history.entries[-1].cleanup_reason == "cpu_fallback"


def test_llama_client_is_built_with_the_configured_timeout(tmp_path):
    h = Harness(tmp_path, settings=cleanup_settings(timeout_ms=4000))
    h.dictate()
    assert h.llama_factory.calls == [("http://127.0.0.1:9002", 4.0)]


def test_missing_llama_url_skips_cleanup(tmp_path):
    engines = FakeEngines()
    engines.llama_url = None  # inconsistent with READY on purpose: the URL is what counts
    h = Harness(tmp_path, engines=engines)
    h.dictate()
    assert h.llama_factory.calls == []
    assert h.history.entries[-1].cleanup_reason == "engine_not_ready"
    assert h.delivered_texts() == [LONG_TEXT]


def test_a_missing_cleanup_model_skips_cleanup_and_delivers_the_raw_text(tmp_path):
    """The real supervisor without a cleanup model, on fake engine processes (spec 14.1, 16).

    llama is FAILED with reason no_model and never launched, so it has no URL; the gate
    reports engine_not_ready and the raw transcript is delivered.
    """
    from spells.engines import EngineSupervisor

    from .fake_process import FakeClock as EngineClock
    from .fake_process import FakeWorld
    from .test_engines import GPU, make_paths

    world = FakeWorld()
    engine_clock = EngineClock()
    supervisor = EngineSupervisor(
        replace(make_paths(tmp_path / "install"), llama_model=None),
        GPU,
        lambda engine, state, reason: None,
        process_backend=world,
        whisper_client_factory=world.whisper_client,
        llama_client_factory=world.llama_client,
        clock=engine_clock.now,
        sleeper=engine_clock.sleep,
    )
    for _ in range(50):
        supervisor.step(WHISPER)
        supervisor.step(LLAMA)
        if supervisor.status(WHISPER) is EngineState.READY:
            break
    assert supervisor.status(WHISPER) is EngineState.READY
    assert (supervisor.status(LLAMA), supervisor.reason(LLAMA)) == (EngineState.FAILED, "no_model")
    assert supervisor.llama_url is None and supervisor.cleanup_available is False

    h = Harness(tmp_path, engines=supervisor)
    h.dictate()
    assert world.spawned("llama") == []
    assert h.llama_factory.calls == []
    assert h.llama.calls == []
    assert h.history.entries[-1].cleanup_reason == "engine_not_ready"
    assert h.delivered_texts() == [LONG_TEXT]
    assert h.last.notice is not Notice.ERROR


def test_guard_rejection_is_counted_per_reason(tmp_path):
    h = Harness(tmp_path, llama=FakeLlamaClient(content=""))
    h.dictate(1)
    h.dictate(2)
    assert h.pipeline.guard_counts() == {"empty": 2}
    assert h.delivered_texts() == [LONG_TEXT, " " + LONG_TEXT]
    assert h.history.entries[-1].cleanup_reason == "empty"


def test_gate_skips_are_not_counted_as_guard_rejections(tmp_path):
    h = Harness(tmp_path, settings=cleanup_settings(enabled=False))
    h.dictate()
    assert h.pipeline.guard_counts() == {}


def test_tone_comes_from_the_cleanup_settings(tmp_path):
    def mutate(s):
        return replace(s, cleanup=replace(s.cleanup, tones={**s.cleanup.tones, "Default": "very casual"}))

    h = Harness(tmp_path, settings=mutate)
    h.dictate()
    assert "very casual" in h.llama.calls[0][1]


def test_terms_are_passed_oldest_first(tmp_path):
    def mutate(s):
        terms = [TermEntry("Zeta", 2.0), TermEntry("Alpha", 1.0)]
        return replace(s, vocabulary=replace(s.vocabulary, terms=terms))

    h = Harness(tmp_path, settings=mutate)
    h.dictate()
    assert h.whisper.calls[0]["prompt"] == "Alpha, Zeta"
    assert "Alpha" in h.llama.calls[0][1]


# Post-processing and delivery ----------------------------------------------------------------


def test_postprocess_receives_the_last_delivery_and_the_clock(tmp_path):
    h = Harness(tmp_path)
    h.dictate(1)
    h.dictate(2)
    h.clock.advance(61.0)
    h.dictate(3)
    assert h.delivered_texts() == [CLEANED, " " + CLEANED, CLEANED]


def test_last_delivery_is_updated_only_for_pasted_and_typed(tmp_path):
    h = Harness(tmp_path, foreground=OTHER_HWND)
    h.dictate(1)
    assert h.history.entries[-1].outcome == "copied_focus_changed"
    h.backends.window.foreground = TARGET_HWND
    h.dictate(2)
    assert h.delivered_texts() == [CLEANED, CLEANED]
    h.ctx_process = "WindowsTerminal.exe"
    h.clock.advance(61.0)
    h.dictate(3)
    h.dictate(4)
    # 3 was typed after the 60 s window and counts for 4; a copy never counts.
    assert h.delivered_texts()[-2:] == [LONG_TEXT, " " + LONG_TEXT]


@pytest.mark.parametrize(
    "kwargs, outcome, notice, notice_text, notification",
    [
        ({}, "pasted", None, "", None),
        ({"foreground": OTHER_HWND}, "copied_focus_changed", Notice.COPIED, "Copied", COPIED_NOTICE),
        ({"elevated": True}, "copied_elevated", Notice.COPIED, "Copied", COPIED_NOTICE),
        ({"input_error": ("send_ctrl_v",)}, "failed", Notice.COPIED, "Copied", COPIED_NOTICE),
        ({"clipboard_error": ("snapshot",)}, "failed", Notice.ERROR, DELIVERY_FAILED, None),
    ],
)
def test_delivery_outcomes_map_to_pill_and_notification(
    tmp_path, kwargs, outcome, notice, notice_text, notification
):
    h = Harness(tmp_path, **kwargs)
    h.dictate()
    assert h.history.entries[-1].outcome == outcome
    assert h.last.notice is notice
    assert h.last.notice_text == notice_text
    if notification is None:
        assert h.last.notification is None or h.last.notification.startswith(DELIVERY_FAILED)
    else:
        assert h.last.notification == notification
    assert h.last.pill is PillState.IDLE
    assert h.last.tray is TrayState.READY
    assert h.last.timings is not None


def test_typed_delivery_for_the_terminal_profile(tmp_path):
    h = Harness(tmp_path)
    h.ctx_process = "pwsh.exe"
    h.dictate()
    assert ("type_unicode", LONG_TEXT) in h.inject_log
    assert h.history.entries[-1].outcome == "typed"
    assert h.last.notice is None


# History and timings ----------------------------------------------------------------------


def test_history_failure_never_fails_a_dictation(tmp_path):
    h = Harness(tmp_path, history=FakeHistory(error=RuntimeError("disk full")))
    h.dictate()
    assert h.delivered_texts() == [CLEANED]
    assert h.last.timings is not None
    assert h.last.notice is None
    assert len(h.pipeline.recent_timings()) == 1


def test_history_entry_contents(tmp_path):
    h = Harness(tmp_path)
    h.dictate()
    entry = h.history.entries[-1]
    assert entry.id is None
    assert entry.created_at > 1_600_000_000
    assert entry.raw_text == LONG_TEXT
    assert entry.cleaned_text == CLEANED
    assert entry.delivered_text == CLEANED
    assert entry.app_process == "notepad.exe"
    assert entry.app_title == "Untitled - Notepad"
    assert entry.language == "en"
    assert entry.used_llm is True
    assert entry.cleanup_reason == "ok"
    assert entry.outcome == "pasted"
    assert isinstance(entry.timings, StageTimings)
    assert entry.timings is h.last.timings


def test_stage_timings_come_from_the_injected_clock(tmp_path):
    h = Harness(tmp_path)
    h.capture_advance_s = 0.005
    h.whisper.on_call = lambda: h.clock.advance(0.4)
    h.llama.on_call = lambda: h.clock.advance(0.3)
    h.dictate()
    timings = h.last.timings
    assert timings.press_to_pill_ms == pytest.approx(5.0)
    assert timings.mic_open_ms == pytest.approx(37.0)
    assert timings.release_to_transcript_ms == pytest.approx(400.0)
    assert timings.release_to_cleaned_ms == pytest.approx(700.0)
    assert timings.delivery_ms == pytest.approx(0.0)
    assert timings.extra["release_to_delivered_ms"] == pytest.approx(700.0)
    assert timings.extra["audio_s"] == pytest.approx(2.0)


def test_timing_ring_keeps_the_last_twenty(tmp_path):
    h = Harness(tmp_path)
    for dictation_id in range(1, 26):
        h.dictate(dictation_id)
    ring = h.pipeline.recent_timings()
    assert len(ring) == 20
    assert len(h.history.entries) == 25
    assert ring[-1] is h.history.entries[-1].timings
    assert ring[0] is h.history.entries[5].timings


# Retry last dictation -----------------------------------------------------------------------


def test_retry_last_delivers_to_the_clipboard_and_keeps_the_pcm_until_done(tmp_path):
    whisper = FakeWhisperClient(responses=[WhisperError("reset")], healthy=True)
    h = Harness(tmp_path, whisper=whisper)
    h.dictate()
    assert h.pipeline.retry_available is True
    whisper.responses = [OK_RESPONSE]
    assert h.pipeline.retry_last() is True
    assert h.pipeline.retry_available is True, "the PCM stays until the attempt completes"
    assert h.last.pill is PillState.PROCESSING
    h.process()
    assert ("set_text", CLEANED) in h.inject_log
    assert ("send_ctrl_v",) not in h.inject_log
    assert h.last.notice is Notice.COPIED
    assert h.last.notification == COPIED_NOTICE
    assert h.last.retry_available is False
    assert h.pipeline.retry_available is False
    assert h.pipeline.retry_last() is False
    assert h.history.entries[-1].outcome == "copied_focus_changed"
    assert h.history.entries[-1].timings.extra["retry"] == 1.0
    assert h.pipeline.recent_timings() == [], "a retry never enters the Diagnostics ring"


def test_failed_retry_drops_the_pcm(tmp_path):
    whisper = FakeWhisperClient(responses=[WhisperError("reset")], healthy=True)
    h = Harness(tmp_path, whisper=whisper)
    h.dictate()
    assert h.pipeline.retry_last() is True
    h.process()
    assert h.last.notice_text == TRANSCRIPTION_FAILED
    assert h.pipeline.retry_available is False


def test_next_dictation_drops_the_retry_pcm(tmp_path):
    whisper = FakeWhisperClient(responses=[WhisperError("reset")], healthy=True)
    h = Harness(tmp_path, whisper=whisper)
    h.dictate(1)
    assert h.pipeline.retry_available is True
    h.press(2)
    assert h.pipeline.retry_available is False
    assert h.last.retry_available is False
    assert h.pipeline.retry_last() is False


# Recorder lifecycle ----------------------------------------------------------------------


def test_mic_device_change_reopens_the_recorder(tmp_path):
    h = Harness(tmp_path)
    h.prime()
    first = h.recorder
    assert first.device is None
    h.config.update(general(mic_device="USB Mic"))
    h.pipeline.drain_controller()
    assert first.closed
    assert h.recorder is not first
    assert h.recorder.device == "USB Mic"


def test_mic_device_change_during_a_recording_waits_for_it_to_end(tmp_path):
    h = Harness(tmp_path)
    h.prime()
    first = h.recorder
    h.press(1)
    h.config.update(general(mic_device="USB Mic"))
    h.pipeline.drain_controller()
    assert h.recorder is first and not first.closed
    h.release(1)
    h.press(2)
    assert first.closed
    assert h.recorder.device == "USB Mic"
    h.release(2)
    h.process()
    assert len(h.whisper.calls) == 2


def test_keep_mic_warm_keeps_the_stream_open(tmp_path):
    h = Harness(tmp_path, settings=general(keep_mic_warm=True))
    h.prime()
    recorder = h.recorder
    assert recorder.keep_open is True
    assert recorder.calls == ["start", "cancel"]
    assert recorder.stream_open
    h.dictate()
    assert h.recorder is recorder
    assert recorder.stream_open
    assert h.last.timings.mic_open_ms == pytest.approx(0.0)


def test_keep_mic_warm_off_closes_the_stream_after_each_recording(tmp_path):
    h = Harness(tmp_path)
    h.prime()
    assert h.recorder.calls == []
    h.dictate()
    assert not h.recorder.stream_open


# Threads and events -----------------------------------------------------------------------


def test_stop_joins_the_threads_and_closes_the_recorder(tmp_path):
    h = Harness(tmp_path)
    h.pipeline.start()
    assert h.pipeline.running
    h.pipeline.stop()
    h.pipeline.stop()
    assert not h.pipeline.running
    names = {t.name for t in threading.enumerate()}
    assert not any(name.startswith("spells-") for name in names)
    assert h.recorders and h.recorder.closed


def test_stop_during_a_recording_cancels_it_and_joins(tmp_path):
    recording = threading.Event()
    events = []

    def on_event(event):
        events.append(event)
        if event.pill is PillState.RECORDING:
            recording.set()

    h = Harness(tmp_path, on_event=on_event)
    h.pipeline.start()
    h.callbacks.pressed(MAIN, 1)
    assert recording.wait(5.0)
    h.pipeline.stop()
    assert not any(t.name.startswith("spells-") for t in threading.enumerate())
    assert h.recorder.calls == ["start", "cancel", "close"]
    assert h.hotkey.ended == [1]
    assert events[-1].pill is PillState.IDLE
    assert events[-1].target_hwnd is None


def test_stop_from_an_event_handler_does_not_join_itself(tmp_path):
    done = threading.Event()

    def on_event(event):
        h.events.append(event)
        if event.pill is PillState.RECORDING:
            h.pipeline.stop()  # runs on the controller thread: it must not join itself
            done.set()

    h = Harness(tmp_path, on_event=on_event)
    h.pipeline.start()
    threads = [t for t in threading.enumerate() if t.name.startswith("spells-")]
    assert len(threads) == 2
    h.callbacks.pressed(MAIN, 1)
    assert done.wait(5.0)
    for thread in threads:
        thread.join(5.0)
    assert not any(t.is_alive() for t in threads)
    assert not h.pipeline.running
    assert h.recorder.closed
    assert h.hotkey.ended == [1]


def test_on_event_exceptions_are_caught(tmp_path):
    def broken(event):
        raise RuntimeError("UI gone")

    h = Harness(tmp_path, on_event=broken)
    h.dictate()
    assert h.delivered_texts() == [CLEANED]
    assert len(h.history.entries) == 1


def test_worker_survives_an_unexpected_exception(tmp_path):
    h = Harness(tmp_path)
    h.llama_factory.error = RuntimeError("boom")
    h.dictate(1)
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == DICTATION_FAILED
    assert h.last.pill is PillState.IDLE
    h.llama_factory.error = None
    h.dictate(2)
    assert h.delivered_texts() == [CLEANED]


def test_hotkey_may_be_given_as_a_bare_end_recording_callable(tmp_path):
    ended: list[int] = []
    h = Harness(tmp_path)
    pipeline = Pipeline(
        config=h.config,
        engines=h.engines,
        hotkey=ended.append,
        history=h.history,
        on_event=h.events.append,
        recorder_factory=h._recorder,
        capture=h._capture,
        inject_backends=h.backends,
        whisper_client_factory=h.whisper_factory,
        llama_client_factory=h.llama_factory,
        scheduler=h.scheduler,
        clock=h.clock.now,
        sleeper=h.sleeps.append,
    )
    callbacks = pipeline.hotkey_callbacks()
    callbacks.pressed(MAIN, 7)
    callbacks.cancelled(7)
    pipeline.drain_controller()
    assert ended == [7]


# Several speech engines (B5-13 to B5-16) -----------------------------------------------------------


QWEN = EngineId(WHISPER, 0)
FLUTRA = EngineId(WHISPER, 1)
LONG_PCM = b"\x01\x00" * (NATIVE_SAMPLE_RATE * 12)


class RoutedEngines(FakeEngines):
    def __init__(self) -> None:
        super().__init__()
        self.slots = (
            SpeechSlot(QWEN, "llama-asr", ("de", "en")),
            SpeechSlot(FLUTRA, "whisper-server", ("sq",)),
        )
        self.states.update({QWEN: EngineState.READY, FLUTRA: EngineState.READY})
        self.variants.update({QWEN: "cpu", FLUTRA: "cpu"})
        self.urls.update({QWEN: "http://127.0.0.1:9101", FLUTRA: "http://127.0.0.1:9102"})

    def url(self, engine) -> str | None:
        return self._url(engine)


def routed_harness(tmp_path, *, qwen=None, flutra=None, engines=None, **kwargs):
    engines = engines if engines is not None else RoutedEngines()
    flutra = flutra if flutra is not None else FakeWhisperClient(
        responses=[{"text": "Përshëndetje nga Prishtina", "language": "albanian"}]
    )
    qwen = qwen if qwen is not None else FakeLlamaAsrClient(
        AsrReply(LONG_TEXT, "en", "English"), reports="English"
    )
    asr_factory = ClientFactory(qwen)
    h = Harness(tmp_path, whisper=flutra, engines=engines, **kwargs)
    h.pipeline._llama_asr_factory = asr_factory
    h.qwen = qwen
    h.asr_factory = asr_factory
    h.recorder_setup = lambda recorder: setattr(recorder, "pcm", LONG_PCM)
    return h


def test_a_single_whisper_engine_keeps_the_classic_path(tmp_path):
    h = Harness(tmp_path)
    h.dictate()
    assert h.whisper_factory.calls == [("http://127.0.0.1:9001", None)]
    assert len(h.whisper.calls) == 1
    assert h.history.entries[-1].outcome == "pasted"


def test_english_goes_to_the_fast_engine_only(tmp_path):
    h = routed_harness(tmp_path)
    h.dictate()
    assert h.asr_factory.calls == [("http://127.0.0.1:9101", None)]
    assert h.whisper_factory.calls == []
    assert h.qwen.asked == ["English"]
    entry = h.history.entries[-1]
    assert (entry.raw_text, entry.language, entry.outcome) == (LONG_TEXT, "en", "pasted")


def test_albanian_is_rerouted_to_its_engine_with_the_language_locked(tmp_path):
    qwen = FakeLlamaAsrClient(AsrReply("", "hu", "Hungarian"), reports="Hungarian")
    h = routed_harness(tmp_path, qwen=qwen)
    h.dictate()
    assert qwen.stopped == 1
    assert h.whisper_factory.calls == [("http://127.0.0.1:9102", None)]
    assert h.whisper.calls[0]["language"] == "sq"
    assert h.history.entries[-1].language == "sq"
    assert h.pipeline._lang_state.last_accepted == "sq"


def test_a_starting_second_engine_is_waited_for_with_the_starting_pill(tmp_path):
    engines = RoutedEngines()
    engines.states[FLUTRA] = EngineState.STARTING
    qwen = FakeLlamaAsrClient(AsrReply("", "pl", "Polish"), reports="Polish")
    h = routed_harness(tmp_path, qwen=qwen, engines=engines)
    h.dictate()
    assert ("wait_ready", FLUTRA, 1.0) in engines.calls
    assert not any(call[:2] == ("wait_ready", QWEN) for call in engines.calls)
    assert any(event.pill is PillState.STARTING_ENGINES for event in h.events)
    assert h.history.entries[-1].language == "sq"


def test_a_failed_second_engine_is_speech_engine_unavailable_with_the_audio_kept(tmp_path):
    engines = RoutedEngines()
    engines.states[FLUTRA] = EngineState.FAILED
    h = routed_harness(tmp_path, engines=engines, settings=general(language_mode="sq"))
    h.dictate()
    assert h.last.notice_text == ENGINE_UNAVAILABLE
    assert h.last.retry_available is True
    assert h.asr_factory.calls == []


def test_an_error_on_a_healthy_second_engine_fails_without_a_restart(tmp_path):
    flutra = FakeWhisperClient(responses=[WhisperError("HTTP 500")], healthy=True)
    h = routed_harness(tmp_path, flutra=flutra, settings=general(language_mode="sq"))
    h.dictate()
    assert flutra.health_calls == 1
    assert not any(call[0] == "restart" for call in h.engines.calls)
    assert h.last.notice_text == TRANSCRIPTION_FAILED


def test_a_dead_second_engine_is_restarted_alone_and_retried_once(tmp_path):
    flutra = FakeWhisperClient(
        responses=[WhisperError("reset"), {"text": "Mirëdita të gjithëve", "language": "albanian"}],
        healthy=False,
    )
    h = routed_harness(tmp_path, flutra=flutra, settings=general(language_mode="sq"))
    h.dictate()
    assert ("restart", FLUTRA) in h.engines.calls
    assert ("restart", WHISPER) not in h.engines.calls
    assert len(flutra.calls) == 2
    assert h.history.entries[-1].language == "sq"


def test_a_dead_fast_engine_is_restarted_by_its_id(tmp_path):
    qwen = FakeLlamaAsrClient(WhisperError("reset"), AsrReply(LONG_TEXT, "en", "English"),
                              healthy=False)
    h = routed_harness(tmp_path, qwen=qwen, settings=general(language_mode="en"))
    h.dictate()
    assert ("restart", QWEN) in h.engines.calls
    assert len(qwen.calls) == 2
    assert qwen.calls[0]["language"] == "en"
    assert h.history.entries[-1].raw_text == LONG_TEXT


def test_processor_hardware_delivers_clean_text_raw(tmp_path):
    engines = FakeEngines()
    engines.cpu_only = True
    h = Harness(tmp_path, engines=engines)
    h.dictate()
    assert h.llama.calls == []
    assert h.history.entries[-1].cleanup_reason == "clean_text"
    assert h.delivered_texts() == [LONG_TEXT]


def test_processor_hardware_cleans_a_transcript_with_a_filler(tmp_path):
    engines = FakeEngines()
    engines.cpu_only = True
    whisper = FakeWhisperClient(responses=[{"text": "um " + LONG_TEXT, "language": "english"}])
    h = Harness(tmp_path, engines=engines, whisper=whisper)
    h.dictate()
    assert len(h.llama.calls) == 1
    assert h.history.entries[-1].cleanup_reason == "ok"


def test_a_language_the_cleanup_model_does_not_score_is_delivered_raw(tmp_path):
    engines = FakeEngines()
    engines.cleanup_languages = frozenset({"de"})
    h = Harness(tmp_path, engines=engines)
    h.dictate()
    assert h.llama.calls == []
    assert h.history.entries[-1].cleanup_reason == "language_unscored"


# Quality signals on the history row (spec 8.4)


SCORED_RESPONSE = {
    "text": LONG_TEXT,
    "language": "english",
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "avg_logprob": -0.22,
            "compression_ratio": 1.5,
            "no_speech_prob": 0.01,
            "temperature": 0.0,
        }
    ],
}
POOR_RESPONSE = {
    "text": LONG_TEXT,
    "language": "english",
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "avg_logprob": -1.6,
            "compression_ratio": 1.5,
            "no_speech_prob": 0.02,
            "temperature": 0.0,
        }
    ],
}


def test_the_history_row_carries_the_engine_scores(tmp_path):
    h = Harness(tmp_path, whisper=FakeWhisperClient(responses=[SCORED_RESPONSE]))
    h.prime()
    h.dictate()
    signals = h.history.entries[0].signals
    assert signals.avg_logprob == pytest.approx(-0.22)
    assert signals.compression_ratio == pytest.approx(1.5)
    assert signals.source == "whisper-server"
    assert signals.audio_s == pytest.approx(2.0)
    assert signals.word_count == 22


def unhurried(recorder) -> None:
    recorder.pcm = recorder.pcm * 6


def test_the_history_row_carries_the_label_and_its_reason(tmp_path):
    h = Harness(
        tmp_path,
        whisper=FakeWhisperClient(responses=[SCORED_RESPONSE]),
        recorder_setup=unhurried,
    )
    h.prime()
    h.dictate()
    entry = h.history.entries[0]
    assert entry.quality_label == "good"
    assert entry.quality_reason.endswith(".")
    assert entry.signals.words_per_minute == pytest.approx(110.0)


def test_racing_through_a_dictation_is_labelled_uncertain(tmp_path):
    h = Harness(tmp_path, whisper=FakeWhisperClient(responses=[SCORED_RESPONSE]))
    h.prime()
    h.dictate()
    entry = h.history.entries[0]
    assert entry.quality_label == "uncertain"
    assert "words per minute" in entry.quality_reason


def test_a_bad_decode_is_labelled_poor(tmp_path):
    h = Harness(tmp_path, whisper=FakeWhisperClient(responses=[POOR_RESPONSE]))
    h.prime()
    h.dictate()
    entry = h.history.entries[0]
    assert entry.quality_label == "poor"
    assert "confidence" in entry.quality_reason


def test_the_row_counts_the_fillers_of_its_language(tmp_path):
    spoken = {"text": "um so I mean we should um ship it", "language": "english"}
    h = Harness(tmp_path, whisper=FakeWhisperClient(responses=[spoken]))
    h.prime()
    h.dictate()
    signals = h.history.entries[0].signals
    assert signals.filler_count == 3
    assert signals.words_per_minute == pytest.approx(signals.word_count * 30.0)


def test_the_row_names_the_engines_that_served_it(tmp_path):
    h = Harness(tmp_path, whisper=FakeWhisperClient(responses=[SCORED_RESPONSE]))
    h.prime()
    h.dictate()
    entry = h.history.entries[0]
    assert entry.asr_engine == "whisper"
    assert entry.cleanup_engine == "llama"


def test_the_row_names_no_cleanup_engine_when_cleanup_was_skipped(tmp_path):
    h = Harness(tmp_path, settings=cleanup_settings(enabled=False))
    h.prime()
    h.dictate()
    entry = h.history.entries[0]
    assert entry.cleanup_engine == ""
    assert entry.cleanup_reason == "disabled"


def test_a_llama_asr_engine_records_its_runtime_and_no_scores(tmp_path):
    h = routed_harness(tmp_path)
    h.prime()
    h.dictate()
    entry = h.history.entries[-1]
    assert entry.signals.source == "llama-asr"
    assert entry.signals.avg_logprob is None
    assert entry.signals.word_count > 0
    assert entry.quality_label == "good"
    assert "no confidence scores" in entry.quality_reason
    assert entry.asr_engine == "whisper"


# Keeping the recording (spec 15, 17)


def test_no_audio_reaches_the_store_by_default(tmp_path):
    h = Harness(tmp_path)
    h.prime()
    h.dictate()
    assert h.history.audio == [None]
    assert h.history.policies == [None]


def test_the_pcm_reaches_the_store_when_the_user_asked_for_it(tmp_path):
    h = Harness(tmp_path, settings=history_settings(keep_audio=True))
    h.prime()
    h.dictate()
    assert h.history.audio[0] == h.recorder.pcm
    policy = h.history.policies[0]
    assert policy.keep is True
    assert policy.max_files == 200
    assert policy.max_mb == 1000
    assert h.history.rates == [NATIVE_SAMPLE_RATE]


def test_the_store_gets_the_limits_the_user_set(tmp_path):
    h = Harness(
        tmp_path,
        settings=history_settings(keep_audio=True, audio_keep_count=10, audio_keep_mb=25),
    )
    h.prime()
    h.dictate()
    policy = h.history.policies[0]
    assert (policy.max_files, policy.max_mb) == (10, 25)


def test_a_history_error_still_lets_the_dictation_finish(tmp_path):
    h = Harness(
        tmp_path,
        history=FakeHistory(error=RuntimeError("disk full")),
        settings=history_settings(keep_audio=True),
    )
    h.prime()
    h.dictate()
    assert h.delivered_texts() == [CLEANED]


# The on-demand check (spec 8.4)


def test_check_transcript_returns_the_parsed_verdict(tmp_path):
    llama = FakeLlamaClient(content="VERDICT: GARBLED\nREASON: the words do not fit together")
    h = Harness(tmp_path, llama=llama)
    result = h.pipeline.check_transcript("krbl mmm what", "en")
    assert result.ok
    assert result.verdict == "GARBLED"
    assert result.reason == "the words do not fit together"


def test_check_transcript_sends_the_fixed_prompt_and_the_language_name(tmp_path):
    llama = FakeLlamaClient(content="VERDICT: GOOD")
    h = Harness(tmp_path, llama=llama)
    h.pipeline.check_transcript("hello there", "de")
    system, user, max_tokens = llama.calls[0]
    assert system == quality.CHECK_SYSTEM_PROMPT
    assert "LANGUAGE: German" in user
    assert "hello there" in user
    assert max_tokens == quality.CHECK_MAX_TOKENS


def test_check_transcript_uses_its_own_timeout_not_the_cleanup_one(tmp_path):
    h = Harness(tmp_path, llama=FakeLlamaClient(content="VERDICT: GOOD"))
    h.pipeline.check_transcript("hello there", "en")
    assert h.llama_factory.calls[-1][1] == CHECK_TIMEOUT_S


def test_check_transcript_skips_politely_when_the_engine_is_not_serving(tmp_path):
    engines = FakeEngines()
    engines.states[LLAMA] = EngineState.STARTING
    h = Harness(tmp_path, engines=engines)
    result = h.pipeline.check_transcript("hello there", "en")
    assert result.status == "unavailable"
    assert h.llama.calls == []


def test_check_transcript_rejects_an_answer_in_the_wrong_shape(tmp_path):
    llama = FakeLlamaClient(content="Here is the cleaned text: hello there.")
    h = Harness(tmp_path, llama=llama)
    assert h.pipeline.check_transcript("hello there", "en").status == "unreadable"


def test_check_transcript_reports_a_timeout(tmp_path):
    llama = FakeLlamaClient(error=CleanupError("too slow", reason="timeout"))
    h = Harness(tmp_path, llama=llama)
    assert h.pipeline.check_transcript("hello there", "en").status == "timeout"


def test_check_transcript_survives_an_unexpected_failure(tmp_path):
    llama = FakeLlamaClient(error=RuntimeError("boom"))
    h = Harness(tmp_path, llama=llama)
    assert h.pipeline.check_transcript("hello there", "en").status == "error"


def test_check_transcript_ignores_an_empty_transcript(tmp_path):
    h = Harness(tmp_path)
    assert h.pipeline.check_transcript("   ", "en").status == "empty"
    assert h.llama.calls == []


def test_check_transcript_never_runs_during_a_dictation(tmp_path):
    h = Harness(tmp_path)
    h.prime()
    h.dictate()
    cleanup_calls = len(h.llama.calls)
    assert cleanup_calls == 1
    assert all(quality.CHECK_SYSTEM_PROMPT not in call[0] for call in h.llama.calls)


# A microphone that delivered nothing (B5-42)


def mute(recorder) -> None:
    recorder.pcm = b""


def barely(recorder) -> None:
    recorder.pcm = bytes(2 * 1000)


def dropping(blocks: int):
    def setup(recorder) -> None:
        recorder.dropped_blocks = blocks

    return setup


def test_an_empty_recording_never_reaches_the_speech_engine(tmp_path):
    h = Harness(tmp_path, recorder_setup=mute)
    h.prime()
    h.dictate()
    assert h.whisper.calls == []
    assert h.llama.calls == []
    assert h.delivered_texts() == []


def test_a_recording_under_a_tenth_of_a_second_is_treated_the_same(tmp_path):
    h = Harness(tmp_path, recorder_setup=barely)
    h.prime()
    h.dictate()
    assert h.whisper.calls == []


def test_a_tenth_of_a_second_of_audio_is_transcribed(tmp_path):
    def just_enough(recorder) -> None:
        recorder.pcm = bytes(2 * (recorder.sample_rate // 10))

    h = Harness(tmp_path, recorder_setup=just_enough)
    h.prime()
    h.dictate()
    assert len(h.whisper.calls) == 1


def test_the_user_is_told_what_actually_happened(tmp_path):
    h = Harness(tmp_path, recorder_setup=mute)
    h.prime()
    h.dictate()
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == NO_AUDIO_TEXT
    assert h.last.notice_text != TRANSCRIPTION_FAILED
    assert h.last.notification == NO_AUDIO_TEXT
    assert h.last.notification_action == SOUND_SETTINGS


def test_the_empty_recording_is_not_offered_for_a_retry(tmp_path):
    h = Harness(tmp_path, recorder_setup=mute)
    h.prime()
    h.dictate()
    assert h.pipeline.retry_available is False
    assert h.last.retry_available is False
    assert h.pipeline.retry_last() is False


def test_the_empty_recording_is_recorded_with_a_reason_of_its_own(tmp_path):
    h = Harness(tmp_path, recorder_setup=mute)
    h.prime()
    h.dictate()
    entry = h.history.entries[0]
    assert entry.outcome == "no_audio"
    assert entry.raw_text == ""
    assert entry.app_process == "notepad.exe"
    assert entry.quality_label == "poor"
    assert entry.quality_reason == NO_AUDIO_REASON
    assert entry.timings.extra["no_audio"] == 1.0
    assert entry.timings.release_to_transcript_ms is None


def test_no_recording_is_kept_for_an_empty_dictation(tmp_path):
    h = Harness(tmp_path, recorder_setup=mute, settings=history_settings(keep_audio=True))
    h.prime()
    h.dictate()
    assert h.history.audio == [None]


def test_the_empty_case_is_logged_once_with_the_device(tmp_path, caplog):
    h = Harness(tmp_path, recorder_setup=mute, settings=general(mic_device="Jabra Evolve2"))
    h.prime()
    with caplog.at_level("WARNING", logger="spells.pipeline"):
        h.dictate()
    lines = [r.getMessage() for r in caplog.records if "delivered" in r.getMessage()]
    assert len(lines) == 1
    assert "Jabra Evolve2" in lines[0]


def test_the_default_device_is_named_when_none_was_chosen(tmp_path, caplog):
    h = Harness(tmp_path, recorder_setup=mute)
    h.prime()
    with caplog.at_level("WARNING", logger="spells.pipeline"):
        h.dictate()
    lines = [r.getMessage() for r in caplog.records if "delivered" in r.getMessage()]
    assert "default device" in lines[0]


def test_the_next_dictation_works_normally(tmp_path):
    h = Harness(tmp_path, recorder_setup=mute)
    h.prime()
    h.dictate()
    h.recorder.pcm = bytes(2 * 32000)
    h.press(2)
    h.release(2)
    h.process()
    assert h.delivered_texts() == [CLEANED]
    assert h.events_for(2)[-1].notice is None


# Blocks the sound card dropped


def test_dropped_blocks_reach_the_timings_and_the_row(tmp_path):
    h = Harness(tmp_path, recorder_setup=dropping(4))
    h.prime()
    h.dictate()
    entry = h.history.entries[0]
    assert entry.timings.extra["dropped_blocks"] == 4.0
    assert entry.signals.dropped_blocks == 4


def test_dropped_blocks_hold_the_label_back(tmp_path):
    def setup(recorder) -> None:
        recorder.dropped_blocks = 2
        recorder.pcm = recorder.pcm * 6

    h = Harness(
        tmp_path, whisper=FakeWhisperClient(responses=[SCORED_RESPONSE]), recorder_setup=setup
    )
    h.prime()
    h.dictate()
    entry = h.history.entries[0]
    assert entry.quality_label == "uncertain"
    assert "dropped" in entry.quality_reason


def test_the_user_is_told_that_audio_was_lost(tmp_path):
    h = Harness(tmp_path, recorder_setup=dropping(1))
    h.prime()
    h.dictate()
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == AUDIO_LOST_TEXT
    assert h.last.notification == AUDIO_LOST_NOTIFICATION
    assert h.last.notification_action is None


def test_a_clean_dictation_says_nothing_about_lost_audio(tmp_path):
    h = Harness(tmp_path)
    h.prime()
    h.dictate()
    assert h.last.notice is None
    assert "dropped_blocks" not in h.history.entries[0].timings.extra
    assert h.history.entries[0].signals.dropped_blocks == 0


def test_the_copied_notice_wins_over_the_lost_audio_one(tmp_path):
    h = Harness(tmp_path, recorder_setup=dropping(3), foreground=OTHER_HWND)
    h.prime()
    h.dictate()
    assert h.last.notice is Notice.COPIED
    assert h.last.notification == COPIED_NOTICE
    assert h.history.entries[0].signals.dropped_blocks == 3


def test_an_empty_recording_with_dropped_blocks_reports_both(tmp_path):
    def setup(recorder) -> None:
        recorder.pcm = b""
        recorder.dropped_blocks = 7

    h = Harness(tmp_path, recorder_setup=setup)
    h.prime()
    h.dictate()
    entry = h.history.entries[0]
    assert entry.outcome == "no_audio"
    assert entry.timings.extra["dropped_blocks"] == 7.0
    assert h.last.notice_text == NO_AUDIO_TEXT


# The capture path (spec 5.2 audio row, 14.4 Diagnostics)


def wav_rate(wav: bytes) -> int:
    with wave.open(io.BytesIO(wav)) as handle:
        return handle.getframerate()


def test_the_recorder_is_asked_for_the_devices_own_rate(tmp_path):
    h = Harness(tmp_path)
    h.press()
    assert h.recorder.requested_sample_rate is None
    assert h.recorder.sample_rate == NATIVE_SAMPLE_RATE


def test_the_captured_rate_reaches_the_wav_and_the_duration(tmp_path):
    h = Harness(tmp_path)
    h.prime()
    h.dictate()
    assert wav_rate(h.whisper.calls[0]["wav"]) == NATIVE_SAMPLE_RATE
    assert h.history.entries[0].signals.audio_s == pytest.approx(2.0)


def test_a_recorder_at_another_rate_is_followed(tmp_path):
    def slow(recorder) -> None:
        recorder.sample_rate = 44100
        recorder.pcm = b"\x01\x00" * (44100 * 3)

    h = Harness(tmp_path, recorder_setup=slow, settings=history_settings(keep_audio=True))
    h.prime()
    h.dictate()
    assert wav_rate(h.whisper.calls[0]["wav"]) == 44100
    assert h.history.rates == [44100]
    assert h.history.entries[0].signals.audio_s == pytest.approx(3.0)


def test_the_capture_of_each_dictation_reaches_diagnostics(tmp_path):
    def captured(recorder) -> None:
        recorder.device_name = "Microphone (HyperX Cloud III Wireless)"
        recorder.dropped_blocks = 2

    h = Harness(tmp_path, recorder_setup=captured)
    h.prime()
    h.dictate(1)
    h.dictate(2)
    captures = h.pipeline.recent_captures()
    assert len(captures) == 2
    assert captures[-1].host_api == "Windows WASAPI"
    assert captures[-1].sample_rate == NATIVE_SAMPLE_RATE
    assert captures[-1].device == "Microphone (HyperX Cloud III Wireless)"
    assert captures[-1].dropped_blocks == 2


def test_a_recorder_without_capture_info_is_tolerated(tmp_path):
    class Bare(FakeRecorder):
        capture_info = None

    h = Harness(tmp_path, recorder_factory=lambda **kwargs: Bare(**kwargs))
    h.prime()
    h.dictate()
    assert h.pipeline.recent_captures() == []
    assert h.history.entries[0].outcome == "pasted"


def test_a_truncated_device_name_is_rewritten_to_the_full_one(tmp_path):
    def moved(recorder) -> None:
        recorder.device_match = "prefix"
        recorder.device_name = "Microphone (HyperX Cloud III Wireless)"

    h = Harness(
        tmp_path,
        recorder_setup=moved,
        settings=general(mic_device="Microphone (HyperX Cloud III Wi"),
    )
    h.prime()
    h.dictate()
    assert h.config.settings.general.mic_device == "Microphone (HyperX Cloud III Wireless)"
    assert h.recorder.device == "Microphone (HyperX Cloud III Wireless)"
    assert h.last.notification is None


def test_a_full_name_is_not_rewritten(tmp_path):
    h = Harness(tmp_path, settings=general(mic_device="Fake Microphone"))
    h.prime()
    h.dictate()
    assert h.config.settings.general.mic_device == "Fake Microphone"


def test_a_device_that_is_gone_is_announced_once(tmp_path):
    def gone(recorder) -> None:
        recorder.device_match = "default"
        recorder.device_name = "OBSBOT Tiny 2 Lite Microphone (6- OBSBOT Tiny 2 Lite Audio)"
        recorder.fallback_notice = "The microphone \"Blue Yeti\" is gone, so Spells is recording from OBSBOT."

    h = Harness(tmp_path, recorder_setup=gone, settings=general(mic_device="Blue Yeti"))
    h.prime()
    h.press(1)
    assert "Blue Yeti" in (h.last.notification or "")
    assert h.last.notification_action == SOUND_SETTINGS
    assert h.last.pill is PillState.RECORDING
    h.release(1)
    h.process()
    h.press(2)
    assert h.last.notification is None
    h.release(2)
    h.process()
    assert h.config.settings.general.mic_device == "Blue Yeti"


def test_a_settings_write_that_fails_does_not_break_the_dictation(tmp_path):
    def moved(recorder) -> None:
        recorder.device_match = "prefix"
        recorder.device_name = "Microphone (HyperX Cloud III Wireless)"

    h = Harness(
        tmp_path,
        recorder_setup=moved,
        settings=general(mic_device="Microphone (HyperX Cloud III Wi"),
    )
    h.prime()
    h.pipeline._config = _BrokenConfig(h.config)
    h.dictate()
    assert h.history.entries[0].outcome == "pasted"


class _BrokenConfig:
    """A ConfigStore whose update() raises, to prove the dictation survives it."""

    def __init__(self, inner) -> None:
        self._inner = inner

    @property
    def settings(self):
        return self._inner.settings

    def update(self, mutator):
        raise OSError("settings file is read-only")

    def subscribe(self, callback):
        return self._inner.subscribe(callback)
