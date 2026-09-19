"""Live pipeline checks (slices of spec 20.2; brief 08).

The engines are real: the supervisor launches build/out/engines with the test-only models
in build/cache/models (ggml-tiny, Silero VAD, Qwen2.5-0.5B). Everything around the
pipeline is a fake so the desktop can stay locked: the hotkey callbacks are invoked from
the test thread, the recorder is fed from a speech clip synthesized with the Windows voice
(data/warmup.wav is still the generated tone of brief 02, which whisper rightly transcribes
to nothing) or from a generated tone, the target context is fixed and the inject backends
only record what would have been pasted. Skips with a reason when a binary, a model or the
speech synthesizer is missing.

Run with:

    .venv/Scripts/python.exe -m pytest -m integration tests/integration/test_pipeline_live.py -s
"""

from __future__ import annotations

import math
import statistics
import struct
import subprocess
import sys
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from unit.fake_recorder import FakeRecorder

from spells.asr import WhisperClient
from spells.config import ConfigStore
from spells.engines import EnginePaths, EngineSupervisor
from spells.gpu import select_device
from spells.history import HistoryStore
from spells.inject import InjectBackends
from spells.models import Chord, Engine, EngineState, StageTimings, TargetContext
from spells.pipeline import PillState, Pipeline, PipelineEvent

pytestmark = pytest.mark.integration

REPO_DIR = Path(__file__).resolve().parents[2]
ENGINES_DIR = REPO_DIR / "build" / "out" / "engines"
MODELS_DIR = REPO_DIR / "build" / "cache" / "models"
WHISPER_TINY = MODELS_DIR / "ggml-tiny.bin"
VAD_MODEL = MODELS_DIR / "ggml-silero-v5.1.2.bin"
TINY_LLAMA = MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
SPEECH_TEXT = "Hello, this is a test of the dictation pipeline. Please deliver this text."
READY_TIMEOUT_S = 90.0
DICTATION_TIMEOUT_S = 30.0
CREATE_NO_WINDOW = 0x08000000
SAMPLE_RATE = 16000
TARGET_HWND = 0x00AA00BB
MAIN = Chord(keys=(0x11, 0x5B))

WHISPER = Engine.WHISPER
LLAMA = Engine.LLAMA


# Helpers -----------------------------------------------------------------------------


def _missing() -> list[str]:
    missing = []
    for variant in ("vulkan", "cpu"):
        for exe in ("whisper-server.exe", "llama-server.exe"):
            path = ENGINES_DIR / variant / exe
            if not path.exists():
                missing.append(str(path.relative_to(REPO_DIR)))
    for model in (WHISPER_TINY, VAD_MODEL, TINY_LLAMA):
        if not model.exists():
            missing.append(str(model.relative_to(REPO_DIR)))
    return missing


def _alive(pid: int) -> bool:
    from spells.win32.window import process_image_path

    return process_image_path(pid) != ""


def _wait(predicate, timeout_s: float, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval_s)


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as reader:
        assert (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) == (
            SAMPLE_RATE,
            1,
            2,
        )
        return reader.readframes(reader.getnframes())


def synthesize_speech(path: Path, text: str) -> str | None:
    """Write `text` as 16 kHz mono speech with System.Speech; a skip reason on failure."""
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{path}', $f); $s.Speak('{text}'); $s.Dispose()"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"speech synthesis failed: {exc}"
    if result.returncode != 0 or not path.exists():
        return f"speech synthesis failed: {result.stderr.strip()[:200]}"
    return None


def tone_pcm(seconds: float = 2.0, hz: float = 440.0) -> bytes:
    n = int(seconds * SAMPLE_RATE)
    samples = (int(8000 * math.sin(2 * math.pi * hz * i / SAMPLE_RATE)) for i in range(n))
    return struct.pack(f"<{n}h", *samples)


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def print_timings(label: str, timings: StageTimings) -> None:
    print(
        f"[{label}] press->pill {_ms(timings.press_to_pill_ms)}, mic open "
        f"{_ms(timings.mic_open_ms)}, release->transcript {_ms(timings.release_to_transcript_ms)}, "
        f"release->cleaned {_ms(timings.release_to_cleaned_ms)}, delivery "
        f"{_ms(timings.delivery_ms)}, release->delivered "
        f"{_ms(timings.extra.get('release_to_delivered_ms'))}, audio "
        f"{timings.extra.get('audio_s', 0.0):.1f} s"
    )


# Fake inject backends: nothing reaches the real clipboard or the real foreground window.


class FakeWindow:
    def __init__(self, foreground: int) -> None:
        self.foreground = foreground

    def foreground_hwnd(self) -> int:
        return self.foreground

    def is_elevated_window(self, hwnd: int) -> bool:
        return False


class FakeInput:
    def __init__(self) -> None:
        self.log: list = []

    def release_held_modifiers(self) -> list:
        self.log.append("release_held_modifiers")
        return []

    def send_ctrl_v(self) -> None:
        self.log.append("send_ctrl_v")

    def type_unicode(self, text: str) -> None:
        self.log.append(("type_unicode", text))


class FakeClipboard:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.sequence = 1

    def snapshot(self) -> object:
        return object()

    def set_text(self, text: str) -> int:
        self.texts.append(text)
        self.sequence += 1
        return self.sequence

    def restore(self, snap: object, expected_sequence: int) -> bool:
        return True


# Fixtures ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_paths(tmp_path_factory) -> EnginePaths:
    missing = _missing()
    if missing:
        pytest.skip("missing: " + ", ".join(missing))
    log_dir = Path(tmp_path_factory.mktemp("engine-logs"))
    print(f"\nengine logs: {log_dir}")
    return EnginePaths(
        vulkan_dir=ENGINES_DIR / "vulkan",
        cpu_dir=ENGINES_DIR / "cpu",
        whisper_model=WHISPER_TINY,
        vad_model=VAD_MODEL,
        llama_model=TINY_LLAMA,
        log_dir=log_dir,
    )


@pytest.fixture(scope="module")
def speech(tmp_path_factory) -> bytes:
    """A real speech clip from the Windows voice, so tiny whisper has something to hear."""
    path = Path(tmp_path_factory.mktemp("speech")) / "speech.wav"
    reason = synthesize_speech(path, SPEECH_TEXT)
    if reason is not None:
        pytest.skip(reason)
    pcm = read_pcm(path)
    print(f"speech clip: {len(pcm) / (2 * SAMPLE_RATE):.1f} s, {SPEECH_TEXT!r}")
    return pcm


@pytest.fixture(scope="module")
def gpu(live_paths):
    started = time.monotonic()
    selection = select_device(
        live_paths.llama_exe("vulkan"), whisper_server=live_paths.whisper_exe("vulkan")
    )
    print(f"GPU: {selection.name!r} (raw index {selection.raw_index}, probe {time.monotonic() - started:.1f} s)")
    return selection


class LiveHarness:
    """Real supervisor plus a started pipeline wired to fakes on every other side."""

    def __init__(
        self,
        paths: EnginePaths,
        gpu,
        tmp_path: Path,
        *,
        idle_unload_minutes: float = 0.0,
        whisper_client_factory=None,
    ) -> None:
        self.cond = threading.Condition()
        self.status_events: list[tuple[float, Engine, EngineState, str]] = []
        self.events: list[PipelineEvent] = []
        self.started_at = time.perf_counter()
        self.engines = EngineSupervisor(
            paths, gpu, self._on_status, idle_unload_minutes=idle_unload_minutes
        )
        self.config = ConfigStore(tmp_path / "settings.json")
        self.history = HistoryStore(tmp_path / "history.db")
        self.ended: list[int] = []
        self.recorders: list[FakeRecorder] = []
        self.pcm = b""
        self.input = FakeInput()
        self.clipboard = FakeClipboard()
        self.backends = InjectBackends(
            window=FakeWindow(TARGET_HWND), input=self.input, clipboard=self.clipboard
        )
        self.next_id = 0
        self.engines.start()
        self.pipeline = Pipeline(
            config=self.config,
            engines=self.engines,
            hotkey=self.ended.append,
            history=self.history,
            on_event=self._on_event,
            recorder_factory=self._recorder,
            capture=self._capture,
            inject_backends=self.backends,
            whisper_client_factory=whisper_client_factory,
        )
        self.pipeline.start()
        self.callbacks = self.pipeline.hotkey_callbacks()

    # Collaborators

    def _on_status(self, engine: Engine, state: EngineState, reason: str) -> None:
        with self.cond:
            self.status_events.append(
                (time.perf_counter() - self.started_at, engine, state, reason)
            )
            self.cond.notify_all()

    def _on_event(self, event: PipelineEvent) -> None:
        with self.cond:
            self.events.append(event)
            self.cond.notify_all()

    def _recorder(self, **kwargs) -> FakeRecorder:
        recorder = FakeRecorder(**kwargs)
        recorder.pcm = self.pcm
        self.recorders.append(recorder)
        return recorder

    @staticmethod
    def _capture() -> TargetContext:
        return TargetContext(TARGET_HWND, "notepad.exe", "Untitled - Notepad", time.perf_counter())

    # Driving

    def wait_engines(self, timeout_s: float = READY_TIMEOUT_S) -> None:
        for engine in Engine:
            assert self.engines.wait_ready(engine, timeout_s), self.describe()

    def wait_for(self, predicate, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self.cond:
            while not predicate():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.cond.wait(remaining)
            return True

    def wait_status(self, engine: Engine, state: EngineState, timeout_s: float, since: int = 0):
        return self.wait_for(
            lambda: any(e is engine and s is state for _, e, s, _ in self.status_events[since:]),
            timeout_s,
        )

    def set_pcm(self, pcm: bytes) -> None:
        self.pcm = pcm
        for recorder in self.recorders:
            recorder.pcm = pcm

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def dictate(self, pcm: bytes, hold_s: float = 0.3, timeout_s: float = DICTATION_TIMEOUT_S):
        self.set_pcm(pcm)
        dictation_id = self.new_id()
        self.callbacks.pressed(MAIN, dictation_id)
        time.sleep(hold_s)
        self.callbacks.released(MAIN, dictation_id)
        return self.final(dictation_id, timeout_s)

    def final(self, dictation_id: int, timeout_s: float) -> PipelineEvent:
        """The event that completes the dictation: timings, a notice, or idle again."""

        def is_final(event: PipelineEvent) -> bool:
            return event.dictation_id == dictation_id and (
                event.timings is not None or event.notice is not None
            )

        assert self.wait_for(lambda: any(is_final(e) for e in self.events), timeout_s), (
            f"dictation {dictation_id} did not complete\n{self.describe()}"
        )
        return next(e for e in self.events if is_final(e))

    def describe(self) -> str:
        with self.cond:
            lines = [f"{t:7.1f} s  {e.value:8} {s.value:13} {r}" for t, e, s, r in self.status_events]
            lines += [
                f"event: id={e.dictation_id} pill={e.pill.value} tray={e.tray.value} "
                f"notice={e.notice} {e.notice_text!r} timings={'yes' if e.timings else 'no'}"
                for e in self.events[-12:]
            ]
        return "\n".join(lines)

    def close(self) -> None:
        self.pipeline.stop()
        self.engines.stop()
        self.history.close()


# Tests --------------------------------------------------------------------------------


def test_full_dictation_end_to_end(live_paths, gpu, speech, tmp_path):
    h = LiveHarness(live_paths, gpu, tmp_path)
    try:
        h.wait_engines()
        started = time.perf_counter()
        event = h.dictate(speech)
        elapsed_s = time.perf_counter() - started
        assert elapsed_s < DICTATION_TIMEOUT_S
        assert event.notice is None, h.describe()
        assert event.timings is not None, h.describe()
        assert h.ended == [1]
        entries = h.history.recent()
        assert entries, "the speech clip produced no transcript"
        entry = entries[0]
        print(
            f"transcript {entry.raw_text!r} ({entry.language}), cleanup {entry.cleanup_reason}, "
            f"delivered {entry.delivered_text!r}, outcome {entry.outcome}"
        )
        print_timings("speech clip", event.timings)
        lowered = entry.raw_text.lower()
        assert "test" in lowered or "dictation" in lowered, entry.raw_text
        assert entry.outcome == "pasted"
        assert "send_ctrl_v" in h.input.log
        assert h.clipboard.texts == [entry.delivered_text]
        assert h.pipeline.recent_timings() == [event.timings]
        tone = h.dictate(tone_pcm())
        assert tone.notice is None, h.describe()
        assert tone.timings is not None
        delivered = "nothing_to_deliver" not in tone.timings.extra
        print("440 Hz tone:", "delivered text" if delivered else "nothing to deliver")
        print_timings("tone", tone.timings)
        assert h.recorders[-1].calls.count("stop") == 2
    finally:
        h.close()


class BurningWhisperClient(WhisperClient):
    """Holds the GIL in a pure Python loop before every request (spec 20.2 stress test)."""

    def __init__(self, base_url: str, burn_s: float) -> None:
        super().__init__(base_url)
        self.burn_s = burn_s

    def inference(self, *args, **kwargs):
        deadline = time.perf_counter() + self.burn_s
        value = 1
        while time.perf_counter() < deadline:
            value = (value * 1103515245 + 12345) & 0x7FFFFFFF
        return super().inference(*args, **kwargs)


def test_hook_callback_enqueue_latency_under_a_cpu_bound_worker(
    live_paths, gpu, speech, tmp_path
):
    burn_s = 0.5
    cycles = 8
    h = LiveHarness(
        live_paths, gpu, tmp_path, whisper_client_factory=lambda url: BurningWhisperClient(url, burn_s)
    )
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(0.001)  # what HotkeyThread.run sets in production (spec 5.1)
    try:
        h.wait_engines()
        h.set_pcm(speech)
        durations: list[float] = []
        ids: list[int] = []
        for _ in range(cycles):
            dictation_id = h.new_id()
            ids.append(dictation_id)
            started = time.perf_counter()
            h.callbacks.pressed(MAIN, dictation_id)
            durations.append(time.perf_counter() - started)
            time.sleep(0.3)
            started = time.perf_counter()
            h.callbacks.released(MAIN, dictation_id)
            durations.append(time.perf_counter() - started)
            time.sleep(0.1)
        worst_ms = max(durations) * 1000.0
        print(
            f"hook callback enqueue while the worker burns CPU: max {worst_ms:.3f} ms, "
            f"median {statistics.median(durations) * 1000.0:.3f} ms over {len(durations)} calls"
        )
        assert worst_ms < 300.0
        for dictation_id in ids:
            event = h.final(dictation_id, 60.0)
            assert event.notice is None, h.describe()
        pills = [e.timings.press_to_pill_ms for e in h.events if e.timings is not None]
        assert len(pills) == cycles
        print(
            f"press->pill under load: median {statistics.median(pills):.1f} ms, "
            f"max {max(pills):.1f} ms over {len(pills)} dictations"
        )
    finally:
        sys.setswitchinterval(previous_interval)
        h.close()


class KillingWhisperFactory:
    """Kills whisper-server 20 ms into the first request, then counts the retry."""

    def __init__(self) -> None:
        self.harness: LiveHarness | None = None
        self.calls = 0
        self.killed_pid: int | None = None

    def __call__(self, base_url: str) -> WhisperClient:
        return KillingWhisperClient(base_url, self)

    def kill(self, pid: int) -> None:
        time.sleep(0.02)
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            check=True,
            capture_output=True,
            creationflags=CREATE_NO_WINDOW,
        )
        self.killed_pid = pid


class KillingWhisperClient(WhisperClient):
    def __init__(self, base_url: str, owner: KillingWhisperFactory) -> None:
        super().__init__(base_url)
        self.owner = owner

    def inference(self, *args, **kwargs):
        self.owner.calls += 1
        if self.owner.calls == 1:
            assert self.owner.harness is not None
            pid = self.owner.harness.engines.pid(WHISPER)
            assert pid
            threading.Thread(target=self.owner.kill, args=(pid,), daemon=True).start()
        return super().inference(*args, **kwargs)


def test_whisper_killed_mid_request_is_retried_exactly_once(live_paths, gpu, speech, tmp_path):
    factory = KillingWhisperFactory()
    h = LiveHarness(live_paths, gpu, tmp_path, whisper_client_factory=factory)
    factory.harness = h
    try:
        h.wait_engines()
        # Locked language: exactly one request per attempt, whatever the clip contains.
        h.config.update(lambda s: replace(s, general=replace(s.general, language_mode="en")))
        pid_before = h.engines.pid(WHISPER)
        since = len(h.status_events)
        started = time.perf_counter()
        event = h.dictate(speech * 2, timeout_s=90.0)
        assert factory.killed_pid == pid_before
        assert factory.calls == 2, f"expected one retry, saw {factory.calls} requests"
        assert event.notice is None, h.describe()
        states = [s for _, e, s, _ in h.status_events[since:] if e is WHISPER]
        print(f"whisper states after the kill: {[s.value for s in states]}")
        assert states, h.describe()
        assert states[-1] is EngineState.READY
        assert any(s in (EngineState.STARTING, EngineState.RESTARTING) for s in states)
        assert h.engines.pid(WHISPER) not in (None, pid_before)
        assert not _alive(pid_before)
        assert any(e.pill is PillState.STARTING_ENGINES for e in h.events)
        print(f"dictation delivered {time.perf_counter() - started:.1f} s after the press")
        assert event.timings is not None
        print_timings("after the kill", event.timings)
        assert h.pipeline.retry_available is False
    finally:
        h.close()


def test_idle_unload_then_a_press_reloads_the_engines(live_paths, gpu, speech, tmp_path):
    h = LiveHarness(live_paths, gpu, tmp_path, idle_unload_minutes=0.05)
    try:
        h.wait_engines()
        assert h.wait_status(WHISPER, EngineState.UNLOADED, 30.0), h.describe()
        assert h.wait_status(LLAMA, EngineState.UNLOADED, 30.0), h.describe()
        assert h.engines.whisper_url is None
        since = len(h.status_events)
        started = time.perf_counter()
        event = h.dictate(speech, timeout_s=90.0)
        assert event.notice is None, h.describe()
        assert event.timings is not None
        assert any(e.pill is PillState.STARTING_ENGINES for e in h.events), h.describe()
        assert any(s is EngineState.READY for _, e, s, _ in h.status_events[since:] if e is WHISPER)
        print(f"reload plus dictation took {time.perf_counter() - started:.1f} s after idle unload")
        print_timings("after reload", event.timings)
    finally:
        h.close()


def test_stop_leaves_no_engine_processes_and_no_pipeline_threads(live_paths, gpu, tmp_path):
    h = LiveHarness(live_paths, gpu, tmp_path)
    h.wait_engines()
    pids = [h.engines.pid(engine) for engine in Engine]
    assert all(pids), pids
    h.pipeline.stop()
    names = {t.name for t in threading.enumerate()}
    assert not any(name.startswith("spells-") for name in names), names
    assert h.recorders and h.recorders[-1].closed
    h.engines.stop()
    assert _wait(lambda: not any(_alive(pid) for pid in pids), 10.0), pids
    assert h.engines.whisper_url is None and h.engines.llama_url is None
    h.history.close()
