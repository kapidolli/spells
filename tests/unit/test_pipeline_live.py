"""Unit tests for the live partial passes and live insertion of spells.pipeline.

Spec 6 (partial passes and the dictation flow), 11 (delivery), 13 (engine states) and
decision B5-57. The harness, the fakes and the drain helpers come from test_pipeline; the
partial loop runs on the calling thread here (`partial_threads=False`), so every tick is
explicit and nothing is timing dependent.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from spells import pipeline as pipeline_module
from spells import platform
from spells.asr import LLAMA_ASR, WHISPER_SERVER, AsrReply, WhisperError
from spells.models import DeliveryOutcome, EngineState, LangMode
from spells.pipeline import (
    COPIED_NOTICE,
    PARTIAL_MAX_LATENCY_CANCELLABLE_MS,
    PARTIAL_MAX_LATENCY_MS,
    PARTIAL_WINDOW_S,
    Notice,
)
from spells.platform import stub
from spells.platform.base import Capabilities
from spells.profiles import BUILTIN_PROFILES

from .fake_clients import FakeLlamaAsrClient, FakeWhisperClient
from .test_inject import OTHER_HWND
from .test_pipeline import (
    CLEANED,
    OK_RESPONSE,
    WHISPER,
    Harness,
    general,
    routed_harness,
)

PARTIAL_TEXT = "so I think we should meet"
PARTIAL_RESPONSE = {"text": PARTIAL_TEXT, "language": "english"}


@pytest.fixture(autouse=True)
def live_capable_platform():
    """Live typing needs a platform that can do it; the stub on Linux and macOS cannot."""
    record = platform.current()
    if record.capabilities.live_typing:
        yield
        return
    previous = platform.swap(replace(stub.build("linux"), capabilities=Capabilities.everything()))
    try:
        yield
    finally:
        platform.swap(previous)


class FakeChoice:
    def __init__(
        self, latency_ms: int | None, runtime: str = WHISPER_SERVER, languages=()
    ) -> None:
        self.latency_ms = latency_ms
        self.runtime = runtime
        self.languages = tuple(languages)


class FakeSelection:
    def __init__(
        self,
        latency_ms: int | None = 300,
        per_language=None,
        runtime: str = WHISPER_SERVER,
        languages=("en", "de", "sq"),
    ) -> None:
        self.latency_ms = latency_ms
        self.per_language = dict(per_language or {})
        self.runtime = runtime
        self.languages = tuple(languages)

    @property
    def asr(self):
        shared = tuple(code for code in self.languages if code not in self.per_language)
        choices = []
        if self.latency_ms is not None and shared:
            choices.append(FakeChoice(self.latency_ms, self.runtime, shared))
        for code, value in self.per_language.items():
            if value is not None:
                choices.append(FakeChoice(value, WHISPER_SERVER, (code,)))
        return tuple(choices)

    def asr_for(self, language: str):
        for choice in self.asr:
            if language in choice.languages:
                return choice
        return None


def live_harness(tmp_path, **kwargs):
    kwargs.setdefault("selection", FakeSelection())
    kwargs.setdefault("whisper", FakeWhisperClient(responses=[PARTIAL_RESPONSE, OK_RESPONSE]))
    return Harness(tmp_path, **kwargs)


def typed(h):
    return [text for text, _ in h.backends.keyboard.typed]


def erased(h):
    return [count for count, _ in h.backends.keyboard.backspaced]



def test_without_a_selection_no_partial_runs(tmp_path):
    h = Harness(tmp_path)
    h.press()
    assert h.pipeline.live is None
    assert h.pipeline.run_partial_tick() == "off"
    assert typed(h) == []


def test_a_tick_transcribes_the_audio_so_far_and_types_it(tmp_path):
    h = live_harness(tmp_path)
    h.press()
    assert h.pipeline.run_partial_tick() == "ran"
    assert typed(h) == [PARTIAL_TEXT]
    assert h.pipeline.live.session.inserted == PARTIAL_TEXT
    assert h.whisper.calls[0]["wav"][:4] == b"RIFF"


def test_the_first_partial_waits_for_enough_audio(tmp_path):
    def short(recorder):
        recorder.pcm = b"\x01\x00" * int(1.0 * recorder.sample_rate)

    h = live_harness(tmp_path, recorder_setup=short)
    h.press()
    assert h.pipeline.run_partial_tick() == "waiting"
    assert h.whisper.calls == []
    assert typed(h) == []


def test_a_tick_while_a_pass_is_in_flight_is_skipped(tmp_path):
    h = live_harness(tmp_path)
    h.press()
    seen: list[str] = []
    h.whisper.on_call = lambda: seen.append(h.pipeline.run_partial_tick())
    assert h.pipeline.run_partial_tick() == "ran"
    assert seen == ["skipped"]
    assert len(h.whisper.calls) == 1
    assert h.pipeline.live.skipped == 1


def test_a_growing_partial_types_only_the_tail(tmp_path):
    h = live_harness(
        tmp_path,
        whisper=FakeWhisperClient(
            responses=[
                {"text": "so I think", "language": "english"},
                {"text": "so I think we should meet", "language": "english"},
                OK_RESPONSE,
            ]
        ),
    )
    h.press()
    h.pipeline.run_partial_tick()
    h.pipeline.run_partial_tick()
    assert typed(h) == ["so I think", " we should meet"]
    assert erased(h) == []


def test_a_rewritten_word_costs_only_its_own_characters(tmp_path):
    h = live_harness(
        tmp_path,
        whisper=FakeWhisperClient(
            responses=[
                {"text": "meet at ten", "language": "english"},
                {"text": "meet at two", "language": "english"},
                OK_RESPONSE,
            ]
        ),
    )
    h.press()
    h.pipeline.run_partial_tick()
    h.pipeline.run_partial_tick()
    assert erased(h) == [2]
    assert h.pipeline.live.session.inserted == "meet at two"


def test_an_unchanged_partial_sends_nothing(tmp_path):
    h = live_harness(tmp_path, whisper=FakeWhisperClient(responses=[PARTIAL_RESPONSE]))
    h.press()
    h.pipeline.run_partial_tick()
    h.backends.keyboard.typed.clear()
    assert h.pipeline.run_partial_tick() == "ran"
    assert typed(h) == []
    assert h.pipeline.live.inserts == 1


def test_a_partial_error_never_touches_the_dictation(tmp_path):
    h = live_harness(
        tmp_path,
        whisper=FakeWhisperClient(responses=[WhisperError("boom"), OK_RESPONSE]),
    )
    h.press()
    assert h.pipeline.run_partial_tick() == "error"
    assert not [call for call in h.engines.calls if call[0] == "restart"]
    h.release()
    h.process()
    assert h.delivered_texts() == [CLEANED]
    assert h.last.notice is None


def test_partials_never_run_while_the_engine_is_not_serving(tmp_path):
    h = live_harness(tmp_path)
    h.press()
    h.engines.states[WHISPER] = EngineState.RESTARTING
    h.engines.calls.clear()
    assert h.pipeline.run_partial_tick() == "off"
    assert h.whisper.calls == []
    assert not [
        call for call in h.engines.calls if call[0] in ("wait_ready", "ensure_ready", "restart")
    ]


def test_a_partial_leaves_the_language_state_to_the_worker(tmp_path):
    h = live_harness(
        tmp_path,
        whisper=FakeWhisperClient(
            responses=[{"text": "guten Tag", "language": "german"}, OK_RESPONSE]
        ),
    )
    h.press()
    h.pipeline.run_partial_tick()
    assert h.pipeline._lang_state.last_accepted == "en"



def test_release_aborts_the_pass_in_flight_and_drops_its_text(tmp_path):
    h = live_harness(tmp_path)
    h.press()
    handles: list = []

    def release_during_the_pass():
        handles.append(h.pipeline.live.abort_handle)
        h.release()

    h.whisper.on_call = release_during_the_pass
    live = h.pipeline.live
    assert live.tick() == "aborted"
    assert handles[0] is not None
    assert handles[0].aborted is True
    assert live.aborted is True
    assert typed(h) == []


def test_the_final_pass_reconciles_the_draft_instead_of_pasting(tmp_path):
    h = live_harness(tmp_path)
    h.press()
    h.pipeline.run_partial_tick()
    h.release()
    h.process()
    assert h.pipeline.live is None
    assert typed(h)[-1].endswith("installer work.")
    assert ("send_paste",) not in h.inject_log
    assert not [entry for entry in h.inject_log if entry[0] == "set_text"]
    assert h.history.entries[-1].outcome == DeliveryOutcome.TYPED.value
    assert h.last.notice is None


def test_the_reconciliation_keeps_the_prefix_the_draft_got_right(tmp_path):
    h = live_harness(
        tmp_path,
        whisper=FakeWhisperClient(
            responses=[{"text": "I think we should", "language": "english"}, OK_RESPONSE]
        ),
        llama=None,
    )
    h.press()
    h.pipeline.run_partial_tick()
    h.backends.keyboard.typed.clear()
    h.release()
    h.process()
    assert erased(h) == []
    assert typed(h) == [CLEANED[len("I think we should") :]]


def test_focus_moving_mid_dictation_stops_the_draft_and_copies(tmp_path):
    h = live_harness(tmp_path)
    h.press()
    h.pipeline.run_partial_tick()
    before = list(typed(h))
    h.backends.focus.foreground_window = OTHER_HWND
    h.pipeline.run_partial_tick()
    assert h.pipeline.live.session.stopped is True
    h.release()
    h.process()
    assert typed(h) == before
    assert erased(h) == []
    assert h.last.notice is Notice.COPIED
    assert h.last.notification == COPIED_NOTICE
    assert h.history.entries[-1].outcome == DeliveryOutcome.COPIED_FOCUS_CHANGED.value
    assert h.delivered_texts()[-1] == CLEANED


def test_focus_moving_at_release_copies_rather_than_typing(tmp_path):
    h = live_harness(tmp_path)
    h.press()
    h.pipeline.run_partial_tick()
    before = list(typed(h))
    h.release()
    h.backends.focus.foreground_window = OTHER_HWND
    h.process()
    assert typed(h) == before
    assert h.last.notice is Notice.COPIED
    assert h.delivered_texts()[-1] == CLEANED


def test_escape_takes_the_draft_back(tmp_path):
    h = live_harness(tmp_path)
    h.press()
    h.pipeline.run_partial_tick()
    h.callbacks.cancelled(1)
    h.pipeline.drain_controller()
    assert erased(h) == [len(PARTIAL_TEXT)]
    assert h.pipeline.live is None


def test_a_capture_without_a_window_never_types(tmp_path):
    h = live_harness(tmp_path)
    h.ctx_window = 0
    h.press()
    assert h.pipeline.live is None



@pytest.mark.parametrize(
    ("latency", "expected"),
    [
        (300, True),
        (PARTIAL_MAX_LATENCY_MS, True),
        (PARTIAL_MAX_LATENCY_MS + 1, False),
        (None, False),
    ],
)
def test_the_enable_rule_follows_the_catalog_latency(tmp_path, latency, expected):
    h = Harness(tmp_path, selection=FakeSelection(latency))
    profile = BUILTIN_PROFILES["Default"]
    assert h.pipeline.live_enabled(h.config.settings, profile, LangMode("auto")) is expected


@pytest.mark.parametrize(
    ("latency", "expected"),
    [
        (1300, True),
        (PARTIAL_MAX_LATENCY_CANCELLABLE_MS, True),
        (PARTIAL_MAX_LATENCY_CANCELLABLE_MS + 1, False),
    ],
)
def test_an_engine_that_cancels_may_be_slower(tmp_path, latency, expected):
    h = Harness(tmp_path, selection=FakeSelection(latency, runtime=LLAMA_ASR))
    profile = BUILTIN_PROFILES["Default"]
    assert h.pipeline.live_enabled(h.config.settings, profile, LangMode("auto")) is expected


def test_auto_mode_follows_the_engine_it_asks_first(tmp_path):
    h = Harness(
        tmp_path, selection=FakeSelection(1300, per_language={"sq": 4500}, runtime=LLAMA_ASR)
    )
    settings = h.config.settings
    profile = BUILTIN_PROFILES["Default"]
    assert h.pipeline.live_enabled(settings, profile, LangMode("auto")) is True
    assert h.pipeline.live_enabled(settings, profile, LangMode("forced", "en")) is True
    assert h.pipeline.live_enabled(settings, profile, LangMode("locked", "sq")) is False


def test_a_language_with_no_speech_model_is_never_live(tmp_path):
    h = Harness(tmp_path, selection=FakeSelection(300, per_language={"de": None}))
    profile = BUILTIN_PROFILES["Default"]
    settings = h.config.settings
    assert h.pipeline.live_enabled(settings, profile, LangMode("forced", "de")) is False
    assert h.pipeline.live_enabled(settings, profile, LangMode("auto")) is True


def test_the_setting_turns_live_typing_off(tmp_path):
    h = live_harness(
        tmp_path,
        settings=general(live_text=False),
        whisper=FakeWhisperClient(responses=[OK_RESPONSE]),
    )
    h.press()
    assert h.pipeline.live is None
    assert h.pipeline.run_partial_tick() == "off"
    h.release()
    h.process()
    assert h.delivered_texts() == [CLEANED]


def test_live_typing_stays_out_of_code_and_terminal_windows(tmp_path):
    h = Harness(tmp_path, selection=FakeSelection())
    settings = h.config.settings
    assert h.pipeline.live_enabled(settings, BUILTIN_PROFILES["Code"], LangMode("auto")) is False
    assert (
        h.pipeline.live_enabled(settings, BUILTIN_PROFILES["Terminal"], LangMode("auto")) is False
    )
    everywhere = h.config.update(general(live_text_everywhere=True))
    assert h.pipeline.live_enabled(everywhere, BUILTIN_PROFILES["Code"], LangMode("auto")) is True


def test_a_long_dictation_commits_the_draft_and_starts_the_window_again(tmp_path):
    def long_recording(recorder):
        recorder.pcm = b"\x01\x00" * int((PARTIAL_WINDOW_S + 1) * recorder.sample_rate)

    h = live_harness(tmp_path, recorder_setup=long_recording)
    h.press()
    assert h.pipeline.run_partial_tick() == "ran"
    live = h.pipeline.live
    assert live.anchor_text == PARTIAL_TEXT
    assert live.anchor_bytes == len(h.recorder.pcm)
    assert h.pipeline.run_partial_tick() == "waiting"
    assert len(h.whisper.calls) == 1


def test_the_committed_prefix_stays_in_front_of_later_partials(tmp_path):
    def long_recording(recorder):
        recorder.pcm = b"\x01\x00" * int((PARTIAL_WINDOW_S + 1) * recorder.sample_rate)

    h = live_harness(
        tmp_path,
        recorder_setup=long_recording,
        whisper=FakeWhisperClient(
            responses=[
                {"text": "first part", "language": "english"},
                {"text": "second part", "language": "english"},
                OK_RESPONSE,
            ]
        ),
    )
    h.press()
    h.pipeline.run_partial_tick()
    live = h.pipeline.live
    live.anchor_bytes = 0
    h.pipeline.run_partial_tick()
    assert live.text == "first part second part"
    assert live.session.inserted == "first part second part"


def test_the_retry_path_never_types_live(tmp_path):
    h = live_harness(
        tmp_path,
        whisper=FakeWhisperClient(responses=[WhisperError("down"), OK_RESPONSE]),
    )
    h.engines.states[WHISPER] = EngineState.FAILED
    h.dictate()
    h.engines.states[WHISPER] = EngineState.READY
    assert h.pipeline.retry_available
    h.pipeline.retry_last()
    h.process()
    assert typed(h) == []



def test_the_partial_thread_ticks_and_stops_with_the_dictation(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "PARTIAL_INTERVAL_S", 0.01)
    h = live_harness(
        tmp_path,
        partial_threads=True,
        whisper=FakeWhisperClient(responses=[PARTIAL_RESPONSE]),
    )
    h.press()
    live = h.pipeline.live
    deadline = time.perf_counter() + 5.0
    while live.passes == 0 and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert live.passes >= 1
    assert live.session.inserted == PARTIAL_TEXT
    h.release()
    live._thread.join(timeout=5.0)
    assert live._thread.is_alive() is False
    h.process()
    assert live.session.inserted == h.history.entries[-1].delivered_text



def fast_and_slow():
    return FakeSelection(1300, per_language={"sq": 4500}, runtime=LLAMA_ASR)


def test_a_partial_runs_on_the_engine_that_cancels(tmp_path):
    h = routed_harness(tmp_path, selection=fast_and_slow())
    h.press()
    assert h.pipeline.live is not None
    assert h.pipeline.run_partial_tick() == "ran"
    assert h.pipeline.live.session.inserted
    assert h.whisper.calls == []


def test_a_partial_never_waits_on_a_slow_engine(tmp_path):
    qwen = FakeLlamaAsrClient(AsrReply("", "hu", "Hungarian"), reports="Hungarian")
    h = routed_harness(tmp_path, qwen=qwen, selection=fast_and_slow())
    h.press()
    assert h.pipeline.run_partial_tick() == "off"
    assert h.whisper.calls == []
    assert typed(h) == []


def test_live_typing_stays_off_where_the_platform_cannot_do_it():
    previous = platform.swap(stub.build("linux"))
    try:
        harness_pipeline = pipeline_module.Pipeline.__new__(pipeline_module.Pipeline)
        assert harness_pipeline.live_enabled(None, None, None) is False
    finally:
        platform.swap(previous)
