"""The instruction path through the pipeline: compose and edit mode (spec 8.5).

Everything is faked, exactly as in test_pipeline: the hotkey, the recorder, the engine
supervisor, the clients, the Win32 backends, the clock. The harness of test_pipeline is
reused, with a writing client factory on top of it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from spells import compose
from spells.compose import NOTHING_SELECTED_TEXT, ComposeError
from spells.compose_prompt import SYSTEM_PROMPT
from spells.history import MODE_COMPOSE, MODE_DICTATE, MODE_EDIT, OUTCOME_NOT_WRITTEN
from spells.models import Chord, ChordMode, Engine, EngineId, EngineState
from spells.pipeline import Notice, PillState

from .test_pipeline import LONG_TEXT, FakeEngines, FakeLlamaClient, FakeWhisperClient, Harness

COMPOSE_CHORD = Chord(keys=(0x11, 0x12, 0x57), mode=ChordMode.COMPOSE)
EDIT_CHORD = Chord(keys=(0x11, 0x12, 0x45), mode=ChordMode.EDIT)
EMAIL = "Hi Marta,\n\nCould you send me the September invoice?\n\nThanks,\nAlex"
SHORTER = "The release plan is long and nobody owns the installer work."
PARAGRAPH = "The release plan is long and the installer work is not owned by anyone yet."


class FakeWriter:
    """A ComposeClient stand-in that records what the pipeline asked for."""

    def __init__(self, content: str = EMAIL, finish: str = "stop", error=None):
        self.content = content
        self.finish = finish
        self.error = error
        self.calls: list[tuple[str, str, int]] = []
        self.cancel = None
        self.on_call = None

    def chat(self, system: str, user: str, max_tokens: int):
        self.calls.append((system, user, max_tokens))
        if self.on_call is not None:
            self.on_call()
        if self.cancel is not None and self.cancel.cancelled:
            raise ComposeError("cancelled", reason="cancelled")
        if self.error is not None:
            raise self.error
        return self.content, self.finish


class WriterFactory:
    def __init__(self, writer: FakeWriter):
        self.writer = writer
        self.urls: list[str] = []
        self.timeouts: list[float] = []

    def __call__(self, url: str, timeout_s: float, cancel):
        self.urls.append(url)
        self.timeouts.append(timeout_s)
        self.writer.cancel = cancel
        return self.writer


class ComposeHarness(Harness):
    """test_pipeline's harness with a writing client and the writing chords recorded."""

    def __init__(self, tmp_path, *, writer: FakeWriter | None = None, selection=None, **kwargs):
        self.writer = writer if writer is not None else FakeWriter()
        self.writer_factory = WriterFactory(self.writer)
        super().__init__(tmp_path, **kwargs)
        self.pipeline._compose_factory = self.writer_factory
        if selection is not None:
            self.backends.clipboard.selection = selection

    def write(self, dictation_id: int = 1, chord: Chord = COMPOSE_CHORD) -> None:
        self.dictate(dictation_id, chord)

    @property
    def entry(self):
        return self.history.entries[-1]


class WritingHotkey:
    """The FakeHotkey of test_pipeline plus the Esc arming of spec 8.5."""

    def __init__(self) -> None:
        self.ended: list[int] = []
        self.armed: list[int | None] = []

    def end_recording(self, dictation_id: int) -> None:
        self.ended.append(dictation_id)

    def set_writing(self, dictation_id: int | None) -> None:
        self.armed.append(dictation_id)


# Composing (spec 8.5) --------------------------------------------------------------------------


def test_a_compose_dictation_delivers_what_the_model_wrote(tmp_path):
    h = ComposeHarness(tmp_path)
    h.write()
    assert h.delivered_texts() == [EMAIL]
    assert h.inject_log[-1][0] != "type_unicode"


def test_a_compose_dictation_never_reaches_the_cleanup_gate(tmp_path):
    h = ComposeHarness(tmp_path)
    h.write()
    assert h.llama.calls == []
    assert h.writer.calls


def test_the_instruction_and_the_fixed_prompt_go_to_the_writing_model(tmp_path):
    h = ComposeHarness(tmp_path)
    h.write()
    system, user, max_tokens = h.writer.calls[0]
    assert system == SYSTEM_PROMPT
    assert LONG_TEXT in user
    assert "[MODE]\nwrite" in user
    assert max_tokens == compose.max_tokens_for(LONG_TEXT)


def test_the_writing_request_uses_the_text_engine_and_its_own_timeout(tmp_path):
    h = ComposeHarness(tmp_path)
    h.write()
    assert h.writer_factory.urls == [h.engines.llama_url]
    assert h.writer_factory.timeouts == [compose.COMPOSE_TIMEOUT_MS / 1000.0]


def test_a_dictation_chord_still_goes_through_cleanup(tmp_path):
    h = ComposeHarness(tmp_path)
    h.dictate()
    assert h.llama.calls
    assert h.writer.calls == []


def test_nothing_is_delivered_before_the_answer_is_ready(tmp_path):
    seen: list[list] = []
    h = ComposeHarness(tmp_path)
    h.writer.on_call = lambda: seen.append(list(h.inject_log))
    h.write()
    assert seen[0] == [], "the cursor saw nothing while the model was still writing"
    assert h.delivered_texts() == [EMAIL]


# Editing (spec 8.5) ----------------------------------------------------------------------------


def test_an_edit_copies_the_selection_and_sends_it_with_the_instruction(tmp_path):
    h = ComposeHarness(tmp_path, selection=PARAGRAPH, writer=FakeWriter(content=SHORTER))
    h.write(chord=EDIT_CHORD)
    _system, user, _max_tokens = h.writer.calls[0]
    assert "[MODE]\nedit" in user
    assert PARAGRAPH in user
    assert h.delivered_texts()[-1] == SHORTER


def test_an_edit_puts_the_users_clipboard_back(tmp_path):
    h = ComposeHarness(tmp_path, selection=PARAGRAPH, writer=FakeWriter(content=SHORTER))
    h.write(chord=EDIT_CHORD)
    kinds = [entry[0] for entry in h.inject_log]
    assert "send_ctrl_c" in kinds
    assert kinds.count("restore") >= 1


def test_nothing_selected_falls_back_to_composing_and_says_so(tmp_path):
    h = ComposeHarness(tmp_path, selection=None)
    h.write(chord=EDIT_CHORD)
    _system, user, _max_tokens = h.writer.calls[0]
    assert "[MODE]\nwrite" in user
    assert "[SELECTED TEXT]" not in user
    assert h.delivered_texts() == [EMAIL]
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == NOTHING_SELECTED_TEXT


def test_nothing_selected_is_recorded_as_a_compose_row(tmp_path):
    h = ComposeHarness(tmp_path, selection=None)
    h.write(chord=EDIT_CHORD)
    assert h.entry.mode == MODE_COMPOSE
    assert h.entry.selection_chars == 0


def test_an_edit_whose_focus_moved_never_copies_from_the_other_window(tmp_path):
    h = ComposeHarness(tmp_path, selection=PARAGRAPH, foreground=0x00CC00DD)
    h.write(chord=EDIT_CHORD)
    assert not [entry for entry in h.inject_log if entry[0] == "send_ctrl_c"]
    assert h.backends.clipboard.copies == 0


def test_an_edit_delivers_over_the_selection_by_the_usual_path(tmp_path):
    h = ComposeHarness(tmp_path, selection=PARAGRAPH, writer=FakeWriter(content=SHORTER))
    h.write(chord=EDIT_CHORD)
    kinds = [entry[0] for entry in h.inject_log]
    assert kinds[-4:] == ["set_text", "release_held_modifiers", "send_ctrl_v", "restore"]


def test_an_edit_never_gains_the_leading_space_of_the_previous_delivery(tmp_path):
    h = ComposeHarness(tmp_path, selection=PARAGRAPH, writer=FakeWriter(content=SHORTER))
    h.dictate(1)
    h.write(2, chord=EDIT_CHORD)
    assert h.delivered_texts()[-1] == SHORTER


def test_a_written_text_never_gains_the_leading_space_of_the_previous_delivery(tmp_path):
    h = ComposeHarness(tmp_path, writer=FakeWriter(content=SHORTER))
    h.dictate(1)
    h.write(2)
    assert h.delivered_texts()[-1] == SHORTER


# The guards (spec 8.5) -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "finish", "reason"),
    [
        ("", "stop", "empty"),
        ("cut off here", "length", "finish_length"),
        ("Here is the email you asked for: hello", "stop", "preamble"),
        ("I cannot write that for you.", "stop", "refusal"),
        (LONG_TEXT, "stop", "echo"),
    ],
)
def test_a_rejected_answer_delivers_nothing_and_the_pill_says_why(
    tmp_path, answer, finish, reason
):
    h = ComposeHarness(tmp_path, writer=FakeWriter(content=answer, finish=finish))
    h.write()
    assert h.delivered_texts() == []
    assert h.last.notice is Notice.ERROR
    assert h.last.notice_text == compose.rejection_text(reason)
    assert h.entry.outcome == OUTCOME_NOT_WRITTEN
    assert h.entry.cleanup_reason == reason


def test_a_rejection_never_delivers_the_raw_instruction(tmp_path):
    h = ComposeHarness(tmp_path, writer=FakeWriter(content="As an AI I cannot do that."))
    h.write()
    assert LONG_TEXT not in "".join(h.delivered_texts())
    assert h.entry.delivered_text == ""


def test_a_rejection_is_counted_for_diagnostics(tmp_path):
    h = ComposeHarness(tmp_path, writer=FakeWriter(content="I cannot write that."))
    h.write()
    assert h.pipeline.guard_counts()["refusal"] == 1


def test_a_writing_engine_that_is_not_serving_delivers_nothing(tmp_path):
    h = ComposeHarness(tmp_path)
    h.engines.states[Engine.LLAMA] = EngineState.UNLOADED
    h.write()
    assert h.delivered_texts() == []
    assert h.entry.cleanup_reason == "engine_not_ready"
    assert h.writer.calls == []


def test_a_writing_request_that_times_out_delivers_nothing(tmp_path):
    h = ComposeHarness(tmp_path, writer=FakeWriter(error=ComposeError("slow", reason="timeout")))
    h.write()
    assert h.delivered_texts() == []
    assert h.last.notice_text == compose.rejection_text("timeout")


# Esc (spec 8.5) --------------------------------------------------------------------------------


def press_escape(h, dictation_id=1):
    """What the hook does when Esc arrives while the writing model is working."""

    def escape() -> None:
        h.callbacks.cancelled(dictation_id)
        h.pipeline.drain_controller()

    return escape


def test_escape_cancels_the_request_and_delivers_nothing(tmp_path):
    h = ComposeHarness(tmp_path)
    h.writer.on_call = press_escape(h)
    h.write()
    assert h.delivered_texts() == []
    assert h.entry.cleanup_reason == "cancelled"


def test_a_cancelled_composition_says_nothing_on_the_pill(tmp_path):
    h = ComposeHarness(tmp_path)
    h.writer.on_call = press_escape(h)
    h.write()
    assert h.last.notice is None


def test_the_pipeline_arms_and_disarms_the_escape_key(tmp_path):
    hotkey = WritingHotkey()
    h = ComposeHarness(tmp_path)
    h.pipeline._set_writing_fn = hotkey.set_writing
    h.write()
    assert hotkey.armed == [1, None]


def test_an_escape_for_another_dictation_cancels_nothing(tmp_path):
    h = ComposeHarness(tmp_path)
    h.writer.on_call = press_escape(h, 99)
    h.write()
    assert h.delivered_texts() == [EMAIL]


# The pill (spec 14.2) --------------------------------------------------------------------------


def test_the_pill_shows_the_writing_state_with_the_instruction(tmp_path):
    states: list = []
    h = ComposeHarness(tmp_path)
    h.writer.on_call = lambda: states.append((h.last.pill, h.last.text))
    h.write()
    assert states == [(PillState.WRITING, LONG_TEXT)]


def test_the_pill_is_back_to_idle_once_the_text_is_delivered(tmp_path):
    h = ComposeHarness(tmp_path)
    h.write()
    assert h.last.pill is PillState.IDLE


# History (spec 8.5) ----------------------------------------------------------------------------


def test_a_compose_row_records_the_mode_the_instruction_and_the_text(tmp_path):
    h = ComposeHarness(tmp_path)
    h.write()
    entry = h.entry
    assert entry.mode == MODE_COMPOSE
    assert entry.instruction == LONG_TEXT
    assert entry.raw_text == LONG_TEXT
    assert entry.delivered_text == EMAIL
    assert entry.cleaned_text == ""
    assert entry.cleanup_reason == "compose"
    assert entry.selection_chars == 0


def test_an_edit_row_records_the_length_of_the_selection(tmp_path):
    h = ComposeHarness(tmp_path, selection=PARAGRAPH, writer=FakeWriter(content=SHORTER))
    h.write(chord=EDIT_CHORD)
    assert h.entry.mode == MODE_EDIT
    assert h.entry.selection_chars == len(PARAGRAPH)


def test_a_dictation_row_is_still_a_dictation_row(tmp_path):
    h = ComposeHarness(tmp_path)
    h.dictate()
    assert h.entry.mode == MODE_DICTATE
    assert h.entry.instruction == ""


def test_the_speech_signals_still_describe_the_instruction(tmp_path):
    h = ComposeHarness(tmp_path)
    h.write()
    assert h.entry.signals.word_count == len(LONG_TEXT.split())
    assert h.entry.quality_label in ("good", "uncertain", "poor")


# Live typing never runs for an instruction (spec 6, B5-57) -------------------------------------


def test_a_compose_dictation_never_types_the_instruction_as_it_goes(tmp_path):
    h = ComposeHarness(tmp_path, settings=lambda s: replace(s, general=replace(s.general, live_text=True)))
    h.press(1, COMPOSE_CHORD)
    assert h.pipeline.live is None
    assert h.last.live_typing is False
    h.release(1, COMPOSE_CHORD)
    h.process()
    assert h.delivered_texts() == [EMAIL]


def test_a_llama_client_is_never_built_for_a_compose_dictation(tmp_path):
    h = ComposeHarness(tmp_path, llama=FakeLlamaClient(content="should not be used"))
    h.write()
    assert h.llama.calls == []


def test_a_retried_composition_goes_to_the_clipboard_like_any_retry(tmp_path):
    from spells.asr import WhisperError

    h = ComposeHarness(tmp_path, whisper=FakeWhisperClient(responses=[WhisperError("down")]))
    h.engines.states[Engine.WHISPER] = EngineState.READY
    h.whisper.healthy = False
    h.write()
    assert h.pipeline.retry_available is True
    h.whisper.responses = [{"text": LONG_TEXT, "language": "english"}]
    h.whisper.healthy = True
    assert h.pipeline.retry_last() is True
    h.process()
    assert h.delivered_texts() == [EMAIL]
    assert [entry[0] for entry in h.inject_log if entry[0] == "send_ctrl_v"] == []


# The writing engine ----------------------------------------------------------------------


WRITER_ENGINE = EngineId(Engine.LLAMA, 1)
WRITER_URL = "http://127.0.0.1:9301"


class WriterEngines(FakeEngines):
    def __init__(self, ready: bool = True) -> None:
        super().__init__()
        self.writer_engine = WRITER_ENGINE
        self.states[WRITER_ENGINE] = EngineState.READY if ready else EngineState.STARTING
        self.urls[WRITER_ENGINE] = WRITER_URL

    def url(self, engine) -> str | None:
        if engine == WRITER_ENGINE:
            return self._url(WRITER_ENGINE)
        return super().url(engine)


def test_writing_goes_to_the_writing_engine_when_there_is_one(tmp_path):
    h = ComposeHarness(tmp_path, engines=WriterEngines())
    h.write()
    assert h.writer_factory.urls == [WRITER_URL]
    assert h.delivered_texts()


def test_writing_waits_for_a_writing_engine_that_is_still_loading(tmp_path):
    engines = WriterEngines(ready=False)
    h = ComposeHarness(tmp_path, engines=engines)
    h.write()
    assert ("wait_ready", WRITER_ENGINE, 1.0) in engines.calls
    assert h.writer_factory.urls == [WRITER_URL]


def test_without_a_writing_engine_writing_uses_the_cleanup_engine(tmp_path):
    h = ComposeHarness(tmp_path)
    h.write()
    assert h.writer_factory.urls == ["http://127.0.0.1:9002"]
