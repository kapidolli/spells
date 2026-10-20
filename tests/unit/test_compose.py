"""Unit tests for spells.compose and spells.compose_prompt (spec 8.5).

No engine, no network, no clipboard: the writing client is a fake, the Win32 backends are
the fakes of test_inject, and the selection wait is driven by a hand-held clock.
"""

from __future__ import annotations

import json

import pytest

from spells import compose
from spells.compose import (
    COMPOSE_MAX_TOKENS,
    COMPOSE_MIN_TOKENS,
    CancelHandle,
    ComposeError,
    echoes_instruction,
    guard,
    max_tokens_for,
    read_selection,
    starts_with_refusal,
)
from spells.compose import (
    compose as write,
)
from spells.compose_prompt import SYSTEM_PROMPT, user_message
from spells.models import ChordMode

from .test_inject import SNAPSHOT, make

INSTRUCTION = "write an email to Marta asking for the September invoice"
EMAIL = (
    "Hi Marta,\n\nCould you send me the September invoice when you have a moment?\n\n"
    "Thanks,\nAlex"
)
PARAGRAPH = "The release plan is long and the installer work is not owned by anyone yet."


class FakeWriter:
    """A ComposeClient stand-in: returns (content, finish) or raises `error`."""

    def __init__(self, content: str = EMAIL, finish: str = "stop", error: Exception | None = None):
        self.content = content
        self.finish = finish
        self.error = error
        self.calls: list[tuple[str, str, int]] = []

    def chat(self, system: str, user: str, max_tokens: int) -> tuple[str, str]:
        self.calls.append((system, user, max_tokens))
        if self.error is not None:
            raise self.error
        return self.content, self.finish


class Clock:
    def __init__(self, step: float = 0.0):
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.value
        self.value += self.step
        return value



def test_the_user_message_names_write_mode_and_carries_no_selection_block():
    message = user_message(INSTRUCTION, tone="casual")
    assert "[MODE]\nwrite" in message
    assert "[TONE]\ncasual" in message
    assert f"[INSTRUCTION]\n{INSTRUCTION}\n[END INSTRUCTION]" in message
    assert "[SELECTED TEXT]" not in message


def test_the_user_message_names_edit_mode_and_delimits_the_selection():
    message = user_message("make this shorter", PARAGRAPH)
    assert "[MODE]\nedit" in message
    assert f"[SELECTED TEXT]\n{PARAGRAPH}\n[END SELECTED TEXT]" in message


def test_a_short_instruction_still_buys_a_whole_email():
    assert max_tokens_for("write to Marta") == COMPOSE_MIN_TOKENS
    assert max_tokens_for(INSTRUCTION) == COMPOSE_MIN_TOKENS


def test_the_selection_lifts_the_bound_so_an_edit_can_answer_in_full():
    selection = " ".join(["word"] * 300)
    assert max_tokens_for("make this shorter", selection) > COMPOSE_MIN_TOKENS


def test_the_bound_never_passes_its_ceiling():
    huge = " ".join(["word"] * 5000)
    assert max_tokens_for(huge, huge) == COMPOSE_MAX_TOKENS


def test_a_composed_email_far_longer_than_the_instruction_passes():
    assert guard(INSTRUCTION, EMAIL, "stop") is None


def test_a_cut_off_answer_is_rejected_before_anything_else():
    assert guard(INSTRUCTION, "", "length") == "finish_length"


def test_an_empty_answer_is_rejected():
    assert guard(INSTRUCTION, "   \n ", "stop") == "empty"


def test_an_assistant_preamble_is_rejected():
    assert guard(INSTRUCTION, "Here is the email you asked for:\n\n" + EMAIL, "stop") == "preamble"


@pytest.mark.parametrize(
    "answer",
    [
        "I cannot write that email for you.",
        "As an AI, I do not have access to your invoices.",
        "Ich kann nicht auf deine Rechnungen zugreifen.",
    ],
)
def test_a_refusal_is_rejected(answer):
    assert guard(INSTRUCTION, answer, "stop") == "refusal"


def test_the_instruction_handed_straight_back_is_rejected():
    assert guard(INSTRUCTION, INSTRUCTION, "stop") == "echo"


def test_the_instruction_handed_back_with_punctuation_is_still_an_echo():
    assert guard(INSTRUCTION, f'"{INSTRUCTION.capitalize()}."', "stop") == "echo"


def test_a_real_answer_that_quotes_the_instruction_is_not_an_echo():
    answer = f"{INSTRUCTION}. {EMAIL}"
    assert guard(INSTRUCTION, answer, "stop") is None


def test_an_empty_instruction_can_never_be_echoed():
    assert echoes_instruction("", "anything at all") is False


def test_a_refusal_only_counts_at_the_start():
    assert starts_with_refusal("The team said it cannot be done before Friday.") is False


def test_a_composed_email_is_delivered_with_the_reason_ok():
    writer = FakeWriter()
    result = write(INSTRUCTION, writer)
    assert result.ok is True
    assert result.reason == "ok"
    assert result.text == EMAIL
    assert result.instruction == INSTRUCTION
    assert result.mode is ChordMode.COMPOSE
    assert result.selection_chars == 0


def test_nothing_said_never_reaches_the_model():
    writer = FakeWriter()
    result = write("   ", writer)
    assert (result.ok, result.reason, result.text) == (False, "no_instruction", "")
    assert writer.calls == []


def test_no_engine_means_no_request_and_no_text():
    result = write(INSTRUCTION, None)
    assert (result.ok, result.reason, result.text) == (False, "engine_not_ready", "")


def test_a_timeout_delivers_nothing_at_all():
    writer = FakeWriter(error=ComposeError("too slow", reason="timeout"))
    result = write(INSTRUCTION, writer)
    assert (result.ok, result.reason, result.text) == (False, "timeout", "")


def test_a_rejected_answer_never_falls_back_to_the_instruction():
    writer = FakeWriter(content="I cannot do that.")
    result = write(INSTRUCTION, writer)
    assert result.ok is False
    assert result.reason == "refusal"
    assert result.text == ""
    assert INSTRUCTION not in result.text


class FakeResponse:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode("utf-8")
        self.closed = False

    def read(self) -> bytes:
        return self.body

    def close(self) -> None:
        self.closed = True


def test_the_request_asks_for_temperature_zero_and_a_cached_prompt(monkeypatch):
    sent: dict = {}

    def fake_urlopen(request, timeout=None):
        sent["payload"] = json.loads(request.data.decode("utf-8"))
        sent["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": EMAIL}, "finish_reason": "stop"}]})

    monkeypatch.setattr(compose, "urlopen", fake_urlopen)
    client = compose.ComposeClient("http://127.0.0.1:9002", timeout_s=42.0)
    assert client.chat(SYSTEM_PROMPT, "user", 256) == (EMAIL, "stop")
    assert sent["payload"]["temperature"] == 0
    assert sent["payload"]["cache_prompt"] is True
    assert sent["payload"]["stream"] is False
    assert sent["payload"]["max_tokens"] == 256
    assert sent["timeout"] == 42.0


def test_a_cancelled_request_never_leaves_the_process(monkeypatch):
    calls = []
    monkeypatch.setattr(compose, "urlopen", lambda *a, **k: calls.append(a))
    cancel = CancelHandle()
    cancel.cancel()
    client = compose.ComposeClient("http://127.0.0.1:9002", cancel=cancel)
    with pytest.raises(ComposeError) as caught:
        client.chat("system", "user", 256)
    assert caught.value.reason == "cancelled"
    assert calls == []


def test_cancelling_mid_answer_closes_the_response_and_reads_as_cancelled(monkeypatch):
    cancel = CancelHandle()

    class Closing(FakeResponse):
        def read(self) -> bytes:
            raise OSError("connection closed")

    response = Closing({})

    def fake_urlopen(request, timeout=None):
        return response

    monkeypatch.setattr(compose, "urlopen", fake_urlopen)
    client = compose.ComposeClient("http://127.0.0.1:9002", cancel=cancel)
    cancel.cancel()
    with pytest.raises(ComposeError) as caught:
        client.chat("system", "user", 256)
    assert caught.value.reason == "cancelled"


def test_the_handle_closes_the_response_it_holds():
    response = FakeResponse({})
    cancel = CancelHandle()
    assert cancel.attach(response) is True
    cancel.cancel()
    assert response.closed is True
    assert cancel.cancelled is True
    assert cancel.attach(FakeResponse({})) is False


def test_a_body_that_is_not_a_chat_completion_is_an_error(monkeypatch):
    monkeypatch.setattr(compose, "urlopen", lambda *a, **k: FakeResponse({"nope": 1}))
    client = compose.ComposeClient("http://127.0.0.1:9002")
    with pytest.raises(ComposeError) as caught:
        client.chat("system", "user", 256)
    assert caught.value.reason == "error"



def test_the_selection_is_copied_and_the_clipboard_put_back():
    log, backends = make(selection=PARAGRAPH)
    picked = read_selection(backends, sleeper=lambda _s: None, clock=Clock())
    assert picked.copied is True
    assert picked.text == PARAGRAPH
    assert picked.restored is True
    assert [entry[0] for entry in log][:4] == [
        "snapshot",
        "sequence_number",
        "release_held_modifiers",
        "send_copy",
    ]
    assert ("restore", SNAPSHOT, backends.clipboard.sequence) in log


def test_nothing_selected_leaves_the_clipboard_alone_and_says_so():
    log, backends = make(selection=None)
    picked = read_selection(backends, sleeper=lambda _s: None, clock=Clock())
    assert picked.copied is False
    assert picked.text == ""
    assert picked.restored is None
    assert not [entry for entry in log if entry[0] == "restore"]
    assert not [entry for entry in log if entry[0] == "get_text"]


def test_a_clock_that_runs_past_the_wait_ends_the_poll():
    sleeps: list[float] = []
    _log, backends = make(selection=None)
    read_selection(backends, sleeper=sleeps.append, clock=Clock(step=1.0))
    assert sleeps == []


def test_a_clipboard_that_cannot_be_snapshotted_reads_as_nothing_selected():
    log, backends = make(clipboard_error={"snapshot"}, selection=PARAGRAPH)
    picked = read_selection(backends, sleeper=lambda _s: None, clock=Clock())
    assert picked.copied is False
    assert picked.text == ""
    assert not [entry for entry in log if entry[0] == "send_copy"]


def test_a_ctrl_c_that_cannot_be_sent_reads_as_nothing_selected():
    _log, backends = make(input_error={"send_copy"}, selection=PARAGRAPH)
    picked = read_selection(backends, sleeper=lambda _s: None, clock=Clock())
    assert picked.copied is False
    assert picked.text == ""


def test_a_restore_the_target_refused_is_reported_and_not_an_error():
    _log, backends = make(selection=PARAGRAPH)
    backends.clipboard.restore_result = False
    picked = read_selection(backends, sleeper=lambda _s: None, clock=Clock())
    assert picked.copied is True
    assert picked.restored is False
    assert "not restored" in picked.detail


def test_a_subject_line_nobody_asked_for_is_dropped():
    from spells.compose import tidy

    text = "Subject: Quick update\n\nHey Marcel,\n\nI am doing well.\n\nKind regards,"
    assert tidy(text, "write an email to Marcel") == "Hey Marcel,\n\nI am doing well.\n\nKind regards,"


def test_a_subject_line_that_was_asked_for_stays():
    from spells.compose import tidy

    text = "Subject: Invoice\n\nHi Marta,"
    assert tidy(text, "write an email with the subject invoice") == text
    german = tidy("Betreff: Rechnung\n\nHallo", "schreib eine Mail mit dem Betreff Rechnung")
    assert german.startswith("Betreff")


def test_a_placeholder_line_is_dropped_and_an_inline_one_is_left_alone():
    from spells.compose import tidy

    assert tidy("Kind regards,\n[Your Name]", "x") == "Kind regards,"
    assert tidy("Hi [name], see you", "x") == "Hi [name], see you"


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("PyMCA ", "iMac "),
        ("\tPyMCA\n", "\tiMac\n"),
    ],
)
def test_an_edit_keeps_the_spaces_around_the_selection_it_replaces(selection, expected):
    result = write("replace it with iMac", FakeWriter(content="iMac"), selection=selection, mode=ChordMode.EDIT)

    assert result.ok
    assert result.text == expected

