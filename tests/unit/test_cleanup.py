"""cleanup: gate (spec 8.1, V2-9), request (8.2, V2-1), guards (8.3, V2-2, V2-6, V2-7), clean().

The LLM is always a fake here; LlamaClient's HTTP layer is tested against a fake urlopen.
"""

import io
import json
from dataclasses import replace
from urllib.error import HTTPError, URLError

import pytest

from spells import cleanup
from spells.cleanup import (
    CleanupError,
    GateInput,
    LlamaClient,
    clean,
    dominant_language,
    guard,
    max_tokens_for,
    should_clean,
    stopwords,
)
from spells.cleanup_prompt import SYSTEM_PROMPT, user_message
from spells.models import CleanResult, DeliveryMethod, Profile, Transcript

ENABLED = ["en", "de", "sq"]
FILLERS = {"en": ["um", "uh", "you know", "like"], "de": ["äh", "ähm", "also"]}
CORRECTIONS = {"en": ["no wait", "scratch that", "sorry I mean"], "de": ["nein warte", "ich meine"]}
GATE_OK = GateInput(
    cleanup_enabled=True, profile_cleanup=True, engine_available=True, cpu_fallback=False
)
PROFILE = Profile(name="default", cleanup=True, tone="Neutral.", delivery=DeliveryMethod.PASTE)

LONG_EN = "we should meet tomorrow at ten in the morning to talk about the release plan"
LONG_DE = (
    "Ich glaube, dass wir uns morgen um zehn Uhr treffen sollten, um über den Plan zu sprechen"
)


def _guard(transcript, output, finish="stop", language="en", enabled=ENABLED):
    return guard(transcript, output, finish, language, enabled, FILLERS, CORRECTIONS)


def test_gate_disabled_wins_over_everything():
    gate = GateInput(
        cleanup_enabled=False, profile_cleanup=False, engine_available=False, cpu_fallback=True
    )
    assert should_clean("short", "en", FILLERS, CORRECTIONS, gate) == (False, "disabled")


def test_gate_profile_off():
    gate = replace(GATE_OK, profile_cleanup=False, engine_available=False)
    assert should_clean(LONG_EN, "en", FILLERS, CORRECTIONS, gate) == (False, "profile_off")


def test_gate_engine_not_ready():
    gate = replace(GATE_OK, engine_available=False, cpu_fallback=True)
    assert should_clean(LONG_EN, "en", FILLERS, CORRECTIONS, gate) == (False, "engine_not_ready")


def test_gate_cpu_fallback_is_runtime_state():
    gate = replace(GATE_OK, cpu_fallback=True)
    assert should_clean(LONG_EN, "en", FILLERS, CORRECTIONS, gate) == (False, "cpu_fallback")


def test_gate_word_count_boundary_is_twelve():
    eleven = "a b c d e f g h i j k"
    twelve = eleven + " l"
    assert should_clean(eleven, "en", FILLERS, CORRECTIONS, GATE_OK) == (False, "short_clean")
    assert should_clean(twelve, "en", FILLERS, CORRECTIONS, GATE_OK) == (True, "ok")


def test_gate_short_transcript_with_filler_or_correction_runs():
    assert should_clean("um can we meet", "en", FILLERS, CORRECTIONS, GATE_OK) == (True, "ok")
    assert should_clean("Um, can we meet?", "en", FILLERS, CORRECTIONS, GATE_OK) == (True, "ok")
    assert should_clean("at ten no wait eleven", "en", FILLERS, CORRECTIONS, GATE_OK) == (
        True,
        "ok",
    )
    assert should_clean("Also, das geht", "de", FILLERS, CORRECTIONS, GATE_OK) == (True, "ok")


def test_gate_lists_are_per_language():
    assert should_clean("um can we meet", "sq", FILLERS, CORRECTIONS, GATE_OK) == (
        False,
        "short_clean",
    )
    assert should_clean("also das geht", "en", FILLERS, CORRECTIONS, GATE_OK) == (
        False,
        "short_clean",
    )
    assert should_clean("humming along", "en", FILLERS, CORRECTIONS, GATE_OK) == (
        False,
        "short_clean",
    )


def test_gate_on_processor_cleans_long_text_even_without_fillers():
    gate = replace(GATE_OK, cpu_selected=True)
    assert should_clean(LONG_EN, "en", FILLERS, CORRECTIONS, gate) == (True, "ok")
    assert should_clean("a b c d e", "en", FILLERS, CORRECTIONS, gate) == (False, "short_clean")
    with_filler = LONG_EN + " you know"
    assert should_clean(with_filler, "en", FILLERS, CORRECTIONS, gate) == (True, "ok")
    assert should_clean("um yes", "en", FILLERS, CORRECTIONS, gate) == (True, "ok")
    assert should_clean("Nein warte, morgen", "de", FILLERS, CORRECTIONS, gate) == (True, "ok")


def test_gate_skips_a_language_the_cleanup_model_has_no_score_for():
    gate = replace(GATE_OK, cleanup_languages=frozenset({"en", "de"}))
    assert should_clean("um yes " + LONG_EN, "sq", FILLERS, CORRECTIONS, gate) == (
        False,
        "language_unscored",
    )
    assert should_clean(LONG_EN, "en", FILLERS, CORRECTIONS, gate) == (True, "ok")
    unrestricted = replace(GATE_OK, cleanup_languages=None)
    assert should_clean(LONG_EN, "sq", FILLERS, CORRECTIONS, unrestricted) == (True, "ok")


def test_gate_language_unscored_comes_after_the_configuration_reasons():
    gate = replace(GATE_OK, profile_cleanup=False, cleanup_languages=frozenset())
    assert should_clean(LONG_EN, "en", FILLERS, CORRECTIONS, gate) == (False, "profile_off")
    gate = replace(GATE_OK, cleanup_languages=frozenset(), cpu_selected=True)
    assert should_clean("a b", "en", FILLERS, CORRECTIONS, gate) == (False, "language_unscored")


def test_clean_skips_short_text_without_calling_the_model():
    client = FakeLlama(content="should not be used")
    transcript = Transcript(text="See you tomorrow", language="en", duration_s=5.0)
    result = clean(
        transcript,
        PROFILE,
        [],
        FILLERS,
        CORRECTIONS,
        ENABLED,
        replace(GATE_OK, cpu_selected=True),
        client,
    )
    assert result == CleanResult(text="See you tomorrow", used_llm=False, reason="short_clean")
    assert client.calls == []


def test_guard_empty_output():
    assert _guard(LONG_EN, "  \n ") == "empty"


def test_guard_finish_length():
    out = "We should meet tomorrow at ten in the morning to talk about the release plan."
    assert _guard(LONG_EN, out, finish="length") == "finish_length"
    assert _guard(LONG_EN, out, finish="stop") is None


def test_guard_length_ratio_too_long():
    out = (
        "We should meet tomorrow at ten in the morning to talk about the release plan "
        "and the roadmap and the budget too."
    )
    assert _guard(LONG_EN, out) == "length_ratio"


def test_guard_length_ratio_boundaries_are_inclusive():
    transcript = "one two three four five six seven eight nine ten"
    assert _guard(transcript, "alpha beta gamma delta") is None
    assert _guard(transcript, "alpha beta gamma") == "length_ratio"
    assert _guard(transcript, " ".join(["w"] * 13)) is None
    assert _guard(transcript, " ".join(["w"] * 14)) == "length_ratio"


def test_guard_floor_uses_non_filler_word_count():
    transcript = "um uh um uh um uh hello there friend"
    assert 2 < 0.4 * 9, "would fail under the full-count floor"
    assert _guard(transcript, "Hello there.") is None


def test_guard_short_filler_heavy_transcript_passes():
    assert _guard("um, uh, yes", "Yes.") is None


def test_guard_preamble_english():
    out = "Here is the cleaned text: we should meet tomorrow at ten to talk about the release plan"
    assert _guard(LONG_EN, out) == "preamble"


def test_guard_preamble_after_leading_quote_and_punctuation():
    out = '"Sure, we should meet tomorrow at ten in the morning to talk about the release plan."'
    assert _guard(LONG_EN, out) == "preamble"
    out = "... Here's the text: we should meet tomorrow at ten to talk about the release plan"
    assert _guard(LONG_EN, out) == "preamble"


def test_guard_preamble_matches_whole_words_only():
    out = "Surely we should meet tomorrow at ten in the morning to talk about the release plan."
    assert _guard(LONG_EN, out) is None


def test_guard_language_switch_on_a_clear_switch():
    out = "I think that we should meet tomorrow at ten to talk about the plan"
    assert _guard(LONG_DE, out, language="de") == "language_switch"


def test_guard_language_passes_when_output_has_no_clear_dominant_language():
    transcript = "Wir treffen uns morgen um zehn und sprechen über den Plan"
    out = "Meeting tomorrow ten o'clock, plan discussion, sounds good"
    assert _guard(transcript, out, language="de") is None


def test_guard_language_passes_when_transcript_has_no_clear_dominant_language():
    transcript = (
        "Meeting tomorrow ten o'clock plan discussion sounds good roadmap budget installer owners"
    )
    out = "I think that we should meet tomorrow at ten to talk about the plan"
    assert _guard(transcript, out) is None


def test_guard_language_only_considers_enabled_languages_with_lists():
    out = "I think that we should meet tomorrow at ten to talk about the plan"
    assert _guard(LONG_DE, out, language="de", enabled=["de", "fr"]) is None
    assert _guard(LONG_DE, out, language="de", enabled=["de"]) is None
    assert (
        _guard(LONG_DE, out, language="de", enabled=["en", "de", "fr", "xx"]) == "language_switch"
    )
    assert _guard(LONG_DE, out, language="de", enabled=[]) is None


def test_dominant_language_per_language():
    assert dominant_language(LONG_EN, ENABLED) == "en"
    assert dominant_language(LONG_DE, ENABLED) == "de"
    sq = "Unë mendoj se duhet të takohemi nesër në orën dhjetë për të folur për planin"
    assert dominant_language(sq, ENABLED) == "sq"


def test_dominant_language_needs_three_hits_and_a_two_to_one_margin():
    assert dominant_language("the der die das and", ENABLED) is None
    assert dominant_language("the cat sat", ENABLED) is None
    assert dominant_language("the cat and the dog", ENABLED) == "en"
    assert dominant_language("", ENABLED) is None


def test_dominant_language_ignores_languages_without_lists():
    assert dominant_language(LONG_DE, ["fr", "it"]) is None
    assert dominant_language(LONG_DE, ["de", "fr"]) == "de"


class FakeLlama(LlamaClient):
    def __init__(self, content="", finish="stop", error=None):
        super().__init__("http://127.0.0.1:1")
        self.content, self.finish, self.error = content, finish, error
        self.calls = []

    def chat(self, system, user, max_tokens):
        self.calls.append((system, user, max_tokens))
        if self.error is not None:
            raise self.error
        return self.content, self.finish


def _transcript(text=LONG_EN, language="en"):
    return Transcript(text=text, language=language, duration_s=8.0)


def _clean(client, transcript=None, gate=GATE_OK, terms=("Spells",)):
    return clean(
        transcript or _transcript(),
        PROFILE,
        list(terms),
        FILLERS,
        CORRECTIONS,
        ENABLED,
        gate,
        client,
    )


def test_clean_accepted_output_is_stripped_and_marked_used_llm():
    raw = (
        "um so we should meet tomorrow at ten in the morning to talk about the release plan "
        "you know"
    )
    client = FakeLlama(
        "  We should meet tomorrow at ten in the morning to talk about the release plan.\n"
    )
    result = _clean(client, _transcript(raw))
    assert result == CleanResult(
        text="We should meet tomorrow at ten in the morning to talk about the release plan.",
        used_llm=True,
        reason="ok",
    )
    system, user, max_tokens = client.calls[0]
    assert system == SYSTEM_PROMPT
    assert user == user_message(raw, PROFILE.tone, ["Spells"])
    assert max_tokens == max_tokens_for(raw)


def test_clean_rejected_output_returns_raw_text():
    client = FakeLlama(
        "Sure! We should meet tomorrow at ten in the morning to talk about the plan."
    )
    assert _clean(client) == CleanResult(text=LONG_EN, used_llm=False, reason="preamble")


def test_clean_timeout_and_error_yield_raw_text():
    timeout = FakeLlama(error=CleanupError("timed out", reason="timeout"))
    assert _clean(timeout) == CleanResult(text=LONG_EN, used_llm=False, reason="timeout")
    error = FakeLlama(error=CleanupError("HTTP 500"))
    assert _clean(error) == CleanResult(text=LONG_EN, used_llm=False, reason="error")


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_chat_posts_the_spec_parameters(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        body = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}
            ]
        }
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(cleanup, "urlopen", fake_urlopen)
    client = LlamaClient("http://127.0.0.1:8081/", timeout_s=1.5)
    assert client.chat("SYS", "USER", 77) == ("Hello.", "stop")
    req = captured["req"]
    assert req.full_url == "http://127.0.0.1:8081/v1/chat/completions"
    assert req.get_method() == "POST"
    assert captured["timeout"] == 1.5
    assert req.get_header("Content-type") == "application/json"
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert payload["temperature"] == 0
    assert payload["cache_prompt"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] == 77
    assert payload.get("stream", False) is False


def test_chat_null_content_is_empty_string(monkeypatch):
    body = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
    monkeypatch.setattr(
        cleanup, "urlopen", lambda req, timeout=None: FakeResponse(json.dumps(body).encode())
    )
    assert LlamaClient("http://127.0.0.1:8081").chat("s", "u", 10) == ("", "stop")


@pytest.mark.parametrize(
    "exc,reason",
    [
        (TimeoutError("timed out"), "timeout"),
        (URLError("connection refused"), "error"),
        (ConnectionResetError(), "error"),
    ],
)
def test_chat_maps_transport_failures_to_cleanup_error(monkeypatch, exc, reason):
    def boom(req, timeout=None):
        raise exc

    monkeypatch.setattr(cleanup, "urlopen", boom)
    with pytest.raises(CleanupError) as info:
        LlamaClient("http://127.0.0.1:8081").chat("s", "u", 10)
    assert info.value.reason == reason


def test_chat_http_error_and_bad_body_are_errors(monkeypatch):
    def http_500(req, timeout=None):
        raise HTTPError(req.full_url, 500, "boom", None, io.BytesIO(b"{}"))

    monkeypatch.setattr(cleanup, "urlopen", http_500)
    with pytest.raises(CleanupError) as info:
        LlamaClient("http://127.0.0.1:8081").chat("s", "u", 10)
    assert info.value.reason == "error"

    monkeypatch.setattr(cleanup, "urlopen", lambda req, timeout=None: FakeResponse(b"not json"))
    with pytest.raises(CleanupError):
        LlamaClient("http://127.0.0.1:8081").chat("s", "u", 10)

    monkeypatch.setattr(
        cleanup, "urlopen", lambda req, timeout=None: FakeResponse(b'{"choices": []}')
    )
    with pytest.raises(CleanupError):
        LlamaClient("http://127.0.0.1:8081").chat("s", "u", 10)


def test_health(monkeypatch):
    seen = {}

    def ok(req, timeout=None):
        seen["url"] = req.full_url
        return FakeResponse(b'{"status":"ok"}')

    monkeypatch.setattr(cleanup, "urlopen", ok)
    assert LlamaClient("http://127.0.0.1:8081").health() is True
    assert seen["url"] == "http://127.0.0.1:8081/health"

    def loading(req, timeout=None):
        raise HTTPError(req.full_url, 503, "loading", None, io.BytesIO(b""))

    monkeypatch.setattr(cleanup, "urlopen", loading)
    assert LlamaClient("http://127.0.0.1:8081").health() is False


@pytest.mark.parametrize("code", ["sq"])
def test_stopword_lists_have_80_to_120_entries(code):
    entries = stopwords(code)
    assert 80 <= len(entries) <= 120
    assert len(set(entries)) == len(entries)
    assert all(entry == entry.lower() and entry == entry.strip() for entry in entries)
