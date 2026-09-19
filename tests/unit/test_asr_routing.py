from __future__ import annotations

import json
from urllib.error import HTTPError, URLError

import pytest

from spells import asr
from spells.asr import (
    AsrReply,
    LanguagePolicyState,
    LlamaAsrClient,
    SpeechEngineUnavailable,
    SpeechRoute,
    WhisperClient,
    WhisperError,
    language_code,
    language_title,
    parse_asr_output,
    serving_languages,
    transcribe_routed,
)
from spells.models import Engine, EngineId, LangMode

from .fake_clients import FakeLlamaAsrClient

SR = 16000
ENABLED = ["en", "de", "sq"]
QWEN = EngineId(Engine.WHISPER, 0)
FLUTRA = EngineId(Engine.WHISPER, 1)
FAST = SpeechRoute(QWEN, "llama-asr", ("en", "de"))
ALBANIAN = SpeechRoute(FLUTRA, "whisper-server", ("sq",))


def pcm(seconds: float) -> bytes:
    return bytes(2 * int(SR * seconds))


class FakeWhisper(WhisperClient):
    def __init__(self, *responses):
        super().__init__("http://127.0.0.1:1")
        self.responses = list(responses)
        self.calls: list[dict] = []

    def inference(self, wav_bytes, **kwargs):
        assert wav_bytes[:4] == b"RIFF"
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class Clients:
    def __init__(self, **by_key):
        self.by_key = by_key
        self.connected: list[str] = []

    def __call__(self, route: SpeechRoute):
        key = str(route.engine)
        self.connected.append(key)
        return self.by_key.get(key)


def routed(seconds, mode, clients, *, routes=(FAST, ALBANIAN), enabled=ENABLED, last="en",
           terms=()):
    state = LanguagePolicyState(last)
    result = transcribe_routed(pcm(seconds), SR, mode, list(terms), list(enabled), state,
                               routes, clients)
    return result, state


# Output parsing ------------------------------------------------------------------------------


def test_the_language_prefix_is_stripped_and_named():
    assert parse_asr_output("language English<asr_text>Hello there.") == ("English", "Hello there.")
    assert parse_asr_output("  language None<asr_text>") == ("None", "")
    assert parse_asr_output("no prefix at all") == (None, "no prefix at all")
    assert parse_asr_output("language Chinese Mandarin<asr_text> x") == ("Chinese Mandarin", "x")


def test_reported_names_map_to_codes():
    assert language_code("English") == "en"
    assert language_code("German") == "de"
    assert language_code("Hungarian") == "hu"
    assert language_code("None") is None
    assert language_code("") is None
    assert language_code("Klingon") is None
    assert language_title("de") == "German"
    assert language_title("sq") == "Albanian"
    assert language_title("xx") == "xx"


# LlamaAsrClient over stdlib urllib ---------------------------------------------------------------


class StreamResponse:
    def __init__(self, lines):
        self.lines = [line if isinstance(line, bytes) else line.encode() for line in lines]
        self.read_lines = 0
        self.closed = False

    def __iter__(self):
        for line in self.lines:
            self.read_lines += 1
            yield line

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


def sse(*pieces, done=True):
    lines = []
    for piece in pieces:
        chunk = {"choices": [{"index": 0, "delta": {"content": piece}}]}
        lines += [f"data: {json.dumps(chunk)}\n", "\n"]
    if done:
        lines.append("data: [DONE]\n")
    return lines


@pytest.fixture
def server(monkeypatch):
    state = {"responses": [], "requests": []}

    def fake_urlopen(req, timeout=None):
        state["requests"].append((req, timeout))
        response = state["responses"].pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(asr, "urlopen", fake_urlopen)
    return state


def payload(state, index=0):
    req, _timeout = state["requests"][index]
    return json.loads(req.data.decode("utf-8"))


def test_the_request_carries_the_audio_the_vocabulary_and_streams(server):
    server["responses"].append(StreamResponse(sse("language", " English", "<asr_text>", "Hi.")))
    client = LlamaAsrClient("http://127.0.0.1:9/", timeout_s=4.0)
    reply = client.transcribe(b"RIFFdata", vocabulary="Arben, Marta", timeout_s=12.5)
    assert reply == AsrReply("Hi.", "en", "English", True)
    req, timeout = server["requests"][0]
    assert req.full_url == "http://127.0.0.1:9/v1/chat/completions"
    assert timeout == 12.5
    body = payload(server)
    assert body["stream"] is True and body["temperature"] == 0
    assert body["cache_prompt"] is False
    system, user = body["messages"]
    assert system == {"role": "system", "content": ""}
    audio, hint = user["content"]
    assert audio["type"] == "input_audio"
    assert audio["input_audio"]["format"] == "wav"
    assert audio["input_audio"]["data"] == "UklGRmRhdGE="
    assert hint == {"type": "text", "text": "Vocabulary: Arben, Marta"}


def test_no_vocabulary_sends_the_audio_alone(server):
    server["responses"].append(StreamResponse(sse("language English<asr_text>ok")))
    LlamaAsrClient("http://127.0.0.1:9").transcribe(b"RIFF")
    assert len(payload(server)["messages"][1]["content"]) == 1


def test_a_forced_language_is_a_prefill_and_labels_the_reply(server):
    server["responses"].append(StreamResponse(sse("Guten", " Tag.")))
    reply = LlamaAsrClient("http://127.0.0.1:9").transcribe(b"RIFF", language="de")
    assert payload(server)["messages"][-1] == {
        "role": "assistant",
        "content": "language German<asr_text>",
    }
    assert reply == AsrReply("Guten Tag.", "de", "German", True)


def test_a_forced_reply_that_repeats_the_prefill_is_parsed_the_same(server):
    server["responses"].append(StreamResponse(sse("language German<asr_text>Guten Tag.")))
    reply = LlamaAsrClient("http://127.0.0.1:9").transcribe(b"RIFF", language="de")
    assert (reply.text, reply.language) == ("Guten Tag.", "de")


def test_accept_is_asked_once_the_tag_arrives_and_a_refusal_closes_the_stream(server):
    response = StreamResponse(sse("language", " Hungarian", "<asr_text>", "Pers", "hendetje"))
    server["responses"].append(response)
    asked: list[str] = []

    def accept(name):
        asked.append(name)
        return False

    reply = LlamaAsrClient("http://127.0.0.1:9").transcribe(b"RIFF", accept=accept)
    assert asked == ["Hungarian"]
    assert reply == AsrReply("", "hu", "Hungarian", complete=False)
    assert response.closed
    assert response.read_lines < len(response.lines)


def test_an_accepted_language_reads_to_the_end(server):
    server["responses"].append(StreamResponse(sse("language English<asr_text>", "Hi", " there")))
    asked: list[str] = []
    reply = LlamaAsrClient("http://127.0.0.1:9").transcribe(
        b"RIFF", accept=lambda name: asked.append(name) or True
    )
    assert asked == ["English"]
    assert reply == AsrReply("Hi there", "en", "English", True)


def test_output_without_a_tag_is_accepted_as_no_language_after_a_while(server):
    server["responses"].append(StreamResponse(sse("x" * 120, "y")))
    asked: list[str] = []
    reply = LlamaAsrClient("http://127.0.0.1:9").transcribe(
        b"RIFF", accept=lambda name: asked.append(name) or True
    )
    assert asked == [""]
    assert reply.text == "x" * 120 + "y" and reply.language is None


def test_silence_answers_no_language_and_no_text(server):
    server["responses"].append(StreamResponse(sse("language None<asr_text>")))
    reply = LlamaAsrClient("http://127.0.0.1:9").transcribe(b"RIFF")
    assert reply == AsrReply("", None, "None", True)


def test_a_plain_json_body_is_read_too(server):
    body = {"choices": [{"message": {"content": "language English<asr_text>Plain."}}]}
    server["responses"].append(StreamResponse([json.dumps(body).encode()]))
    assert LlamaAsrClient("http://127.0.0.1:9").transcribe(b"RIFF").text == "Plain."


@pytest.mark.parametrize(
    "failure",
    [
        HTTPError("http://x", 400, "bad", {}, None),
        URLError("refused"),
        TimeoutError("slow"),
        ConnectionResetError("reset"),
    ],
)
def test_request_failures_are_speech_errors(server, failure):
    server["responses"].append(failure)
    with pytest.raises(WhisperError):
        LlamaAsrClient("http://127.0.0.1:9").transcribe(b"RIFF")


def test_a_malformed_or_error_chunk_is_a_speech_error(server):
    server["responses"].append(StreamResponse(["data: {not json\n"]))
    with pytest.raises(WhisperError):
        LlamaAsrClient("http://127.0.0.1:9").transcribe(b"RIFF")
    server["responses"].append(StreamResponse(['data: {"error": {"message": "too long"}}\n']))
    with pytest.raises(WhisperError):
        LlamaAsrClient("http://127.0.0.1:9").transcribe(b"RIFF")


def test_health_follows_the_server(monkeypatch):
    monkeypatch.setattr(asr, "urlopen", lambda req, timeout=None: StreamResponse([]))
    assert LlamaAsrClient("http://127.0.0.1:9").health() is True

    def refuse(req, timeout=None):
        raise URLError("refused")

    monkeypatch.setattr(asr, "urlopen", refuse)
    assert LlamaAsrClient("http://127.0.0.1:9").health() is False


# Routing -------------------------------------------------------------------------------------


def test_languages_are_assigned_to_the_engine_that_lists_them_and_the_rest_to_the_first():
    assert serving_languages((FAST, ALBANIAN), ["sq", "en", "fr", "de"]) == [
        ("en", "fr", "de"),
        ("sq",),
    ]
    catch_all = SpeechRoute("legacy", "whisper-server", ())
    assert serving_languages((ALBANIAN, catch_all), ["en", "sq"]) == [("sq",), ("en",)]


def test_a_locked_language_goes_to_its_engine():
    flutra = FakeWhisper({"text": "Mirëdita", "language": "albanian"})
    clients = Clients(**{"whisper-2": flutra})
    result, state = routed(5.0, LangMode("locked", "sq"), clients)
    assert clients.connected == ["whisper-2"]
    assert flutra.calls[0]["language"] == "sq"
    assert (result.text, result.language, result.engine) == ("Mirëdita", "sq", "whisper-2")
    assert state.last_accepted == "sq"


def test_a_forced_language_on_the_fast_engine_is_prefilled():
    qwen = FakeLlamaAsrClient(AsrReply("Guten Tag.", "de", "German"))
    clients = Clients(whisper=qwen)
    result, state = routed(1.0, LangMode("forced", "de"), clients, terms=["Arben"])
    assert qwen.calls == [{"language": "de", "accept": None, "vocabulary": "Arben",
                           "timeout_s": pytest.approx(12.0)}]
    assert (result.language, result.engine, result.fallback_used) == ("de", "whisper", False)
    assert state.last_accepted == "de"


def test_a_short_auto_recording_uses_the_engine_of_the_last_accepted_language():
    flutra = FakeWhisper({"text": "Po", "language": "albanian"})
    clients = Clients(**{"whisper-2": flutra})
    result, _state = routed(1.5, LangMode("auto"), clients, last="sq")
    assert flutra.calls[0]["language"] == "sq"
    assert result.engine == "whisper-2"


def test_auto_asks_the_fast_engine_first_and_keeps_a_language_it_serves():
    qwen = FakeLlamaAsrClient(AsrReply("Hello there.", "en", "English"), reports="English")
    clients = Clients(whisper=qwen)
    result, state = routed(15.0, LangMode("auto"), clients, last="sq")
    assert clients.connected == ["whisper"]
    assert qwen.asked == ["English"]
    assert (result.text, result.language, result.engine) == ("Hello there.", "en", "whisper")
    assert result.fallback_used is False
    assert state.last_accepted == "en"


def test_auto_closes_the_stream_on_another_language_and_locks_the_albanian_engine():
    qwen = FakeLlamaAsrClient(AsrReply("Pers hendetje", "hu", "Hungarian"), reports="Hungarian")
    flutra = FakeWhisper({"text": "Përshëndetje", "language": "albanian"})
    clients = Clients(whisper=qwen, **{"whisper-2": flutra})
    result, state = routed(15.0, LangMode("auto"), clients)
    assert qwen.stopped == 1
    assert [call["language"] for call in flutra.calls] == ["sq"]
    assert (result.text, result.language, result.engine) == ("Përshëndetje", "sq", "whisper-2")
    assert result.fallback_used is False
    assert state.last_accepted == "sq"


def test_auto_accepts_no_language_and_silence_returns_nothing():
    qwen = FakeLlamaAsrClient(AsrReply("", None, "None"), reports="None")
    clients = Clients(whisper=qwen)
    result, state = routed(3.0, LangMode("auto"), clients, last="de")
    assert result is None
    assert clients.connected == ["whisper"]
    assert state.last_accepted == "de"


def test_a_reported_language_served_elsewhere_is_locked_on_that_engine():
    qwen = FakeLlamaAsrClient(AsrReply("", "sq", "Albanian"), reports="Albanian")
    flutra = FakeWhisper({"text": "Tungjatjeta", "language": "albanian"})
    clients = Clients(whisper=qwen, **{"whisper-2": flutra})
    result, _state = routed(12.0, LangMode("auto"), clients)
    assert qwen.stopped == 1
    assert flutra.calls[0]["language"] == "sq"
    assert result.language == "sq"


def test_several_remaining_languages_use_the_next_engines_own_auto_policy():
    italian = SpeechRoute(EngineId(Engine.WHISPER, 1), "whisper-server", ("sq", "it"))
    qwen = FakeLlamaAsrClient(AsrReply("", "pl", "Polish"), reports="Polish")
    whisper = FakeWhisper(
        {"text": "Ciao a tutti", "language": "polish"},
        {"language_probabilities": {"it": 0.7, "sq": 0.2, "en": 0.9}},
        {"text": "Ciao a tutti", "language": "italian"},
    )
    clients = Clients(whisper=qwen, **{"whisper-2": whisper})
    result, _state = routed(12.0, LangMode("auto"), clients, routes=(FAST, italian),
                            enabled=["en", "de", "sq", "it"])
    assert [call["language"] for call in whisper.calls] == ["auto", "auto", "it"]
    assert whisper.calls[1]["detect_only"] is True
    assert (result.language, result.fallback_used) == ("it", True)


def test_without_another_engine_the_fast_engine_keeps_reading_and_guesses_the_language():
    qwen = FakeLlamaAsrClient(AsrReply("Ni hao", "zh", "Chinese"), reports="Chinese")
    clients = Clients(whisper=qwen)
    result, _state = routed(12.0, LangMode("auto"), clients, routes=(FAST,),
                            enabled=["en", "de"], last="de")
    assert qwen.stopped == 0
    assert (result.text, result.language, result.fallback_used) == ("Ni hao", "de", True)


def test_a_single_whisper_engine_follows_the_classic_policy():
    whisper = FakeWhisper(
        {"text": "Bonjour", "language": "french"},
        {"language_probabilities": {"de": 0.2, "sq": 0.6}},
        {"text": "Mirëdita", "language": "albanian"},
    )
    only = SpeechRoute(EngineId(Engine.WHISPER, 0), "whisper-server", ())
    clients = Clients(whisper=whisper)
    result, state = routed(12.0, LangMode("auto"), clients, routes=(only,))
    assert [call["language"] for call in whisper.calls] == ["auto", "auto", "sq"]
    assert (result.language, result.fallback_used, state.last_accepted) == ("sq", True, "sq")


def test_whisper_first_hands_a_language_it_does_not_serve_to_its_engine():
    general = SpeechRoute(EngineId(Engine.WHISPER, 0), "whisper-server", ("en", "de", "sq"))
    french = SpeechRoute(EngineId(Engine.WHISPER, 1), "llama-asr", ("fr",))
    whisper = FakeWhisper({"text": "Bonjour", "language": "french"})
    qwen = FakeLlamaAsrClient(AsrReply("Bonjour à tous", "fr", "French"))
    clients = Clients(whisper=whisper, **{"whisper-2": qwen})
    result, _state = routed(12.0, LangMode("auto"), clients, routes=(general, french),
                            enabled=["en", "de", "sq", "fr"])
    assert qwen.calls[0]["language"] == "fr"
    assert (result.text, result.engine) == ("Bonjour à tous", "whisper-2")


def test_a_recording_too_long_for_the_fast_engine_goes_to_whisper():
    flutra = FakeWhisper({"text": "long english", "language": "english"})
    clients = Clients(**{"whisper-2": flutra})
    seconds = asr.LLAMA_ASR_MAX_AUDIO_S + 1
    result, _state = routed(seconds, LangMode("locked", "en"), clients)
    assert clients.connected == ["whisper-2"]
    assert flutra.calls[0]["language"] == "en"
    assert flutra.calls[0]["timeout_s"] == pytest.approx(10.0 + 2.0 * seconds)
    assert result.engine == "whisper-2"


def test_a_hallucination_from_the_fast_engine_is_dropped():
    qwen = FakeLlamaAsrClient(AsrReply("Thanks for watching!", "en", "English"),
                              reports="English")
    result, state = routed(5.0, LangMode("auto"), Clients(whisper=qwen), last="de")
    assert result is None and state.last_accepted == "de"


def test_an_engine_that_does_not_serve_raises_unavailable_with_its_id():
    with pytest.raises(SpeechEngineUnavailable) as caught:
        routed(5.0, LangMode("locked", "sq"), Clients())
    assert caught.value.engine == FLUTRA


def test_a_failed_request_names_the_engine_that_failed():
    qwen = FakeLlamaAsrClient(WhisperError("boom"))
    with pytest.raises(WhisperError) as caught:
        routed(5.0, LangMode("locked", "en"), Clients(whisper=qwen))
    assert caught.value.engine == QWEN
    flutra = FakeWhisper(WhisperError("down"))
    with pytest.raises(WhisperError) as caught:
        routed(5.0, LangMode("locked", "sq"), Clients(**{"whisper-2": flutra}))
    assert caught.value.engine == FLUTRA


def test_routing_needs_an_engine_and_a_code_for_locked_modes():
    with pytest.raises(ValueError):
        routed(5.0, LangMode("auto"), Clients(), routes=())
    with pytest.raises(ValueError):
        routed(5.0, LangMode("forced"), Clients())


# confidence scores across engines (spec 8.4)


def whisper_body(text: str, language: str = "albanian", avg_logprob: float = -0.3) -> dict:
    return {
        "text": text,
        "language": language,
        "segments": [
            {
                "start": 0.0,
                "end": 4.0,
                "avg_logprob": avg_logprob,
                "compression_ratio": 1.2,
                "no_speech_prob": 0.03,
                "temperature": 0.0,
            }
        ],
    }


def test_a_whisper_route_reports_its_scores():
    clients = Clients(**{str(FLUTRA): FakeWhisper(whisper_body("Mire se vini"))})
    seen: list = []
    state = LanguagePolicyState("sq")
    transcript = transcribe_routed(
        pcm(5), SR, LangMode("locked", "sq"), [], ENABLED, state, (FAST, ALBANIAN), clients, seen.append
    )
    assert transcript is not None
    assert seen[0].avg_logprob == pytest.approx(-0.3)
    assert seen[0].source == "whisper-server"


def test_a_llama_asr_route_reports_its_runtime_and_nothing_else():
    clients = Clients(**{str(QWEN): FakeLlamaAsrClient(AsrReply("Hello there", "en", "English"))})
    seen: list = []
    state = LanguagePolicyState("en")
    transcript = transcribe_routed(
        pcm(5), SR, LangMode("locked", "en"), [], ENABLED, state, (FAST, ALBANIAN), clients, seen.append
    )
    assert transcript is not None
    assert seen[0].source == "llama-asr"
    assert seen[0].scored is False


def test_auto_mode_on_a_whisper_route_reports_the_accepted_scores():
    clients = Clients(**{str(FLUTRA): FakeWhisper(whisper_body("Mire se vini", avg_logprob=-0.42))})
    seen: list = []
    state = LanguagePolicyState("sq")
    transcribe_routed(
        pcm(5), SR, LangMode("auto"), [], ["sq"], state, (ALBANIAN,), clients, seen.append
    )
    assert seen[0].avg_logprob == pytest.approx(-0.42)
