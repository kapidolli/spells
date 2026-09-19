"""asr: whisper client, language policy (spec 7.2 with V2-8), prompt rule, hallucination filter.

All requests go to fakes; nothing here touches the network.
"""

import io
import json
import re
import wave
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from spells import asr
from spells.asr import (
    LanguagePolicyState,
    WhisperClient,
    WhisperError,
    build_prompt,
    is_hallucination,
    name_to_code,
    pcm16_to_wav,
    transcribe,
)
from spells.models import LangMode, Transcript

REPO = Path(__file__).resolve().parents[2]
LANGUAGES_JSON = REPO / "data" / "whisper_languages.json"
WHISPER_SOURCE = REPO / "build" / "cache" / "whisper.cpp-1.9.4" / "src" / "whisper.cpp"
LANG_ROW = re.compile(r'\{\s*"(\w+)",\s*\{\s*(\d+),\s*"([^"]+)",?\s*\}\s*\}')

SR = 16000
ENABLED = ["en", "de", "sq"]


def language_table(source: str) -> dict[str, str]:
    """Generator for data/whisper_languages.json: {"english": "en", ...} in whisper id order.

    The source is whisper.cpp v1.9.4 src/whisper.cpp (its g_lang table maps a code to an id
    and a lowercase full name). To regenerate the committed file from the build cache, run
    with .venv/Scripts/python.exe:

        import json, sys
        sys.path.insert(0, "tests/unit")
        from test_asr import LANGUAGES_JSON, WHISPER_SOURCE, language_table
        table = language_table(WHISPER_SOURCE.read_text(encoding="utf-8"))
        LANGUAGES_JSON.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")

    Without the cache, fetch the same file from
    https://raw.githubusercontent.com/ggml-org/whisper.cpp/v1.9.4/src/whisper.cpp first.
    """
    table = source[source.index("g_lang = {") :]
    table = table[: table.index("\n};")]
    rows = [(int(id_), name, code) for code, id_, name in LANG_ROW.findall(table)]
    return {name: code for _, name, code in sorted(rows)}


def pcm(seconds: float) -> bytes:
    return bytes(2 * int(SR * seconds))


def resp(text: str, language: str = "english", **extra) -> dict:
    return {"text": text, "language": language, **extra}


class FakeWhisper(WhisperClient):
    """Returns canned responses in order and records the policy-relevant call arguments."""

    def __init__(self, *responses: dict):
        super().__init__("http://127.0.0.1:1")
        self.responses = list(responses)
        self.calls: list[dict] = []

    def inference(self, wav_bytes, **kwargs):
        assert wav_bytes[:4] == b"RIFF"
        self.calls.append(
            {
                "language": kwargs.get("language"),
                "prompt": kwargs.get("prompt"),
                "detect_only": kwargs.get("detect_only", False),
                "want_probabilities": kwargs.get("want_probabilities", False),
                "carry_initial_prompt": kwargs.get("carry_initial_prompt", True),
                "temperature": kwargs.get("temperature", 0.0),
                "timeout_s": kwargs.get("timeout_s"),
            }
        )
        return self.responses.pop(0)


# wav


def test_pcm16_to_wav_header_and_payload():
    data = b"\x01\x02" * SR
    out = pcm16_to_wav(data, SR)
    assert out[:4] == b"RIFF" and out[8:12] == b"WAVE"
    with wave.open(io.BytesIO(out)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == SR
        assert w.getnframes() == SR
        assert w.readframes(SR) == data


def test_pcm16_to_wav_header_for_a_non_16khz_rate():
    rate = 44100
    data = b"\x03\x04" * rate
    out = pcm16_to_wav(data, rate)
    with wave.open(io.BytesIO(out)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == rate
        assert w.getnframes() == rate
        assert w.readframes(rate) == data
    byte_rate = int.from_bytes(out[28:32], "little")
    block_align = int.from_bytes(out[32:34], "little")
    assert byte_rate == rate * 1 * 2
    assert block_align == 1 * 2


# vocabulary prompt (spec 7.3, decisions V2-5 and E8)


def test_build_prompt_joins_oldest_to_newest():
    assert build_prompt(["Alpha", "Beta", "Gamma"]) == "Alpha, Beta, Gamma"


def test_build_prompt_skips_blank_terms_and_trims():
    assert build_prompt(["", " Alpha ", "  "]) == "Alpha"
    assert build_prompt([]) == ""


def test_build_prompt_truncates_from_the_front_to_500_chars():
    terms = [f"term{i:03d}" for i in range(100)]
    joined = ", ".join(terms)
    assert len(joined) > 500
    prompt = build_prompt(terms)
    assert len(prompt) <= 500
    assert joined.endswith(prompt)
    assert prompt.endswith("term099")
    assert "term000" not in prompt
    assert not prompt.startswith((",", " "))


def test_build_prompt_exactly_500_is_untouched():
    term = "x" * 500
    assert build_prompt([term]) == term


# language table


def test_language_json_has_100_lowercase_names_and_unique_codes():
    table = json.loads(LANGUAGES_JSON.read_text(encoding="utf-8"))
    assert len(table) == 100
    assert table["english"] == "en"
    assert table["german"] == "de"
    assert table["albanian"] == "sq"
    assert all(name == name.lower() for name in table)
    codes = set(table.values())
    assert len(codes) == 100
    assert {"haw", "yue", "jw"} <= codes


@pytest.mark.skipif(not WHISPER_SOURCE.exists(), reason="whisper.cpp source not in build cache")
def test_language_json_matches_whisper_source():
    generated = language_table(WHISPER_SOURCE.read_text(encoding="utf-8"))
    committed = json.loads(LANGUAGES_JSON.read_text(encoding="utf-8"))
    assert generated == committed
    assert list(generated) == list(committed)


def test_name_to_code():
    assert name_to_code("english") == "en"
    assert name_to_code(" German ") == "de"
    assert name_to_code("albanian") == "sq"
    assert name_to_code("cantonese") == "yue"
    with pytest.raises(KeyError):
        name_to_code("klingon")


# hallucination filter (spec 7.4)


@pytest.mark.parametrize(
    "text",
    [
        "Thanks for watching!",
        "Thank you for watching.",
        "  thanks for watching  ",
        "Untertitel im Auftrag des ZDF",
        "Untertitelung des ZDF, 2020",
        "Untertitel von Stephanie Geiges",
        "Subtitles by the Amara.org community",
    ],
)
def test_known_hallucinations_are_filtered(text):
    assert is_hallucination(text)


@pytest.mark.parametrize(
    "text",
    [
        "Thanks for the update",
        "Untertitel",
        "",
        "We watched the film",
        "Thanks for watching it",
        "Faleminderit.",
        "faleminderit",
    ],
)
def test_ordinary_text_is_not_a_hallucination(text):
    assert not is_hallucination(text)


# language policy (spec 7.2 with V2-8)


def test_locked_mode_forces_the_locked_language():
    client = FakeWhisper(resp(" Hallo Welt ", "german"))
    state = LanguagePolicyState("en")
    t = transcribe(pcm(3), SR, LangMode("locked", "de"), ["Spells"], ENABLED, state, client)
    assert t == Transcript(text="Hallo Welt", language="de", duration_s=3.0, fallback_used=False)
    assert client.calls == [
        {
            "language": "de",
            "prompt": "Spells",
            "detect_only": False,
            "want_probabilities": False,
            "carry_initial_prompt": True,
            "temperature": 0.0,
            "timeout_s": 16.0,
        }
    ]
    assert state.last_accepted == "de"


def test_forced_mode_forces_the_chord_language_even_for_short_clips():
    client = FakeWhisper(resp("Tungjatjeta", "albanian"))
    state = LanguagePolicyState("en")
    t = transcribe(pcm(1), SR, LangMode("forced", "sq"), [], ENABLED, state, client)
    assert t is not None and t.language == "sq" and not t.fallback_used
    assert client.calls[0]["language"] == "sq"
    assert state.last_accepted == "sq"


def test_auto_short_clip_uses_last_accepted_language():
    client = FakeWhisper(resp("Guten Morgen", "german"))
    state = LanguagePolicyState("de")
    t = transcribe(pcm(1.9), SR, LangMode("auto"), [], ENABLED, state, client)
    assert t is not None
    assert t.language == "de" and t.duration_s == pytest.approx(1.9) and not t.fallback_used
    assert [c["language"] for c in client.calls] == ["de"]
    assert state.last_accepted == "de"


def test_auto_two_seconds_is_not_short():
    client = FakeWhisper(resp("Good morning", "english"))
    state = LanguagePolicyState("de")
    t = transcribe(pcm(2.0), SR, LangMode("auto"), [], ENABLED, state, client)
    assert t is not None and t.language == "en"
    assert [c["language"] for c in client.calls] == ["auto"]
    assert state.last_accepted == "en"


def test_auto_long_clip_accepts_a_detected_enabled_language():
    client = FakeWhisper(resp("Guten Morgen", "german"))
    state = LanguagePolicyState("en")
    t = transcribe(pcm(5), SR, LangMode("auto"), ["Spells"], ENABLED, state, client)
    assert t == Transcript(text="Guten Morgen", language="de", duration_s=5.0, fallback_used=False)
    assert client.calls == [
        {
            "language": "auto",
            "prompt": "Spells",
            "detect_only": False,
            "want_probabilities": False,
            "carry_initial_prompt": True,
            "temperature": 0.0,
            "timeout_s": 20.0,
        }
    ]
    assert state.last_accepted == "de"


def test_auto_fallback_when_detected_language_is_not_enabled():
    probabilities = {"fr": 0.8, "de": 0.15, "en": 0.05}
    client = FakeWhisper(
        resp("Bonjour", "french"),
        resp("", "french", language_probabilities=probabilities),
        resp("Guten Tag", "german"),
    )
    state = LanguagePolicyState("en")
    t = transcribe(pcm(5), SR, LangMode("auto"), ["Spells"], ENABLED, state, client)
    assert t == Transcript(text="Guten Tag", language="de", duration_s=5.0, fallback_used=True)
    common = {"carry_initial_prompt": True, "temperature": 0.0, "timeout_s": 20.0}
    assert client.calls == [
        {"language": "auto", "prompt": "Spells", "detect_only": False, "want_probabilities": False, **common},
        {"language": "auto", "prompt": "Spells", "detect_only": True, "want_probabilities": True, **common},
        {"language": "de", "prompt": "Spells", "detect_only": False, "want_probabilities": False, **common},
    ]
    assert state.last_accepted == "de"


def test_auto_fallback_when_the_name_is_unknown():
    client = FakeWhisper(
        resp("???", "klingon"),
        resp("", "klingon", language_probabilities={"sq": 0.6, "en": 0.3}),
        resp("Mirëdita", "albanian"),
    )
    state = LanguagePolicyState("en")
    t = transcribe(pcm(5), SR, LangMode("auto"), [], ENABLED, state, client)
    assert t is not None and t.language == "sq" and t.fallback_used
    assert client.calls[2]["language"] == "sq"


def test_auto_fallback_ignores_probabilities_of_disabled_languages():
    client = FakeWhisper(
        resp("Ciao", "italian"),
        resp("", "italian", language_probabilities={"it": 0.9, "sq": 0.05, "en": 0.04}),
        resp("Mirëdita", "albanian"),
    )
    state = LanguagePolicyState("en")
    t = transcribe(pcm(5), SR, LangMode("auto"), [], ENABLED, state, client)
    assert t is not None and t.language == "sq"
    assert client.calls[2]["language"] == "sq"


def test_auto_fallback_missing_entry_counts_as_zero():
    client = FakeWhisper(
        resp("Ciao", "italian"),
        resp("", "italian", language_probabilities={"it": 0.99, "de": 0.0009}),
        resp("Guten Tag", "german"),
    )
    state = LanguagePolicyState("sq")
    t = transcribe(pcm(5), SR, LangMode("auto"), [], ENABLED, state, client)
    assert t is not None and t.language == "de"
    assert client.calls[2]["language"] == "de"


def test_auto_fallback_all_missing_uses_last_accepted():
    client = FakeWhisper(
        resp("Ciao", "italian"),
        resp("", "italian", language_probabilities={"it": 0.99}),
        resp("Mirëdita", "albanian"),
    )
    state = LanguagePolicyState("sq")
    t = transcribe(pcm(5), SR, LangMode("auto"), [], ENABLED, state, client)
    assert t is not None and t.language == "sq" and t.fallback_used
    assert client.calls[2]["language"] == "sq"


def test_auto_fallback_without_probabilities_key_uses_last_accepted():
    client = FakeWhisper(resp("Ciao", "italian"), {"text": ""}, resp("Guten Tag", "german"))
    state = LanguagePolicyState("de")
    t = transcribe(pcm(5), SR, LangMode("auto"), [], ENABLED, state, client)
    assert t is not None and t.language == "de"


def test_auto_fallback_tie_prefers_enabled_order():
    client = FakeWhisper(
        resp("Ciao", "italian"),
        resp("", "italian", language_probabilities={"de": 0.2, "en": 0.2}),
        resp("Good day", "english"),
    )
    state = LanguagePolicyState("sq")
    t = transcribe(pcm(5), SR, LangMode("auto"), [], ["en", "de", "sq"], state, client)
    assert t is not None and t.language == "en"


def test_empty_text_returns_none_and_keeps_state():
    client = FakeWhisper(resp("  \n ", "german"))
    state = LanguagePolicyState("en")
    assert transcribe(pcm(5), SR, LangMode("auto"), [], ENABLED, state, client) is None
    assert state.last_accepted == "en"
    assert len(client.calls) == 1


def test_hallucination_returns_none_and_keeps_state():
    client = FakeWhisper(resp(" Thanks for watching! ", "english"))
    state = LanguagePolicyState("de")
    assert transcribe(pcm(5), SR, LangMode("locked", "en"), [], ENABLED, state, client) is None
    assert state.last_accepted == "de"


def test_prompt_is_built_from_terms_for_every_request():
    client = FakeWhisper(resp("hi", "english"))
    terms = [f"term{i:03d}" for i in range(100)]
    state = LanguagePolicyState("en")
    transcribe(pcm(5), SR, LangMode("locked", "en"), terms, ENABLED, state, client)
    assert client.calls[0]["prompt"] == build_prompt(terms)


def test_carry_initial_prompt_is_sent_on_the_policy_path():
    client = FakeWhisper(resp("Guten Morgen", "german"))
    state = LanguagePolicyState("en")
    transcribe(pcm(5), SR, LangMode("auto"), ["Spells"], ENABLED, state, client)
    assert client.calls[0]["carry_initial_prompt"] is True


@pytest.mark.parametrize("seconds,expected_timeout", [(2.0, 14.0), (60.0, 130.0)])
def test_transcribe_scales_timeout_with_audio_length_for_all_policy_requests(
    seconds, expected_timeout
):
    probabilities = {"fr": 0.8, "de": 0.15, "en": 0.05}
    client = FakeWhisper(
        resp("Bonjour", "french"),
        resp("", "french", language_probabilities=probabilities),
        resp("Guten Tag", "german"),
    )
    state = LanguagePolicyState("en")
    transcribe(pcm(seconds), SR, LangMode("auto"), ["Spells"], ENABLED, state, client)
    assert len(client.calls) == 3
    for call in client.calls:
        assert call["timeout_s"] == pytest.approx(expected_timeout)


def test_locked_mode_timeout_scales_with_a_sixty_second_clip():
    client = FakeWhisper(resp("A long dictation.", "english"))
    state = LanguagePolicyState("en")
    transcribe(pcm(60.0), SR, LangMode("locked", "en"), [], ENABLED, state, client)
    assert client.calls[0]["timeout_s"] == pytest.approx(130.0)


def test_short_clip_auto_request_gets_the_scaled_timeout():
    client = FakeWhisper(resp("Hi", "english"))
    state = LanguagePolicyState("en")
    transcribe(pcm(1.0), SR, LangMode("auto"), [], ENABLED, state, client)
    assert client.calls[0]["timeout_s"] == pytest.approx(12.0)


# WhisperClient over stdlib urllib


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.body = body
        self.status = status

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def parse_multipart(body: bytes, boundary: str) -> dict[str, tuple[str | None, str | None, bytes]]:
    parts = {}
    delimiter = b"--" + boundary.encode()
    chunks = body.split(delimiter)
    assert chunks[0] == b""
    assert chunks[-1] == b"--\r\n"
    for chunk in chunks[1:-1]:
        assert chunk.startswith(b"\r\n") and chunk.endswith(b"\r\n")
        head, _, payload = chunk[2:-2].partition(b"\r\n\r\n")
        headers = head.decode()
        name = re.search(r'name="([^"]+)"', headers).group(1)
        filename = re.search(r'filename="([^"]+)"', headers)
        ctype = re.search(r"Content-Type: (\S+)", headers)
        parts[name] = (
            filename.group(1) if filename else None,
            ctype.group(1) if ctype else None,
            payload,
        )
    return parts


@pytest.fixture
def capture(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        return FakeResponse(b'{"text": " hi ", "language": "english"}')

    monkeypatch.setattr(asr, "urlopen", fake_urlopen)
    return captured


def _fields(req) -> dict[str, tuple[str | None, str | None, bytes]]:
    boundary = req.get_header("Content-type").split("boundary=", 1)[1]
    return parse_multipart(req.data, boundary)


def test_inference_posts_multipart_with_the_spec_fields(capture):
    client = WhisperClient("http://127.0.0.1:8080/", timeout_s=3.0)
    result = client.inference(b"RIFFwav", language="auto", prompt="Spells, Vulkan")
    assert result == {"text": " hi ", "language": "english"}
    req = capture["req"]
    assert req.full_url == "http://127.0.0.1:8080/inference"
    assert req.get_method() == "POST"
    assert capture["timeout"] == 3.0
    assert req.get_header("Content-type").startswith("multipart/form-data; boundary=")
    parts = _fields(req)
    assert parts["file"] == ("audio.wav", "audio/wav", b"RIFFwav")
    fields = {k: v[2].decode() for k, v in parts.items() if k != "file"}
    assert fields == {
        "response_format": "verbose_json",
        "temperature": "0.0",
        "language": "auto",
        "prompt": "Spells, Vulkan",
        "detect_language": "false",
        "no_language_probabilities": "true",
        "carry_initial_prompt": "true",
    }


def test_inference_per_request_timeout_overrides_the_client_default(capture):
    client = WhisperClient("http://127.0.0.1:8080/", timeout_s=3.0)
    client.inference(b"RIFFwav", language="auto", prompt="Spells", timeout_s=45.0)
    assert capture["timeout"] == 45.0


def test_inference_without_per_request_timeout_uses_the_client_default(capture):
    client = WhisperClient("http://127.0.0.1:8080/", timeout_s=3.0)
    client.inference(b"RIFFwav", language="auto", prompt="Spells")
    assert capture["timeout"] == 3.0


def test_inference_omits_language_when_none_and_sets_detect_flags(capture):
    client = WhisperClient("http://127.0.0.1:8080")
    client.inference(
        b"RIFF",
        language=None,
        prompt="",
        detect_only=True,
        want_probabilities=True,
        carry_initial_prompt=False,
        temperature=0.2,
    )
    fields = {k: v[2].decode() for k, v in _fields(capture["req"]).items() if k != "file"}
    assert "language" not in fields
    assert fields["detect_language"] == "true"
    assert fields["no_language_probabilities"] == "false"
    assert fields["carry_initial_prompt"] == "false"
    assert fields["temperature"] == "0.2"


def test_inference_prompt_survives_unicode(capture):
    client = WhisperClient("http://127.0.0.1:8080")
    client.inference(b"RIFF", language="sq", prompt="Përshëndetje, Zürich")
    assert _fields(capture["req"])["prompt"][2].decode("utf-8") == "Përshëndetje, Zürich"


def test_inference_http_error_raises_whisper_error(monkeypatch):
    def boom(req, timeout=None):
        raise HTTPError(req.full_url, 500, "Internal Server Error", None, io.BytesIO(b"bad"))

    monkeypatch.setattr(asr, "urlopen", boom)
    with pytest.raises(WhisperError):
        WhisperClient("http://127.0.0.1:8080").inference(b"RIFF", language="en", prompt="")


def test_inference_connection_error_raises_whisper_error(monkeypatch):
    def boom(req, timeout=None):
        raise URLError("connection refused")

    monkeypatch.setattr(asr, "urlopen", boom)
    with pytest.raises(WhisperError):
        WhisperClient("http://127.0.0.1:8080").inference(b"RIFF", language="en", prompt="")


def test_inference_bad_json_raises_whisper_error(monkeypatch):
    monkeypatch.setattr(asr, "urlopen", lambda req, timeout=None: FakeResponse(b"not json"))
    with pytest.raises(WhisperError):
        WhisperClient("http://127.0.0.1:8080").inference(b"RIFF", language="en", prompt="")


def test_health_true_on_200(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return FakeResponse(b'{"status":"ok"}')

    monkeypatch.setattr(asr, "urlopen", fake_urlopen)
    assert WhisperClient("http://127.0.0.1:8080").health() is True
    assert seen["url"] == "http://127.0.0.1:8080/health"


def test_health_false_on_503_or_connection_error(monkeypatch):
    def loading(req, timeout=None):
        raise HTTPError(req.full_url, 503, "loading", None, io.BytesIO(b""))

    monkeypatch.setattr(asr, "urlopen", loading)
    assert WhisperClient("http://127.0.0.1:8080").health() is False

    def refused(req, timeout=None):
        raise URLError("refused")

    monkeypatch.setattr(asr, "urlopen", refused)
    assert WhisperClient("http://127.0.0.1:8080").health() is False


# confidence scores of the accepted answer (spec 8.4)


def segments(**overrides) -> list[dict]:
    base = {
        "start": 0.0,
        "end": 4.0,
        "avg_logprob": -0.25,
        "compression_ratio": 1.3,
        "no_speech_prob": 0.02,
        "temperature": 0.0,
    }
    base.update(overrides)
    return [base]


def test_transcribe_reports_the_scores_of_the_accepted_answer():
    client = FakeWhisper(resp("Hello there", segments=segments()))
    state = LanguagePolicyState("en")
    seen: list = []
    t = transcribe(pcm(3), SR, LangMode("locked", "en"), [], ENABLED, state, client, seen.append)
    assert t is not None
    assert len(seen) == 1
    assert seen[0].avg_logprob == pytest.approx(-0.25)
    assert seen[0].source == "whisper-server"


def test_transcribe_without_a_sink_still_works():
    client = FakeWhisper(resp("Hello there", segments=segments()))
    state = LanguagePolicyState("en")
    assert transcribe(pcm(3), SR, LangMode("locked", "en"), [], ENABLED, state, client) is not None


def test_an_empty_transcript_reports_no_scores():
    client = FakeWhisper(resp("", segments=segments()))
    state = LanguagePolicyState("en")
    seen: list = []
    assert transcribe(pcm(3), SR, LangMode("locked", "en"), [], ENABLED, state, client, seen.append) is None
    assert seen == []


def test_a_hallucination_reports_no_scores():
    client = FakeWhisper(resp("Thanks for watching!", segments=segments()))
    state = LanguagePolicyState("en")
    seen: list = []
    assert transcribe(pcm(3), SR, LangMode("locked", "en"), [], ENABLED, state, client, seen.append) is None
    assert seen == []


def test_the_fallback_path_reports_the_scores_of_the_final_request():
    client = FakeWhisper(
        resp("wrong", "french", segments=segments(avg_logprob=-1.8)),
        {"language_probabilities": {"de": 0.9}},
        resp("Guten Tag", "german", segments=segments(avg_logprob=-0.2)),
    )
    state = LanguagePolicyState("en")
    seen: list = []
    t = transcribe(pcm(5), SR, LangMode("auto"), [], ENABLED, state, client, seen.append)
    assert t is not None and t.fallback_used
    assert seen[0].avg_logprob == pytest.approx(-0.2)


# Aborting a request in flight (spec 6 live partials, B5-57) ------------------------------


class ClosableResponse:
    """A response whose read() blocks until it is closed, the way a socket read does."""

    def __init__(self, body: bytes = b'{"text": "hi", "language": "english"}', on_read=None):
        self.body = body
        self.closed = False
        self.reads = 0
        self.on_read = on_read

    def read(self):
        self.reads += 1
        if self.on_read is not None:
            self.on_read()
        if self.closed:
            raise ValueError("read of closed file")
        return self.body

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_tail_pcm_keeps_the_last_seconds_on_a_whole_sample():
    data = bytes(2 * 16000 * 5)
    assert len(asr.tail_pcm(data, 16000, 2.0)) == 2 * 16000 * 2
    assert asr.tail_pcm(data, 16000, 30.0) is data
    assert asr.tail_pcm(data, 16000, 0.0) is data


def test_abort_handle_closes_the_attached_response():
    handle = asr.AbortHandle()
    response = ClosableResponse()
    assert handle.attach(response) is True
    handle.abort()
    assert response.closed is True
    assert handle.aborted is True


def test_abort_before_attach_refuses_the_response():
    handle = asr.AbortHandle()
    handle.abort()
    assert handle.attach(ClosableResponse()) is False


def test_a_detached_handle_closes_nothing():
    handle = asr.AbortHandle()
    response = ClosableResponse()
    handle.attach(response)
    handle.detach()
    handle.abort()
    assert response.closed is False


def test_an_aborted_handle_never_sends_the_request(monkeypatch):
    sent = []
    monkeypatch.setattr(asr, "urlopen", lambda req, timeout=None: sent.append(req))
    handle = asr.AbortHandle()
    handle.abort()
    client = WhisperClient("http://127.0.0.1:9")
    with pytest.raises(asr.AbortedRequest):
        client.inference(b"RIFFfake", language="en", prompt="", abort=handle)
    assert sent == []


def test_closing_the_response_mid_read_raises_aborted_request(monkeypatch):
    handle = asr.AbortHandle()
    response = ClosableResponse(on_read=handle.abort)
    monkeypatch.setattr(asr, "urlopen", lambda req, timeout=None: response)
    client = WhisperClient("http://127.0.0.1:9")
    with pytest.raises(asr.AbortedRequest):
        client.inference(b"RIFFfake", language="en", prompt="", abort=handle)
    assert response.closed is True


def test_an_aborted_request_is_still_a_whisper_error():
    assert issubclass(asr.AbortedRequest, WhisperError)


def test_a_request_without_a_handle_detaches_nothing(monkeypatch):
    response = ClosableResponse()
    monkeypatch.setattr(asr, "urlopen", lambda req, timeout=None: response)
    client = WhisperClient("http://127.0.0.1:9")
    assert client.inference(b"RIFFfake", language="en", prompt="")["text"] == "hi"
    assert response.closed is False


class RecordingClient:
    def __init__(self):
        self.kwargs = []

    def inference(self, wav, **kwargs):
        self.kwargs.append(kwargs)
        return {"text": "hi", "language": "english"}

    def transcribe(self, wav, **kwargs):
        self.kwargs.append(kwargs)
        return asr.AsrReply("hi", "en", "English")

    def health(self):
        return True


def test_abortable_client_binds_the_handle_to_every_request():
    inner = RecordingClient()
    handle = asr.AbortHandle()
    client = asr.AbortableClient(inner, handle)
    client.inference(b"RIFFfake", language="en", prompt="")
    client.transcribe(b"RIFFfake", language="en")
    assert [kwargs["abort"] for kwargs in inner.kwargs] == [handle, handle]
    assert client.health() is True
