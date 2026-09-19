"""Whisper requests, language policy, hallucination filter (spec 7; decisions V2-5, V2-8, V2-10).

Stdlib only (urllib plus a hand-built multipart body) so the benchmark (spec 18) and the app run
the same code without extra dependencies.

Every entry point takes the PCM with the sample rate it was captured at and sends a WAV
carrying that rate. Nothing here resamples: whisper.cpp reads the uploaded WAV through
miniaudio and llama.cpp mtmd does the same for a Qwen3-ASR request, so the engines convert
to the 16 kHz their front ends want with a better converter than the capture path could
reach. The audio row of spec 5.2 captures at the device's own rate for that reason.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import socket
import threading
import uuid
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request

from spells.datafiles import data_path, read_lines
from spells.loopback import urlopen
from spells.models import LangMode, LanguageCode, Transcript
from spells.quality import AsrMetrics, metrics_from_verbose_json
from spells.textutil import normalize_for_match

log = logging.getLogger(__name__)

MetricsSink = Callable[[AsrMetrics], None]

PROMPT_MAX_CHARS = 500
SHORT_CLIP_S = 2.0
WHISPER_TIMEOUT_BASE_S = 10.0
WHISPER_TIMEOUT_PER_SECOND_S = 2.0
WHISPER_SERVER = "whisper-server"
LLAMA_ASR = "llama-asr"
LLAMA_ASR_MAX_AUDIO_S = 180.0
ASR_TEXT_TAG = "<asr_text>"
NO_LANGUAGE = "none"
PREFIX_SCAN_CHARS = 96
_LANGUAGE_PREFIX = re.compile(r"^\s*language\s+(?P<name>[^<]*?)\s*<asr_text>", re.IGNORECASE)


class WhisperError(Exception):
    """whisper-server was unreachable, timed out, or answered with an error or a bad body."""

    engine: Any = None


SpeechError = WhisperError


class AbortedRequest(WhisperError):
    """The caller closed this request's connection before the answer was read."""


class SpeechEngineUnavailable(Exception):
    def __init__(self, engine: Any) -> None:
        super().__init__(f"speech engine {engine} is not serving")
        self.engine = engine


class AbortHandle:
    """Closes the connection of one in-flight request from another thread (B5-57).

    A partial pass runs on its own thread while the recording controller waits for
    nothing. When the key is released the controller calls abort(): the response object
    is closed under it, the socket goes with it, and the engine sees the client hang up
    rather than the final pass queuing behind a request nobody wants any more. A request
    started after abort() never leaves the process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closer: Callable[[], None] | None = None
        self._socket: Any = None
        self._aborted = False

    @property
    def aborted(self) -> bool:
        with self._lock:
            return self._aborted

    def attach(self, response: Any) -> bool:
        """Hold the response so abort() can end it; False when abort() already ran.

        The socket under the response is what abort() really wants: closing the response
        object takes the buffered reader's lock, which the thread blocked in read() is
        holding, so the caller would wait for the very answer it is trying to throw away
        (measured at about 2 s on a 30 s clip). Shutting the socket down returns at once
        and makes that read fail, which is the point.
        """
        with self._lock:
            if self._aborted:
                return False
            self._closer = getattr(response, "close", None)
            self._socket = _socket_of(response)
            return True

    def detach(self) -> None:
        with self._lock:
            self._closer = None
            self._socket = None

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            closer, self._closer = self._closer, None
            sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
                return
            except OSError:
                log.debug("shutting down an aborted speech socket raised", exc_info=True)
        if closer is None:
            return
        try:
            closer()
        except Exception:
            log.debug("closing an aborted speech request raised", exc_info=True)


def _socket_of(response: Any) -> Any:
    """The socket an http.client response reads from, or None for anything else."""
    raw = getattr(getattr(response, "fp", None), "raw", None)
    return getattr(raw, "_sock", None)


def _abort_guard(abort: AbortHandle | None) -> None:
    if abort is not None and abort.aborted:
        raise AbortedRequest("the request was aborted before it was sent")


def _abort_error(abort: AbortHandle | None, exc: BaseException, label: str) -> WhisperError:
    if abort is not None and abort.aborted:
        return AbortedRequest(f"{label} request aborted")
    return WhisperError(f"{label} request failed: {exc}")


class AbortableClient:
    """A speech client whose requests all carry one AbortHandle.

    The language policy builds its requests through the plain client interface, so the
    handle is bound here instead of threaded through every call site.
    """

    def __init__(self, client: Any, abort: AbortHandle) -> None:
        self.client = client
        self.abort = abort

    def inference(self, wav_bytes: bytes, **kwargs: Any) -> dict:
        return self.client.inference(wav_bytes, abort=self.abort, **kwargs)

    def transcribe(self, wav_bytes: bytes, **kwargs: Any) -> AsrReply:
        return self.client.transcribe(wav_bytes, abort=self.abort, **kwargs)

    def health(self) -> bool:
        return bool(self.client.health())


def tail_pcm(pcm16: bytes, sample_rate: int, seconds: float) -> bytes:
    """The last `seconds` of 16-bit mono PCM, aligned to a whole sample."""
    if seconds <= 0 or sample_rate <= 0:
        return pcm16
    limit = int(seconds * sample_rate) * 2
    if limit <= 0 or len(pcm16) <= limit:
        return pcm16
    return pcm16[len(pcm16) - limit :]


@dataclass
class LanguagePolicyState:
    """Mutable state of the language policy: the last accepted language (spec 7.2, V2-8).

    Initially the first enabled language; the pipeline owns one instance for the app lifetime.
    """

    last_accepted: LanguageCode


class WhisperClient:
    """HTTP client for whisper-server (spec 7.1). Subclass and override inference() in tests."""

    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def inference(
        self,
        wav_bytes: bytes,
        *,
        language: str | None,
        prompt: str,
        detect_only: bool = False,
        want_probabilities: bool = False,
        carry_initial_prompt: bool = True,
        temperature: float = 0.0,
        timeout_s: float | None = None,
        abort: AbortHandle | None = None,
    ) -> dict:
        """POST multipart to /inference and return the parsed verbose_json body.

        language None omits the field (server default), "auto" asks for detection. Every request
        carries the vocabulary prompt across 30 s windows (carry_initial_prompt, decision V2-5).
        timeout_s overrides the constructor timeout for this one request when given; callers that
        omit it (for example the engine supervisor) keep using the constructor timeout.
        """
        fields = [("response_format", "verbose_json"), ("temperature", str(float(temperature)))]
        if language is not None:
            fields.append(("language", language))
        fields += [
            ("prompt", prompt),
            ("detect_language", _flag(detect_only)),
            ("no_language_probabilities", _flag(not want_probabilities)),
            ("carry_initial_prompt", _flag(carry_initial_prompt)),
        ]
        body, content_type = _multipart(fields, ("file", "audio.wav", "audio/wav", wav_bytes))
        request = Request(
            self.base_url + "/inference",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        request_timeout = self.timeout_s if timeout_s is None else timeout_s
        _abort_guard(abort)
        try:
            with urlopen(request, timeout=request_timeout) as response:
                if abort is not None and not abort.attach(response):
                    raise AbortedRequest("the request was aborted before it was read")
                try:
                    raw = response.read()
                finally:
                    if abort is not None:
                        abort.detach()
        except HTTPError as exc:
            raise WhisperError(f"whisper-server returned HTTP {exc.code}") from exc
        except AbortedRequest:
            raise
        except Exception as exc:  # URLError, TimeoutError, resets, a shut-down socket
            raise _abort_error(abort, exc, "whisper-server") from exc
        _abort_guard(abort)
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise WhisperError("whisper-server returned a non-JSON body") from exc
        if not isinstance(result, dict):
            raise WhisperError("whisper-server returned an unexpected JSON shape")
        return result

    def health(self) -> bool:
        """True when GET /health answers 200 (model loaded)."""
        try:
            with urlopen(Request(self.base_url + "/health"), timeout=self.timeout_s):
                return True
        except OSError:
            return False


@dataclass(frozen=True)
class AsrReply:
    text: str
    language: str | None
    language_name: str
    complete: bool = True


class LlamaAsrClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def transcribe(
        self,
        wav_bytes: bytes,
        *,
        vocabulary: str = "",
        language: str | None = None,
        accept: Callable[[str], bool] | None = None,
        timeout_s: float | None = None,
        abort: AbortHandle | None = None,
    ) -> AsrReply:
        audio = base64.b64encode(wav_bytes).decode("ascii")
        content: list[dict] = [
            {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}
        ]
        if vocabulary:
            content.append({"type": "text", "text": f"Vocabulary: {vocabulary}"})
        messages: list[dict] = [
            {"role": "system", "content": ""},
            {"role": "user", "content": content},
        ]
        forced_name = language_title(language) if language else ""
        if forced_name:
            prefill = f"language {forced_name}{ASR_TEXT_TAG}"
            messages.append({"role": "assistant", "content": prefill})
        payload = {"messages": messages, "temperature": 0, "stream": True, "cache_prompt": False}
        request = Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request_timeout = self.timeout_s if timeout_s is None else timeout_s
        _abort_guard(abort)
        try:
            with urlopen(request, timeout=request_timeout) as response:
                if abort is not None and not abort.attach(response):
                    raise AbortedRequest("the request was aborted before it was read")
                try:
                    reply = _read_asr_stream(
                        response, accept if not forced_name else None, forced_name
                    )
                finally:
                    if abort is not None:
                        abort.detach()
                _abort_guard(abort)
                return reply
        except HTTPError as exc:
            raise WhisperError(f"llama-server returned HTTP {exc.code}") from exc
        except AbortedRequest:
            raise
        except WhisperError:
            raise
        except Exception as exc:
            raise _abort_error(abort, exc, "llama-server") from exc

    def health(self) -> bool:
        try:
            with urlopen(Request(self.base_url + "/health"), timeout=self.timeout_s):
                return True
        except OSError:
            return False


def _read_asr_stream(
    response: Any, accept: Callable[[str], bool] | None, forced_name: str
) -> AsrReply:
    content = ""
    decided = accept is None
    streamed = False
    others: list[bytes] = []
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if not line.startswith("data:"):
            others.append(raw)
            continue
        streamed = True
        data = line[5:].strip()
        if data == "[DONE]":
            break
        content += _chunk_text(data)
        if decided:
            continue
        if ASR_TEXT_TAG in content or len(content) > PREFIX_SCAN_CHARS:
            decided = True
            name, _text = parse_asr_output(content)
            if not accept(name or ""):
                return AsrReply("", language_code(name), name or "", complete=False)
    if not streamed:
        content = _message_text(b"".join(others))
    name, text = parse_asr_output(content)
    if name is None and forced_name:
        name = forced_name
    return AsrReply(text, language_code(name), name or "")


def _chunk_text(data: str) -> str:
    try:
        chunk = json.loads(data)
    except ValueError as exc:
        raise WhisperError("llama-server sent a malformed stream chunk") from exc
    if not isinstance(chunk, dict):
        raise WhisperError("llama-server sent an unexpected stream chunk")
    if "error" in chunk:
        raise WhisperError(f"llama-server reported an error: {chunk['error']}")
    try:
        delta = chunk["choices"][0].get("delta") or {}
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    piece = delta.get("content") if isinstance(delta, dict) else None
    return piece if isinstance(piece, str) else ""


def _message_text(body: bytes) -> str:
    try:
        parsed = json.loads(body.decode("utf-8"))
        piece = parsed["choices"][0]["message"]["content"]
    except (UnicodeDecodeError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise WhisperError("llama-server returned an unexpected body") from exc
    return piece if isinstance(piece, str) else ""


def parse_asr_output(content: str) -> tuple[str | None, str]:
    match = _LANGUAGE_PREFIX.match(content)
    if match is None:
        return None, content.strip()
    return match.group("name").strip(), content[match.end() :].strip()


def language_code(name: str | None) -> str | None:
    if not name or name.strip().lower() == NO_LANGUAGE:
        return None
    try:
        return name_to_code(name)
    except KeyError:
        return None


@lru_cache(maxsize=1)
def _code_titles() -> dict[str, str]:
    titles: dict[str, str] = {}
    for name, code in _language_table().items():
        titles.setdefault(code, name.title())
    return titles


def language_title(code: str) -> str:
    return _code_titles().get(code, code)


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _multipart(
    fields: list[tuple[str, str]], file_part: tuple[str, str, str, bytes]
) -> tuple[bytes, str]:
    boundary = "----SpellsFormBoundary" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields:
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += value.encode("utf-8") + b"\r\n"
    name, filename, content_type, data = file_part
    out += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"'
        f"\r\nContent-Type: {content_type}\r\n\r\n"
    ).encode()
    out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def pcm16_to_wav(pcm16: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container at its capture rate (decision V3-17)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm16)
    return buffer.getvalue()


@lru_cache(maxsize=1)
def _language_table() -> dict[str, str]:
    """Full name to code, generated from whisper.cpp's language table (decision V2-10)."""
    return json.loads(data_path("whisper_languages.json").read_text(encoding="utf-8"))


def name_to_code(name: str) -> str:
    """Code ("en") for whisper-server's full language name ("english"); KeyError if unknown."""
    return _language_table()[name.strip().lower()]


def build_prompt(terms: list[str]) -> str:
    """Join vocabulary terms (oldest first, most recent last) and keep the last 500 characters.

    Spec 7.3 with decisions V2-5 and E8: truncation is from the front so the newest terms survive.
    """
    joined = ", ".join(term.strip() for term in terms if term.strip())
    if len(joined) > PROMPT_MAX_CHARS:
        joined = joined[-PROMPT_MAX_CHARS:].lstrip(", ")
    return joined


@lru_cache(maxsize=1)
def _hallucinations() -> frozenset[str]:
    return frozenset(normalize_for_match(line) for line in read_lines("hallucinations.txt"))


def is_hallucination(text: str) -> bool:
    """True when the normalized text exactly matches a known Whisper silence output (spec 7.4)."""
    normalized = normalize_for_match(text)
    return bool(normalized) and normalized in _hallucinations()


def transcribe(
    pcm16: bytes,
    sample_rate: int,
    lang_mode: LangMode,
    terms: list[str],
    enabled_languages: list[str],
    state: LanguagePolicyState,
    client: WhisperClient,
    on_metrics: MetricsSink | None = None,
    timeout_base_s: float = WHISPER_TIMEOUT_BASE_S,
) -> Transcript | None:
    """Run the language policy of spec 7.2 and return the transcript, or None for nothing usable.

    Locked and forced modes send their language. Auto mode sends the last accepted language for
    raw recordings under 2.0 s, otherwise "auto"; a detected language outside the enabled set
    triggers a detect-only request with probabilities, the best enabled language (missing counts
    as 0, all missing means the last accepted language) and a forced re-transcription with
    fallback_used set. Every successful transcription updates state.last_accepted (V2-8); empty
    text and hallucinations return None and leave the state alone.

    Every request this function makes gets a per-request timeout scaled with audio length,
    WHISPER_TIMEOUT_BASE_S + WHISPER_TIMEOUT_PER_SECOND_S * audio_seconds computed from the raw
    PCM length, so a long locked or forced dictation is not cut off by the client's flat default.

    on_metrics, when given, receives the confidence scores of the accepted answer alone
    (spec 8.4). It is a sink rather than a field on Transcript so that every existing caller,
    the benchmark among them, keeps the signature it has.
    """
    wav = pcm16_to_wav(pcm16, sample_rate)
    duration_s = len(pcm16) / (2 * sample_rate)
    prompt = build_prompt(terms)
    fallback_used = False
    timeout_s = timeout_base_s + WHISPER_TIMEOUT_PER_SECOND_S * duration_s

    if lang_mode.kind in ("locked", "forced"):
        if lang_mode.code is None:
            raise ValueError(f"{lang_mode.kind} mode needs a language code")
        language = lang_mode.code
        result = client.inference(
            wav, language=language, prompt=prompt, carry_initial_prompt=True, timeout_s=timeout_s
        )
    elif duration_s < SHORT_CLIP_S:
        language = state.last_accepted
        result = client.inference(
            wav, language=language, prompt=prompt, carry_initial_prompt=True, timeout_s=timeout_s
        )
    else:
        result = client.inference(
            wav, language="auto", prompt=prompt, carry_initial_prompt=True, timeout_s=timeout_s
        )
        detected = _reported_code(result)
        if detected is not None and detected in enabled_languages:
            language = detected
        else:
            detection = client.inference(
                wav,
                language="auto",
                prompt=prompt,
                detect_only=True,
                want_probabilities=True,
                carry_initial_prompt=True,
                timeout_s=timeout_s,
            )
            language = _best_enabled(
                detection.get("language_probabilities"), enabled_languages, state.last_accepted
            )
            result = client.inference(
                wav, language=language, prompt=prompt, carry_initial_prompt=True, timeout_s=timeout_s
            )
            fallback_used = True

    text = str(result.get("text") or "").strip()
    if not text or is_hallucination(text):
        return None
    state.last_accepted = language
    if on_metrics is not None:
        on_metrics(metrics_from_verbose_json(result))
    return Transcript(
        text=text, language=language, duration_s=duration_s, fallback_used=fallback_used
    )


def _reported_code(result: dict) -> str | None:
    name = result.get("language")
    if not isinstance(name, str):
        return None
    try:
        return name_to_code(name)
    except KeyError:
        return None


def _best_enabled(probabilities: object, enabled_languages: list[str], fallback: str) -> str:
    """The enabled code with the highest probability; ties keep the enabled order."""
    if not isinstance(probabilities, dict):
        probabilities = {}
    best: str | None = None
    best_probability = 0.0
    for code in enabled_languages:
        try:
            probability = float(probabilities.get(code) or 0.0)
        except (TypeError, ValueError):
            probability = 0.0
        if probability > best_probability:
            best, best_probability = code, probability
    return best if best is not None else fallback


@dataclass(frozen=True)
class SpeechRoute:
    engine: Any
    runtime: str
    languages: tuple[str, ...] = ()


SpeechClient = Any
Connect = Callable[[SpeechRoute], SpeechClient | None]


@dataclass(frozen=True)
class _Outcome:
    text: str
    language: str
    route: SpeechRoute
    fallback_used: bool = False
    metrics: AsrMetrics | None = None


def serving_languages(
    routes: Sequence[SpeechRoute], languages: Sequence[str]
) -> list[tuple[str, ...]]:
    assigned: list[list[str]] = [[] for _ in routes]
    for code in dict.fromkeys(languages):
        index = _owner(routes, code)
        if index is not None:
            assigned[index].append(code)
    return [tuple(codes) for codes in assigned]


def first_auto_route(routes: Sequence[Any], languages: Sequence[str]) -> int:
    assigned = serving_languages(routes, languages)
    return min(range(len(routes)), key=lambda index: (-len(assigned[index]), index))


def _owner(routes: Sequence[SpeechRoute], code: str) -> int | None:
    if not routes:
        return None
    for index, route in enumerate(routes):
        if code in route.languages:
            return index
    for index, route in enumerate(routes):
        if not route.languages:
            return index
    return 0


class _Caller:
    def __init__(self, wav: bytes, prompt: str, timeout_s: float, connect: Connect) -> None:
        self.wav = wav
        self.prompt = prompt
        self.timeout_s = timeout_s
        self.connect = connect
        self.clients: dict[int, SpeechClient] = {}

    def client(self, route: SpeechRoute) -> SpeechClient:
        key = id(route)
        if key not in self.clients:
            client = self.connect(route)
            if client is None:
                raise SpeechEngineUnavailable(route.engine)
            self.clients[key] = client
        return self.clients[key]

    def whisper(self, route: SpeechRoute, **kwargs: Any) -> dict:
        client = self.client(route)
        try:
            return client.inference(
                self.wav,
                prompt=self.prompt,
                carry_initial_prompt=True,
                timeout_s=self.timeout_s,
                **kwargs,
            )
        except WhisperError as exc:
            exc.engine = route.engine
            raise

    def llama(self, route: SpeechRoute, **kwargs: Any) -> AsrReply:
        client = self.client(route)
        try:
            return client.transcribe(
                self.wav, vocabulary=self.prompt, timeout_s=self.timeout_s, **kwargs
            )
        except WhisperError as exc:
            exc.engine = route.engine
            raise

    def locked(self, route: SpeechRoute, code: str, fallback_used: bool = False) -> _Outcome:
        if route.runtime == LLAMA_ASR:
            text = self.llama(route, language=code).text
            metrics = AsrMetrics(source=LLAMA_ASR)
        else:
            result = self.whisper(route, language=code)
            text = str(result.get("text") or "").strip()
            metrics = metrics_from_verbose_json(result)
        return _Outcome(text, code, route, fallback_used, metrics)


def transcribe_routed(
    pcm16: bytes,
    sample_rate: int,
    lang_mode: LangMode,
    terms: list[str],
    enabled_languages: list[str],
    state: LanguagePolicyState,
    routes: Sequence[SpeechRoute],
    connect: Connect,
    on_metrics: MetricsSink | None = None,
    timeout_base_s: float = WHISPER_TIMEOUT_BASE_S,
) -> Transcript | None:
    """The language policy across several speech engines (spec 7.2).

    on_metrics receives the confidence scores of the engine that produced the accepted
    answer, which is AsrMetrics(source="llama-asr") and nothing else on a llama-asr engine,
    because that runtime reports no scores (spec 8.4).
    """
    if not routes:
        raise ValueError("transcribe_routed needs at least one speech engine")
    wav = pcm16_to_wav(pcm16, sample_rate)
    duration_s = len(pcm16) / (2 * sample_rate)
    timeout_s = timeout_base_s + WHISPER_TIMEOUT_PER_SECOND_S * duration_s
    enabled = list(dict.fromkeys(enabled_languages))
    usable = [
        route
        for route in routes
        if not (route.runtime == LLAMA_ASR and duration_s > LLAMA_ASR_MAX_AUDIO_S)
    ] or list(routes)
    call = _Caller(wav, build_prompt(terms), timeout_s, connect)

    if lang_mode.kind in ("locked", "forced"):
        if lang_mode.code is None:
            raise ValueError(f"{lang_mode.kind} mode needs a language code")
        outcome = call.locked(_route_for(usable, lang_mode.code), lang_mode.code)
    elif duration_s < SHORT_CLIP_S:
        outcome = call.locked(_route_for(usable, state.last_accepted), state.last_accepted)
    else:
        outcome = _auto(call, usable, enabled, state.last_accepted)

    text = outcome.text.strip()
    if not text or is_hallucination(text):
        return None
    state.last_accepted = outcome.language
    if on_metrics is not None:
        on_metrics(outcome.metrics or AsrMetrics(source=outcome.route.runtime))
    return Transcript(
        text=text,
        language=outcome.language,
        duration_s=duration_s,
        fallback_used=outcome.fallback_used,
        engine=str(outcome.route.engine),
    )


def _route_for(routes: Sequence[SpeechRoute], code: str) -> SpeechRoute:
    index = _owner(routes, code)
    return routes[index if index is not None else 0]


def _auto(
    call: _Caller, routes: Sequence[SpeechRoute], enabled: list[str], last_accepted: str
) -> _Outcome:
    assigned = serving_languages(routes, enabled)
    first_index = first_auto_route(routes, enabled)
    first = routes[first_index]
    mine = assigned[first_index]
    others = [route for index, route in enumerate(routes) if index != first_index]
    elsewhere = [
        code for index, codes in enumerate(assigned) if index != first_index for code in codes
    ]
    if first.runtime == LLAMA_ASR:
        return _auto_llama(call, first, mine, others, elsewhere, enabled, last_accepted)
    return _auto_whisper(call, first, mine, routes, enabled, last_accepted)


def _auto_llama(call, first, mine, others, elsewhere, enabled, last_accepted) -> _Outcome:
    def accept(name: str) -> bool:
        code = language_code(name)
        if code is None and (not name or name.strip().lower() == NO_LANGUAGE):
            return True
        if code in mine:
            return True
        if code in elsewhere:
            return False
        return not elsewhere

    reply = call.llama(first, accept=accept)
    if reply.complete:
        metrics = AsrMetrics(source=LLAMA_ASR)
        if reply.language in mine:
            return _Outcome(reply.text, reply.language, first, metrics=metrics)
        guess = last_accepted if last_accepted in mine or not mine else mine[0]
        return _Outcome(
            reply.text, guess, first, reply.language is not None, metrics
        )
    if reply.language in elsewhere:
        return call.locked(_route_for(others, reply.language), reply.language)
    if len(elsewhere) == 1:
        return call.locked(_route_for(others, elsewhere[0]), elsewhere[0])
    return _auto(call, others, elsewhere, last_accepted)


def _auto_whisper(call, first, mine, routes, enabled, last_accepted) -> _Outcome:
    result = call.whisper(first, language="auto")
    detected = _reported_code(result)
    if detected is not None and detected in mine:
        text = str(result.get("text") or "").strip()
        return _Outcome(text, detected, first, metrics=metrics_from_verbose_json(result))
    if detected is not None and detected in enabled:
        return call.locked(_route_for(routes, detected), detected)
    detection = call.whisper(first, language="auto", detect_only=True, want_probabilities=True)
    best = _best_enabled(detection.get("language_probabilities"), enabled, last_accepted)
    return call.locked(_route_for(routes, best), best, fallback_used=True)
