"""Cleanup gate, llama-server request, and output guards (spec 8; decisions V2-1, V2-2, V2-6,
V2-7, V2-9, V2-15).

Stdlib only so the benchmark (spec 18) and the app share it. The filler and correction lists come
in as dicts keyed by language code (config owns them; default_fillers() and default_corrections()
load the shipped defaults). Preambles and stopwords are read from data/ here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, lru_cache
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from spells.cleanup_prompt import system_prompt, user_message
from spells.datafiles import data_dir, read_lines
from spells.models import CleanResult, Profile, Transcript
from spells.textutil import contains_phrase, normalize_for_match, word_count

MIN_WORDS_FOR_CLEANUP = 12
LENGTH_FLOOR_PERCENT = 40
LENGTH_CEILING_PERCENT = 130
MIN_STOPWORD_HITS = 3
DOMINANCE_RATIO = 2
GATE_REASON_TEXT = {
    "disabled": "Cleanup is turned off.",
    "profile_off": "Cleanup is off for this app.",
    "language_unscored": "The cleanup model has not been measured for this language.",
    "short_clean": "Short and already clean.",
    "clean_text": "No filler words or corrections to clean up.",
    "engine_not_ready": "The cleanup engine was not ready.",
    "cpu_fallback": "The cleanup engine fell back to the processor.",
    "compose": "This was an instruction, so the writing model answered it.",
}


class CleanupError(Exception):
    """The cleanup request failed. reason is "timeout" or "error" (spec 8.3, CleanResult.reason)."""

    def __init__(self, message: str, reason: str = "error") -> None:
        super().__init__(message)
        self.reason = reason


class LlamaClient:
    """HTTP client for llama-server's OpenAI-compatible endpoint. Override chat() in tests."""

    def __init__(self, base_url: str, timeout_s: float = 2.5) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def chat(self, system: str, user: str, max_tokens: int) -> tuple[str, str]:
        """POST one chat completion and return (content, finish_reason). Raises CleanupError."""
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": max_tokens,
            "stream": False,
        }
        request = Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except HTTPError as exc:
            raise CleanupError(f"llama-server returned HTTP {exc.code}") from exc
        except TimeoutError as exc:
            raise CleanupError("llama-server request timed out", reason="timeout") from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise CleanupError("llama-server request timed out", reason="timeout") from exc
            raise CleanupError(f"llama-server request failed: {exc.reason}") from exc
        except OSError as exc:
            raise CleanupError(f"llama-server request failed: {exc}") from exc
        try:
            body = json.loads(raw.decode("utf-8"))
            choice = body["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (UnicodeDecodeError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise CleanupError("llama-server returned an unexpected body") from exc
        return ("" if content is None else str(content), str(finish_reason or ""))

    def health(self) -> bool:
        """True when GET /health answers 200 (model loaded)."""
        try:
            with urlopen(Request(self.base_url + "/health"), timeout=self.timeout_s):
                return True
        except OSError:
            return False


@dataclass(frozen=True)
class GateInput:
    """Everything the gate needs besides the text (spec 8.1 with V2-9).

    cpu_fallback is runtime state from engines, never a stored setting: while llama runs on its
    CPU build, cleanup is skipped and the on/off setting stays untouched.
    """

    cleanup_enabled: bool
    profile_cleanup: bool
    engine_available: bool
    cpu_fallback: bool
    cpu_selected: bool = False
    cleanup_languages: frozenset[str] | None = None


def _phrases(
    language: str, fillers: dict[str, list[str]], corrections: dict[str, list[str]]
) -> list[str]:
    return list(fillers.get(language, ())) + list(corrections.get(language, ()))


def should_clean(
    text: str,
    language: str,
    fillers: dict[str, list[str]],
    corrections: dict[str, list[str]],
    gate: GateInput,
) -> tuple[bool, str]:
    """Spec 8.1 gate. Returns (run, reason); reason is "ok" when cleanup should run.

    Configuration reasons come first, then the text rule, then engine state, so a transcript that
    would never be cleaned is not counted against the engine in Diagnostics.
    """
    if not gate.cleanup_enabled:
        return False, "disabled"
    if not gate.profile_cleanup:
        return False, "profile_off"
    if gate.cleanup_languages is not None and language not in gate.cleanup_languages:
        return False, "language_unscored"
    if gate.cpu_selected:
        if not _has_marker(text, language, fillers, corrections):
            return False, "clean_text"
    elif word_count(text) < MIN_WORDS_FOR_CLEANUP and not _has_marker(
        text, language, fillers, corrections
    ):
        return False, "short_clean"
    if not gate.engine_available:
        return False, "engine_not_ready"
    if gate.cpu_fallback:
        return False, "cpu_fallback"
    return True, "ok"


def _has_marker(
    text: str, language: str, fillers: dict[str, list[str]], corrections: dict[str, list[str]]
) -> bool:
    return any(
        contains_phrase(text, phrase) for phrase in _phrases(language, fillers, corrections)
    )


def max_tokens_for(text: str) -> int:
    """Word-based completion bound, no tokenizer bundled (spec 8.2 with V2-1)."""
    return 3 * word_count(text) + 32


def _matched_words(text: str, phrases: list[str]) -> int:
    """Number of words of text covered by non-overlapping whole-phrase matches, longest first."""
    tokens = normalize_for_match(text).split()
    needles = [normalize_for_match(phrase).split() for phrase in phrases]
    needles = sorted((n for n in needles if n), key=len, reverse=True)
    covered = [False] * len(tokens)
    for needle in needles:
        span = len(needle)
        i = 0
        while i + span <= len(tokens):
            if not any(covered[i : i + span]) and tokens[i : i + span] == needle:
                covered[i : i + span] = [True] * span
                i += span
            else:
                i += 1
    return sum(covered)


@lru_cache(maxsize=1)
def _preamble_needles() -> tuple[tuple[str, ...], ...]:
    needles = (tuple(normalize_for_match(line).split()) for line in preambles())
    return tuple(needle for needle in needles if needle)


def preambles() -> list[str]:
    """The combined preamble list of data/preambles.txt (decision V2-7), as written."""
    return read_lines("preambles.txt")


def starts_with_preamble(output: str) -> bool:
    """Whether the output opens with an entry of data/preambles.txt (spec 8.3, V2-7).

    Shared with the writing guards of spec 8.5, which reject the same assistant openings.
    """
    tokens = normalize_for_match(output).split()
    return any(tokens[: len(needle)] == list(needle) for needle in _preamble_needles())




@cache
def _stopword_entries(code: str) -> tuple[str, ...]:
    path = data_dir() / "stopwords" / f"{code}.txt"
    if not path.is_file():
        return ()
    return tuple(read_lines(f"stopwords/{code}.txt"))


def stopwords(code: str) -> list[str]:
    """The shipped stopword list for a language code, or [] when none ships (V2-6)."""
    return list(_stopword_entries(code))


@cache
def _stopword_matcher(code: str) -> tuple[frozenset[str], tuple[str, ...]] | None:
    """(exact words, clitic prefixes) for a language, or None when it has no list."""
    entries = _stopword_entries(code)
    if not entries:
        return None
    exact = set()
    prefixes = []
    for entry in entries:
        if entry.endswith(("'", "’")):
            prefixes.append(normalize_for_match(entry[:-1]) + "'")
        else:
            exact.add(normalize_for_match(entry))
    exact.discard("")
    return frozenset(exact), tuple(prefix for prefix in prefixes if prefix != "'")


def _stopword_hits(tokens: list[str], matcher: tuple[frozenset[str], tuple[str, ...]]) -> int:
    exact, prefixes = matcher
    return sum(1 for token in tokens if token in exact or token.startswith(prefixes))


def dominant_language(text: str, enabled_languages: list[str]) -> str | None:
    """Stopword-dominant language of text among the enabled languages that ship a list (8.3, V2-6).

    Clear when the top language has at least 3 hits and at least twice the runner-up's hits;
    otherwise None.
    """
    tokens = normalize_for_match(text).split()
    scores = []
    for code in dict.fromkeys(enabled_languages):
        matcher = _stopword_matcher(code)
        if matcher is not None:
            scores.append((_stopword_hits(tokens, matcher), code))
    if not scores:
        return None
    scores.sort(key=lambda score: score[0], reverse=True)
    top_hits, top_code = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0
    if top_hits >= MIN_STOPWORD_HITS and top_hits >= DOMINANCE_RATIO * runner_up:
        return top_code
    return None


def guard(
    transcript_text: str,
    output: str,
    finish_reason: str,
    language: str,
    enabled_languages: list[str],
    fillers: dict[str, list[str]],
    corrections: dict[str, list[str]],
) -> str | None:
    """Spec 8.3 output guards. Returns the rejection reason, or None when the output is accepted.

    Length: floor 40% of the transcript's non-filler word count (words minus filler and
    correction matches, V2-2), ceiling 130% of the full word count. Preamble: combined list,
    whole words, case-insensitive (V2-7). Language: reject only on a clear switch (V2-6).
    """
    cleaned = output.strip()
    if finish_reason == "length":
        return "finish_length"
    if not cleaned:
        return "empty"
    total = word_count(transcript_text)
    matched = _matched_words(transcript_text, _phrases(language, fillers, corrections))
    non_filler = max(0, total - matched)
    out_count = word_count(cleaned)
    if out_count * 100 < LENGTH_FLOOR_PERCENT * non_filler:
        return "length_ratio"
    if out_count * 100 > LENGTH_CEILING_PERCENT * total:
        return "length_ratio"
    if starts_with_preamble(cleaned):
        return "preamble"
    before = dominant_language(transcript_text, enabled_languages)
    after = dominant_language(cleaned, enabled_languages)
    if before is not None and after is not None and before != after:
        return "language_switch"
    return None


def clean(
    transcript: Transcript,
    profile: Profile,
    terms: list[str],
    fillers: dict[str, list[str]],
    corrections: dict[str, list[str]],
    enabled_languages: list[str],
    gate: GateInput,
    client: LlamaClient,
) -> CleanResult:
    """Gate, request, guard. Skipped, failed, and rejected results carry the raw text (spec 8)."""
    text = transcript.text
    run, reason = should_clean(text, transcript.language, fillers, corrections, gate)
    if not run:
        return CleanResult(text=text, used_llm=False, reason=reason)
    try:
        output, finish_reason = client.chat(
            system_prompt(), user_message(text, profile.tone, terms), max_tokens_for(text)
        )
    except CleanupError as exc:
        return CleanResult(text=text, used_llm=False, reason=exc.reason)
    rejection = guard(
        text, output, finish_reason, transcript.language, enabled_languages, fillers, corrections
    )
    if rejection is not None:
        return CleanResult(text=text, used_llm=False, reason=rejection)
    return CleanResult(text=output.strip(), used_llm=True, reason="ok")


def warmup_messages() -> tuple[str, str]:
    """System prompt and an empty-transcript user message for the llama warm-up (V2-13)."""
    return system_prompt(), user_message("", "", [])


def _language_lists(kind: str) -> dict[str, list[str]]:
    folder = data_dir() / kind
    return {path.stem: read_lines(f"{kind}/{path.name}") for path in sorted(folder.glob("*.txt"))}


def default_fillers() -> dict[str, list[str]]:
    """Shipped filler lists by language code (data/fillers/<code>.txt, spec 8.1 table)."""
    return _language_lists("fillers")


def default_corrections() -> dict[str, list[str]]:
    """Shipped self-correction lists by language code (data/corrections/<code>.txt)."""
    return _language_lists("corrections")
