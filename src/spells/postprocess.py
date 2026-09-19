"""Post-processing of the cleaned (or raw) transcript before delivery (spec 10).

Steps, in this order:

1. Replacements: case-insensitive whole-phrase find and replace.
2. Snippets: when the whole utterance, lowercased and stripped of punctuation,
   equals a trigger, the output becomes the snippet text.
3. Profile formatting: Chat drops the trailing period on single-sentence text;
   Terminal drops trailing punctuation.
4. Leading space: one space when the previous delivery went to the same window
   less than 60 s ago, was pasted or typed, and did not end in whitespace.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from spells.config import Replacement, Snippet, Vocabulary
from spells.models import DeliveryOutcome, LastDelivery, Profile, TargetContext

LEADING_SPACE_WINDOW_S = 60.0

# Sentence-ending punctuation that Terminal strips from the end of the text.
# Quotes and brackets are left alone: they are usually part of the command.
TRAILING_PUNCTUATION = ".!?,;:…"

_DELIVERED = frozenset({DeliveryOutcome.PASTED, DeliveryOutcome.TYPED})

# A sentence terminator run followed by whitespace and more text: the text goes
# on after a sentence ended. "3.50" and "example.com" do not count because their
# period is followed directly by a non-space character.
_INNER_SENTENCE_END = re.compile(r"[.!?]+\s+\S")


def apply(
    text: str,
    profile: Profile,
    ctx: TargetContext,
    last_delivery: LastDelivery | None,
    vocabulary: Vocabulary,
    now: float,
) -> str:
    """Run every spec 10 step in order and return the text to deliver."""
    text = apply_replacements(text, vocabulary.replacements)
    text = apply_snippets(text, vocabulary.snippets)
    text = apply_profile_formatting(text, profile)
    if needs_leading_space(text, ctx, last_delivery, now):
        text = " " + text
    return text


# --- 1. replacements ---


def _phrase_pattern(find: str) -> re.Pattern[str] | None:
    """A case-insensitive pattern matching find as a whole phrase (no letter or digit
    touching either end). Whitespace inside the phrase matches any whitespace run."""
    words = find.split()
    if not words:
        return None
    body = r"\s+".join(re.escape(word) for word in words)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _literal(replacement: str) -> Callable[[re.Match[str]], str]:
    """A substitution callback that inserts replacement verbatim (no backslash escapes)."""
    return lambda _match: replacement


def apply_replacements(text: str, replacements: list[Replacement]) -> str:
    for replacement in replacements:
        pattern = _phrase_pattern(replacement.find)
        if pattern is None:
            continue
        text = pattern.sub(_literal(replacement.replace), text)
    return text


# --- 2. snippets ---


def normalize_utterance(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace (the spec 10 trigger comparison)."""
    kept = "".join(
        char for char in text.casefold() if not unicodedata.category(char).startswith("P")
    )
    return " ".join(kept.split())


def apply_snippets(text: str, snippets: list[Snippet]) -> str:
    utterance = normalize_utterance(text)
    if not utterance:
        return text
    for snippet in snippets:
        if normalize_utterance(snippet.trigger) == utterance:
            return snippet.text
    return text


# --- 3. profile formatting ---


def is_single_sentence(text: str) -> bool:
    """True when no sentence terminator is followed by more text."""
    return _INNER_SENTENCE_END.search(text) is None


def _split_trailing_whitespace(text: str) -> tuple[str, str]:
    body = text.rstrip()
    return body, text[len(body) :]


def _drop_trailing_period(text: str) -> str:
    body, tail = _split_trailing_whitespace(text)
    if body.endswith(".") and not body.endswith("..") and is_single_sentence(body):
        return body[:-1] + tail
    return text


def _drop_trailing_punctuation(text: str) -> str:
    body, _tail = _split_trailing_whitespace(text)
    stripped = body.rstrip(TRAILING_PUNCTUATION)
    if stripped == body:
        return text
    return stripped.rstrip()


def apply_profile_formatting(text: str, profile: Profile) -> str:
    if profile.drop_trailing_period_single_sentence:
        text = _drop_trailing_period(text)
    if profile.drop_trailing_punctuation:
        text = _drop_trailing_punctuation(text)
    return text


# --- 4. leading space ---


def needs_leading_space(
    text: str, ctx: TargetContext, last_delivery: LastDelivery | None, now: float
) -> bool:
    """The spec 10 step 4 rule. Also false for empty text or text that already
    starts with whitespace, so a snippet beginning with a newline is not padded."""
    if last_delivery is None:
        return False
    if last_delivery.hwnd != ctx.hwnd:
        return False
    if last_delivery.outcome not in _DELIVERED:
        return False
    if now - last_delivery.finished_at >= LEADING_SPACE_WINDOW_S:
        return False
    if last_delivery.ended_with_whitespace:
        return False
    return bool(text) and not text[0].isspace()
