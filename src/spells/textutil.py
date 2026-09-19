"""Text helpers shared by asr, cleanup, and the benchmark.

Word counting follows batch 2 decision V2-15: whitespace-separated tokens after dropping tokens
that contain no letter or digit. The same count feeds the gate (8.1), max_tokens (8.2), and the
length guard (8.3).
"""

from __future__ import annotations

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})


def words(text: str) -> list[str]:
    """Whitespace-split tokens that contain at least one letter or digit (decision V2-15)."""
    return [token for token in text.split() if any(ch.isalnum() for ch in token)]


def word_count(text: str) -> int:
    return len(words(text))


def normalize_for_match(text: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace.

    Letters with diacritics survive (str.isalnum is Unicode aware). Apostrophes survive inside a
    word ("don't", Albanian "s'ka") so that contractions and clitics stay one token; leading and
    trailing apostrophes are dropped. Both sides of every comparison go through this function.
    """
    lowered = text.lower().translate(_APOSTROPHES)
    spaced = "".join(ch if ch.isalnum() or ch == "'" else " " for ch in lowered)
    tokens = (token.strip("'") for token in spaced.split())
    return " ".join(token for token in tokens if token)


def contains_phrase(text: str, phrase: str) -> bool:
    """True when phrase occurs in text as whole words, case-insensitively.

    A one-word phrase matches a whole word ("like" but not "likes"); a multi-word phrase matches
    the same words in sequence, whatever punctuation sat between them in the original text.
    """
    needle = normalize_for_match(phrase).split()
    if not needle:
        return False
    haystack = normalize_for_match(text).split()
    span = len(needle)
    return any(haystack[i : i + span] == needle for i in range(len(haystack) - span + 1))
