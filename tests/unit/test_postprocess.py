"""Tests for spells.postprocess: replacements, snippets, profile formatting, leading space."""

from __future__ import annotations

import pytest

from spells import postprocess
from spells.config import Replacement, Snippet, TermEntry, Vocabulary
from spells.models import DeliveryMethod, DeliveryOutcome, LastDelivery, Profile, TargetContext

CHAT = Profile(
    name="Chat",
    cleanup=True,
    tone="casual",
    delivery=DeliveryMethod.PASTE,
    drop_trailing_period_single_sentence=True,
)
TERMINAL = Profile(
    name="Terminal",
    cleanup=False,
    tone="",
    delivery=DeliveryMethod.TYPE,
    drop_trailing_punctuation=True,
)
DEFAULT = Profile(name="Default", cleanup=True, tone="standard", delivery=DeliveryMethod.PASTE)

CTX = TargetContext(hwnd=1, process="notepad.exe", title="Untitled", captured_at=100.0)
EMPTY = Vocabulary(terms=[], replacements=[], snippets=[])


def _vocab(
    replacements: list[tuple[str, str]] = (),
    snippets: list[tuple[str, str]] = (),
) -> Vocabulary:
    return Vocabulary(
        terms=[TermEntry(text="Contoso", edited_at=0.0)],
        replacements=[Replacement(find=f, replace=r) for f, r in replacements],
        snippets=[Snippet(trigger=t, text=x) for t, x in snippets],
    )


def _last(
    hwnd: int = 1,
    finished_at: float = 100.0,
    outcome: DeliveryOutcome = DeliveryOutcome.PASTED,
    ended_with_whitespace: bool = False,
) -> LastDelivery:
    return LastDelivery(
        hwnd=hwnd,
        finished_at=finished_at,
        outcome=outcome,
        ended_with_whitespace=ended_with_whitespace,
    )


def _apply(
    text: str,
    profile: Profile = DEFAULT,
    vocabulary: Vocabulary = EMPTY,
    last: LastDelivery | None = None,
    now: float = 1000.0,
) -> str:
    return postprocess.apply(text, profile, CTX, last, vocabulary, now)


# --- nothing to do ---


def test_no_rules_leaves_text_untouched():
    assert _apply("Hello, world.") == "Hello, world."


def test_empty_text_stays_empty():
    assert _apply("", CHAT, _vocab([("a", "b")], [("a", "b")]), _last()) == ""


# --- replacements ---


def test_replacement_is_case_insensitive_whole_phrase():
    vocab = _vocab([("con toso", "Contoso")])

    assert _apply("I use Con Toso daily.", vocabulary=vocab) == "I use Contoso daily."
    assert _apply("CON TOSO rocks", vocabulary=vocab) == "Contoso rocks"


def test_replacement_does_not_touch_partial_words():
    vocab = _vocab([("con toso", "Contoso"), ("flex", "FLEX")])

    assert _apply("inflexible network flex", vocabulary=vocab) == "inflexible network FLEX"


def test_replacement_replaces_every_occurrence():
    vocab = _vocab([("teh", "the")])

    assert _apply("teh cat and teh dog", vocabulary=vocab) == "the cat and the dog"


def test_replacement_phrase_tolerates_extra_whitespace():
    vocab = _vocab([("con toso", "Contoso")])

    assert _apply("con  toso and con\ntoso", vocabulary=vocab) == "Contoso and Contoso"


def test_replacements_apply_in_list_order():
    vocab = _vocab([("alpha", "beta"), ("beta", "gamma")])

    assert _apply("alpha", vocabulary=vocab) == "gamma"


def test_replacement_handles_regex_special_characters():
    vocab = _vocab([("c++", "C++"), ("e.g.", "for example")])

    assert _apply("i like c++.", vocabulary=vocab) == "i like C++."
    assert _apply("c++x is not c++", vocabulary=vocab) == "c++x is not C++"
    assert _apply("use e.g. this, not eXgX", vocabulary=vocab) == "use for example this, not eXgX"


def test_replacement_text_is_literal():
    vocab = _vocab([("path", r"C:\1\g<0>")])

    assert _apply("the path", vocabulary=vocab) == r"the C:\1\g<0>"


def test_replacement_with_blank_find_is_ignored():
    vocab = _vocab([("", "x"), ("   ", "y")])

    assert _apply("keep me", vocabulary=vocab) == "keep me"


def test_replacement_handles_unicode_case():
    vocab = _vocab([("äpfel", "Äpfel")])

    assert _apply("ÄPFEL und äpfel", vocabulary=vocab) == "Äpfel und Äpfel"


def test_apply_replacements_alone():
    replacements = [Replacement(find="con toso", replace="Contoso")]

    assert postprocess.apply_replacements("con toso.", replacements) == "Contoso."


# --- snippets ---


def test_snippet_replaces_whole_utterance():
    vocab = _vocab(snippets=[("sign off", "Best regards,\nAlex")])

    assert _apply("Sign off!", vocabulary=vocab) == "Best regards,\nAlex"
    assert _apply("sign off", vocabulary=vocab) == "Best regards,\nAlex"


def test_snippet_ignores_punctuation_case_and_spacing():
    vocab = _vocab(snippets=[("E-mail  Sig.", "sig")])

    assert _apply("  e-mail, sig!  ", vocabulary=vocab) == "sig"
    assert _apply("Email sig", vocabulary=vocab) == "sig"
    assert _apply("e mail sig", vocabulary=vocab) == "e mail sig"


def test_snippet_not_expanded_inside_longer_utterance():
    vocab = _vocab(snippets=[("sign off", "Best regards")])

    assert _apply("please sign off now", vocabulary=vocab) == "please sign off now"


def test_first_matching_snippet_wins():
    vocab = _vocab(snippets=[("hi", "first"), ("hi", "second")])

    assert _apply("Hi.", vocabulary=vocab) == "first"


def test_snippet_with_blank_trigger_is_ignored():
    vocab = _vocab(snippets=[("", "nothing"), ("...", "dots")])

    assert _apply("keep", vocabulary=vocab) == "keep"
    assert _apply("", vocabulary=vocab) == ""


def test_apply_snippets_alone():
    snippets = [Snippet(trigger="sign off", text="Best regards")]

    assert postprocess.apply_snippets("Sign off.", snippets) == "Best regards"
    assert postprocess.apply_snippets("no", snippets) == "no"


# --- order: replacements run before snippets ---


def test_replacement_can_create_a_snippet_trigger():
    vocab = _vocab([("signature", "sign off")], [("sign off", "Best regards,\nAlex")])

    assert _apply("Signature.", vocabulary=vocab) == "Best regards,\nAlex"


def test_snippet_output_is_not_fed_back_into_replacements():
    vocab = _vocab([("regards", "REGARDS")], [("sig", "Best regards")])

    assert _apply("sig", vocabulary=vocab) == "Best regards"


# --- profile formatting ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sounds good.", "Sounds good"),
        ("It costs 3.50.", "It costs 3.50"),
        ("See example.com.", "See example.com"),
        ("Sounds good. ", "Sounds good "),
        ("Sounds good.\n", "Sounds good\n"),
        ("Sounds good", "Sounds good"),
        ("Really?", "Really?"),
        ("Great!", "Great!"),
        ("Wait...", "Wait..."),
        ("Sounds good. See you tomorrow.", "Sounds good. See you tomorrow."),
        ("Really? Yes.", "Really? Yes."),
        ("Wow! Great.", "Wow! Great."),
        ("Send it to Dr. Smith.", "Send it to Dr. Smith."),
        ("", ""),
        (".", ""),
    ],
)
def test_chat_drops_trailing_period_on_single_sentence(text, expected):
    assert _apply(text, CHAT) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ls -la.", "ls -la"),
        ("git status?!", "git status"),
        ("git status...", "git status"),
        ("echo hi;", "echo hi"),
        ("cd src,", "cd src"),
        ("ls -la. ", "ls -la"),
        ("ls -la", "ls -la"),
        ('echo "hi."', 'echo "hi."'),
        ("ls -la. cat x.", "ls -la. cat x"),
        ("", ""),
    ],
)
def test_terminal_drops_trailing_punctuation(text, expected):
    assert _apply(text, TERMINAL) == expected


def test_default_profile_keeps_punctuation():
    assert _apply("Sounds good.", DEFAULT) == "Sounds good."
    assert _apply("ls -la.", DEFAULT) == "ls -la."


def test_email_profile_keeps_punctuation():
    email = Profile(name="Email and docs", cleanup=True, tone="", delivery=DeliveryMethod.PASTE)
    assert _apply("Sounds good.", email) == "Sounds good."


def test_apply_profile_formatting_alone():
    assert postprocess.apply_profile_formatting("Hi.", CHAT) == "Hi"
    assert postprocess.apply_profile_formatting("ls.", TERMINAL) == "ls"
    assert postprocess.apply_profile_formatting("Hi.", DEFAULT) == "Hi."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sounds good.", True),
        ("It costs 3.50.", True),
        ("Sounds good", True),
        ("Sounds good! Really", False),
        ("One. Two.", False),
        ("One.\nTwo.", False),
        ("", True),
    ],
)
def test_is_single_sentence(text, expected):
    assert postprocess.is_single_sentence(text) is expected


def test_profile_formatting_applies_to_snippet_output():
    vocab = _vocab(snippets=[("thanks", "Thanks a lot.")])

    assert _apply("thanks", CHAT, vocab) == "Thanks a lot"


# --- leading space ---


def test_leading_space_added_when_all_conditions_hold():
    assert _apply("hello", last=_last(), now=150.0) == " hello"


def test_leading_space_added_after_typed_delivery():
    assert _apply("hello", last=_last(outcome=DeliveryOutcome.TYPED), now=150.0) == " hello"


def test_no_leading_space_without_last_delivery():
    assert _apply("hello", last=None, now=150.0) == "hello"


def test_no_leading_space_for_a_different_window():
    assert _apply("hello", last=_last(hwnd=2), now=150.0) == "hello"


def test_leading_space_window_is_sixty_seconds():
    assert _apply("hello", last=_last(finished_at=100.0), now=159.9) == " hello"
    assert _apply("hello", last=_last(finished_at=100.0), now=160.0) == "hello"
    assert _apply("hello", last=_last(finished_at=100.0), now=500.0) == "hello"


@pytest.mark.parametrize(
    "outcome",
    [
        DeliveryOutcome.COPIED_FOCUS_CHANGED,
        DeliveryOutcome.COPIED_ELEVATED,
        DeliveryOutcome.FAILED,
    ],
)
def test_no_leading_space_after_copy_or_failure(outcome):
    assert _apply("hello", last=_last(outcome=outcome), now=150.0) == "hello"


def test_no_leading_space_when_previous_text_ended_with_whitespace():
    assert _apply("hello", last=_last(ended_with_whitespace=True), now=150.0) == "hello"


def test_no_leading_space_when_text_already_starts_with_whitespace():
    assert _apply(" hello", last=_last(), now=150.0) == " hello"
    assert _apply("\nhello", last=_last(), now=150.0) == "\nhello"


def test_no_leading_space_for_empty_text():
    assert _apply("", last=_last(), now=150.0) == ""


def test_needs_leading_space_alone():
    assert postprocess.needs_leading_space("x", CTX, _last(), 150.0) is True
    assert postprocess.needs_leading_space("x", CTX, None, 150.0) is False


# --- everything together, in spec order ---


def test_full_pipeline_order():
    vocab = _vocab([("signature", "sign off")], [("sign off", "Thanks a lot.")])

    result = _apply("Signature!", CHAT, vocab, _last(), now=150.0)

    assert result == " Thanks a lot"


def test_leading_space_is_added_after_terminal_formatting():
    assert _apply("ls -la.", TERMINAL, last=_last(), now=150.0) == " ls -la"
