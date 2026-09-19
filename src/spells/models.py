"""Shared data types for Spells.

Every module imports its cross-module types from here so that builders working
in parallel agree on names and shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# Language codes are the codes whisper.cpp uses (100 entries, ISO 639-1 where one
# exists, plus "haw", "yue", "jw"). See data/whisper_languages.json.
LanguageCode = str


@dataclass(frozen=True)
class LangMode:
    """How the language is chosen for one dictation (spec 7.2).

    kind "auto": detect; "locked": tray-selected language for every dictation;
    "forced": the per-language chord that started this dictation.
    """

    kind: Literal["auto", "locked", "forced"]
    code: LanguageCode | None = None


class ChordMode(str, Enum):
    """What a chord does with the transcript it produces (spec 6, 8.5).

    "dictate": the transcript is the text, cleaned and delivered.
    "compose": the transcript is an instruction and the model writes the text.
    "edit": the same, applied to the text selected in the target window; with
    nothing selected it falls back to composing.
    """

    DICTATE = "dictate"
    COMPOSE = "compose"
    EDIT = "edit"

    @property
    def writes(self) -> bool:
        """True for the two modes that send the transcript to the writing model."""
        return self is not ChordMode.DICTATE


@dataclass(frozen=True)
class Chord:
    """A hotkey chord. All keys must be down for the chord to fire (spec 6).

    language is None for the main dictation chord and a language code for a
    per-language chord. mode says whether the transcript is text (dictate) or an
    instruction for the writing model (compose, edit).
    """

    keys: tuple[int, ...]
    language: LanguageCode | None = None
    mode: ChordMode = ChordMode.DICTATE


@dataclass(frozen=True)
class TargetContext:
    """Snapshot of the focused window at press time (spec 6 step 1, 5.2).

    Elevation is not captured here: the hotkey cannot fire while an elevated
    window is in the foreground, and inject re-checks the current foreground
    window at delivery time.
    """

    hwnd: int
    process: str
    title: str
    captured_at: float


class DeliveryMethod(str, Enum):
    PASTE = "paste"
    TYPE = "type"


@dataclass(frozen=True)
class Profile:
    """An app profile (spec 9)."""

    name: str
    cleanup: bool
    tone: str
    delivery: DeliveryMethod
    drop_trailing_period_single_sentence: bool = False
    drop_trailing_punctuation: bool = False


@dataclass(frozen=True)
class Transcript:
    """Output of asr.transcribe (spec 7)."""

    text: str
    language: LanguageCode
    duration_s: float
    fallback_used: bool = False
    engine: str = ""


@dataclass(frozen=True)
class CleanResult:
    """Output of cleanup.clean (spec 8).

    reason is "ok" when the LLM output was accepted, otherwise the gate or guard
    reason: "disabled", "profile_off", "language_unscored", "short_clean",
    "clean_text", "engine_not_ready", "cpu_fallback", "timeout", "error", "empty",
    "length_ratio", "finish_length", "preamble", "language_switch".
    """

    text: str
    used_llm: bool
    reason: str


@dataclass(frozen=True)
class ComposeResult:
    """Output of compose.compose (spec 8.5).

    ok is True only when the writing model answered and every guard passed; nothing is
    delivered otherwise, because delivering the raw instruction would be worse than
    delivering nothing. reason is "ok" or the failure: "engine_not_ready", "no_instruction",
    "timeout", "error", "cancelled", "finish_length", "empty", "preamble", "refusal",
    "echo".
    """

    text: str
    ok: bool
    reason: str
    mode: ChordMode = ChordMode.COMPOSE
    instruction: str = ""
    selection_chars: int = 0


class DeliveryOutcome(str, Enum):
    PASTED = "pasted"
    TYPED = "typed"
    COPIED_FOCUS_CHANGED = "copied_focus_changed"
    COPIED_ELEVATED = "copied_elevated"
    FAILED = "failed"


@dataclass(frozen=True)
class DeliveryResult:
    outcome: DeliveryOutcome
    detail: str = ""


@dataclass(frozen=True)
class LastDelivery:
    """The previous delivery, for the leading-space rule (spec 10.4)."""

    hwnd: int
    finished_at: float
    outcome: DeliveryOutcome
    ended_with_whitespace: bool


class Engine(str, Enum):
    WHISPER = "whisper"
    LLAMA = "llama"


@dataclass(frozen=True)
class EngineId:
    role: Engine
    slot: int = 0

    @property
    def key(self) -> str:
        return self.role.value if self.slot == 0 else f"{self.role.value}-{self.slot + 1}"

    def __str__(self) -> str:
        return self.key


@dataclass(frozen=True)
class CpuPlan:
    threads: int | None = None
    affinity_mask: int | None = None

    @property
    def cpu_mask_hex(self) -> str | None:
        return None if self.affinity_mask is None else format(self.affinity_mask, "x")


class EngineState(str, Enum):
    """Per-engine supervisor state (spec 13, batch 2 decision V3-SM)."""

    STARTING = "starting"
    READY = "ready"
    RESTARTING = "restarting"
    PAUSED = "paused"
    CPU_FALLBACK = "cpu_fallback"
    UNLOADED = "unloaded"
    FAILED = "failed"


@dataclass(frozen=True)
class StageTimings:
    """Milliseconds per pipeline stage for one dictation (spec 6 step 9, 12)."""

    press_to_pill_ms: float | None = None
    mic_open_ms: float | None = None
    release_to_transcript_ms: float | None = None
    release_to_cleaned_ms: float | None = None
    delivery_ms: float | None = None
    extra: dict[str, float] = field(default_factory=dict)
