"""Settings schema, serialization, migration, atomic persistence, and change notification.

Spec: sections 8.1 (filler and correction lists), 9 (profile tones), 14.4 (every
setting the UI exposes), 15 (file location, corrupt-file recovery, atomic writes)
plus batch 2 decisions V2-9, V3-F2, and V3-F8.

The settings file is %APPDATA%\\Spells\\settings.json. It carries a
``schema_version``; ``migrate`` moves older files forward before ``from_dict``
validates them. A file that cannot be parsed or fails validation is renamed to
``settings.corrupt-<YYYYMMDD-HHMMSS>.json`` (the newest three backups are kept)
and defaults are used, with a notice for the UI.

Everything here is stdlib only and has no Qt dependency.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from spells.cleanup import default_corrections, default_fillers
from spells.datafiles import data_path
from spells.history import DEFAULT_AUDIO_KEEP_COUNT, DEFAULT_AUDIO_KEEP_MB
from spells.models import Chord, ChordMode, DeliveryMethod, Profile
from spells.vk import generic_modifier

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

CORRUPT_BACKUPS_KEPT = 3

RETENTION_CHOICES = ("100", "7d", "30d", "off")

# Ctrl+Win: VK_CONTROL and VK_LWIN.
DEFAULT_MAIN_CHORD = Chord(keys=(0x11, 0x5B))

DEFAULT_ENABLED_LANGUAGES = ["en", "de", "sq"]

# Tone instruction per built-in profile, from the spec 9 table (the LLM user
# message carries it, spec 8.2). profiles.py builds its Profile objects from it.
DEFAULT_TONES: dict[str, str] = {
    "Chat": "casual; no trailing period on single-sentence messages",
    "Email and docs": "full sentences, standard capitalization and punctuation",
    "Code": "minimal rewriting; never alter identifiers, paths, or symbols",
    "Terminal": "no trailing punctuation",
    "Default": "standard cleanup",
}

# The filler and self-correction defaults (spec 8.1 table) are the data files
# data/fillers/<code>.txt and data/corrections/<code>.txt, read through
# cleanup.default_fillers() and cleanup.default_corrections().


class SettingsError(ValueError):
    """A settings dictionary has the wrong shape or a value of the wrong type."""


# --- schema ---


@dataclass(frozen=True)
class GeneralSettings:
    """The General tab plus the tray language choice (spec 14.1, 14.4)."""

    main_chord: Chord = DEFAULT_MAIN_CHORD
    language_chords: list[Chord] = field(default_factory=list)
    # The writing hotkeys (spec 8.5). Both are None until the user records one: an
    # empty chord cannot exist, and nothing should fire before he asks for it.
    compose_chord: Chord | None = None
    edit_chord: Chord | None = None
    mic_device: str | None = None
    # "auto", or a language code for Locked mode (spec 7.2).
    language_mode: str = "auto"
    enabled_languages: list[str] = field(default_factory=lambda: list(DEFAULT_ENABLED_LANGUAGES))
    autostart: bool = True
    keep_mic_warm: bool = False
    sounds: bool = True
    # Live insertion while speaking (spec 6, B5-57): on where the hardware allows it,
    # and only in the plain profiles until live_text_everywhere is switched on.
    live_text: bool = True
    live_text_everywhere: bool = False
    # 0 means never unload (spec 13).
    idle_unload_minutes: int = 0


@dataclass(frozen=True)
class CleanupSettings:
    """The Cleanup tab (spec 8.1, 8.2, 14.4). Keys of the dicts: profile name or language code."""

    enabled: bool = True
    timeout_ms: int = 2500
    tones: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TONES))
    # Read from data/ on every construction; the reader returns fresh lists each time.
    fillers: dict[str, list[str]] = field(default_factory=default_fillers)
    corrections: dict[str, list[str]] = field(default_factory=default_corrections)


@dataclass(frozen=True)
class ProfileRule:
    """One row of the Apps tab (spec 9).

    User rules: with both matchers set, both must match; with one, that one. Built-in
    rules (profiles.BUILTIN_RULES) use the same shape with the spec 9 semantics.
    """

    name: str
    match_process: list[str]
    match_title: list[str]
    profile: Profile


@dataclass(frozen=True)
class TermEntry:
    """A vocabulary term; edited_at orders the Whisper prompt (spec 7.3, V2-5)."""

    text: str
    edited_at: float


@dataclass(frozen=True)
class Replacement:
    find: str
    replace: str


@dataclass(frozen=True)
class Snippet:
    trigger: str
    text: str


@dataclass(frozen=True)
class Vocabulary:
    """The Vocabulary tab (spec 10)."""

    terms: list[TermEntry] = field(default_factory=list)
    replacements: list[Replacement] = field(default_factory=list)
    snippets: list[Snippet] = field(default_factory=list)


@dataclass(frozen=True)
class HistorySettings:
    """The History page (spec 15).

    retention is one of RETENTION_CHOICES. keep_audio is the explicit exception to the
    promise of spec 17 that audio is never written to disk: off by default, and with its
    own retention (the newest audio_keep_count recordings within audio_keep_mb megabytes,
    oldest deleted first) so it cannot fill the disk.
    """

    retention: str = "100"
    keep_audio: bool = False
    audio_keep_count: int = DEFAULT_AUDIO_KEEP_COUNT
    audio_keep_mb: int = DEFAULT_AUDIO_KEEP_MB


@dataclass(frozen=True)
class UpdateSettings:
    """The About page's update switch and its bookkeeping (spec 3, 17, 19.7).

    ``weekly_check`` is off by default and is the only thing that lets Spells open a
    connection without the user pressing a button. The three timestamps are what the check
    and the tray balloon need between restarts: when a check last completed, and which
    version the tray last told the user about and when, so a balloon shows at most once a
    day. A settings file written before this release simply has no ``updates`` block and
    loads with these defaults, which is the whole migration.
    """

    weekly_check: bool = False
    last_check: float = 0.0
    last_offer_version: str = ""
    last_offer_at: float = 0.0


@dataclass(frozen=True)
class DiagnosticsSettings:
    """The Diagnostics tab (spec 14.4) plus the cached GPU choice (decision V3-F2).

    gpu_device_override is the user's dropdown choice (None: automatic).
    gpu_device_index and gpu_device_name cache the automatic probe result so the
    probe reruns only when the GPU name set changes.
    """

    debug_logging: bool = False
    gpu_device_override: int | None = None
    gpu_device_index: int | None = None
    gpu_device_name: str | None = None


@dataclass(frozen=True)
class Settings:
    schema_version: int = SCHEMA_VERSION
    general: GeneralSettings = field(default_factory=GeneralSettings)
    cleanup: CleanupSettings = field(default_factory=CleanupSettings)
    profiles: list[ProfileRule] = field(default_factory=list)
    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    history: HistorySettings = field(default_factory=HistorySettings)
    updates: UpdateSettings = field(default_factory=UpdateSettings)
    diagnostics: DiagnosticsSettings = field(default_factory=DiagnosticsSettings)


def default_settings() -> Settings:
    """A fresh Settings with every default; each call returns independent lists and dicts."""
    return Settings()


def settings_path() -> Path:
    """%APPDATA%\\Spells\\settings.json (spec 15)."""
    return Path(os.environ["APPDATA"]) / "Spells" / "settings.json"


# --- the installer's language choice (spec 19.3 step 7, B5-32) ---

FIRST_RUN_FILE = "first-run.json"


@lru_cache(maxsize=1)
def known_language_codes() -> frozenset[str]:
    """Every language code the app recognises, from data/whisper_languages.json."""
    try:
        table = json.loads(data_path("whisper_languages.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        log.warning("The language table could not be read", exc_info=True)
        return frozenset()
    if not isinstance(table, dict):
        return frozenset()
    return frozenset(code for code in table.values() if isinstance(code, str) and code)


def first_run_languages(path: Path) -> list[str]:
    """The language codes the online installer wrote beside the executable, else an empty list.

    The online installer of 19.3 asks which languages the user will dictate in, downloads the
    models those need, and writes the answer to ``{app}\\first-run.json`` as
    ``{"enabled_languages": ["en", "de"]}``. Only a start that finds no settings file reads it
    (``app._start``), so a reinstall never overwrites a choice made in Settings later.

    Nothing here raises. A missing file, an unreadable one, a file that is not JSON, the wrong
    shape, or codes the app does not know all read as "nothing to seed", and the defaults of
    ``GeneralSettings`` stand. Unknown codes are dropped one by one, so a file naming one
    known and one unknown language still seeds the known one.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    except Exception:
        log.warning("The first-run language file %s could not be read", path, exc_info=True)
        return []
    if not isinstance(raw, dict):
        return []
    codes = raw.get("enabled_languages")
    if not isinstance(codes, list):
        return []
    known = known_language_codes()
    if not known:
        return []
    found: list[str] = []
    for entry in codes:
        if not isinstance(entry, str):
            continue
        code = entry.strip().lower()
        if code in known and code not in found:
            found.append(code)
    return found


def with_enabled_languages(settings: Settings, languages: Sequence[str]) -> Settings:
    """A copy whose general.enabled_languages is ``languages``; an empty list changes nothing.

    ``language_mode`` falls back to "auto" when the language it locks to is not in the new
    set, because ``from_dict`` refuses that combination and the result has to load back.
    """
    if not languages:
        return settings
    general = settings.general
    mode = general.language_mode
    if mode != "auto" and mode not in languages:
        mode = "auto"
    return replace(
        settings,
        general=replace(general, enabled_languages=list(languages), language_mode=mode),
    )


# --- to_dict ---


def _chord_to_dict(chord: Chord, *, with_language: bool) -> dict:
    raw: dict = {"keys": list(chord.keys)}
    if with_language:
        raw["language"] = chord.language
    if chord.mode is not ChordMode.DICTATE:
        raw["mode"] = chord.mode.value
    return raw


def _opt_chord_to_dict(chord: Chord | None, *, with_language: bool = False) -> dict | None:
    return None if chord is None else _chord_to_dict(chord, with_language=with_language)


def _profile_to_dict(profile: Profile) -> dict:
    return {
        "name": profile.name,
        "cleanup": profile.cleanup,
        "tone": profile.tone,
        "delivery": profile.delivery.value,
        "drop_trailing_period_single_sentence": profile.drop_trailing_period_single_sentence,
        "drop_trailing_punctuation": profile.drop_trailing_punctuation,
    }


def to_dict(settings: Settings) -> dict:
    """A JSON-compatible dict (plain dicts, lists, strings, numbers, booleans, None)."""
    general = settings.general
    cleanup = settings.cleanup
    vocabulary = settings.vocabulary
    diagnostics = settings.diagnostics
    return {
        "schema_version": settings.schema_version,
        "general": {
            "main_chord": _chord_to_dict(general.main_chord, with_language=False),
            "language_chords": [
                _chord_to_dict(chord, with_language=True) for chord in general.language_chords
            ],
            "compose_chord": _opt_chord_to_dict(general.compose_chord),
            "edit_chord": _opt_chord_to_dict(general.edit_chord),
            "mic_device": general.mic_device,
            "language_mode": general.language_mode,
            "enabled_languages": list(general.enabled_languages),
            "autostart": general.autostart,
            "keep_mic_warm": general.keep_mic_warm,
            "sounds": general.sounds,
            "live_text": general.live_text,
            "live_text_everywhere": general.live_text_everywhere,
            "idle_unload_minutes": general.idle_unload_minutes,
        },
        "cleanup": {
            "enabled": cleanup.enabled,
            "timeout_ms": cleanup.timeout_ms,
            "tones": dict(cleanup.tones),
            "fillers": {code: list(words) for code, words in cleanup.fillers.items()},
            "corrections": {code: list(words) for code, words in cleanup.corrections.items()},
        },
        "profiles": [
            {
                "name": rule.name,
                "match_process": list(rule.match_process),
                "match_title": list(rule.match_title),
                "profile": _profile_to_dict(rule.profile),
            }
            for rule in settings.profiles
        ],
        "vocabulary": {
            "terms": [{"text": t.text, "edited_at": t.edited_at} for t in vocabulary.terms],
            "replacements": [
                {"find": r.find, "replace": r.replace} for r in vocabulary.replacements
            ],
            "snippets": [{"trigger": s.trigger, "text": s.text} for s in vocabulary.snippets],
        },
        "history": {
            "retention": settings.history.retention,
            "keep_audio": settings.history.keep_audio,
            "audio_keep_count": settings.history.audio_keep_count,
            "audio_keep_mb": settings.history.audio_keep_mb,
        },
        "updates": {
            "weekly_check": settings.updates.weekly_check,
            "last_check": settings.updates.last_check,
            "last_offer_version": settings.updates.last_offer_version,
            "last_offer_at": settings.updates.last_offer_at,
        },
        "diagnostics": {
            "debug_logging": diagnostics.debug_logging,
            "gpu_device_override": diagnostics.gpu_device_override,
            "gpu_device_index": diagnostics.gpu_device_index,
            "gpu_device_name": diagnostics.gpu_device_name,
        },
    }


# --- from_dict ---

_T = TypeVar("_T")
_REQUIRED = object()


def _type_name(value: object) -> str:
    return "null" if value is None else type(value).__name__


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class _Node:
    """A JSON object under validation. Every getter raises SettingsError on a wrong type.

    Missing keys take the given default; ``_REQUIRED`` makes a key mandatory.
    Unknown keys are ignored.
    """

    def __init__(self, raw: object, where: str) -> None:
        if not isinstance(raw, dict):
            raise SettingsError(f"{where or 'settings'}: expected an object, got {_type_name(raw)}")
        self._raw = raw
        self._where = where

    def _label(self, key: str) -> str:
        return f"{self._where}.{key}" if self._where else key

    def _value(self, key: str, default: object) -> object:
        if key in self._raw:
            return self._raw[key]
        if default is _REQUIRED:
            raise SettingsError(f"{self._label(key)}: missing")
        return default

    def child(self, key: str) -> _Node:
        """The nested object under key; a missing key reads as an empty object."""
        return _Node(self._value(key, {}), self._label(key))

    def get_bool(self, key: str, default: object = _REQUIRED) -> bool:
        value = self._value(key, default)
        if not isinstance(value, bool):
            raise SettingsError(
                f"{self._label(key)}: expected true or false, got {_type_name(value)}"
            )
        return value

    def get_int(self, key: str, default: object = _REQUIRED, *, minimum: int | None = None) -> int:
        value = self._value(key, default)
        if not _is_int(value):
            raise SettingsError(f"{self._label(key)}: expected an integer, got {_type_name(value)}")
        assert isinstance(value, int)
        if minimum is not None and value < minimum:
            raise SettingsError(f"{self._label(key)}: expected at least {minimum}, got {value}")
        return value

    def get_opt_int(self, key: str, default: object = _REQUIRED) -> int | None:
        value = self._value(key, default)
        if value is None:
            return None
        if not _is_int(value):
            raise SettingsError(
                f"{self._label(key)}: expected an integer or null, got {_type_name(value)}"
            )
        assert isinstance(value, int)
        return value

    def get_float(self, key: str, default: object = _REQUIRED) -> float:
        value = self._value(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{self._label(key)}: expected a number, got {_type_name(value)}")
        return float(value)

    def get_str(
        self, key: str, default: object = _REQUIRED, *, choices: tuple[str, ...] | None = None
    ) -> str:
        value = self._value(key, default)
        if not isinstance(value, str):
            raise SettingsError(f"{self._label(key)}: expected a string, got {_type_name(value)}")
        if choices is not None and value not in choices:
            raise SettingsError(f"{self._label(key)}: expected one of {choices}, got {value!r}")
        return value

    def get_opt_str(self, key: str, default: object = _REQUIRED) -> str | None:
        value = self._value(key, default)
        if value is None:
            return None
        if not isinstance(value, str):
            raise SettingsError(
                f"{self._label(key)}: expected a string or null, got {_type_name(value)}"
            )
        return value

    def get_list(
        self, key: str, default: object, parse: Callable[[object, str], _T]
    ) -> list[_T]:
        value = self._value(key, default)
        if not isinstance(value, list):
            raise SettingsError(f"{self._label(key)}: expected a list, got {_type_name(value)}")
        label = self._label(key)
        return [parse(item, f"{label}[{index}]") for index, item in enumerate(value)]

    def get_str_list(self, key: str, default: object = _REQUIRED) -> list[str]:
        return self.get_list(key, default, _parse_str)

    def get_dict(
        self, key: str, default: object, parse: Callable[[object, str], _T]
    ) -> dict[str, _T]:
        value = self._value(key, default)
        if not isinstance(value, dict):
            raise SettingsError(f"{self._label(key)}: expected an object, got {_type_name(value)}")
        label = self._label(key)
        return {name: parse(item, f"{label}.{name}") for name, item in value.items()}


def _parse_str(raw: object, where: str) -> str:
    if not isinstance(raw, str):
        raise SettingsError(f"{where}: expected a string, got {_type_name(raw)}")
    return raw


def _parse_str_list(raw: object, where: str) -> list[str]:
    if not isinstance(raw, list):
        raise SettingsError(f"{where}: expected a list, got {_type_name(raw)}")
    return [_parse_str(item, f"{where}[{index}]") for index, item in enumerate(raw)]


def _parse_key(raw: object, where: str) -> int:
    if not _is_int(raw):
        raise SettingsError(f"{where}: expected a virtual-key code, got {_type_name(raw)}")
    assert isinstance(raw, int)
    return raw


def _parse_chord(
    raw: object, where: str, *, with_language: bool, mode: ChordMode = ChordMode.DICTATE
) -> Chord:
    """One chord. The mode is optional in the file and has to match the field it sits in.

    A chord's mode is decided by which setting holds it (spec 8.5), so a stored mode is a
    record rather than a choice: a hand-edited file that names the wrong one is refused
    instead of quietly turning a dictation hotkey into a writing one.
    """
    node = _Node(raw, where)
    keys = node.get_list("keys", _REQUIRED, _parse_key)
    if not keys:
        raise SettingsError(f"{where}.keys: a chord needs at least one key")
    language = node.get_str("language") if with_language else None
    stored = node.get_str("mode", mode.value, choices=tuple(m.value for m in ChordMode))
    if stored != mode.value:
        raise SettingsError(
            f"{where}.mode: expected {mode.value!r} here, got {stored!r}"
        )
    return Chord(keys=tuple(keys), language=language, mode=mode)


def _parse_main_chord(raw: object, where: str) -> Chord:
    return _parse_chord(raw, where, with_language=False)


def _parse_language_chord(raw: object, where: str) -> Chord:
    return _parse_chord(raw, where, with_language=True)


def _parse_writing_chord(raw: object, where: str, mode: ChordMode) -> Chord | None:
    if raw is None:
        return None
    return _parse_chord(raw, where, with_language=False, mode=mode)


def _generic_keys(chord: Chord) -> frozenset[int]:
    """The chord's keys, with left/right modifier variants folded together.

    Uses spells.vk.generic_modifier: the hook can report either spelling for a
    chord the user holds, so a chord configured with one spelling must still
    conflict with the other. The Win keys are the exception and stay distinct
    (spec 6, decision B3-24).
    """
    return frozenset(generic_modifier(key) for key in chord.keys)


def _format_keys(keys: frozenset[int]) -> str:
    return "{" + ", ".join(f"0x{key:X}" for key in sorted(keys)) + "}"


def _check_chord_conflicts(general: GeneralSettings) -> None:
    """Reject a chord set where one chord's keys equal or are a subset of another's.

    A chord that is a prefix of another fires first on the hook, so the longer
    chord could never trigger, whether the two chords are exactly equal or one
    is a subset of the other (spec 6, decision B3-24). Chords are compared by
    their generic key sets in configuration order (general.main_chord, then
    general.language_chords by index, then the compose and edit chords of spec
    8.5, which is the order the hook matches them in), and the first conflicting
    pair found is named in the error.
    """
    labeled: list[tuple[str, frozenset[int]]] = [
        ("general.main_chord", _generic_keys(general.main_chord))
    ]
    labeled.extend(
        (f"general.language_chords[{index}]", _generic_keys(chord))
        for index, chord in enumerate(general.language_chords)
    )
    labeled.extend(
        (f"general.{name}", _generic_keys(chord))
        for name, chord in (
            ("compose_chord", general.compose_chord),
            ("edit_chord", general.edit_chord),
        )
        if chord is not None
    )
    for i, (label_a, keys_a) in enumerate(labeled):
        for label_b, keys_b in labeled[i + 1 :]:
            if keys_a == keys_b:
                relation = "equal"
            elif keys_a <= keys_b or keys_b <= keys_a:
                relation = "a subset of one another"
            else:
                continue
            raise SettingsError(
                f"{label_a} conflicts with {label_b}: generic keys "
                f"{_format_keys(keys_a)} and {_format_keys(keys_b)} are {relation}; "
                "a prefix chord fires first on the hook, so the longer chord can "
                "never fire"
            )


def _parse_profile(raw: object, where: str) -> Profile:
    node = _Node(raw, where)
    delivery = node.get_str(
        "delivery", DeliveryMethod.PASTE.value, choices=tuple(m.value for m in DeliveryMethod)
    )
    return Profile(
        name=node.get_str("name"),
        cleanup=node.get_bool("cleanup", True),
        tone=node.get_str("tone", ""),
        delivery=DeliveryMethod(delivery),
        drop_trailing_period_single_sentence=node.get_bool(
            "drop_trailing_period_single_sentence", False
        ),
        drop_trailing_punctuation=node.get_bool("drop_trailing_punctuation", False),
    )


def _parse_rule(raw: object, where: str) -> ProfileRule:
    node = _Node(raw, where)
    return ProfileRule(
        name=node.get_str("name", ""),
        match_process=node.get_str_list("match_process", []),
        match_title=node.get_str_list("match_title", []),
        profile=_parse_profile(node._value("profile", _REQUIRED), f"{where}.profile"),
    )


def _parse_term(raw: object, where: str) -> TermEntry:
    node = _Node(raw, where)
    return TermEntry(text=node.get_str("text"), edited_at=node.get_float("edited_at", 0.0))


def _parse_replacement(raw: object, where: str) -> Replacement:
    node = _Node(raw, where)
    return Replacement(find=node.get_str("find"), replace=node.get_str("replace"))


def _parse_snippet(raw: object, where: str) -> Snippet:
    node = _Node(raw, where)
    return Snippet(trigger=node.get_str("trigger"), text=node.get_str("text"))


def _parse_general(node: _Node) -> GeneralSettings:
    defaults = GeneralSettings()
    enabled_languages = node.get_str_list("enabled_languages", defaults.enabled_languages)
    if not enabled_languages:
        raise SettingsError(
            f"{node._label('enabled_languages')}: at least one language must be enabled"
        )
    language_mode = node.get_str("language_mode", defaults.language_mode)
    if language_mode != "auto" and language_mode not in enabled_languages:
        raise SettingsError(
            f"{node._label('language_mode')}: expected 'auto' or one of the enabled "
            f"languages {enabled_languages}, got {language_mode!r}"
        )
    general = GeneralSettings(
        main_chord=_parse_main_chord(
            node._value("main_chord", _chord_to_dict(defaults.main_chord, with_language=False)),
            node._label("main_chord"),
        ),
        language_chords=node.get_list("language_chords", [], _parse_language_chord),
        compose_chord=_parse_writing_chord(
            node._value("compose_chord", None),
            node._label("compose_chord"),
            ChordMode.COMPOSE,
        ),
        edit_chord=_parse_writing_chord(
            node._value("edit_chord", None), node._label("edit_chord"), ChordMode.EDIT
        ),
        mic_device=node.get_opt_str("mic_device", defaults.mic_device),
        language_mode=language_mode,
        enabled_languages=enabled_languages,
        autostart=node.get_bool("autostart", defaults.autostart),
        keep_mic_warm=node.get_bool("keep_mic_warm", defaults.keep_mic_warm),
        sounds=node.get_bool("sounds", defaults.sounds),
        live_text=node.get_bool("live_text", defaults.live_text),
        live_text_everywhere=node.get_bool(
            "live_text_everywhere", defaults.live_text_everywhere
        ),
        idle_unload_minutes=node.get_int(
            "idle_unload_minutes", defaults.idle_unload_minutes, minimum=0
        ),
    )
    _check_chord_conflicts(general)
    return general


def _parse_cleanup(node: _Node) -> CleanupSettings:
    defaults = CleanupSettings()
    return CleanupSettings(
        enabled=node.get_bool("enabled", defaults.enabled),
        timeout_ms=node.get_int("timeout_ms", defaults.timeout_ms, minimum=1),
        tones=node.get_dict("tones", defaults.tones, _parse_str),
        fillers=node.get_dict("fillers", defaults.fillers, _parse_str_list),
        corrections=node.get_dict("corrections", defaults.corrections, _parse_str_list),
    )


def _parse_vocabulary(node: _Node) -> Vocabulary:
    return Vocabulary(
        terms=node.get_list("terms", [], _parse_term),
        replacements=node.get_list("replacements", [], _parse_replacement),
        snippets=node.get_list("snippets", [], _parse_snippet),
    )


def _parse_history(node: _Node) -> HistorySettings:
    defaults = HistorySettings()
    return HistorySettings(
        retention=node.get_str("retention", defaults.retention, choices=RETENTION_CHOICES),
        keep_audio=node.get_bool("keep_audio", defaults.keep_audio),
        audio_keep_count=node.get_int(
            "audio_keep_count", defaults.audio_keep_count, minimum=1
        ),
        audio_keep_mb=node.get_int("audio_keep_mb", defaults.audio_keep_mb, minimum=1),
    )


def _parse_updates(node: _Node) -> UpdateSettings:
    defaults = UpdateSettings()
    return UpdateSettings(
        weekly_check=node.get_bool("weekly_check", defaults.weekly_check),
        last_check=node.get_float("last_check", defaults.last_check),
        last_offer_version=node.get_str("last_offer_version", defaults.last_offer_version),
        last_offer_at=node.get_float("last_offer_at", defaults.last_offer_at),
    )


def _parse_diagnostics(node: _Node) -> DiagnosticsSettings:
    defaults = DiagnosticsSettings()
    return DiagnosticsSettings(
        debug_logging=node.get_bool("debug_logging", defaults.debug_logging),
        gpu_device_override=node.get_opt_int("gpu_device_override", defaults.gpu_device_override),
        gpu_device_index=node.get_opt_int("gpu_device_index", defaults.gpu_device_index),
        gpu_device_name=node.get_opt_str("gpu_device_name", defaults.gpu_device_name),
    )


def from_dict(raw: dict) -> Settings:
    """Validate a dict of the current schema version into Settings.

    Unknown keys are ignored, missing keys take their defaults, and a value of the
    wrong type or an out-of-range value raises SettingsError. Run ``migrate`` first
    on data read from disk.
    """
    root = _Node(raw, "")
    version = root.get_int("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise SettingsError(
            f"schema_version: expected {SCHEMA_VERSION}, got {version} (run migrate first)"
        )
    return Settings(
        schema_version=version,
        general=_parse_general(root.child("general")),
        cleanup=_parse_cleanup(root.child("cleanup")),
        profiles=root.get_list("profiles", [], _parse_rule),
        vocabulary=_parse_vocabulary(root.child("vocabulary")),
        history=_parse_history(root.child("history")),
        updates=_parse_updates(root.child("updates")),
        diagnostics=_parse_diagnostics(root.child("diagnostics")),
    )


# --- migrate ---


def migrate(raw: dict) -> dict:
    """Bring a raw settings dict forward to SCHEMA_VERSION (spec 15).

    Returns a new dict; the input is not modified. A missing ``schema_version`` is
    treated as 1. Future versions add one step each, chained in order. A version
    newer than SCHEMA_VERSION cannot be read and raises SettingsError.
    """
    if not isinstance(raw, dict):
        raise SettingsError(f"settings: expected an object, got {_type_name(raw)}")
    migrated = dict(raw)
    version = migrated.get("schema_version", 1)
    if not _is_int(version) or version < 1:
        raise SettingsError(f"schema_version: expected a positive integer, got {version!r}")
    if version > SCHEMA_VERSION:
        raise SettingsError(
            f"schema_version: {version} is newer than this app supports ({SCHEMA_VERSION})"
        )
    # Future migrations: `if version == 1: migrated = _migrate_1_to_2(migrated); version = 2`
    migrated["schema_version"] = version
    return migrated


# --- files ---


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _backup_pattern(path: Path) -> str:
    return f"{path.stem}.corrupt-*.json"


def _backup_corrupt(path: Path) -> Path:
    """Rename path to <stem>.corrupt-<timestamp>.json, then prune to the newest backups."""
    stamp = _timestamp()
    target = path.with_name(f"{path.stem}.corrupt-{stamp}.json")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.stem}.corrupt-{stamp}-{counter}.json")
        counter += 1
    os.replace(path, target)
    try:
        # The backup made last is the newest, whatever mtime the corrupt file carried
        # (a file restored from an old copy keeps that copy's mtime).
        os.utime(target, None)
    except OSError:
        pass
    _prune_backups(path)
    return target


def _prune_backups(path: Path) -> None:
    """Delete all but the CORRUPT_BACKUPS_KEPT most recently modified backups."""
    backups: list[tuple[float, Path]] = []
    for backup in path.parent.glob(_backup_pattern(path)):
        try:
            backups.append((backup.stat().st_mtime, backup))
        except OSError:
            continue
    backups.sort(key=lambda item: item[0])
    for _mtime, stale in backups[:-CORRUPT_BACKUPS_KEPT]:
        try:
            stale.unlink()
        except OSError:
            log.warning("Could not remove old settings backup %s", stale, exc_info=True)


def save(settings: Settings, path: Path) -> None:
    """Write settings atomically: <path>.tmp, fsync, then os.replace. Creates parent dirs.

    Raises SettingsError before touching the disk when the settings would not load
    back, so a file that the next start would move aside as corrupt is never written.
    """
    raw = to_dict(settings)
    from_dict(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    text = json.dumps(raw, indent=2, ensure_ascii=False) + "\n"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load(path: Path) -> tuple[Settings, str | None]:
    """Read settings from path.

    Returns (settings, notice). A missing file yields the defaults and no notice.
    A file that cannot be parsed, migrated, or validated is moved aside as
    ``<stem>.corrupt-<YYYYMMDD-HHMMSS>.json`` (newest three kept) and the defaults
    are returned with a human-readable notice. A file that cannot be read at all
    (permissions, a directory in the way) is left alone; defaults and a notice.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return default_settings(), None
    except OSError as exc:
        log.warning("Settings file %s could not be read", path, exc_info=True)
        return default_settings(), (
            f"Your settings could not be read ({exc.strerror or exc}). "
            "Spells is using its defaults for this session."
        )

    try:
        raw = json.loads(data.decode("utf-8"))
        settings = from_dict(migrate(raw))
    except (ValueError, TypeError) as exc:
        # ValueError covers UnicodeDecodeError, JSONDecodeError, and SettingsError.
        reason = str(exc) or type(exc).__name__
        try:
            backup = _backup_corrupt(path)
        except OSError:
            log.warning("Corrupt settings file %s could not be moved aside", path, exc_info=True)
            return default_settings(), (
                f"Your settings file could not be read ({reason}) and could not be moved "
                "aside. Spells is using its defaults."
            )
        log.warning("Settings file %s was corrupt (%s); moved to %s", path, reason, backup.name)
        return default_settings(), (
            f"Your settings could not be read ({reason}) and were reset to defaults. "
            f"The old file was kept as {backup.name} next to it."
        )
    return settings, None


# --- store ---


class ConfigStore:
    """Owns the loaded Settings, persists updates, and notifies subscribers.

    ``update`` is thread-safe. Subscribers run on the updating thread, after the
    write succeeded and outside the store's lock. A leftover ``<path>.tmp`` from a
    write interrupted by a crash is removed on construction (decision V3-F8).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else settings_path()
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[Settings], None]] = []
        self._remove_leftover_tmp()
        self._settings, self._notice = load(self._path)

    def _remove_leftover_tmp(self) -> None:
        tmp = _tmp_path(self._path)
        try:
            if tmp.exists():
                tmp.unlink()
                log.info("Removed leftover settings temp file %s", tmp)
        except OSError:
            log.warning("Could not remove leftover settings temp file %s", tmp, exc_info=True)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def notice(self) -> str | None:
        """A human-readable message when load fell back to defaults, else None."""
        return self._notice

    def update(self, mutator: Callable[[Settings], Settings]) -> Settings:
        """Apply mutator to the current settings, save, notify subscribers, return the result.

        If the mutator raises or the save fails (including SettingsError for a
        result that would not load back), the in-memory settings stay unchanged
        and nobody is notified.
        """
        with self._lock:
            new_settings = mutator(self._settings)
            if not isinstance(new_settings, Settings):
                raise TypeError(
                    f"settings mutator must return Settings, got {_type_name(new_settings)}"
                )
            save(new_settings, self._path)
            self._settings = new_settings
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(new_settings)
            except Exception:
                log.exception("Settings subscriber %r failed", callback)
        return new_settings

    def subscribe(self, callback: Callable[[Settings], None]) -> Callable[[], None]:
        """Register callback for every successful update; returns an unsubscribe function."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return unsubscribe
