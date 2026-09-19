"""Tests for spells.config: schema, serialization, migration, load/save, ConfigStore."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from pathlib import Path

import pytest

from spells import config
from spells.config import (
    DEFAULT_TONES,
    SCHEMA_VERSION,
    ConfigStore,
    DiagnosticsSettings,
    HistorySettings,
    ProfileRule,
    Replacement,
    Settings,
    SettingsError,
    Snippet,
    TermEntry,
    Vocabulary,
    default_settings,
    from_dict,
    load,
    migrate,
    save,
    settings_path,
    to_dict,
)
from spells.datafiles import read_lines
from spells.models import Chord, ChordMode, DeliveryMethod, Profile

BACKUP_NAME = re.compile(r"^settings\.corrupt-\d{8}-\d{6}\.json$")


def _custom_settings() -> Settings:
    """A Settings value that differs from the defaults in every section."""
    base = default_settings()
    general = dataclasses.replace(
        base.general,
        main_chord=Chord(keys=(0x11, 0x12)),
        language_chords=[Chord(keys=(0x11, 0x5B, 0x44), language="de")],
        mic_device="USB Mic",
        language_mode="de",
        enabled_languages=["en", "de"],
        autostart=False,
        keep_mic_warm=True,
        sounds=False,
        live_text=False,
        live_text_everywhere=True,
        idle_unload_minutes=15,
    )
    cleanup = dataclasses.replace(
        base.cleanup,
        enabled=False,
        timeout_ms=1000,
        tones={**base.cleanup.tones, "Chat": "be brief"},
        fillers={"en": ["um"], "de": [], "sq": ["hm"]},
        corrections={"en": ["no wait"], "de": [], "sq": []},
    )
    rule = ProfileRule(
        name="Jira in Chrome",
        match_process=["chrome.exe"],
        match_title=["Jira"],
        profile=Profile(
            name="Email and docs",
            cleanup=True,
            tone="formal",
            delivery=DeliveryMethod.TYPE,
        ),
    )
    vocabulary = Vocabulary(
        terms=[TermEntry(text="Contoso", edited_at=1700000000.5)],
        replacements=[Replacement(find="con toso", replace="Contoso")],
        snippets=[Snippet(trigger="sign off", text="Best regards,\nAlex")],
    )
    return Settings(
        general=general,
        cleanup=cleanup,
        profiles=[rule],
        vocabulary=vocabulary,
        history=HistorySettings(retention="7d"),
        diagnostics=DiagnosticsSettings(
            debug_logging=True,
            gpu_device_override=1,
            gpu_device_index=0,
            gpu_device_name="Radeon 780M",
        ),
    )


def _backup_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("settings.corrupt-*.json"))


# --- defaults and shape ---


def test_default_settings_match_spec():
    settings = default_settings()

    assert settings.schema_version == SCHEMA_VERSION == 1
    general = settings.general
    assert general.main_chord == Chord(keys=(0x11, 0x5B))
    assert general.main_chord.language is None
    assert general.language_chords == []
    assert general.mic_device is None
    assert general.language_mode == "auto"
    assert general.enabled_languages == ["en", "de", "sq"]
    assert general.autostart is True
    assert general.keep_mic_warm is False
    assert general.sounds is True
    assert general.live_text is True
    assert general.live_text_everywhere is False
    assert general.idle_unload_minutes == 0

    cleanup = settings.cleanup
    assert cleanup.enabled is True
    assert cleanup.timeout_ms == 2500
    assert set(cleanup.tones) == {"Chat", "Email and docs", "Code", "Terminal", "Default"}
    assert cleanup.tones == DEFAULT_TONES
    assert "identifiers" in cleanup.tones["Code"]
    assert cleanup.fillers["en"][:3] == ["um", "uh", "erm"]
    assert "I mean" in cleanup.fillers["en"]
    assert "let me rephrase" in cleanup.corrections["en"]
    assert "ähm" in cleanup.fillers["de"]
    assert "besser gesagt" in cleanup.corrections["de"]
    assert "domethanë" in cleanup.fillers["sq"]
    assert "jo prit" in cleanup.corrections["sq"]

    assert settings.profiles == []
    assert settings.vocabulary == Vocabulary(terms=[], replacements=[], snippets=[])
    assert settings.history.retention == "100"
    assert settings.diagnostics.debug_logging is False
    assert settings.diagnostics.gpu_device_override is None
    assert settings.diagnostics.gpu_device_index is None
    assert settings.diagnostics.gpu_device_name is None


@pytest.mark.parametrize("code", ["en", "de", "sq"])
def test_default_lists_come_from_the_data_files(code):
    cleanup = default_settings().cleanup

    assert cleanup.fillers[code] == read_lines(f"fillers/{code}.txt")
    assert cleanup.corrections[code] == read_lines(f"corrections/{code}.txt")


def test_default_lists_cover_exactly_the_shipped_languages():
    cleanup = default_settings().cleanup

    assert set(cleanup.fillers) == {"en", "de", "sq"}
    assert set(cleanup.corrections) == {"en", "de", "sq"}


def test_default_settings_returns_independent_copies():
    first = default_settings()
    second = default_settings()
    first.cleanup.fillers["en"].append("mutated")
    first.general.enabled_languages.append("fr")

    assert "mutated" not in second.cleanup.fillers["en"]
    assert second.general.enabled_languages == ["en", "de", "sq"]


def test_settings_are_frozen():
    settings = default_settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.general = settings.general  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.general.autostart = False  # type: ignore[misc]


def test_settings_error_is_a_value_error():
    assert issubclass(SettingsError, ValueError)


# --- to_dict / from_dict ---


def test_to_dict_round_trips_through_json():
    original = _custom_settings()

    raw = to_dict(original)
    restored = from_dict(json.loads(json.dumps(raw)))

    assert restored == original


def test_to_dict_uses_plain_json_values():
    raw = to_dict(_custom_settings())

    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["general"]["main_chord"] == {"keys": [0x11, 0x12]}
    assert raw["general"]["language_chords"] == [{"keys": [0x11, 0x5B, 0x44], "language": "de"}]
    assert raw["profiles"][0]["profile"]["delivery"] == "type"
    assert raw["profiles"][0]["profile"]["drop_trailing_period_single_sentence"] is False
    assert raw["vocabulary"]["terms"] == [{"text": "Contoso", "edited_at": 1700000000.5}]
    assert raw["history"] == {
        "retention": "7d",
        "keep_audio": False,
        "audio_keep_count": 200,
        "audio_keep_mb": 1000,
    }
    assert raw["diagnostics"]["gpu_device_override"] == 1


def test_defaults_round_trip():
    assert from_dict(to_dict(default_settings())) == default_settings()


def test_from_dict_ignores_unknown_keys():
    raw = to_dict(default_settings())
    raw["future_section"] = {"x": 1}
    raw["general"]["unknown_flag"] = True
    raw["general"]["main_chord"]["label"] = "Ctrl+Win"

    assert from_dict(raw) == default_settings()


def test_from_dict_fills_missing_keys_with_defaults():
    raw = {"schema_version": 1, "general": {"autostart": False}, "history": {}}

    settings = from_dict(raw)

    assert settings.general.autostart is False
    assert settings.general.main_chord == Chord(keys=(0x11, 0x5B))
    assert settings.cleanup == default_settings().cleanup
    assert settings.history.retention == "100"


def test_from_dict_accepts_missing_schema_version():
    settings = from_dict({"general": {"sounds": False}})

    assert settings.schema_version == SCHEMA_VERSION
    assert settings.general.sounds is False


def test_from_dict_main_chord_language_is_always_none():
    raw = to_dict(default_settings())
    raw["general"]["main_chord"]["language"] = "de"

    assert from_dict(raw).general.main_chord.language is None


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"general": "nope"},
        {"general": {"autostart": "yes"}},
        {"general": {"autostart": 1}},
        {"general": {"idle_unload_minutes": True}},
        {"general": {"idle_unload_minutes": "5"}},
        {"general": {"idle_unload_minutes": -1}},
        {"general": {"enabled_languages": "en"}},
        {"general": {"enabled_languages": ["en", 2]}},
        {"general": {"enabled_languages": []}},
        {"general": {"language_mode": "fr"}},
        {"general": {"language_mode": ""}},
        {"general": {"language_mode": "de", "enabled_languages": ["en"]}},
        {"general": {"mic_device": 3}},
        {"general": {"language_mode": None}},
        {"general": {"main_chord": [17, 91]}},
        {"general": {"main_chord": {"keys": ["ctrl", "win"]}}},
        {"general": {"main_chord": {"keys": []}}},
        {"general": {"language_chords": [{"keys": [17]}]}},
        {"general": {"language_chords": [{"keys": [17], "language": 7}]}},
        {"cleanup": {"timeout_ms": 2500.0}},
        {"cleanup": {"tones": {"Chat": 1}}},
        {"cleanup": {"tones": ["casual"]}},
        {"cleanup": {"fillers": {"en": "um"}}},
        {"cleanup": {"fillers": {"en": ["um", None]}}},
        {"profiles": {}},
        {"profiles": [{"name": "x", "match_process": "chrome.exe", "profile": {"name": "Chat"}}]},
        {"profiles": [{"name": "x", "profile": {"name": "Chat", "delivery": "fax"}}]},
        {"profiles": [{"name": "x", "profile": {"name": "Chat", "cleanup": "on"}}]},
        {"profiles": [{"name": "x", "profile": {"tone": "no name"}}]},
        {"profiles": [{"name": "x"}]},
        {"vocabulary": {"terms": [{"text": 1}]}},
        {"vocabulary": {"terms": [{"text": "a", "edited_at": "now"}]}},
        {"vocabulary": {"replacements": [{"find": "a"}]}},
        {"vocabulary": {"snippets": [{"trigger": "a", "text": None}]}},
        {"history": {"retention": "forever"}},
        {"history": {"retention": 100}},
        {"diagnostics": {"debug_logging": None}},
        {"diagnostics": {"gpu_device_override": "0"}},
        {"schema_version": "1"},
    ],
)
def test_from_dict_rejects_wrong_types(raw):
    with pytest.raises(SettingsError):
        from_dict(raw)


def test_from_dict_accepts_locked_mode_for_an_enabled_language():
    assert from_dict({"general": {"language_mode": "de"}}).general.language_mode == "de"

    raw = {"general": {"language_mode": "fr", "enabled_languages": ["fr", "en"]}}
    assert from_dict(raw).general.language_mode == "fr"


# --- chord conflicts (hotkey verification): nested key sets can never both fire ---


def _chords_raw(main: list[int], *language_chords: tuple[list[int], str]) -> dict:
    return {
        "general": {
            "main_chord": {"keys": main},
            "language_chords": [
                {"keys": keys, "language": language} for keys, language in language_chords
            ],
        }
    }


def test_equal_chords_are_rejected():
    with pytest.raises(SettingsError, match=r"main_chord.*language_chords\[0\]"):
        from_dict(_chords_raw([0x11, 0x5B], ([0x11, 0x5B], "de")))


def test_chord_that_is_a_subset_of_another_is_rejected():
    # Ctrl+Win fires on the hook before Ctrl+Win+D is complete, so the longer chord is dead.
    with pytest.raises(SettingsError, match=r"main_chord.*language_chords\[0\]"):
        from_dict(_chords_raw([0x11, 0x5B], ([0x11, 0x5B, 0x44], "de")))


def test_chord_that_is_a_superset_of_another_is_rejected():
    with pytest.raises(SettingsError, match=r"main_chord.*language_chords\[0\]"):
        from_dict(_chords_raw([0x11, 0x5B, 0x44], ([0x11, 0x5B], "de")))


def test_nested_language_chords_are_rejected():
    with pytest.raises(SettingsError, match=r"language_chords\[0\].*language_chords\[1\]"):
        from_dict(_chords_raw([0x11, 0x5B], ([0x11, 0x44], "de"), ([0x11, 0x44, 0x45], "en")))


def test_disjoint_chords_are_accepted():
    raw = _chords_raw([0x11, 0x5B], ([0x11, 0x12, 0x44], "de"), ([0x11, 0x12, 0x45], "en"))

    settings = from_dict(raw)

    assert len(settings.general.language_chords) == 2


def test_chord_key_order_does_not_matter():
    with pytest.raises(SettingsError):
        from_dict(_chords_raw([0x11, 0x5B], ([0x5B, 0x11], "de")))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([0xA2, 0x5B], [0x11, 0x5B]),  # VK_LCONTROL counts as VK_CONTROL
        ([0xA3, 0x5B], [0x11, 0x5B]),  # VK_RCONTROL counts as VK_CONTROL
        ([0xA0, 0x44], [0xA1, 0x44]),  # VK_LSHIFT and VK_RSHIFT both count as VK_SHIFT
        ([0xA0, 0x44], [0x10, 0x44]),  # VK_LSHIFT counts as VK_SHIFT
        ([0xA4, 0x44], [0x12, 0x44]),  # VK_LMENU counts as VK_MENU
        ([0xA5, 0x44], [0xA4, 0x44]),  # VK_RMENU and VK_LMENU both count as VK_MENU
        ([0xA2, 0x5B], [0x11, 0x5B, 0x44]),  # generic form applies to subsets too
        ([0x11, 0x11, 0x5B], [0x11, 0x5B]),  # a duplicated key folds harmlessly
        ([0xA2, 0xA3, 0x44], [0x11, 0x44]),  # VK_LCONTROL and VK_RCONTROL both fold to VK_CONTROL
    ],
)
def test_left_and_right_modifiers_count_as_the_generic_key(first, second):
    with pytest.raises(SettingsError):
        from_dict(_chords_raw(first, (second, "de")))


def test_left_and_right_win_keys_are_distinct():
    settings = from_dict(_chords_raw([0x11, 0x5B], ([0x11, 0x5C], "de")))

    assert settings.general.language_chords[0].keys == (0x11, 0x5C)


def test_save_rejects_conflicting_chords(tmp_path):
    general = dataclasses.replace(
        default_settings().general,
        language_chords=[Chord(keys=(0x11, 0x5B, 0x44), language="de")],
    )
    invalid = dataclasses.replace(default_settings(), general=general)

    with pytest.raises(SettingsError):
        save(invalid, tmp_path / "settings.json")

    assert not (tmp_path / "settings.json").exists()


def test_from_dict_error_names_the_offending_key():
    with pytest.raises(SettingsError, match="general.autostart"):
        from_dict({"general": {"autostart": "yes"}})


# --- migrate ---


def test_migrate_treats_missing_schema_version_as_one():
    raw = {"general": {"sounds": False}}

    migrated = migrate(raw)

    assert migrated["schema_version"] == 1
    assert migrated["general"] == {"sounds": False}
    assert "schema_version" not in raw, "migrate must not mutate its input"


def test_migrate_keeps_current_version_unchanged():
    raw = to_dict(_custom_settings())
    assert migrate(raw) == raw


def test_migrate_rejects_newer_version():
    with pytest.raises(SettingsError):
        migrate({"schema_version": SCHEMA_VERSION + 1})


@pytest.mark.parametrize("version", ["1", 1.0, True, None, 0, -3])
def test_migrate_rejects_invalid_version(version):
    with pytest.raises(SettingsError):
        migrate({"schema_version": version})


def test_migrate_rejects_non_object():
    with pytest.raises(SettingsError):
        migrate([1, 2, 3])


# --- paths, save, load ---


def test_settings_path_uses_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert settings_path() == tmp_path / "Spells" / "settings.json"


def test_save_creates_parent_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "deep" / "er" / "settings.json"
    settings = _custom_settings()

    save(settings, path)

    assert path.is_file()
    assert not path.with_name("settings.json.tmp").exists()
    assert sorted(p.name for p in path.parent.iterdir()) == ["settings.json"]
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == SCHEMA_VERSION
    assert from_dict(on_disk) == settings


def test_save_writes_utf8_without_escaping(tmp_path):
    path = tmp_path / "settings.json"

    save(default_settings(), path)

    assert "ähm" in path.read_text(encoding="utf-8")


def test_save_overwrites_existing_file(tmp_path):
    path = tmp_path / "settings.json"
    save(default_settings(), path)

    save(_custom_settings(), path)

    assert load(path) == (_custom_settings(), None)


def test_save_rejects_settings_that_would_not_load_back(tmp_path):
    path = tmp_path / "settings.json"
    general = dataclasses.replace(default_settings().general, language_mode="fr")
    invalid = dataclasses.replace(default_settings(), general=general)

    with pytest.raises(SettingsError):
        save(invalid, path)

    assert not path.exists()
    assert not path.with_name("settings.json.tmp").exists()


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    settings = _custom_settings()

    save(settings, path)

    assert load(path) == (settings, None)


def test_load_missing_file_returns_defaults_without_notice(tmp_path):
    assert load(tmp_path / "missing" / "settings.json") == (default_settings(), None)


def test_load_corrupt_json_backs_up_and_returns_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ this is not json", encoding="utf-8")

    settings, notice = load(path)

    assert settings == default_settings()
    assert notice is not None
    assert not path.exists()
    backups = _backup_files(tmp_path)
    assert len(backups) == 1
    assert BACKUP_NAME.match(backups[0].name), backups[0].name
    assert backups[0].read_text(encoding="utf-8") == "{ this is not json"
    assert backups[0].name in notice


def test_load_invalid_schema_is_treated_as_corrupt(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"schema_version": 1, "general": {"autostart": "yes"}}))

    settings, notice = load(path)

    assert settings == default_settings()
    assert notice is not None
    assert not path.exists()
    assert len(_backup_files(tmp_path)) == 1


def test_load_newer_schema_is_treated_as_corrupt(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 5}))

    settings, notice = load(path)

    assert settings == default_settings()
    assert notice is not None
    assert len(_backup_files(tmp_path)) == 1


def test_load_undecodable_bytes_are_treated_as_corrupt(tmp_path):
    path = tmp_path / "settings.json"
    path.write_bytes(b"\xff\xfe\x00garbage")

    settings, notice = load(path)

    assert settings == default_settings()
    assert notice is not None
    assert len(_backup_files(tmp_path)) == 1


def test_load_keeps_only_three_newest_backups(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_timestamp", lambda: "20300101-000000")
    old_names = [
        "settings.corrupt-20200101-000000.json",
        "settings.corrupt-20200102-000000.json",
        "settings.corrupt-20200103-000000.json",
    ]
    for age, name in enumerate(old_names):
        backup = tmp_path / name
        backup.write_text("old", encoding="utf-8")
        os.utime(backup, (1_600_000_000 + age, 1_600_000_000 + age))
    path = tmp_path / "settings.json"
    path.write_text("broken", encoding="utf-8")

    load(path)

    assert [p.name for p in _backup_files(tmp_path)] == [
        "settings.corrupt-20200102-000000.json",
        "settings.corrupt-20200103-000000.json",
        "settings.corrupt-20300101-000000.json",
    ]


def test_load_five_corrupt_files_in_one_second_keep_the_three_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_timestamp", lambda: "20300101-000000")
    path = tmp_path / "settings.json"
    base = 1_700_000_000.0
    seen: set[str] = set()

    for version in range(1, 6):
        path.write_text(f"v{version}", encoding="utf-8")
        load(path)
        created = [p for p in _backup_files(tmp_path) if p.name not in seen]
        assert len(created) == 1, "each load moves exactly one file aside"
        # Pin the mtime so the test does not depend on filesystem timestamp resolution.
        os.utime(created[0], (base + version, base + version))
        seen = {p.name for p in _backup_files(tmp_path)}

    survivors = sorted(p.read_text(encoding="utf-8") for p in _backup_files(tmp_path))
    assert survivors == ["v3", "v4", "v5"]


def test_prune_backups_orders_by_mtime_not_by_name(tmp_path):
    path = tmp_path / "settings.json"
    # Name order is the reverse of age: the unsuffixed name sorts last but is the oldest.
    names = [
        "settings.corrupt-20300101-000000.json",
        "settings.corrupt-20300101-000000-1.json",
        "settings.corrupt-20300101-000000-2.json",
        "settings.corrupt-20300101-000000-3.json",
        "settings.corrupt-20300101-000000-4.json",
    ]
    for age, name in enumerate(names):
        backup = tmp_path / name
        backup.write_text(name, encoding="utf-8")
        os.utime(backup, (1_700_000_000 + age, 1_700_000_000 + age))

    config._prune_backups(path)

    assert [p.name for p in _backup_files(tmp_path)] == sorted(names[2:])


def test_backup_of_a_stale_file_counts_as_the_newest(tmp_path, monkeypatch):
    # A corrupt settings.json restored from an old copy keeps its old mtime; the backup
    # made from it must still be the newest so it is not pruned first.
    monkeypatch.setattr(config, "_timestamp", lambda: "20300101-000000")
    for age in range(3):
        backup = tmp_path / f"settings.corrupt-2020010{age + 1}-000000.json"
        backup.write_text("old", encoding="utf-8")
        os.utime(backup, (1_600_000_000 + age, 1_600_000_000 + age))
    path = tmp_path / "settings.json"
    path.write_text("stale and broken", encoding="utf-8")
    os.utime(path, (1_500_000_000, 1_500_000_000))

    load(path)

    survivors = {p.read_text(encoding="utf-8") for p in _backup_files(tmp_path)}
    assert "stale and broken" in survivors
    assert len(_backup_files(tmp_path)) == 3


def test_load_two_corrupt_files_in_the_same_second_keep_both(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_timestamp", lambda: "20300101-000000")
    path = tmp_path / "settings.json"

    path.write_text("first", encoding="utf-8")
    load(path)
    path.write_text("second", encoding="utf-8")
    load(path)

    contents = sorted(p.read_text(encoding="utf-8") for p in _backup_files(tmp_path))
    assert contents == ["first", "second"]


def test_load_unreadable_file_returns_defaults_with_notice(tmp_path):
    path = tmp_path / "settings.json"
    path.mkdir()

    settings, notice = load(path)

    assert settings == default_settings()
    assert notice is not None
    assert path.is_dir(), "an unreadable path is left alone"


# --- ConfigStore ---


def test_store_loads_existing_file(tmp_path):
    path = tmp_path / "settings.json"
    save(_custom_settings(), path)

    store = ConfigStore(path)

    assert store.settings == _custom_settings()
    assert store.notice is None
    assert store.path == path


def test_store_defaults_to_appdata_path(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))

    store = ConfigStore()

    assert store.path == tmp_path / "Spells" / "settings.json"
    assert store.settings == default_settings()


def test_store_exposes_corruption_notice(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("nope", encoding="utf-8")

    store = ConfigStore(path)

    assert store.settings == default_settings()
    assert store.notice is not None


def test_store_removes_leftover_tmp_file(tmp_path):
    path = tmp_path / "settings.json"
    save(_custom_settings(), path)
    leftover = tmp_path / "settings.json.tmp"
    leftover.write_text("half-written", encoding="utf-8")

    store = ConfigStore(path)

    assert not leftover.exists()
    assert store.settings == _custom_settings()


def test_store_update_saves_and_returns_new_settings(tmp_path):
    path = tmp_path / "settings.json"
    store = ConfigStore(path)

    def mutate(settings: Settings) -> Settings:
        general = dataclasses.replace(settings.general, autostart=False)
        return dataclasses.replace(settings, general=general)

    result = store.update(mutate)

    assert result.general.autostart is False
    assert store.settings is result
    assert load(path)[0] == result
    assert not path.with_name("settings.json.tmp").exists()


def test_store_subscribe_and_unsubscribe(tmp_path):
    store = ConfigStore(tmp_path / "settings.json")
    seen: list[Settings] = []
    unsubscribe = store.subscribe(seen.append)

    first = store.update(lambda s: dataclasses.replace(s, history=HistorySettings("off")))
    assert seen == [first]

    unsubscribe()
    unsubscribe()  # a second call is harmless
    store.update(lambda s: dataclasses.replace(s, history=HistorySettings("30d")))
    assert seen == [first]


def test_store_notifies_every_subscriber_in_order(tmp_path):
    store = ConfigStore(tmp_path / "settings.json")
    calls: list[str] = []
    store.subscribe(lambda s: calls.append("a"))
    store.subscribe(lambda s: calls.append("b"))

    store.update(lambda s: s)

    assert calls == ["a", "b"]


def test_store_subscriber_error_does_not_block_others(tmp_path, caplog):
    store = ConfigStore(tmp_path / "settings.json")
    calls: list[str] = []

    def broken(settings: Settings) -> None:
        raise RuntimeError("boom")

    store.subscribe(broken)
    store.subscribe(lambda s: calls.append("ok"))

    result = store.update(lambda s: dataclasses.replace(s, history=HistorySettings("7d")))

    assert calls == ["ok"]
    assert store.settings == result
    assert any(
        record.levelno == logging.ERROR and record.exc_info for record in caplog.records
    ), "the failure is logged with its traceback"


def test_store_update_rejects_non_settings_result(tmp_path):
    path = tmp_path / "settings.json"
    store = ConfigStore(path)
    seen: list[Settings] = []
    store.subscribe(seen.append)

    with pytest.raises(TypeError):
        store.update(lambda s: None)  # type: ignore[arg-type, return-value]

    assert store.settings == default_settings()
    assert seen == []
    assert not path.exists(), "nothing is written when the mutator misbehaves"


def test_store_update_rejects_settings_that_would_not_load_back(tmp_path):
    path = tmp_path / "settings.json"
    store = ConfigStore(path)
    seen: list[Settings] = []
    store.subscribe(seen.append)

    def lock_unknown_language(settings: Settings) -> Settings:
        general = dataclasses.replace(settings.general, language_mode="fr")
        return dataclasses.replace(settings, general=general)

    with pytest.raises(SettingsError):
        store.update(lock_unknown_language)

    assert store.settings == default_settings()
    assert seen == []
    assert not path.exists()


def test_store_update_keeps_old_settings_when_save_fails(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    store = ConfigStore(path)
    seen: list[Settings] = []
    store.subscribe(seen.append)

    def failing_save(settings: Settings, target: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(config, "save", failing_save)

    with pytest.raises(OSError):
        store.update(lambda s: dataclasses.replace(s, history=HistorySettings("off")))

    assert store.settings == default_settings()
    assert seen == []


# --- the online installer's first-run language file (B5-32) ---


def _write_first_run(tmp_path: Path, text: str) -> Path:
    path = tmp_path / config.FIRST_RUN_FILE
    path.write_text(text, encoding="utf-8")
    return path


def test_first_run_languages_reads_what_the_installer_wrote(tmp_path):
    path = _write_first_run(tmp_path, '{"enabled_languages": ["en", "de"]}')

    assert config.first_run_languages(path) == ["en", "de"]


def test_first_run_languages_keeps_the_installers_order(tmp_path):
    path = _write_first_run(tmp_path, '{"enabled_languages": ["sq", "en"]}')

    assert config.first_run_languages(path) == ["sq", "en"]


def test_first_run_languages_accepts_a_single_language(tmp_path):
    path = _write_first_run(tmp_path, '{"enabled_languages": ["sq"]}')

    assert config.first_run_languages(path) == ["sq"]


def test_first_run_languages_drops_unknown_codes_and_keeps_the_rest(tmp_path):
    path = _write_first_run(tmp_path, '{"enabled_languages": ["en", "klingon", "sq"]}')

    assert config.first_run_languages(path) == ["en", "sq"]


def test_first_run_languages_drops_repeats_and_normalises_case(tmp_path):
    path = _write_first_run(tmp_path, '{"enabled_languages": ["EN", " en ", "De"]}')

    assert config.first_run_languages(path) == ["en", "de"]


def test_first_run_languages_of_a_missing_file_is_empty(tmp_path):
    assert config.first_run_languages(tmp_path / "nothing-here.json") == []


def test_first_run_languages_of_a_directory_is_empty(tmp_path):
    folder = tmp_path / config.FIRST_RUN_FILE
    folder.mkdir()

    assert config.first_run_languages(folder) == []


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json at all",
        "[]",
        '"just a string"',
        "null",
        "{}",
        '{"enabled_languages": null}',
        '{"enabled_languages": "en"}',
        '{"enabled_languages": {}}',
        '{"enabled_languages": []}',
        '{"enabled_languages": [1, 2]}',
        '{"enabled_languages": ["zzz"]}',
        '{"other": ["en"]}',
        '{"enabled_languages": ["en"',
    ],
)
def test_a_malformed_first_run_file_never_raises_and_seeds_nothing(tmp_path, text):
    path = _write_first_run(tmp_path, text)

    assert config.first_run_languages(path) == []


def test_a_first_run_file_with_a_byte_order_mark_still_reads(tmp_path):
    path = tmp_path / config.FIRST_RUN_FILE
    path.write_bytes(b"\xef\xbb\xbf" + b'{"enabled_languages": ["de"]}')

    assert config.first_run_languages(path) == ["de"]


def test_the_known_codes_are_the_ones_the_language_table_names():
    known = config.known_language_codes()

    assert {"en", "de", "sq"} <= known
    assert "klingon" not in known


def test_with_enabled_languages_replaces_the_default_set():
    seeded = config.with_enabled_languages(default_settings(), ["en", "de"])

    assert seeded.general.enabled_languages == ["en", "de"]
    assert default_settings().general.enabled_languages == ["en", "de", "sq"]


def test_with_enabled_languages_changes_nothing_else():
    before = default_settings()
    after = config.with_enabled_languages(before, ["sq"])

    assert dataclasses.replace(
        after, general=dataclasses.replace(after.general, enabled_languages=list(
            before.general.enabled_languages))) == before


def test_with_enabled_languages_of_an_empty_list_leaves_the_settings_alone():
    before = default_settings()

    assert config.with_enabled_languages(before, []) is before


def test_with_enabled_languages_drops_a_locked_language_that_is_no_longer_enabled():
    locked = dataclasses.replace(
        default_settings(),
        general=dataclasses.replace(default_settings().general, language_mode="sq"))

    seeded = config.with_enabled_languages(locked, ["en"])

    assert seeded.general.language_mode == "auto"


def test_with_enabled_languages_keeps_a_locked_language_that_is_still_enabled():
    locked = dataclasses.replace(
        default_settings(),
        general=dataclasses.replace(default_settings().general, language_mode="sq"))

    seeded = config.with_enabled_languages(locked, ["en", "sq"])

    assert seeded.general.language_mode == "sq"


def test_a_seeded_settings_object_survives_a_save_and_a_load(tmp_path):
    path = tmp_path / "settings.json"
    seeded = config.with_enabled_languages(default_settings(), ["en", "de"])

    config.save(seeded, path)
    loaded, notice = load(path)

    assert notice is None
    assert loaded.general.enabled_languages == ["en", "de"]


def test_history_audio_defaults_to_off_with_its_own_limits():
    settings = default_settings()
    assert settings.history.keep_audio is False
    assert settings.history.audio_keep_count == 200
    assert settings.history.audio_keep_mb == 1000


def test_history_audio_settings_round_trip():
    settings = dataclasses.replace(
        default_settings(),
        history=HistorySettings(retention="30d", keep_audio=True, audio_keep_count=50, audio_keep_mb=250),
    )
    again = from_dict(to_dict(settings))
    assert again.history == settings.history


def test_a_settings_file_from_the_current_release_still_loads():
    raw = {"schema_version": 1, "history": {"retention": "30d"}}
    settings = from_dict(migrate(raw))
    assert settings.history.retention == "30d"
    assert settings.history.keep_audio is False
    assert settings.history.audio_keep_count == 200


@pytest.mark.parametrize(
    "raw",
    [
        {"history": {"keep_audio": "yes"}},
        {"history": {"audio_keep_count": 0}},
        {"history": {"audio_keep_count": "many"}},
        {"history": {"audio_keep_mb": 0}},
        {"history": {"audio_keep_mb": None}},
    ],
)
def test_bad_history_audio_values_are_refused(raw):
    with pytest.raises(SettingsError):
        from_dict({"schema_version": 1, **raw})


# --- the writing chords (spec 8.5): compose and edit ---


def test_the_writing_chords_are_empty_by_default():
    general = default_settings().general
    assert general.compose_chord is None
    assert general.edit_chord is None


def test_a_chord_dictates_unless_it_says_otherwise():
    assert default_settings().general.main_chord.mode is ChordMode.DICTATE
    assert Chord(keys=(0x11,)).mode is ChordMode.DICTATE
    assert ChordMode.DICTATE.writes is False
    assert ChordMode.COMPOSE.writes is True
    assert ChordMode.EDIT.writes is True


def test_the_writing_chords_round_trip_with_their_mode():
    raw = to_dict(default_settings())
    raw["general"]["compose_chord"] = {"keys": [0x11, 0x12, 0x57], "mode": "compose"}
    raw["general"]["edit_chord"] = {"keys": [0x11, 0x12, 0x45], "mode": "edit"}

    general = from_dict(raw).general

    assert general.compose_chord == Chord(keys=(0x11, 0x12, 0x57), mode=ChordMode.COMPOSE)
    assert general.edit_chord == Chord(keys=(0x11, 0x12, 0x45), mode=ChordMode.EDIT)
    again = to_dict(from_dict(raw))
    assert again["general"]["compose_chord"] == {"keys": [0x11, 0x12, 0x57], "mode": "compose"}
    assert again["general"]["edit_chord"] == {"keys": [0x11, 0x12, 0x45], "mode": "edit"}


def test_an_unset_writing_chord_serializes_as_null():
    raw = to_dict(default_settings())
    assert raw["general"]["compose_chord"] is None
    assert raw["general"]["edit_chord"] is None


def test_a_writing_chord_without_a_stored_mode_takes_the_mode_of_its_setting():
    raw = to_dict(default_settings())
    raw["general"]["compose_chord"] = {"keys": [0x11, 0x12, 0x57]}
    assert from_dict(raw).general.compose_chord.mode is ChordMode.COMPOSE


def test_a_settings_file_that_predates_compose_mode_reads_unchanged():
    raw = to_dict(default_settings())
    raw["general"].pop("compose_chord")
    raw["general"].pop("edit_chord")
    general = from_dict(raw).general
    assert (general.compose_chord, general.edit_chord) == (None, None)


@pytest.mark.parametrize(
    "raw_chord",
    [
        {"keys": [0x11, 0x12, 0x57], "mode": "dictate"},
        {"keys": [0x11, 0x12, 0x57], "mode": "edit"},
        {"keys": [0x11, 0x12, 0x57], "mode": "shout"},
        {"keys": [0x11, 0x12, 0x57], "mode": 3},
        {"keys": []},
    ],
)
def test_a_compose_chord_with_the_wrong_mode_or_no_keys_is_refused(raw_chord):
    raw = to_dict(default_settings())
    raw["general"]["compose_chord"] = raw_chord
    with pytest.raises(SettingsError):
        from_dict(raw)


def test_the_main_chord_may_not_claim_a_writing_mode():
    raw = to_dict(default_settings())
    raw["general"]["main_chord"] = {"keys": [0x11, 0x5B], "mode": "compose"}
    with pytest.raises(SettingsError, match="main_chord.mode"):
        from_dict(raw)


def test_a_writing_chord_equal_to_the_main_chord_is_rejected():
    raw = to_dict(default_settings())
    raw["general"]["compose_chord"] = {"keys": [0x11, 0x5B], "mode": "compose"}
    with pytest.raises(SettingsError, match=r"main_chord.*compose_chord"):
        from_dict(raw)


def test_a_writing_chord_that_is_a_subset_of_a_language_chord_is_rejected():
    raw = to_dict(default_settings())
    raw["general"]["language_chords"] = [{"keys": [0x11, 0x12, 0x44, 0x45], "language": "de"}]
    raw["general"]["edit_chord"] = {"keys": [0x11, 0x12, 0x44], "mode": "edit"}
    with pytest.raises(SettingsError, match=r"language_chords\[0\].*edit_chord"):
        from_dict(raw)


def test_the_compose_and_edit_chords_may_not_be_the_same_chord():
    raw = to_dict(default_settings())
    raw["general"]["compose_chord"] = {"keys": [0x11, 0x12, 0x57], "mode": "compose"}
    raw["general"]["edit_chord"] = {"keys": [0x12, 0x11, 0x57], "mode": "edit"}
    with pytest.raises(SettingsError, match=r"compose_chord.*edit_chord"):
        from_dict(raw)


def test_writing_chords_that_share_no_key_set_are_accepted():
    raw = to_dict(default_settings())
    raw["general"]["compose_chord"] = {"keys": [0x11, 0x12, 0x57], "mode": "compose"}
    raw["general"]["edit_chord"] = {"keys": [0x11, 0x12, 0x45], "mode": "edit"}
    general = from_dict(raw).general
    assert general.compose_chord is not None
    assert general.edit_chord is not None


def test_a_left_control_writing_chord_still_conflicts_with_the_generic_main_chord():
    raw = to_dict(default_settings())
    raw["general"]["compose_chord"] = {"keys": [0xA2, 0x5B], "mode": "compose"}
    with pytest.raises(SettingsError):
        from_dict(raw)
