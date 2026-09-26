"""spells.ui.settings: the tabs of spec 14.4, every change through ConfigStore.update().

The dialog is driven programmatically; the message boxes, the confirm prompt, the device
list, the RegisterHotKey probe and the level meter's recorder are injected fakes.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from spells.audio import AudioDevice, MicError
from spells.config import ProfileRule, Replacement, Snippet
from spells.models import Chord, ChordMode, DeliveryMethod
from spells.ui.settings import (
    BUILTIN_PROFILES,
    MIC_DEVICE_HINT,
    NO_WRITING_MODEL,
    TAB_NAMES,
    WRITING_TONE_HINT,
    ChordCaptureDialog,
    HotkeyRecorder,
    RuleEditor,
    SettingsDialog,
    mic_device_hint,
    probe_arguments,
)
from spells.ui.widgets import SegmentedControl, ToggleSwitch

from .fake_recorder import FakeRecorder
from .test_ui_support import (
    SELECTION,
    FakeEngines,
    FakeHistory,
    FakeHotkey,
    FakePipeline,
    Messages,
    flush,
    history_entry,
    make_config,
    qt_app,
)

CTRL, ALT, SHIFT, LWIN, LCONTROL = 0x11, 0x12, 0x10, 0x5B, 0xA2
KEY_D, KEY_E = 0x44, 0x45
DEVICES = [
    AudioDevice(index=1, name="Headset Microphone", is_default=True),
    AudioDevice(index=3, name="Webcam Mic", is_default=False),
]


@pytest.fixture(scope="module")
def app():
    return qt_app()


class FakeProbe:
    def __init__(self, result: bool | Exception = True) -> None:
        self.result = result
        self.calls: list[tuple[int, int]] = []

    def __call__(self, modifiers: int, vk: int) -> bool:
        self.calls.append((modifiers, vk))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_dialog(tmp_path: Path, *, probe=None, devices=None, history=None, entries=None):
    config = make_config(tmp_path)
    messages = Messages()
    probe = probe if probe is not None else FakeProbe(True)
    hotkey = FakeHotkey()
    store = history if history is not None else FakeHistory(entries or [])
    dialog = SettingsDialog(
        config=config,
        engines=FakeEngines(),
        pipeline=FakePipeline(),
        hotkey=hotkey,
        history=store,
        gpu_selection=SELECTION,
        log_dir=tmp_path / "logs",
        devices=(lambda: list(DEVICES)) if devices is None else devices,
        probe=probe,
        notify=messages,
        confirm=lambda title, text: True,
        recorder_factory=FakeRecorder,
    )
    return dialog, config, messages, probe, hotkey, store


# Dialog shape ---------------------------------------------------------------------------------


def test_pages_are_the_spec_sections_plus_languages_and_about(app, tmp_path):
    dialog, *_ = make_dialog(tmp_path)
    assert dialog.page_labels() == [
        "General",
        "Languages",
        "Writing",
        "Apps",
        "Vocabulary",
        "History",
        "Upload",
        "Diagnostics",
        "About",
    ]
    assert dialog.page_names() == list(TAB_NAMES)
    dialog.show_tab("upload")
    assert dialog.tabs.currentWidget() is dialog.upload
    dialog.show_tab("history")
    assert dialog.tabs.currentWidget() is dialog.history
    assert dialog.nav.current() == "history"
    dialog.show_tab("diagnostics")
    assert dialog.tabs.currentWidget() is dialog.diagnostics
    dialog.show_tab("nonsense")
    assert dialog.current_page() == "general"
    dialog.close()


def test_clicking_the_navigation_switches_the_page(app, tmp_path):
    dialog, *_ = make_dialog(tmp_path)
    dialog.nav.items["languages"].click()
    assert dialog.current_page() == "languages"
    assert dialog.pages.currentWidget() is dialog.languages
    dialog.nav.items["about"].click()
    assert dialog.pages.currentWidget() is dialog.about
    dialog.close()


def test_general_uses_toggle_switches(app, tmp_path):
    dialog, *_ = make_dialog(tmp_path)
    for switch in (
        dialog.general.autostart,
        dialog.general.keep_mic_warm,
        dialog.general.sounds,
        dialog.general.live_text,
        dialog.general.live_text_everywhere,
        dialog.cleanup.enabled,
    ):
        assert isinstance(switch, ToggleSwitch)
    assert isinstance(dialog.history.retention, SegmentedControl)
    dialog.close()


# General tab ----------------------------------------------------------------------------------


def test_general_checkboxes_round_trip_through_the_store(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    general = dialog.general
    before = config.settings
    general.autostart.setChecked(False)
    assert config.settings.general.autostart is False
    general.keep_mic_warm.setChecked(True)
    assert config.settings.general.keep_mic_warm is True
    general.sounds.setChecked(False)
    assert config.settings.general.sounds is False
    general.idle_unload.setValue(15)
    general.idle_unload.editingFinished.emit()
    assert config.settings.general.idle_unload_minutes == 15
    assert config.settings is not before
    assert before.general.autostart is True
    dialog.close()


def test_live_text_rows_round_trip_and_gate_each_other(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    general = dialog.general
    assert general.live_text.isChecked() is True
    assert general.live_text_everywhere.isChecked() is False
    general.live_text_everywhere.setChecked(True)
    assert config.settings.general.live_text_everywhere is True
    general.live_text.setChecked(False)
    assert config.settings.general.live_text is False
    assert general.live_text_everywhere.isEnabled() is False
    dialog.close()


def test_mic_picker_lists_the_devices_and_stores_the_name(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    picker = dialog.general.mic_picker
    assert [picker.itemText(i) for i in range(picker.count())] == [
        "System default (Headset Microphone)",
        "Headset Microphone",
        "Webcam Mic",
    ]
    picker.setCurrentIndex(2)
    assert config.settings.general.mic_device == "Webcam Mic"
    picker.setCurrentIndex(0)
    assert config.settings.general.mic_device is None
    dialog.close()


def test_the_mic_row_names_the_capture_path(app, tmp_path):
    assert mic_device_hint(lambda: "Windows WASAPI") == (
        "Spells records from this microphone through Windows WASAPI, at its own sample rate."
    )
    assert mic_device_hint(lambda: "") == MIC_DEVICE_HINT


def test_the_mic_row_falls_back_when_the_host_api_cannot_be_read(app, tmp_path):
    def broken() -> str:
        raise MicError("unknown", "PortAudio is unhappy")

    assert mic_device_hint(broken) == MIC_DEVICE_HINT


def test_mic_picker_survives_an_enumeration_error(app, tmp_path):
    def broken():
        raise MicError("unknown", "PortAudio is unhappy")

    dialog, _config, messages, *_ = make_dialog(tmp_path, devices=broken)
    picker = dialog.general.mic_picker
    assert picker.count() == 1
    assert any("PortAudio is unhappy" in text for text in messages.texts)
    dialog.close()


def test_enabled_languages_round_trip_and_keep_the_mode_valid(app, tmp_path):
    dialog, config, messages, *_ = make_dialog(tmp_path)
    general = dialog.languages
    general.set_language_enabled("fr", True)
    assert config.settings.general.enabled_languages == ["en", "de", "sq", "fr"]
    config.update(lambda s: replace(s, general=replace(s.general, language_mode="de")))
    dialog.apply_settings(config.settings)
    general.set_language_enabled("de", False)
    assert config.settings.general.enabled_languages == ["en", "sq", "fr"]
    assert config.settings.general.language_mode == "auto"
    for code in ("en", "sq", "fr"):
        general.set_language_enabled(code, False)
    assert config.settings.general.enabled_languages == ["fr"]
    assert any("at least one" in text.lower() for text in messages.texts)
    dialog.close()


def test_language_chords_are_added_and_removed(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    languages = dialog.languages
    assert languages.add_language_chord("de", (CTRL, ALT, KEY_D))
    assert config.settings.general.language_chords == [Chord(keys=(CTRL, ALT, KEY_D), language="de")]
    assert languages.language_list.language_rows["de"].hotkey_labels() == ["Ctrl+Alt+D"]
    languages.remove_language_chord(0)
    assert config.settings.general.language_chords == []
    assert languages.language_list.language_rows["de"].hotkey_labels() == []
    dialog.close()


def test_language_chord_that_is_a_superset_of_the_main_chord_is_refused(app, tmp_path):
    dialog, config, messages, *_ = make_dialog(tmp_path)
    assert not dialog.languages.add_language_chord("de", (CTRL, LWIN, KEY_D, KEY_E))
    assert config.settings.general.language_chords == []
    assert any("main_chord" in text and "language_chords[0]" in text for text in messages.texts)
    dialog.close()


def test_welcome_button_asks_for_the_welcome_page(app, tmp_path):
    dialog, *_ = make_dialog(tmp_path)
    asked: list[int] = []
    dialog.welcome_requested.connect(lambda: asked.append(1))
    dialog.general.welcome_button.click()
    assert asked == [1]
    dialog.close()


# Hotkey recorder -------------------------------------------------------------------------------


def test_recorder_accepts_a_free_chord_and_folds_side_modifiers(app, tmp_path):
    dialog, config, _messages, probe, *_ = make_dialog(tmp_path)
    recorder = dialog.general.main_recorder
    assert recorder.accept_chord((LCONTROL, ALT, KEY_D))
    assert config.settings.general.main_chord == Chord(keys=(CTRL, ALT, KEY_D))
    assert probe.calls == [(0x0002 | 0x0001, KEY_D)]
    assert "Ctrl+Alt+D" in recorder.label.text()
    dialog.close()


def test_recorder_refuses_a_chord_another_app_owns(app, tmp_path):
    dialog, config, messages, _probe, *_ = make_dialog(tmp_path, probe=FakeProbe(False))
    before = config.settings.general.main_chord
    assert not dialog.general.main_recorder.accept_chord((CTRL, SHIFT, KEY_E))
    assert config.settings.general.main_chord == before
    assert any("another app" in text.lower() for text in messages.texts)
    dialog.close()


def test_recorder_shows_a_probe_failure_as_a_warning_and_keeps_the_chord(app, tmp_path):
    dialog, config, messages, *_ = make_dialog(tmp_path, probe=FakeProbe(OSError(1400, "bad hwnd")))
    assert dialog.general.main_recorder.accept_chord((CTRL, SHIFT, KEY_E))
    assert config.settings.general.main_chord == Chord(keys=(CTRL, SHIFT, KEY_E))
    assert any("bad hwnd" in text for text in messages.texts)
    assert messages.shown[-1][0] == "warning"
    dialog.close()


def test_modifier_only_chords_are_not_probed(app, tmp_path):
    dialog, config, _messages, probe, *_ = make_dialog(tmp_path)
    assert dialog.general.main_recorder.accept_chord((CTRL, SHIFT))
    assert probe.calls == []
    assert config.settings.general.main_chord == Chord(keys=(CTRL, SHIFT))
    dialog.close()


def test_recorder_surfaces_the_subset_rule_from_config(app, tmp_path):
    dialog, config, messages, *_ = make_dialog(tmp_path)
    de = Chord(keys=(CTRL, ALT, KEY_D), language="de")
    config.update(lambda s: replace(s, general=replace(s.general, language_chords=[de])))
    dialog.apply_settings(config.settings)
    assert not dialog.general.main_recorder.accept_chord((CTRL, ALT, KEY_D))
    assert config.settings.general.main_chord == Chord(keys=(CTRL, LWIN))
    text = messages.texts[-1]
    assert "general.main_chord" in text and "general.language_chords[0]" in text
    assert "equal" in text
    assert not dialog.general.main_recorder.accept_chord((ALT,))
    assert "subset" in messages.texts[-1]
    assert config.settings.general.main_chord == Chord(keys=(CTRL, LWIN))
    dialog.close()


def test_probe_arguments_map_generic_modifiers_to_mod_flags():
    assert probe_arguments((CTRL, ALT, SHIFT, LWIN, KEY_D)) == (0x0002 | 0x0001 | 0x0004 | 0x0008, KEY_D)
    assert probe_arguments((LCONTROL, KEY_D)) == (0x0002, KEY_D)
    assert probe_arguments((CTRL, LWIN)) is None
    assert probe_arguments((CTRL, KEY_D, KEY_E)) is None


def key_event(kind, vk, text=""):
    return QtGui.QKeyEvent(kind, 0, QtCore.Qt.KeyboardModifier.NoModifier, 0, vk, 0, text)


def test_capture_dialog_reads_the_chord_from_real_key_events(app, tmp_path):
    hotkey = FakeHotkey(chords=(Chord(keys=(CTRL, LWIN)),))
    dialog = ChordCaptureDialog(hotkey=hotkey, active_chords=(Chord(keys=(CTRL, LWIN)),))
    dialog.show()
    flush(app)
    assert hotkey.pushed[-1] == ()
    press = QtCore.QEvent.Type.KeyPress
    release = QtCore.QEvent.Type.KeyRelease
    QtWidgets.QApplication.sendEvent(dialog, key_event(press, LCONTROL))
    QtWidgets.QApplication.sendEvent(dialog, key_event(press, KEY_D, "d"))
    assert dialog.chord is None
    QtWidgets.QApplication.sendEvent(dialog, key_event(release, KEY_D, "d"))
    assert dialog.chord == (CTRL, KEY_D)
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    flush(app)
    assert hotkey.pushed[-1] == (Chord(keys=(CTRL, LWIN)),)


def test_capture_dialog_escape_cancels_and_restores_the_chords(app):
    hotkey = FakeHotkey(chords=(Chord(keys=(CTRL, LWIN)),))
    dialog = ChordCaptureDialog(hotkey=hotkey, active_chords=(Chord(keys=(CTRL, LWIN)),))
    dialog.show()
    flush(app)
    QtWidgets.QApplication.sendEvent(dialog, key_event(QtCore.QEvent.Type.KeyPress, 0x1B))
    assert dialog.chord is None
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    flush(app)
    assert hotkey.pushed[-1] == (Chord(keys=(CTRL, LWIN)),)


def test_standalone_recorder_uses_its_setter(app, tmp_path):
    config = make_config(tmp_path)
    messages = Messages()
    recorder = HotkeyRecorder(
        config=config,
        hotkey=FakeHotkey(),
        probe=FakeProbe(True),
        notify=messages,
        chord_of=lambda s: s.general.main_chord,
        with_chord=lambda s, keys: replace(s, general=replace(s.general, main_chord=Chord(keys=keys))),
    )
    assert recorder.accept_chord((CTRL, KEY_E))
    assert config.settings.general.main_chord == Chord(keys=(CTRL, KEY_E))


# Cleanup tab --------------------------------------------------------------------------------


def test_cleanup_switch_and_timeout(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    cleanup = dialog.cleanup
    cleanup.enabled.setChecked(False)
    assert config.settings.cleanup.enabled is False
    cleanup.timeout.setValue(1800)
    cleanup.timeout.editingFinished.emit()
    assert config.settings.cleanup.timeout_ms == 1800
    dialog.close()


def test_tones_are_edited_per_profile_name(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    cleanup = dialog.cleanup
    assert set(cleanup.tone_edits) == set(config.settings.cleanup.tones)
    cleanup.tone_edits["Chat"].setText("very casual")
    cleanup.tone_edits["Chat"].editingFinished.emit()
    assert config.settings.cleanup.tones["Chat"] == "very casual"
    assert config.settings.cleanup.tones["Code"] == "minimal rewriting; never alter identifiers, paths, or symbols"
    dialog.close()


def test_filler_and_correction_lists_per_language(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    cleanup = dialog.cleanup
    codes = [cleanup.language.itemData(i) for i in range(cleanup.language.count())]
    assert codes == ["en", "de", "sq"]
    cleanup.language.setCurrentIndex(1)
    cleanup.fillers.setPlainText("ähm\näh\n\n")
    cleanup.corrections.setPlainText("ich meine")
    cleanup.save_lists()
    assert config.settings.cleanup.fillers["de"] == ["ähm", "äh"]
    assert config.settings.cleanup.corrections["de"] == ["ich meine"]
    assert config.settings.cleanup.fillers["en"] == dialog.cleanup.original_fillers["en"]
    dialog.close()


# Apps tab -----------------------------------------------------------------------------------


def test_rule_editor_has_no_tone_and_builds_a_rule(app):
    editor = RuleEditor(profile_names=list(BUILTIN_PROFILES))
    assert not hasattr(editor, "tone")
    editor.name.setText("Slack")
    editor.process.setText("slack.exe, teams.exe")
    editor.title.setText("Direct message")
    editor.profile.setCurrentText("Chat")
    editor.delivery.setCurrentIndex(editor.delivery.findData(DeliveryMethod.TYPE))
    rule = editor.rule()
    assert rule.name == "Slack"
    assert rule.match_process == ["slack.exe", "teams.exe"]
    assert rule.match_title == ["Direct message"]
    assert rule.profile.name == "Chat"
    assert rule.profile.delivery is DeliveryMethod.TYPE
    assert rule.profile.tone == BUILTIN_PROFILES["Chat"].tone
    assert rule.profile.cleanup is BUILTIN_PROFILES["Chat"].cleanup
    editor.close()


def test_apps_tab_adds_edits_and_removes_rules(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    apps = dialog.apps
    rule = ProfileRule(
        name="Slack", match_process=["slack.exe"], match_title=[], profile=BUILTIN_PROFILES["Chat"]
    )
    apps.apply_rule(None, rule)
    assert config.settings.profiles == [rule]
    assert apps.table.rowCount() == 1
    edited = replace(rule, match_title=["#general"])
    apps.apply_rule(0, edited)
    assert config.settings.profiles == [edited]
    apps.remove_rule(0)
    assert config.settings.profiles == []
    dialog.close()


def test_apps_tab_picks_a_running_window(app, tmp_path):
    dialog, *_ = make_dialog(tmp_path)
    dialog.apps.window_picker = lambda: ("Code.exe", "settings.py - spells")
    editor = dialog.apps.editor_for_picked_window()
    assert editor.process.text() == "Code.exe"
    assert editor.title.text() == "settings.py - spells"
    editor.close()
    dialog.close()


# Vocabulary tab -------------------------------------------------------------------------------


def test_vocabulary_terms_replacements_and_snippets(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    vocab = dialog.vocabulary
    vocab.add_term("Kubernetes")
    assert [t.text for t in config.settings.vocabulary.terms] == ["Kubernetes"]
    assert config.settings.vocabulary.terms[0].edited_at > 0
    vocab.add_replacement("brb", "be right back")
    assert config.settings.vocabulary.replacements == [Replacement("brb", "be right back")]
    vocab.add_snippet("sig", "Kind regards, Alex")
    assert config.settings.vocabulary.snippets == [Snippet("sig", "Kind regards, Alex")]
    vocab.remove_term(0)
    vocab.remove_replacement(0)
    vocab.remove_snippet(0)
    assert config.settings.vocabulary.terms == []
    assert config.settings.vocabulary.replacements == []
    assert config.settings.vocabulary.snippets == []
    dialog.close()


def test_vocabulary_export_and_import(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    vocab = dialog.vocabulary
    vocab.add_term("Gitea")
    vocab.add_replacement("k8s", "Kubernetes")
    target = tmp_path / "vocab.json"
    vocab.export_to(target)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["terms"][0]["text"] == "Gitea"
    vocab.remove_term(0)
    vocab.remove_replacement(0)
    vocab.import_from(target)
    assert [t.text for t in config.settings.vocabulary.terms] == ["Gitea"]
    assert config.settings.vocabulary.replacements == [Replacement("k8s", "Kubernetes")]
    dialog.close()


def test_vocabulary_import_rejects_a_bad_file(app, tmp_path):
    dialog, config, messages, *_ = make_dialog(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    dialog.vocabulary.import_from(bad)
    assert config.settings.vocabulary == replace(config.settings.vocabulary)
    assert messages.shown[-1][0] == "warning"
    dialog.close()


# History tab -------------------------------------------------------------------------------


def test_history_tab_lists_searches_copies_and_clears(app, tmp_path):
    entries = [history_entry(1, "hello there", "Hello there."), history_entry(2, "zwei", "Zwei.")]
    dialog, _config, _messages, _probe, _hotkey, store = make_dialog(tmp_path, entries=entries)
    history = dialog.history
    history.refresh()
    assert history.table.rowCount() == 2
    history.table.selectRow(0)
    assert history.raw.toPlainText() == "hello there"
    assert history.cleaned.toPlainText() == "Hello there."
    history.copy_button.click()
    assert QtWidgets.QApplication.clipboard().text() == "Hello there."
    history.search.setText("zwei")
    history.run_search()
    assert history.table.rowCount() == 1
    assert ("search", "zwei", 100) in store.calls
    history.clear_button.click()
    assert ("clear",) in store.calls
    assert history.table.rowCount() == 0
    dialog.close()


def test_history_retention_goes_through_the_store(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    retention = dialog.history.retention
    codes = [retention.itemData(i) for i in range(retention.count())]
    assert codes == ["100", "7d", "30d", "off"]
    retention.setCurrentIndex(3)
    assert config.settings.history.retention == "off"
    dialog.close()


# Refresh from a foreign update ------------------------------------------------------------


def test_dialog_refreshes_from_a_settings_change_without_writing_back(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    writes: list[int] = []
    config.subscribe(lambda s: writes.append(1))
    config.update(lambda s: replace(s, general=replace(s.general, keep_mic_warm=True, sounds=False)))
    dialog.apply_settings(config.settings)
    assert dialog.general.keep_mic_warm.isChecked()
    assert not dialog.general.sounds.isChecked()
    assert writes == [1]
    dialog.close()


def test_every_setting_edit_replaces_the_frozen_snapshot(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    snapshots = [config.settings]
    config.subscribe(snapshots.append)
    dialog.general.autostart.setChecked(False)
    dialog.cleanup.enabled.setChecked(False)
    dialog.vocabulary.add_term("x")
    assert len(snapshots) == 4
    assert len({id(s) for s in snapshots}) == 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshots[-1].general = None
    dialog.close()


def test_capture_dialog_restores_a_paused_hook(app):
    hotkey = FakeHotkey(chords=())
    dialog = ChordCaptureDialog(hotkey=hotkey, active_chords=(Chord(keys=(CTRL, LWIN)),))
    dialog.show()
    flush(app)
    assert hotkey.pushed[-1] == ()
    QtWidgets.QApplication.sendEvent(dialog, key_event(QtCore.QEvent.Type.KeyPress, KEY_D, "d"))
    QtWidgets.QApplication.sendEvent(dialog, key_event(QtCore.QEvent.Type.KeyRelease, KEY_D, "d"))
    flush(app)
    assert dialog.chord == (KEY_D,)
    assert hotkey.chords == ()
    assert hotkey.pushed == [(), ()]


def test_capture_dialog_snapshots_the_running_chords(app):
    running = (Chord(keys=(CTRL, ALT, KEY_D)), Chord(keys=(CTRL, ALT, KEY_E), language="de"))
    hotkey = FakeHotkey(chords=running)
    dialog = ChordCaptureDialog(hotkey=hotkey, active_chords=(Chord(keys=(CTRL, LWIN)),))
    dialog.show()
    flush(app)
    assert hotkey.chords == ()
    dialog.reject()
    flush(app)
    assert hotkey.chords == running


def test_capture_dialog_falls_back_to_the_given_chords_without_the_property(app):
    class BareHotkey:
        def __init__(self) -> None:
            self.pushed: list[tuple] = []

        def update_chords(self, chords) -> None:
            self.pushed.append(tuple(chords))

    hotkey = BareHotkey()
    given = (Chord(keys=(CTRL, LWIN)),)
    dialog = ChordCaptureDialog(hotkey=hotkey, active_chords=given)
    dialog.show()
    flush(app)
    dialog.reject()
    flush(app)
    assert hotkey.pushed == [(), given]


# The writing hotkeys and the writing model (spec 8.5, 14.4) ------------------------------------

KEY_W = 0x57


def test_both_writing_hotkeys_start_empty(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    assert config.settings.general.compose_chord is None
    assert config.settings.general.edit_chord is None
    assert dialog.general.compose_recorder.label.text() == ""
    assert dialog.general.edit_recorder.label.text() == ""
    dialog.close()


def test_recording_a_write_hotkey_stores_it_in_compose_mode(app, tmp_path):
    dialog, config, _messages, probe, *_ = make_dialog(tmp_path)
    assert dialog.general.compose_recorder.accept_chord((LCONTROL, ALT, KEY_W))
    assert config.settings.general.compose_chord == Chord(
        keys=(CTRL, ALT, KEY_W), mode=ChordMode.COMPOSE
    )
    assert probe.calls == [(0x0002 | 0x0001, KEY_W)]
    assert "Ctrl+Alt+W" in dialog.general.compose_recorder.label.text()
    dialog.close()


def test_recording_an_edit_hotkey_stores_it_in_edit_mode(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    assert dialog.general.edit_recorder.accept_chord((CTRL, ALT, KEY_E))
    assert config.settings.general.edit_chord == Chord(
        keys=(CTRL, ALT, KEY_E), mode=ChordMode.EDIT
    )
    dialog.close()


def test_a_writing_hotkey_that_clashes_is_refused_with_the_same_rule(app, tmp_path):
    dialog, config, messages, *_ = make_dialog(tmp_path)
    main = config.settings.general.main_chord
    assert not dialog.general.compose_recorder.accept_chord(main.keys)
    assert config.settings.general.compose_chord is None
    assert any("conflicts with" in text for text in messages.texts)
    dialog.close()


def test_the_two_writing_hotkeys_may_not_be_the_same_chord(app, tmp_path):
    dialog, config, messages, *_ = make_dialog(tmp_path)
    assert dialog.general.compose_recorder.accept_chord((CTRL, ALT, KEY_W))
    assert not dialog.general.edit_recorder.accept_chord((CTRL, ALT, KEY_W))
    assert config.settings.general.edit_chord is None
    assert any("conflicts with" in text for text in messages.texts)
    dialog.close()


def test_a_writing_hotkey_another_app_owns_is_refused(app, tmp_path):
    dialog, config, messages, *_ = make_dialog(tmp_path, probe=FakeProbe(False))
    assert not dialog.general.compose_recorder.accept_chord((CTRL, SHIFT, KEY_W))
    assert config.settings.general.compose_chord is None
    assert any("another app" in text.lower() for text in messages.texts)
    dialog.close()


def test_the_writing_page_names_the_model_that_writes(app, tmp_path):
    dialog, *_ = make_dialog(tmp_path)
    card = dialog.cleanup.writing
    assert card.model_row is not None
    if card.choice is not None:
        assert card.model_row.title_label.text() == card.choice.display_name
        assert card.choice.reason
    else:
        assert card.model_row.title_label.text() == "No writing model"
    dialog.close()


def test_the_writing_card_says_where_composing_takes_its_tone_from(app, tmp_path):
    dialog, *_ = make_dialog(tmp_path)
    labels = [
        widget.text()
        for widget in dialog.cleanup.writing.findChildren(QtWidgets.QLabel)
        if widget.text()
    ]
    assert WRITING_TONE_HINT in labels
    dialog.close()


def test_the_writing_card_follows_the_enabled_languages(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    before = dialog.cleanup.writing._key
    config.update(lambda s: replace(s, general=replace(s.general, enabled_languages=["en"])))
    dialog.cleanup.apply_settings(config.settings)
    assert dialog.cleanup.writing._key != before
    assert dialog.cleanup.writing._key[0] == ("en",)
    dialog.close()


def test_the_writing_page_still_edits_the_cleanup_settings(app, tmp_path):
    dialog, config, *_ = make_dialog(tmp_path)
    dialog.cleanup.timeout.setValue(1234)
    dialog.cleanup._on_timeout()
    assert config.settings.cleanup.timeout_ms == 1234
    assert NO_WRITING_MODEL
    dialog.close()
