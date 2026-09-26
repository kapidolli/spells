from __future__ import annotations

from dataclasses import fields

import pytest
from PySide6 import QtWidgets

from spells.platform.base import Capabilities
from spells.ui.capabilities import UNAVAILABLE, apply_capability, note_for
from spells.ui.icons import TrayIconState
from spells.ui.settings import BUILTIN_PROFILES, WRITE_HOTKEY_HINT, WRITING_HOTKEYS_HINT, RuleEditor
from spells.ui.welcome import STEP_INTRO, STEP_MICROPHONE, WRITE_HOTKEY_NOTE, WRITING_NOTE
from spells.ui.widgets import Card, SectionHeader, SettingRow

from .fake_platform import fake_platform
from .test_ui_diagnostics import make_tab
from .test_ui_settings import make_dialog
from .test_ui_support import qt_app
from .test_ui_tray import make_tray
from .test_ui_welcome import make_page

NOTES = {
    "hold_to_talk": "Dictation hotkeys are not available on this system yet.",
    "esc_cancel": "Esc cannot cancel a dictation on this system yet.",
    "live_typing": "Live typing is not available on this system yet.",
    "app_profiles": (
        "Spells cannot tell which app is in front on this system yet, so every app uses the "
        "Default profile."
    ),
    "window_titles": "Window titles cannot be read on this system yet.",
    "edit_hotkey": "The edit hotkey is not available on this system yet.",
    "clipboard_restore": "On this system the dictated text stays on the clipboard after it is pasted.",
    "autostart": "Starting with the session is not available on this system yet.",
    "self_update": "Updates are installed from the download page on this system.",
    "app_records_chords": "Change this shortcut in your system's keyboard settings.",
    "elevation_check": "Spells cannot check whether an app runs as administrator on this system.",
}


@pytest.fixture(scope="module")
def app():
    return qt_app()


@pytest.fixture
def folder(tmp_path):
    def make(name: str):
        path = tmp_path / name
        path.mkdir()
        return path

    return make


def visible_rows(page: QtWidgets.QWidget) -> list[str]:
    return [row.title_label.text() for row in page.findChildren(SettingRow) if row.isVisibleTo(page)]


def recorder_buttons(dialog) -> list[QtWidgets.QAbstractButton]:
    general = dialog.general
    buttons = [general.main_recorder.change_button, general.compose_recorder.change_button, general.edit_recorder.change_button]
    buttons.extend(row.add_button for row in dialog.languages.language_list.language_rows.values())
    return buttons


def profile_choices(editor: RuleEditor) -> list[str]:
    return [editor.profile.itemText(index) for index in range(editor.profile.count())]


def section_description(page: QtWidgets.QWidget, title: str) -> str:
    headers = [header for header in page.findChildren(SectionHeader) if header.title_label.text() == title]
    assert len(headers) == 1
    return headers[0].description_label.text()


def card_of(widget: QtWidgets.QWidget) -> Card:
    parent = widget.parentWidget()
    while not isinstance(parent, Card):
        parent = parent.parentWidget()
    return parent


def label_texts(page: QtWidgets.QWidget) -> list[str]:
    return [label.text() for label in page.findChildren(QtWidgets.QLabel) if label.text()]


def form_position(editor: RuleEditor, widget: QtWidgets.QWidget) -> tuple:
    layout = editor.layout()
    forms = [layout.itemAt(index).layout() for index in range(layout.count())]
    (form,) = [item for item in forms if isinstance(item, QtWidgets.QFormLayout)]
    return form.getWidgetPosition(widget)


def test_every_flag_has_its_own_plain_note():
    assert {item.name: note_for(item.name) for item in fields(Capabilities)} == NOTES
    assert UNAVAILABLE == "Not available on this system yet."
    assert note_for("no_such_flag") == UNAVAILABLE
    assert not any("\u2014" in text for text in NOTES.values())


def test_the_writing_texts_name_one_hotkey_or_two():
    assert WRITE_HOTKEY_HINT == "It is empty until you record it, and it cannot share a chord with the others."
    assert WRITE_HOTKEY_NOTE == (
        "One more hotkey waits for you in Settings: what you say with it is an instruction, so "
        "\"write an email to Marta asking for the September invoice\" inserts the email."
    )
    assert WRITING_HOTKEYS_HINT == (
        "Both are empty until you record them, and neither can share a chord with the others."
    )
    assert "Two more hotkeys" in WRITING_NOTE


def test_apply_capability_disables_with_the_note_only_when_unavailable(app):
    kept = QtWidgets.QPushButton("Change")
    kept.setToolTip("as today")
    apply_capability(kept, True, "live_typing")
    assert kept.isEnabled()
    assert kept.toolTip() == "as today"
    lost = QtWidgets.QPushButton("Change")
    apply_capability(lost, False, "live_typing")
    assert not lost.isEnabled()
    assert lost.toolTip() == NOTES["live_typing"]


def test_hold_to_talk_note_shows_in_the_welcome_hotkey_step_and_the_tray(app, folder, use_platform):
    use_platform(fake_platform())
    page, *_ = make_page(folder("welcome"))
    assert page.hotkey_note.text() == NOTES["hold_to_talk"]
    assert page.hotkey_note.isVisibleTo(page)
    assert page.steps.widget(STEP_INTRO).isAncestorOf(page.hotkey_note)
    tray, *_ = make_tray(folder("tray"))
    assert NOTES["hold_to_talk"] in tray.tooltip
    assert tray.icon.toolTip() == tray.tooltip
    assert tray.state is TrayIconState.READY


def test_hold_to_talk_note_shows_in_the_settings_dictation_card(app, folder, use_platform):
    use_platform(fake_platform())
    dialog, *_ = make_dialog(folder("settings"))
    general = dialog.general
    assert general.hotkey_note.text() == NOTES["hold_to_talk"]
    assert general.hotkey_note.isVisibleTo(general)
    assert card_of(general.main_recorder).isAncestorOf(general.hotkey_note)
    dialog.close()


def test_live_typing_toggles_are_disabled_with_the_note(app, folder, use_platform):
    use_platform(fake_platform())
    dialog, *_ = make_dialog(folder("settings"))
    for toggle in (dialog.general.live_text, dialog.general.live_text_everywhere):
        assert not toggle.isEnabled()
        assert toggle.toolTip() == NOTES["live_typing"]
    dialog.close()


def test_live_typing_group_has_one_visible_line_with_the_note(app, folder, use_platform):
    use_platform(fake_platform())
    dialog, *_ = make_dialog(folder("settings"))
    general = dialog.general
    note = general.live_typing_note
    assert note.text() == note_for("live_typing")
    assert note.isVisibleTo(general)
    assert card_of(general.live_text).isAncestorOf(note)
    assert label_texts(general).count(note_for("live_typing")) == 1
    dialog.close()


def test_app_profiles_offers_only_default_with_the_note(app, folder, use_platform):
    use_platform(fake_platform())
    dialog, *_ = make_dialog(folder("settings"))
    apps = dialog.apps
    assert apps.profiles_note.text() == NOTES["app_profiles"]
    assert apps.profiles_note.isVisibleTo(apps)
    assert apps.profile_names == ["Default"]
    editor = apps.editor_for_picked_window()
    assert profile_choices(editor) == ["Default"]
    assert not apps.pick_button.isEnabled()
    assert apps.pick_button.toolTip() == NOTES["app_profiles"]
    editor.close()
    dialog.close()


def test_window_titles_disable_the_title_field_with_the_note(app, use_platform):
    use_platform(fake_platform())
    editor = RuleEditor(profile_names=list(BUILTIN_PROFILES))
    assert not editor.title.isEnabled()
    assert editor.title.toolTip() == NOTES["window_titles"]
    assert editor.process.isEnabled()
    editor.close()


def test_window_titles_show_one_line_under_the_title_field(app, use_platform):
    use_platform(fake_platform())
    editor = RuleEditor(profile_names=list(BUILTIN_PROFILES))
    note = editor.title_note
    assert note.text() == note_for("window_titles")
    assert note.isVisibleTo(editor)
    title_row, _role = form_position(editor, editor.title)
    assert form_position(editor, note) == (title_row + 1, QtWidgets.QFormLayout.ItemRole.FieldRole)
    assert label_texts(editor).count(note_for("window_titles")) == 1
    editor.close()


def test_edit_hotkey_row_is_hidden(app, folder, use_platform):
    use_platform(fake_platform())
    dialog, *_ = make_dialog(folder("settings"))
    general = dialog.general
    assert "Edit hotkey" not in visible_rows(general)
    assert not general.edit_recorder.isVisibleTo(general)
    assert "Write hotkey" in visible_rows(general)
    assert section_description(general, "Writing") == WRITE_HOTKEY_HINT
    dialog.close()


def test_without_the_edit_hotkey_the_welcome_names_only_the_write_hotkey(app, tmp_path, use_platform):
    use_platform(fake_platform())
    page, *_ = make_page(tmp_path)
    labels = label_texts(page)
    assert WRITE_HOTKEY_NOTE in labels
    assert WRITING_NOTE not in labels


def test_clipboard_restore_note_shows_under_the_delivery_setting(app, use_platform):
    use_platform(fake_platform())
    editor = RuleEditor(profile_names=list(BUILTIN_PROFILES))
    assert editor.delivery_note.text() == NOTES["clipboard_restore"]
    assert editor.delivery_note.isVisibleTo(editor)
    assert editor.delivery.isEnabled()
    editor.close()


def test_autostart_toggles_are_hidden(app, folder, use_platform):
    use_platform(fake_platform())
    dialog, *_ = make_dialog(folder("settings"))
    general = dialog.general
    assert "Start with Windows" not in visible_rows(general)
    assert not general.autostart.isVisibleTo(general)
    assert "Keep the microphone warm" in visible_rows(general)
    page, *_ = make_page(folder("welcome"))
    microphone = page.steps.widget(STEP_MICROPHONE)
    assert "Start with Windows" not in visible_rows(microphone)
    assert not page.autostart.isVisibleTo(microphone)
    assert "Microphone" in visible_rows(microphone)
    dialog.close()


def test_chord_recorder_buttons_are_disabled_with_the_note(app, folder, use_platform):
    use_platform(fake_platform())
    dialog, *_ = make_dialog(folder("settings"))
    buttons = recorder_buttons(dialog)
    assert len(buttons) > 3
    for button in buttons:
        assert not button.isEnabled()
        assert button.toolTip() == NOTES["app_records_chords"]
    dialog.close()


def test_each_section_with_chord_recorders_has_one_visible_line(app, folder, use_platform):
    use_platform(fake_platform())
    dialog, *_ = make_dialog(folder("settings"))
    general = dialog.general
    languages = dialog.languages
    notes = [general.dictation_chord_note, general.writing_chord_note, languages.chord_note]
    for note in notes:
        assert note.text() == note_for("app_records_chords")
    assert general.dictation_chord_note.isVisibleTo(general)
    assert general.writing_chord_note.isVisibleTo(general)
    assert languages.chord_note.isVisibleTo(languages)
    assert len(languages.language_list.language_rows) > 0
    assert label_texts(general).count(note_for("app_records_chords")) == 2
    assert label_texts(languages).count(note_for("app_records_chords")) == 1
    dialog.close()


def test_diagnostics_system_card_names_the_system_and_its_gaps(app, folder, use_platform):
    use_platform(fake_platform())
    tab, *_ = make_tab(folder("diagnostics"))
    rows = tab.system_card.rows()
    assert rows[0] is tab.system_row
    assert tab.system_row.title_label.text() == "Linux"
    assert [row.title_label.text() for row in rows[1:]] == [NOTES[flag] for flag in Capabilities().missing()]
    assert tab.system_card.isVisibleTo(tab)


def test_with_every_capability_the_controls_stay_as_today(app, folder, use_platform):
    use_platform(fake_platform(capabilities=Capabilities.everything()))
    dialog, *_ = make_dialog(folder("settings"))
    general = dialog.general
    for toggle in (general.live_text, general.live_text_everywhere, general.autostart):
        assert toggle.isEnabled()
        assert toggle.toolTip() == ""
    for button in recorder_buttons(dialog):
        assert button.isEnabled()
        assert button.toolTip() not in NOTES.values()
    rows = visible_rows(general)
    assert "Edit hotkey" in rows
    assert "Start with Windows" in rows
    assert general.hotkey_note is None
    assert general.live_typing_note is None
    assert general.dictation_chord_note is None
    assert general.writing_chord_note is None
    assert dialog.languages.chord_note is None
    for page in (general, dialog.languages):
        texts = label_texts(page)
        assert note_for("live_typing") not in texts
        assert note_for("app_records_chords") not in texts
    assert section_description(general, "Writing") == WRITING_HOTKEYS_HINT
    apps = dialog.apps
    assert apps.profiles_note is None
    assert apps.profile_names == list(BUILTIN_PROFILES)
    assert apps.pick_button.isEnabled()
    assert apps.pick_button.toolTip() == ""
    editor = apps.editor_for_picked_window()
    assert profile_choices(editor) == list(BUILTIN_PROFILES)
    assert editor.title.isEnabled()
    assert editor.title.toolTip() == ""
    assert editor.delivery_note is None
    assert editor.title_note is None
    assert note_for("window_titles") not in label_texts(editor)
    editor.close()
    dialog.close()
    page, *_ = make_page(folder("welcome"))
    assert page.hotkey_note is None
    assert WRITING_NOTE in label_texts(page)
    assert WRITE_HOTKEY_NOTE not in label_texts(page)
    microphone = page.steps.widget(STEP_MICROPHONE)
    assert page.autostart.isVisibleTo(microphone)
    assert "Start with Windows" in visible_rows(microphone)
    tray, *_ = make_tray(folder("tray"))
    assert not any(text in tray.tooltip for text in NOTES.values())
    tab, *_ = make_tab(folder("diagnostics"))
    assert [row.title_label.text() for row in tab.system_card.rows()] == ["Linux"]
