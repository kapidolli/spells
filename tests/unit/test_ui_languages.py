"""spells.ui.languages: the language chips, the search, per-language hotkeys and the models card."""

from __future__ import annotations

from dataclasses import replace

import pytest
from PySide6 import QtCore

from spells import modelcatalog
from spells.gpu import NO_GPU, GpuDevice, GpuSelection
from spells.modelcatalog import Hardware, ModelChoice, ModelKind, Selection
from spells.models import Chord
from spells.ui.languages import (
    POPULAR_LANGUAGES,
    LanguageAdder,
    LanguagesPage,
    ModelsCard,
    hardware_for,
)

from .test_ui_support import SELECTION, FakeHotkey, Messages, flush, make_config, qt_app

CTRL, ALT, KEY_D = 0x11, 0x12, 0x44


@pytest.fixture(scope="module")
def app():
    return qt_app()


class SpyStore:
    """Counts the updates that go through ConfigStore.update()."""

    def __init__(self, config) -> None:
        self.updates = 0
        original = config.update

        def update(mutator):
            self.updates += 1
            return original(mutator)

        config.update = update


class FakeSelect:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], Hardware]] = []
        self.cleanup = True
        self.uncovered: set[str] = set()
        self.error: Exception | None = None

    def __call__(self, enabled_languages, hardware, installed_ids=None, catalog=None) -> Selection:
        self.calls.append((tuple(enabled_languages), hardware))
        if self.error is not None:
            raise self.error
        covered = tuple(code for code in enabled_languages if code not in self.uncovered)
        asr = ModelChoice(
            kind=ModelKind.ASR,
            model_id="asr-model",
            display_name=f"Speech model for {len(covered)}",
            languages=covered,
            reason="Chosen by the fake.",
            installed=True,
        )
        cleanup = None
        if self.cleanup:
            cleanup = ModelChoice(
                kind=ModelKind.CLEANUP,
                model_id="cleanup-model",
                display_name="Cleanup model",
                languages=tuple(enabled_languages),
                reason="Cleans everything.",
                installed=False,
            )
        return Selection(hardware=hardware, asr=(asr,), cleanup=cleanup, notes=("A note from the catalog.",))


@pytest.fixture
def fake_select(monkeypatch):
    fake = FakeSelect()
    monkeypatch.setattr(modelcatalog, "select_models", fake)
    return fake


def make_page(tmp_path, *, gpu=SELECTION, probe=None):
    config = make_config(tmp_path)
    messages = Messages()
    page = LanguagesPage(
        config=config,
        hotkey=FakeHotkey(),
        notify=messages,
        probe=probe or (lambda _modifiers, _key: True),
        gpu_selection=gpu,
    )
    return page, config, messages


# The enabled languages ------------------------------------------------------------------------


def test_chips_show_the_enabled_languages(app, tmp_path, fake_select):
    page, _config, _messages = make_page(tmp_path)
    assert page.language_list.codes() == ["en", "de", "sq"]
    chip = page.language_list.language_rows["de"].chip
    assert chip.text() == "German"
    assert chip.code_label.text() == "DE"
    page.close()


def test_the_chip_remove_button_removes_through_the_store(app, tmp_path, fake_select):
    page, config, _messages = make_page(tmp_path)
    spy = SpyStore(config)
    page.language_list.language_rows["sq"].chip.remove_button.click()
    flush(app)
    assert config.settings.general.enabled_languages == ["en", "de"]
    assert spy.updates == 1
    assert page.language_list.codes() == ["en", "de"]
    page.close()


def test_the_last_language_stays(app, tmp_path, fake_select):
    page, config, messages = make_page(tmp_path)
    page.set_language_enabled("de", False)
    page.set_language_enabled("sq", False)
    page.set_language_enabled("en", False)
    assert config.settings.general.enabled_languages == ["en"]
    assert any("at least one" in text.lower() for text in messages.texts)
    page.close()


def test_removing_the_locked_language_returns_the_mode_to_auto(app, tmp_path, fake_select):
    page, config, _messages = make_page(tmp_path)
    config.update(lambda s: replace(s, general=replace(s.general, language_mode="sq")))
    page.set_language_enabled("sq", False)
    assert config.settings.general.language_mode == "auto"
    page.close()


# Adding a language ------------------------------------------------------------------------------


def test_popular_languages_come_first_and_search_ranks_prefixes(app):
    adder = LanguageAdder()
    empty = [code for code, _name in adder.matches("")]
    assert tuple(empty[:3]) == POPULAR_LANGUAGES
    assert len(empty) == len(set(empty)) == 100
    assert adder.matches("ger")[0][0] == "de"
    assert "fr" in [code for code, _name in adder.matches("fr")]
    assert adder.matches("zzzz") == []
    adder.close()


def test_a_search_result_click_adds_the_language(app, tmp_path, fake_select):
    page, config, _messages = make_page(tmp_path)
    spy = SpyStore(config)
    page.adder.search.setText("French")
    assert page.adder.visible_codes()[0] == "fr"
    page.adder.results.itemClicked.emit(page.adder.results.item(0))
    assert config.settings.general.enabled_languages == ["en", "de", "sq", "fr"]
    assert spy.updates == 1
    assert "fr" in page.language_list.codes()
    page.adder.results.itemClicked.emit(page.adder.results.item(0))
    assert spy.updates == 1
    page.close()


def test_enter_adds_the_first_language_not_yet_added(app, tmp_path, fake_select):
    page, config, _messages = make_page(tmp_path)
    page.adder.search.setText("a")
    page.adder.search.returnPressed.emit()
    added = config.settings.general.enabled_languages[-1]
    assert added not in ("en", "de", "sq")
    assert page.adder.search.text() == ""
    page.close()


def test_suggestions_offer_only_missing_popular_languages(app, tmp_path, fake_select):
    page, config, _messages = make_page(tmp_path)
    page.show()
    flush(app)
    assert not page.adder.suggestions.isVisible()
    page.set_language_enabled("de", False)
    flush(app)
    assert page.adder.suggestions.isVisible()
    assert page.adder.suggestion_buttons["de"].isVisible()
    assert not page.adder.suggestion_buttons["en"].isVisible()
    page.adder.suggestion_buttons["de"].click()
    assert "de" in config.settings.general.enabled_languages
    page.close()


# Per-language hotkeys ---------------------------------------------------------------------------


def test_language_hotkeys_sit_beside_their_chip(app, tmp_path, fake_select):
    page, config, _messages = make_page(tmp_path)
    assert page.add_language_chord("de", (CTRL, ALT, KEY_D))
    row = page.language_list.language_rows["de"]
    assert row.hotkey_labels() == ["Ctrl+Alt+D"]
    assert page.language_list.language_rows["en"].hotkey_labels() == []
    page.language_list.remove_hotkey.emit(0)
    assert config.settings.general.language_chords == []
    page.close()


def test_a_taken_hotkey_is_refused_before_the_store(app, tmp_path, fake_select):
    page, config, messages = make_page(tmp_path, probe=lambda _m, _k: False)
    spy = SpyStore(config)
    assert not page.add_language_chord("de", (CTRL, ALT, KEY_D))
    assert spy.updates == 0
    assert any("another app" in text for text in messages.texts)
    page.close()


def test_a_hotkey_for_a_removed_language_stays_visible(app, tmp_path, fake_select):
    page, config, _messages = make_page(tmp_path)
    chord = Chord(keys=(CTRL, ALT, KEY_D), language="de")
    config.update(lambda s: replace(s, general=replace(s.general, language_chords=[chord], enabled_languages=["en"])))
    page.apply_settings(config.settings)
    assert page.language_list.codes() == ["en", "de"]
    orphan = page.language_list.language_rows["de"]
    assert orphan.chip.remove_button is None
    assert orphan.hotkey_labels() == ["Ctrl+Alt+D"]
    page.close()


# Models -----------------------------------------------------------------------------------------


def test_models_card_asks_the_catalog_for_the_languages_and_hardware(app, tmp_path, fake_select):
    page, _config, _messages = make_page(tmp_path)
    assert fake_select.calls == [(("en", "de", "sq"), Hardware.GPU)]
    assert page.models.names() == ["Speech model for 3", "Cleanup model"]
    reasons = [row.reason_label.text() for row in page.models.model_rows if hasattr(row, "reason_label")]
    assert reasons == ["Chosen by the fake.", "Cleans everything."]
    assert "RTX 5060" in page.hardware_badge.text()
    page.close()


def test_models_card_refreshes_when_the_language_set_changes(app, tmp_path, fake_select):
    page, config, _messages = make_page(tmp_path)
    page.set_language_enabled("fr", True)
    assert fake_select.calls[-1] == (("en", "de", "sq", "fr"), Hardware.GPU)
    assert page.models.names()[0] == "Speech model for 4"
    calls = len(fake_select.calls)
    config.update(lambda s: replace(s, general=replace(s.general, sounds=False)))
    page.apply_settings(config.settings)
    assert len(fake_select.calls) == calls
    page.close()


def test_models_card_on_the_processor(app, tmp_path, fake_select):
    page, _config, _messages = make_page(tmp_path, gpu=NO_GPU)
    assert fake_select.calls[-1][1] is Hardware.CPU
    assert page.hardware_badge.text() == "For your processor"
    page.close()


ARC_DEVICE = GpuDevice(
    raw_index=0, list_name="Vulkan0", name="Intel(R) Arc(TM) Pro Graphics", memory_mb=37032,
    free_mb=30000, uma=True,
)
ARC = GpuSelection(raw_index=0, name=ARC_DEVICE.name, memory_mb=37032, devices=(ARC_DEVICE,))


def store_measurements(config, gpu_ms: int, cpu_ms: int) -> None:
    from spells import calibrate

    results = [
        calibrate.SpeechCalibration(
            model.id,
            ARC.name,
            calibrate.Run("vulkan", gpu_ms, 1.0),
            calibrate.Run("cpu", cpu_ms, 1.0),
        )
        for model in modelcatalog.load_catalog()
        if model.kind is ModelKind.ASR
    ]
    calibrate.save(calibrate.store_path(config.path), results)


def test_models_card_shows_a_measured_win_of_the_integrated_graphics(app, tmp_path, fake_select):
    page, config, _messages = make_page(tmp_path, gpu=ARC)
    assert fake_select.calls[-1][1] is Hardware.CPU
    store_measurements(config, 400, 2300)
    page.refresh_models()
    assert fake_select.calls[-1][1] is Hardware.GPU
    assert page.hardware_badge.text() == f"For your graphics, {ARC.name}, measured faster"
    page.close()


def test_models_card_shows_a_measured_win_of_the_processor(app, tmp_path, fake_select):
    page, config, _messages = make_page(tmp_path, gpu=ARC)
    store_measurements(config, 3000, 1300)
    page.refresh_models()
    assert fake_select.calls[-1][1] is Hardware.CPU
    assert page.hardware_badge.text() == "For your processor, measured faster than the graphics"
    page.close()


def test_models_card_names_missing_models_and_errors(app, fake_select):
    card = ModelsCard()
    fake_select.cleanup = False
    fake_select.uncovered = {"sq"}
    card.refresh(["en", "sq"], Hardware.CPU)
    titles = card.names()
    assert "No speech model for Albanian" in titles
    assert "No cleanup model" in titles
    fake_select.error = RuntimeError("catalog unreadable")
    card.refresh(["en", "sq"], Hardware.CPU, force=True)
    assert card.error == "catalog unreadable"
    assert card.names() == ["Models could not be chosen"]
    card.close()


def test_hardware_follows_the_gpu_selection():
    assert hardware_for(SELECTION) is Hardware.GPU
    assert hardware_for(NO_GPU) is Hardware.CPU
    assert hardware_for(None) is Hardware.CPU
    integrated = GpuDevice(raw_index=0, list_name="Vulkan0", name="Radeon 610M", memory_mb=8000, free_mb=7000, uma=True)
    only_integrated = GpuSelection(raw_index=0, name="Radeon 610M", memory_mb=8000, devices=(integrated,))
    assert hardware_for(only_integrated) is modelcatalog.hardware_from_gpu(only_integrated)


def test_the_settings_dialog_refreshes_models_after_a_gpu_override(app, tmp_path, fake_select):
    from spells.ui.settings import SettingsDialog

    from .fake_recorder import FakeRecorder
    from .test_ui_support import FakeEngines, FakeHistory, FakePipeline

    dialog = SettingsDialog(
        config=make_config(tmp_path),
        engines=FakeEngines(),
        pipeline=FakePipeline(),
        hotkey=FakeHotkey(),
        history=FakeHistory(),
        gpu_selection=SELECTION,
        log_dir=tmp_path / "logs",
        devices=list,
        probe=lambda _m, _k: True,
        notify=Messages(),
        confirm=lambda _t, _x: True,
        recorder_factory=FakeRecorder,
    )
    before = len(fake_select.calls)
    dialog.diagnostics.gpu_override.setCurrentIndex(1)
    assert len(fake_select.calls) == before + 1
    assert fake_select.calls[-1][1] is hardware_for(dialog.diagnostics.selection)
    dialog.close()


def test_search_list_rows_have_a_fixed_height(app):
    from PySide6 import QtWidgets

    adder = LanguageAdder()
    index = adder.results.model().index(0, 0)
    option = QtWidgets.QStyleOptionViewItem()
    option.rect = QtCore.QRect(0, 0, 300, 36)
    assert adder.results.itemDelegate().sizeHint(option, index).height() == 36
    assert index.data(QtCore.Qt.ItemDataRole.UserRole) == "en"
    adder.close()
