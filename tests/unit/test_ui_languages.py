"""spells.ui.languages: the language chips, the search, per-language hotkeys and the models card."""

from __future__ import annotations

from dataclasses import replace

import pytest

from spells import modelcatalog, platform
from spells.gpu import GpuDevice, GpuSelection
from spells.modelcatalog import Hardware, ModelChoice, ModelKind, Selection
from spells.platform.keys_windows import WindowsKeys
from spells.ui.languages import (
    LanguagesPage,
    ModelsCard,
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


def test_language_hotkeys_sit_beside_their_chip(app, tmp_path, fake_select, use_platform):
    use_platform(replace(platform.current(), keys=WindowsKeys()))
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

