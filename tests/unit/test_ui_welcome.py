"""spells.ui.welcome: the first-run page of spec 14.3 with the elevated-window note."""

from __future__ import annotations

from dataclasses import replace

import pytest

from spells import platform
from spells.audio import AudioDevice
from spells.models import Chord
from spells.platform.keys_windows import WindowsKeys
from spells.ui.welcome import (
    STEP_MICROPHONE,
    MicMeter,
    MicPicker,
    WelcomePage,
)

from .fake_recorder import FakeRecorder
from .test_ui_support import FakeHotkey, Messages, flush, make_config, qt_app

DEVICES = [
    AudioDevice(index=1, name="Headset Microphone", is_default=True),
    AudioDevice(index=3, name="Webcam Mic", is_default=False),
]


@pytest.fixture(scope="module")
def app():
    return qt_app()


def make_page(tmp_path):
    config = make_config(tmp_path)
    recorders: list[FakeRecorder] = []

    def factory(**kwargs):
        recorder = FakeRecorder(**kwargs)
        recorders.append(recorder)
        return recorder

    asked: list[int] = []
    page = WelcomePage(
        config=config,
        hotkey=FakeHotkey(),
        devices=lambda: list(DEVICES),
        recorder_factory=factory,
        on_change_hotkey=lambda: asked.append(1),
        notify=Messages(),
    )
    return page, config, recorders, asked


def test_page_shows_the_hotkey_and_offers_to_change_it(app, tmp_path, use_platform):
    use_platform(replace(platform.current(), keys=WindowsKeys()))
    page, config, _recorders, asked = make_page(tmp_path)
    assert "Ctrl+Win" in page.hotkey_label.text()
    page.change_button.click()
    assert asked == [1]
    config.update(lambda s: replace(s, general=replace(s.general, main_chord=Chord(keys=(0x11, 0x12, 0x44)))))
    page.apply_settings(config.settings)
    assert "Ctrl+Alt+D" in page.hotkey_label.text()
    page.close()


def test_autostart_toggle_writes_the_setting(app, tmp_path):
    page, config, *_ = make_page(tmp_path)
    assert page.autostart.isChecked()
    page.autostart.setChecked(False)
    assert config.settings.general.autostart is False
    page.close()


def test_mic_picker_and_meter_share_the_device_choice(app, tmp_path):
    page, config, recorders, _asked = make_page(tmp_path)
    page.show()
    flush(app)
    assert recorders == []
    page.set_step(STEP_MICROPHONE)
    flush(app)
    assert recorders and recorders[-1].calls[:1] == ["start"]
    page.mic_picker.setCurrentIndex(2)
    assert config.settings.general.mic_device == "Webcam Mic"
    flush(app)
    assert recorders[-1].device == "Webcam Mic"
    page.close()
    flush(app)
    assert recorders[-1].closed
    page.close()


def test_meter_reports_a_mic_error_instead_of_raising(app, tmp_path):
    from spells.audio import MicError

    def factory(**kwargs):
        recorder = FakeRecorder(**kwargs)
        recorder.start_error = MicError("busy", "Microphone busy")
        return recorder

    messages = Messages()
    meter = MicMeter(recorder_factory=factory, notify=messages)
    meter.start(device=None)
    assert "Microphone busy" in meter.status.text()
    meter.close()


def test_the_tour_moves_through_three_steps_and_finishes(app, tmp_path):
    page, *_ = make_page(tmp_path)
    page.show()
    flush(app)
    assert page.step() == 0
    assert page.back_button.isHidden()
    assert page.next_button.text() == "Next"
    page.next_button.click()
    assert page.step() == 1 and not page.back_button.isHidden()
    page.next_button.click()
    assert page.step() == STEP_MICROPHONE and page.next_button.text() == "Finish"
    page.back_button.click()
    assert page.step() == 1
    page.set_step(99)
    assert page.step() == STEP_MICROPHONE
    page.next_button.click()
    flush(app)
    assert not page.isVisible()


def test_the_meter_runs_only_on_the_microphone_step(app, tmp_path):
    page, _config, recorders, _asked = make_page(tmp_path)
    page.show()
    page.set_step(STEP_MICROPHONE)
    flush(app)
    assert page.meter.running
    page.set_step(0)
    assert not page.meter.running
    assert recorders[-1].closed
    page.close()


def test_the_language_step_edits_the_enabled_languages(app, tmp_path):
    page, config, *_ = make_page(tmp_path)
    page.set_step(1)
    page.language_list.language_rows["sq"].chip.remove_button.click()
    assert config.settings.general.enabled_languages == ["en", "de"]
    page.language_adder.add_language.emit("fr")
    assert config.settings.general.enabled_languages == ["en", "de", "fr"]
    assert page.language_list.codes() == ["en", "de", "fr"]
    page.close()


def test_the_picker_matches_a_name_stored_while_capture_went_through_mme(app, tmp_path):
    config = make_config(tmp_path)
    config.update(
        lambda s: replace(s, general=replace(s.general, mic_device="Microphone (HyperX Cloud III Wi"))
    )
    devices = [
        AudioDevice(index=5, name="Headset Microphone", is_default=True),
        AudioDevice(index=7, name="Microphone (HyperX Cloud III Wireless)", is_default=False),
    ]
    picker = MicPicker(config=config, devices=lambda: list(devices), notify=Messages())
    assert picker.current_device() == "Microphone (HyperX Cloud III Wireless)"
    assert picker.index_for("Headset Microphone") == 1
    assert picker.index_for("Blue Yeti") == 0
    picker.deleteLater()

