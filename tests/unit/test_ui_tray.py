"""spells.ui.tray and spells.ui.icons: state composition (spec 14.1), the menu, toasts.

The seven tray states are composed from the engine status per engine, the hotkey install
error, the GPU probe result and the pipeline's own tray state. Menu actions are checked
against fakes of the collaborators they must call.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from PySide6 import QtGui, QtWidgets

from spells import platform
from spells.models import Chord, EngineState
from spells.platform.base import SETTINGS_MICROPHONE_PRIVACY, SETTINGS_SOUND
from spells.platform.stub import StubShell
from spells.ui.icons import TrayIconState, render_tray_pixmap
from spells.ui.tray import NO_CLEANUP_MODEL, Tray, TrayInputs, compose_tray_state

from .fake_platform import fake_platform
from .test_ui_support import (
    LLAMA,
    WHISPER,
    FakeEngines,
    FakeHotkey,
    FakePipeline,
    Notice,
    PillState,
    TrayState,
    event,
    make_config,
    qt_app,
)

READY = EngineState.READY


@pytest.fixture(scope="module")
def app():
    return qt_app()


class SettingsShell(StubShell):
    def __init__(self) -> None:
        self.opened: list[str] = []
        self.paths: list[str] = []

    def open_path(self, path):
        self.paths.append(path)

    def has_settings(self, kind):
        return True

    def open_settings(self, kind):
        self.opened.append(kind)
        return True


@pytest.fixture(autouse=True)
def recording_shell(use_platform):
    shell = SettingsShell()
    use_platform(replace(platform.current(), shell=shell))
    return shell


def inputs(**changes) -> TrayInputs:
    base = TrayInputs(
        engines={WHISPER: (READY, "ok"), LLAMA: (READY, "ok")},
        pipeline=TrayState.READY,
    )
    return replace(base, **changes)


def engines(whisper=READY, llama=READY, whisper_reason="ok", llama_reason="ok"):
    return {WHISPER: (whisper, whisper_reason), LLAMA: (llama, llama_reason)}



def test_all_ready_is_ready():
    state, tooltip = compose_tray_state(inputs())
    assert state is TrayIconState.READY
    assert "Ready" in tooltip


@pytest.mark.parametrize(
    "engine, engine_state, reason, expected_words",
    [
        (WHISPER, EngineState.RESTARTING, "crash", ("restarting", "crash")),
        (LLAMA, EngineState.RESTARTING, "crash", ("restarting",)),
        (LLAMA, EngineState.PAUSED, "oom", ("paused", "GPU memory")),
        (WHISPER, EngineState.CPU_FALLBACK, "oom", ("CPU", "GPU memory")),
        (LLAMA, EngineState.CPU_FALLBACK, "no_vulkan_gpu", ("CPU", "Vulkan")),
        (LLAMA, EngineState.CPU_FALLBACK, "dll_not_found", ("CPU", "DLL")),
    ],
)
def test_restarting_paused_and_cpu_fallback_are_warnings_with_the_reason(
    engine, engine_state, reason, expected_words
):
    states = engines()
    states[engine] = (engine_state, reason)
    state, tooltip = compose_tray_state(inputs(engines=states))
    assert state is TrayIconState.WARNING
    for word in expected_words:
        assert word.lower() in tooltip.lower()


def test_a_cleanup_engine_without_a_model_is_a_warning_never_an_error():
    """Dictation still works without cleanup, so a missing model only warns (spec 14.1, 16)."""
    state, tooltip = compose_tray_state(
        inputs(engines=engines(llama=EngineState.FAILED, llama_reason="no_model"))
    )
    assert state is TrayIconState.WARNING
    assert NO_CLEANUP_MODEL in tooltip
    assert "Cleanup is off" in tooltip
    assert "failed" not in tooltip.lower()


@pytest.mark.parametrize("whisper_reason", ["crash", "dll_not_found", "no_model"])
def test_a_failed_speech_engine_is_still_an_error_beside_a_missing_cleanup_model(whisper_reason):
    state, tooltip = compose_tray_state(
        inputs(
            engines=engines(
                whisper=EngineState.FAILED,
                llama=EngineState.FAILED,
                whisper_reason=whisper_reason,
                llama_reason="no_model",
            )
        )
    )
    assert state is TrayIconState.ERROR
    assert NO_CLEANUP_MODEL in tooltip


@pytest.mark.parametrize("engine", [WHISPER, LLAMA])
def test_a_failed_engine_is_an_error(engine):
    states = engines()
    states[engine] = (EngineState.FAILED, "crash")
    state, tooltip = compose_tray_state(inputs(engines=states))
    assert state is TrayIconState.ERROR
    assert "failed" in tooltip.lower()


def test_a_dismissed_mic_error_is_an_error():
    state, tooltip = compose_tray_state(inputs(mic_error="Microphone busy"))
    assert state is TrayIconState.ERROR
    assert "Microphone busy" in tooltip


def test_recording_and_processing_come_from_the_pipeline():
    assert compose_tray_state(inputs(pipeline=TrayState.RECORDING))[0] is TrayIconState.RECORDING
    assert compose_tray_state(inputs(pipeline=TrayState.PROCESSING))[0] is TrayIconState.PROCESSING


def test_recording_outranks_a_warning_but_not_an_error():
    warned = inputs(pipeline=TrayState.RECORDING, no_vulkan_gpu=True)
    assert compose_tray_state(warned)[0] is TrayIconState.RECORDING
    failed = inputs(pipeline=TrayState.RECORDING, engines=engines(EngineState.FAILED))
    assert compose_tray_state(failed)[0] is TrayIconState.ERROR


def test_warning_outranks_starting_and_idle():
    starting = inputs(engines=engines(EngineState.STARTING), hook_error="refused")
    assert compose_tray_state(starting)[0] is TrayIconState.WARNING
    idle = inputs(engines=engines(EngineState.UNLOADED, EngineState.UNLOADED), no_vulkan_gpu=True)
    assert compose_tray_state(idle)[0] is TrayIconState.WARNING


def _opaque_pixels(pixmap: QtGui.QPixmap) -> int:
    image = pixmap.toImage()
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 0
    )


@pytest.mark.parametrize("light_taskbar", [False, True])
def test_every_state_has_a_distinct_non_empty_glyph(app, light_taskbar):
    renders = {}
    for state in TrayIconState:
        pixmap = render_tray_pixmap(state, 32, light_taskbar=light_taskbar)
        assert not pixmap.isNull()
        assert pixmap.width() == 32 and pixmap.height() == 32
        assert _opaque_pixels(pixmap) > 40
        renders[state] = bytes(pixmap.toImage().constBits())
    assert len(set(renders.values())) == len(TrayIconState)


class RecordingIcon(QtWidgets.QSystemTrayIcon):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple] = []

    def showMessage(self, *args) -> None:
        self.messages.append(args)


def make_tray(tmp_path, *, no_vulkan_gpu=False, llama=(READY, "ok")):
    config = make_config(tmp_path)
    pipeline = FakePipeline()
    hotkey = FakeHotkey()
    fake_engines = FakeEngines()
    fake_engines.states = {WHISPER: READY, LLAMA: llama[0]}
    fake_engines.reasons = {WHISPER: "ok", LLAMA: llama[1]}
    icon = RecordingIcon()
    launched: list[str] = []
    tray = Tray(
        config=config,
        pipeline=pipeline,
        hotkey=hotkey,
        engines=fake_engines,
        no_vulkan_gpu=no_vulkan_gpu,
        icon=icon,
        launcher=launched.append,
        light_taskbar=False,
    )
    return tray, config, pipeline, hotkey, icon, launched


def test_a_missing_cleanup_model_from_the_supervisor_keeps_the_tray_in_warning(app, tmp_path):
    tray, *_ = make_tray(tmp_path, llama=(EngineState.FAILED, "no_model"))
    assert tray.state is TrayIconState.WARNING
    assert NO_CLEANUP_MODEL in tray.tooltip
    tray.set_engine_status(LLAMA, EngineState.FAILED, "no_model")
    assert tray.state is TrayIconState.WARNING
    tray.set_engine_status(LLAMA, EngineState.FAILED, "crash")
    assert tray.state is TrayIconState.ERROR


def test_pipeline_events_drive_recording_processing_and_retry(app, tmp_path):
    tray, _config, _pipeline, _hotkey, _icon, _launched = make_tray(tmp_path)
    tray.apply_event(event(PillState.RECORDING, TrayState.RECORDING))
    assert tray.state is TrayIconState.RECORDING
    tray.apply_event(event(PillState.PROCESSING, TrayState.PROCESSING, busy=True))
    assert tray.state is TrayIconState.PROCESSING
    assert not tray.retry_action.isEnabled()
    tray.apply_event(event(retry_available=True))
    assert tray.state is TrayIconState.READY
    assert tray.retry_action.isEnabled()


def test_a_mic_error_keeps_the_tray_in_error_until_the_next_recording(app, tmp_path):
    tray, *_ = make_tray(tmp_path)
    tray.apply_event(
        event(
            notice=Notice.ERROR,
            notice_text="Mic is busy",
            notification="Microphone busy: another app holds it",
            notification_action=SETTINGS_SOUND,
        )
    )
    assert tray.state is TrayIconState.ERROR
    assert "Microphone busy" in tray.tooltip
    tray.apply_event(event(PillState.RECORDING, TrayState.RECORDING))
    assert tray.state is TrayIconState.RECORDING
    tray.apply_event(event())
    assert tray.state is TrayIconState.READY


def test_language_menu_lists_auto_plus_each_enabled_language(app, tmp_path):
    tray, config, *_ = make_tray(tmp_path)
    assert list(tray.language_actions) == ["auto", "en", "de", "sq"]
    assert tray.language_actions["auto"].isChecked()
    assert "German" in tray.language_actions["de"].text()
    config.update(lambda s: replace(s, general=replace(s.general, enabled_languages=["en", "fr"])))
    tray.apply_settings(config.settings)
    assert list(tray.language_actions) == ["auto", "en", "fr"]


def test_language_action_sets_the_language_mode_through_the_store(app, tmp_path):
    tray, config, *_ = make_tray(tmp_path)
    before = config.settings
    tray.language_actions["de"].trigger()
    assert config.settings.general.language_mode == "de"
    assert config.settings is not before
    tray.apply_settings(config.settings)
    assert tray.language_actions["de"].isChecked()
    assert not tray.language_actions["auto"].isChecked()
    tray.language_actions["auto"].trigger()
    assert config.settings.general.language_mode == "auto"


def test_cleanup_action_toggles_the_cleanup_setting(app, tmp_path):
    tray, config, *_ = make_tray(tmp_path)
    assert tray.cleanup_action.isChecked()
    tray.cleanup_action.trigger()
    assert config.settings.cleanup.enabled is False
    tray.cleanup_action.trigger()
    assert config.settings.cleanup.enabled is True


def test_pause_clears_the_chords_and_resume_restores_them(app, tmp_path):
    tray, config, _pipeline, hotkey, *_ = make_tray(tmp_path)
    tray.pause_action.trigger()
    assert tray.paused
    assert hotkey.pushed[-1] == ()
    assert "paused" in tray.tooltip.lower()
    tray.pause_action.trigger()
    assert not tray.paused
    assert hotkey.pushed[-1] == (config.settings.general.main_chord,)


def test_settings_change_while_paused_keeps_the_chords_cleared(app, tmp_path):
    tray, config, _pipeline, hotkey, *_ = make_tray(tmp_path)
    tray.pause_action.trigger()
    de = Chord(keys=(0x11, 0x44), language="de")
    config.update(lambda s: replace(s, general=replace(s.general, language_chords=[de])))
    tray.apply_settings(config.settings)
    assert hotkey.pushed[-1] == ()
    tray.pause_action.trigger()
    assert hotkey.pushed[-1] == (config.settings.general.main_chord, de)


def test_retry_calls_the_pipeline(app, tmp_path):
    tray, _config, pipeline, *_ = make_tray(tmp_path)
    tray.apply_event(event(retry_available=True))
    tray.retry_action.trigger()
    assert pipeline.calls == ["retry_last"]


def test_quit_settings_and_history_emit_their_requests(app, tmp_path):
    tray, *_ = make_tray(tmp_path)
    quits: list[int] = []
    opened: list[str] = []
    tray.quit_requested.connect(lambda: quits.append(1))
    tray.open_requested.connect(opened.append)
    tray.quit_action.trigger()
    tray.settings_action.trigger()
    tray.history_action.trigger()
    assert quits == [1]
    assert opened == ["general", "history"]


def test_left_click_opens_settings(app, tmp_path):
    tray, _config, _pipeline, _hotkey, icon, _launched = make_tray(tmp_path)
    opened: list[str] = []
    tray.open_requested.connect(opened.append)
    icon.activated.emit(QtWidgets.QSystemTrayIcon.ActivationReason.Trigger)
    icon.activated.emit(QtWidgets.QSystemTrayIcon.ActivationReason.Context)
    assert opened == ["general"]


def test_notification_with_a_settings_kind_opens_it_on_click(app, tmp_path, use_platform):
    shell = SettingsShell()
    use_platform(fake_platform(shell=shell))
    tray, _config, _pipeline, _hotkey, icon, launched = make_tray(tmp_path)
    tray.apply_event(
        event(
            notice=Notice.ERROR,
            notice_text="Mic is blocked",
            notification="Microphone blocked by the privacy settings",
            notification_action=SETTINGS_MICROPHONE_PRIVACY,
        )
    )
    assert len(icon.messages) == 1
    assert "Microphone blocked by the privacy settings" in icon.messages[0][1]
    assert "Open settings" in icon.messages[0][1]
    icon.messageClicked.emit()
    assert shell.opened == ["microphone_privacy"]
    icon.messageClicked.emit()
    assert shell.opened == ["microphone_privacy"]
    assert launched == []


def test_engines_on_the_processor_by_choice_are_ready_not_a_warning():
    state, tooltip = compose_tray_state(
        inputs(engines=engines(whisper=EngineState.CPU_FALLBACK, whisper_reason="cpu_selected", llama=EngineState.CPU_FALLBACK, llama_reason="cpu_selected"))
    )
    assert state is TrayIconState.READY
    assert "chosen for this computer" in tooltip
    state, _tooltip = compose_tray_state(inputs(engines=engines(llama=EngineState.CPU_FALLBACK, llama_reason="oom")))
    assert state is TrayIconState.WARNING


def test_clicking_an_update_balloon_opens_the_about_page_and_launches_nothing(app, tmp_path):
    tray, _config, _pipeline, _hotkey, _icon, launched = make_tray(tmp_path)
    opened: list[str] = []
    tray.open_requested.connect(opened.append)

    tray.notify_update("0.3.0")
    tray._on_message_clicked()

    assert opened == ["about"]
    assert launched == []


@pytest.mark.parametrize("state", [TrayState.RECORDING, TrayState.PROCESSING])
def test_no_update_balloon_interrupts_a_dictation(app, tmp_path, state):
    tray, _config, _pipeline, _hotkey, icon, _launched = make_tray(tmp_path)
    tray.apply_event(event(pill=PillState.RECORDING, tray=state))
    before = len(icon.messages)

    assert tray.busy()
    assert not tray.notify_update("0.3.0")
    assert len(icon.messages) == before

