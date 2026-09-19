"""spells.ui.create_ui: one call wires the bridge, the tray, the pill and the dialogs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from spells.models import EngineState
from spells.ui import UiHandles, create_ui
from spells.ui.bridge import UiBridge
from spells.ui.icons import TrayIconState

from .fake_recorder import FakeRecorder
from .test_ui_support import (
    SELECTION,
    WHISPER,
    FakeEngines,
    FakeHistory,
    FakeHotkey,
    FakePipeline,
    PillState,
    TrayState,
    event,
    flush,
    make_config,
    qt_app,
)


@pytest.fixture(scope="module")
def app():
    return qt_app()


def make_ui(app, tmp_path, *, first_run=False, bridge=None):
    config = make_config(tmp_path)
    engines = FakeEngines()
    engines.states = {e: EngineState.READY for e in engines.states}
    quits: list[int] = []
    handles = create_ui(
        app,
        config=config,
        engines=engines,
        pipeline=FakePipeline(),
        hotkey=FakeHotkey(),
        history=FakeHistory(),
        gpu_selection=SELECTION,
        log_dir=tmp_path / "logs",
        first_run=first_run,
        bridge=bridge,
        on_quit=lambda: quits.append(1),
        devices=list,
        recorder_factory=FakeRecorder,
        reduced_motion=True,
    )
    return handles, config, engines, quits


def test_handles_carry_the_parts(app, tmp_path):
    handles, *_ = make_ui(app, tmp_path)
    assert isinstance(handles, UiHandles)
    assert isinstance(handles.bridge, UiBridge)
    assert handles.tray.state is TrayIconState.READY
    assert handles.pill.content is None
    assert callable(handles.open_settings)
    assert handles.welcome is None
    handles.close()


def test_a_supplied_bridge_is_used(app, tmp_path):
    bridge = UiBridge()
    handles, *_ = make_ui(app, tmp_path, bridge=bridge)
    assert handles.bridge is bridge
    handles.close()


def test_bridge_events_reach_tray_and_pill(app, tmp_path):
    handles, *_ = make_ui(app, tmp_path)
    handles.bridge.on_pipeline_event(event(PillState.RECORDING, TrayState.RECORDING))
    handles.bridge.on_engine_status(WHISPER, EngineState.RESTARTING, "crash")
    handles.bridge.on_hotkey_error("refused")
    flush(app)
    assert handles.tray.state is TrayIconState.RECORDING
    assert handles.pill.content is not None and handles.pill.content.meter
    handles.bridge.on_pipeline_event(event())
    flush(app)
    assert handles.tray.state is TrayIconState.WARNING
    assert "crash" in handles.tray.tooltip and "refused" in handles.tray.tooltip
    handles.close()


def test_config_changes_are_subscribed_and_marshalled(app, tmp_path):
    handles, config, *_ = make_ui(app, tmp_path)
    config.update(lambda s: replace(s, general=replace(s.general, language_mode="de")))
    assert not handles.tray.language_actions["de"].isChecked()
    flush(app)
    assert handles.tray.language_actions["de"].isChecked()
    handles.close()


def test_open_settings_shows_the_dialog_on_the_requested_tab(app, tmp_path):
    handles, *_ = make_ui(app, tmp_path)
    assert handles.settings_dialog is None
    handles.open_settings("history")
    flush(app)
    dialog = handles.settings_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog.tabs.currentWidget() is dialog.history
    handles.tray.open_requested.emit("diagnostics")
    flush(app)
    assert dialog.tabs.currentWidget() is dialog.diagnostics
    assert handles.settings_dialog is dialog
    handles.close()


def test_first_run_shows_the_welcome_page_once(app, tmp_path):
    handles, *_ = make_ui(app, tmp_path, first_run=True)
    flush(app)
    assert handles.welcome is not None and handles.welcome.isVisible()
    handles.close()


def test_quit_goes_to_the_callback(app, tmp_path):
    handles, _config, _engines, quits = make_ui(app, tmp_path)
    handles.tray.quit_action.trigger()
    assert quits == [1]
    handles.close()
