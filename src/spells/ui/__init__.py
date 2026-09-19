"""The Qt user interface (spec 14): tray, pill, welcome page, settings, notifications.

`create_ui` wires everything in one call and returns the handles the app module keeps. It
starts no thread and no engine; it only builds widgets on the calling (Qt) thread and
subscribes the bridge to the settings store. PySide6 is imported only inside this package.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6 import QtWidgets

from spells.config import ConfigStore
from spells.gpu import GpuSelection
from spells.ui import style
from spells.ui.bridge import UiBridge
from spells.ui.pill import Pill
from spells.ui.settings import SettingsDialog
from spells.ui.tray import Tray
from spells.ui.updatecheck import UpdateCoordinator
from spells.ui.welcome import WelcomePage

log = logging.getLogger(__name__)


@dataclass
class UiHandles:
    """What the app module keeps after create_ui."""

    bridge: UiBridge
    tray: Tray
    pill: Pill
    open_settings: Callable[..., None]
    welcome: WelcomePage | None = None
    settings_dialog: SettingsDialog | None = None
    updates: UpdateCoordinator | None = None
    _closers: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def close(self) -> None:
        """Tear the widgets down (quit path and tests); idempotent."""
        closers, self._closers = self._closers, []
        for closer in closers:
            try:
                closer()
            except Exception:
                log.exception("ui close step failed")


def create_ui(
    app: QtWidgets.QApplication,
    *,
    config: ConfigStore,
    engines: Any,
    pipeline: Any,
    hotkey: Any,
    history: Any,
    gpu_selection: GpuSelection,
    log_dir: Path,
    first_run: bool,
    update_source: str = "",
    bridge: UiBridge | None = None,
    on_quit: Callable[[], None] | None = None,
    devices: Callable[[], list] | None = None,
    recorder_factory: Callable[..., Any] | None = None,
    reduced_motion: bool | None = None,
) -> UiHandles:
    """Build the tray, the pill, the dialogs and the bridge; nothing starts here.

    `bridge` may be created earlier (on the Qt thread) so its callbacks can be handed to the
    Pipeline, the EngineSupervisor and the HotkeyThread before this call; otherwise one is
    created here and the app forwards to `handles.bridge.on_*` through late-binding lambdas.
    `on_quit` defaults to app.quit; the app should stop the threads before quitting.

    `update_source` is the address this build checks for a newer version, fixed when the
    build was made (spec 19.7). An empty one, which is what a development checkout has, gives
    an About page that says so and a coordinator that never opens a connection.
    """
    bridge = bridge if bridge is not None else UiBridge()
    try:
        style.install(app)
    except Exception:
        log.exception("could not install the Spells style")
    tray = Tray(
        config=config,
        pipeline=pipeline,
        hotkey=hotkey,
        engines=engines,
        no_vulkan_gpu=gpu_selection.raw_index is None,
    )
    pill = Pill(reduced_motion=reduced_motion)
    quit_app = on_quit if on_quit is not None else app.quit
    coordinator = UpdateCoordinator(
        config=config,
        source_url=update_source,
        busy=tray.busy,
        on_quit=quit_app,
    )
    handles = UiHandles(
        bridge=bridge,
        tray=tray,
        pill=pill,
        open_settings=lambda tab="general": None,
        updates=coordinator,
    )

    bridge.connect_pipeline(tray.apply_event)
    bridge.connect_pipeline(pill.apply)
    bridge.connect_engine_status(tray.set_engine_status)
    bridge.connect_hotkey_error(tray.set_hook_error)
    bridge.connect_settings(tray.apply_settings)
    bridge.connect_settings(coordinator.apply_settings)
    coordinator.update_found.connect(tray.notify_update)
    unsubscribe = config.subscribe(bridge.on_settings)
    handles._closers.append(unsubscribe)

    def show_welcome() -> None:
        page = handles.welcome
        if page is None:
            page = WelcomePage(
                config=config,
                hotkey=hotkey,
                devices=devices,
                recorder_factory=recorder_factory,
                on_change_hotkey=lambda: open_settings("general"),
                reduced_motion=reduced_motion,
            )
            bridge.connect_settings(page.apply_settings)
            handles.welcome = page
        page.show()
        page.raise_()
        page.activateWindow()

    def open_settings(tab: str = "general") -> None:
        dialog = handles.settings_dialog
        if dialog is None:
            dialog = SettingsDialog(
                config=config,
                engines=engines,
                pipeline=pipeline,
                hotkey=hotkey,
                history=history,
                gpu_selection=gpu_selection,
                log_dir=log_dir,
                devices=devices,
                recorder_factory=recorder_factory,
            )
            bridge.connect_settings(dialog.apply_settings)
            bridge.connect_engine_status(dialog.on_engine_status)
            dialog.welcome_requested.connect(show_welcome)
            dialog.about.attach(coordinator)
            handles.settings_dialog = dialog
        dialog.show_tab(tab)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    handles.open_settings = open_settings
    tray.open_requested.connect(open_settings)
    tray.quit_requested.connect(quit_app)

    def close_widgets() -> None:
        pill.close()
        tray.hide()
        if handles.settings_dialog is not None:
            handles.settings_dialog.close()
        if handles.welcome is not None:
            handles.welcome.close()

    handles._closers.append(close_widgets)
    handles._closers.append(coordinator.stop_timer)
    coordinator.start_timer()

    tray.show()
    if first_run:
        show_welcome()
    return handles


__all__ = ["UiBridge", "UiHandles", "UpdateCoordinator", "create_ui"]
