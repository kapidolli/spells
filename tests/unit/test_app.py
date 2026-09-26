"""The process entry point with every collaborator faked (spec 15, 19.5, 20.1).

No Qt, no engines, no registry, no real hidden window: `Deps` is the seam, so these tests
drive the real start order, the real shutdown order and the real message-window handlers
against recording fakes. The only real collaborator is `ConfigStore`, which is pure
filesystem and whose subscriber contract the app depends on.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from spells import app as app_module
from spells import calibrate, paths
from spells.app import MUTEX_NAME, WINDOW_CLASS, Deps, main, parse_args
from spells.config import ConfigStore, Settings, default_settings, load
from spells.engines import SpeechEngine
from spells.gpu import GpuDevice, GpuSelection
from spells.modelcatalog import CatalogError, load_catalog, parse_catalog
from spells.models import CpuPlan, Engine, EngineId, EngineState
from spells.pipeline import PillState, PipelineEvent, TrayState
from spells.win32.msgwindow import MessageHandlers

RTX = "NVIDIA GeForce RTX 5060 Laptop GPU"
SELECTION = GpuSelection(
    raw_index=1,
    name=RTX,
    memory_mb=7810,
    devices=(GpuDevice(1, "Vulkan0", RTX, 7810, 7042),),
)


# Fakes ------------------------------------------------------------------------------------


class Signal:
    """A Qt signal stand-in: connect() collects slots, emit() calls them at once."""

    def __init__(self, log: list[str], name: str) -> None:
        self._log = log
        self._name = name
        self._slots: list = []
        self.emitted: list[tuple] = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def emit(self, *args) -> None:
        self._log.append(self._name)
        self.emitted.append(args)
        for slot in list(self._slots):
            slot(*args)


class FakeTray:
    def __init__(self, log: list[str]) -> None:
        self.quit_requested = Signal(log, "tray.quit_requested")
        self.open_requested = Signal(log, "tray.open_requested")
        self.messages: list[tuple[str, bool]] = []
        self.extra_warnings: list[str | None] = []

    def notify(self, text: str, action=None, *, warning: bool = False) -> None:
        self.messages.append((text, warning))

    def set_extra_warning(self, reason: str | None) -> None:
        self.extra_warnings.append(reason)


class FakePlaceholder:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.hidden = 0
        self.deleted = 0

    def hide(self) -> None:
        self.hidden += 1
        self._log.append("placeholder.hide")

    def deleteLater(self) -> None:  # the Qt spelling
        self.deleted += 1


class FakeHandles:
    def __init__(self, log: list[str], bridge) -> None:
        self._log = log
        self.bridge = bridge
        self.tray = FakeTray(log)
        self.opened: list[str] = []
        self.closed = 0
        self.tray.open_requested.connect(self.open_settings)

    def open_settings(self, tab: str = "general") -> None:
        self.opened.append(tab)

    def close(self) -> None:
        self.closed += 1
        self._log.append("ui.close")


class FakeBridge:
    def __init__(self) -> None:
        self.events: list = []

    def on_pipeline_event(self, event) -> None:
        self.events.append(event)

    def on_engine_status(self, engine, state, reason) -> None:
        self.events.append((engine, state, reason))

    def on_hotkey_error(self, message: str) -> None:
        self.events.append(message)

    def on_settings(self, settings) -> None:
        self.events.append(settings)


class FakeQApp:
    def __init__(self, log: list[str], on_exec) -> None:
        self._log = log
        self._on_exec = on_exec
        self.quits = 0

    def exec(self) -> int:
        self._log.append("app.exec")
        if self._on_exec is not None:
            self._on_exec()
        return 0

    def quit(self) -> None:
        self.quits += 1
        self._log.append("app.quit")


class FakeEngines:
    def __init__(
        self,
        log: list[str],
        engine_paths,
        gpu,
        on_status,
        idle_unload_minutes=0,
        cpu_only=False,
        cpu_cleanup_allowed=False,
        cpu_plan=None,
    ) -> None:
        self.cpu_plan = cpu_plan
        self._log = log
        self.paths = engine_paths
        self.gpu = gpu
        self.on_status = on_status
        self.idle_minutes = [idle_unload_minutes]
        self.cpu_only = cpu_only
        self.cpu_cleanup_allowed = cpu_cleanup_allowed
        self.model_changes: list[tuple] = []
        self.started = 0
        self.stopped = 0

    def set_models(self, paths, *, cpu_only: bool, cpu_cleanup_allowed: bool, gpu=None):
        if gpu is not None:
            self.gpu = gpu
        self.model_changes.append((paths, cpu_only, cpu_cleanup_allowed))
        changed = tuple(
            EngineId(engine)
            for engine, key in ((Engine.WHISPER, "whisper"), (Engine.LLAMA, "llama"))
            if (getattr(paths, f"{key}_model"), getattr(paths, f"{key}_args"), cpu_only)
            != (getattr(self.paths, f"{key}_model"), getattr(self.paths, f"{key}_args"), self.cpu_only)
        )
        self.paths = paths
        self.cpu_only = cpu_only
        self.cpu_cleanup_allowed = cpu_cleanup_allowed
        return changed

    def start(self) -> None:
        self.started += 1
        self._log.append("engines.start")

    def stop(self) -> None:
        self.stopped += 1
        self._log.append("engines.stop")

    def set_idle_unload_minutes(self, minutes: float) -> None:
        self.idle_minutes.append(minutes)

    def status(self, engine: Engine) -> EngineState:
        return EngineState.STARTING

    def reason(self, engine: Engine) -> str:
        return "ok"

    def wait_ready(self, engine, timeout_s: float) -> bool:
        return True


class FakeCalibrator:
    def __init__(self, speeds: dict, **kwargs) -> None:
        self.speeds = speeds
        self.kwargs = kwargs
        self.measured: list[str] = []
        self.cancelled = False

    def calibrate(self, choice):
        self.measured.append(choice.model_id)
        gpu_ms, cpu_ms = self.speeds.get(choice.model_id, (300, 1300))
        return calibrate.SpeechCalibration(
            choice.model_id,
            self.kwargs["gpu"].name,
            calibrate.Run("vulkan", gpu_ms, 1.0),
            calibrate.Run("cpu", cpu_ms, 1.0),
        )

    def cancel(self) -> None:
        self.cancelled = True


class FakePipeline:
    def __init__(self, log: list[str], **kwargs) -> None:
        self._log = log
        self.kwargs = kwargs
        self.on_event = kwargs["on_event"]
        self.end_recording = kwargs["hotkey"]
        self.started = 0
        self.stopped = 0

    def hotkey_callbacks(self):
        return ("callbacks",)

    def set_selection(self, selection):
        self.kwargs["selection"] = selection

    def start(self) -> None:
        self.started += 1
        self._log.append("pipeline.start")

    def stop(self) -> None:
        self.stopped += 1
        self._log.append("pipeline.stop")


class FakeHotkey:
    def __init__(self, log: list[str], chords, callbacks, on_error=None) -> None:
        self._log = log
        self.chords = list(chords)
        self.callbacks = callbacks
        self.on_error = on_error
        self.started = 0
        self.stopped = 0
        self.reinstalls = 0
        self.ended: list[int] = []

    def start(self) -> None:
        self.started += 1
        self._log.append("hotkey.start")

    def stop(self) -> None:
        self.stopped += 1
        self._log.append("hotkey.stop")

    def end_recording(self, dictation_id: int) -> None:
        self.ended.append(dictation_id)

    def request_reinstall(self) -> None:
        self.reinstalls += 1


class FakeHistory:
    def __init__(self, log: list[str], path, retention, upload_hold: bool = False) -> None:
        self._log = log
        self.path = path
        self.retention = retention
        self.upload_hold = upload_hold
        self.retentions: list[str] = []
        self.closed = 0
        self.closing: threading.Event | None = None
        self.close_gate: threading.Event | None = None

    def set_retention(self, retention: str) -> None:
        self.retentions.append(retention)

    def close(self) -> None:
        if self.closing is not None:
            self.closing.set()
        if self.close_gate is not None:
            self.close_gate.wait(10.0)
        self.closed += 1
        self._log.append("history.close")


class FakeWindow:
    def __init__(self, log: list[str], class_name: str, handlers: MessageHandlers) -> None:
        self._log = log
        self.class_name = class_name
        self.handlers = handlers
        self.destroyed = 0
        self.ran = threading.Event()

    def run(self, stop: threading.Event) -> None:
        self.ran.set()
        stop.wait(5.0)

    def destroy(self) -> None:
        self.destroyed += 1
        self._log.append("window.destroy")


class CountingStore(ConfigStore):
    """ConfigStore that counts the writes the app makes."""

    def __init__(self, path: Path) -> None:
        self.updates = 0
        super().__init__(path)

    def update(self, mutator):
        self.updates += 1
        return super().update(mutator)


@dataclass
class World:
    """Everything the fakes create, plus the Deps that build them."""

    tmp_path: Path
    log: list[str] = field(default_factory=list)
    taken: bool = False
    window_present: bool = True
    quit_clears_window: bool = True
    window_after_finds: int = 0
    selection: GpuSelection = SELECTION
    on_exec: object = None
    config: CountingStore | None = None
    app: FakeQApp | None = None
    bridge: FakeBridge | None = None
    engines: FakeEngines | None = None
    pipeline: FakePipeline | None = None
    hotkey: FakeHotkey | None = None
    history: FakeHistory | None = None
    handles: FakeHandles | None = None
    window: FakeWindow | None = None
    placeholder: FakePlaceholder | None = None
    layout: paths.Layout | None = None
    signals: list[tuple[str, str]] = field(default_factory=list)
    autostart_calls: list[tuple[bool, str]] = field(default_factory=list)
    gpu_calls: list[dict] = field(default_factory=list)
    released: list[str] = field(default_factory=list)
    log_dirs: list[Path] = field(default_factory=list)
    debug_levels: list[bool] = field(default_factory=list)
    probe_saw_placeholder: list[bool] = field(default_factory=list)
    finds: int = 0
    real_logging: bool = False
    catalog: object = None
    catalog_broken: bool = False
    cpu_plan: object = field(default_factory=CpuPlan)
    cpu_plan_calls: int = 0
    calibrators: list = field(default_factory=list)
    calibration_speeds: dict = field(default_factory=dict)

    # Fake implementations -----------------------------------------------------------------

    def make_layout(self) -> paths.Layout:
        env = {
            paths.SETTINGS_DIR_ENV: str(self.tmp_path / "cfg"),
            paths.DATA_DIR_ENV: str(self.tmp_path / "data"),
        }
        self.layout = paths.resolve(frozen=False, repo_root=self.tmp_path, env=env)
        return self.layout

    def setup_logging(self, log_dir: Path, *, debug: bool = False):
        self.log.append("logging")
        self.log_dirs.append(log_dir)
        if self.real_logging:
            return app_module.setup_logging(log_dir, debug=debug)
        return None

    def acquire(self, name: str) -> bool:
        self.log.append("mutex")
        assert name == MUTEX_NAME
        return not self.taken

    def release(self, name: str) -> None:
        self.released.append(name)
        self.log.append("mutex.release")

    def signal(self, class_name: str, command: str) -> bool:
        self.signals.append((class_name, command))
        if command == "quit" and self.quit_clears_window:
            self.window_present = False
        return True

    def find(self, class_name: str) -> int:
        self.finds += 1
        if self.finds <= self.window_after_finds:
            return 0
        return 4242 if self.window_present else 0

    def set_debug_logging(self, enabled: bool) -> None:
        self.debug_levels.append(enabled)
        if self.real_logging:
            app_module.set_debug_logging(enabled)

    def placeholder_tray(self) -> FakePlaceholder:
        self.log.append("placeholder")
        self.placeholder = FakePlaceholder(self.log)
        return self.placeholder

    def config_store(self, path: Path) -> CountingStore:
        self.log.append("config")
        self.config = CountingStore(path)
        return self.config

    def select_device(self, llama_server, **kwargs) -> GpuSelection:
        self.log.append("gpu")
        self.probe_saw_placeholder.append(
            self.placeholder is not None and self.placeholder.hidden == 0
        )
        self.gpu_calls.append({"llama_server": llama_server, **kwargs})
        return self.selection

    def supervisor(self, engine_paths, gpu, on_status, **kwargs) -> FakeEngines:
        self.engines = FakeEngines(self.log, engine_paths, gpu, on_status, **kwargs)
        return self.engines

    def make_history(self, path, retention, *, upload_hold: bool = False) -> FakeHistory:
        self.log.append("history")
        self.history = FakeHistory(self.log, path, retention, upload_hold)
        return self.history

    def make_pipeline(self, **kwargs) -> FakePipeline:
        self.pipeline = FakePipeline(self.log, **kwargs)
        return self.pipeline

    def make_hotkey(self, chords, callbacks, on_error=None) -> FakeHotkey:
        self.hotkey = FakeHotkey(self.log, chords, callbacks, on_error=on_error)
        return self.hotkey

    def make_app(self, argv) -> FakeQApp:
        self.log.append("qapp")
        self.app = FakeQApp(self.log, self.on_exec)
        return self.app

    def make_bridge(self) -> FakeBridge:
        self.log.append("bridge")
        self.bridge = FakeBridge()
        return self.bridge

    def create_ui(self, app, **kwargs) -> FakeHandles:
        self.log.append("ui.create")
        self.ui_kwargs = kwargs
        self.handles = FakeHandles(self.log, kwargs.get("bridge"))
        on_quit = kwargs.get("on_quit")
        if on_quit is not None:
            self.handles.tray.quit_requested.connect(on_quit)
        return self.handles

    def make_window(self, class_name: str, handlers: MessageHandlers) -> FakeWindow:
        self.log.append("window.create")
        self.window = FakeWindow(self.log, class_name, handlers)
        return self.window

    def autostart_apply(self, enabled: bool, command: str) -> bool:
        self.log.append("autostart")
        self.autostart_calls.append((enabled, command))
        return True

    def autostart_command(self) -> str:
        return '"pythonw.exe" -m spells'

    def detect_cpu_plan(self) -> CpuPlan:
        self.cpu_plan_calls += 1
        if isinstance(self.cpu_plan, Exception):
            raise self.cpu_plan
        return self.cpu_plan

    def make_calibrator(self, **kwargs) -> FakeCalibrator:
        calibrator = FakeCalibrator(self.calibration_speeds, **kwargs)
        self.calibrators.append(calibrator)
        return calibrator

    def model_catalog(self):
        if self.catalog_broken:
            raise CatalogError("catalog.json: broken on purpose")
        return load_catalog() if self.catalog is None else self.catalog

    def deps(self) -> Deps:
        return Deps(
            layout=self.make_layout,
            acquire_single_instance=self.acquire,
            release_single_instance=self.release,
            signal_running_instance=self.signal,
            find_message_window=self.find,
            config_store=self.config_store,
            select_device=self.select_device,
            model_catalog=self.model_catalog,
            supervisor=self.supervisor,
            history=self.make_history,
            pipeline=self.make_pipeline,
            hotkey=self.make_hotkey,
            qapplication=self.make_app,
            placeholder_tray=self.placeholder_tray,
            bridge=self.make_bridge,
            create_ui=self.create_ui,
            message_window=self.make_window,
            autostart_apply=self.autostart_apply,
            autostart_command=self.autostart_command,
            setup_logging=self.setup_logging,
            set_debug_logging=self.set_debug_logging,
            cpu_plan=self.detect_cpu_plan,
            calibrator=self.make_calibrator,
        )

    # Helpers -----------------------------------------------------------------------------

    def handlers(self) -> MessageHandlers:
        assert self.window is not None
        return self.window.handlers

    def settings_file(self) -> Path:
        return self.tmp_path / "cfg" / "settings.json"

    def models_dir(self) -> Path:
        return self.tmp_path / "build" / "cache" / "models"

    def install(self, *model_ids: str) -> None:
        catalog = {model.id: model for model in load_catalog()}
        folder = self.models_dir()
        folder.mkdir(parents=True, exist_ok=True)
        for model_id in model_ids:
            for name in catalog[model_id].files:
                (folder / name).write_bytes(b"")

    def set_languages(self, *codes: str) -> None:
        settings = ConfigStore(self.settings_file()).settings
        self.write_settings(
            replace(settings, general=replace(settings.general, enabled_languages=list(codes)))
        )

    def write_settings(self, settings: Settings) -> None:
        from spells.config import save

        path = self.settings_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        save(settings, path)


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path=tmp_path)


@pytest.fixture
def quiet_logging():
    """Keep a test that configures real logging from leaking into the rest of the suite."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            handler.close()
            root.removeHandler(handler)
    root.handlers = handlers
    root.setLevel(level)


# Arguments ---------------------------------------------------------------------------------


def test_parse_args_recognises_every_flag():
    assert parse_args([]) == app_module.Args()
    assert parse_args(["--quit"]).quit is True
    assert parse_args(["--settings"]).settings is True
    assert parse_args(["--version"]).version is True
    assert parse_args(["--QUIT"]).quit is True


def test_parse_args_keeps_unknown_arguments_without_failing():
    args = parse_args(["--wat", "extra"])
    assert args.unknown == ("--wat", "extra")
    assert args.quit is False


def test_version_prints_and_starts_nothing(world, capsys):
    assert main(["--version"], deps=world.deps()) == 0
    assert app_module.__version__ in capsys.readouterr().out
    assert world.log == []


# Single instance and --quit ------------------------------------------------------------------


def test_second_launch_asks_the_running_instance_to_open_settings(world):
    world.taken = True
    assert main([], deps=world.deps()) == 0
    assert world.signals == [(WINDOW_CLASS, "open-settings")]
    assert "qapp" not in world.log
    assert world.engines is None


def test_settings_flag_reaches_the_running_instance(world):
    world.taken = True
    assert main(["--settings"], deps=world.deps()) == 0
    assert world.signals == [(WINDOW_CLASS, "open-settings")]


def test_settings_flag_opens_settings_in_a_fresh_instance(world):
    world.on_exec = lambda: None
    assert main(["--settings"], deps=world.deps()) == 0
    assert world.handles is not None and world.handles.opened == ["general"]


def test_quit_signals_the_running_instance_and_waits_for_its_window(world):
    world.taken = True
    assert main(["--quit"], deps=world.deps()) == 0
    assert world.signals == [(WINDOW_CLASS, "quit")]
    assert world.window_present is False
    assert world.engines is None


def test_quit_without_a_running_instance_exits_zero(world):
    """A free mutex, not a missing window, is what says nothing runs (spec 19.5)."""
    world.window_present = False
    assert main(["--quit"], deps=world.deps()) == 0
    assert world.signals == []
    assert world.released == [MUTEX_NAME]


def test_quit_waits_for_the_window_of_an_instance_that_is_still_starting(world):
    world.taken = True
    world.window_after_finds = 8  # it is still probing GPUs
    deps, slept = _fake_clock_deps(world)
    assert main(["--quit"], deps=deps) == 0
    assert world.signals == [(WINDOW_CLASS, "quit")]
    assert world.window_present is False
    assert 0 < sum(slept) < app_module.QUIT_TIMEOUT_S


def test_quit_gives_up_when_the_window_never_appears(world, caplog):
    world.taken = True
    world.window_present = False
    deps, slept = _fake_clock_deps(world)
    with caplog.at_level(logging.WARNING, logger="spells.app"):
        assert main(["--quit"], deps=deps) == 0
    assert world.signals == []
    assert sum(slept) >= app_module.QUIT_TIMEOUT_S
    assert "no window" in caplog.text.lower()


def _fake_clock_deps(world: World):
    """Deps on a fake clock, so every wait in these tests is instant."""
    now = [0.0]
    slept: list[float] = []

    def clock() -> float:
        return now[0]

    def sleeper(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    return replace(world.deps(), clock=clock, sleeper=sleeper), slept


def test_quit_gives_up_after_ten_seconds(world):
    world.taken = True
    world.quit_clears_window = False
    deps, slept = _fake_clock_deps(world)
    assert main(["--quit"], deps=deps) == 0
    assert sum(slept) >= app_module.QUIT_TIMEOUT_S
    assert world.window_present is True


def test_a_second_launch_waits_for_the_window_before_signalling(world):
    world.taken = True
    world.window_after_finds = 5
    deps, slept = _fake_clock_deps(world)
    assert main([], deps=deps) == 0
    assert world.signals == [(WINDOW_CLASS, "open-settings")]
    assert 0 < sum(slept) < app_module.WINDOW_WAIT_S


def test_a_second_launch_drops_the_request_when_no_window_appears(world, caplog):
    world.taken = True
    world.window_present = False
    deps, slept = _fake_clock_deps(world)
    with caplog.at_level(logging.WARNING, logger="spells.app"):
        assert main([], deps=deps) == 0
    assert world.signals == []
    assert sum(slept) >= app_module.WINDOW_WAIT_S
    assert "never showed its window" in caplog.text


# Start and shutdown ---------------------------------------------------------------------------


def test_start_order_follows_the_spec(world):
    """The mutex comes before the config store: a second launch never touches settings.json."""
    world.on_exec = lambda: None
    assert main([], deps=world.deps()) == 0
    assert world.log[: world.log.index("app.exec") + 1] == [
        "logging",
        "mutex",
        "config",
        "qapp",
        "placeholder",
        "bridge",
        "gpu",
        "engines.start",
        "history",
        "pipeline.start",
        "hotkey.start",
        "ui.create",
        "placeholder.hide",
        "window.create",
        "autostart",
        "app.exec",
    ]


def test_wiring_hands_each_collaborator_the_bridge_callbacks(world):
    world.on_exec = lambda: None
    main([], deps=world.deps())
    bridge = world.bridge
    assert world.pipeline.kwargs["on_event"] == bridge.on_pipeline_event
    assert world.engines.on_status == bridge.on_engine_status
    assert world.hotkey.on_error == bridge.on_hotkey_error
    assert world.ui_kwargs["bridge"] is bridge
    assert world.ui_kwargs["gpu_selection"] is SELECTION
    assert world.ui_kwargs["log_dir"] == world.layout.log_dir
    assert world.hotkey.callbacks == ("callbacks",)
    settings = world.config.settings
    assert world.hotkey.chords == [settings.general.main_chord, *settings.general.language_chords]
    assert world.history.retention == settings.history.retention
    assert world.history.upload_hold is settings.upload.enabled
    assert world.engines.paths.whisper_model == world.layout.whisper_model


def test_the_history_holds_unsent_rows_from_the_start_when_uploading_is_on(world):
    settings = ConfigStore(world.settings_file()).settings
    world.write_settings(
        replace(
            settings,
            upload=replace(settings.upload, enabled=True, url="https://example.com/spells"),
        )
    )
    world.on_exec = lambda: None
    main([], deps=world.deps())
    assert world.history.upload_hold is True


def test_the_pipeline_ends_recordings_through_the_hotkey_thread(world):
    world.on_exec = lambda: None
    main([], deps=world.deps())
    world.pipeline.end_recording(7)
    assert world.hotkey.ended == [7]


def test_shutdown_order_and_idempotence(world):
    world.on_exec = lambda: world.handles.tray.quit_requested.emit()
    assert main([], deps=world.deps()) == 0
    tail = [
        step
        for step in world.log[world.log.index("app.exec") + 1 :]
        if not step.startswith("tray.")
    ]
    assert tail[:7] == [
        "ui.close",
        "pipeline.stop",
        "hotkey.stop",
        "engines.stop",
        "history.close",
        "window.destroy",
        "mutex.release",
    ]
    assert world.handles.closed == 1
    assert world.pipeline.stopped == 1
    assert world.hotkey.stopped == 1
    assert world.engines.stopped == 1
    assert world.history.closed == 1
    assert world.window.destroyed == 1
    assert world.released == [MUTEX_NAME]
    assert world.app.quits == 1


def test_a_second_quit_request_changes_nothing(world):
    def quit_twice() -> None:
        world.handles.tray.quit_requested.emit()
        world.handles.tray.quit_requested.emit()

    world.on_exec = quit_twice
    main([], deps=world.deps())
    assert world.pipeline.stopped == 1
    assert world.engines.stopped == 1
    assert world.released == [MUTEX_NAME]


def test_end_session_shuts_down_before_it_returns(world):
    done: list[list[str]] = []

    def end_session() -> None:
        handlers = world.handlers()
        assert handlers.on_query_end_session(0) is True
        thread = threading.Thread(target=lambda: handlers.on_end_session(0x80000000))
        thread.start()
        thread.join(10.0)
        assert not thread.is_alive()
        done.append(list(world.log))

    world.on_exec = end_session
    main([], deps=world.deps())
    during = done[0]
    for step in ("pipeline.stop", "hotkey.stop", "engines.stop", "history.close", "mutex.release"):
        assert step in during
    assert world.pipeline.stopped == 1
    assert world.engines.stopped == 1


def test_the_starting_icon_covers_the_gpu_probe_and_goes_once_the_ui_is_up(world):
    world.on_exec = lambda: None
    main([], deps=world.deps())
    assert world.probe_saw_placeholder == [True]
    assert world.log.index("placeholder") < world.log.index("gpu")
    assert world.log.index("placeholder.hide") > world.log.index("ui.create")
    assert (world.placeholder.hidden, world.placeholder.deleted) == (1, 1)


def test_a_failed_start_returns_one_and_releases_the_mutex(world):
    def boom(*args, **kwargs):
        raise RuntimeError("no engines today")

    deps = replace(world.deps(), supervisor=boom)
    assert main([], deps=deps) == 1
    assert world.released == [MUTEX_NAME]
    assert world.placeholder.hidden == 1
    assert world.app.quits == 0


def test_end_session_waits_for_a_quit_already_running(world):
    gate = threading.Event()
    entered = threading.Event()
    finished: list[list[str]] = []

    def run() -> None:
        world.history.closing = entered
        world.history.close_gate = gate
        quitter = threading.Thread(target=world.handles.tray.quit_requested.emit)
        quitter.start()
        assert entered.wait(10.0), "the quit never reached the teardown"
        opener = threading.Timer(0.2, gate.set)
        opener.start()
        world.handlers().on_end_session(0x80000000)
        finished.append(list(world.log))
        opener.cancel()
        quitter.join(10.0)

    world.on_exec = run
    main([], deps=world.deps())
    assert "mutex.release" in finished[0], "the handler returned mid-teardown"
    assert world.history.closed == 1


def test_end_session_gives_up_after_its_cap(world, monkeypatch, caplog):
    monkeypatch.setattr(app_module, "END_SESSION_WAIT_S", 0.05)
    gate = threading.Event()
    entered = threading.Event()
    finished: list[list[str]] = []

    def run() -> None:
        world.history.closing = entered
        world.history.close_gate = gate
        quitter = threading.Thread(target=world.handles.tray.quit_requested.emit)
        quitter.start()
        assert entered.wait(10.0)
        with caplog.at_level(logging.WARNING, logger="spells.app"):
            world.handlers().on_end_session(0)
        finished.append(list(world.log))
        gate.set()
        quitter.join(10.0)

    world.on_exec = run
    main([], deps=world.deps())
    assert "mutex.release" not in finished[0]
    assert "did not finish" in caplog.text


def test_the_window_thread_survives_a_window_that_cannot_be_created(world):
    def boom(class_name, handlers):
        raise OSError("no window station")

    deps = replace(world.deps(), message_window=boom)
    world.on_exec = lambda: None
    assert main([], deps=deps) == 0
    assert world.engines.stopped == 1


# Hidden window commands ------------------------------------------------------------------------


def test_copydata_open_settings_is_marshalled_to_the_qt_thread(world):
    world.on_exec = lambda: world.handlers().on_copydata("open-settings")
    main([], deps=world.deps())
    assert world.handles.opened == ["general"]
    assert world.handles.tray.open_requested.emitted == [("general",)]


def test_copydata_quit_shuts_down(world):
    world.on_exec = lambda: world.handlers().on_copydata("quit")
    main([], deps=world.deps())
    assert world.pipeline.stopped == 1
    assert world.app.quits == 1


def test_an_unknown_command_is_ignored(world, caplog):
    world.on_exec = lambda: world.handlers().on_copydata("rm -rf")
    with caplog.at_level(logging.WARNING, logger="spells.app"):
        main([], deps=world.deps())
    assert world.pipeline.stopped == 1  # only the shutdown after exec
    assert world.handles.opened == []
    assert "rm -rf" in caplog.text


def test_resume_and_unlock_reinstall_the_hook(world):
    def wake() -> None:
        world.handlers().on_resume()
        world.handlers().on_session_unlock()

    world.on_exec = wake
    main([], deps=world.deps())
    assert world.hotkey.reinstalls == 2


# Settings reactions ------------------------------------------------------------------------------


def test_autostart_is_applied_at_start_and_on_change(world):
    def toggle() -> None:
        world.config.update(lambda s: replace(s, general=replace(s.general, autostart=False)))

    world.on_exec = toggle
    main([], deps=world.deps())
    assert world.autostart_calls == [(True, '"pythonw.exe" -m spells'), (False, '"pythonw.exe" -m spells')]


def test_idle_unload_minutes_reach_the_supervisor(world):
    def change() -> None:
        world.config.update(
            lambda s: replace(s, general=replace(s.general, idle_unload_minutes=12))
        )

    world.on_exec = change
    main([], deps=world.deps())
    assert world.engines.idle_minutes == [0, 12]


def test_retention_is_left_to_the_settings_dialog(world):
    def change() -> None:
        world.config.update(lambda s: replace(s, history=replace(s.history, retention="7d")))

    world.on_exec = change
    main([], deps=world.deps())
    assert world.history.retentions == []


def test_debug_logging_takes_effect_at_once(world):
    def change() -> None:
        world.config.update(
            lambda s: replace(s, diagnostics=replace(s.diagnostics, debug_logging=True))
        )
        world.config.update(
            lambda s: replace(s, diagnostics=replace(s.diagnostics, debug_logging=False))
        )

    world.on_exec = change
    main([], deps=world.deps())
    assert world.debug_levels == [True, False]


def test_an_unchanged_setting_is_not_reapplied(world):
    def touch() -> None:
        world.config.update(lambda s: replace(s, cleanup=replace(s.cleanup, enabled=False)))

    world.on_exec = touch
    main([], deps=world.deps())
    assert len(world.autostart_calls) == 1
    assert world.engines.idle_minutes == [0]
    assert world.debug_levels == []


# GPU selection ---------------------------------------------------------------------------------


def test_the_gpu_choice_is_probed_with_the_cache_and_the_override(world):
    settings = ConfigStore(world.settings_file()).settings
    world.write_settings(
        replace(
            settings,
            diagnostics=replace(
                settings.diagnostics, gpu_device_override=3, gpu_device_index=0, gpu_device_name="old"
            ),
        )
    )
    world.on_exec = lambda: None
    main([], deps=world.deps())
    call = world.gpu_calls[0]
    assert call["llama_server"] == world.layout.llama_exe("vulkan")
    assert call["whisper_server"] == world.layout.whisper_exe("vulkan")
    assert call["override"] == 3
    assert call["cached"] == (0, "old")


def test_the_gpu_cache_is_written_only_when_it_changed(world):
    world.on_exec = lambda: None
    main([], deps=world.deps())
    diagnostics = world.config.settings.diagnostics
    assert (diagnostics.gpu_device_index, diagnostics.gpu_device_name) == (1, RTX)
    first_writes = world.config.updates

    second = World(tmp_path=world.tmp_path)
    second.on_exec = lambda: None
    main([], deps=second.deps())
    assert second.config.updates == 0
    assert first_writes >= 1


def test_no_gpu_keeps_the_cache_empty(world):
    world.selection = GpuSelection(raw_index=None, name="", memory_mb=0, devices=())
    world.on_exec = lambda: None
    main([], deps=world.deps())
    diagnostics = world.config.settings.diagnostics
    assert diagnostics.gpu_device_index is None
    assert diagnostics.gpu_device_name is None


# First run and warnings ---------------------------------------------------------------------------


def test_first_run_shows_the_welcome_page_and_writes_the_settings_file(world):
    world.on_exec = lambda: None
    main([], deps=world.deps())
    assert world.ui_kwargs["first_run"] is True
    assert world.settings_file().exists()


def test_a_second_start_is_not_a_first_run(world):
    world.on_exec = lambda: None
    main([], deps=world.deps())
    second = World(tmp_path=world.tmp_path)
    second.on_exec = lambda: None
    main([], deps=second.deps())
    assert second.ui_kwargs["first_run"] is False


def test_a_settings_load_notice_becomes_a_tray_warning(world):
    path = world.settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ broken", encoding="utf-8")
    world.on_exec = lambda: None
    main([], deps=world.deps())
    assert any(warning for _text, warning in world.handles.tray.messages)


def test_a_missing_cleanup_model_is_a_balloon_but_not_a_second_tooltip_line(world):
    """The engine's no_model state already warns in the tray; Windows cuts a tooltip at 127."""
    models = world.tmp_path / "build" / "cache" / "models"
    models.mkdir(parents=True)
    (models / paths.WHISPER_DEV_MODEL_NAME).write_bytes(b"")
    world.on_exec = lambda: None
    main([], deps=world.deps())
    texts = [text for text, _warning in world.handles.tray.messages]
    assert any("cleanup" in text.lower() for text in texts)
    assert not any(world.handles.tray.extra_warnings)
    # The supervisor gets no model at all, so it reports llama as FAILED with no_model.
    assert world.layout.llama_model is None
    assert world.engines.paths.llama_model is None


def test_only_a_missing_speech_model_becomes_the_lasting_tray_warning(world):
    world.on_exec = lambda: None
    main([], deps=world.deps())
    texts = [text for text, _warning in world.handles.tray.messages]
    assert any("cleanup" in text.lower() for text in texts)
    assert any(world.layout.whisper_notice == text for text in texts)
    standing = world.handles.tray.extra_warnings
    assert standing and standing[-1] == world.layout.whisper_notice
    assert "cleanup" not in standing[-1].lower()


# Models for the dictation languages (spec 13, 15, batch 5) --------------------------------------

WHISPER_ID = "whisper-large-v3-turbo-q8_0"
GEMMA_ID = "gemma-4-e2b-it-q4_0"
QWEN_ID = "qwen3.5-4b-q4_k_m"
ALL_IDS = tuple(model.id for model in load_catalog())
NGRAM = (
    "--spec-type",
    "ngram-simple",
    "--spec-ngram-simple-size-n",
    "2",
    "--spec-ngram-simple-size-m",
    "24",
)
INTEGRATED = GpuSelection(
    raw_index=0,
    name="AMD Radeon(TM) 610M",
    memory_mb=16000,
    devices=(GpuDevice(0, "Vulkan0", "AMD Radeon(TM) 610M", 16000, 15000, uma=True),),
)
NO_DEVICE = GpuSelection(raw_index=None, name="", memory_mb=0, devices=())


def change_languages(world, *codes: str):
    def change() -> None:
        world.config.update(
            lambda s: replace(s, general=replace(s.general, enabled_languages=list(codes)))
        )

    return change


def test_the_models_for_the_enabled_languages_reach_the_supervisor(world):
    world.install(*ALL_IDS)
    world.cpu_plan = CpuPlan(threads=8, affinity_mask=0x5555)
    world.on_exec = lambda: None
    main([], deps=world.deps())
    engines = world.engines
    models = world.models_dir()
    assert engines.paths.whisper_model == models / "Qwen3-ASR-0.6B-Q4_K_M.gguf"
    assert engines.paths.whisper_runtime == "llama-asr"
    assert engines.paths.whisper_args == ("--mmproj", "{extra:0}", "--no-webui")
    assert engines.paths.whisper_extra_files == (models / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",)
    assert engines.paths.whisper_languages == ("de", "en")
    assert engines.paths.extra_speech == (
        SpeechEngine(
            "whisper-server", models / "ggml-large-v3-turbo-sq-flutra-v2-q8_0.bin", (), (),
            ("sq",),
        ),
    )
    assert engines.paths.llama_model == models / "Qwen3.5-4B-Q4_K_M.gguf"
    assert engines.paths.llama_args == NGRAM
    assert engines.paths.llama_languages == ("de", "en", "sq")
    assert engines.cpu_only is False
    assert engines.cpu_cleanup_allowed is False
    assert engines.gpu is SELECTION
    assert engines.cpu_plan == CpuPlan(threads=8, affinity_mask=0x5555)
    assert world.cpu_plan_calls == 1


def test_a_computer_without_a_gpu_runs_a_model_that_is_fast_enough_on_the_processor(world):
    world.install(*ALL_IDS)
    world.selection = NO_DEVICE
    world.on_exec = lambda: None
    main([], deps=world.deps())
    engines = world.engines
    assert engines.paths.llama_model == world.models_dir() / "gemma-4-E2B-it-Q4_0.gguf"
    assert engines.cpu_only is True
    assert engines.cpu_cleanup_allowed is True


def test_an_integrated_gpu_counts_as_the_processor_until_measured(world):
    world.install(*ALL_IDS)
    world.selection = INTEGRATED
    world.calibration_speeds = {model.id: (3000, 1300) for model in load_catalog()}
    world.on_exec = join_calibration
    main([], deps=world.deps())
    assert world.engines.cpu_only is True
    assert world.engines.gpu is INTEGRATED
    assert world.engines.paths.llama_model.name == "gemma-4-E2B-it-Q4_0.gguf"


def test_a_language_change_hands_the_supervisor_the_new_models(world):
    world.install(*ALL_IDS)
    world.on_exec = change_languages(world, "en", "de")
    main([], deps=world.deps())
    assert len(world.engines.model_changes) == 1
    paths, cpu_only, cpu_cleanup_allowed = world.engines.model_changes[0]
    assert paths.llama_model == world.models_dir() / "gemma-4-E2B-it-Q4_0.gguf"
    assert paths.llama_extra_files == (world.models_dir() / "mtp-gemma-4-E2B-it-Q4_0.gguf",)
    assert paths.whisper_model == world.models_dir() / "Qwen3-ASR-0.6B-Q4_K_M.gguf"
    assert paths.extra_speech == ()
    assert (cpu_only, cpu_cleanup_allowed) == (False, False)


def test_a_language_change_that_keeps_the_models_changes_nothing(world):
    world.install(*ALL_IDS)

    def reorder() -> None:
        change_languages(world, "sq", "de", "en")()
        change_languages(world, "de", "sq", "en")()

    world.on_exec = reorder
    main([], deps=world.deps())
    assert world.engines.model_changes == []


def test_a_setting_other_than_the_languages_never_touches_the_models(world):
    world.install(*ALL_IDS)

    def touch() -> None:
        world.config.update(lambda s: replace(s, cleanup=replace(s.cleanup, enabled=False)))

    world.on_exec = touch
    main([], deps=world.deps())
    assert world.engines.model_changes == []


def test_gpu_override_rebuilds_cpu_plan_and_caches_the_selected_device(world):
    world.install(*ALL_IDS)
    world.selection = replace(INTEGRATED, devices=INTEGRATED.devices + SELECTION.devices)
    world.calibration_speeds = {model.id: (3000, 1300) for model in load_catalog()}

    def switch():
        join_calibration()
        assert world.engines.cpu_only is True
        world.config.update(lambda s: replace(
            s, diagnostics=replace(s.diagnostics, gpu_device_override=SELECTION.raw_index)))
        assert world.engines.cpu_only is False
        assert world.engines.gpu.raw_index == SELECTION.raw_index
        assert world.engines.paths.llama_model.name == "Qwen3.5-4B-Q4_K_M.gguf"
        assert world.pipeline.kwargs["selection"].hardware.value == "gpu"
        assert world.config.settings.diagnostics.gpu_device_name == RTX
        assert len(world.gpu_calls) == 1
        change_languages(world, "en", "de")()
        assert world.engines.cpu_only is False

    world.on_exec = switch
    assert main([], deps=world.deps()) == 0


def test_returning_to_automatic_chooses_the_best_gpu_without_the_manual_cache(world):
    world.install(*ALL_IDS)
    small = GpuDevice(0, "Vulkan0", "Small GPU", 4000, 3900, uma=False)
    large = GpuDevice(1, "Vulkan1", "Large GPU", 16000, 15000, uma=False)
    world.selection = GpuSelection(1, large.name, large.memory_mb, (small, large))

    def switch():
        world.config.update(lambda s: replace(
            s, diagnostics=replace(s.diagnostics, gpu_device_override=0)))
        assert world.engines.gpu.name == "Small GPU"
        world.config.update(lambda s: replace(
            s, diagnostics=replace(s.diagnostics, gpu_device_override=None)))
        assert world.engines.cpu_only is False
        assert world.engines.gpu.name == "Large GPU"
        assert world.config.settings.diagnostics.gpu_device_name == "Large GPU"
        assert len(world.gpu_calls) == 1

    world.on_exec = switch
    assert main([], deps=world.deps()) == 0


def test_languages_no_installed_cleanup_model_suits_start_llama_without_a_model(world):
    world.install(WHISPER_ID, GEMMA_ID)
    world.set_languages("fr")
    world.on_exec = lambda: None
    main([], deps=world.deps())
    assert world.engines.paths.llama_model is None
    texts = [text for text, _warning in world.handles.tray.messages]
    assert app_module.NO_SUITABLE_CLEANUP in texts
    assert not any(world.handles.tray.extra_warnings)


def test_a_chosen_cleanup_model_replaces_the_layouts_cleanup_notice(world):
    world.install(*ALL_IDS)
    world.on_exec = lambda: None
    main([], deps=world.deps())
    assert "several models" in world.layout.cleanup_notice
    texts = [text for text, _warning in world.handles.tray.messages]
    assert not any("cleanup" in text.lower() for text in texts)


def test_without_catalog_models_the_layout_models_are_kept(world):
    folder = world.models_dir()
    folder.mkdir(parents=True)
    (folder / paths.WHISPER_DEV_MODEL_NAME).write_bytes(b"")
    (folder / "qwen2.5-0.5b-instruct-q4_k_m.gguf").write_bytes(b"")
    world.on_exec = lambda: None
    main([], deps=world.deps())
    assert world.engines.paths == world.layout.engine_paths()
    assert world.engines.cpu_only is False


def test_an_unreadable_catalog_keeps_todays_models_and_builds(world, caplog):
    world.install(*ALL_IDS)
    world.catalog_broken = True
    world.selection = NO_DEVICE
    world.on_exec = change_languages(world, "en")
    main([], deps=world.deps())
    assert world.engines.paths == world.layout.engine_paths()
    assert (world.engines.cpu_only, world.engines.cpu_cleanup_allowed) == (False, False)
    assert world.engines.model_changes == []
    assert "model catalog could not be read" in caplog.text


def test_a_second_speech_model_gets_its_own_engine(world, caplog):
    raw = {
        "schema_version": 1,
        "models": [
            {
                "id": "whisper-main",
                "kind": "asr",
                "display_name": "Whisper",
                "file": "main.bin",
                "size_bytes": 1,
                "license": "MIT",
                "runtime": "whisper-server",
                "languages": {"en": 87, "de": 85, "sq": 48},
                "hardware": {"gpu": 300, "cpu": 5000},
            },
            {
                "id": "whisper-sq",
                "kind": "asr",
                "display_name": "Albanian Whisper",
                "file": "sq.bin",
                "size_bytes": 1,
                "license": "MIT",
                "runtime": "whisper-server",
                "languages": {"sq": 90},
                "hardware": {"gpu": 300, "cpu": 5000},
                "engine_args": ["--beam-size", "5"],
            },
        ],
    }
    world.catalog = parse_catalog(raw)
    folder = world.models_dir()
    folder.mkdir(parents=True)
    (folder / "main.bin").write_bytes(b"")
    (folder / "sq.bin").write_bytes(b"")
    world.on_exec = lambda: None
    with caplog.at_level(logging.INFO):
        main([], deps=world.deps())
    assert world.engines.paths.whisper_model == folder / "main.bin"
    assert world.engines.paths.whisper_languages == ("de", "en")
    assert world.engines.paths.extra_speech == (
        SpeechEngine("whisper-server", folder / "sq.bin", ("--beam-size", "5"), (), ("sq",)),
    )
    assert "whisper-sq (whisper-server) for sq" in caplog.text
    assert "not supported" not in caplog.text
    texts = [text for text, _warning in world.handles.tray.messages]
    assert not any("speech model" in text for text in texts)
    assert not any(world.handles.tray.extra_warnings)


def test_a_failed_processor_probe_leaves_the_engines_unpinned(world, caplog):
    world.cpu_plan = OSError("no topology")
    world.on_exec = lambda: None
    main([], deps=world.deps())
    assert world.engines.cpu_plan == CpuPlan()
    assert "processor topology could not be read" in caplog.text


# Logging -------------------------------------------------------------------------------------------


def test_the_log_carries_no_transcript_text_at_info(world, quiet_logging):
    world.real_logging = True
    secret = "zebra crossing at midnight"

    def leak() -> None:
        event = PipelineEvent(
            pill=PillState.PROCESSING,
            tray=TrayState.PROCESSING,
            text=secret,
            notice_text=secret,
            notification=secret,
        )
        world.pipeline.on_event(event)
        logging.getLogger("spells.pipeline").debug("delivered %s", secret)

    world.on_exec = leak
    main([], deps=world.deps())
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = (world.layout.log_dir / app_module.LOG_NAME).read_text(encoding="utf-8", errors="replace")
    assert secret not in text
    assert "spells.app" in text
    assert "MainThread" in text


def test_debug_logging_from_the_settings_raises_the_level(world, quiet_logging):
    """The stored setting reaches logging through the same seam the checkbox uses."""
    settings = ConfigStore(world.settings_file()).settings
    world.write_settings(
        replace(settings, diagnostics=replace(settings.diagnostics, debug_logging=True))
    )
    world.real_logging = True
    world.on_exec = lambda: None
    main([], deps=world.deps())
    assert logging.getLogger().level == logging.DEBUG


# --- the online installer's language choice, used once (B5-32) ---


def _seed_file(world: World, text: str) -> Path:
    """Write what the online installer leaves beside Spells.exe, which is layout.root here."""
    path = world.tmp_path / app_module.FIRST_RUN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_first_run_seeds_the_languages_the_installer_asked_for(world):
    _seed_file(world, '{"enabled_languages": ["en", "de"]}')
    world.on_exec = lambda: None

    main([], deps=world.deps())

    assert world.config.settings.general.enabled_languages == ["en", "de"]


def test_the_seeded_languages_reach_the_settings_file_on_disk(world):
    _seed_file(world, '{"enabled_languages": ["sq"]}')
    world.on_exec = lambda: None

    main([], deps=world.deps())

    saved, notice = load(world.settings_file())
    assert notice is None
    assert saved.general.enabled_languages == ["sq"]


def test_without_the_file_the_first_run_keeps_the_default_languages(world):
    world.on_exec = lambda: None

    main([], deps=world.deps())

    assert world.config.settings.general.enabled_languages == ["en", "de", "sq"]


def test_a_malformed_seed_file_never_stops_the_start(world):
    _seed_file(world, "{not json")
    world.on_exec = lambda: None

    assert main([], deps=world.deps()) == 0
    assert world.config.settings.general.enabled_languages == ["en", "de", "sq"]


def test_a_seed_file_naming_only_unknown_codes_changes_nothing(world):
    _seed_file(world, '{"enabled_languages": ["klingon"]}')
    world.on_exec = lambda: None

    main([], deps=world.deps())

    assert world.config.settings.general.enabled_languages == ["en", "de", "sq"]


def test_the_seed_file_is_ignored_once_a_settings_file_exists(world):
    world.on_exec = lambda: None
    main([], deps=world.deps())

    _seed_file(world, '{"enabled_languages": ["sq"]}')
    second = World(tmp_path=world.tmp_path)
    second.on_exec = lambda: None
    main([], deps=second.deps())

    assert second.ui_kwargs["first_run"] is False
    assert second.config.settings.general.enabled_languages == ["en", "de", "sq"]


def test_first_run_seed_is_a_pure_function_of_the_file_beside_the_executable(tmp_path):
    (tmp_path / app_module.FIRST_RUN_FILE).write_text(
        '{"enabled_languages": ["de"]}', encoding="utf-8")

    seeded = app_module.first_run_seed(default_settings(), tmp_path)

    assert seeded.general.enabled_languages == ["de"]
    assert app_module.first_run_seed(default_settings(), tmp_path / "elsewhere") \
        == default_settings()


# Measuring an integrated GPU -----------------------------------------------------------


def join_calibration() -> None:
    for thread in threading.enumerate():
        if thread.name == "spells-calibration":
            thread.join(10)


def test_an_integrated_gpu_is_measured_once_and_moves_to_it_when_faster(world):
    world.install(*ALL_IDS)
    world.selection = INTEGRATED
    world.on_exec = join_calibration
    main([], deps=world.deps())
    assert len(world.calibrators) == 1
    measured = world.calibrators[0].measured
    assert measured
    assert len(measured) == len(set(measured))
    assert world.engines.cpu_only is False
    stored = calibrate.load(calibrate.store_path(world.settings_file()), INTEGRATED.name)
    assert set(stored) == set(measured)


def test_an_integrated_gpu_that_loses_stays_on_the_processor(world):
    world.install(*ALL_IDS)
    world.selection = INTEGRATED
    world.calibration_speeds = {model.id: (3000, 1300) for model in load_catalog()}
    world.on_exec = join_calibration
    main([], deps=world.deps())
    assert world.calibrators[0].measured
    assert world.engines.cpu_only is True
    assert all(change[1] is True for change in world.engines.model_changes)


def test_stored_measurements_are_used_at_start_without_measuring_again(world):
    world.install(*ALL_IDS)
    world.selection = INTEGRATED
    world.on_exec = join_calibration
    main([], deps=world.deps())
    first = list(world.calibrators[0].measured)
    later = World(tmp_path=world.tmp_path, selection=INTEGRATED, on_exec=join_calibration)
    main([], deps=later.deps())
    assert first
    assert later.calibrators == []
    assert later.engines.cpu_only is False
    assert later.engines.model_changes == []


def test_a_discrete_gpu_is_never_measured(world):
    world.install(*ALL_IDS)
    world.on_exec = join_calibration
    main([], deps=world.deps())
    assert world.calibrators == []


def test_quitting_cancels_a_measurement(world):
    world.install(*ALL_IDS)
    world.selection = INTEGRATED
    world.calibration_speeds = {}
    gate = threading.Event()
    original = FakeCalibrator.calibrate

    def slow(self, choice):
        gate.wait(5)
        return original(self, choice)

    world.on_exec = lambda: None
    FakeCalibrator.calibrate = slow
    try:
        main([], deps=world.deps())
    finally:
        gate.set()
        FakeCalibrator.calibrate = original
        join_calibration()
    assert world.calibrators[0].cancelled is True


# The writing engine ----------------------------------------------------------------------


def test_the_processor_writes_with_the_strong_model_in_an_engine_of_its_own(world):
    world.install(*ALL_IDS)
    world.selection = NO_DEVICE
    world.on_exec = lambda: None
    main([], deps=world.deps())
    paths = world.engines.paths
    assert paths.llama_model.name == "gemma-4-E2B-it-Q4_0.gguf"
    assert paths.writer_model == world.models_dir() / "Qwen3.5-4B-Q4_K_M.gguf"


def test_a_graphics_card_writes_and_cleans_with_one_engine(world):
    world.install(*ALL_IDS)
    world.on_exec = lambda: None
    main([], deps=world.deps())
    paths = world.engines.paths
    assert paths.llama_model.name == "Qwen3.5-4B-Q4_K_M.gguf"
    assert paths.writer_model is None


def test_with_only_the_small_model_there_is_one_text_engine(world):
    world.install(WHISPER_ID, GEMMA_ID, "gemma-4-e2b-it-q4_0-compose")
    world.selection = NO_DEVICE
    world.on_exec = lambda: None
    main([], deps=world.deps())
    assert world.engines.paths.writer_model is None
