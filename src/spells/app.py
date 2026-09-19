"""Process entry point: arguments, single instance, wiring, shutdown (spec 15, 19.5).

One process per user (spec 5.1). `main()` parses the arguments, sets up logging, takes the
named mutex, builds every module in the order the spec asks for, and hands control to the Qt
event loop. A second launch never starts anything: it signals the running instance to open
settings and exits, and `Spells.exe --quit` asks it to exit and waits for its hidden window
to disappear (spec 19.5, what the installer and the uninstaller run).

Threads: the Qt event loop owns the widgets, the hidden top-level window owns its own thread
because it is thread-affine (spec 5.1), and the pipeline, hotkey and engine threads belong to
their own modules. Everything those threads hand to the UI goes through `UiBridge`, whose
callbacks are handed out here before the threads start. Nothing in this module touches a
widget from another thread: a command arriving on the hidden window is marshalled to the Qt
thread through a tray signal, which Qt queues because it crosses a thread boundary.

Qt lives in `spells.ui`; the QApplication and the Qt message handler are built here in a lazy
factory so the unit tests can run this module without Qt at all.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from spells import __version__, autostart, calibrate, paths, updates
from spells.config import (
    FIRST_RUN_FILE,
    ConfigStore,
    Settings,
    first_run_languages,
    with_enabled_languages,
)
from spells.engines import EnginePaths, EngineSupervisor
from spells.gpu import NO_GPU, GpuSelection, select_device
from spells.history import HistoryStore
from spells.hotkey import HotkeyThread
from spells.modelcatalog import (
    CatalogModel,
    Hardware,
    ModelKind,
    Selection,
    installed_ids,
    load_catalog,
    select_models,
)
from spells.models import CpuPlan, Engine
from spells.paths import Layout
from spells.pipeline import Pipeline
from spells.win32.cpu import detect_cpu_plan
from spells.win32.instance import (
    acquire_single_instance,
    find_message_window,
    release_single_instance,
    signal_running_instance,
)
from spells.win32.msgwindow import MessageHandlers, MessageWindow

log = logging.getLogger(__name__)

APP_NAME = "Spells"
# The installer's AppMutex and the [Code] PrepareToInstall step use this exact name (19.5).
MUTEX_NAME = "Spells"
WINDOW_CLASS = "SpellsMessageWindow"
OPEN_SETTINGS = "open-settings"
QUIT = "quit"

QUIT_TIMEOUT_S = 10.0
QUIT_POLL_S = 0.1
# An instance that holds the mutex may still be starting: its window appears about 1.5 s in,
# longer behind a first-run GPU probe, so both CLI paths wait for it (spec 19.5).
WINDOW_WAIT_S = 10.0
# How long WM_ENDSESSION waits for a teardown another thread already started (spec 19.5).
END_SESSION_WAIT_S = 3.0
WINDOW_READY_TIMEOUT_S = 5.0
WINDOW_JOIN_TIMEOUT_S = 5.0

LOG_NAME = "app.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # spec 15: five files of up to 5 MB, the live one included
LOG_KEEP = 5
LOG_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"

# QtMsgType order: debug, warning, critical, fatal, info.
QT_LEVELS = (logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL, logging.INFO)


# Arguments ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Args:
    """The command line (spec 19.5). Unknown arguments are logged, never fatal."""

    quit: bool = False
    settings: bool = False
    version: bool = False
    unknown: tuple[str, ...] = ()


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """Parse by hand: argparse writes to a console this process does not have."""
    raw = list(sys.argv[1:] if argv is None else argv)
    flags = {"--quit": False, "--settings": False, "--version": False}
    unknown: list[str] = []
    for item in raw:
        key = item.strip().lower()
        if key in flags:
            flags[key] = True
        else:
            unknown.append(item)
    return Args(
        quit=flags["--quit"],
        settings=flags["--settings"],
        version=flags["--version"],
        unknown=tuple(unknown),
    )


# Logging (spec 15, 17) ------------------------------------------------------------------------


def setup_logging(log_dir: Path, *, debug: bool = False) -> logging.Handler | None:
    """A rotating app log and nothing else: there is no console to write to.

    INFO by default, DEBUG when Diagnostics asks for it. Transcript text is never logged at
    INFO anywhere in the app (spec 17); the debug level is the user's explicit choice.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    handler: logging.Handler | None = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_dir / LOG_NAME,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_KEEP - 1,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    except OSError:
        # Without a log file the app still runs; it just says nothing.
        pass
    _install_excepthooks()
    return handler


def set_debug_logging(enabled: bool) -> None:
    logging.getLogger().setLevel(logging.DEBUG if enabled else logging.INFO)


def _install_excepthooks() -> None:
    def on_exception(exc_type, exc, traceback) -> None:
        log.critical("Unhandled exception", exc_info=(exc_type, exc, traceback))

    def on_thread_exception(args) -> None:
        log.critical(
            "Unhandled exception in %s",
            getattr(args.thread, "name", "a thread"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = on_exception
    threading.excepthook = on_thread_exception


def _qt_message(mode: Any, context: Any, message: str) -> None:
    index = int(mode) if 0 <= int(mode) < len(QT_LEVELS) else 4
    logging.getLogger("qt").log(QT_LEVELS[index], "%s", message)


# Lazy Qt factories ------------------------------------------------------------------------------


def _default_qapplication(argv: list[str]) -> Any:
    from PySide6 import QtCore
    from PySide6.QtWidgets import QApplication

    QtCore.qInstallMessageHandler(_qt_message)
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setQuitOnLastWindowClosed(False)  # a tray app outlives its dialogs
    return app


def _default_placeholder_tray() -> Any:
    """A Starting tray icon for the seconds the GPU probe blocks (spec 13, 14.1).

    The real tray only exists after create_ui, which waits for the probe, so without this
    nothing at all is on screen while Spells starts.
    """
    from PySide6 import QtWidgets

    from spells.ui import theme
    from spells.ui.icons import TrayIconState, tray_icon

    icon = QtWidgets.QSystemTrayIcon(
        tray_icon(TrayIconState.STARTING, light_taskbar=theme.taskbar_is_light())
    )
    icon.setToolTip(f"{APP_NAME}: starting")
    icon.show()
    return icon


def _default_bridge() -> Any:
    from spells.ui import UiBridge

    return UiBridge()


def _default_create_ui(app: Any, **kwargs: Any) -> Any:
    from spells.ui import create_ui

    return create_ui(app, **kwargs)


# Dependencies ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Deps:
    """Every collaborator the entry point builds; the unit tests replace them with fakes."""

    layout: Callable[[], Layout] = paths.resolve
    acquire_single_instance: Callable[[str], bool] = acquire_single_instance
    release_single_instance: Callable[[str], None] = release_single_instance
    signal_running_instance: Callable[[str, str], bool] = signal_running_instance
    find_message_window: Callable[[str], int] = find_message_window
    config_store: Callable[[Path], ConfigStore] = ConfigStore
    select_device: Callable[..., GpuSelection] = select_device
    model_catalog: Callable[[], Sequence[CatalogModel]] = load_catalog
    supervisor: Callable[..., Any] = EngineSupervisor
    cpu_plan: Callable[[], CpuPlan] = detect_cpu_plan
    history: Callable[[Path, str], Any] = HistoryStore
    pipeline: Callable[..., Any] = Pipeline
    hotkey: Callable[..., Any] = HotkeyThread
    qapplication: Callable[[list[str]], Any] = _default_qapplication
    placeholder_tray: Callable[[], Any] = _default_placeholder_tray
    bridge: Callable[[], Any] = _default_bridge
    create_ui: Callable[..., Any] = _default_create_ui
    message_window: Callable[[str, MessageHandlers], Any] = MessageWindow
    autostart_apply: Callable[[bool, str], bool] = autostart.apply
    autostart_command: Callable[[], str] = autostart.current_command
    setup_logging: Callable[..., Any] = setup_logging
    set_debug_logging: Callable[[bool], None] = set_debug_logging
    calibrator: Callable[..., Any] = calibrate.Calibrator
    clock: Callable[[], float] = time.perf_counter
    sleeper: Callable[[float], None] = time.sleep


# Entry point ---------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None, *, deps: Deps | None = None) -> int:
    args = parse_args(argv)
    deps = deps if deps is not None else Deps()
    if args.version:
        _write_line(f"{APP_NAME} {__version__}")
        return 0
    layout = deps.layout()
    deps.setup_logging(layout.log_dir)
    log.info(
        "%s %s starting (frozen=%s, root=%s)", APP_NAME, __version__, layout.frozen, layout.root
    )
    # The layout is resolved before logging exists (its log directory comes from it), so the
    # models it picked are reported here rather than where they were chosen.
    log.info(
        "Layout: engines=%s, whisper=%s, vad=%s, cleanup=%s, settings=%s",
        layout.vulkan_dir.parent,
        layout.whisper_model.name,
        layout.vad_model.name,
        layout.llama_model.name if layout.llama_model else "none",
        layout.settings_path,
    )
    if args.unknown:
        log.warning("Ignoring unknown arguments: %s", " ".join(args.unknown))
    if args.quit:
        return _quit_running_instance(deps)
    if not deps.acquire_single_instance(MUTEX_NAME):
        log.info("Another instance is running; asking it to open settings")
        if not _wait_for_window(deps, deps.clock() + WINDOW_WAIT_S):
            log.warning(
                "The running instance never showed its window within %.0f s; "
                "the open-settings request is dropped",
                WINDOW_WAIT_S,
            )
            return 0
        deps.signal_running_instance(WINDOW_CLASS, OPEN_SETTINGS)
        return 0
    instance = _App(args, layout, deps)
    try:
        return instance.run()
    except BaseException:
        log.exception("Spells could not start")
        instance.shutdown()
        return 1


def _quit_running_instance(deps: Deps) -> int:
    """Ask the running instance to exit and wait up to 10 s all told (spec 19.5).

    The mutex, not the window, says whether anything runs: an instance that is still probing
    GPUs holds the mutex but has no window yet, and Setup runs this the moment the user
    clicks Install. So take the mutex to prove the field is clear, and otherwise wait for the
    window to appear before asking it to quit.
    """
    if deps.acquire_single_instance(MUTEX_NAME):
        deps.release_single_instance(MUTEX_NAME)
        log.info("No running instance to quit")
        return 0
    deadline = deps.clock() + QUIT_TIMEOUT_S
    if not _wait_for_window(deps, deadline):
        log.warning(
            "An instance holds the mutex but showed no window within %.0f s", QUIT_TIMEOUT_S
        )
        return 0
    if not deps.signal_running_instance(WINDOW_CLASS, QUIT):
        log.warning("The running instance did not accept the quit command")
    while deps.clock() < deadline:
        if not deps.find_message_window(WINDOW_CLASS):
            log.info("The running instance has exited")
            return 0
        deps.sleeper(QUIT_POLL_S)
    log.warning("The running instance is still there after %.0f s", QUIT_TIMEOUT_S)
    return 0


def _wait_for_window(deps: Deps, deadline: float) -> int:
    """The running instance's hidden window, polled until the deadline; 0 when it never came."""
    while True:
        hwnd = deps.find_message_window(WINDOW_CLASS)
        if hwnd:
            return hwnd
        if deps.clock() >= deadline:
            return 0
        deps.sleeper(QUIT_POLL_S)


def _write_line(text: str) -> None:
    try:
        print(text)
    except OSError:
        pass  # a windowed build has no stdout


NO_SUITABLE_CLEANUP = (
    "Cleanup is off: no installed cleanup model suits your dictation languages, "
    "so raw transcripts are delivered."
)


@dataclass(frozen=True)
class ModelSetup:
    paths: EnginePaths
    cpu_only: bool
    cpu_cleanup_allowed: bool
    selection: Selection | None
    cleanup_notice: str
    speech_notice: str = ""


def first_run_seed(settings: Settings, root: Path) -> Settings:
    """The online installer's language choice, used as the default on the very first start.

    The online installer of spec 19.3 asks which languages the user will dictate in, downloads
    the models those need, and leaves the answer in ``<root>\\first-run.json``. Only a start
    that found no settings file comes here, so the file is read once in the life of an
    installation and a reinstall never overwrites what the user later chose in Settings. A
    missing or malformed file leaves the defaults alone (config.first_run_languages).
    """
    codes = first_run_languages(root / FIRST_RUN_FILE)
    if not codes:
        return settings
    log.info("First run: the installer asked for %s", ", ".join(codes))
    return with_enabled_languages(settings, codes)


def model_setup(
    layout: Layout,
    gpu: GpuSelection,
    languages: Sequence[str],
    catalog: Sequence[CatalogModel] | None,
    calibrations: Mapping[str, calibrate.SpeechCalibration] | None = None,
) -> ModelSetup:
    base = layout.engine_paths()
    if catalog is None:
        return ModelSetup(base, False, False, None, layout.cleanup_notice, layout.whisper_notice)
    installed = installed_ids(layout.models_dir, catalog)
    hardware, integrated, measured = calibrate.choose_hardware(
        gpu, languages, installed, catalog, calibrations or {}
    )
    selection = select_models(languages, hardware, installed, measured)
    if integrated:
        selection = replace(selection, integrated=True)
    paths = base.with_speech(layout.models_dir, selection.asr)
    notice = layout.cleanup_notice
    cleanup = selection.cleanup
    if cleanup is not None:
        paths = paths.with_cleanup(layout.models_dir, cleanup)
        notice = ""
    elif any(model.kind is ModelKind.CLEANUP and model.id in installed for model in catalog):
        paths = paths.with_cleanup(layout.models_dir, None)
        notice = NO_SUITABLE_CLEANUP
    writer = selection.compose
    if writer is not None and not selection.one_text_model:
        paths = paths.with_writer(layout.models_dir, writer)
    return ModelSetup(
        paths=paths,
        cpu_only=selection.hardware is Hardware.CPU,
        cpu_cleanup_allowed=selection.cpu_cleanup_allowed,
        selection=selection,
        cleanup_notice=notice,
        speech_notice="" if selection.asr else layout.whisper_notice,
    )


def describe_setup(languages: Sequence[str], setup: ModelSetup) -> str:
    selection = setup.selection
    if selection is None:
        return "Models: no catalog, using the layout's models"
    asr = ", ".join(
        f"{choice.model_id} ({choice.runtime}) for {'/'.join(choice.languages)}"
        for choice in selection.asr
    )
    cleanup = selection.cleanup.model_id if selection.cleanup else "none"
    where = selection.hardware.value
    if selection.integrated:
        where += " (integrated graphics, measured faster than the processor)"
    return f"Models for {'/'.join(languages)} on {where}: speech {asr or 'none'}, cleanup {cleanup}"


def chords_of(settings: Settings) -> list:
    """The chord set in the order the hook matches it (spec 6, 8.5).

    The main chord, the per-language chords, then the writing chords when they are set.
    ui.tray.chords_of answers the same list; this one is the startup copy.
    """
    general = settings.general
    writing = [general.compose_chord, general.edit_chord]
    return [
        general.main_chord,
        *general.language_chords,
        *[chord for chord in writing if chord is not None],
    ]


class _App:
    """The wiring, the hidden window thread and the one shutdown path."""

    def __init__(self, args: Args, layout: Layout, deps: Deps) -> None:
        self._args = args
        self._layout = layout
        self._deps = deps
        self._main_thread = threading.current_thread()
        self._config: ConfigStore | None = None
        self._app: Any = None
        self._bridge: Any = None
        self._engines: Any = None
        self._history: Any = None
        self._pipeline: Any = None
        self._hotkey: Any = None
        self._handles: Any = None
        self._placeholder: Any = None
        self._window: Any = None
        self._window_thread: threading.Thread | None = None
        self._window_stop = threading.Event()
        self._window_ready = threading.Event()
        self._unsubscribe: Callable[[], None] | None = None
        self._autostart_enabled: bool | None = None
        self._idle_minutes: int | None = None
        self._debug_logging: bool | None = None
        self._gpu: GpuSelection = NO_GPU
        self._catalog: Sequence[CatalogModel] | None = None
        self._languages: tuple[str, ...] = ()
        self._models: ModelSetup | None = None
        self._calibrations: dict[str, calibrate.SpeechCalibration] = {}
        self._calibrator: Any = None
        self._calibration_thread: threading.Thread | None = None
        self._plan: CpuPlan = CpuPlan()
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_finished = threading.Event()

    # Startup ---------------------------------------------------------------------------------

    def run(self) -> int:
        self._start()
        try:
            code = int(self._app.exec())
        finally:
            self.shutdown()
        return code

    def _start(self) -> None:
        deps = self._deps
        layout = self._layout

        first_run = not _exists(layout.settings_path)
        config = deps.config_store(layout.settings_path)
        self._config = config
        settings = config.settings
        self._debug_logging = settings.diagnostics.debug_logging
        if settings.diagnostics.debug_logging:
            deps.set_debug_logging(True)
            log.debug("Debug logging is on")
        if first_run:
            log.info("First run: writing %s", layout.settings_path)
            self._save_settings(lambda current: first_run_seed(current, layout.root))
            settings = config.settings

        self._app = deps.qapplication(list(sys.argv[:1]) or [APP_NAME])
        self._show_placeholder()
        self._bridge = deps.bridge()

        selection = self._probe_gpu(settings)
        self._gpu = selection
        self._catalog = self._load_catalog()
        self._languages = tuple(settings.general.enabled_languages)
        self._calibrations = self._load_calibrations(selection)
        models = model_setup(layout, selection, self._languages, self._catalog, self._calibrations)
        self._models = models
        log.info("%s", describe_setup(self._languages, models))
        self._plan = self._cpu_plan()

        self._engines = deps.supervisor(
            models.paths,
            selection,
            self._bridge.on_engine_status,
            idle_unload_minutes=settings.general.idle_unload_minutes,
            cpu_only=models.cpu_only,
            cpu_cleanup_allowed=models.cpu_cleanup_allowed,
            cpu_plan=self._plan,
        )
        self._engines.start()

        self._history = deps.history(layout.history_path, settings.history.retention)

        self._pipeline = deps.pipeline(
            config=config,
            engines=self._engines,
            hotkey=self._end_recording,
            history=self._history,
            on_event=self._bridge.on_pipeline_event,
            selection=models.selection,
        )
        self._pipeline.start()

        self._hotkey = deps.hotkey(
            chords_of(settings),
            self._pipeline.hotkey_callbacks(),
            on_error=self._bridge.on_hotkey_error,
        )
        self._hotkey.start()

        self._handles = deps.create_ui(
            self._app,
            config=config,
            engines=self._engines,
            pipeline=self._pipeline,
            hotkey=self._hotkey,
            history=self._history,
            gpu_selection=selection,
            log_dir=layout.log_dir,
            first_run=first_run,
            update_source=updates.source_url(layout.data_dir),
            bridge=self._bridge,
            on_quit=self._on_quit,
        )
        self._hide_placeholder()

        self._start_window_thread()

        self._autostart_enabled = settings.general.autostart
        self._idle_minutes = settings.general.idle_unload_minutes
        self._apply_autostart(settings.general.autostart)
        self._unsubscribe = config.subscribe(self._on_settings)

        self._warn_at_start(config.notice)
        if self._args.settings:
            self._open_settings()
        self._start_calibration()

    def _show_placeholder(self) -> None:
        """Something on screen while the GPU probe blocks; the real tray replaces it."""
        try:
            self._placeholder = self._deps.placeholder_tray()
        except Exception:
            log.exception("The starting tray icon could not be shown")
            self._placeholder = None

    def _hide_placeholder(self) -> None:
        placeholder, self._placeholder = self._placeholder, None
        if placeholder is None:
            return
        try:
            placeholder.hide()
            delete = getattr(placeholder, "deleteLater", None)
            if delete is not None:
                delete()
        except Exception:
            log.exception("The starting tray icon could not be removed")

    def _probe_gpu(self, settings: Settings) -> GpuSelection:
        """Pick the GPU (spec 13). Blocking, seconds on a first run, so it says so first."""
        diagnostics = settings.diagnostics
        cached = None
        if diagnostics.gpu_device_index is not None and diagnostics.gpu_device_name:
            cached = (diagnostics.gpu_device_index, diagnostics.gpu_device_name)
        log.info("Probing GPUs, which takes a few seconds on a first run")
        try:
            selection = self._deps.select_device(
                self._layout.llama_exe("vulkan"),
                override=diagnostics.gpu_device_override,
                cached=cached,
                whisper_server=self._layout.whisper_exe("vulkan"),
            )
        except Exception:
            log.exception("The GPU probe failed; continuing without a device")
            selection = NO_GPU
        log.info("GPU: %s (raw index %s)", selection.name or "none", selection.raw_index)
        name = selection.name or None
        if selection.raw_index != diagnostics.gpu_device_index or name != diagnostics.gpu_device_name:
            self._save_settings(
                lambda current: replace(
                    current,
                    diagnostics=replace(
                        current.diagnostics,
                        gpu_device_index=selection.raw_index,
                        gpu_device_name=name,
                    ),
                )
            )
        return selection

    def _load_catalog(self) -> Sequence[CatalogModel] | None:
        try:
            return tuple(self._deps.model_catalog())
        except Exception:
            log.exception("The model catalog could not be read; using the layout's models")
            return None

    def _cpu_plan(self) -> CpuPlan:
        try:
            plan = self._deps.cpu_plan()
        except Exception:
            log.exception("The processor topology could not be read; engines are not pinned")
            return CpuPlan()
        if plan.affinity_mask is not None:
            log.info(
                "Engines pinned to %s performance cores (mask %s)", plan.threads, plan.cpu_mask_hex
            )
        return plan

    def _load_calibrations(self, gpu: GpuSelection) -> dict[str, calibrate.SpeechCalibration]:
        if not calibrate.needs_measuring(gpu):
            return {}
        try:
            results = calibrate.load(calibrate.store_path(self._layout.settings_path), gpu.name)
        except Exception:
            log.exception("The earlier speed measurements could not be read")
            return {}
        for result in results.values():
            log.info("Measured earlier: %s", result.describe())
        return results

    def _start_calibration(self) -> None:
        catalog, models = self._catalog, self._models
        if catalog is None or models is None or not calibrate.needs_measuring(self._gpu):
            return
        running = self._calibration_thread
        if running is not None and running.is_alive():
            return
        try:
            installed = installed_ids(self._layout.models_dir, catalog)
            todo = calibrate.pending(
                self._gpu, self._languages, installed, catalog, self._calibrations
            )
            if not todo:
                return
            plan = self._plan
            calibrator = self._deps.calibrator(
                paths=models.paths,
                gpu=self._gpu,
                cpu_plan=plan,
                whisper_threads=plan.threads or max(1, min(8, (os.cpu_count() or 4) // 2)),
                models_dir=self._layout.models_dir,
            )
        except Exception:
            log.exception("The speed measurement could not be prepared")
            return
        self._calibrator = calibrator
        thread = threading.Thread(
            target=self._calibrate,
            args=(calibrator, todo),
            name="spells-calibration",
            daemon=True,
        )
        self._calibration_thread = thread
        thread.start()

    def _calibrate(self, calibrator: Any, todo: Sequence[Any]) -> None:
        names = ", ".join(choice.model_id for choice in todo)
        log.info(
            "Measuring %s on %s and on the processor, once for this computer",
            names,
            self._gpu.name,
        )
        try:
            self._engines.wait_ready(Engine.WHISPER, calibrate.STARTUP_TIMEOUT_S)
        except Exception:
            log.debug("The speech engine state could not be read", exc_info=True)
        results = []
        for choice in todo:
            if calibrator.cancelled:
                return
            try:
                result = calibrator.calibrate(choice)
            except Exception:
                log.exception("Measuring %s failed", choice.model_id)
                continue
            if calibrator.cancelled:
                return
            log.info("Measured %s", result.describe())
            results.append(result)
        if not results:
            return
        try:
            calibrate.save(calibrate.store_path(self._layout.settings_path), results)
        except Exception:
            log.exception("The speed measurements could not be saved")
        merged = dict(self._calibrations)
        merged.update({result.model_id: result for result in results})
        self._calibrations = merged
        self._apply_models(self._languages)

    def _cancel_calibration(self) -> None:
        calibrator, self._calibrator = self._calibrator, None
        if calibrator is not None:
            calibrator.cancel()

    def _apply_models(self, languages: tuple[str, ...]) -> None:
        try:
            models = model_setup(
                self._layout, self._gpu, languages, self._catalog, self._calibrations
            )
        except Exception:
            log.exception("The models for %s could not be chosen", "/".join(languages))
            return
        log.info("%s", describe_setup(languages, models))
        current = self._models
        self._models = models
        set_selection = getattr(self._pipeline, "set_selection", None)
        if callable(set_selection):
            set_selection(models.selection)
        if current is not None and (
            current.paths,
            current.cpu_only,
            current.cpu_cleanup_allowed,
        ) == (models.paths, models.cpu_only, models.cpu_cleanup_allowed):
            return
        try:
            changed = self._engines.set_models(
                models.paths,
                cpu_only=models.cpu_only,
                cpu_cleanup_allowed=models.cpu_cleanup_allowed,
            )
        except Exception:
            log.exception("The engines could not switch models")
            return
        log.info("Engines changed for the new models: %s", [e.key for e in changed] or "none")

    def _warn_at_start(self, notice: str | None) -> None:
        """A balloon for each problem, plus a standing tray warning for a missing speech model.

        A missing cleanup model gets its balloon only: the supervisor already reports llama as
        Failed with reason no_model, which the tray shows as a Warning with its own line, and
        Windows cuts a tooltip at 127 characters, so the reason is not repeated there.
        """
        whisper_notice = self._models.speech_notice if self._models else self._layout.whisper_notice
        cleanup_notice = self._models.cleanup_notice if self._models else self._layout.cleanup_notice
        for message in (notice, whisper_notice, cleanup_notice):
            if message:
                log.warning("%s", message)
                self._notify(message)
        if whisper_notice:
            self._set_tray_warning(whisper_notice)

    # The hidden window (spec 15, 19.5) ---------------------------------------------------------

    def _start_window_thread(self) -> None:
        self._window_thread = threading.Thread(
            target=self._window_loop, name="spells-window", daemon=True
        )
        self._window_thread.start()
        if not self._window_ready.wait(WINDOW_READY_TIMEOUT_S):
            log.warning("The hidden window is taking longer than expected to appear")

    def _window_loop(self) -> None:
        window = None
        try:
            window = self._deps.message_window(WINDOW_CLASS, self._handlers())
            self._window = window
            log.info("Hidden window %s is up", WINDOW_CLASS)
        except Exception:
            log.exception("The hidden window could not be created; --quit cannot reach this instance")
        finally:
            self._window_ready.set()
        if window is None:
            return
        try:
            window.run(self._window_stop)
        except Exception:
            log.exception("The hidden window loop failed")
        finally:
            try:
                window.destroy()
            except Exception:
                log.exception("The hidden window could not be destroyed")
            self._window = None

    def _handlers(self) -> MessageHandlers:
        return MessageHandlers(
            on_copydata=self._on_command,
            on_query_end_session=self._on_query_end_session,
            on_end_session=self._on_end_session,
            on_resume=self._reinstall_hook,
            on_session_unlock=self._reinstall_hook,
        )

    def _on_command(self, text: str) -> None:
        """A single-instance command, on the window's thread: marshal, never act here."""
        command = (text or "").strip().lower()
        if command == OPEN_SETTINGS:
            self._request_open_settings()
        elif command == QUIT:
            self._request_quit()
        else:
            log.warning("Unknown command on the single-instance channel: %r", text)

    def _on_query_end_session(self, flags: int) -> bool:
        log.info("Windows asks to end the session (flags 0x%X); answering yes", flags)
        return True

    def _on_end_session(self, flags: int) -> None:
        """Restart Manager: this must finish before it returns (spec 15, 19.5)."""
        log.info("Session ending (flags 0x%X); shutting down now", flags)
        self.shutdown(wait_s=END_SESSION_WAIT_S)
        self._request_quit()

    def _reinstall_hook(self) -> None:
        hotkey = self._hotkey
        if hotkey is None:
            return
        try:
            hotkey.request_reinstall()
        except Exception:
            log.exception("The hook reinstall request failed")

    # Marshalling to the Qt thread ------------------------------------------------------------------

    def _tray_signal(self, name: str) -> Any:
        tray = getattr(self._handles, "tray", None)
        return getattr(tray, name, None)

    def _request_open_settings(self) -> None:
        """Qt queues this signal because it is emitted from the window's thread (spec 5.1)."""
        signal = self._tray_signal("open_requested")
        if signal is None:
            log.warning("No tray yet; the open-settings command is dropped")
            return
        signal.emit("general")

    def _request_quit(self) -> None:
        signal = self._tray_signal("quit_requested")
        if signal is None:
            self._on_quit()
            return
        signal.emit()

    def _open_settings(self) -> None:
        opener = getattr(self._handles, "open_settings", None)
        if opener is None:
            return
        try:
            opener("general")
        except Exception:
            log.exception("Settings could not be opened")

    def _set_tray_warning(self, reason: str | None) -> None:
        tray = getattr(self._handles, "tray", None)
        setter = getattr(tray, "set_extra_warning", None)
        if setter is None:
            log.warning("No tray to carry the warning: %s", reason)
            return
        try:
            setter(reason)
        except Exception:
            log.exception("The tray warning could not be set")

    def _notify(self, text: str) -> None:
        tray = getattr(self._handles, "tray", None)
        notify = getattr(tray, "notify", None)
        if notify is None:
            return
        try:
            notify(text, warning=True)
        except Exception:
            log.exception("The tray warning could not be shown")

    # Settings reactions ------------------------------------------------------------------------------

    def _on_settings(self, settings: Settings) -> None:
        """Runs on whichever thread called ConfigStore.update (spec 5.2).

        The tray owns the chords and the settings dialog owns history retention; what is left
        for the app is the Run key and the idle unload interval.
        """
        if settings.general.autostart != self._autostart_enabled:
            self._autostart_enabled = settings.general.autostart
            self._apply_autostart(settings.general.autostart)
        languages = tuple(settings.general.enabled_languages)
        if languages != self._languages:
            self._languages = languages
            self._apply_models(languages)
            self._start_calibration()
        if settings.general.idle_unload_minutes != self._idle_minutes:
            self._idle_minutes = settings.general.idle_unload_minutes
            try:
                self._engines.set_idle_unload_minutes(settings.general.idle_unload_minutes)
            except Exception:
                log.exception("The idle unload interval could not be changed")
        if settings.diagnostics.debug_logging != self._debug_logging:
            # This is the one setting whose effect is the logging itself, and Diagnostics
            # warns that the logs then carry transcript text (spec 17), so it takes effect
            # now rather than at the next start.
            self._debug_logging = settings.diagnostics.debug_logging
            self._deps.set_debug_logging(settings.diagnostics.debug_logging)
            log.info("Debug logging is now %s", "on" if self._debug_logging else "off")

    def _apply_autostart(self, enabled: bool) -> None:
        try:
            self._deps.autostart_apply(enabled, self._deps.autostart_command())
        except Exception:
            log.exception("The autostart setting could not be applied")

    def _save_settings(self, mutator: Callable[[Settings], Settings]) -> None:
        try:
            self._config.update(mutator)
        except Exception:
            log.exception("The settings could not be saved")

    def _end_recording(self, dictation_id: int) -> None:
        """What the pipeline calls; the hotkey thread exists only after the pipeline does."""
        hotkey = self._hotkey
        if hotkey is not None:
            hotkey.end_recording(dictation_id)

    # Shutdown -----------------------------------------------------------------------------------------

    def _on_quit(self) -> None:
        """The tray's Quit, and every marshalled quit; runs on the Qt thread."""
        self.shutdown()
        try:
            self._app.quit()
        except Exception:
            log.exception("The Qt event loop could not be stopped")

    def shutdown(self, *, wait_s: float = 0.0) -> None:
        """Stop everything, once, whoever asks and from whichever thread.

        `wait_s` is for WM_ENDSESSION: when another thread is already tearing down, that
        handler must not return while the teardown runs, so it waits for it to finish. The
        lock guards the flag only, never the teardown, which joins the very thread that may
        be calling this.
        """
        with self._shutdown_lock:
            mine = not self._shutdown_started
            self._shutdown_started = True
        if not mine:
            if wait_s > 0 and not self._shutdown_finished.wait(wait_s):
                log.warning("The shutdown already running did not finish within %.1f s", wait_s)
            return
        started = self._deps.clock()
        log.info("Shutting down")
        self._step("settings subscription", self._unsubscribe_settings)
        self._step("starting icon", self._hide_placeholder)
        self._step("ui", self._close_ui)
        self._step("speed measurement", self._cancel_calibration)
        self._step("pipeline", lambda: self._pipeline and self._pipeline.stop())
        self._step("hotkey", lambda: self._hotkey and self._hotkey.stop())
        self._step("engines", lambda: self._engines and self._engines.stop())
        self._step("history", lambda: self._history and self._history.close())
        self._step("hidden window", self._stop_window)
        self._step("mutex", lambda: self._deps.release_single_instance(MUTEX_NAME))
        log.info("Shutdown took %.0f ms", (self._deps.clock() - started) * 1000.0)
        self._shutdown_finished.set()

    @staticmethod
    def _step(name: str, action: Callable[[], Any]) -> None:
        try:
            action()
        except Exception:
            log.exception("Shutdown step %s failed", name)

    def _unsubscribe_settings(self) -> None:
        unsubscribe, self._unsubscribe = self._unsubscribe, None
        if unsubscribe is not None:
            unsubscribe()

    def _close_ui(self) -> None:
        handles = self._handles
        if handles is None:
            return
        if threading.current_thread() is not self._main_thread:
            # Widgets belong to the Qt thread; an end-session shutdown leaves them alone.
            log.info("Leaving the widgets alone: this shutdown is not on the Qt thread")
            return
        handles.close()

    def _stop_window(self) -> None:
        self._window_stop.set()
        thread = self._window_thread
        if thread is None or thread is threading.current_thread():
            return  # the window's own thread destroys it when run() returns
        if thread.is_alive():
            thread.join(WINDOW_JOIN_TIMEOUT_S)
        if thread.is_alive():
            log.warning("The hidden window thread did not stop; destroying the window from here")
            window = self._window
            if window is not None:
                window.destroy()
            thread.join(WINDOW_JOIN_TIMEOUT_S)
        self._window_thread = None


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


__all__ = ["APP_NAME", "MUTEX_NAME", "WINDOW_CLASS", "Args", "Deps", "main", "parse_args"]
