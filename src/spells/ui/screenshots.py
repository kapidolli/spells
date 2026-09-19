"""Render every window of the ui with fake data and save PNGs, in the light and dark themes.

    QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m spells.ui.screenshots <out dir>

Options: --themes light,dark (default both), --accent system|brand (default brand),
--only <prefix> to render only the shots whose name starts with it. Nothing here starts an
engine, opens a microphone or touches the settings file of the real app: the collaborators
are fakes and the settings live in a temporary folder.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spells.audio import AudioDevice, CaptureInfo
from spells.config import ConfigStore, ProfileRule, Replacement, Snippet, TermEntry
from spells.gpu import GpuDevice, GpuSelection
from spells.history import HistoryEntry
from spells.hotkey import HotkeyStats
from spells.models import Chord, ChordMode, DeliveryMethod, Engine, EngineState, StageTimings
from spells.pipeline import Notice, PillState, PipelineEvent, TrayState
from spells.profiles import BUILTIN_PROFILES
from spells.ui import style
from spells.ui.hotkeys import ChordCaptureDialog
from spells.ui.icons import TrayIconState, render_tray_pixmap
from spells.ui.pill import Pill
from spells.ui.settings import RuleEditor, SettingsDialog
from spells.ui.tray import Tray
from spells.ui.welcome import STEP_NAMES, WelcomePage


def _prepare_environment() -> None:
    if os.environ.get("QT_QPA_PLATFORM", "") == "offscreen" and "QT_QPA_FONTDIR" not in os.environ:
        fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        if fonts.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(fonts)


DEVICES = (
    GpuDevice(raw_index=0, list_name="Vulkan0", name="AMD Radeon 610M", memory_mb=8000, free_mb=7000, uma=True),
    GpuDevice(raw_index=1, list_name="Vulkan1", name="NVIDIA GeForce RTX 5060 Laptop GPU", memory_mb=8151, free_mb=7412, uma=False),
)
SELECTION = GpuSelection(raw_index=1, name="NVIDIA GeForce RTX 5060 Laptop GPU", memory_mb=8151, devices=DEVICES)
MICROPHONES = [
    AudioDevice(index=1, name="Microphone Array (Realtek Audio)", is_default=True),
    AudioDevice(index=3, name="Headset Microphone (Jabra Evolve2 65)", is_default=False),
]


class FakeEngines:
    def __init__(self) -> None:
        self.states = {Engine.WHISPER: EngineState.READY, Engine.LLAMA: EngineState.CPU_FALLBACK}
        self.reasons = {Engine.WHISPER: "ok", Engine.LLAMA: "oom"}
        self.variants = {Engine.WHISPER: "vulkan", Engine.LLAMA: "cpu"}
        self.verified = {Engine.WHISPER: True, Engine.LLAMA: None}

    def status(self, engine: Engine) -> EngineState:
        return self.states[engine]

    def reason(self, engine: Engine) -> str:
        return self.reasons[engine]

    def variant(self, engine: Engine) -> str:
        return self.variants[engine]

    def gpu_verified(self, engine: Engine) -> bool | None:
        return self.verified[engine]

    def restart(self, engine: Engine) -> None:
        return None

    def set_gpu(self, selection: GpuSelection) -> None:
        return None


class FakePipeline:
    def __init__(self) -> None:
        self.timings = [
            StageTimings(press_to_pill_ms=38.0 + i, mic_open_ms=260.0 + 12 * i, release_to_transcript_ms=590.0 + 9 * i, delivery_ms=6.0)
            for i in range(5)
        ]
        self.guards = {"length_ratio": 2, "preamble": 1}
        self.captures = [
            CaptureInfo(
                device="Microphone (HyperX Cloud III Wireless)",
                host_api="Windows WASAPI",
                sample_rate=48000,
                channels=1,
            )
        ]

    def retry_last(self) -> bool:
        return False

    def recent_timings(self) -> list[StageTimings]:
        return list(self.timings)

    def recent_captures(self) -> list[CaptureInfo]:
        return list(self.captures)

    def guard_counts(self) -> dict[str, int]:
        return dict(self.guards)


class FakeHotkey:
    def __init__(self) -> None:
        self._chords: tuple = ()
        self.stats = HotkeyStats(probes_sent=412, probes_missed=0, reinstalls=1, hook_exceptions=0, install_failures=0)

    @property
    def chords(self) -> tuple:
        return self._chords

    def update_chords(self, chords: Iterable[Chord]) -> None:
        self._chords = tuple(chords)


class FakeHistory:
    def __init__(self) -> None:
        now = time.time()
        samples = [
            ("slack.exe", "en", "hey can you send me the deck before the call", "Hey, can you send me the deck before the call?"),
            ("outlook.exe", "de", "danke fuer die schnelle antwort ich melde mich morgen", "Danke für die schnelle Antwort, ich melde mich morgen."),
            ("Code.exe", "en", "rename the function to load settings", "Rename the function to load_settings"),
            ("WINWORD.EXE", "sq", "faleminderit per ndihmen", "Faleminderit për ndihmën."),
            ("slack.exe", "en", "um so i mean lets meet at ten", "Let's meet at ten"),
        ]
        self.entries = [
            HistoryEntry(
                id=i,
                created_at=now - 60 * 37 * i,
                raw_text=raw,
                cleaned_text=cleaned,
                delivered_text=cleaned,
                app_process=app,
                app_title="",
                language=language,
                used_llm=True,
                cleanup_reason="ok",
                outcome="pasted",
                timings=StageTimings(press_to_pill_ms=40.0),
            )
            for i, (app, language, raw, cleaned) in enumerate(samples)
        ]

    def recent(self, limit: int = 100) -> list[HistoryEntry]:
        return list(self.entries[:limit])

    def search(self, query: str, limit: int = 100) -> list[HistoryEntry]:
        needle = query.casefold()
        return [e for e in self.entries if needle in e.raw_text.casefold() or needle in e.cleaned_text.casefold()][:limit]

    def clear(self) -> None:
        self.entries = []

    def set_retention(self, retention: str) -> None:
        return None


class SilentRecorder:
    def __init__(self, **kwargs: Any) -> None:
        self.on_level = kwargs.get("on_level")

    def start(self) -> None:
        return None

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None


def _silent_notify(kind: str, title: str, text: str) -> None:
    print(f"[{kind}] {title}: {text}", file=sys.stderr)


def _seed(config: ConfigStore) -> None:
    def mutate(s: Any) -> Any:
        general = replace(
            s.general,
            language_chords=[Chord(keys=(0x11, 0x12, 0x44), language="de")],
            compose_chord=Chord(keys=(0x11, 0x12, 0x57), mode=ChordMode.COMPOSE),
            edit_chord=Chord(keys=(0x11, 0x12, 0x45), mode=ChordMode.EDIT),
            idle_unload_minutes=30,
        )
        vocabulary = replace(
            s.vocabulary,
            terms=[TermEntry(text=t, edited_at=float(i)) for i, t in enumerate(["Contoso", "PostgreSQL", "Kubernetes", "PySide6", "Prishtina", "whisper.cpp"])],
            replacements=[Replacement("k8s", "Kubernetes"), Replacement("brb", "be right back")],
            snippets=[Snippet("thanks", "Thanks for your help, talk soon."), Snippet("sig", "Kind regards, Alex")],
        )
        profiles = [
            ProfileRule(name="Slack", match_process=["slack.exe"], match_title=[], profile=BUILTIN_PROFILES["Chat"]),
            ProfileRule(
                name="Terminal",
                match_process=["WindowsTerminal.exe"],
                match_title=[],
                profile=replace(BUILTIN_PROFILES["Terminal"], delivery=DeliveryMethod.TYPE),
            ),
        ]
        return replace(s, general=general, vocabulary=vocabulary, profiles=profiles)

    config.update(mutate)


def _backdrop(size: QtCore.QSize, dark: bool) -> QtGui.QImage:
    image = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    painter = QtGui.QPainter(image)
    try:
        gradient = QtGui.QLinearGradient(0, 0, size.width(), size.height())
        if dark:
            gradient.setColorAt(0.0, QtGui.QColor("#2C4080"))
            gradient.setColorAt(1.0, QtGui.QColor("#1A1030"))
        else:
            gradient.setColorAt(0.0, QtGui.QColor("#BBD2F3"))
            gradient.setColorAt(1.0, QtGui.QColor("#E2CCF1"))
        painter.fillRect(image.rect(), gradient)
    finally:
        painter.end()
    return image


def _padded(pixmap: QtGui.QPixmap, dark: bool, pad: int = 24) -> QtGui.QImage:
    size = QtCore.QSize(pixmap.width() + 2 * pad, pixmap.height() + 2 * pad)
    image = _backdrop(size, dark)
    painter = QtGui.QPainter(image)
    try:
        painter.drawPixmap(pad, pad, pixmap)
    finally:
        painter.end()
    return image


class Shooter:
    def __init__(self, out_dir: Path, only: str | None) -> None:
        self.out_dir = out_dir
        self.only = only
        self.saved: list[Path] = []

    def wanted(self, name: str) -> bool:
        return self.only is None or name.startswith(self.only)

    def save(self, name: str, image: QtGui.QPixmap | QtGui.QImage) -> None:
        path = self.out_dir / f"{name}.png"
        image.save(str(path))
        self.saved.append(path)
        print(path)


def _settle(app: QtWidgets.QApplication, rounds: int = 4) -> None:
    for _ in range(rounds):
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _show_offscreen(widget: QtWidgets.QWidget) -> None:
    widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.show()


def shoot_settings(app: QtWidgets.QApplication, shooter: Shooter, theme_name: str, folder: Path) -> None:
    config = ConfigStore(folder / "settings.json")
    _seed(config)
    dialog = SettingsDialog(
        config=config,
        engines=FakeEngines(),
        pipeline=FakePipeline(),
        hotkey=FakeHotkey(),
        history=FakeHistory(),
        gpu_selection=SELECTION,
        log_dir=folder / "logs",
        devices=lambda: list(MICROPHONES),
        probe=lambda _m, _k: True,
        notify=_silent_notify,
        confirm=lambda _t, _x: False,
        recorder_factory=SilentRecorder,
        window_picker=lambda: ("slack.exe", "general"),
    )
    dialog.resize(1080, 760)
    _show_offscreen(dialog)
    for name in dialog.page_names():
        shot = f"settings-{name}-{theme_name}"
        full = f"settings-{name}-full-{theme_name}"
        if not (shooter.wanted(shot) or shooter.wanted(full)):
            continue
        dialog.show_tab(name)
        if name == "history":
            dialog.history.table.selectRow(1)
        if name == "about":
            dialog.about.licenses.setCurrentRow(0)
        _settle(app)
        if shooter.wanted(shot):
            shooter.save(shot, dialog.grab())
        page = dialog.pages.currentWidget()
        body = getattr(page, "body", None)
        if body is not None and shooter.wanted(full):
            canvas = QtGui.QPixmap(body.size())
            canvas.fill(style.palette().color("layer"))
            body.render(canvas)
            shooter.save(full, canvas)
    if shooter.wanted(f"dialog-hotkey-{theme_name}"):
        capture = ChordCaptureDialog(hotkey=FakeHotkey(), active_chords=())
        capture.preview.set_keys(["Ctrl", "Alt", "D"])
        capture.adjustSize()
        capture.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        capture.ensurePolished()
        capture.layout().activate()
        _settle(app)
        shooter.save(f"dialog-hotkey-{theme_name}", capture.grab())
        capture.deleteLater()
    if shooter.wanted(f"dialog-rule-{theme_name}"):
        editor = RuleEditor(profile_names=list(BUILTIN_PROFILES), rule=config.settings.profiles[0])
        _show_offscreen(editor)
        editor.adjustSize()
        _settle(app)
        shooter.save(f"dialog-rule-{theme_name}", editor.grab())
        editor.close()
    dialog.close()
    dialog.deleteLater()
    _settle(app)


def shoot_welcome(app: QtWidgets.QApplication, shooter: Shooter, theme_name: str, folder: Path) -> None:
    names = [f"welcome-{index + 1}-{step}-{theme_name}" for index, step in enumerate(STEP_NAMES)]
    if not any(shooter.wanted(name) for name in names):
        return
    config = ConfigStore(folder / "welcome-settings.json")
    page = WelcomePage(
        config=config,
        hotkey=FakeHotkey(),
        devices=lambda: list(MICROPHONES),
        recorder_factory=SilentRecorder,
        notify=_silent_notify,
        reduced_motion=True,
    )
    _show_offscreen(page)
    for index, name in enumerate(names):
        if not shooter.wanted(name):
            continue
        page.set_step(index)
        if index == 2:
            page.meter.bar.set_level(0.42)
        _settle(app)
        shooter.save(name, page.grab())
    if shooter.wanted(f"welcome-hint-processing-{theme_name}"):
        page.set_step(0)
        page._phase = 2600.0
        _settle(app)
        shooter.save(f"welcome-hint-processing-{theme_name}", page.grab())
    page.close()
    page.deleteLater()
    _settle(app)


def shoot_tray(app: QtWidgets.QApplication, shooter: Shooter, theme_name: str, folder: Path) -> None:
    dark = theme_name == "dark"
    if shooter.wanted(f"tray-menu-{theme_name}"):
        config = ConfigStore(folder / "tray-settings.json")
        engines = FakeEngines()
        tray = Tray(
            config=config,
            pipeline=FakePipeline(),
            hotkey=FakeHotkey(),
            engines=engines,
            icon=QtWidgets.QSystemTrayIcon(),
            launcher=lambda _uri: None,
            light_taskbar=not dark,
        )
        tray.retry_action.setEnabled(True)
        for menu, suffix in ((tray.menu, ""), (tray.language_menu, "-language")):
            menu.ensurePolished()
            menu.adjustSize()
            _settle(app)
            shooter.save(f"tray-menu{suffix}-{theme_name}", QtGui.QPixmap.fromImage(_padded(menu.grab(), dark)))
    if shooter.wanted(f"tray-icons-{theme_name}"):
        sizes = (16, 24, 32, 48)
        states = list(TrayIconState)
        cell = 64
        image = QtGui.QImage(cell * len(sizes), cell * len(states), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QtGui.QColor("#1F1F1F" if dark else "#EEEEEE"))
        painter = QtGui.QPainter(image)
        try:
            for row, state in enumerate(states):
                for column, size in enumerate(sizes):
                    pixmap = render_tray_pixmap(state, size, light_taskbar=not dark)
                    painter.drawPixmap(column * cell + (cell - size) // 2, row * cell + (cell - size) // 2, pixmap)
        finally:
            painter.end()
        shooter.save(f"tray-icons-{theme_name}", image)


PILL_STATES: tuple[tuple[str, PipelineEvent], ...] = (
    ("recording", PipelineEvent(pill=PillState.RECORDING, tray=TrayState.RECORDING, level=0.6)),
    ("recording-busy", PipelineEvent(pill=PillState.RECORDING, tray=TrayState.RECORDING, level=0.6, busy=True)),
    ("recording-live", PipelineEvent(pill=PillState.RECORDING, tray=TrayState.RECORDING, level=0.6, live_typing=True)),
    ("latched", PipelineEvent(pill=PillState.LATCHED, tray=TrayState.RECORDING, level=0.6)),
    ("processing", PipelineEvent(pill=PillState.PROCESSING, tray=TrayState.PROCESSING)),
    ("starting", PipelineEvent(pill=PillState.STARTING_ENGINES, tray=TrayState.PROCESSING)),
    ("error", PipelineEvent(pill=PillState.IDLE, tray=TrayState.READY, notice=Notice.ERROR, notice_text="Mic is busy")),
    ("copied", PipelineEvent(pill=PillState.IDLE, tray=TrayState.READY, notice=Notice.COPIED, notice_text="Copied")),
)


def shoot_pill(app: QtWidgets.QApplication, shooter: Shooter, theme_name: str, _folder: Path) -> None:
    dark = theme_name == "dark"
    for state, event in PILL_STATES:
        name = f"pill-{state}-{theme_name}"
        if not shooter.wanted(name):
            continue
        pill = Pill(
            reduced_motion=True,
            screen_for=lambda _hwnd: app.primaryScreen(),
            apply_no_activate=lambda _hwnd: None,
        )
        pill.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        pill.apply(event)
        if event.pill in (PillState.RECORDING, PillState.LATCHED):
            pill.bar_heights = [3 + 17 * abs(math.sin(i * 0.55 + 0.4)) * (0.35 + 0.65 * math.sin(math.pi * (i + 0.5) / 15)) for i in range(15)]
        _settle(app)
        pixmap = pill.grab()
        shooter.save(name, _padded(pixmap, dark, pad=12))
        pill.close()
        pill.deleteLater()
    _settle(app)


SHOOTERS: tuple[Callable[[QtWidgets.QApplication, Shooter, str, Path], None], ...] = (
    shoot_settings,
    shoot_welcome,
    shoot_tray,
    shoot_pill,
)


def render(out_dir: Path, *, themes: Iterable[str] = ("light", "dark"), accent: str = "system", only: str | None = None) -> list[Path]:
    """Render the shots into out_dir; returns the saved paths."""
    _prepare_environment()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["spells-screenshots"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shooter = Shooter(out_dir, only)
    shades = None if accent == "system" else style.BRAND_SHADES
    with tempfile.TemporaryDirectory(prefix="spells-shots-") as temp:
        folder = Path(temp)
        for theme_name in themes:
            style.install(app, dark=theme_name == "dark", accent=shades, follow_system_accent=accent == "system", watch=False)
            theme_folder = folder / theme_name
            theme_folder.mkdir()
            for shoot in SHOOTERS:
                shoot(app, shooter, theme_name, theme_folder)
    return shooter.saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spells.ui.screenshots", description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", nargs="?", default="ui-shots", type=Path)
    parser.add_argument("--themes", default="light,dark")
    parser.add_argument("--accent", choices=("system", "brand"), default="brand")
    parser.add_argument("--only", default=None)
    args = parser.parse_args(argv)
    themes = [name.strip() for name in args.themes.split(",") if name.strip()]
    saved = render(args.out_dir, themes=themes, accent=args.accent, only=args.only)
    print(f"{len(saved)} screenshots in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
