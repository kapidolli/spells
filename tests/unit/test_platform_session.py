from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from spells import app as app_module
from spells.config import ConfigStore
from spells.models import CpuPlan
from spells.platform.base import Capabilities, MessageHandlers
from spells.platform.stub import StubAutostart, StubInstance, StubProcesses, StubUpdater
from spells.ui import updatecheck
from spells.ui.about import AboutPage
from spells.ui.updatecheck import Phase, UpdateCoordinator, direct_runner
from spells.updates import Release

from .fake_platform import fake_platform
from .test_ui_support import qt_app

SOURCE = "https://spells.example.com/latest.json"
PAYLOAD = b"MZ the real installer"


@pytest.fixture(scope="module")
def app():
    return qt_app()


class RecordingInstance(StubInstance):
    def __init__(self):
        self.calls = []

    def acquire(self, name):
        self.calls.append(("acquire", name))
        return False

    def release(self, name):
        self.calls.append(("release", name))

    def signal_running(self, name, command):
        self.calls.append(("signal", name, command))
        return True

    def find_running(self, name):
        self.calls.append(("find", name))
        return 42

    def message_window(self, name, handlers):
        self.calls.append(("window", name, handlers))
        return "the window"


class RecordingAutostart(StubAutostart):
    def __init__(self, name="spells"):
        self.name = name
        self.calls = []

    def apply(self, enabled, command):
        self.calls.append((enabled, command))
        return True

    def current_command(self):
        return f'"{self.name}" --from-the-fake'


class RecordingProcesses(StubProcesses):
    def __init__(self):
        super().__init__("linux")
        self.plan = CpuPlan(threads=3, affinity_mask=0b111)

    def cpu_plan(self):
        return self.plan


class RecordingUpdater(StubUpdater):
    def __init__(self):
        super().__init__("linux")
        self.applied = []

    @property
    def can_apply(self):
        return True

    def apply(self, installer_path):
        self.applied.append(Path(installer_path))


def test_deps_reach_the_instance_guard_autostart_and_cpu_plan_through_the_platform(
    use_platform,
):
    instance = RecordingInstance()
    autostart = RecordingAutostart()
    processes = RecordingProcesses()
    use_platform(fake_platform(instance=instance, autostart=autostart, processes=processes))
    handlers = MessageHandlers()

    deps = app_module.Deps()

    assert deps.autostart_command() == '"spells" --from-the-fake'
    assert deps.cpu_plan() is processes.plan
    assert deps.acquire_single_instance("Spells") is False
    deps.release_single_instance("Spells")
    assert deps.signal_running_instance("SpellsMessageWindow", "quit") is True
    assert deps.find_message_window("SpellsMessageWindow") == 42
    assert deps.message_window("SpellsMessageWindow", handlers) == "the window"
    assert instance.calls == [
        ("acquire", "Spells"),
        ("release", "Spells"),
        ("signal", "SpellsMessageWindow", "quit"),
        ("find", "SpellsMessageWindow"),
        ("window", "SpellsMessageWindow", handlers),
    ]
    assert deps.autostart_apply(True, "cmd") is True
    assert autostart.calls == [(True, "cmd")]


def test_deps_follow_the_platform_installed_when_they_are_built(use_platform):
    first = RecordingAutostart("first")
    use_platform(fake_platform(autostart=first))
    early = app_module.Deps()
    second = RecordingAutostart("second")
    use_platform(fake_platform(autostart=second))
    late = app_module.Deps()

    assert early.autostart_command() == '"first" --from-the-fake'
    assert late.autostart_command() == '"second" --from-the-fake'
    early.autostart_apply(True, "early")
    late.autostart_apply(False, "late")
    assert first.calls == [(True, "early")]
    assert second.calls == [(False, "late")]


def test_the_stub_lets_the_app_run_as_the_only_instance_with_a_quiet_window(use_platform):
    use_platform(fake_platform())

    deps = app_module.Deps()

    assert deps.autostart_command() == ""
    assert deps.autostart_apply(True, "cmd") is False
    assert deps.acquire_single_instance("Spells") is True
    assert deps.find_message_window("SpellsMessageWindow") == 0
    assert deps.signal_running_instance("SpellsMessageWindow", "quit") is False
    window = deps.message_window("SpellsMessageWindow", MessageHandlers())
    window.run(None)
    window.destroy()


@pytest.mark.windows
def test_windows_builds_the_instance_autostart_and_updater_from_win32():
    from spells.platform import windows

    record = windows.build()

    assert isinstance(record.instance, windows.WindowsInstance)
    assert isinstance(record.autostart, windows.WindowsAutostart)
    assert isinstance(record.updater, windows.WindowsUpdater)
    assert record.updater.can_apply is True


@pytest.mark.windows
def test_windows_instance_wraps_the_mutex_the_window_and_the_copydata_channel(monkeypatch):
    from spells.platform import windows
    from spells.win32 import instance as win32_instance
    from spells.win32 import msgwindow

    calls = []
    monkeypatch.setattr(
        win32_instance,
        "acquire_single_instance",
        lambda name: calls.append(("acquire", name)) or True,
    )
    monkeypatch.setattr(
        win32_instance, "release_single_instance", lambda name: calls.append(("release", name))
    )
    monkeypatch.setattr(
        win32_instance,
        "signal_running_instance",
        lambda name, command: calls.append(("signal", name, command)) or True,
    )
    monkeypatch.setattr(
        win32_instance, "find_message_window", lambda name: calls.append(("find", name)) or 7
    )

    class FakeWindow:
        def __init__(self, class_name, handlers):
            calls.append(("window", class_name, handlers))

    monkeypatch.setattr(msgwindow, "MessageWindow", FakeWindow)
    handlers = MessageHandlers()
    instance = windows.build().instance

    assert instance.acquire("Spells") is True
    instance.release("Spells")
    assert instance.signal_running("SpellsMessageWindow", "quit") is True
    assert instance.find_running("SpellsMessageWindow") == 7
    assert isinstance(instance.message_window("SpellsMessageWindow", handlers), FakeWindow)
    assert calls == [
        ("acquire", "Spells"),
        ("release", "Spells"),
        ("signal", "SpellsMessageWindow", "quit"),
        ("find", "SpellsMessageWindow"),
        ("window", "SpellsMessageWindow", handlers),
    ]


@pytest.mark.windows
def test_windows_autostart_writes_the_run_value_for_this_installation(monkeypatch):
    from spells.platform import windows
    from spells.win32 import autostart as win32_autostart

    calls = []
    monkeypatch.setattr(
        win32_autostart, "apply", lambda enabled, command: calls.append((enabled, command)) or True
    )
    autostart = windows.build().autostart

    assert autostart.current_command() == win32_autostart.current_command()
    assert autostart.apply(True, '"Spells.exe"') is True
    assert calls == [(True, '"Spells.exe"')]


@pytest.mark.windows
def test_windows_updater_starts_the_installer(monkeypatch, tmp_path):
    from spells import updates
    from spells.platform import windows

    started = []
    monkeypatch.setattr(updates, "launch_installer", lambda path: started.append(path))
    installer = tmp_path / "Spells-Online-Setup-0.3.0.exe"

    windows.build().updater.apply(installer)

    assert started == [installer]


def release_for(payload: bytes = PAYLOAD) -> Release:
    return Release(
        version="0.3.0",
        released="2026-10-01",
        url="https://spells.example.com/Spells-Online-Setup-0.3.0.exe",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        changes=("A faster Albanian.",),
    )


def coordinator_for(tmp_path, **kwargs) -> UpdateCoordinator:
    kwargs.setdefault("fetch", lambda url: release_for())
    return UpdateCoordinator(
        config=ConfigStore(tmp_path / "settings.json"),
        source_url=SOURCE,
        runner=direct_runner,
        clock=lambda: 1_000_000.0,
        current_version="0.2.0",
        **kwargs,
    )


def write_installer(release, directory, progress=None, cancel=None):
    path = Path(directory) / "Spells-Online-Setup-0.3.0.exe"
    path.write_bytes(PAYLOAD)
    return path


def test_the_releases_page_is_the_latest_github_release():
    assert updatecheck.RELEASES_URL == "https://github.com/kapidolli/spells/releases/latest"


def test_without_self_update_the_check_links_to_the_releases_page(app, tmp_path, use_platform):
    use_platform(fake_platform())
    downloads = []
    coordinator = coordinator_for(
        tmp_path, download=lambda *args, **kwargs: downloads.append(args), temp_dir=tmp_path
    )

    coordinator.check()

    view = coordinator.view
    assert view.phase is Phase.BLOCKED
    assert view.release is not None and view.release.version == "0.3.0"
    assert updatecheck.RELEASES_URL in view.message
    coordinator.start_download()
    coordinator.install()
    assert downloads == []
    assert coordinator.view.phase is Phase.BLOCKED


def test_without_self_update_the_about_page_shows_the_link_instead_of_install(
    app, tmp_path, use_platform
):
    use_platform(fake_platform())
    coordinator = coordinator_for(tmp_path)
    page = AboutPage()
    page.attach(coordinator)

    coordinator.check()

    assert not page.release_panel.isHidden()
    assert page.install_button.isHidden()
    assert updatecheck.RELEASES_URL in page.version_row.description_label.text()
    page.close()


def test_without_self_update_the_weekly_check_still_announces_the_version(
    app, tmp_path, use_platform
):
    use_platform(fake_platform())
    seen = []
    coordinator = coordinator_for(tmp_path)
    coordinator.update_found.connect(lambda version, when: seen.append(version))
    coordinator.set_weekly_check(True)

    coordinator.tick()

    assert seen == ["0.3.0"]
    assert coordinator.view.phase is Phase.BLOCKED


def test_a_self_updating_platform_installs_through_its_updater(app, tmp_path, use_platform):
    updater = RecordingUpdater()
    use_platform(fake_platform(capabilities=Capabilities.everything(), updater=updater))
    quits = []
    coordinator = coordinator_for(
        tmp_path, download=write_installer, on_quit=lambda: quits.append(1), temp_dir=tmp_path
    )

    coordinator.check()
    assert coordinator.view.phase is Phase.AVAILABLE
    assert updatecheck.RELEASES_URL not in coordinator.view.message
    coordinator.start_download()
    coordinator.install()

    assert [path.name for path in updater.applied] == ["Spells-Online-Setup-0.3.0.exe"]
    assert quits == [1]
    assert coordinator.view.phase is Phase.INSTALLING
