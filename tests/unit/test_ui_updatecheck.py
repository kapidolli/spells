"""spells.ui.updatecheck: the coordinator that owns checking, downloading and installing.

Every job runs through `direct_runner`, so a check or a download is one call on this thread:
no worker thread, no event loop and no network. The clock is a fake too, so the weekly rule
and the one-balloon-a-day rule are tested in milliseconds.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from spells import updates
from spells.config import ConfigStore
from spells.ui.updatecheck import Phase, UpdateCoordinator, direct_runner

from .test_ui_support import qt_app

SOURCE = "https://spells.example.com/latest.json"
INSTALLER_URL = "https://spells.example.com/Spells-Online-Setup-0.3.0.exe"


@pytest.fixture(scope="module")
def app():
    return qt_app()


def release(version: str = "0.3.0", **overrides) -> updates.Release:
    fields = {
        "version": version,
        "released": "2026-10-01",
        "url": INSTALLER_URL,
        "size_bytes": 4096,
        "sha256": "a" * 64,
        "changes": ("A faster Albanian.",),
    }
    fields.update(overrides)
    return updates.Release(**fields)


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def build(tmp_path, **kwargs) -> tuple[UpdateCoordinator, ConfigStore, Clock]:
    config = ConfigStore(tmp_path / "settings.json")
    clock = kwargs.pop("clock", Clock())
    kwargs.setdefault("source_url", SOURCE)
    kwargs.setdefault("current_version", "0.2.0")
    kwargs.setdefault("runner", direct_runner)
    coordinator = UpdateCoordinator(config=config, clock=clock, **kwargs)
    return coordinator, config, clock


# --- nothing happens without permission -----------------------------------------------------


def test_a_build_without_an_address_says_so_and_never_fetches(app, tmp_path):
    def fetch(url):
        raise AssertionError("this must never run")

    coordinator, _config, _clock = build(tmp_path, source_url="", fetch=fetch)

    assert not coordinator.configured
    assert coordinator.view.phase is Phase.UNCONFIGURED
    coordinator.check(manual=True)
    coordinator.tick()
    assert coordinator.view.phase is Phase.UNCONFIGURED


def test_the_tick_does_nothing_while_the_weekly_check_is_off(app, tmp_path):
    calls: list[str] = []
    coordinator, _config, clock = build(tmp_path, fetch=lambda url: calls.append(url) or release())

    clock.advance(updates.WEEK_S * 5)
    coordinator.tick()

    assert calls == []
    assert coordinator.view.phase is Phase.IDLE


def test_switching_the_weekly_check_on_writes_the_setting_and_starts_no_check(app, tmp_path):
    calls: list[str] = []
    coordinator, config, _clock = build(
        tmp_path, fetch=lambda url: calls.append(url) or release()
    )

    coordinator.set_weekly_check(True)

    assert config.settings.updates.weekly_check is True
    assert coordinator.view.weekly_check is True
    assert calls == []


def test_the_weekly_check_runs_once_a_week_and_not_sooner(app, tmp_path):
    calls: list[str] = []
    coordinator, config, clock = build(
        tmp_path, fetch=lambda url: calls.append(url) or release()
    )
    coordinator.set_weekly_check(True)

    coordinator.tick()
    assert calls == [SOURCE]

    clock.advance(updates.WEEK_S - 10)
    coordinator.tick()
    assert calls == [SOURCE]

    clock.advance(20)
    coordinator.tick()
    assert calls == [SOURCE, SOURCE]
    assert config.settings.updates.last_check == clock.now


def test_the_weekly_check_waits_while_a_dictation_is_in_flight(app, tmp_path):
    calls: list[str] = []
    dictating = {"now": True}
    coordinator, _config, clock = build(
        tmp_path,
        fetch=lambda url: calls.append(url) or release(),
        busy=lambda: dictating["now"],
    )
    coordinator.set_weekly_check(True)

    clock.advance(updates.WEEK_S)
    coordinator.tick()
    assert calls == []

    dictating["now"] = False
    coordinator.tick()
    assert calls == [SOURCE]


def test_the_button_always_works_even_with_the_weekly_check_off(app, tmp_path):
    calls: list[str] = []
    coordinator, config, _clock = build(
        tmp_path, fetch=lambda url: calls.append(url) or release()
    )

    coordinator.check(manual=True)

    assert calls == [SOURCE]
    assert config.settings.updates.weekly_check is False
    assert coordinator.view.phase is Phase.AVAILABLE


# --- what the check produces ------------------------------------------------------------------


def test_a_newer_version_becomes_an_offer_with_its_change_list(app, tmp_path):
    coordinator, _config, _clock = build(tmp_path, fetch=lambda url: release())

    coordinator.check()

    view = coordinator.view
    assert view.phase is Phase.AVAILABLE
    assert view.release is not None
    assert view.release.version == "0.3.0"
    assert view.release.changes == ("A faster Albanian.",)


def test_the_newest_version_already_installed_says_so(app, tmp_path):
    coordinator, _config, _clock = build(
        tmp_path, fetch=lambda url: release("0.2.0"), current_version="0.2.0"
    )

    coordinator.check()

    assert coordinator.view.phase is Phase.UP_TO_DATE
    assert "0.2.0" in coordinator.view.message


def test_a_release_that_needs_an_older_one_first_offers_no_install(app, tmp_path):
    coordinator, _config, _clock = build(
        tmp_path,
        fetch=lambda url: release(minimum_version="0.9.0"),
        current_version="0.2.0",
    )

    coordinator.check()

    assert coordinator.view.phase is Phase.BLOCKED
    assert "0.9.0" in coordinator.view.message
    coordinator.start_download()
    assert coordinator.view.phase is Phase.BLOCKED


def test_a_check_that_fails_is_a_line_on_the_page_and_nothing_else(app, tmp_path):
    def fetch(url):
        raise updates.UpdateError("Spells could not reach the update server (timed out).")

    coordinator, config, clock = build(tmp_path, fetch=fetch)

    coordinator.check()

    assert coordinator.view.phase is Phase.ERROR
    assert "could not reach" in coordinator.view.message
    assert config.settings.updates.last_check == clock.now


def test_a_failed_check_still_counts_as_the_week_s_check(app, tmp_path):
    calls: list[str] = []

    def fetch(url):
        calls.append(url)
        raise updates.UpdateError("no")

    coordinator, _config, clock = build(tmp_path, fetch=fetch)
    coordinator.set_weekly_check(True)

    coordinator.tick()
    clock.advance(60)
    coordinator.tick()

    assert calls == [SOURCE]


# --- the balloon ------------------------------------------------------------------------------


def test_the_weekly_check_offers_the_version_once_a_day(app, tmp_path):
    seen: list[tuple[str, str]] = []
    coordinator, _config, clock = build(tmp_path, fetch=lambda url: release())
    coordinator.update_found.connect(lambda version, when: seen.append((version, when)))
    coordinator.set_weekly_check(True)

    coordinator.tick()
    assert seen == [("0.3.0", "1 October 2026")]

    clock.advance(updates.WEEK_S)
    coordinator.tick()
    assert len(seen) == 2


def test_a_manual_check_never_raises_a_balloon(app, tmp_path):
    seen: list[tuple[str, str]] = []
    coordinator, _config, _clock = build(tmp_path, fetch=lambda url: release())
    coordinator.update_found.connect(lambda version, when: seen.append((version, when)))

    coordinator.check(manual=True)

    assert seen == []
    assert coordinator.view.phase is Phase.AVAILABLE


def test_the_balloon_is_held_back_while_a_dictation_is_in_flight(app, tmp_path):
    seen: list[tuple[str, str]] = []
    busy = {"now": False}
    coordinator, config, _clock = build(
        tmp_path, fetch=lambda url: release(), busy=lambda: busy["now"]
    )
    coordinator.update_found.connect(lambda version, when: seen.append((version, when)))
    coordinator.set_weekly_check(True)
    busy["now"] = True

    coordinator._checked(release(), manual=False)

    assert seen == []
    assert config.settings.updates.last_offer_version == ""


# --- downloading and installing -----------------------------------------------------------------


def test_a_verified_download_leads_to_the_install_button(app, tmp_path):
    payload = b"MZ setup bytes"
    target = tmp_path / "downloaded"
    target.mkdir()

    def download(release_arg, directory, progress=None, cancel=None):
        assert Path(directory) == target
        progress(len(payload), release_arg.size_bytes)
        path = Path(directory) / "Spells-Online-Setup-0.3.0.exe"
        path.write_bytes(payload)
        return path

    coordinator, _config, _clock = build(
        tmp_path,
        fetch=lambda url: release(size_bytes=len(payload)),
        download=download,
        temp_dir=target,
    )
    coordinator.check()
    coordinator.start_download()

    assert coordinator.view.phase is Phase.READY
    assert "close" in coordinator.view.message


def test_a_tampered_download_never_reaches_the_install_button(app, tmp_path):
    def download(release_arg, directory, progress=None, cancel=None):
        raise updates.UpdateError(
            "The downloaded installer does not match its checksum, so Spells deleted it "
            "and will not run it."
        )

    launched: list[Path] = []
    coordinator, _config, _clock = build(
        tmp_path,
        fetch=lambda url: release(),
        download=download,
        launch=lambda path: launched.append(path),
        temp_dir=tmp_path,
    )
    coordinator.check()
    coordinator.start_download()

    assert coordinator.view.phase is Phase.ERROR
    assert "checksum" in coordinator.view.message
    coordinator.install()
    assert launched == []


def test_cancelling_puts_the_offer_back(app, tmp_path):
    holder: dict[str, UpdateCoordinator] = {}

    def download(release_arg, directory, progress=None, cancel=None):
        holder["it"].cancel_download()
        assert cancel()
        raise updates.UpdateCancelled("The download was cancelled.")

    coordinator, _config, _clock = build(
        tmp_path, fetch=lambda url: release(), download=download, temp_dir=tmp_path
    )
    holder["it"] = coordinator
    coordinator.check()
    coordinator.start_download()

    assert coordinator.view.phase is Phase.AVAILABLE
    assert "cancelled" in coordinator.view.message


def test_installing_starts_the_installer_and_quits_the_app(app, tmp_path):
    payload = b"MZ the real installer"
    target = tmp_path / "downloaded"
    target.mkdir()

    def download(release_arg, directory, progress=None, cancel=None):
        path = Path(directory) / "Spells-Online-Setup-0.3.0.exe"
        path.write_bytes(payload)
        return path

    launched: list[Path] = []
    quits: list[int] = []
    coordinator, _config, _clock = build(
        tmp_path,
        fetch=lambda url: release(
            size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        ),
        download=download,
        launch=lambda path: launched.append(Path(path)),
        on_quit=lambda: quits.append(1),
        temp_dir=target,
    )
    coordinator.check()
    coordinator.start_download()
    coordinator.install()

    assert [p.name for p in launched] == ["Spells-Online-Setup-0.3.0.exe"]
    assert quits == [1]
    assert coordinator.view.phase is Phase.INSTALLING


def test_a_file_changed_after_it_was_verified_is_refused_at_the_last_moment(app, tmp_path):
    payload = b"MZ the real installer"
    target = tmp_path / "downloaded"
    target.mkdir()

    def download(release_arg, directory, progress=None, cancel=None):
        path = Path(directory) / "Spells-Online-Setup-0.3.0.exe"
        path.write_bytes(payload)
        return path

    launched: list[Path] = []
    quits: list[int] = []
    coordinator, _config, _clock = build(
        tmp_path,
        fetch=lambda url: release(
            size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        ),
        download=download,
        launch=lambda path: launched.append(Path(path)),
        on_quit=lambda: quits.append(1),
        temp_dir=target,
    )
    coordinator.check()
    coordinator.start_download()
    (target / "Spells-Online-Setup-0.3.0.exe").write_bytes(b"MZ somebody else wrote this")

    coordinator.install()

    assert launched == []
    assert quits == []
    assert coordinator.view.phase is Phase.ERROR


def test_nothing_installs_without_a_download(app, tmp_path):
    launched: list[Path] = []
    coordinator, _config, _clock = build(
        tmp_path, fetch=lambda url: release(), launch=lambda path: launched.append(path)
    )
    coordinator.check()

    coordinator.install()

    assert launched == []
    assert coordinator.view.phase is Phase.AVAILABLE


def test_the_progress_of_a_download_reaches_the_view(app, tmp_path):
    seen: list[tuple[int, int]] = []

    def download(release_arg, directory, progress=None, cancel=None):
        progress(1024, 4096)
        progress(4096, 4096)
        path = Path(directory) / "Spells-Online-Setup-0.3.0.exe"
        path.write_bytes(b"x" * 4096)
        return path

    coordinator, _config, _clock = build(
        tmp_path, fetch=lambda url: release(), download=download, temp_dir=tmp_path
    )
    coordinator.changed.connect(
        lambda view: seen.append((view.done_bytes, view.total_bytes))
    )
    coordinator.check()
    coordinator.start_download()

    assert (1024, 4096) in seen


# --- following the settings file ---------------------------------------------------------------


def test_a_switch_flipped_elsewhere_reaches_the_view(app, tmp_path):
    coordinator, config, _clock = build(tmp_path, fetch=lambda url: release())

    config.update(
        lambda settings: replace(
            settings, updates=replace(settings.updates, weekly_check=True)
        )
    )
    coordinator.apply_settings(config.settings)

    assert coordinator.view.weekly_check is True


def test_a_coordinator_built_on_a_file_that_already_has_the_switch_on_follows_it(app, tmp_path):
    config = ConfigStore(tmp_path / "settings.json")
    config.update(
        lambda settings: replace(
            settings, updates=replace(settings.updates, weekly_check=True, last_check=500.0)
        )
    )

    coordinator = UpdateCoordinator(
        config=config, source_url=SOURCE, runner=direct_runner, clock=Clock()
    )

    assert coordinator.view.weekly_check is True
    assert coordinator.view.last_check == 500.0
