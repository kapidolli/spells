"""spells.updates: version comparison, the version file, the download and the two timers.

Nothing here opens a connection or starts a process: every worker takes its opener, its
progress callback, its cancel flag and its runner as arguments, and the tests pass fakes.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from spells import config, updates

SOURCE = "https://spells.example.com/latest.json"
INSTALLER = "https://spells.example.com/Spells-Online-Setup-0.3.0.exe"


def manifest(**overrides) -> dict:
    body = {
        "schema": 1,
        "product": "Spells",
        "version": "0.3.0",
        "released": "2026-10-01",
        "minimum_version": "",
        "installer": {
            "url": INSTALLER,
            "size_bytes": 97_296_166,
            "sha256": "a" * 64,
        },
        "changes": ["A faster Albanian.", "The tray remembers where it was."],
    }
    if "installer" in overrides:
        body["installer"] = overrides.pop("installer")
    body.update(overrides)
    return body


def text_of(**overrides) -> str:
    return json.dumps(manifest(**overrides))


class FakeResponse:
    """The parts of an http response the workers use, and a record of the reads."""

    def __init__(self, body: bytes, *, chunk: int = 1 << 20, fail_after: int | None = None):
        self.body = body
        self.chunk = chunk
        self.fail_after = fail_after
        self.position = 0
        self.reads = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise OSError("the connection dropped")
        want = self.chunk if size is None or size < 0 else min(size, self.chunk)
        piece = self.body[self.position : self.position + want]
        self.position += len(piece)
        return piece

    def close(self) -> None:
        self.closed = True


def opener_for(body: bytes, **kwargs):
    calls: list[tuple[str, float]] = []
    response = FakeResponse(body, **kwargs)

    def opener(url: str, timeout: float):
        calls.append((url, timeout))
        return response

    opener.calls = calls
    opener.response = response
    return opener


# --- version comparison ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["0.2.0", "v0.2.0", "1.2", "1", "1.2.3.4", "0.2.0-dev", " 0.2.0 ", "10.20.30"],
)
def test_a_version_this_app_can_read(text):
    assert updates.parse_version(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "abc",
        "1.2.x",
        "-1.2",
        "1.2.3.4.5",
        "1234567.0",
        "0.2.0; rm -rf",
        "0..2",
        1,
        2.0,
        ["0.2.0"],
        {"version": "0.2.0"},
        "0.2.0\n1.0.0",
    ],
)
def test_a_version_this_app_refuses_to_read(text):
    assert updates.parse_version(text) is None


@pytest.mark.parametrize(
    ("newer", "older"),
    [
        ("0.2.0", "0.1.0"),
        ("0.10.0", "0.9.0"),
        ("1.0", "0.99.99"),
        ("0.2.1", "0.2.0"),
        ("0.2.0", "0.2.0-dev"),
        ("0.2.0.1", "0.2.0"),
    ],
)
def test_the_higher_version_wins(newer, older):
    assert updates.compare_versions(newer, older) == 1
    assert updates.compare_versions(older, newer) == -1
    assert updates.is_newer(newer, older)
    assert not updates.is_newer(older, newer)


def test_the_same_version_written_two_ways_is_the_same_version():
    assert updates.compare_versions("0.2", "0.2.0") == 0
    assert updates.compare_versions("v0.2.0", "0.2.0.0") == 0
    assert not updates.is_newer("0.2.0", "0.2")


def test_comparing_an_unreadable_version_raises():
    with pytest.raises(updates.UpdateError):
        updates.compare_versions("0.2.0", "banana")


@pytest.mark.parametrize(
    ("candidate", "current"),
    [("banana", "0.1.0"), ("0.2.0", "banana"), (None, "0.1.0"), ("", "")],
)
def test_an_update_this_app_cannot_reason_about_is_never_newer(candidate, current):
    assert not updates.is_newer(candidate, current)


# --- the version file -----------------------------------------------------------------------


def test_a_good_version_file_parses():
    release = updates.parse_manifest(text_of(), source_url=SOURCE)

    assert release.version == "0.3.0"
    assert release.released == "2026-10-01"
    assert release.url == INSTALLER
    assert release.size_bytes == 97_296_166
    assert release.sha256 == "a" * 64
    assert release.changes == ("A faster Albanian.", "The tray remembers where it was.")


def test_the_file_may_arrive_as_bytes_with_a_byte_order_mark():
    raw = text_of().encode("utf-8-sig")

    assert updates.parse_manifest(raw, source_url=SOURCE).version == "0.3.0"


def test_an_uppercase_hash_is_read_as_the_same_hash():
    release = updates.parse_manifest(
        text_of(installer={"url": INSTALLER, "size_bytes": 2048, "sha256": "A" * 64}),
        source_url=SOURCE,
    )

    assert release.sha256 == "a" * 64


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json at all",
        "[]",
        '"a string"',
        "null",
        b"\xff\xfe\x00",
    ],
)
def test_a_file_that_is_not_a_version_file_is_refused(raw):
    with pytest.raises(updates.UpdateError):
        updates.parse_manifest(raw, source_url=SOURCE)


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": "banana"},
        {"version": None},
        {"version": 3},
        {"product": "NotSpells"},
        {"schema": 2},
        {"schema": "one"},
        {"schema": 0},
        {"released": "the first of October"},
        {"released": 20261001},
        {"minimum_version": "banana"},
        {"changes": "a single string"},
        {"changes": ["fine", 7]},
        {"changes": ["fine", ""]},
        {"changes": [f"line {index}" for index in range(updates.MAX_CHANGES + 1)]},
    ],
)
def test_a_malformed_version_file_is_refused(overrides):
    with pytest.raises(updates.UpdateError):
        updates.parse_manifest(text_of(**overrides), source_url=SOURCE)


@pytest.mark.parametrize(
    "installer",
    [
        None,
        "https://spells.example.com/setup.exe",
        {"url": INSTALLER, "size_bytes": 97_296_166},
        {"url": INSTALLER, "size_bytes": 97_296_166, "sha256": "not a hash"},
        {"url": INSTALLER, "size_bytes": 97_296_166, "sha256": "a" * 63},
        {"url": INSTALLER, "size_bytes": 0, "sha256": "a" * 64},
        {"url": INSTALLER, "size_bytes": -5, "sha256": "a" * 64},
        {"url": INSTALLER, "size_bytes": 10**12, "sha256": "a" * 64},
        {"url": INSTALLER, "size_bytes": True, "sha256": "a" * 64},
        {"url": "", "size_bytes": 2048, "sha256": "a" * 64},
    ],
)
def test_a_version_file_whose_installer_is_wrong_is_refused(installer):
    with pytest.raises(updates.UpdateError):
        updates.parse_manifest(text_of(installer=installer), source_url=SOURCE)


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/Windows/System32/cmd.exe",
        r"\\attacker\share\setup.exe",
        "ftp://spells.example.com/setup.exe",
        "javascript:alert(1)",
        "https://attacker.example.com/Spells-Setup.exe",
        "http://spells.example.com/Spells-Setup.exe",
        "https://spells.example.com:8443/Spells-Setup.exe",
    ],
)
def test_an_installer_that_is_not_on_the_version_file_s_own_server_is_refused(url):
    body = text_of(installer={"url": url, "size_bytes": 2048, "sha256": "a" * 64})

    with pytest.raises(updates.UpdateError):
        updates.parse_manifest(body, source_url=SOURCE)


def test_a_version_file_far_too_large_is_refused_before_it_is_parsed():
    with pytest.raises(updates.UpdateError):
        updates.parse_manifest("x" * (updates.MAX_MANIFEST_BYTES + 1), source_url=SOURCE)


def test_control_characters_in_a_change_line_never_reach_the_page():
    body = text_of(changes=["Fixed the meter.\r\nInstall now: https://attacker.example.com"])

    release = updates.parse_manifest(body, source_url=SOURCE)

    assert "\n" not in release.changes[0]
    assert "\r" not in release.changes[0]


def test_a_very_long_change_line_is_trimmed():
    release = updates.parse_manifest(text_of(changes=["x" * 5000]), source_url=SOURCE)

    assert len(release.changes[0]) == updates.MAX_CHANGE_CHARS


def test_a_version_file_without_a_source_to_compare_against_still_checks_the_scheme():
    body = text_of(installer={"url": "file:///c:/x.exe", "size_bytes": 2048, "sha256": "a" * 64})

    with pytest.raises(updates.UpdateError):
        updates.parse_manifest(body)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (INSTALLER, "Spells-Online-Setup-0.3.0.exe"),
        ("https://x.example.com/a/b/Setup.exe", "Setup.exe"),
        ("https://x.example.com/..%2f..%2fevil.exe", "Spells-Setup.exe"),
        ("https://x.example.com/setup.exe?a=b", "setup.exe"),
        ("https://x.example.com/", "Spells-Setup.exe"),
        ("https://x.example.com/setup.bat", "Spells-Setup.exe"),
        ("https://x.example.com/.exe", "Spells-Setup.exe"),
    ],
)
def test_the_download_never_chooses_its_own_place_on_the_disk(url, expected):
    assert updates.installer_name(url) == expected


# --- deciding -------------------------------------------------------------------------------


def test_a_newer_release_is_an_update():
    release = updates.parse_manifest(text_of(), source_url=SOURCE)

    decision = updates.decide(release, "0.2.0")

    assert decision.available
    assert not decision.blocked
    assert decision.release is release


@pytest.mark.parametrize("current", ["0.3.0", "0.4.0", "1.0.0"])
def test_the_same_or_an_older_release_is_not_an_update(current):
    decision = updates.decide(updates.parse_manifest(text_of(), source_url=SOURCE), current)

    assert not decision.available
    assert current in decision.reason


def test_nothing_read_is_not_an_update():
    assert not updates.decide(None, "0.2.0").available


def test_a_release_that_needs_an_older_one_first_is_blocked_and_names_it():
    release = updates.parse_manifest(text_of(minimum_version="0.2.0"), source_url=SOURCE)

    decision = updates.decide(release, "0.1.0")

    assert decision.available
    assert decision.blocked
    assert "0.2.0" in decision.reason


def test_a_minimum_version_already_installed_does_not_block():
    release = updates.parse_manifest(text_of(minimum_version="0.2.0"), source_url=SOURCE)

    assert not updates.decide(release, "0.2.0").blocked


# --- the two timers -------------------------------------------------------------------------


def test_the_weekly_check_never_runs_without_permission():
    assert not updates.check_due(enabled=False, now=10 * updates.WEEK_S, last_check=0.0)


def test_the_weekly_check_runs_once_when_it_has_never_run():
    assert updates.check_due(enabled=True, now=1000.0, last_check=0.0)


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [(0, False), (updates.WEEK_S - 1, False), (updates.WEEK_S, True), (updates.WEEK_S * 3, True)],
)
def test_the_weekly_check_waits_a_week(elapsed, expected):
    now = 2_000_000.0

    assert updates.check_due(enabled=True, now=now, last_check=now - elapsed) is expected


def test_a_clock_that_moved_backwards_checks_once_rather_than_never_again():
    assert updates.check_due(enabled=True, now=1000.0, last_check=9_000_000.0)


def test_a_version_never_mentioned_before_gets_its_balloon_at_once():
    assert updates.balloon_due(now=1000.0, last_at=999.0, version="0.3.0", last_version="0.2.0")


def test_the_same_version_gets_one_balloon_a_day():
    now = 5_000_000.0

    assert not updates.balloon_due(
        now=now, last_at=now - 60.0, version="0.3.0", last_version="0.3.0"
    )
    assert updates.balloon_due(
        now=now, last_at=now - updates.DAY_S, version="0.3.0", last_version="0.3.0"
    )


def test_no_version_means_no_balloon():
    assert not updates.balloon_due(now=1.0, last_at=0.0, version="", last_version="")


# --- the address fixed at build time --------------------------------------------------------


def test_the_build_s_address_is_read_from_the_app_folder(tmp_path):
    (tmp_path / updates.SOURCE_FILE).write_text(
        json.dumps({"manifest_url": SOURCE}), encoding="utf-8"
    )

    assert updates.source_url(tmp_path) == SOURCE


@pytest.mark.parametrize(
    "body",
    [
        "",
        "{",
        "[]",
        json.dumps({}),
        json.dumps({"manifest_url": ""}),
        json.dumps({"manifest_url": "file:///c:/latest.json"}),
        json.dumps({"manifest_url": "not an address"}),
        json.dumps({"manifest_url": 7}),
    ],
)
def test_a_build_without_a_usable_address_checks_nothing(tmp_path, body):
    (tmp_path / updates.SOURCE_FILE).write_text(body, encoding="utf-8")

    assert updates.source_url(tmp_path) == ""


def test_a_development_checkout_has_no_address_and_does_not_raise(tmp_path):
    assert updates.source_url(tmp_path / "nowhere") == ""


# --- fetching -------------------------------------------------------------------------------


def test_fetching_reads_the_address_it_was_given_and_times_out_quickly():
    opener = opener_for(text_of().encode("utf-8"))

    release = updates.fetch_release(SOURCE, opener=opener, timeout_s=3.0)

    assert release.version == "0.3.0"
    assert opener.calls == [(SOURCE, 3.0)]
    assert opener.response.closed


def test_a_server_that_cannot_be_reached_is_a_sentence_not_a_traceback():
    def opener(url, timeout):
        raise OSError("name or service not known")

    with pytest.raises(updates.UpdateError) as caught:
        updates.fetch_release(SOURCE, opener=opener)

    assert "could not reach" in str(caught.value)


def test_an_address_that_is_not_http_is_never_opened():
    opened: list[str] = []

    def opener(url, timeout):
        opened.append(url)
        raise AssertionError("this must never run")

    with pytest.raises(updates.UpdateError):
        updates.fetch_release("file:///c:/latest.json", opener=opener)
    assert opened == []


def test_an_installer_on_another_host_is_refused_even_when_the_server_answered():
    body = text_of(
        installer={
            "url": "https://attacker.example.com/Setup.exe",
            "size_bytes": 2048,
            "sha256": "a" * 64,
        }
    )

    with pytest.raises(updates.UpdateError):
        updates.fetch_release(SOURCE, opener=opener_for(body.encode("utf-8")))


# --- downloading ----------------------------------------------------------------------------


def release_for(payload: bytes, **overrides) -> updates.Release:
    fields = {
        "version": "0.3.0",
        "url": INSTALLER,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    fields.update(overrides)
    return updates.Release(**fields)


def test_a_good_download_lands_verified_with_progress_reported(tmp_path):
    payload = b"setup bytes" * 900
    release = release_for(payload)
    seen: list[tuple[int, int]] = []

    path = updates.download_installer(
        release,
        tmp_path,
        opener=opener_for(payload, chunk=1000),
        progress=lambda done, total: seen.append((done, total)),
    )

    assert path.name == "Spells-Online-Setup-0.3.0.exe"
    assert path.read_bytes() == payload
    assert seen[-1] == (len(payload), len(payload))
    assert all(total == len(payload) for _done, total in seen)


def test_a_tampered_download_is_deleted_and_nothing_is_launched(tmp_path):
    payload = b"the real installer"
    release = release_for(payload)
    tampered = b"an installer somebody else wrote"

    with pytest.raises(updates.UpdateError) as caught:
        updates.download_installer(
            release, tmp_path, opener=opener_for(tampered + payload[: len(payload) - 1])
        )

    assert "checksum" in str(caught.value) or "larger" in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_a_download_with_the_right_size_and_the_wrong_bytes_is_refused(tmp_path):
    payload = b"x" * 4096
    release = release_for(payload)

    with pytest.raises(updates.UpdateError) as caught:
        updates.download_installer(release, tmp_path, opener=opener_for(b"y" * 4096))

    assert "checksum" in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_a_download_that_ends_early_is_refused(tmp_path):
    payload = b"z" * 4096
    release = release_for(payload)

    with pytest.raises(updates.UpdateError) as caught:
        updates.download_installer(release, tmp_path, opener=opener_for(payload[:100]))

    assert "ended early" in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_a_download_larger_than_promised_stops_rather_than_filling_the_disk(tmp_path):
    release = release_for(b"q" * 100)

    with pytest.raises(updates.UpdateError) as caught:
        updates.download_installer(release, tmp_path, opener=opener_for(b"q" * 50_000, chunk=64))

    assert "larger than" in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_a_connection_that_drops_mid_download_leaves_nothing_behind(tmp_path):
    payload = b"p" * 8192
    release = release_for(payload)

    with pytest.raises(updates.UpdateError):
        updates.download_installer(
            release, tmp_path, opener=opener_for(payload, chunk=512, fail_after=3)
        )

    assert list(tmp_path.iterdir()) == []


def test_cancelling_stops_the_download_and_removes_the_partial_file(tmp_path):
    payload = b"c" * 20_000
    release = release_for(payload)
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(updates.UpdateCancelled):
        updates.download_installer(release, tmp_path, opener=opener_for(payload, chunk=64),
                                   cancel=cancel)

    assert list(tmp_path.iterdir()) == []


def test_a_release_without_a_usable_hash_is_never_downloaded(tmp_path):
    release = updates.Release(version="0.3.0", url=INSTALLER, size_bytes=10, sha256="")

    def opener(url, timeout):
        raise AssertionError("this must never run")

    with pytest.raises(updates.UpdateError):
        updates.download_installer(release, tmp_path, opener=opener)


def test_verify_file_agrees_with_the_download(tmp_path):
    payload = b"m" * 300
    release = release_for(payload)
    path = updates.download_installer(release, tmp_path, opener=opener_for(payload))

    assert updates.verify_file(path, release)
    path.write_bytes(b"m" * 299 + b"n")
    assert not updates.verify_file(path, release)
    path.unlink()
    assert not updates.verify_file(path, release)


# --- launching ------------------------------------------------------------------------------


def test_the_installer_is_started_silently_and_asks_for_the_app_to_come_back(tmp_path):
    installer = tmp_path / "Spells-Setup.exe"
    installer.write_bytes(b"MZ")
    seen: list[list[str]] = []

    updates.launch_installer(installer, runner=lambda command: seen.append(list(command)))

    assert seen[0][0] == str(installer)
    assert seen[0][1:] == list(updates.INSTALL_ARGS)
    assert "/SILENT" in seen[0]
    assert "/RELAUNCH" in seen[0]


@pytest.mark.parametrize("name", ["Spells-Setup.txt", "missing.exe"])
def test_only_an_installer_that_is_there_is_started(tmp_path, name):
    path = tmp_path / name
    if path.suffix == ".txt":
        path.write_bytes(b"MZ")

    def runner(command):
        raise AssertionError("this must never run")

    with pytest.raises(updates.UpdateError):
        updates.launch_installer(path, runner=runner)


def test_a_launch_that_fails_is_a_sentence_not_a_traceback(tmp_path):
    installer = tmp_path / "Spells-Setup.exe"
    installer.write_bytes(b"MZ")

    def runner(command):
        raise OSError("access denied")

    with pytest.raises(updates.UpdateError) as caught:
        updates.launch_installer(installer, runner=runner)

    assert "could not be started" in str(caught.value)


# --- the words on the page ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "expected"),
    [(500, "500 bytes"), (2048, "2 kB"), (97_296_166, "97.3 MB"), (3_873_093_093, "3.87 GB")],
)
def test_a_size_reads_the_way_the_rest_of_the_app_writes_one(size, expected):
    assert updates.size_text(size) == expected


@pytest.mark.parametrize(
    ("released", "expected"),
    [
        ("2026-09-18", "18 September 2026"),
        ("2026-01-01", "1 January 2026"),
        ("", ""),
        ("soon", "soon"),
        ("2026-13-45", "2026-13-45"),
    ],
)
def test_a_date_reads_as_a_date(released, expected):
    assert updates.date_text(released) == expected


# --- the setting ----------------------------------------------------------------------------


def test_the_weekly_check_is_off_in_a_fresh_settings_file():
    settings = config.default_settings()

    assert settings.updates.weekly_check is False
    assert settings.updates.last_check == 0.0
    assert settings.updates.last_offer_version == ""


def test_the_switch_round_trips_through_the_file(tmp_path):
    from dataclasses import replace

    path = tmp_path / "settings.json"
    settings = config.default_settings()
    settings = replace(
        settings,
        updates=replace(settings.updates, weekly_check=True, last_check=1234.5,
                        last_offer_version="0.3.0", last_offer_at=1200.0),
    )
    config.save(settings, path)

    loaded, notice = config.load(path)

    assert notice is None
    assert loaded.updates.weekly_check is True
    assert loaded.updates.last_check == 1234.5
    assert loaded.updates.last_offer_version == "0.3.0"
    assert loaded.updates.last_offer_at == 1200.0


def test_a_settings_file_from_before_this_release_migrates_silently(tmp_path):
    """0.1.0 wrote no updates block at all; it loads with the check off and nothing else moves."""
    path = tmp_path / "settings.json"
    raw = config.to_dict(config.default_settings())
    raw.pop("updates")
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    loaded, notice = config.load(path)

    assert notice is None
    assert loaded.updates.weekly_check is False
    assert loaded.updates.last_check == 0.0
    assert loaded.general.enabled_languages == config.DEFAULT_ENABLED_LANGUAGES
    assert not list(tmp_path.glob("settings.corrupt-*.json"))


@pytest.mark.parametrize(
    "block",
    [
        {"weekly_check": "yes"},
        {"weekly_check": 1},
        {"last_check": "recently"},
        {"last_offer_version": 3},
        {"last_offer_at": None},
    ],
)
def test_an_updates_block_of_the_wrong_shape_is_refused(block):
    raw = config.to_dict(config.default_settings())
    raw["updates"].update(block)

    with pytest.raises(config.SettingsError):
        config.from_dict(raw)


def test_an_unknown_key_in_the_updates_block_is_ignored():
    raw = config.to_dict(config.default_settings())
    raw["updates"]["check_hourly_please"] = True

    assert config.from_dict(raw).updates.weekly_check is False
