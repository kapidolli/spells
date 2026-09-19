"""Tests for spells.profiles: built-in table, user rules, matching order."""

from __future__ import annotations

import dataclasses

import pytest

from spells import profiles
from spells.config import DEFAULT_TONES, ProfileRule, Settings, default_settings
from spells.models import DeliveryMethod, Profile, TargetContext

BROWSERS = ["chrome.exe", "msedge.exe", "firefox.exe"]

# A user rule's profile snapshot: Email and docs, typed instead of pasted. Its tone
# is the Cleanup tab's tone for that name, which wins over any snapshot (see
# test_tone_override_applies_to_user_rules_by_profile_name).
CUSTOM = Profile(
    name="Email and docs",
    cleanup=True,
    tone=DEFAULT_TONES["Email and docs"],
    delivery=DeliveryMethod.TYPE,
)


def _ctx(process: str, title: str = "", hwnd: int = 1) -> TargetContext:
    return TargetContext(hwnd=hwnd, process=process, title=title, captured_at=0.0)


def _settings(*rules: ProfileRule, tones: dict[str, str] | None = None) -> Settings:
    settings = default_settings()
    if tones is not None:
        cleanup = dataclasses.replace(settings.cleanup, tones=tones)
        settings = dataclasses.replace(settings, cleanup=cleanup)
    return dataclasses.replace(settings, profiles=list(rules))


def _match(ctx: TargetContext, settings: Settings | None = None) -> Profile:
    return profiles.match(ctx, settings if settings is not None else default_settings())


# --- built-in table ---


def test_builtin_rules_cover_the_spec_table():
    by_name = {rule.name: rule for rule in profiles.BUILTIN_RULES}

    assert list(by_name) == ["Chat", "Email and docs", "Code", "Terminal"]
    assert by_name["Chat"].match_process == ["ms-teams.exe", "slack.exe", "WhatsApp.exe"]
    assert by_name["Chat"].match_title == ["Slack", "WhatsApp"]
    assert by_name["Email and docs"].match_process == ["OUTLOOK.EXE", "olk.exe", "WINWORD.EXE"]
    assert by_name["Email and docs"].match_title == ["Gmail", "Outlook"]
    assert by_name["Code"].match_process == ["Code.exe", "idea64.exe", "devenv.exe"]
    assert by_name["Code"].match_title == []
    assert by_name["Terminal"].match_process == [
        "WindowsTerminal.exe",
        "powershell.exe",
        "pwsh.exe",
        "cmd.exe",
    ]
    assert by_name["Terminal"].match_title == []
    for rule in profiles.BUILTIN_RULES:
        assert rule.profile.name == rule.name
        assert rule.profile is profiles.BUILTIN_PROFILES[rule.name]


def test_builtin_profiles_have_the_spec_shapes():
    chat = profiles.BUILTIN_PROFILES["Chat"]
    assert chat.cleanup is True
    assert chat.delivery is DeliveryMethod.PASTE
    assert chat.drop_trailing_period_single_sentence is True
    assert chat.drop_trailing_punctuation is False
    assert chat.tone == DEFAULT_TONES["Chat"]

    email = profiles.BUILTIN_PROFILES["Email and docs"]
    assert email.cleanup is True
    assert email.delivery is DeliveryMethod.PASTE
    assert email.drop_trailing_period_single_sentence is False
    assert email.drop_trailing_punctuation is False

    code = profiles.BUILTIN_PROFILES["Code"]
    assert code.cleanup is True
    assert code.delivery is DeliveryMethod.PASTE
    assert "identifiers" in code.tone and "paths" in code.tone and "symbols" in code.tone

    terminal = profiles.BUILTIN_PROFILES["Terminal"]
    assert terminal.cleanup is False
    assert terminal.delivery is DeliveryMethod.TYPE
    assert terminal.drop_trailing_punctuation is True
    assert terminal.drop_trailing_period_single_sentence is False

    default = profiles.BUILTIN_PROFILES["Default"]
    assert default is profiles.DEFAULT_PROFILE
    assert default.cleanup is True
    assert default.delivery is DeliveryMethod.PASTE
    assert default.drop_trailing_period_single_sentence is False
    assert default.drop_trailing_punctuation is False

    expected_order = ["Chat", "Email and docs", "Code", "Terminal", "Default"]
    assert list(profiles.BUILTIN_PROFILES) == expected_order


@pytest.mark.parametrize(
    ("process", "expected"),
    [
        ("ms-teams.exe", "Chat"),
        ("slack.exe", "Chat"),
        ("WhatsApp.exe", "Chat"),
        ("OUTLOOK.EXE", "Email and docs"),
        ("olk.exe", "Email and docs"),
        ("WINWORD.EXE", "Email and docs"),
        ("Code.exe", "Code"),
        ("idea64.exe", "Code"),
        ("devenv.exe", "Code"),
        ("WindowsTerminal.exe", "Terminal"),
        ("powershell.exe", "Terminal"),
        ("pwsh.exe", "Terminal"),
        ("cmd.exe", "Terminal"),
        ("notepad.exe", "Default"),
        ("", "Default"),
    ],
)
def test_builtin_process_matches(process, expected):
    assert _match(_ctx(process)).name == expected


@pytest.mark.parametrize(
    ("process", "expected"),
    [
        ("CODE.EXE", "Code"),
        ("code.exe", "Code"),
        ("outlook.exe", "Email and docs"),
        ("Outlook.exe", "Email and docs"),
        ("whatsapp.exe", "Chat"),
        ("WINDOWSTERMINAL.EXE", "Terminal"),
    ],
)
def test_process_match_is_case_insensitive(process, expected):
    assert _match(_ctx(process)).name == expected


def test_process_may_be_a_full_path():
    assert _match(_ctx(r"C:\Program Files\Slack\slack.exe")).name == "Chat"
    forward_slashes = "C:/Users/x/AppData/Local/Programs/Microsoft VS Code/Code.exe"
    assert _match(_ctx(forward_slashes)).name == "Code"


# --- browser title rules ---


@pytest.mark.parametrize("browser", BROWSERS)
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("general - Contoso - Slack", "Chat"),
        ("WhatsApp", "Chat"),
        ("(3) WhatsApp - Google Chrome", "Chat"),
        ("Inbox (12) - alex@example.com - Gmail", "Email and docs"),
        ("Mail - Alex Morgan - Outlook", "Email and docs"),
        ("GitHub - spells", "Default"),
        ("", "Default"),
    ],
)
def test_browser_title_rules(browser, title, expected):
    assert _match(_ctx(browser, title)).name == expected


def test_builtin_title_match_is_case_sensitive():
    # Product names are capitalized; "outlook" as a plain word must not select Email and docs.
    assert _match(_ctx("chrome.exe", "Economic outlook 2026")).name == "Default"
    assert _match(_ctx("chrome.exe", "my slack workspace")).name == "Default"
    assert _match(_ctx("firefox.exe", "GMAIL")).name == "Default"
    assert _match(_ctx("ApplicationFrameHost.exe", "whatsapp")).name == "Default"
    assert _match(_ctx("msedge.exe", "Mail - Alex Morgan - Outlook")).name == "Email and docs"


def test_browser_title_rules_follow_table_order():
    # Chat is listed before Email and docs, so a title matching both goes to Chat.
    assert _match(_ctx("msedge.exe", "Slack message about Gmail")).name == "Chat"


def test_title_rules_do_not_apply_to_other_processes():
    assert _match(_ctx("notepad.exe", "Slack notes.txt - Notepad")).name == "Default"
    assert _match(_ctx("explorer.exe", "Gmail attachments")).name == "Default"


def test_process_match_wins_over_title_for_builtins():
    assert _match(_ctx("Code.exe", "gmail_client.py - spells - Visual Studio Code")).name == "Code"
    assert _match(_ctx("WindowsTerminal.exe", "Slack CLI")).name == "Terminal"


def test_application_frame_host_whatsapp():
    assert _match(_ctx("ApplicationFrameHost.exe", "WhatsApp")).name == "Chat"
    assert _match(_ctx("applicationframehost.exe", "WhatsApp Business")).name == "Chat"
    assert _match(_ctx("ApplicationFrameHost.exe", "Calculator")).name == "Default"


def test_application_frame_host_uses_every_title_rule():
    # V2-11: title matching applies to Store apps in general, not only WhatsApp.
    assert _match(_ctx("ApplicationFrameHost.exe", "Mail - Outlook")).name == "Email and docs"


# --- user rules ---


def test_user_rule_is_checked_before_builtins():
    rule = ProfileRule(
        name="Slack typed",
        match_process=["slack.exe"],
        match_title=[],
        profile=CUSTOM,
    )

    result = _match(_ctx("slack.exe", "general - Slack"), _settings(rule))

    assert result == CUSTOM
    assert result.delivery is DeliveryMethod.TYPE


def test_user_rule_process_match_is_case_insensitive():
    rule = ProfileRule(name="r", match_process=["SLACK.EXE"], match_title=[], profile=CUSTOM)

    assert _match(_ctx("slack.exe"), _settings(rule)) == CUSTOM


def test_user_rule_with_both_matchers_requires_both():
    rule = ProfileRule(
        name="Jira",
        match_process=["chrome.exe"],
        match_title=["Jira"],
        profile=CUSTOM,
    )
    settings = _settings(rule)

    assert _match(_ctx("chrome.exe", "BT-1 - Jira"), settings) == CUSTOM
    assert _match(_ctx("chrome.exe", "GitHub"), settings).name == "Default"
    assert _match(_ctx("notepad.exe", "Jira notes"), settings).name == "Default"
    assert _match(_ctx("chrome.exe", "Slack"), settings).name == "Chat"


def test_user_rule_with_only_process_matches_any_title():
    rule = ProfileRule(name="r", match_process=["notepad.exe"], match_title=[], profile=CUSTOM)

    assert _match(_ctx("notepad.exe", "anything - Notepad"), _settings(rule)) == CUSTOM
    assert _match(_ctx("wordpad.exe", "anything"), _settings(rule)).name == "Default"


def test_user_rule_with_only_title_matches_any_process():
    rule = ProfileRule(name="r", match_process=[], match_title=["Jira"], profile=CUSTOM)

    assert _match(_ctx("notepad.exe", "BT-1 - Jira"), _settings(rule)) == CUSTOM
    assert _match(_ctx("chrome.exe", "BT-1 - Jira"), _settings(rule)) == CUSTOM
    assert _match(_ctx("chrome.exe", "Slack"), _settings(rule)).name == "Chat"


def test_user_rule_title_match_is_case_insensitive_substring():
    rule = ProfileRule(name="r", match_process=[], match_title=["jira"], profile=CUSTOM)

    assert _match(_ctx("chrome.exe", "BT-1 - JIRA"), _settings(rule)) == CUSTOM


def test_user_rule_title_match_ignores_case_unlike_builtins():
    # The user chose "outlook" deliberately, so it matches the plain word too.
    rule = ProfileRule(name="r", match_process=[], match_title=["outlook"], profile=CUSTOM)

    assert _match(_ctx("chrome.exe", "Economic outlook 2026"), _settings(rule)) == CUSTOM
    assert _match(_ctx("chrome.exe", "Economic outlook 2026")).name == "Default"


def test_user_rule_any_listed_process_or_title_counts():
    rule = ProfileRule(
        name="r",
        match_process=["a.exe", "b.exe"],
        match_title=["One", "Two"],
        profile=CUSTOM,
    )
    settings = _settings(rule)

    assert _match(_ctx("b.exe", "Two"), settings) == CUSTOM
    assert _match(_ctx("a.exe", "One"), settings) == CUSTOM
    assert _match(_ctx("a.exe", "Three"), settings).name == "Default"


def test_first_matching_user_rule_wins():
    first = ProfileRule(
        name="first",
        match_process=["x.exe"],
        match_title=[],
        profile=dataclasses.replace(CUSTOM, name="First"),
    )
    second = ProfileRule(
        name="second",
        match_process=["x.exe"],
        match_title=[],
        profile=dataclasses.replace(CUSTOM, name="Second"),
    )

    assert _match(_ctx("x.exe"), _settings(first, second)).name == "First"
    assert _match(_ctx("x.exe"), _settings(second, first)).name == "Second"


def test_user_rule_with_custom_profile_name_keeps_its_own_tone():
    custom = dataclasses.replace(CUSTOM, name="Mine", tone="my tone")
    rule = ProfileRule(name="r", match_process=["x.exe"], match_title=[], profile=custom)

    assert _match(_ctx("x.exe"), _settings(rule)) == custom


def test_user_rule_without_matchers_never_matches():
    rule = ProfileRule(name="empty", match_process=[], match_title=[], profile=CUSTOM)

    assert _match(_ctx("notepad.exe", "anything"), _settings(rule)).name == "Default"


def test_user_rule_ignores_blank_title_substrings():
    rule = ProfileRule(name="blank", match_process=[], match_title=["", "  "], profile=CUSTOM)

    assert _match(_ctx("notepad.exe", "anything"), _settings(rule)).name == "Default"


def test_default_fallback_when_nothing_matches():
    assert _match(_ctx("unknown.exe", "Unknown")) is profiles.DEFAULT_PROFILE


# --- tone override from the Cleanup tab ---


def test_tone_comes_from_settings_when_edited():
    tones = {**DEFAULT_TONES, "Chat": "be brief"}

    result = _match(_ctx("slack.exe"), _settings(tones=tones))

    assert result.name == "Chat"
    assert result.tone == "be brief"
    assert result.drop_trailing_period_single_sentence is True


def test_tone_override_applies_to_default_profile():
    tones = {**DEFAULT_TONES, "Default": "plain"}

    assert _match(_ctx("unknown.exe"), _settings(tones=tones)).tone == "plain"


def test_tone_override_applies_to_user_rules_by_profile_name():
    rule = ProfileRule(name="r", match_process=["x.exe"], match_title=[], profile=CUSTOM)
    tones = {**DEFAULT_TONES, "Email and docs": "very formal"}

    assert _match(_ctx("x.exe"), _settings(rule, tones=tones)).tone == "very formal"


def test_unedited_tones_return_the_builtin_profile_object():
    assert _match(_ctx("slack.exe")) is profiles.BUILTIN_PROFILES["Chat"]
