"""Match the target window to an app profile (spec 9, decision V2-11).

Order: the user's rules (Apps tab) first, then the built-in table, first match
wins, Default when nothing matches. Process names compare case-insensitively.
Built-in title substrings are product names (Slack, Gmail) and match
case-sensitively, only when the process is a browser or ApplicationFrameHost.exe
(Store apps). User rules may match titles for any process, case-insensitively,
and a user rule with both matchers set requires both.
"""

from __future__ import annotations

import dataclasses
from pathlib import PureWindowsPath

from spells.config import DEFAULT_TONES, ProfileRule, Settings
from spells.models import DeliveryMethod, Profile, TargetContext

# Processes whose window title is consulted for built-in rules (spec 9, V2-11).
TITLE_MATCH_PROCESSES = frozenset(
    {"chrome.exe", "msedge.exe", "firefox.exe", "applicationframehost.exe"}
)

CHAT_PROFILE = Profile(
    name="Chat",
    cleanup=True,
    tone=DEFAULT_TONES["Chat"],
    delivery=DeliveryMethod.PASTE,
    drop_trailing_period_single_sentence=True,
)
EMAIL_PROFILE = Profile(
    name="Email and docs",
    cleanup=True,
    tone=DEFAULT_TONES["Email and docs"],
    delivery=DeliveryMethod.PASTE,
)
CODE_PROFILE = Profile(
    name="Code",
    cleanup=True,
    tone=DEFAULT_TONES["Code"],
    delivery=DeliveryMethod.PASTE,
)
TERMINAL_PROFILE = Profile(
    name="Terminal",
    cleanup=False,
    tone=DEFAULT_TONES["Terminal"],
    delivery=DeliveryMethod.TYPE,
    drop_trailing_punctuation=True,
)
DEFAULT_PROFILE = Profile(
    name="Default",
    cleanup=True,
    tone=DEFAULT_TONES["Default"],
    delivery=DeliveryMethod.PASTE,
)

# Every built-in profile by name, in spec 9 table order (for the Apps tab picker).
BUILTIN_PROFILES: dict[str, Profile] = {
    profile.name: profile
    for profile in (CHAT_PROFILE, EMAIL_PROFILE, CODE_PROFILE, TERMINAL_PROFILE, DEFAULT_PROFILE)
}

# The spec 9 table. Default is the fallback, not a rule.
BUILTIN_RULES: list[ProfileRule] = [
    ProfileRule(
        name="Chat",
        match_process=["ms-teams.exe", "slack.exe", "WhatsApp.exe"],
        match_title=["Slack", "WhatsApp"],
        profile=CHAT_PROFILE,
    ),
    ProfileRule(
        name="Email and docs",
        match_process=["OUTLOOK.EXE", "olk.exe", "WINWORD.EXE"],
        match_title=["Gmail", "Outlook"],
        profile=EMAIL_PROFILE,
    ),
    ProfileRule(
        name="Code",
        match_process=["Code.exe", "idea64.exe", "devenv.exe"],
        match_title=[],
        profile=CODE_PROFILE,
    ),
    ProfileRule(
        name="Terminal",
        match_process=["WindowsTerminal.exe", "powershell.exe", "pwsh.exe", "cmd.exe"],
        match_title=[],
        profile=TERMINAL_PROFILE,
    ),
]


def _process_name(process: str) -> str:
    """The casefolded executable name; tolerates a full path in ctx.process."""
    return PureWindowsPath(process).name.casefold()


def _process_matches(rule: ProfileRule, process: str) -> bool:
    return any(candidate.casefold() == process for candidate in rule.match_process)


def _title_matches(rule: ProfileRule, title: str, *, case_sensitive: bool) -> bool:
    haystack = title if case_sensitive else title.casefold()
    for candidate in rule.match_title:
        if not candidate.strip():
            continue
        needle = candidate if case_sensitive else candidate.casefold()
        if needle in haystack:
            return True
    return False


def _user_rule_matches(rule: ProfileRule, process: str, title: str) -> bool:
    has_process = bool(rule.match_process)
    has_title = any(candidate.strip() for candidate in rule.match_title)
    if not has_process and not has_title:
        return False
    if has_process and not _process_matches(rule, process):
        return False
    if has_title:
        return _title_matches(rule, title, case_sensitive=False)
    return True


def _builtin_rule_matches(rule: ProfileRule, process: str, title: str) -> bool:
    if _process_matches(rule, process):
        return True
    return process in TITLE_MATCH_PROCESSES and _title_matches(rule, title, case_sensitive=True)


def _with_tone(profile: Profile, settings: Settings) -> Profile:
    """The profile with the Cleanup tab's tone for its name, when the user edited it."""
    tone = settings.cleanup.tones.get(profile.name)
    if tone is None or tone == profile.tone:
        return profile
    return dataclasses.replace(profile, tone=tone)


def match(ctx: TargetContext, settings: Settings) -> Profile:
    """The profile for the target captured at press time (spec 6 step 1)."""
    process = _process_name(ctx.process)
    title = ctx.title
    for rule in settings.profiles:
        if _user_rule_matches(rule, process, title):
            return _with_tone(rule.profile, settings)
    for rule in BUILTIN_RULES:
        if _builtin_rule_matches(rule, process, title):
            return _with_tone(rule.profile, settings)
    return _with_tone(DEFAULT_PROFILE, settings)
