"""Synthetic keyboard input: modifier release, Ctrl+V and Unicode typing.

Spec section 11.2 step 3 and 11.3; batch 2 decisions V2-3 and V2-12. Releasing a held Win
key also injects the Start menu mask of spec 6. The pure parts (split_batches, plan_batch)
carry the batching and newline rules so they can be unit tested without touching Win32.
"""

from dataclasses import dataclass

from spells.vk import (
    VK_CONTROL,
    VK_LCONTROL,
    VK_LMENU,
    VK_LSHIFT,
    VK_LWIN,
    VK_MENU,
    VK_RCONTROL,
    VK_RMENU,
    VK_RSHIFT,
    VK_RWIN,
    VK_SHIFT,
)

from .hook import INPUT, VK_PROBE, is_key_down, keyboard_input, send_inputs, unicode_input

VK_RETURN = 0x0D
VK_V = 0x56
VK_C = 0x43
VK_BACK = 0x08

# Generic and side-specific modifier keys, in the order they are checked and released.
MODIFIER_VKS = (
    VK_SHIFT,
    VK_CONTROL,
    VK_MENU,
    VK_LWIN,
    VK_RWIN,
    VK_LSHIFT,
    VK_RSHIFT,
    VK_LCONTROL,
    VK_RCONTROL,
    VK_LMENU,
    VK_RMENU,
)
WIN_VKS = frozenset({VK_LWIN, VK_RWIN})

# The Start menu mask: the unassigned key VK 0xE8 carrying the hotkey layer's private
# dwExtraInfo tag. Both values must stay equal to spells.hotkey.MASK_VK and
# spells.hotkey.MASK_TAG, which a unit test asserts; they are repeated rather than imported
# because spells.win32 is the bottom layer and never imports the layers above it.
MASK_VK = VK_PROBE
MASK_TAG = 0x5554544D

DEFAULT_BATCH_SIZE = 50


@dataclass(frozen=True)
class TypedKey:
    """One planned key event: either a UTF-16 unit (vk 0) or a virtual key such as Enter."""

    keydown: bool
    unit: int = 0
    vk: int = 0

    @property
    def is_enter(self) -> bool:
        return self.vk == VK_RETURN


def held_modifiers() -> list[int]:
    """The modifier vks a real keyboard is holding down right now."""
    return [vk for vk in MODIFIER_VKS if is_key_down(vk)]


def modifier_release_records(held: list[int], extra_info: int = 0) -> list[INPUT]:
    """The records that release `held`, without sending them.

    Built separately from release_held_modifiers so a caller that types while the chord
    is still down can put them at the front of the same SendInput call as its characters:
    SendInput inserts a whole array atomically, so no physical auto-repeat of the held
    modifier can land between the release and the text. `extra_info` carries the private
    tag for such a caller, which keeps the hotkey hook from reading our key-up as the
    user letting the chord go (spec 6, the hook ignores tagged events).
    """
    records: list[INPUT] = []
    ordered = [vk for vk in held if vk in WIN_VKS] + [vk for vk in held if vk not in WIN_VKS]
    for vk in ordered:
        if vk in WIN_VKS:
            records.append(keyboard_input(MASK_VK, True, MASK_TAG))
            records.append(keyboard_input(MASK_VK, False, MASK_TAG))
        records.append(keyboard_input(vk, False, extra_info))
    return records


def release_held_modifiers() -> list[int]:
    """Send a synthetic key-up for every modifier currently down; returns the vks released.

    A held Win key gets the Start menu mask first: a press and release of VK 0xE8 carrying
    the private tag, exactly as the hook injects it on a Win key-up (spec 6 and 13), so
    Windows does not read our synthetic release as a lone Win tap and open the Start menu.
    Everything goes in one SendInput call, so no real key can land between mask and release.
    """
    held = held_modifiers()
    records = modifier_release_records(held)
    if records:
        send_inputs(records)
    return held


def send_ctrl_v() -> None:
    """Send Ctrl down, V down, V up, Ctrl up in one SendInput call."""
    send_inputs(
        [
            keyboard_input(VK_CONTROL, True),
            keyboard_input(VK_V, True),
            keyboard_input(VK_V, False),
            keyboard_input(VK_CONTROL, False),
        ]
    )


def send_ctrl_c() -> None:
    """Send Ctrl down, C down, C up, Ctrl up in one SendInput call.

    Edit mode copies the target window's selection with it (spec 8.5) before it asks the
    writing model anything; the clipboard is snapshotted first and put back afterwards
    under the rules of spec 11.
    """
    send_inputs(
        [
            keyboard_input(VK_CONTROL, True),
            keyboard_input(VK_C, True),
            keyboard_input(VK_C, False),
            keyboard_input(VK_CONTROL, False),
        ]
    )


def split_batches(text: str, batch_size: int = DEFAULT_BATCH_SIZE) -> list[str]:
    """Split text into batches of at most batch_size code points.

    "\\r\\n" (and a lone "\\r") is normalized to "\\n" first, so each line break is one
    code point and later becomes one Enter. Python strings index by code point, so a
    non-BMP character never straddles two batches.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [normalized[i : i + batch_size] for i in range(0, len(normalized), batch_size)]


def plan_batch(batch: str) -> list[TypedKey]:
    """Expand one batch into key events.

    "\\n" becomes VK_RETURN down and up. Every other character becomes KEYEVENTF_UNICODE
    down and up per UTF-16 code unit, so a non-BMP character yields its high surrogate
    down/up followed by its low surrogate down/up.
    """
    keys: list[TypedKey] = []
    for char in batch:
        if char == "\n":
            keys.append(TypedKey(True, vk=VK_RETURN))
            keys.append(TypedKey(False, vk=VK_RETURN))
            continue
        encoded = char.encode("utf-16-le")
        for offset in range(0, len(encoded), 2):
            unit = int.from_bytes(encoded[offset : offset + 2], "little")
            keys.append(TypedKey(True, unit=unit))
            keys.append(TypedKey(False, unit=unit))
    return keys


def _records_for(keys: list[TypedKey], extra_info: int = 0) -> list[INPUT]:
    records: list[INPUT] = []
    for key in keys:
        if key.vk:
            records.append(keyboard_input(key.vk, key.keydown, extra_info))
        else:
            records.append(unicode_input(key.unit, key.keydown, extra_info))
    return records


def plan_backspaces(count: int) -> list[TypedKey]:
    """`count` Backspace taps as key events."""
    keys: list[TypedKey] = []
    for _ in range(max(0, int(count))):
        keys.append(TypedKey(True, vk=VK_BACK))
        keys.append(TypedKey(False, vk=VK_BACK))
    return keys


def type_unicode(
    text: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    extra_info: int = 0,
    release_modifiers: bool = False,
) -> None:
    """Type text into the focused window with KEYEVENTF_UNICODE, one SendInput per batch.

    With release_modifiers the key-ups of every modifier a real keyboard is holding lead
    each batch inside the same SendInput call, which is what live insertion needs while
    the dictation chord is still held down.
    """
    for batch in split_batches(text, batch_size):
        _send_typed(plan_batch(batch), extra_info, release_modifiers)


def send_backspaces(
    count: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    extra_info: int = 0,
    release_modifiers: bool = False,
) -> None:
    """Send `count` Backspace taps, batched the way type_unicode batches characters."""
    total = max(0, int(count))
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    sent = 0
    while sent < total:
        chunk = min(batch_size, total - sent)
        _send_typed(plan_backspaces(chunk), extra_info, release_modifiers)
        sent += chunk


def _send_typed(keys: list[TypedKey], extra_info: int, release_modifiers: bool) -> None:
    records: list[INPUT] = []
    if release_modifiers:
        records += modifier_release_records(held_modifiers(), extra_info)
    records += _records_for(keys, extra_info)
    send_inputs(records)
