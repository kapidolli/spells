"""Tests for spells.vk: shared virtual-key constants and the generic-modifier fold."""

from __future__ import annotations

import pytest

from spells.vk import (
    GENERIC_OF,
    SPECIFIC_OF,
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
    generic_modifier,
)


def test_constants_match_the_win32_vk_table():
    assert VK_SHIFT == 0x10
    assert VK_CONTROL == 0x11
    assert VK_MENU == 0x12
    assert VK_LWIN == 0x5B
    assert VK_RWIN == 0x5C
    assert VK_LSHIFT == 0xA0
    assert VK_RSHIFT == 0xA1
    assert VK_LCONTROL == 0xA2
    assert VK_RCONTROL == 0xA3
    assert VK_LMENU == 0xA4
    assert VK_RMENU == 0xA5


@pytest.mark.parametrize(
    ("side_specific", "generic"),
    [
        (VK_LSHIFT, VK_SHIFT),
        (VK_RSHIFT, VK_SHIFT),
        (VK_LCONTROL, VK_CONTROL),
        (VK_RCONTROL, VK_CONTROL),
        (VK_LMENU, VK_MENU),
        (VK_RMENU, VK_MENU),
    ],
)
def test_generic_modifier_folds_left_and_right_variants(side_specific, generic):
    assert generic_modifier(side_specific) == generic
    assert GENERIC_OF[side_specific] == generic


def test_generic_modifier_leaves_win_keys_distinct():
    assert generic_modifier(VK_LWIN) == VK_LWIN
    assert generic_modifier(VK_RWIN) == VK_RWIN
    assert VK_LWIN not in GENERIC_OF
    assert VK_RWIN not in GENERIC_OF


def test_generic_modifier_leaves_already_generic_codes_unchanged():
    assert generic_modifier(VK_SHIFT) == VK_SHIFT
    assert generic_modifier(VK_CONTROL) == VK_CONTROL
    assert generic_modifier(VK_MENU) == VK_MENU


def test_generic_modifier_leaves_unknown_codes_unchanged():
    assert generic_modifier(0x44) == 0x44
    assert generic_modifier(0) == 0


def test_specific_of_maps_each_generic_code_to_its_left_and_right_pair():
    assert SPECIFIC_OF == {
        VK_SHIFT: (VK_LSHIFT, VK_RSHIFT),
        VK_CONTROL: (VK_LCONTROL, VK_RCONTROL),
        VK_MENU: (VK_LMENU, VK_RMENU),
    }


def test_specific_of_is_the_inverse_of_generic_of():
    for generic, pair in SPECIFIC_OF.items():
        for side_specific in pair:
            assert GENERIC_OF[side_specific] == generic
    assert set(GENERIC_OF) == {vk for pair in SPECIFIC_OF.values() for vk in pair}


def test_specific_of_leaves_win_keys_out():
    assert VK_LWIN not in SPECIFIC_OF
    assert VK_RWIN not in SPECIFIC_OF
