"""textutil (word count per decision V2-15, normalized phrase matching) and datafiles."""

import sys

from spells import datafiles
from spells.textutil import contains_phrase, normalize_for_match, words


def test_words_splits_on_whitespace_and_drops_symbol_only_tokens():
    assert words("Hello,  world!\n- yes ... 42 --") == ["Hello,", "world!", "yes", "42"]


def test_words_empty_inputs():
    assert words("") == []
    assert words("  \n\t ") == []
    assert words("... - !!") == []


def test_normalize_lowercases_strips_punctuation_and_collapses_whitespace():
    assert normalize_for_match("  Thanks   for\twatching!!  ") == "thanks for watching"
    assert normalize_for_match("Untertitel im Auftrag des ZDF, 2017.") == (
        "untertitel im auftrag des zdf 2017"
    )


def test_normalize_keeps_apostrophes_inside_words_only():
    assert normalize_for_match("Don’t say 'hello' s'ka") == "don't say hello s'ka"


def test_contains_phrase_whole_word_case_insensitive():
    assert contains_phrase("I LIKE it, um, a lot", "um")
    assert contains_phrase("I LIKE it a lot", "like")
    assert not contains_phrase("He likes it", "like")
    assert not contains_phrase("It is unlike him", "like")


def test_contains_phrase_multiword():
    assert contains_phrase("So, you know, we went", "you know")
    assert contains_phrase("Sorry, I mean tomorrow", "sorry I mean")
    assert not contains_phrase("You should know this", "you know")


def test_contains_phrase_empty_phrase_never_matches():
    assert not contains_phrase("anything", "")
    assert not contains_phrase("anything", "...")


def test_data_dir_is_next_to_the_executable_when_frozen(monkeypatch, tmp_path):
    exe = tmp_path / "Spells" / "Spells.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert datafiles.data_dir() == exe.parent / "data"


def test_read_lines_strips_and_skips_blanks_and_comments(monkeypatch, tmp_path):
    (tmp_path / "list.txt").write_text(
        "﻿# header comment\n\n  first  \n#another\nsecond\n   \n", encoding="utf-8"
    )
    monkeypatch.setattr(datafiles, "data_dir", lambda: tmp_path)
    assert datafiles.read_lines("list.txt") == ["first", "second"]
