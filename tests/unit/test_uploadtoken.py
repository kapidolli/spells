from __future__ import annotations

import pytest

from spells.uploadtoken import TOKEN_FILE, TokenStore, token_path_for
from spells.win32 import dpapi

TOKEN = "3f9a1c0d8e7b6a5f4e3d2c1b0a9f8e7d-ünïcode"


def test_dpapi_round_trips_and_hides_the_plaintext():
    blob = dpapi.protect(TOKEN.encode("utf-8"), "test")
    assert TOKEN.encode("utf-8") not in blob
    assert dpapi.unprotect(blob) == TOKEN.encode("utf-8")
    assert dpapi.protect(TOKEN.encode("utf-8")) != TOKEN.encode("utf-8")


def test_dpapi_refuses_a_blob_it_did_not_make():
    with pytest.raises(dpapi.DpapiError):
        dpapi.unprotect(b"not a protected blob at all")


def test_the_token_file_sits_beside_the_history(tmp_path):
    assert token_path_for(tmp_path / "history.db") == tmp_path / TOKEN_FILE
    assert TOKEN_FILE == "upload-token.bin"


def test_the_store_writes_an_encrypted_file_and_reads_it_back(tmp_path):
    path = tmp_path / "Spells" / TOKEN_FILE
    store = TokenStore(path)
    assert store.read() == "" and not store.exists()
    store.write(f"  {TOKEN}\n")
    assert store.exists()
    assert TOKEN.encode("utf-8") not in path.read_bytes()
    assert TokenStore(path).read() == TOKEN
    assert not (tmp_path / "Spells" / (TOKEN_FILE + ".tmp")).exists()


def test_an_empty_token_removes_the_file(tmp_path):
    store = TokenStore(tmp_path / TOKEN_FILE)
    store.write(TOKEN)
    store.write("   ")
    assert not store.exists()
    assert store.read() == ""
    store.delete()


def test_a_file_that_cannot_be_decrypted_reads_as_no_token(tmp_path, caplog):
    path = tmp_path / TOKEN_FILE
    path.write_bytes(b"garbage")
    assert TokenStore(path).read() == ""
    assert TOKEN not in caplog.text


def test_the_store_never_logs_the_token(tmp_path, caplog):
    caplog.set_level("DEBUG")
    store = TokenStore(tmp_path / TOKEN_FILE)
    store.write(TOKEN)
    store.read()
    store.delete()
    assert TOKEN not in caplog.text


def test_without_a_path_the_token_lives_in_memory_only(tmp_path):
    store = TokenStore(None)
    store.write(TOKEN)
    assert store.read() == TOKEN and store.exists()
    store.write("")
    assert store.read() == "" and not store.exists()
    assert list(tmp_path.iterdir()) == []


def test_the_encryption_is_injectable(tmp_path):
    store = TokenStore(
        tmp_path / TOKEN_FILE,
        protect=lambda data: data[::-1],
        unprotect=lambda blob: blob[::-1],
    )
    store.write("abc")
    assert (tmp_path / TOKEN_FILE).read_bytes() == b"cba"
    assert store.read() == "abc"
