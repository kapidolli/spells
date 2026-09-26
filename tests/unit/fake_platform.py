from __future__ import annotations

from dataclasses import replace

import pytest

from spells import platform
from spells.platform import stub
from spells.platform.base import Platform


class ReversibleSecrets:
    MARK = b"fake-protected:"

    def __init__(self) -> None:
        self.descriptions: list[str] = []

    def protect(self, data: bytes, description: str = "") -> bytes:
        self.descriptions.append(description)
        return self.MARK + bytes(value ^ 0x5A for value in data)

    def unprotect(self, blob: bytes) -> bytes:
        if not blob.startswith(self.MARK):
            raise OSError("not a blob these secrets protected")
        return bytes(value ^ 0x5A for value in blob[len(self.MARK) :])


def fake_platform(**overrides) -> Platform:
    return replace(stub.build("linux"), **overrides)


def secrets_where_missing(install) -> None:
    record = platform.current()
    if isinstance(record.secrets, stub.StubSecrets):
        install(replace(record, secrets=ReversibleSecrets()))


@pytest.fixture
def use_platform():
    installed: list[Platform | None] = []

    def install(record: Platform) -> Platform:
        installed.append(platform.swap(record))
        return record

    yield install
    while installed:
        platform.swap(installed.pop())
