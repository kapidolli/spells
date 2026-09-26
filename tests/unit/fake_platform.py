from __future__ import annotations

from dataclasses import replace

import pytest

from spells import platform
from spells.platform import stub
from spells.platform.base import Platform


def fake_platform(**overrides) -> Platform:
    return replace(stub.build("linux"), **overrides)


@pytest.fixture
def use_platform():
    installed: list[Platform | None] = []

    def install(record: Platform) -> Platform:
        installed.append(platform.swap(record))
        return record

    yield install
    while installed:
        platform.swap(installed.pop())
