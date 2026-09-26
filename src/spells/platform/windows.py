from __future__ import annotations

from dataclasses import replace

from spells.platform import stub
from spells.platform.base import Capabilities, Platform


def build() -> Platform:
    return replace(stub.build("windows"), capabilities=Capabilities.everything())
