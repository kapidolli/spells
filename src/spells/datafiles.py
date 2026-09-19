"""Locate and read the bundled data files (spec 5.3, data/ folder).

In development the folder is the repository's data/. In a frozen PyInstaller build the build
copies it next to the executable, so it is resolved relative to sys.executable there.
"""

from __future__ import annotations

import sys
from pathlib import Path


def data_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "data"
    return Path(__file__).resolve().parents[2] / "data"


def data_path(*parts: str) -> Path:
    return data_dir().joinpath(*parts)


def read_lines(name: str) -> list[str]:
    """Non-blank, non-comment lines of data/<name>, stripped. Lines starting with # are comments.

    name is relative to the data folder and may contain forward slashes ("stopwords/en.txt").
    A UTF-8 byte order mark is tolerated because the lists are edited on Windows.
    """
    lines = []
    for raw in data_path(name).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines
