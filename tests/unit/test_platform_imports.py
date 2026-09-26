import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "spells"
OS_PACKAGES = ("win32", "linux", "macos", "platform")
FORBIDDEN = {"ctypes", "winreg", "msvcrt", "_winapi", "spells.win32"}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                package = path.relative_to(SRC.parent).parent.parts
                anchor = package[: len(package) - (node.level - 1)]
                base = ".".join((*anchor, base)) if base else ".".join(anchor)
            yield base
            for alias in node.names:
                yield f"{base}.{alias.name}"


def _forbidden(name: str) -> bool:
    return any(name == item or name.startswith(f"{item}.") for item in FORBIDDEN)


def _shared_modules():
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC)
        if relative.parts[0] in OS_PACKAGES:
            continue
        yield path


@pytest.mark.parametrize("path", list(_shared_modules()), ids=lambda p: str(p.relative_to(SRC)))
def test_shared_code_does_not_touch_the_operating_system(path):
    bad = sorted({name for name in _imports(path) if _forbidden(name)})
    assert bad == []


def test_only_the_windows_assembly_imports_win32():
    for path in sorted((SRC / "platform").rglob("*.py")):
        if path.name == "windows.py":
            continue
        bad = sorted({name for name in _imports(path) if name.startswith("spells.win32")})
        assert bad == [], path.name
