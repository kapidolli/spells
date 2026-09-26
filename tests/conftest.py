import sys

import pytest


def pytest_collection_modifyitems(config, items):
    if sys.platform == "win32":
        return
    skip = pytest.mark.skip(reason="exercises Windows itself")
    for item in items:
        if "windows" in item.keywords:
            item.add_marker(skip)
