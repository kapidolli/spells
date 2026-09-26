import sys

import pytest


def pytest_collection_modifyitems(config, items):
    if sys.platform == "win32":
        return
    skip = pytest.mark.skip(reason="exercises Windows itself")
    for item in items:
        if item.get_closest_marker("windows") is not None:
            item.add_marker(skip)
