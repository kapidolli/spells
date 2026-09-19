from __future__ import annotations

from urllib.request import ProxyHandler, build_opener

urlopen = build_opener(ProxyHandler({})).open
