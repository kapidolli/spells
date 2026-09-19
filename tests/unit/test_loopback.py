from __future__ import annotations

import http.server
import threading
from urllib.request import Request

import pytest

from spells import asr, cleanup, compose, loopback


class _Direct(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"direct")

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Direct)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


def test_a_loopback_request_never_goes_through_a_configured_proxy(monkeypatch, local_server):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)

    with loopback.urlopen(Request(local_server), timeout=5) as response:
        assert response.read() == b"direct"


@pytest.mark.parametrize("module", [asr, cleanup, compose])
def test_every_engine_client_reaches_its_engine_without_a_proxy(module):
    assert module.urlopen is loopback.urlopen
