"""Fake engine clients for the pipeline tests: a WhisperClient and a LlamaClient with
scripted responses and a call log, plus a factory that records the URLs the pipeline
builds them from (spec 5.2: the supervisor exposes URLs, the pipeline builds clients).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from spells.asr import AsrReply, LlamaAsrClient, WhisperClient
from spells.cleanup import LlamaClient

DEFAULT_TEXT = "hello world"


class FakeWhisperClient(WhisperClient):
    """Answers inference() from `responses` in order; the last entry repeats.

    An exception instance in the list is raised instead of returned. health() answers
    `healthy`. `on_call` runs before every inference (tests advance a fake clock there).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1",
        timeout_s: float = 10.0,
        *,
        responses: list[Any] | None = None,
        healthy: bool = True,
    ) -> None:
        super().__init__(base_url, timeout_s)
        self.responses: list[Any] = (
            list(responses)
            if responses is not None
            else [{"text": DEFAULT_TEXT, "language": "english"}]
        )
        self.healthy = healthy
        self.calls: list[dict] = []
        self.health_calls = 0
        self.on_call: Callable[[], None] | None = None

    def inference(self, wav_bytes: bytes, **kwargs: Any) -> dict:
        assert wav_bytes[:4] == b"RIFF"
        if self.on_call is not None:
            self.on_call()
        self.calls.append(
            {
                "wav": wav_bytes,
                "language": kwargs.get("language"),
                "prompt": kwargs.get("prompt"),
                "detect_only": kwargs.get("detect_only", False),
                "want_probabilities": kwargs.get("want_probabilities", False),
                "timeout_s": kwargs.get("timeout_s"),
            }
        )
        if not self.responses:
            raise AssertionError("FakeWhisperClient has no response left")
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, BaseException):
            raise response
        return dict(response)

    def health(self) -> bool:
        self.health_calls += 1
        return self.healthy


class FakeLlamaAsrClient(LlamaAsrClient):
    def __init__(
        self,
        *replies: Any,
        reports: str | None = None,
        base_url: str = "http://127.0.0.1:3",
        timeout_s: float = 10.0,
        healthy: bool = True,
    ) -> None:
        super().__init__(base_url, timeout_s)
        self.replies: list[Any] = list(replies) or [AsrReply(DEFAULT_TEXT, "en", "English")]
        self.reports = reports
        self.healthy = healthy
        self.calls: list[dict] = []
        self.asked: list[str] = []
        self.stopped = 0
        self.health_calls = 0
        self.on_call: Callable[[], None] | None = None

    def transcribe(self, wav_bytes: bytes, **kwargs: Any) -> AsrReply:
        assert wav_bytes[:4] == b"RIFF"
        if self.on_call is not None:
            self.on_call()
        accept = kwargs.get("accept")
        self.calls.append(
            {
                "language": kwargs.get("language"),
                "accept": accept,
                "vocabulary": kwargs.get("vocabulary", ""),
                "timeout_s": kwargs.get("timeout_s"),
            }
        )
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        if accept is not None:
            name = self.reports if self.reports is not None else reply.language_name
            self.asked.append(name)
            if not accept(name):
                self.stopped += 1
                return AsrReply("", reply.language, name, complete=False)
        return reply

    def health(self) -> bool:
        self.health_calls += 1
        return self.healthy


class FakeLlamaClient(LlamaClient):
    """Returns (content, finish) from chat(), or raises `error` when set."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:2",
        timeout_s: float = 2.5,
        *,
        content: str = "",
        finish: str = "stop",
        error: Exception | None = None,
        healthy: bool = True,
    ) -> None:
        super().__init__(base_url, timeout_s)
        self.content = content
        self.finish = finish
        self.error = error
        self.healthy = healthy
        self.calls: list[tuple[str, str, int]] = []
        self.on_call: Callable[[], None] | None = None

    def chat(self, system: str, user: str, max_tokens: int) -> tuple[str, str]:
        if self.on_call is not None:
            self.on_call()
        self.calls.append((system, user, max_tokens))
        if self.error is not None:
            raise self.error
        return self.content, self.finish

    def health(self) -> bool:
        return self.healthy


class ClientFactory:
    """A client factory that hands out one client and records every request for one.

    Records (base_url, timeout_s) per call and points the client at the requested URL,
    so a test can check which engine URL the pipeline used for each request.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.calls: list[tuple[str, float | None]] = []
        self.error: Exception | None = None

    def __call__(self, base_url: str, timeout_s: float | None = None) -> Any:
        self.calls.append((base_url, timeout_s))
        if self.error is not None:
            raise self.error
        self.client.base_url = base_url
        if timeout_s is not None:
            self.client.timeout_s = timeout_s
        return self.client
