"""
Tests for app.services.llm_service.

requests.post is monkeypatched -- these tests verify OUR error
handling and request shape, not Ollama itself (that's covered by the
manual end-to-end verification against a real running Ollama server).
"""

import pytest
import requests

from app.core.config import settings
from app.services import llm_service


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json_data


def test_ollama_provider_returns_response_text(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"response": "  The answer is 42.  "})

    monkeypatch.setattr(requests, "post", fake_post)

    provider = llm_service.OllamaProvider()
    result = provider.generate("What is the answer?")

    assert result == "The answer is 42."
    assert captured["url"] == f"{settings.ollama_base_url}/api/generate"
    assert captured["json"]["model"] == settings.ollama_model
    assert captured["json"]["prompt"] == "What is the answer?"
    assert captured["json"]["stream"] is False
    assert captured["timeout"] == settings.llm_request_timeout_seconds


def test_ollama_provider_raises_clear_error_on_connection_failure(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fake_post)

    provider = llm_service.OllamaProvider()
    with pytest.raises(RuntimeError, match="Is the Ollama service running"):
        provider.generate("anything")


def test_ollama_provider_raises_clear_error_on_timeout(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.exceptions.Timeout("too slow")

    monkeypatch.setattr(requests, "post", fake_post)

    provider = llm_service.OllamaProvider()
    with pytest.raises(RuntimeError, match="did not respond"):
        provider.generate("anything")


def test_ollama_provider_raises_on_http_error(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse({}, status_code=500)

    monkeypatch.setattr(requests, "post", fake_post)

    provider = llm_service.OllamaProvider()
    with pytest.raises(RuntimeError, match="Ollama returned an error"):
        provider.generate("anything")


def test_get_llm_provider_returns_ollama_by_default():
    provider = llm_service.get_llm_provider()
    assert isinstance(provider, llm_service.OllamaProvider)


def test_get_llm_provider_raises_on_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "some_unsupported_provider")
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        llm_service.get_llm_provider()
