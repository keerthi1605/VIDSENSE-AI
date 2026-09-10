"""
Tests for POST /api/chat, with rag_service's dependencies mocked so
these don't require a running Ollama server.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import SearchResponse, SearchResult
from app.services import llm_service, retrieval_service

client = TestClient(app)


class FakeProvider:
    def generate(self, prompt):
        return "This is the synthesized answer."


def test_chat_returns_answer_with_sources(monkeypatch):
    result = SearchResult(
        video_id="v1", chunk_id="v1_0000", text="binary search halves the array",
        start_time=12.0, end_time=20.0, score=0.9,
    )
    monkeypatch.setattr(
        retrieval_service, "search",
        lambda query, top_k=None, video_id=None: SearchResponse(query=query, results=[result]),
    )
    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: FakeProvider())

    response = client.post("/api/chat", json={"query": "explain binary search"})
    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "explain binary search"
    assert body["answer"] == "This is the synthesized answer."
    assert body["has_sufficient_context"] is True
    assert len(body["sources"]) == 1
    assert body["sources"][0]["video_id"] == "v1"
    assert body["sources"][0]["start_time"] == 12.0


def test_chat_returns_no_context_answer_when_nothing_indexed(monkeypatch):
    monkeypatch.setattr(
        retrieval_service, "search",
        lambda query, top_k=None, video_id=None: SearchResponse(query=query, results=[]),
    )

    response = client.post("/api/chat", json={"query": "anything"})
    assert response.status_code == 200
    body = response.json()

    assert body["has_sufficient_context"] is False
    assert body["sources"] == []


def test_chat_returns_503_when_llm_unreachable(monkeypatch):
    result = SearchResult(
        video_id="v1", chunk_id="v1_0000", text="some text",
        start_time=0.0, end_time=5.0, score=0.9,
    )
    monkeypatch.setattr(
        retrieval_service, "search",
        lambda query, top_k=None, video_id=None: SearchResponse(query=query, results=[result]),
    )

    class UnreachableProvider:
        def generate(self, prompt):
            raise RuntimeError("Could not reach Ollama at http://localhost:11434")

    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: UnreachableProvider())

    response = client.post("/api/chat", json={"query": "anything"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]
