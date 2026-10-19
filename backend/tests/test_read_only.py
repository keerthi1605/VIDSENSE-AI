"""
Tests for read-only demo mode (settings.read_only): the API refuses anything that changes data or
starts heavy processing, while reads, search and chat keep working, and /health reports the flag.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture
def read_only(monkeypatch):
    monkeypatch.setattr(settings, "read_only", True)


@pytest.mark.parametrize(
    "method, path",
    [
        ("post", "/api/videos/upload"),
        ("post", "/api/videos/from-url"),
        ("post", "/api/videos/transcribe"),
        ("post", "/api/videos/abc/chunks"),
        ("post", "/api/videos/abc/index"),
        ("post", "/api/videos/abc/summary"),
        ("delete", "/api/videos/abc"),
    ],
)
def test_changes_are_refused_with_403(read_only, method, path):
    response = getattr(client, method)(path)
    assert response.status_code == 403
    assert "read-only" in response.json()["detail"]


def test_reads_and_search_and_chat_stay_open(read_only):
    assert client.get("/health").status_code == 200
    assert client.get("/api/videos").status_code == 200
    # Search and chat are POSTs but only read; an empty body reaches validation (422), not the guard.
    assert client.post("/api/search", json={}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422


def test_refusal_carries_cors_headers_so_the_browser_can_read_it(read_only):
    response = client.post("/api/videos/upload", headers={"Origin": "https://demo.vercel.app"})
    assert response.status_code == 403
    assert response.headers.get("access-control-allow-origin") == "*"


def test_health_reports_the_flag(read_only):
    assert client.get("/health").json()["read_only"] is True


def test_default_is_not_read_only():
    assert settings.read_only is False
    assert client.get("/health").json()["read_only"] is False
