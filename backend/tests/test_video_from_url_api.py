"""
Tests for POST /api/videos/from-url -- thin wiring test, since the real
logic is covered by test_video_from_url.py. Confirms the endpoint calls
through correctly and surfaces video_service's HTTPExceptions unchanged.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import video_service

client = TestClient(app)


class FakeYoutubeDL:
    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        return {"title": "API Test Video", "duration": 42, "is_live": False}

    def download(self, urls):
        from pathlib import Path
        Path(self.opts["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"fake mp4 bytes")


@pytest.fixture(autouse=True)
def _cleanup(monkeypatch):
    monkeypatch.setattr(video_service.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    before = set(settings.videos_dir.glob("*"))
    yield
    after = set(settings.videos_dir.glob("*"))
    for path in after - before:
        path.unlink(missing_ok=True)


def test_from_url_endpoint_returns_video_metadata():
    response = client.post("/api/videos/from-url", json={"url": "https://youtube.com/watch?v=test"})
    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "API Test Video.mp4"
    assert body["source_url"] == "https://youtube.com/watch?v=test"
    assert "video_id" in body


def test_from_url_endpoint_rejects_bad_url(monkeypatch):
    import yt_dlp

    class FailingYoutubeDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            raise yt_dlp.utils.DownloadError("Unsupported URL")

    monkeypatch.setattr(video_service.yt_dlp, "YoutubeDL", FailingYoutubeDL)

    response = client.post("/api/videos/from-url", json={"url": "not-a-url"})
    assert response.status_code == 400
