"""
Tests for GET /api/videos, GET /api/videos/{video_id}, and
GET /api/videos/{video_id}/file -- added to support the frontend
(Phase 8 groundwork): a library view, single-video lookup, and a
streamable file for the <video> player.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture
def uploaded_video():
    response = client.post(
        "/api/videos/upload",
        files={"file": ("meta_test.mp4", b"fake video bytes", "video/mp4")},
    )
    metadata = response.json()
    yield metadata
    for path in settings.videos_dir.glob(f"{metadata['video_id']}*"):
        path.unlink(missing_ok=True)


def test_list_videos_includes_uploaded_video(uploaded_video):
    response = client.get("/api/videos")
    assert response.status_code == 200
    video_ids = [v["video_id"] for v in response.json()]
    assert uploaded_video["video_id"] in video_ids


def test_get_single_video_returns_metadata(uploaded_video):
    response = client.get(f"/api/videos/{uploaded_video['video_id']}")
    assert response.status_code == 200
    assert response.json() == uploaded_video


def test_get_single_video_404s_for_unknown_id():
    response = client.get("/api/videos/no_such_video")
    assert response.status_code == 404


def test_get_video_file_streams_the_actual_bytes(uploaded_video):
    response = client.get(f"/api/videos/{uploaded_video['video_id']}/file")
    assert response.status_code == 200
    assert response.content == b"fake video bytes"
    assert response.headers["content-type"] == "video/mp4"


def test_get_video_file_404s_for_unknown_id():
    response = client.get("/api/videos/no_such_video/file")
    assert response.status_code == 404
