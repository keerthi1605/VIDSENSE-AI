"""
Tests for POST /api/videos/upload.

Covers: happy path (valid small file), rejected extension, and the
streaming size-cap enforcement (using a monkeypatched low limit so we
don't need to generate a real multi-GB file to test it).
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_uploaded_files():
    """Track files created during each test and remove them afterwards,
    so repeated test runs don't accumulate junk in backend/storage/videos/."""
    before = set(settings.videos_dir.glob("*"))
    yield
    after = set(settings.videos_dir.glob("*"))
    for path in after - before:
        path.unlink(missing_ok=True)


def test_upload_valid_video_returns_metadata():
    fake_video_bytes = b"\x00\x01\x02\x03" * 100  # not a real mp4, just bytes
    response = client.post(
        "/api/videos/upload",
        files={"file": ("lecture.mp4", fake_video_bytes, "video/mp4")},
    )

    assert response.status_code == 201
    body = response.json()

    assert body["original_filename"] == "lecture.mp4"
    assert body["extension"] == ".mp4"
    assert body["size_bytes"] == len(fake_video_bytes)
    assert body["status"] == "uploaded"
    assert "video_id" in body and len(body["video_id"]) > 0

    # The file and its metadata sidecar should actually exist on disk.
    stored_path = settings.videos_dir / body["stored_filename"]
    metadata_path = settings.videos_dir / f"{body['video_id']}.json"
    assert stored_path.exists()
    assert stored_path.stat().st_size == len(fake_video_bytes)
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text())["video_id"] == body["video_id"]


def test_upload_rejects_disallowed_extension():
    response = client.post(
        "/api/videos/upload",
        files={"file": ("notes.txt", b"just text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_upload_rejects_file_over_size_cap(monkeypatch):
    # Rather than uploading a real multi-hundred-MB file, shrink the
    # configured cap to a few bytes and confirm the streaming check
    # catches it mid-write instead of only after the fact.
    monkeypatch.setattr(settings, "max_upload_size_mb", 0.000001)  # ~1 byte

    # Snapshot before, not an assumption of an empty directory -- other
    # legitimate videos (uploaded by other tests, or real manual usage)
    # may already be sitting in storage/videos/.
    before = set(settings.videos_dir.glob("*.mp4"))

    response = client.post(
        "/api/videos/upload",
        files={"file": ("lecture.mp4", b"more than one byte of data", "video/mp4")},
    )
    assert response.status_code == 413

    # No NEW partial file should have been left behind by this request.
    after = set(settings.videos_dir.glob("*.mp4"))
    assert after == before
