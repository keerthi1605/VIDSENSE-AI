"""
Tests for POST/GET /api/videos/{video_id}/chunks.

Writes a fake transcript JSON directly to storage (bypassing real
Whisper) since the chunking endpoints only care that a transcript
file exists and is valid -- they don't care how it got there.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.schemas import Transcript, TranscriptSegment

client = TestClient(app)

TEST_VIDEO_ID = "chunks_api_test_video"


def _write_fake_transcript(video_id: str, num_segments: int = 5) -> None:
    segments = [
        TranscriptSegment(
            segment_id=i,
            start_time=float(i * 5),
            end_time=float(i * 5 + 4),
            text=f"segment {i} word word word word",
        )
        for i in range(num_segments)
    ]
    transcript = Transcript(
        video_id=video_id,
        language="en",
        duration_seconds=segments[-1].end_time,
        model_size="base",
        created_at=datetime.now(timezone.utc),
        segments=segments,
    )
    path = settings.transcripts_dir / f"{video_id}.json"
    path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    (settings.transcripts_dir / f"{TEST_VIDEO_ID}.json").unlink(missing_ok=True)
    (settings.chunks_dir / f"{TEST_VIDEO_ID}.json").unlink(missing_ok=True)


def test_chunk_endpoint_requires_existing_transcript():
    response = client.post("/api/videos/no_such_video/chunks")
    assert response.status_code == 404
    assert "Transcribe it first" in response.json()["detail"]


def test_get_chunks_before_chunking_returns_404():
    _write_fake_transcript(TEST_VIDEO_ID)
    response = client.get(f"/api/videos/{TEST_VIDEO_ID}/chunks")
    assert response.status_code == 404


def test_chunk_then_get_roundtrip(monkeypatch):
    monkeypatch.setattr(settings, "chunk_target_words", 10)
    monkeypatch.setattr(settings, "chunk_overlap_segments", 1)
    _write_fake_transcript(TEST_VIDEO_ID, num_segments=5)

    post_response = client.post(f"/api/videos/{TEST_VIDEO_ID}/chunks")
    assert post_response.status_code == 200
    posted = post_response.json()
    assert posted["video_id"] == TEST_VIDEO_ID
    assert len(posted["chunks"]) >= 1
    assert posted["chunks"][0]["chunk_id"] == f"{TEST_VIDEO_ID}_0000"

    get_response = client.get(f"/api/videos/{TEST_VIDEO_ID}/chunks")
    assert get_response.status_code == 200
    assert get_response.json() == posted
