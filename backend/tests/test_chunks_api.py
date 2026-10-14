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


# ---------------------------------------------------------------------
# Content-aligned chunking through the endpoint
# ---------------------------------------------------------------------

def _write_fake_frame_manifest(video_id: str, timestamps) -> None:
    from app.models.schemas import Frame, FrameSet

    frames = [
        Frame(
            frame_id=f"{video_id}_{i:04d}", video_id=video_id, frame_index=i,
            timestamp=float(t), image_path=f"frame_{i:04d}.jpg",
        )
        for i, t in enumerate(timestamps)
    ]
    frame_set = FrameSet(
        video_id=video_id, created_at=datetime.now(timezone.utc),
        sample_interval_seconds=5.0, diff_threshold=0.6, diff_method="ssim", frames=frames,
    )
    frame_dir = settings.frames_dir / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    (frame_dir / "manifest.json").write_text(frame_set.model_dump_json(indent=2), encoding="utf-8")


@pytest.fixture
def _aligned_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "frames_dir", tmp_path / "frames_storage")
    monkeypatch.setattr(settings, "chunk_target_words", 1000)
    monkeypatch.setattr(settings, "chunk_min_words", 8)
    monkeypatch.setattr(settings, "frame_sample_interval_seconds", 5.0)


def test_chunk_endpoint_uses_scene_boundaries_when_frames_exist(_aligned_settings):
    _write_fake_transcript(TEST_VIDEO_ID, num_segments=10)  # 6 words/segment, starts 0,5,...
    # Frames at t=0 (video start) and t=27.5 -> one scene boundary -> cut before segment 5.
    _write_fake_frame_manifest(TEST_VIDEO_ID, [0.0, 27.5])

    body = client.post(f"/api/videos/{TEST_VIDEO_ID}/chunks").json()
    assert body["alignment"] == "scene_aligned"
    assert body["scene_boundaries_available"] == 1
    assert body["scene_cuts_applied"] == 1
    assert len(body["chunks"]) == 2
    assert body["chunks"][1]["start_time"] == 25.0


def test_chunk_endpoint_falls_back_to_word_count_without_frames(_aligned_settings):
    _write_fake_transcript(TEST_VIDEO_ID, num_segments=10)
    body = client.post(f"/api/videos/{TEST_VIDEO_ID}/chunks").json()
    assert body["alignment"] == "word_count"
    assert body["scene_boundaries_available"] == 0


def test_chunk_endpoint_ignores_frames_when_alignment_is_disabled(_aligned_settings, monkeypatch):
    monkeypatch.setattr(settings, "chunk_align_to_scenes", False)
    _write_fake_transcript(TEST_VIDEO_ID, num_segments=10)
    _write_fake_frame_manifest(TEST_VIDEO_ID, [0.0, 27.5])
    body = client.post(f"/api/videos/{TEST_VIDEO_ID}/chunks").json()
    assert body["alignment"] == "word_count"
    assert len(body["chunks"]) == 1
