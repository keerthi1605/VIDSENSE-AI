"""
Tests for POST/GET /api/videos/{video_id}/frame-embeddings.

Uses a fake CLIP model (monkeypatched) so these tests don't pay the
cost of loading real CLIP weights -- that's covered by
test_vision_service.py's real-model test.
"""

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app
from app.models.schemas import Frame, FrameSet
from app.services import vision_service

client = TestClient(app)

TEST_VIDEO_ID = "frame_embeddings_api_test_video"


class FakeClipModel:
    def encode(self, inputs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        return np.array([[1.0, 0.0] for _ in inputs], dtype=np.float32)


def _write_fake_frames(video_id: str, n: int = 2) -> None:
    frame_dir = settings.frames_dir / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        image_filename = f"frame_{i:04d}.jpg"
        Image.new("RGB", (8, 8), color=(0, 255, 0)).save(frame_dir / image_filename)
        frames.append(
            Frame(
                frame_id=f"{video_id}_{i:04d}",
                video_id=video_id,
                frame_index=i,
                timestamp=float(i * 5),
                image_path=image_filename,
            )
        )
    frame_set = FrameSet(
        video_id=video_id,
        created_at=datetime.now(timezone.utc),
        sample_interval_seconds=5.0,
        diff_threshold=10.0,
        frames=frames,
    )
    (frame_dir / "manifest.json").write_text(frame_set.model_dump_json(indent=2), encoding="utf-8")


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(vision_service, "get_clip_model", lambda: FakeClipModel())
    yield
    import shutil
    frame_dir = settings.frames_dir / TEST_VIDEO_ID
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    (settings.embeddings_dir / f"{TEST_VIDEO_ID}_frames.json").unlink(missing_ok=True)
    (settings.embeddings_dir / f"{TEST_VIDEO_ID}_frames.npy").unlink(missing_ok=True)


def test_frame_embed_endpoint_requires_existing_frames():
    response = client.post("/api/videos/no_such_video/frame-embeddings")
    assert response.status_code == 404
    assert "Extract frames first" in response.json()["detail"]


def test_get_frame_embeddings_before_embedding_returns_404():
    _write_fake_frames(TEST_VIDEO_ID)
    response = client.get(f"/api/videos/{TEST_VIDEO_ID}/frame-embeddings")
    assert response.status_code == 404


def test_frame_embed_then_get_roundtrip():
    _write_fake_frames(TEST_VIDEO_ID, n=2)

    post_response = client.post(f"/api/videos/{TEST_VIDEO_ID}/frame-embeddings")
    assert post_response.status_code == 200
    body = post_response.json()

    assert body["video_id"] == TEST_VIDEO_ID
    assert body["frame_count"] == 2
    assert body["dimension"] == 2
    assert body["frame_ids"] == [f"{TEST_VIDEO_ID}_0000", f"{TEST_VIDEO_ID}_0001"]
    assert "vectors" not in body
    assert "embeddings" not in body

    get_response = client.get(f"/api/videos/{TEST_VIDEO_ID}/frame-embeddings")
    assert get_response.status_code == 200
    assert get_response.json() == body
