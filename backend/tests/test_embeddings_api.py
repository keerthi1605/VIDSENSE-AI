"""
Tests for POST/GET /api/videos/{video_id}/embeddings.

Uses a fake embedding model (monkeypatched) so these tests don't pay
the cost of loading real Sentence-Transformers weights -- that
semantic behavior is already covered by test_embedding_service.py.
"""

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.schemas import Chunk, ChunkSet
from app.services import embedding_service

client = TestClient(app)

TEST_VIDEO_ID = "embeddings_api_test_video"


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _write_fake_chunks(video_id: str, n: int = 2) -> None:
    chunks = [
        Chunk(
            chunk_id=f"{video_id}_{i:04d}",
            video_id=video_id,
            chunk_index=i,
            start_time=float(i * 10),
            end_time=float(i * 10 + 9),
            text=f"chunk {i}",
            word_count=2,
        )
        for i in range(n)
    ]
    chunk_set = ChunkSet(
        video_id=video_id,
        created_at=datetime.now(timezone.utc),
        chunk_target_words=150,
        chunk_overlap_segments=1,
        chunks=chunks,
    )
    path = settings.chunks_dir / f"{video_id}.json"
    path.write_text(chunk_set.model_dump_json(indent=2), encoding="utf-8")


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(embedding_service, "get_embedding_model", lambda: FakeEmbeddingModel())
    yield
    (settings.chunks_dir / f"{TEST_VIDEO_ID}.json").unlink(missing_ok=True)
    (settings.embeddings_dir / f"{TEST_VIDEO_ID}.json").unlink(missing_ok=True)
    (settings.embeddings_dir / f"{TEST_VIDEO_ID}.npy").unlink(missing_ok=True)


def test_embed_endpoint_requires_existing_chunks():
    response = client.post("/api/videos/no_such_video/embeddings")
    assert response.status_code == 404
    assert "Chunk it first" in response.json()["detail"]


def test_get_embeddings_before_embedding_returns_404():
    _write_fake_chunks(TEST_VIDEO_ID)
    response = client.get(f"/api/videos/{TEST_VIDEO_ID}/embeddings")
    assert response.status_code == 404


def test_embed_then_get_roundtrip():
    _write_fake_chunks(TEST_VIDEO_ID, n=2)

    post_response = client.post(f"/api/videos/{TEST_VIDEO_ID}/embeddings")
    assert post_response.status_code == 200
    body = post_response.json()

    assert body["video_id"] == TEST_VIDEO_ID
    assert body["chunk_count"] == 2
    assert body["dimension"] == 2
    assert body["chunk_ids"] == [f"{TEST_VIDEO_ID}_0000", f"{TEST_VIDEO_ID}_0001"]
    # Raw vectors must NOT be in the response.
    assert "vectors" not in body
    assert "embeddings" not in body

    get_response = client.get(f"/api/videos/{TEST_VIDEO_ID}/embeddings")
    assert get_response.status_code == 200
    assert get_response.json() == body
