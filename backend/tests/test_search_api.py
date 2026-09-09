"""
Full-pipeline integration test: real ChromaDB (temp dir) + real
chunking_service/vector_store_service/retrieval_service code paths,
with only the embedding *model* faked (deterministic, controlled
vectors) so the test is fast and results are exactly predictable --
while everything downstream of "text -> vector" is genuinely exercised,
including the actual HTTP endpoints.

This is the test that proves the whole Phase 1-3 pipeline composes
correctly end-to-end, not just each piece in isolation.
"""

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.schemas import Chunk, ChunkSet
from app.services import chunking_service, embedding_service, vector_store_service

client = TestClient(app)

# Deterministic 2D "embeddings": exact text match -> that vector.
# Lets us construct queries with predictable, checkable rankings.
VECTOR_MAP = {
    "Binary search halves a sorted array.": [1.0, 0.0],
    "Deadlock needs four conditions to occur.": [0.0, 1.0],
}


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        return np.array([VECTOR_MAP[t] for t in texts], dtype=np.float32)


def _index_video(video_id: str, text: str, monkeypatch) -> None:
    """Run the real chunk -> embed -> index pipeline for a single
    single-chunk video, via the actual service functions."""
    chunk = Chunk(
        chunk_id=f"{video_id}_0000", video_id=video_id, chunk_index=0,
        start_time=0.0, end_time=5.0, text=text, word_count=len(text.split()),
    )
    chunk_set = ChunkSet(
        video_id=video_id, created_at=datetime.now(timezone.utc),
        chunk_target_words=150, chunk_overlap_segments=1, chunks=[chunk],
    )
    (settings.chunks_dir / f"{video_id}.json").write_text(
        chunk_set.model_dump_json(indent=2), encoding="utf-8"
    )

    # embed_and_save already persists its own metadata JSON + .npy under
    # settings.embeddings_dir -- no need to write it again here.
    embedding_service.embed_and_save(chunk_set)

    response = client.post(f"/api/videos/{video_id}/index")
    assert response.status_code == 200, response.text


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_service, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / "chroma_db")
    monkeypatch.setattr(vector_store_service, "_client", None)
    monkeypatch.setattr(vector_store_service, "_collection", None)

    yield

    for video_id in ("search_test_v1", "search_test_v2"):
        (settings.chunks_dir / f"{video_id}.json").unlink(missing_ok=True)
        (settings.embeddings_dir / f"{video_id}.json").unlink(missing_ok=True)
        (settings.embeddings_dir / f"{video_id}.npy").unlink(missing_ok=True)
    monkeypatch.setattr(vector_store_service, "_client", None)
    monkeypatch.setattr(vector_store_service, "_collection", None)


def test_index_endpoint_requires_chunks_and_embeddings():
    response = client.post("/api/videos/no_such_video/index")
    assert response.status_code == 404


def test_search_post_ranks_matching_video_first(monkeypatch):
    _index_video("search_test_v1", "Binary search halves a sorted array.", monkeypatch)
    _index_video("search_test_v2", "Deadlock needs four conditions to occur.", monkeypatch)

    response = client.post(
        "/api/search", json={"query": "Binary search halves a sorted array.", "top_k": 5}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Binary search halves a sorted array."
    assert len(body["results"]) == 2
    assert body["results"][0]["video_id"] == "search_test_v1"
    assert body["results"][0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert body["results"][0]["start_time"] == 0.0
    assert body["results"][0]["end_time"] == 5.0
    assert body["results"][1]["video_id"] == "search_test_v2"
    assert body["results"][0]["score"] > body["results"][1]["score"]


def test_search_get_matches_post(monkeypatch):
    _index_video("search_test_v1", "Binary search halves a sorted array.", monkeypatch)
    _index_video("search_test_v2", "Deadlock needs four conditions to occur.", monkeypatch)

    response = client.get(
        "/api/search",
        params={"query": "Deadlock needs four conditions to occur.", "top_k": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["video_id"] == "search_test_v2"


def test_search_video_id_filter_restricts_results(monkeypatch):
    _index_video("search_test_v1", "Binary search halves a sorted array.", monkeypatch)
    _index_video("search_test_v2", "Deadlock needs four conditions to occur.", monkeypatch)

    # Query semantically closer to v1's content, but filtered to v2 --
    # should still only return v2's chunk despite the weaker match.
    response = client.post(
        "/api/search",
        json={"query": "Binary search halves a sorted array.", "video_id": "search_test_v2"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["video_id"] == "search_test_v2"
