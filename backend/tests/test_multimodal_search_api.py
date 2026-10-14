"""
Integration test for fused multimodal search: real ChromaDB (temp dir),
real retrieval_service fusion code, real HTTP endpoints -- with only the
two query embedders faked (MiniLM and CLIP text encoders) so rankings
are deterministic. Vectors are upserted directly into both collections
with hand-picked 2D values: [1, 0] means "matches the query",
[0, 1] means "does not".

Proves the Phase 6 claim end to end: transcript hits and frame hits for
the same query get aligned by timestamp into one ranked list, across two
genuinely separate collections.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import embedding_service, vector_store_service, vision_service

client = TestClient(app)

MATCH = [1.0, 0.0]
NO_MATCH = [0.0, 1.0]
PARTIAL = [0.8, 0.6]  # unit length, cosine 0.8 vs MATCH


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / "chroma_db")
    monkeypatch.setattr(vector_store_service, "_client", None)
    monkeypatch.setattr(vector_store_service, "_collections", {})
    # Both query encoders map every query onto the "match" direction.
    monkeypatch.setattr(
        embedding_service, "embed_texts", lambda texts: np.array([MATCH], dtype=np.float32)
    )
    monkeypatch.setattr(
        vision_service, "embed_text_for_visual", lambda text: np.array(MATCH, dtype=np.float32)
    )
    yield
    monkeypatch.setattr(vector_store_service, "_client", None)
    monkeypatch.setattr(vector_store_service, "_collections", {})


def _index_chunk(video_id, idx, start, end, vector):
    vector_store_service.upsert(
        collection_name=settings.chroma_collection_name,
        ids=[f"{video_id}_{idx:04d}"],
        embeddings=[vector],
        documents=[f"transcript text {idx}"],
        metadatas=[{"video_id": video_id, "start_time": start, "end_time": end, "chunk_index": idx}],
    )


def _index_frame(video_id, idx, timestamp, vector):
    vector_store_service.upsert(
        collection_name=settings.chroma_frames_collection_name,
        ids=[f"{video_id}_f{idx:04d}"],
        embeddings=[vector],
        documents=[f"frame_{idx:04d}.jpg"],
        metadatas=[{
            "video_id": video_id, "timestamp": timestamp,
            "frame_index": idx, "image_path": f"frame_{idx:04d}.jpg",
        }],
    )


def _seed_video(video_id="mm_v1"):
    # Chunk 0 [0-10] matches; chunk 1 [20-30] does not.
    _index_chunk(video_id, 0, 0.0, 10.0, MATCH)
    _index_chunk(video_id, 1, 20.0, 30.0, NO_MATCH)
    # Frame at 5s (inside chunk 0) matches -> merges into a "both" result.
    _index_frame(video_id, 0, 5.0, MATCH)
    # Frame at 25s (inside chunk 1) does not match.
    _index_frame(video_id, 1, 25.0, NO_MATCH)
    # Frame at 100s is outside every chunk, partially matches -> visual-only.
    _index_frame(video_id, 2, 100.0, PARTIAL)


def test_fused_search_aligns_modalities_by_timestamp():
    _seed_video()
    response = client.post("/api/search/multimodal", json={"query": "anything", "top_k": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["text_weight"] == settings.multimodal_text_weight

    results = body["results"]
    top = results[0]
    assert top["evidence_type"] == "both"
    assert top["chunk_id"] == "mm_v1_0000"
    assert top["frame_id"] == "mm_v1_f0000"
    assert top["text"] == "transcript text 0"
    assert top["image_path"] == "frame_0000.jpg"
    assert (top["start_time"], top["end_time"]) == (0.0, 10.0)
    assert top["text_score"] == pytest.approx(1.0)
    assert top["visual_score"] == pytest.approx(1.0)

    visual_only = [r for r in results if r["evidence_type"] == "visual"]
    assert [r["frame_id"] for r in visual_only if r["frame_id"] == "mm_v1_f0002"]
    far_frame = next(r for r in results if r["frame_id"] == "mm_v1_f0002")
    assert far_frame["start_time"] == far_frame["end_time"] == 100.0

    # The matched frame is folded into the "both" result, not repeated.
    assert sum(1 for r in results if r["frame_id"] == "mm_v1_f0000") == 1
    # Fused list is sorted best-first.
    scores = [r["final_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_fused_get_matches_post_and_weights_are_overridable():
    _seed_video()
    response = client.get(
        "/api/search/multimodal",
        params={"query": "anything", "text_weight": 0.0, "visual_weight": 1.0, "top_k": 10},
    )
    assert response.status_code == 200
    body = response.json()
    assert (body["text_weight"], body["visual_weight"]) == (0.0, 1.0)
    # With text switched off, a text-only result can no longer score.
    text_only = [r for r in body["results"] if r["evidence_type"] == "text"]
    assert all(r["final_score"] == 0.0 for r in text_only)


def test_fused_search_respects_video_id_filter():
    _seed_video("mm_v1")
    _seed_video("mm_v2")
    response = client.post(
        "/api/search/multimodal", json={"query": "anything", "video_id": "mm_v2", "top_k": 10}
    )
    assert response.status_code == 200
    assert {r["video_id"] for r in response.json()["results"]} == {"mm_v2"}


def test_fused_search_on_empty_index_returns_empty_list():
    response = client.post("/api/search/multimodal", json={"query": "anything"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_negative_weight_is_rejected_with_422():
    response = client.post(
        "/api/search/multimodal", json={"query": "anything", "text_weight": -1}
    )
    assert response.status_code == 422
    response = client.get("/api/search/multimodal", params={"query": "x", "visual_weight": -0.5})
    assert response.status_code == 422
