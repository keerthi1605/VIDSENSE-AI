"""
Tests for app.services.vector_store_service.

Uses a real ChromaDB instance rooted at a temp directory per test (not
a mock) -- this layer's whole job is correctly talking to Chroma, so
that's exactly what's worth testing for real. Hand-crafted, tiny
embedding vectors keep tests fast and let similarity be reasoned about
by hand (no need to run the real embedding model here -- that's
already validated in test_embedding_service.py).
"""

import pytest

from app.core.config import settings
from app.services import vector_store_service


@pytest.fixture(autouse=True)
def _fresh_chroma(tmp_path, monkeypatch):
    """A brand-new, empty ChromaDB directory for every test, and reset
    the module's cached client/collection singleton so each test
    actually opens a fresh one instead of reusing a prior test's."""
    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / "chroma_db")
    monkeypatch.setattr(vector_store_service, "_client", None)
    monkeypatch.setattr(vector_store_service, "_collection", None)
    yield
    monkeypatch.setattr(vector_store_service, "_client", None)
    monkeypatch.setattr(vector_store_service, "_collection", None)


def test_upsert_then_query_finds_the_matching_chunk():
    vector_store_service.upsert_chunks(
        chunk_ids=["v1_0000", "v1_0001"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["about apples", "about oranges"],
        metadatas=[
            {"video_id": "v1", "start_time": 0.0, "end_time": 5.0, "chunk_index": 0},
            {"video_id": "v1", "start_time": 5.0, "end_time": 10.0, "chunk_index": 1},
        ],
    )

    results = vector_store_service.query(query_embedding=[1.0, 0.0], top_k=2)

    assert results[0]["chunk_id"] == "v1_0000"
    assert results[0]["text"] == "about apples"
    assert results[0]["video_id"] == "v1"
    assert results[0]["start_time"] == 0.0
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
    # The non-matching chunk should score lower.
    assert results[1]["score"] < results[0]["score"]


def test_query_respects_top_k():
    vector_store_service.upsert_chunks(
        chunk_ids=["v1_0000", "v1_0001", "v1_0002"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        documents=["a", "b", "c"],
        metadatas=[
            {"video_id": "v1", "start_time": float(i), "end_time": float(i + 1), "chunk_index": i}
            for i in range(3)
        ],
    )
    results = vector_store_service.query(query_embedding=[1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "v1_0000"


def test_query_filters_by_video_id():
    vector_store_service.upsert_chunks(
        chunk_ids=["v1_0000"],
        embeddings=[[1.0, 0.0]],
        documents=["video one content"],
        metadatas=[{"video_id": "v1", "start_time": 0.0, "end_time": 5.0, "chunk_index": 0}],
    )
    vector_store_service.upsert_chunks(
        chunk_ids=["v2_0000"],
        embeddings=[[1.0, 0.0]],  # identical vector, different video
        documents=["video two content"],
        metadatas=[{"video_id": "v2", "start_time": 0.0, "end_time": 5.0, "chunk_index": 0}],
    )

    results = vector_store_service.query(query_embedding=[1.0, 0.0], top_k=5, video_id="v2")
    assert len(results) == 1
    assert results[0]["video_id"] == "v2"


def test_upsert_is_idempotent_for_the_same_chunk_id():
    vector_store_service.upsert_chunks(
        chunk_ids=["v1_0000"],
        embeddings=[[1.0, 0.0]],
        documents=["original text"],
        metadatas=[{"video_id": "v1", "start_time": 0.0, "end_time": 5.0, "chunk_index": 0}],
    )
    vector_store_service.upsert_chunks(
        chunk_ids=["v1_0000"],
        embeddings=[[1.0, 0.0]],
        documents=["updated text"],
        metadatas=[{"video_id": "v1", "start_time": 0.0, "end_time": 5.0, "chunk_index": 0}],
    )

    assert vector_store_service.count() == 1
    results = vector_store_service.query(query_embedding=[1.0, 0.0], top_k=5)
    assert results[0]["text"] == "updated text"


def test_upsert_empty_list_is_a_noop():
    assert vector_store_service.upsert_chunks([], [], [], []) == 0
    assert vector_store_service.count() == 0
