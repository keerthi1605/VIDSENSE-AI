"""
Tests for app.services.retrieval_service, with vector_store_service
and embedding_service mocked out -- this layer's own job is just
correctly wiring those two together (embed -> query -> shape results),
which is what's under test here. Real end-to-end behavior (real
Chroma, real search ranking) is covered in test_search_api.py.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from app.models.schemas import Chunk, ChunkSet, EmbeddingSet
from app.services import embedding_service, retrieval_service, vector_store_service


def _chunk_set(video_id="v1", chunk_ids=("v1_0000", "v1_0001")):
    chunks = [
        Chunk(
            chunk_id=cid, video_id=video_id, chunk_index=i,
            start_time=float(i * 10), end_time=float(i * 10 + 5),
            text=f"text {i}", word_count=2,
        )
        for i, cid in enumerate(chunk_ids)
    ]
    return ChunkSet(
        video_id=video_id, created_at=datetime.now(timezone.utc),
        chunk_target_words=150, chunk_overlap_segments=1, chunks=chunks,
    )


def _embedding_meta(video_id="v1", chunk_ids=("v1_0000", "v1_0001")):
    return EmbeddingSet(
        video_id=video_id, model_name="fake", dimension=2,
        chunk_count=len(chunk_ids), created_at=datetime.now(timezone.utc),
        chunk_ids=list(chunk_ids),
    )


def test_index_video_raises_on_chunk_embedding_id_mismatch():
    chunk_set = _chunk_set(chunk_ids=("v1_0000", "v1_0001"))
    mismatched_meta = _embedding_meta(chunk_ids=("v1_0000", "v1_0099"))  # different!

    with pytest.raises(ValueError, match="mismatch"):
        retrieval_service.index_video(chunk_set, mismatched_meta)


def test_index_video_raises_when_embeddings_missing_on_disk(monkeypatch):
    monkeypatch.setattr(embedding_service, "load_embeddings", lambda video_id: None)
    chunk_set = _chunk_set()
    meta = _embedding_meta()

    with pytest.raises(ValueError, match="No usable embeddings"):
        retrieval_service.index_video(chunk_set, meta)


def test_index_video_calls_upsert_with_aligned_data(monkeypatch):
    fake_vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(embedding_service, "load_embeddings", lambda video_id: fake_vectors)

    captured = {}

    def fake_upsert(chunk_ids, embeddings, documents, metadatas):
        captured.update(
            chunk_ids=chunk_ids, embeddings=embeddings,
            documents=documents, metadatas=metadatas,
        )
        return len(chunk_ids)

    monkeypatch.setattr(vector_store_service, "upsert_chunks", fake_upsert)

    chunk_set = _chunk_set()
    meta = _embedding_meta()
    result = retrieval_service.index_video(chunk_set, meta)

    assert result.chunks_indexed == 2
    assert result.video_id == "v1"
    assert captured["chunk_ids"] == ["v1_0000", "v1_0001"]
    assert captured["documents"] == ["text 0", "text 1"]
    assert captured["metadatas"][0]["video_id"] == "v1"
    assert captured["metadatas"][0]["start_time"] == 0.0


def test_search_embeds_query_and_shapes_results(monkeypatch):
    monkeypatch.setattr(
        embedding_service, "embed_texts", lambda texts: np.array([[1.0, 0.0]], dtype=np.float32)
    )

    def fake_query(query_embedding, top_k, video_id=None):
        assert query_embedding == [1.0, 0.0]
        assert top_k == 3
        assert video_id == "v1"
        return [
            {
                "chunk_id": "v1_0000", "text": "matched text", "video_id": "v1",
                "start_time": 1.0, "end_time": 5.0, "score": 0.9,
            }
        ]

    monkeypatch.setattr(vector_store_service, "query", fake_query)

    response = retrieval_service.search(query="my question", top_k=3, video_id="v1")

    assert response.query == "my question"
    assert len(response.results) == 1
    assert response.results[0].chunk_id == "v1_0000"
    assert response.results[0].score == 0.9


def test_search_uses_default_top_k_when_not_specified(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        embedding_service, "embed_texts", lambda texts: np.array([[1.0, 0.0]], dtype=np.float32)
    )
    captured = {}

    def fake_query(query_embedding, top_k, video_id=None):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(vector_store_service, "query", fake_query)
    retrieval_service.search(query="anything")
    assert captured["top_k"] == settings.search_default_top_k
