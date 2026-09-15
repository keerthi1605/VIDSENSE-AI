"""
Tests for app.services.retrieval_service, with vector_store_service and
embedding_service/vision_service mocked out -- this layer's own job is
just correctly wiring those together (embed -> query -> shape results),
which is what's under test here. Real end-to-end behavior (real Chroma,
real search ranking) is covered in test_search_api.py.

Split into a text (chunks) section and a visual (frames) section,
mirroring retrieval_service.py's own layout.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from app.models.schemas import Chunk, ChunkSet, EmbeddingSet, Frame, FrameEmbeddingSet, FrameSet
from app.services import embedding_service, retrieval_service, vector_store_service, vision_service


# ---------------------------------------------------------------------
# Text (transcript chunks)
# ---------------------------------------------------------------------

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

    def fake_upsert(collection_name, ids, embeddings, documents, metadatas):
        captured.update(
            collection_name=collection_name, ids=ids,
            embeddings=embeddings, documents=documents, metadatas=metadatas,
        )
        return len(ids)

    monkeypatch.setattr(vector_store_service, "upsert", fake_upsert)

    chunk_set = _chunk_set()
    meta = _embedding_meta()
    result = retrieval_service.index_video(chunk_set, meta)

    assert result.chunks_indexed == 2
    assert result.video_id == "v1"
    assert captured["ids"] == ["v1_0000", "v1_0001"]
    assert captured["documents"] == ["text 0", "text 1"]
    assert captured["metadatas"][0]["video_id"] == "v1"
    assert captured["metadatas"][0]["start_time"] == 0.0


def test_search_embeds_query_and_shapes_results(monkeypatch):
    monkeypatch.setattr(
        embedding_service, "embed_texts", lambda texts: np.array([[1.0, 0.0]], dtype=np.float32)
    )

    def fake_query(collection_name, query_embedding, top_k, video_id=None):
        assert query_embedding == [1.0, 0.0]
        assert top_k == 3
        assert video_id == "v1"
        return [
            {
                "id": "v1_0000", "document": "matched text",
                "metadata": {"video_id": "v1", "start_time": 1.0, "end_time": 5.0},
                "score": 0.9,
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

    def fake_query(collection_name, query_embedding, top_k, video_id=None):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(vector_store_service, "query", fake_query)
    retrieval_service.search(query="anything")
    assert captured["top_k"] == settings.search_default_top_k


# ---------------------------------------------------------------------
# Visual (video frames)
# ---------------------------------------------------------------------

def _frame_set(video_id="v1", frame_ids=("v1_0000", "v1_0001")):
    frames = [
        Frame(
            frame_id=fid, video_id=video_id, frame_index=i,
            timestamp=float(i * 5), image_path=f"frame_{i:04d}.jpg",
        )
        for i, fid in enumerate(frame_ids)
    ]
    return FrameSet(
        video_id=video_id, created_at=datetime.now(timezone.utc),
        sample_interval_seconds=5.0, diff_threshold=10.0, frames=frames,
    )


def _frame_embedding_meta(video_id="v1", frame_ids=("v1_0000", "v1_0001")):
    return FrameEmbeddingSet(
        video_id=video_id, model_name="fake-clip", dimension=2,
        frame_count=len(frame_ids), created_at=datetime.now(timezone.utc),
        frame_ids=list(frame_ids),
    )


def test_index_video_frames_raises_on_id_mismatch():
    frame_set = _frame_set(frame_ids=("v1_0000", "v1_0001"))
    mismatched_meta = _frame_embedding_meta(frame_ids=("v1_0000", "v1_0099"))

    with pytest.raises(ValueError, match="mismatch"):
        retrieval_service.index_video_frames(frame_set, mismatched_meta)


def test_index_video_frames_raises_when_embeddings_missing_on_disk(monkeypatch):
    monkeypatch.setattr(vision_service, "load_frame_embeddings", lambda video_id: None)
    frame_set = _frame_set()
    meta = _frame_embedding_meta()

    with pytest.raises(ValueError, match="No usable frame embeddings"):
        retrieval_service.index_video_frames(frame_set, meta)


def test_index_video_frames_calls_upsert_with_aligned_data(monkeypatch):
    fake_vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(vision_service, "load_frame_embeddings", lambda video_id: fake_vectors)

    captured = {}

    def fake_upsert(collection_name, ids, embeddings, documents, metadatas):
        captured.update(
            collection_name=collection_name, ids=ids,
            embeddings=embeddings, documents=documents, metadatas=metadatas,
        )
        return len(ids)

    monkeypatch.setattr(vector_store_service, "upsert", fake_upsert)

    from app.core.config import settings
    frame_set = _frame_set()
    meta = _frame_embedding_meta()
    result = retrieval_service.index_video_frames(frame_set, meta)

    assert result.frames_indexed == 2
    assert result.video_id == "v1"
    assert result.collection_name == settings.chroma_frames_collection_name
    assert captured["collection_name"] == settings.chroma_frames_collection_name
    assert captured["ids"] == ["v1_0000", "v1_0001"]
    assert captured["metadatas"][0]["video_id"] == "v1"
    assert captured["metadatas"][0]["timestamp"] == 0.0
    assert captured["metadatas"][0]["image_path"] == "frame_0000.jpg"


def test_search_frames_uses_clip_text_encoder_not_minilm(monkeypatch):
    """The critical distinction from text search: the query must go
    through vision_service's CLIP text encoder, never
    embedding_service's MiniLM -- the two are non-comparable spaces."""
    minilm_called = False

    def fake_minilm(texts):
        nonlocal minilm_called
        minilm_called = True
        return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(embedding_service, "embed_texts", fake_minilm)
    monkeypatch.setattr(vision_service, "embed_text_for_visual", lambda text: np.array([1.0, 0.0], dtype=np.float32))

    def fake_query(collection_name, query_embedding, top_k, video_id=None):
        from app.core.config import settings
        assert collection_name == settings.chroma_frames_collection_name
        assert query_embedding == [1.0, 0.0]
        return [
            {
                "id": "v1_0000",
                "document": "frame_0000.jpg",
                "metadata": {"video_id": "v1", "timestamp": 12.5, "image_path": "frame_0000.jpg"},
                "score": 0.75,
            }
        ]

    monkeypatch.setattr(vector_store_service, "query", fake_query)

    response = retrieval_service.search_frames(query="a slide with a diagram", top_k=5, video_id="v1")

    assert minilm_called is False
    assert response.query == "a slide with a diagram"
    assert len(response.results) == 1
    assert response.results[0].frame_id == "v1_0000"
    assert response.results[0].timestamp == 12.5
    assert response.results[0].image_path == "frame_0000.jpg"
    assert response.results[0].score == 0.75


def test_search_frames_uses_default_top_k_when_not_specified(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(vision_service, "embed_text_for_visual", lambda text: np.array([1.0, 0.0], dtype=np.float32))
    captured = {}

    def fake_query(collection_name, query_embedding, top_k, video_id=None):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(vector_store_service, "query", fake_query)
    retrieval_service.search_frames(query="anything")
    assert captured["top_k"] == settings.search_default_top_k
