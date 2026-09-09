"""
Tests for app.services.embedding_service.

Split into two groups:
  - Fast tests using a fake model (pure math, persistence, edge cases)
  - One real-model test that actually loads Sentence-Transformers and
    checks a genuine semantic property: two similar sentences should
    score higher on cosine similarity than two unrelated ones. This is
    the actual claim embeddings make -- worth verifying for real, not
    just trusting the library.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from app.core.config import settings
from app.models.schemas import Chunk, ChunkSet
from app.services import embedding_service


# ---------------------------------------------------------------------
# cosine_similarity: pure math, no model needed
# ---------------------------------------------------------------------

def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert embedding_service.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert embedding_service.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert embedding_service.cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_does_not_crash():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert embedding_service.cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------
# embed_and_save / persistence: fake model, fast and deterministic
# ---------------------------------------------------------------------

class FakeEmbeddingModel:
    """Returns a distinct, deterministic 3-dim vector per input text,
    based on its length -- enough to verify ordering and shape without
    needing real semantics."""

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        return np.array([[float(len(t)), 0.0, 1.0] for t in texts], dtype=np.float32)


def _chunk_set(video_id="vid1", n=3):
    chunks = [
        Chunk(
            chunk_id=f"{video_id}_{i:04d}",
            video_id=video_id,
            chunk_index=i,
            start_time=float(i * 10),
            end_time=float(i * 10 + 9),
            text=f"chunk number {i} text",
            word_count=4,
        )
        for i in range(n)
    ]
    return ChunkSet(
        video_id=video_id,
        created_at=datetime.now(timezone.utc),
        chunk_target_words=150,
        chunk_overlap_segments=1,
        chunks=chunks,
    )


@pytest.fixture(autouse=True)
def _fake_model_and_tmp_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(embedding_service, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(settings, "embeddings_dir", tmp_path)
    monkeypatch.setattr(settings, "embedding_model_name", "fake-model-for-tests")


def test_embed_and_save_persists_correct_shape_and_order():
    chunk_set = _chunk_set(n=3)
    metadata = embedding_service.embed_and_save(chunk_set)

    assert metadata.video_id == "vid1"
    assert metadata.chunk_count == 3
    assert metadata.dimension == 3
    assert metadata.chunk_ids == ["vid1_0000", "vid1_0001", "vid1_0002"]
    assert metadata.model_name == "fake-model-for-tests"

    vectors = embedding_service.load_embeddings("vid1")
    assert vectors.shape == (3, 3)
    # Row order matches chunk order (row i <-> chunk_ids[i]).
    for i, chunk in enumerate(chunk_set.chunks):
        assert vectors[i][0] == float(len(chunk.text))


def test_embed_and_save_handles_empty_chunk_set():
    empty = _chunk_set(n=0)
    metadata = embedding_service.embed_and_save(empty)

    assert metadata.chunk_count == 0
    assert metadata.dimension == 0
    assert metadata.chunk_ids == []

    vectors = embedding_service.load_embeddings("vid1")
    assert vectors.shape == (0, 0)


def test_get_embedding_metadata_returns_none_when_missing():
    assert embedding_service.get_embedding_metadata("nonexistent") is None


def test_load_embeddings_returns_none_when_missing():
    assert embedding_service.load_embeddings("nonexistent") is None


# ---------------------------------------------------------------------
# Real model: proves the actual semantic-similarity claim
# ---------------------------------------------------------------------

def test_real_model_ranks_similar_sentence_above_unrelated_one(monkeypatch):
    # Undo the autouse fake-model patch for this one test -- we want
    # the real Sentence-Transformers model here. Safe because pytest
    # hands the autouse fixture and this test the *same* monkeypatch
    # instance for a given test (function-scoped fixture caching).
    monkeypatch.undo()

    anchor = "The professor explained how binary search works."
    similar = "Binary search was described by the instructor."
    unrelated = "The stock market rallied sharply this afternoon."

    vectors = embedding_service.embed_texts([anchor, similar, unrelated])

    # Vectors are normalized -> each should have unit length.
    for v in vectors:
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)

    sim_to_similar = embedding_service.cosine_similarity(vectors[0], vectors[1])
    sim_to_unrelated = embedding_service.cosine_similarity(vectors[0], vectors[2])

    assert sim_to_similar > sim_to_unrelated
