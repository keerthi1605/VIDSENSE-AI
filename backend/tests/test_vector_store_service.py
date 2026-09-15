"""
Tests for app.services.vector_store_service.

Uses a real ChromaDB instance rooted at a temp directory per test (not
a mock) -- this layer's whole job is correctly talking to Chroma, so
that's exactly what's worth testing for real. Hand-crafted, tiny
embedding vectors keep tests fast and let similarity be reasoned about
by hand (no need to run a real embedding model here).

The module is collection-agnostic (Phase 5, Step 3) -- every call
passes an explicit collection_name, and results come back as generic
{id, document, metadata, score} dicts, not chunk-specific fields. These
tests use a plain "test_items" collection name; which domain (chunks vs
frames) a collection represents is retrieval_service's concern, not
this module's.
"""

import pytest

from app.core.config import settings
from app.services import vector_store_service

COLLECTION = "test_items"


@pytest.fixture(autouse=True)
def _fresh_chroma(tmp_path, monkeypatch):
    """A brand-new, empty ChromaDB directory for every test, and reset
    the module's cached client/collections singletons so each test
    actually opens a fresh one instead of reusing a prior test's."""
    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / "chroma_db")
    monkeypatch.setattr(vector_store_service, "_client", None)
    monkeypatch.setattr(vector_store_service, "_collections", {})
    yield
    monkeypatch.setattr(vector_store_service, "_client", None)
    monkeypatch.setattr(vector_store_service, "_collections", {})


def test_upsert_then_query_finds_the_matching_item():
    vector_store_service.upsert(
        collection_name=COLLECTION,
        ids=["v1_0000", "v1_0001"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["about apples", "about oranges"],
        metadatas=[
            {"video_id": "v1", "start_time": 0.0, "end_time": 5.0},
            {"video_id": "v1", "start_time": 5.0, "end_time": 10.0},
        ],
    )

    results = vector_store_service.query(collection_name=COLLECTION, query_embedding=[1.0, 0.0], top_k=2)

    assert results[0]["id"] == "v1_0000"
    assert results[0]["document"] == "about apples"
    assert results[0]["metadata"]["video_id"] == "v1"
    assert results[0]["metadata"]["start_time"] == 0.0
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
    # The non-matching item should score lower.
    assert results[1]["score"] < results[0]["score"]


def test_query_respects_top_k():
    vector_store_service.upsert(
        collection_name=COLLECTION,
        ids=["v1_0000", "v1_0001", "v1_0002"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        documents=["a", "b", "c"],
        metadatas=[{"video_id": "v1", "start_time": float(i), "end_time": float(i + 1)} for i in range(3)],
    )
    results = vector_store_service.query(collection_name=COLLECTION, query_embedding=[1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "v1_0000"


def test_query_filters_by_video_id():
    vector_store_service.upsert(
        collection_name=COLLECTION,
        ids=["v1_0000"],
        embeddings=[[1.0, 0.0]],
        documents=["video one content"],
        metadatas=[{"video_id": "v1", "start_time": 0.0, "end_time": 5.0}],
    )
    vector_store_service.upsert(
        collection_name=COLLECTION,
        ids=["v2_0000"],
        embeddings=[[1.0, 0.0]],  # identical vector, different video
        documents=["video two content"],
        metadatas=[{"video_id": "v2", "start_time": 0.0, "end_time": 5.0}],
    )

    results = vector_store_service.query(
        collection_name=COLLECTION, query_embedding=[1.0, 0.0], top_k=5, video_id="v2"
    )
    assert len(results) == 1
    assert results[0]["metadata"]["video_id"] == "v2"


def test_upsert_is_idempotent_for_the_same_id():
    vector_store_service.upsert(
        collection_name=COLLECTION,
        ids=["v1_0000"],
        embeddings=[[1.0, 0.0]],
        documents=["original text"],
        metadatas=[{"video_id": "v1", "start_time": 0.0, "end_time": 5.0}],
    )
    vector_store_service.upsert(
        collection_name=COLLECTION,
        ids=["v1_0000"],
        embeddings=[[1.0, 0.0]],
        documents=["updated text"],
        metadatas=[{"video_id": "v1", "start_time": 0.0, "end_time": 5.0}],
    )

    assert vector_store_service.count(COLLECTION) == 1
    results = vector_store_service.query(collection_name=COLLECTION, query_embedding=[1.0, 0.0], top_k=5)
    assert results[0]["document"] == "updated text"


def test_upsert_empty_list_is_a_noop():
    assert vector_store_service.upsert(COLLECTION, [], [], [], []) == 0
    assert vector_store_service.count(COLLECTION) == 0


def test_separate_collections_are_independent():
    """The whole point of Phase 5 Step 3's refactor: two differently-
    dimensioned collections (e.g. 384-dim chunks, 512-dim frames) must
    coexist without interfering."""
    vector_store_service.upsert(
        collection_name="collection_a",
        ids=["a_0000"],
        embeddings=[[1.0, 0.0, 0.0]],
        documents=["in collection a"],
        metadatas=[{"video_id": "v1"}],
    )
    vector_store_service.upsert(
        collection_name="collection_b",
        ids=["b_0000"],
        embeddings=[[1.0, 0.0]],  # different dimensionality entirely
        documents=["in collection b"],
        metadatas=[{"video_id": "v1"}],
    )

    assert vector_store_service.count("collection_a") == 1
    assert vector_store_service.count("collection_b") == 1

    results_a = vector_store_service.query(collection_name="collection_a", query_embedding=[1.0, 0.0, 0.0], top_k=5)
    assert len(results_a) == 1
    assert results_a[0]["id"] == "a_0000"
