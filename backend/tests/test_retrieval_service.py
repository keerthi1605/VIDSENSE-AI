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

    monkeypatch.setattr(vector_store_service, "delete_video", lambda collection_name, video_id: None)
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

    monkeypatch.setattr(vector_store_service, "delete_video", lambda collection_name, video_id: None)
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


# ---------------------------------------------------------------------
# Multimodal fusion (Phase 6)
# ---------------------------------------------------------------------

from app.models.schemas import FrameSearchResult, SearchResponse, SearchResult, VisualSearchResponse


def _text_hit(chunk_id, start, end, score, video_id="v1"):
    return SearchResult(
        video_id=video_id, chunk_id=chunk_id, text=f"text of {chunk_id}",
        start_time=start, end_time=end, score=score,
    )


def _frame_hit(frame_id, timestamp, score, video_id="v1"):
    return FrameSearchResult(
        video_id=video_id, frame_id=frame_id, timestamp=timestamp,
        image_path=f"{frame_id}.jpg", score=score,
    )


def _stub_modalities(monkeypatch, text_hits, frame_hits):
    monkeypatch.setattr(
        retrieval_service, "search",
        lambda query, top_k=None, video_id=None, **kw: SearchResponse(query=query, results=text_hits),
    )
    monkeypatch.setattr(
        retrieval_service, "search_frames",
        lambda query, top_k=None, video_id=None: VisualSearchResponse(query=query, results=frame_hits),
    )


def test_min_max_normalize_rescales_to_unit_range():
    assert retrieval_service._min_max_normalize([0.5, 0.75, 1.0]) == [0.0, 0.5, 1.0]


def test_min_max_normalize_handles_empty_and_identical_scores():
    assert retrieval_service._min_max_normalize([]) == []
    # No spread to rescale (incl. a lone result): "best of its modality".
    assert retrieval_service._min_max_normalize([0.3]) == [1.0]
    assert retrieval_service._min_max_normalize([0.3, 0.3]) == [1.0, 1.0]


def test_fusion_normalizes_each_modality_independently(monkeypatch):
    """MiniLM-scale (0.8) and CLIP-scale (0.3) top scores must both end
    up as 1.0 -- otherwise text would silently dominate."""
    _stub_modalities(
        monkeypatch,
        [_text_hit("c0", 0, 5, 0.80), _text_hit("c1", 100, 105, 0.60)],
        [_frame_hit("f0", 200.0, 0.30), _frame_hit("f1", 300.0, 0.20)],
    )
    response = retrieval_service.search_multimodal("q", top_k=10)
    by_id = {(r.chunk_id or r.frame_id): r for r in response.results}
    assert by_id["c0"].text_score == 1.0
    assert by_id["f0"].visual_score == 1.0
    assert by_id["c1"].text_score == 0.0
    assert by_id["f1"].visual_score == 0.0


def test_frame_inside_chunk_merges_into_both_result(monkeypatch):
    _stub_modalities(
        monkeypatch,
        [_text_hit("c0", 10, 20, 0.8), _text_hit("c1", 50, 60, 0.4)],
        [_frame_hit("f0", 15.0, 0.3), _frame_hit("f1", 90.0, 0.1)],
    )
    response = retrieval_service.search_multimodal("q", top_k=10)

    both = [r for r in response.results if r.evidence_type == "both"]
    assert len(both) == 1
    assert both[0].chunk_id == "c0" and both[0].frame_id == "f0"
    assert both[0].text == "text of c0"
    assert both[0].image_path == "f0.jpg"
    assert (both[0].start_time, both[0].end_time) == (10, 20)
    assert both[0].final_score == pytest.approx(0.5 * 1.0 + 0.5 * 1.0)

    # The matched frame is NOT also emitted as a separate visual result.
    assert [r.frame_id for r in response.results if r.evidence_type == "visual"] == ["f1"]
    assert [r.chunk_id for r in response.results if r.evidence_type == "text"] == ["c1"]


def test_visual_only_result_has_degenerate_but_valid_time_range(monkeypatch):
    _stub_modalities(monkeypatch, [], [_frame_hit("f0", 42.5, 0.3)])
    result = retrieval_service.search_multimodal("q").results[0]
    assert result.evidence_type == "visual"
    assert result.start_time == result.end_time == 42.5
    assert result.text is None and result.chunk_id is None


def test_agreement_between_modalities_can_outrank_stronger_single_match(monkeypatch):
    """The point of fusion: a moment both modalities like beats a moment
    only text likes, even if that text match is the single best chunk."""
    _stub_modalities(
        monkeypatch,
        [_text_hit("top_text_only", 0, 5, 0.85), _text_hit("agreed", 50, 60, 0.80)],
        [_frame_hit("f_agreed", 55.0, 0.30), _frame_hit("f_other", 500.0, 0.10)],
    )
    results = retrieval_service.search_multimodal("q", top_k=10).results
    assert results[0].chunk_id == "agreed"
    assert results[0].evidence_type == "both"


def test_frame_from_a_different_video_is_not_merged(monkeypatch):
    _stub_modalities(
        monkeypatch,
        [_text_hit("c0", 10, 20, 0.8, video_id="v1")],
        [_frame_hit("f0", 15.0, 0.3, video_id="v2")],
    )
    evidence = {r.evidence_type for r in retrieval_service.search_multimodal("q").results}
    assert evidence == {"text", "visual"}


def test_weights_default_to_settings_and_can_be_overridden(monkeypatch):
    from app.core.config import settings

    _stub_modalities(monkeypatch, [_text_hit("c0", 0, 5, 0.8)], [_frame_hit("f0", 99.0, 0.3)])

    default = retrieval_service.search_multimodal("q")
    assert default.text_weight == settings.multimodal_text_weight
    assert default.visual_weight == settings.multimodal_visual_weight

    visual_heavy = retrieval_service.search_multimodal("q", text_weight=0.0, visual_weight=1.0)
    assert visual_heavy.results[0].evidence_type == "visual"
    assert visual_heavy.results[0].final_score == 1.0
    assert visual_heavy.results[1].final_score == 0.0


def test_negative_weights_are_rejected(monkeypatch):
    _stub_modalities(monkeypatch, [], [])
    with pytest.raises(ValueError, match="non-negative"):
        retrieval_service.search_multimodal("q", text_weight=-0.1)


def test_top_k_truncates_fused_results_and_candidate_pool_is_wider(monkeypatch):
    from app.core.config import settings

    captured = {}

    def fake_search(query, top_k=None, video_id=None, **kw):
        captured["text_k"] = top_k
        return SearchResponse(query=query, results=[_text_hit(f"c{i}", i * 10, i * 10 + 5, 0.9 - i * 0.05) for i in range(6)])

    monkeypatch.setattr(retrieval_service, "search", fake_search)
    monkeypatch.setattr(
        retrieval_service, "search_frames",
        lambda query, top_k=None, video_id=None: VisualSearchResponse(query=query, results=[]),
    )
    response = retrieval_service.search_multimodal("q", top_k=2)
    assert len(response.results) == 2
    assert captured["text_k"] == max(settings.multimodal_candidate_k, 2)


def test_empty_index_returns_empty_results(monkeypatch):
    _stub_modalities(monkeypatch, [], [])
    assert retrieval_service.search_multimodal("q").results == []


# ---------------------------------------------------------------------
# Cross-encoder reranking inside text search
# ---------------------------------------------------------------------

def _three_dense_hits():
    return [
        {"id": f"v1_000{i}", "document": f"text {i}",
         "metadata": {"video_id": "v1", "start_time": float(i), "end_time": float(i + 1)},
         "score": 0.9 - 0.1 * i}
        for i in range(3)
    ]


@pytest.fixture
def rerank_on(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "retrieval_rerank", True)
    monkeypatch.setattr(settings, "rerank_candidates", 10)
    monkeypatch.setattr(
        embedding_service, "embed_texts", lambda texts: np.array([[1.0, 0.0]], dtype=np.float32)
    )


def test_rerank_reorders_results_and_exposes_both_scores(monkeypatch, rerank_on):
    import math
    from app.services import rerank_service

    monkeypatch.setattr(vector_store_service, "query", lambda **kw: _three_dense_hits())
    # The cross-encoder disagrees with the embedding order: it likes chunk 2 best.
    monkeypatch.setattr(
        rerank_service, "rerank_scores",
        lambda query, passages: np.array([-1.0, 0.5, 3.0], dtype=np.float32),
    )

    response = retrieval_service.search("q", top_k=3)

    assert [r.chunk_id for r in response.results] == ["v1_0002", "v1_0001", "v1_0000"]
    best = response.results[0]
    assert best.rerank_score == pytest.approx(3.0)
    from app.core.config import settings

    assert best.score == pytest.approx(1 / (1 + math.exp(-3.0 / settings.rerank_score_temperature)))
    # Temperature keeps scores graded instead of pinned to ~0/~1.
    assert 0.05 < response.results[-1].score < response.results[0].score < 0.95
    scores = [r.score for r in response.results]
    assert scores == sorted(scores, reverse=True)  # score stays monotone with rank


def test_rerank_retrieves_a_wider_shortlist_then_truncates_to_top_k(monkeypatch, rerank_on):
    from app.services import rerank_service

    seen = {}

    def fake_query(collection_name, query_embedding, top_k, video_id=None):
        seen["retrieve_k"] = top_k
        return _three_dense_hits()

    monkeypatch.setattr(vector_store_service, "query", fake_query)
    monkeypatch.setattr(
        rerank_service, "rerank_scores",
        lambda query, passages: np.array([0.0, 1.0, 2.0], dtype=np.float32),
    )
    response = retrieval_service.search("q", top_k=2)

    assert seen["retrieve_k"] == 10  # max(top_k, rerank_candidates)
    assert len(response.results) == 2


def test_no_rerank_keeps_dense_order_and_never_touches_the_reranker(monkeypatch):
    from app.core.config import settings
    from app.services import rerank_service

    assert settings.retrieval_rerank is False  # conftest default
    monkeypatch.setattr(
        embedding_service, "embed_texts", lambda texts: np.array([[1.0, 0.0]], dtype=np.float32)
    )
    seen = {}

    def fake_query(collection_name, query_embedding, top_k, video_id=None):
        seen["retrieve_k"] = top_k
        return _three_dense_hits()

    monkeypatch.setattr(vector_store_service, "query", fake_query)

    def boom(*a, **k):
        raise AssertionError("reranker must not run when retrieval_rerank is off")

    monkeypatch.setattr(rerank_service, "rerank_scores", boom)
    response = retrieval_service.search("q", top_k=3)

    assert seen["retrieve_k"] == 3
    assert [r.chunk_id for r in response.results] == ["v1_0000", "v1_0001", "v1_0002"]
    assert all(r.rerank_score is None for r in response.results)
    assert response.results[0].score == pytest.approx(0.9)


def test_rerank_is_skipped_when_nothing_was_retrieved(monkeypatch, rerank_on):
    from app.services import rerank_service

    monkeypatch.setattr(vector_store_service, "query", lambda **kw: [])

    def boom(*a, **k):
        raise AssertionError("nothing to rerank")

    monkeypatch.setattr(rerank_service, "rerank_scores", boom)
    assert retrieval_service.search("q").results == []


# ---------------------------------------------------------------------
# Re-indexing replaces a video's records instead of merging into them
# ---------------------------------------------------------------------

def test_index_video_deletes_the_videos_old_chunks_before_upserting(monkeypatch):
    from app.core.config import settings

    calls = []
    monkeypatch.setattr(
        embedding_service, "load_embeddings",
        lambda video_id: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        vector_store_service, "delete_video",
        lambda collection_name, video_id: calls.append(("delete", collection_name, video_id)),
    )
    monkeypatch.setattr(
        vector_store_service, "upsert",
        lambda collection_name, ids, embeddings, documents, metadatas: calls.append(("upsert", collection_name)) or len(ids),
    )
    retrieval_service.index_video(_chunk_set(), _embedding_meta())

    assert calls[0] == ("delete", settings.chroma_collection_name, "v1")  # delete FIRST
    assert calls[1][0] == "upsert"


def test_index_video_frames_deletes_the_videos_old_frames_before_upserting(monkeypatch):
    from app.core.config import settings

    calls = []
    monkeypatch.setattr(
        vision_service, "load_frame_embeddings",
        lambda video_id: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        vector_store_service, "delete_video",
        lambda collection_name, video_id: calls.append(("delete", collection_name, video_id)),
    )
    monkeypatch.setattr(
        vector_store_service, "upsert",
        lambda collection_name, ids, embeddings, documents, metadatas: calls.append(("upsert", collection_name)) or len(ids),
    )
    retrieval_service.index_video_frames(_frame_set(), _frame_embedding_meta())

    assert calls[0] == ("delete", settings.chroma_frames_collection_name, "v1")
    assert calls[1][0] == "upsert"


# ---------------------------------------------------------------------
# Citation localization: land on the best SEGMENT inside the chunk
# ---------------------------------------------------------------------

def _transcript_with_segments(starts_texts, video_id="v1"):
    from datetime import datetime, timezone
    from app.models.schemas import Transcript, TranscriptSegment

    segments = [
        TranscriptSegment(segment_id=i, start_time=float(st), end_time=float(st + 4), text=tx)
        for i, (st, tx) in enumerate(starts_texts)
    ]
    return Transcript(
        video_id=video_id, language="en", duration_seconds=segments[-1].end_time, model_size="base",
        created_at=datetime.now(timezone.utc), segments=segments,
    )


def _result(start=10.0, end=30.0):
    return SearchResult(video_id="v1", chunk_id="v1_0000", text="chunk", start_time=start, end_time=end, score=0.5)


def test_localize_returns_the_start_of_the_best_scoring_segment_inside_the_chunk(monkeypatch):
    from app.services import rerank_service, transcription_service

    transcript = _transcript_with_segments([(0, "before"), (10, "intro"), (15, "the answer"), (20, "more"), (35, "after")])
    monkeypatch.setattr(transcription_service, "get_transcript", lambda vid: transcript)
    seen = {}

    def fake_scores(query, passages):
        seen["passages"] = passages
        return np.array([0.1, 2.0, 0.3], dtype=np.float32)  # best = the middle segment

    monkeypatch.setattr(rerank_service, "rerank_scores", fake_scores)
    assert retrieval_service.localize("q", _result(10, 30)) == 15.0
    assert seen["passages"] == ["intro", "the answer", "more"]  # only segments inside the chunk


def test_localize_without_a_transcript_or_segments_returns_none(monkeypatch):
    from app.services import transcription_service

    monkeypatch.setattr(transcription_service, "get_transcript", lambda vid: None)
    assert retrieval_service.localize("q", _result()) is None

    monkeypatch.setattr(transcription_service, "get_transcript", lambda vid: _transcript_with_segments([(100, "elsewhere")]))
    assert retrieval_service.localize("q", _result(10, 30)) is None


def test_localize_single_segment_chunk_skips_the_reranker(monkeypatch):
    from app.services import rerank_service, transcription_service

    monkeypatch.setattr(transcription_service, "get_transcript", lambda vid: _transcript_with_segments([(10, "only")]))

    def boom(*a, **k):
        raise AssertionError("nothing to choose between")

    monkeypatch.setattr(rerank_service, "rerank_scores", boom)
    assert retrieval_service.localize("q", _result(10, 20)) == 10.0


def test_search_attaches_jump_time_to_the_top_results_only(monkeypatch, rerank_on):
    from app.core.config import settings
    from app.services import rerank_service

    monkeypatch.setattr(settings, "localize_top_n", 2)
    monkeypatch.setattr(vector_store_service, "query", lambda **kw: _three_dense_hits())
    monkeypatch.setattr(rerank_service, "rerank_scores", lambda q, p: np.array([3.0, 2.0, 1.0][: len(p)], dtype=np.float32))
    monkeypatch.setattr(retrieval_service, "localize", lambda q, r: r.start_time + 0.5)

    results = retrieval_service.search("q", top_k=3).results
    assert [r.jump_time for r in results] == [results[0].start_time + 0.5, results[1].start_time + 0.5, None]


def test_search_can_skip_or_disable_localization_and_survive_its_failure(monkeypatch, rerank_on):
    from app.core.config import settings
    from app.services import rerank_service

    monkeypatch.setattr(vector_store_service, "query", lambda **kw: _three_dense_hits())
    monkeypatch.setattr(rerank_service, "rerank_scores", lambda q, p: np.array([3.0, 2.0, 1.0][: len(p)], dtype=np.float32))

    def boom(q, r):
        raise RuntimeError("segment scoring exploded")

    monkeypatch.setattr(retrieval_service, "localize", boom)
    # a failure inside localization must not break search -- it just leaves jump_time unset
    assert all(r.jump_time is None for r in retrieval_service.search("q", top_k=3).results)

    monkeypatch.setattr(retrieval_service, "localize", lambda q, r: 99.0)
    assert all(r.jump_time is None for r in retrieval_service.search("q", top_k=3, localize_results=False).results)
    monkeypatch.setattr(settings, "localize_citations", False)
    assert all(r.jump_time is None for r in retrieval_service.search("q", top_k=3).results)
