"""
Semantic search: the domain-level operation of "given a question, find
the most relevant chunks (or frames)" -- independent of HTTP and
independent of which vector database backs it.

This is intentionally a thin layer: embed the query the same way the
target was embedded (MiniLM for chunks, Phase 2; CLIP for frames,
Phase 5), ask the vector store for nearest neighbors, and shape the
result. vector_store_service returns generic {id, document, metadata,
score} dicts -- interpreting those into a chunk-shaped or frame-shaped
result happens here, split into two clearly separate sections below.
No LLM involved yet -- that's Phase 4 (rag_service builds on `search()`).
"""

import math
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    ChunkSet,
    EmbeddingSet,
    FrameIndexResult,
    FrameSearchResult,
    FrameSet,
    FrameEmbeddingSet,
    IndexResult,
    MultimodalResult,
    MultimodalSearchResponse,
    SearchResponse,
    SearchResult,
    VisualSearchResponse,
)
from app.services import embedding_service, rerank_service, vector_store_service, vision_service

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Text (transcript chunks) -- Phase 2/3
# ---------------------------------------------------------------------

def index_video(chunk_set: ChunkSet, embedding_meta: EmbeddingSet) -> IndexResult:
    """
    Push a video's chunks + their precomputed embeddings into the
    vector store.

    Requires chunk_set and embedding_meta to agree on chunk order --
    they're both derived from the same chunking run, but we check
    explicitly rather than trusting that silently, per the project's
    "fail loudly, don't index misaligned data" principle.
    """
    chunk_ids_from_chunks = [c.chunk_id for c in chunk_set.chunks]
    if chunk_ids_from_chunks != embedding_meta.chunk_ids:
        raise ValueError(
            f"Chunk/embedding mismatch for video '{chunk_set.video_id}': "
            "chunks and embeddings were computed from different chunking "
            "runs. Re-run chunking and embedding together before indexing."
        )

    vectors = embedding_service.load_embeddings(chunk_set.video_id)
    if vectors is None or len(vectors) != len(chunk_set.chunks):
        raise ValueError(
            f"No usable embeddings found on disk for video '{chunk_set.video_id}'."
        )

    documents = [c.text for c in chunk_set.chunks]
    metadatas = [
        {
            "video_id": c.video_id,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "chunk_index": c.chunk_index,
        }
        for c in chunk_set.chunks
    ]

    # Replace, don't merge: drop this video's old records first, so a re-run
    # that produced fewer chunks leaves no stale ones behind.
    vector_store_service.delete_video(settings.chroma_collection_name, chunk_set.video_id)
    count = vector_store_service.upsert(
        collection_name=settings.chroma_collection_name,
        ids=chunk_ids_from_chunks,
        embeddings=vectors.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    logger.info("Indexed %d chunks for video %s", count, chunk_set.video_id)
    return IndexResult(
        video_id=chunk_set.video_id,
        chunks_indexed=count,
        collection_name=settings.chroma_collection_name,
    )


def search(query: str, top_k: Optional[int] = None, video_id: Optional[str] = None) -> SearchResponse:
    """
    Embed a natural-language query (via MiniLM) and return the most
    semantically similar indexed chunks, optionally restricted to one
    video.
    """
    resolved_top_k = top_k or settings.search_default_top_k
    query_embedding = embedding_service.embed_texts([query])[0]

    # With reranking, fetch a wider shortlist than asked for: the point is
    # to let the cross-encoder pull a good chunk up from just below the cut.
    retrieve_k = max(resolved_top_k, settings.rerank_candidates) if settings.retrieval_rerank else resolved_top_k
    raw_results = vector_store_service.query(
        collection_name=settings.chroma_collection_name,
        query_embedding=query_embedding.tolist(),
        top_k=retrieve_k,
        video_id=video_id,
    )

    results = [
        SearchResult(
            video_id=r["metadata"]["video_id"],
            chunk_id=r["id"],
            text=r["document"],
            start_time=r["metadata"]["start_time"],
            end_time=r["metadata"]["end_time"],
            score=r["score"],
        )
        for r in raw_results
    ]

    if settings.retrieval_rerank and results:
        logits = rerank_service.rerank_scores(query, [r.text for r in results])
        order = sorted(range(len(results)), key=lambda i: -float(logits[i]))
        results = [
            results[i].model_copy(
                update={
                    "score": _sigmoid(float(logits[i]) / settings.rerank_score_temperature),
                    "rerank_score": float(logits[i]),
                }
            )
            for i in order
        ]
    return SearchResponse(query=query, results=results[:resolved_top_k])


def _sigmoid(x: float) -> float:
    """Squash a cross-encoder logit into (0, 1), keeping rank order."""
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------
# Visual (video frames) -- Phase 5, Step 3
# ---------------------------------------------------------------------

def index_video_frames(frame_set: FrameSet, frame_embedding_meta: FrameEmbeddingSet) -> FrameIndexResult:
    """
    Push a video's frames + their precomputed CLIP embeddings into the
    (separate) frames vector store collection. Mirrors index_video()
    above -- same "verify chunk/embedding order matches" discipline,
    applied to frames.
    """
    frame_ids_from_frames = [f.frame_id for f in frame_set.frames]
    if frame_ids_from_frames != frame_embedding_meta.frame_ids:
        raise ValueError(
            f"Frame/embedding mismatch for video '{frame_set.video_id}': "
            "frames and embeddings were computed from different "
            "extraction runs. Re-run frame extraction and embedding "
            "together before indexing."
        )

    vectors = vision_service.load_frame_embeddings(frame_set.video_id)
    if vectors is None or len(vectors) != len(frame_set.frames):
        raise ValueError(
            f"No usable frame embeddings found on disk for video '{frame_set.video_id}'."
        )

    # No natural "document" text for a frame -- store the image path so
    # a raw Chroma inspection is still human-readable; retrieval code
    # reads image_path from metadata, not from this field.
    documents = [f.image_path for f in frame_set.frames]
    metadatas = [
        {
            "video_id": f.video_id,
            "timestamp": f.timestamp,
            "frame_index": f.frame_index,
            "image_path": f.image_path,
        }
        for f in frame_set.frames
    ]

    # Same replace-not-merge rule as index_video(): re-extracting frames with
    # a stricter scene threshold yields fewer frames, and the old frame
    # records (whose image files are gone) must not linger in search results.
    vector_store_service.delete_video(settings.chroma_frames_collection_name, frame_set.video_id)
    count = vector_store_service.upsert(
        collection_name=settings.chroma_frames_collection_name,
        ids=frame_ids_from_frames,
        embeddings=vectors.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    logger.info("Indexed %d frames for video %s", count, frame_set.video_id)
    return FrameIndexResult(
        video_id=frame_set.video_id,
        frames_indexed=count,
        collection_name=settings.chroma_frames_collection_name,
    )


def search_frames(
    query: str, top_k: Optional[int] = None, video_id: Optional[str] = None
) -> VisualSearchResponse:
    """
    Embed a natural-language query through CLIP's TEXT encoder (not
    MiniLM -- a completely different, non-comparable vector space; see
    docs/concepts.md) and return the most visually-matching indexed
    frames, optionally restricted to one video.
    """
    resolved_top_k = top_k or settings.search_default_top_k
    query_embedding = vision_service.embed_text_for_visual(query)

    raw_results = vector_store_service.query(
        collection_name=settings.chroma_frames_collection_name,
        query_embedding=query_embedding.tolist(),
        top_k=resolved_top_k,
        video_id=video_id,
    )

    results = [
        FrameSearchResult(
            video_id=r["metadata"]["video_id"],
            frame_id=r["id"],
            timestamp=r["metadata"]["timestamp"],
            image_path=r["metadata"]["image_path"],
            score=r["score"],
        )
        for r in raw_results
    ]
    return VisualSearchResponse(query=query, results=results)


# ---------------------------------------------------------------------
# Multimodal fusion -- Phase 6
# ---------------------------------------------------------------------

def _min_max_normalize(scores: list[float]) -> list[float]:
    """
    Rescale a list of scores to [0, 1] relative to ITS OWN min and max.

    Needed because MiniLM and CLIP cosine scores live on different
    natural scales (a good text match is ~0.7-0.85, a good CLIP match is
    only ~0.25-0.35): summing raw scores would let text silently win
    whatever the weights say. When every score is identical (including
    a single result) there is no spread to rescale, so each gets 1.0 --
    "the best this modality found" -- rather than 0/0.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def search_multimodal(
    query: str,
    top_k: Optional[int] = None,
    video_id: Optional[str] = None,
    text_weight: Optional[float] = None,
    visual_weight: Optional[float] = None,
) -> MultimodalSearchResponse:
    """
    Fuse transcript search and visual search into one ranked list of
    moments in a video.

    1. Pull `multimodal_candidate_k` candidates from EACH modality
       (MiniLM over chunks, CLIP over frames).
    2. Min-max normalize each modality's scores within its own list.
    3. Align by time: a frame whose timestamp falls inside a chunk's
       [start_time, end_time] (same video) is evidence for the SAME
       moment, so they merge into one "both" result carrying the chunk's
       text plus its best-matching frame. Unmatched chunks stay "text";
       unmatched frames stay "visual" (start == end == frame timestamp).
    4. final_score = text_weight * text_score + visual_weight * visual_score.

    A moment backed by both modalities can therefore outrank a stronger
    single-modality match -- that agreement is the point of fusion.
    Fusion only sees frames among the visual candidates: a frame that
    would overlap a retrieved chunk but wasn't itself retrieved is not
    pulled in (keeps this a rank-fusion, not a second lookup).
    """
    resolved_top_k = top_k or settings.search_default_top_k
    tw = settings.multimodal_text_weight if text_weight is None else text_weight
    vw = settings.multimodal_visual_weight if visual_weight is None else visual_weight
    if tw < 0 or vw < 0:
        raise ValueError("Fusion weights must be non-negative.")

    candidate_k = max(settings.multimodal_candidate_k, resolved_top_k)
    text_hits = search(query, top_k=candidate_k, video_id=video_id).results
    frame_hits = search_frames(query, top_k=candidate_k, video_id=video_id).results

    text_scores = _min_max_normalize([h.score for h in text_hits])
    frame_scores = _min_max_normalize([f.score for f in frame_hits])

    fused: list[MultimodalResult] = []
    matched_frame_ids: set[str] = set()

    for hit, t_score in zip(text_hits, text_scores):
        overlapping = [
            (frame, f_score)
            for frame, f_score in zip(frame_hits, frame_scores)
            if frame.video_id == hit.video_id and hit.start_time <= frame.timestamp <= hit.end_time
        ]
        if overlapping:
            best_frame, best_f_score = max(overlapping, key=lambda pair: pair[1])
            matched_frame_ids.update(frame.frame_id for frame, _ in overlapping)
            fused.append(
                MultimodalResult(
                    video_id=hit.video_id,
                    start_time=hit.start_time,
                    end_time=hit.end_time,
                    evidence_type="both",
                    text=hit.text,
                    chunk_id=hit.chunk_id,
                    frame_id=best_frame.frame_id,
                    image_path=best_frame.image_path,
                    text_score=t_score,
                    visual_score=best_f_score,
                    final_score=tw * t_score + vw * best_f_score,
                )
            )
        else:
            fused.append(
                MultimodalResult(
                    video_id=hit.video_id,
                    start_time=hit.start_time,
                    end_time=hit.end_time,
                    evidence_type="text",
                    text=hit.text,
                    chunk_id=hit.chunk_id,
                    text_score=t_score,
                    final_score=tw * t_score,
                )
            )

    for frame, f_score in zip(frame_hits, frame_scores):
        if frame.frame_id in matched_frame_ids:
            continue
        fused.append(
            MultimodalResult(
                video_id=frame.video_id,
                start_time=frame.timestamp,
                end_time=frame.timestamp,
                evidence_type="visual",
                frame_id=frame.frame_id,
                image_path=frame.image_path,
                visual_score=f_score,
                final_score=vw * f_score,
            )
        )

    # Stable sort: ties keep text-before-frames order, and "both" wins a
    # tie against single-modality evidence.
    fused.sort(key=lambda r: (-r.final_score, r.evidence_type != "both"))
    return MultimodalSearchResponse(
        query=query, text_weight=tw, visual_weight=vw, results=fused[:resolved_top_k]
    )
