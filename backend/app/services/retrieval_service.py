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
    SearchResponse,
    SearchResult,
    VisualSearchResponse,
)
from app.services import embedding_service, vector_store_service, vision_service

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

    raw_results = vector_store_service.query(
        collection_name=settings.chroma_collection_name,
        query_embedding=query_embedding.tolist(),
        top_k=resolved_top_k,
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
    return SearchResponse(query=query, results=results)


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
