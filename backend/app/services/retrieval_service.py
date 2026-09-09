"""
Semantic search: the domain-level operation of "given a question, find
the most relevant chunks" -- independent of HTTP and independent of
which vector database backs it.

This is intentionally a thin layer: embed the query the same way
chunks were embedded (Phase 2), ask the vector store for nearest
neighbors (Phase 3), and shape the result. No LLM involved yet --
that's Phase 4.
"""

from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import ChunkSet, EmbeddingSet, IndexResult, SearchResponse, SearchResult
from app.services import embedding_service, vector_store_service

logger = get_logger(__name__)


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

    count = vector_store_service.upsert_chunks(
        chunk_ids=chunk_ids_from_chunks,
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
    Embed a natural-language query and return the most semantically
    similar indexed chunks, optionally restricted to one video.
    """
    resolved_top_k = top_k or settings.search_default_top_k
    query_embedding = embedding_service.embed_texts([query])[0]

    raw_results = vector_store_service.query(
        query_embedding=query_embedding.tolist(),
        top_k=resolved_top_k,
        video_id=video_id,
    )

    results = [
        SearchResult(
            video_id=r["video_id"],
            chunk_id=r["chunk_id"],
            text=r["text"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            score=r["score"],
        )
        for r in raw_results
    ]
    return SearchResponse(query=query, results=results)
