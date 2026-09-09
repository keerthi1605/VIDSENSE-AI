"""
Vector database access, via ChromaDB.

This module is the ONLY place that imports chromadb. Everything above
it (retrieval_service, the API layer) talks to plain Python types --
so swapping the vector database later (e.g. to FAISS) means rewriting
this one file, not chasing chromadb-specific calls through the codebase.
That's the "model abstraction" principle applied to the vector DB itself.

We always pass our own precomputed embeddings (from embedding_service)
rather than letting Chroma embed text itself -- see this module's
design note in the Phase 3 completion report for why.
"""

from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None
_collection = None


def get_collection():
    """
    Lazy-loaded singleton ChromaDB collection, configured for cosine
    distance.

    Chroma's default distance metric is squared-L2, not cosine. Since
    our vectors are already unit-normalized (Phase 2), we explicitly
    request "cosine" here -- otherwise search results would be ranked
    by a different metric than the one we've validated since Phase 2.
    """
    global _client, _collection
    if _collection is None:
        logger.info("Opening ChromaDB at %s", settings.chroma_persist_dir)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            # Disable anonymized usage telemetry -- no reason to phone
            # home for a local project database.
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_chunks(
    chunk_ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict],
) -> int:
    """
    Insert or update chunk records in the vector store.

    `upsert` (not `add`) so re-indexing an already-indexed video
    updates its existing records instead of erroring on duplicate IDs
    -- chunk_id is deterministic (video_id + index), so re-running the
    pipeline on the same video should just overwrite, not fail.
    """
    if not chunk_ids:
        return 0
    collection = get_collection()
    collection.upsert(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunk_ids)


def query(
    query_embedding: List[float],
    top_k: int,
    video_id: Optional[str] = None,
) -> List[Dict]:
    """
    Find the top_k chunks most similar to a query embedding.

    Returns a list of dicts: {chunk_id, text, video_id, start_time,
    end_time, score}, already sorted best-first (Chroma returns
    results in distance-ascending order, which is score-descending).

    `score` is 1 - cosine_distance, matching the "higher is better"
    convention used everywhere else in this codebase (Phase 2's
    cosine_similarity()) rather than Chroma's raw distance.
    """
    collection = get_collection()
    where = {"video_id": video_id} if video_id else None

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {
            "chunk_id": chunk_id,
            "text": document,
            "video_id": metadata["video_id"],
            "start_time": metadata["start_time"],
            "end_time": metadata["end_time"],
            "score": 1.0 - distance,
        }
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
    ]


def count() -> int:
    """Total number of chunks currently indexed, across all videos."""
    return get_collection().count()
