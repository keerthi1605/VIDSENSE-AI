"""
Vector database access, via ChromaDB.

This module is the ONLY place that imports chromadb. Everything above
it (retrieval_service, the API layer) talks to plain Python types --
so swapping the vector database later (e.g. to FAISS) means rewriting
this one file, not chasing chromadb-specific calls through the codebase.
That's the "model abstraction" principle applied to the vector DB itself.

We always pass our own precomputed embeddings (from embedding_service /
vision_service) rather than letting Chroma embed things itself -- see
this module's design note in the Phase 3 completion report for why.

COLLECTION-AGNOSTIC BY DESIGN (Phase 5, Step 3): text chunks (384-dim,
MiniLM) and video frames (512-dim, CLIP) live in separate collections --
Chroma collections are single-dimension, and the two vector spaces are
not comparable anyway (see docs/concepts.md). Every function here takes
an explicit `collection_name` and returns generic {id, document,
metadata, score} shapes; it's retrieval_service's job to interpret those
generic fields into a chunk-shaped or frame-shaped result, not this
module's -- this file has no idea what a "chunk" or "frame" is.
"""

from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None
_collections: Dict[str, "chromadb.Collection"] = {}


def get_collection(collection_name: str):
    """
    Lazy-loaded, cached-per-name ChromaDB collection, configured for
    cosine distance.

    Chroma's default distance metric is squared-L2, not cosine. Since
    our vectors are already unit-normalized (Phase 2/5), we explicitly
    request "cosine" here -- otherwise search results would be ranked
    by a different metric than the one our embeddings were validated
    against.
    """
    global _client
    if _client is None:
        logger.info("Opening ChromaDB at %s", settings.chroma_persist_dir)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            # Disable anonymized usage telemetry -- no reason to phone
            # home for a local project database.
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    if collection_name not in _collections:
        _collections[collection_name] = _client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[collection_name]


def upsert(
    collection_name: str,
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict],
) -> int:
    """
    Insert or update records in a collection.

    `upsert` (not `add`) so re-indexing an already-indexed video
    updates its existing records instead of erroring on duplicate IDs
    -- ids are deterministic (video_id + index), so re-running the
    pipeline on the same video should just overwrite, not fail.
    """
    if not ids:
        return 0
    collection = get_collection(collection_name)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(ids)


def query(
    collection_name: str,
    query_embedding: List[float],
    top_k: int,
    video_id: Optional[str] = None,
) -> List[Dict]:
    """
    Find the top_k records in a collection most similar to a query
    embedding.

    Returns generic dicts -- {id, document, metadata, score} -- already
    sorted best-first (Chroma returns results in distance-ascending
    order, which is score-descending). Interpreting `metadata` into a
    chunk-shaped or frame-shaped result is the caller's job.

    `score` is 1 - cosine_distance, matching the "higher is better"
    convention used everywhere else in this codebase (Phase 2's
    cosine_similarity()) rather than Chroma's raw distance.
    """
    collection = get_collection(collection_name)
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
            "id": record_id,
            "document": document,
            "metadata": metadata,
            "score": 1.0 - distance,
        }
        for record_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
    ]


def count(collection_name: str) -> int:
    """Total number of records currently indexed in a collection."""
    return get_collection(collection_name).count()
