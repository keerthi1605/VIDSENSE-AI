"""
Text embeddings via Sentence-Transformers.

Same lazy-singleton pattern as transcription_service's Whisper model:
loading the model (reading weights + building the network) takes real
time, so we pay that cost once per process, not once per request.

Vectors are stored as a compact binary .npy file, NOT as JSON -- a
150-chunk video at 384 dimensions is ~230KB as float32 binary but
would be several times larger (and much slower to parse) as a JSON
array of floats. A small JSON sidecar (EmbeddingSet) carries the
metadata needed to make sense of that binary blob: which model made
it, and which chunk_id each row corresponds to.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import ChunkSet, EmbeddingSet

logger = get_logger(__name__)

_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(
            "Loading embedding model '%s' (device=%s)...",
            settings.embedding_model_name,
            settings.embedding_device,
        )
        _model = SentenceTransformer(
            settings.embedding_model_name, device=settings.embedding_device
        )
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts into an (n, dimension) float32 array.

    normalize_embeddings=True L2-normalizes every vector to unit
    length. Sentence-transformer models are trained/evaluated using
    cosine similarity, and for unit-length vectors cosine similarity
    reduces to a plain dot product -- cheaper to compute, and what
    vector databases (Phase 3) optimize for internally.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors: the cosine of the angle
    between them, ranging from -1 (opposite) to 1 (identical
    direction), independent of their magnitudes.

    Computed generally (not assuming pre-normalized input) so this
    function is correct on its own -- but note that for the vectors
    this service actually produces (already unit-length), this is
    equivalent to a plain np.dot(a, b).
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _vectors_path(video_id: str) -> Path:
    return settings.embeddings_dir / f"{video_id}.npy"


def _metadata_path(video_id: str) -> Path:
    return settings.embeddings_dir / f"{video_id}.json"


def embed_and_save(chunk_set: ChunkSet) -> EmbeddingSet:
    """Embed every chunk in a ChunkSet and persist the result."""
    texts = [chunk.text for chunk in chunk_set.chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunk_set.chunks]

    if not texts:
        vectors = np.empty((0, 0), dtype=np.float32)
        dimension = 0
    else:
        vectors = embed_texts(texts)
        dimension = int(vectors.shape[1])

    np.save(_vectors_path(chunk_set.video_id), vectors)

    metadata = EmbeddingSet(
        video_id=chunk_set.video_id,
        model_name=settings.embedding_model_name,
        dimension=dimension,
        chunk_count=len(chunk_ids),
        created_at=datetime.now(timezone.utc),
        chunk_ids=chunk_ids,
    )
    _metadata_path(chunk_set.video_id).write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )

    logger.info(
        "Embedded %s: %d chunks -> %d-dim vectors (model=%s)",
        chunk_set.video_id,
        len(chunk_ids),
        dimension,
        settings.embedding_model_name,
    )
    return metadata


def get_embedding_metadata(video_id: str) -> Optional[EmbeddingSet]:
    path = _metadata_path(video_id)
    if not path.exists():
        return None
    return EmbeddingSet.model_validate_json(path.read_text(encoding="utf-8"))


def load_embeddings(video_id: str) -> Optional[np.ndarray]:
    """Load the raw vector matrix for a video, for internal use by
    retrieval code (Phase 3) -- never exposed directly over the API."""
    path = _vectors_path(video_id)
    if not path.exists():
        return None
    return np.load(path)
