"""
Visual embeddings via CLIP.

CLIP embeds images AND text into the SAME vector space -- that joint
space is the entire reason CLIP was chosen (see config.py's comment on
clip_model_name). Two functions matter here:

  embed_images()            -- frame JPEGs -> vectors, run once per
                                video during indexing (this module).
  embed_text_for_visual()   -- a user's text query -> a vector in the
                                SAME space, used by Phase 6's visual
                                search. Included now (not deferred)
                                because it's what proves CLIP's joint-
                                space property actually holds -- see
                                test_vision_service.py's real-model test.

Same lazy-singleton pattern as transcription_service and
embedding_service: the model loads once per process, not per request.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import FrameEmbeddingSet, FrameSet

logger = get_logger(__name__)

_model: Optional[SentenceTransformer] = None


def get_clip_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(
            "Loading CLIP model '%s' (device=%s)...", settings.clip_model_name, settings.clip_device
        )
        _model = SentenceTransformer(settings.clip_model_name, device=settings.clip_device)
        logger.info("CLIP model loaded.")
    return _model


def embed_images(image_paths: List[Path]) -> np.ndarray:
    """Embed a list of image files into CLIP's joint space."""
    model = get_clip_model()
    images = [Image.open(p).convert("RGB") for p in image_paths]
    embeddings = model.encode(
        images,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def embed_text_for_visual(text: str) -> np.ndarray:
    """
    Embed a text query into CLIP's joint space -- for comparing against
    frame image vectors (visual search, Phase 6). This is NOT the same
    embedding as embedding_service.embed_texts(): that's MiniLM's
    384-dim text-only space, used for matching against transcript
    chunks. This is CLIP's 512-dim joint space, used for matching
    against frame images. Never mix vectors from the two spaces.
    """
    model = get_clip_model()
    embedding = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return embedding.astype(np.float32)


def _vectors_path(video_id: str) -> Path:
    return settings.embeddings_dir / f"{video_id}_frames.npy"


def _metadata_path(video_id: str) -> Path:
    return settings.embeddings_dir / f"{video_id}_frames.json"


def embed_and_save_frames(frame_set: FrameSet) -> FrameEmbeddingSet:
    """Embed every frame in a FrameSet and persist the result."""
    frame_dir = settings.frames_dir / frame_set.video_id
    image_paths = [frame_dir / frame.image_path for frame in frame_set.frames]
    frame_ids = [frame.frame_id for frame in frame_set.frames]

    if not image_paths:
        vectors = np.empty((0, 0), dtype=np.float32)
        dimension = 0
    else:
        vectors = embed_images(image_paths)
        dimension = int(vectors.shape[1])

    np.save(_vectors_path(frame_set.video_id), vectors)

    metadata = FrameEmbeddingSet(
        video_id=frame_set.video_id,
        model_name=settings.clip_model_name,
        dimension=dimension,
        frame_count=len(frame_ids),
        created_at=datetime.now(timezone.utc),
        frame_ids=frame_ids,
    )
    _metadata_path(frame_set.video_id).write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )

    logger.info(
        "Embedded %s: %d frames -> %d-dim vectors (model=%s)",
        frame_set.video_id,
        len(frame_ids),
        dimension,
        settings.clip_model_name,
    )
    return metadata


def get_frame_embedding_metadata(video_id: str) -> Optional[FrameEmbeddingSet]:
    path = _metadata_path(video_id)
    if not path.exists():
        return None
    return FrameEmbeddingSet.model_validate_json(path.read_text(encoding="utf-8"))


def load_frame_embeddings(video_id: str) -> Optional[np.ndarray]:
    """Load the raw frame-vector matrix for a video, for internal use
    by retrieval code (Phase 6) -- never exposed directly over the API."""
    path = _vectors_path(video_id)
    if not path.exists():
        return None
    return np.load(path)
