"""
Pydantic schemas shared across the API and services.

These serve double duty right now: they're both the FastAPI response
model (controls the JSON shape returned to clients) AND the on-disk
JSON shape we persist to `storage/videos/{video_id}.json`. That's a
deliberate simplification for the filesystem+JSON storage stage --
once PostgreSQL is introduced (Phase 9), the DB row and the API
response model will likely diverge and we'll split them.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class VideoMetadata(BaseModel):
    video_id: str
    original_filename: str
    stored_filename: str
    extension: str
    content_type: Optional[str] = None
    size_bytes: int
    uploaded_at: datetime
    # Simple string for now. Becomes a real enum
    # (UPLOADED/PROCESSING_AUDIO/TRANSCRIBING/INDEXING/READY/FAILED)
    # once background processing exists (Phase 9).
    status: str = "uploaded"


class TranscribeRequest(BaseModel):
    video_id: str


class TranscriptSegment(BaseModel):
    """
    One chunk of spoken text with its exact position in the source video.

    start_time/end_time are in seconds, as floats (Whisper's native
    unit). Every later phase (chunking, embedding, retrieval, RAG
    citations, frontend "jump to video") depends on these surviving
    unchanged -- this is the load-bearing field of the whole project.
    """

    segment_id: int
    start_time: float
    end_time: float
    text: str


class Transcript(BaseModel):
    video_id: str
    language: str
    duration_seconds: float
    model_size: str
    created_at: datetime
    segments: List[TranscriptSegment]


class Chunk(BaseModel):
    """
    A group of consecutive transcript segments, sized for embedding.

    chunk_id is globally unique (prefixed with video_id) and doubles as
    the record ID we'll hand to the vector database in Phase 3 -- no
    separate ID scheme needed there.

    start_time/end_time come directly from the first/last merged
    segment's real timestamps (never interpolated), per the project's
    timestamp-preservation principle.
    """

    chunk_id: str
    video_id: str
    chunk_index: int
    start_time: float
    end_time: float
    text: str
    word_count: int


class ChunkSet(BaseModel):
    """The full set of chunks for one video, plus the settings used to
    produce them -- useful for debugging "why did chunking look like
    this" without cross-referencing config history."""

    video_id: str
    created_at: datetime
    chunk_target_words: int
    chunk_overlap_segments: int
    chunks: List[Chunk]


class EmbeddingSet(BaseModel):
    """
    Metadata describing a video's stored embedding vectors.

    Deliberately does NOT include the vectors themselves -- returning
    e.g. 150 x 384 floats over HTTP is both large and not actually
    useful to a client; the vectors live in a `.npy` file on disk and
    are loaded directly by retrieval code (Phase 3), not fetched via
    this API. `chunk_ids[i]` names which chunk row `i` of that array
    corresponds to.
    """

    video_id: str
    model_name: str
    dimension: int
    chunk_count: int
    created_at: datetime
    chunk_ids: List[str]
