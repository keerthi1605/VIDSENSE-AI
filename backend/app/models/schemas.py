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
