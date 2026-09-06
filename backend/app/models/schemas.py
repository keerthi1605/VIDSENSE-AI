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
from typing import Optional

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
