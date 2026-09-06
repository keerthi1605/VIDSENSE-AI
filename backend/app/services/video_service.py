"""
Video upload handling: validation, safe storage, metadata persistence.

Kept independent of FastAPI's `Request`/`Response` types (only
`UploadFile` is used, which is just a thin file-like wrapper) so this
logic is testable and reusable without spinning up the web layer --
that's the "services stay separate from the API layer" principle.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import VideoMetadata

logger = get_logger(__name__)

# Read/write in 1 MB chunks so we never hold an entire upload in memory
# at once, and can abort mid-stream if the size cap is exceeded.
CHUNK_SIZE = 1024 * 1024


def _validate_extension(filename: str) -> str:
    """Return the lowercase extension if allowed, else raise 400."""
    extension = Path(filename).suffix.lower()
    if extension not in settings.allowed_video_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file extension '{extension}'. "
                f"Allowed: {', '.join(settings.allowed_video_extensions)}"
            ),
        )
    return extension


def _metadata_path(video_id: str) -> Path:
    return settings.videos_dir / f"{video_id}.json"


async def save_video_upload(file: UploadFile) -> VideoMetadata:
    """
    Validate and stream an uploaded video to disk, then persist its
    metadata as a JSON sidecar file.

    Raises HTTPException(400) for a rejected file type, or
    HTTPException(413) if the upload exceeds the configured size cap.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    extension = _validate_extension(file.filename)

    # Never trust the client-supplied name for building a path -- only
    # for display. The actual on-disk name is the generated video_id.
    original_filename = Path(file.filename).name
    video_id = uuid.uuid4().hex
    stored_filename = f"{video_id}{extension}"
    dest_path = settings.videos_dir / stored_filename

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    size_bytes = 0

    try:
        with open(dest_path, "wb") as out_file:
            while chunk := await file.read(CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    out_file.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds maximum upload size of "
                            f"{settings.max_upload_size_mb} MB."
                        ),
                    )
                out_file.write(chunk)
    except HTTPException:
        raise
    except Exception:
        # Any unexpected I/O failure: clean up the partial file rather
        # than leaving a corrupt/truncated video sitting in storage.
        dest_path.unlink(missing_ok=True)
        logger.exception("Failed to save uploaded video %s", original_filename)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")
    finally:
        await file.close()

    metadata = VideoMetadata(
        video_id=video_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        extension=extension,
        content_type=file.content_type,
        size_bytes=size_bytes,
        uploaded_at=datetime.now(timezone.utc),
        status="uploaded",
    )

    _metadata_path(video_id).write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )

    logger.info(
        "Saved video %s (%s, %.2f MB) as %s",
        video_id,
        original_filename,
        size_bytes / (1024 * 1024),
        stored_filename,
    )
    return metadata


def get_video_metadata(video_id: str) -> VideoMetadata:
    """Load a previously saved video's metadata, or raise 404."""
    path = _metadata_path(video_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return VideoMetadata(**data)
