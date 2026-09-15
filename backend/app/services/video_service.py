"""
Video upload handling: validation, safe storage, metadata persistence.

Kept independent of FastAPI's `Request`/`Response` types (only
`UploadFile` is used, which is just a thin file-like wrapper) so this
logic is testable and reusable without spinning up the web layer --
that's the "services stay separate from the API layer" principle.
"""

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp
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


def _cleanup_partial_download(video_id: str) -> None:
    """Remove any file(s) a failed/partial download left behind under
    this video_id, so a failed ingestion never leaves stray media."""
    for path in settings.videos_dir.glob(f"{video_id}.*"):
        path.unlink(missing_ok=True)


def save_video_from_url(url: str) -> VideoMetadata:
    """
    Download a video from a URL (YouTube, and anything else yt-dlp
    supports) and store it exactly like a direct upload -- every
    downstream endpoint (transcribe, chunks, embeddings, frames, ...)
    works identically afterward regardless of how the video arrived.

    Two-phase, on purpose:
      1. Probe metadata WITHOUT downloading (yt-dlp's `download=False`)
         so an oversized/live video is rejected before spending any
         bandwidth or disk -- not after.
      2. Only then actually download, directly into storage/videos/
         under the generated video_id (never the remote title -- same
         "don't trust external names for paths" discipline as the
         direct-upload path above).
    """
    probe_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch video info: {exc}")

    if info.get("is_live"):
        raise HTTPException(
            status_code=400,
            detail="Live streams are not supported -- they have no fixed duration.",
        )

    duration = info.get("duration")
    if duration and duration > settings.max_youtube_duration_seconds:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Video is {duration / 60:.0f} minutes long, exceeding the "
                f"{settings.max_youtube_duration_seconds / 60:.0f}-minute limit "
                "for this deployment."
            ),
        )

    title = info.get("title") or "video"
    video_id = uuid.uuid4().hex
    stored_filename = f"{video_id}.mp4"
    dest_path = settings.videos_dir / stored_filename

    download_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": settings.yt_dlp_format,
        "merge_output_format": "mp4",
        # Guarantees an actual .mp4 container even if the chosen source
        # format wasn't one to begin with (merge_output_format alone
        # only applies when a separate video+audio merge happens).
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
        "outtmpl": str(settings.videos_dir / f"{video_id}.%(ext)s"),
        "noplaylist": True,  # a playlist URL downloads only its first video
    }
    # yt-dlp's ffmpeg_location wants a resolved path, not a bare command
    # name to be PATH-searched -- resolve settings.ffmpeg_binary (e.g.
    # "ffmpeg") ourselves via shutil.which, same as the shell would, and
    # only pass it through if found; otherwise let yt-dlp fall back to
    # its own PATH search rather than pass it something it can't use.
    resolved_ffmpeg = shutil.which(settings.ffmpeg_binary)
    if resolved_ffmpeg:
        download_opts["ffmpeg_location"] = resolved_ffmpeg
    try:
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        _cleanup_partial_download(video_id)
        raise HTTPException(status_code=502, detail=f"Failed to download video: {exc}")
    except Exception:
        _cleanup_partial_download(video_id)
        logger.exception("Unexpected failure downloading %s", url)
        raise HTTPException(status_code=500, detail="Failed to download video.")

    if not dest_path.exists():
        _cleanup_partial_download(video_id)
        raise HTTPException(
            status_code=500,
            detail="Download completed but the expected output file was not found.",
        )

    size_bytes = dest_path.stat().st_size
    metadata = VideoMetadata(
        video_id=video_id,
        original_filename=f"{title}.mp4",
        stored_filename=stored_filename,
        extension=".mp4",
        content_type="video/mp4",
        size_bytes=size_bytes,
        uploaded_at=datetime.now(timezone.utc),
        status="uploaded",
        source_url=url,
    )
    _metadata_path(video_id).write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "Downloaded video from %s as %s (%.2f MB, title=%r)",
        url, video_id, size_bytes / (1024 * 1024), title,
    )
    return metadata


def get_video_metadata(video_id: str) -> VideoMetadata:
    """Load a previously saved video's metadata, or raise 404."""
    path = _metadata_path(video_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return VideoMetadata(**data)


def get_video_file_path(video_id: str) -> Path:
    """Resolve a video_id to its stored file path on disk, or raise 404."""
    metadata = get_video_metadata(video_id)
    path = settings.videos_dir / metadata.stored_filename
    if not path.exists():
        # Metadata exists but the file is gone -- treat as not found
        # rather than silently proceeding with a missing file.
        raise HTTPException(
            status_code=404,
            detail=f"Video file for '{video_id}' is missing from storage.",
        )
    return path


def update_video_status(video_id: str, status: str) -> VideoMetadata:
    """Update and persist a video's processing status."""
    metadata = get_video_metadata(video_id)
    metadata.status = status
    _metadata_path(video_id).write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )
    return metadata
