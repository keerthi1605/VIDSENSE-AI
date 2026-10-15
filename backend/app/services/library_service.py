"""
Library maintenance: remove a video and everything derived from it.

A video leaves traces in several places -- the file and its metadata, the
extracted audio, transcript, chunks, text and frame embedding matrices, the
frame images, and records in both vector collections. Deleting only some of
them leaves a half-alive video (search results pointing at a file that is
gone), so removal lives in one function that knows all of them.

Every path is built from the video_id with an exact name, never a prefix
glob: ids such as "test_video_1" and "test_video_10" must not be able to
delete each other's files.
"""

import re
import shutil
from typing import Dict

from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import get_logger
from app.services import vector_store_service, video_service

logger = get_logger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def delete_video(video_id: str) -> Dict[str, int]:
    """
    Delete a video, its derived artifacts and its vector-store records.

    Raises HTTPException 400 for an id that is not a plain identifier (it is
    interpolated into file names) and 404 for an unknown video. Returns how
    many files and directories were removed.
    """
    if not _SAFE_ID.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid video id.")
    metadata = video_service.get_video_metadata(video_id)  # 404s if unknown

    candidates = [
        settings.videos_dir / metadata.stored_filename,
        settings.videos_dir / f"{video_id}.json",
        settings.transcripts_dir / f"{video_id}.json",
        settings.chunks_dir / f"{video_id}.json",
        settings.embeddings_dir / f"{video_id}.npy",
        settings.embeddings_dir / f"{video_id}.json",
        settings.embeddings_dir / f"{video_id}_frames.npy",
        settings.embeddings_dir / f"{video_id}_frames.json",
        *settings.audio_dir.glob(f"{video_id}.*"),  # ".*" cannot match a longer id
    ]
    removed = 0
    for path in candidates:
        if path.exists():
            path.unlink()
            removed += 1
    frame_dir = settings.frames_dir / video_id
    if frame_dir.is_dir():
        shutil.rmtree(frame_dir)
        removed += 1

    vector_store_service.delete_video(settings.chroma_collection_name, video_id)
    vector_store_service.delete_video(settings.chroma_frames_collection_name, video_id)
    logger.info("Deleted video %s (%d files/directories and its vector records)", video_id, removed)
    return {"files_removed": removed}
