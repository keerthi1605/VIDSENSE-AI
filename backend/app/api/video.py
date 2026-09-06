"""
Video-related HTTP routes.

Thin by design: parse the request, delegate to app.services.video_service,
return its result. No business logic lives here.
"""

from fastapi import APIRouter, File, UploadFile

from app.models.schemas import VideoMetadata
from app.services import video_service

router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.post("/upload", response_model=VideoMetadata, status_code=201)
async def upload_video(file: UploadFile = File(...)) -> VideoMetadata:
    """
    Upload a video file.

    Validates extension and size, stores the file under
    `storage/videos/{video_id}{ext}`, and persists a JSON metadata
    sidecar. Returns the generated `video_id`, which subsequent
    transcription/search calls will reference.
    """
    return await video_service.save_video_upload(file)
