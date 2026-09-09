"""
Video-related HTTP routes.

Thin by design: parse the request, delegate to app.services.video_service,
return its result. No business logic lives here.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import Transcript, TranscribeRequest, VideoMetadata
from app.services import audio_service, transcription_service, video_service

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


@router.post("/transcribe", response_model=Transcript)
def transcribe_video(request: TranscribeRequest) -> Transcript:
    """
    Extract audio and transcribe a previously uploaded video.

    Synchronous by design for Phase 1 (per the project's "synchronous
    first, background processing later" roadmap) -- the request blocks
    until transcription completes. Defined as a plain `def`, not
    `async def`, so FastAPI runs it in its worker thread pool rather
    than on the event loop: this CPU-bound work would otherwise stall
    every other in-flight request (including /health) for its duration.
    """
    video_path = video_service.get_video_file_path(request.video_id)

    try:
        audio_path = audio_service.extract_audio(video_path, request.video_id)
        transcript = transcription_service.transcribe_and_save(
            request.video_id, audio_path
        )
    except HTTPException:
        video_service.update_video_status(request.video_id, "failed")
        raise

    video_service.update_video_status(request.video_id, "transcribed")
    return transcript


@router.get("/{video_id}/transcript", response_model=Transcript)
def get_transcript(video_id: str) -> Transcript:
    """Fetch a previously generated transcript for a video."""
    transcript = transcription_service.get_transcript(video_id)
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found for video '{video_id}'. "
            "Has it been transcribed yet?",
        )
    return transcript
