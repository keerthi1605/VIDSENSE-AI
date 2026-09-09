"""
Video-related HTTP routes.

Thin by design: parse the request, delegate to app.services.video_service,
return its result. No business logic lives here.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import ChunkSet, EmbeddingSet, Transcript, TranscribeRequest, VideoMetadata
from app.services import (
    audio_service,
    chunking_service,
    embedding_service,
    transcription_service,
    video_service,
)

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


@router.post("/{video_id}/chunks", response_model=ChunkSet)
def chunk_video(video_id: str) -> ChunkSet:
    """
    Chunk a video's saved transcript into embedding-sized pieces.

    Requires transcription to have already run -- chunking has nothing
    to operate on otherwise. This is deliberately its own step (not
    folded into /transcribe) so each pipeline stage stays independently
    inspectable and re-runnable, matching how the phases themselves
    are separated.
    """
    transcript = transcription_service.get_transcript(video_id)
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found for video '{video_id}'. "
            "Transcribe it first via POST /api/videos/transcribe.",
        )
    return chunking_service.chunk_and_save(transcript)


@router.get("/{video_id}/chunks", response_model=ChunkSet)
def get_chunks(video_id: str) -> ChunkSet:
    """Fetch a previously computed chunk set for a video."""
    chunk_set = chunking_service.get_chunk_set(video_id)
    if chunk_set is None:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for video '{video_id}'. "
            "Has it been chunked yet?",
        )
    return chunk_set


@router.post("/{video_id}/embeddings", response_model=EmbeddingSet)
def embed_video(video_id: str) -> EmbeddingSet:
    """
    Generate and store text embeddings for a video's chunks.

    Requires chunking to have already run. Sync `def` for the same
    reason as /transcribe: model loading + encoding is CPU-bound work
    that shouldn't block the event loop.
    """
    chunk_set = chunking_service.get_chunk_set(video_id)
    if chunk_set is None:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for video '{video_id}'. "
            "Chunk it first via POST /api/videos/{video_id}/chunks.",
        )
    return embedding_service.embed_and_save(chunk_set)


@router.get("/{video_id}/embeddings", response_model=EmbeddingSet)
def get_embeddings(video_id: str) -> EmbeddingSet:
    """
    Fetch embedding metadata for a video (model name, dimension, chunk
    count) -- NOT the raw vectors themselves; see EmbeddingSet's
    docstring for why.
    """
    metadata = embedding_service.get_embedding_metadata(video_id)
    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"No embeddings found for video '{video_id}'. "
            "Has it been embedded yet?",
        )
    return metadata
