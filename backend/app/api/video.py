"""
Video-related HTTP routes.

Thin by design: parse the request, delegate to app.services.video_service,
return its result. No business logic lives here.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.models.schemas import (
    ChunkSet,
    EmbeddingSet,
    FrameEmbeddingSet,
    FrameSet,
    IndexResult,
    Transcript,
    TranscribeRequest,
    VideoMetadata,
)
from app.services import (
    audio_service,
    chunking_service,
    embedding_service,
    frame_service,
    retrieval_service,
    transcription_service,
    video_service,
    vision_service,
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


@router.post("/{video_id}/index", response_model=IndexResult)
def index_video(video_id: str) -> IndexResult:
    """
    Push a video's chunks + embeddings into the vector database,
    making it searchable via /api/search.

    Requires both chunking and embedding to have already run.
    """
    chunk_set = chunking_service.get_chunk_set(video_id)
    if chunk_set is None:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for video '{video_id}'. "
            "Chunk it first via POST /api/videos/{video_id}/chunks.",
        )
    embedding_meta = embedding_service.get_embedding_metadata(video_id)
    if embedding_meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"No embeddings found for video '{video_id}'. "
            "Embed it first via POST /api/videos/{video_id}/embeddings.",
        )

    try:
        return retrieval_service.index_video(chunk_set, embedding_meta)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/{video_id}/frames", response_model=FrameSet)
def extract_frames(video_id: str) -> FrameSet:
    """
    Sample representative frames from a video (see frame_service for
    the sampling strategy). Sync `def`: OpenCV decoding is CPU-bound.
    """
    video_path = video_service.get_video_file_path(video_id)
    return frame_service.extract_frames(video_path, video_id)


@router.get("/{video_id}/frames", response_model=FrameSet)
def get_frames(video_id: str) -> FrameSet:
    """Fetch a previously extracted frame set for a video."""
    frame_set = frame_service.get_frame_set(video_id)
    if frame_set is None:
        raise HTTPException(
            status_code=404,
            detail=f"No frames found for video '{video_id}'. "
            "Has it been frame-extracted yet?",
        )
    return frame_set


@router.get("/{video_id}/frames/{frame_id}/image")
def get_frame_image(video_id: str, frame_id: str) -> FileResponse:
    """Serve one extracted frame's actual JPEG image -- useful for a
    frontend preview or manual inspection."""
    image_path = frame_service.get_frame_image_path(video_id, frame_id)
    return FileResponse(image_path, media_type="image/jpeg")


@router.post("/{video_id}/frame-embeddings", response_model=FrameEmbeddingSet)
def embed_frames(video_id: str) -> FrameEmbeddingSet:
    """
    Generate and store CLIP visual embeddings for a video's frames.

    Requires frame extraction to have already run. Sync `def`: model
    loading + encoding is CPU-bound.
    """
    frame_set = frame_service.get_frame_set(video_id)
    if frame_set is None:
        raise HTTPException(
            status_code=404,
            detail=f"No frames found for video '{video_id}'. "
            "Extract frames first via POST /api/videos/{video_id}/frames.",
        )
    return vision_service.embed_and_save_frames(frame_set)


@router.get("/{video_id}/frame-embeddings", response_model=FrameEmbeddingSet)
def get_frame_embeddings(video_id: str) -> FrameEmbeddingSet:
    """Fetch CLIP frame-embedding metadata (model, dimension, frame
    count) -- NOT the raw vectors; see FrameEmbeddingSet's docstring."""
    metadata = vision_service.get_frame_embedding_metadata(video_id)
    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"No frame embeddings found for video '{video_id}'. "
            "Has it been frame-embedded yet?",
        )
    return metadata
