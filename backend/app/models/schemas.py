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
    # Set only for videos ingested via /api/videos/from-url; None for a
    # direct file upload. Provenance only -- every downstream pipeline
    # step treats a URL-ingested video identically to an uploaded one.
    source_url: Optional[str] = None


class VideoUrlRequest(BaseModel):
    url: str


class TranscribeRequest(BaseModel):
    video_id: str
    # Optional per-video overrides for domain vocabulary, so testing
    # across genres (a technical lecture vs. a cooking video vs. an
    # interview) doesn't require changing global config between runs.
    # Falls back to settings.whisper_initial_prompt/hotwords when omitted.
    initial_prompt: Optional[str] = None
    hotwords: Optional[str] = None


class WordTimestamp(BaseModel):
    """One word within a segment, with its own exact start/end time.

    Only populated when transcription runs with word_timestamps enabled
    (see settings.whisper_word_timestamps); otherwise a segment's `words`
    is left as None rather than an empty list, so callers can tell
    "not requested" apart from "segment had no words" (silence)."""

    word: str
    start_time: float
    end_time: float


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
    words: Optional[List[WordTimestamp]] = None


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


class IndexResult(BaseModel):
    """Result of pushing a video's chunks+embeddings into the vector store."""

    video_id: str
    chunks_indexed: int
    collection_name: str


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    # Restrict search to one video's chunks. Omitted = search the whole
    # library -- the more powerful default for a multi-video system.
    video_id: Optional[str] = None


class SearchResult(BaseModel):
    """
    One retrieved chunk, with everything needed to show and cite it:
    where it's from (video_id), what it says (text), where in the
    video it is (start_time/end_time), and how relevant it was (score).
    """

    video_id: str
    chunk_id: str
    text: str
    start_time: float
    end_time: float
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    video_id: Optional[str] = None


class SourceRef(BaseModel):
    """
    A citation: exactly which chunk (and thus which moment in which
    video) informed the answer. Built directly from retrieval results,
    never parsed out of the LLM's own text -- so a timestamp shown to
    the user is always a real, exact chunk boundary, never something
    the model could get wrong or invent.
    """

    video_id: str
    chunk_id: str
    start_time: float
    end_time: float
    score: float


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceRef]
    # False when retrieval found nothing at all to answer from -- lets
    # a client distinguish "the model tried and covered this" from
    # "there was nothing to work with in the first place" without
    # parsing the answer text.
    has_sufficient_context: bool


class Frame(BaseModel):
    """
    One sampled video frame -- a candidate that survived both the
    fixed-interval sampling cadence AND the near-duplicate check (it
    differs enough from the previously saved frame to be worth keeping).

    timestamp is in seconds, same unit/precision convention as
    TranscriptSegment/Chunk, so frame results can sit alongside text
    results in a fused multimodal response (Phase 6) without unit
    mismatches.
    """

    frame_id: str
    video_id: str
    frame_index: int
    timestamp: float
    image_path: str  # relative to storage/frames/{video_id}/


class FrameSet(BaseModel):
    """The full set of sampled frames for one video, plus the sampling
    settings used to produce them (mirrors ChunkSet's role for text)."""

    video_id: str
    created_at: datetime
    sample_interval_seconds: float
    diff_threshold: float
    frames: List[Frame]


class FrameEmbeddingSet(BaseModel):
    """
    Metadata describing a video's stored CLIP frame-embedding vectors.

    Mirrors EmbeddingSet's role for text chunks, with the same
    reasoning: vectors live in a `.npy` file, never returned raw over
    the API. dimension is CLIP's (512), a completely different and
    non-comparable space from EmbeddingSet's text dimension (384) --
    the two are never mixed.
    """

    video_id: str
    model_name: str
    dimension: int
    frame_count: int
    created_at: datetime
    frame_ids: List[str]


class FrameIndexResult(BaseModel):
    """Result of pushing a video's frames+CLIP embeddings into the
    vector store (mirrors IndexResult for text chunks)."""

    video_id: str
    frames_indexed: int
    collection_name: str


class VisualSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    video_id: Optional[str] = None


class FrameSearchResult(BaseModel):
    """
    One retrieved frame, with everything needed to show and cite it:
    where it's from, when it occurs, and how relevant it was. No `text`
    field -- unlike a chunk, a frame's "content" IS the image; see
    image_path (relative to storage/frames/{video_id}/) to display it.
    """

    video_id: str
    frame_id: str
    timestamp: float
    image_path: str
    score: float


class VisualSearchResponse(BaseModel):
    query: str
    results: List[FrameSearchResult]


class MultimodalSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    video_id: Optional[str] = None
    # Override the configured fusion weights per-request, for
    # experimentation without restarting the server. Omitted = use
    # settings.multimodal_text_weight / multimodal_visual_weight.
    text_weight: Optional[float] = None
    visual_weight: Optional[float] = None


class MultimodalResult(BaseModel):
    """
    One fused piece of evidence for a query -- a moment in a video that
    matched via its transcript, via a frame's image, or (the case that
    makes this phase worthwhile) both at once.

    start_time/end_time is ALWAYS a valid, non-empty range, even for a
    visual-only result (start_time == end_time == the frame's exact
    timestamp) -- a consistent contract for a frontend video player to
    seek/highlight against, regardless of which modality produced the
    match. evidence_type says explicitly which case this is, rather
    than making a caller infer it from which optional fields are set.
    """

    video_id: str
    start_time: float
    end_time: float
    evidence_type: str  # "text", "visual", or "both"
    text: Optional[str] = None
    chunk_id: Optional[str] = None
    frame_id: Optional[str] = None
    image_path: Optional[str] = None
    # Each raw score, min-max normalized within its own modality's
    # result set (see retrieval_service.search_multimodal) -- 0.0 for
    # whichever modality had no match at this moment.
    text_score: float = 0.0
    visual_score: float = 0.0
    final_score: float


class MultimodalSearchResponse(BaseModel):
    query: str
    text_weight: float
    visual_weight: float
    results: List[MultimodalResult]
