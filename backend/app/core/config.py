"""
Centralized application configuration.

WHY THIS FILE EXISTS (design principle: Model/Config Abstraction)
-------------------------------------------------------------------
Nothing in this project should hard-code a file path, model name, or
provider choice directly inside business logic. Every service reads
its configuration from a single `Settings` object built here.

This means that later, when we swap:
  - the Whisper model size (e.g. "base" -> "small")
  - the embedding model
  - the LLM provider
  - storage paths (e.g. moving from local disk to a mounted volume)
we change ONE place, not every file that happens to use it.

Settings are loaded from environment variables (via a `.env` file in
`backend/`), using pydantic-settings. This is the standard FastAPI
pattern: `.env` holds machine/deployment-specific values, `.env.example`
documents what variables exist without leaking real secrets.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    app_name: str = "VidSense AI"
    api_v1_prefix: str = "/api"
    debug: bool = True

    # --- Storage paths (relative to backend/) ---
    # BASE_DIR = backend/app/core/config.py -> parents[2] = backend/
    base_dir: Path = Path(__file__).resolve().parents[2]
    storage_dir: Path = base_dir / "storage"
    videos_dir: Path = storage_dir / "videos"
    audio_dir: Path = storage_dir / "audio"
    frames_dir: Path = storage_dir / "frames"
    transcripts_dir: Path = storage_dir / "transcripts"
    embeddings_dir: Path = storage_dir / "embeddings"
    chunks_dir: Path = storage_dir / "chunks"

    # --- Upload constraints (Phase 1) ---
    max_upload_size_mb: int = 1024  # 1 GB ceiling, revisit later
    allowed_video_extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm")

    # --- Video ingestion from URL, e.g. YouTube (Phase 1, Step 4) ---
    # Rejected BEFORE downloading (probed via yt-dlp with download=False)
    # -- no point spending bandwidth/disk on something we'll reject anyway.
    max_youtube_duration_seconds: int = 7200  # 2 hours
    # Capped at 720p on purpose: this machine's whole design is CPU-only
    # and storage-conscious (see Hardware note in README) -- no reason to
    # pull 4K just because a source offers it. merge_output_format +
    # the FFmpegVideoConvertor postprocessor (in video_service.py)
    # guarantee a consistent .mp4 regardless of the source container.
    yt_dlp_format: str = (
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]"
        "/best[height<=720][ext=mp4]/best[height<=720]/best"
    )

    # --- Transcription (Faster-Whisper) ---
    # Model size is deliberately small/base-tier by default: this machine's
    # GPU (2GB VRAM, old driver) cannot accelerate anything bigger, so we
    # run on CPU. int8 compute type keeps CPU inference fast enough.
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # --- Audio extraction (FFmpeg) ---
    # Bare command name: resolved via PATH. Override in .env with a full
    # path if ffmpeg isn't on PATH on a given machine.
    ffmpeg_binary: str = "ffmpeg"
    audio_sample_rate_hz: int = 16000  # Whisper's expected input rate

    # --- Chunking (Phase 2) ---
    # Chunks are built by merging consecutive Whisper segments until this
    # many words is reached -- NOT by cutting raw text at a fixed character
    # offset. This keeps every chunk's start/end time exact (taken directly
    # from real segment boundaries) instead of interpolated.
    chunk_target_words: int = 150
    # How many trailing segments of a chunk get repeated at the start of
    # the next chunk, so ideas that span a chunk boundary still appear in
    # both chunks' embeddings.
    chunk_overlap_segments: int = 1

    # --- Text embeddings (Sentence-Transformers) ---
    # A small (384-dim), fast, general-purpose model -- the right
    # tradeoff for CPU-only inference on this machine. Swappable to a
    # larger model (e.g. "all-mpnet-base-v2") via .env once retrieval
    # quality can actually be measured (Phase 10).
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # --- Vector database (ChromaDB) ---
    # Persisted outside storage/ (matches this project's own .gitignore,
    # set up back in Phase 1) since it's a database directory, not a
    # per-video artifact folder.
    chroma_persist_dir: Path = base_dir / "chroma_db"
    chroma_collection_name: str = "video_chunks"
    # A SEPARATE collection for CLIP frame vectors (512-dim) -- Chroma
    # collections are single-dimension, and text-chunk vectors (384-dim,
    # MiniLM) and frame vectors are different, non-comparable spaces
    # anyway (see docs/concepts.md), so they could never share one
    # collection even if dimensions matched.
    chroma_frames_collection_name: str = "video_frames"
    search_default_top_k: int = 5

    # --- LLM (RAG answer generation) ---
    # "llm_provider" is the actual swap point: llm_service.py picks an
    # implementation based on this string. Only "ollama" exists today,
    # but a future "anthropic"/"openai" provider is a new branch there,
    # not a rewrite -- this field is what makes that a config change.
    llm_provider: str = "ollama"
    # A small (~2GB), CPU-runnable instruction-tuned model -- chosen for
    # this machine's 8GB total RAM, shared with everything else already
    # running (torch, sentence-transformers, faster-whisper). A larger
    # model would be higher quality but risks starving the machine.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    llm_request_timeout_seconds: int = 120
    # How many top search results get passed to the LLM as context.
    rag_context_chunks: int = 5

    # --- Frame extraction (Phase 5, OpenCV) ---
    # Ceiling on sampling frequency: one candidate frame every N seconds.
    # Lecture visuals (slides, code, diagrams) rarely change faster than
    # this, so sampling more often would mostly just find duplicates.
    frame_sample_interval_seconds: float = 5.0
    # A candidate frame is compared to the last SAVED frame via mean
    # absolute pixel difference on a small downsampled grayscale copy
    # (cheap to compute). Below this threshold, it's treated as a
    # duplicate (same slide still on screen) and skipped. This is a
    # starting heuristic -- worth tuning once retrieval quality can
    # actually be measured (Phase 10), not derived from any formal study.
    frame_diff_threshold: float = 10.0
    frame_diff_downsample_size: tuple[int, int] = (64, 64)
    frame_jpeg_quality: int = 85

    # --- Visual embeddings (CLIP, Phase 5 Step 2) ---
    # CLIP is the only real choice here, not one option among several: it
    # jointly embeds images AND text into the SAME vector space, which is
    # what makes "search frames by a text query" possible at all. This is
    # a SEPARATE embedding space from embedding_model_name above (512-dim
    # here vs. 384-dim there) -- the two are never compared to each other.
    # clip-ViT-B-32 via sentence-transformers reuses the dependency already
    # installed for text embeddings, same .encode() API for both images
    # and text.
    clip_model_name: str = "clip-ViT-B-32"
    clip_device: str = "cpu"

    # --- Multimodal fusion (Phase 6) ---
    # Simple weighted-sum fusion, per the master roadmap: final_score =
    # text_weight * text_score + visual_weight * visual_score. Each
    # modality's raw scores are min-max normalized to [0,1] within their
    # OWN returned result set first (see retrieval_service.search_multimodal)
    # -- MiniLM and CLIP cosine scores are not on the same natural scale
    # (empirically: MiniLM matches score ~0.7-0.85, CLIP matches ~0.25-0.35),
    # so a naive unnormalized sum would let text silently dominate
    # regardless of these weights. Defaults are neutral (0.5/0.5), NOT
    # tuned against any real evaluation -- that's Phase 10's job.
    multimodal_text_weight: float = 0.5
    multimodal_visual_weight: float = 0.5
    # How many results to pull from EACH modality before fusing -- more
    # than the final top_k, since fusion needs enough candidates from
    # both sides to actually find time-based overlaps.
    multimodal_candidate_k: int = 10

    def ensure_storage_dirs(self) -> None:
        """Create all storage subdirectories if they don't already exist."""
        for d in (
            self.videos_dir,
            self.audio_dir,
            self.frames_dir,
            self.transcripts_dir,
            self.embeddings_dir,
            self.chunks_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# Single shared instance imported everywhere else, e.g.:
#   from app.core.config import settings
settings = Settings()
