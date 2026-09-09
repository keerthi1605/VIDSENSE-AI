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

    # --- Upload constraints (Phase 1) ---
    max_upload_size_mb: int = 1024  # 1 GB ceiling, revisit later
    allowed_video_extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm")

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

    def ensure_storage_dirs(self) -> None:
        """Create all storage subdirectories if they don't already exist."""
        for d in (
            self.videos_dir,
            self.audio_dir,
            self.frames_dir,
            self.transcripts_dir,
            self.embeddings_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# Single shared instance imported everywhere else, e.g.:
#   from app.core.config import settings
settings = Settings()
