"""
Speech-to-text transcription via Faster-Whisper.

The model is loaded lazily and cached as a module-level singleton:
loading weights from disk takes real time (seconds, even for "base"
on CPU), so we pay that cost once per process, not once per request.

`get_whisper_model()` is the one seam tests should monkeypatch to
avoid loading real model weights in unit tests.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import Transcript, TranscriptSegment

logger = get_logger(__name__)

_model: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    global _model
    if _model is None:
        logger.info(
            "Loading Faster-Whisper model '%s' (device=%s, compute_type=%s)...",
            settings.whisper_model_size,
            settings.whisper_device,
            settings.whisper_compute_type,
        )
        _model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("Whisper model loaded.")
    return _model


def _transcript_path(video_id: str) -> Path:
    return settings.transcripts_dir / f"{video_id}.json"


def transcribe_and_save(video_id: str, audio_path: Path) -> Transcript:
    """
    Run Whisper transcription on the given audio file and persist the
    result as `storage/transcripts/{video_id}.json`.
    """
    model = get_whisper_model()

    logger.info("Transcribing %s...", audio_path.name)
    segments_iter, info = model.transcribe(str(audio_path))

    # faster-whisper returns segments as a lazy generator -- it does
    # the actual decoding as you iterate, not up front. We materialize
    # it into our own schema here, which also drives that decoding.
    segments = [
        TranscriptSegment(
            segment_id=i,
            start_time=round(seg.start, 2),
            end_time=round(seg.end, 2),
            text=seg.text.strip(),
        )
        for i, seg in enumerate(segments_iter)
    ]

    transcript = Transcript(
        video_id=video_id,
        language=info.language,
        duration_seconds=round(info.duration, 2),
        model_size=settings.whisper_model_size,
        created_at=datetime.now(timezone.utc),
        segments=segments,
    )

    _transcript_path(video_id).write_text(
        transcript.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info(
        "Transcribed %s: %d segments, %.1fs audio, language=%s",
        video_id,
        len(segments),
        transcript.duration_seconds,
        transcript.language,
    )
    return transcript


def get_transcript(video_id: str) -> Optional[Transcript]:
    """Load a previously saved transcript, or None if it doesn't exist."""
    path = _transcript_path(video_id)
    if not path.exists():
        return None
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))
