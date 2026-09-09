"""
Audio extraction: video file -> mono 16kHz WAV, via FFmpeg.

We shell out to the `ffmpeg` binary rather than using a Python
wrapper library. FFmpeg's command-line interface is the stable,
well-documented contract here; a Python binding would just be a thin
(and sometimes lagging) layer over the same subprocess call.
"""

import subprocess
from pathlib import Path

from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def extract_audio(video_path: Path, video_id: str) -> Path:
    """
    Extract mono, 16kHz PCM WAV audio from a video file.

    Raises HTTPException(500) if FFmpeg fails (e.g. the file has no
    audio track, or is corrupt) -- with the tail of FFmpeg's stderr
    included, since that's where the actual error lives.
    """
    output_path = settings.audio_dir / f"{video_id}.wav"

    command = [
        settings.ffmpeg_binary,
        "-y",  # overwrite output if it already exists (e.g. re-transcribe)
        "-i", str(video_path),
        "-vn",  # drop any video stream -- we only want audio
        "-acodec", "pcm_s16le",  # uncompressed PCM, what Whisper expects
        "-ar", str(settings.audio_sample_rate_hz),
        "-ac", "1",  # mono
        str(output_path),
    ]

    logger.info("Extracting audio: %s -> %s", video_path.name, output_path.name)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Keep only the last portion of stderr -- FFmpeg's full output
        # is often long and mostly build/codec banner noise.
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-10:])
        output_path.unlink(missing_ok=True)
        logger.error("FFmpeg failed for %s: %s", video_path.name, stderr_tail)
        raise HTTPException(
            status_code=500,
            detail=f"Audio extraction failed: {stderr_tail}",
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail="Audio extraction produced no output file.",
        )

    return output_path
