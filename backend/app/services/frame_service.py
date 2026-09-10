"""
Frame extraction: video -> a small set of representative frames.

Sampling strategy (deliberately NOT "extract every frame"):
  1. Fixed-interval cadence (settings.frame_sample_interval_seconds) sets
     a ceiling on how often we even LOOK at a frame -- lecture visuals
     rarely change faster than a few seconds, so sampling more often
     would mostly just find duplicates of the same slide.
  2. Near-duplicate skip: each candidate is compared to the last SAVED
     frame via mean absolute pixel difference on a small downsampled
     grayscale copy. Below settings.frame_diff_threshold, it's treated
     as the same slide/scene still on screen and skipped.

Uses OpenCV (not FFmpeg) because step 2 needs actual pixel arrays to
compare -- FFmpeg extraction would mean writing every candidate frame
to disk first, then reading it back to decide whether to keep it.
OpenCV's VideoCapture reads frames with exact timestamps in one pass.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import Frame, FrameSet

logger = get_logger(__name__)


def _frame_dir(video_id: str) -> Path:
    return settings.frames_dir / video_id


def _manifest_path(video_id: str) -> Path:
    return _frame_dir(video_id) / "manifest.json"


def _downsample_gray(frame: np.ndarray) -> np.ndarray:
    """Small grayscale copy for cheap frame-to-frame comparison."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, settings.frame_diff_downsample_size, interpolation=cv2.INTER_AREA)


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def extract_frames(video_path: Path, video_id: str) -> FrameSet:
    """
    Sample frames from a video and save the ones that pass both the
    interval cadence and the near-duplicate check.

    Raises HTTPException(500) if the video can't be opened/read at all.
    """
    frame_dir = _frame_dir(video_id)
    # Re-extraction should start clean, not accumulate stale frames
    # from a previous run under different settings.
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(
            status_code=500, detail=f"Could not open video file for frame extraction: {video_path.name}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise HTTPException(
            status_code=500, detail=f"Could not determine frame rate for {video_path.name}"
        )

    interval_frames = max(1, round(fps * settings.frame_sample_interval_seconds))

    frames: List[Frame] = []
    last_saved_gray: Optional[np.ndarray] = None
    frame_index = 0
    saved_count = 0
    candidates_checked = 0

    # Sequential grab()/retrieve() instead of seeking to each target
    # position: cap.set(CAP_PROP_POS_FRAMES, ...) can land on the
    # nearest keyframe rather than the exact requested frame depending
    # on codec, silently drifting timestamps. grab() is a cheap
    # "advance one frame" with no decode cost; retrieve() only decodes
    # the frames we actually sample.
    try:
        while cap.grab():
            if frame_index % interval_frames == 0:
                ok, bgr_frame = cap.retrieve()
                if ok:
                    candidates_checked += 1
                    timestamp = frame_index / fps
                    candidate_gray = _downsample_gray(bgr_frame)

                    is_duplicate = (
                        last_saved_gray is not None
                        and _mean_abs_diff(candidate_gray, last_saved_gray) < settings.frame_diff_threshold
                    )

                    if not is_duplicate:
                        frame_id = f"{video_id}_{saved_count:04d}"
                        image_filename = f"frame_{saved_count:04d}.jpg"
                        image_path = frame_dir / image_filename
                        cv2.imwrite(
                            str(image_path),
                            bgr_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, settings.frame_jpeg_quality],
                        )
                        frames.append(
                            Frame(
                                frame_id=frame_id,
                                video_id=video_id,
                                frame_index=saved_count,
                                timestamp=round(timestamp, 2),
                                image_path=image_filename,
                            )
                        )
                        last_saved_gray = candidate_gray
                        saved_count += 1

            frame_index += 1
    finally:
        cap.release()

    frame_set = FrameSet(
        video_id=video_id,
        created_at=datetime.now(timezone.utc),
        sample_interval_seconds=settings.frame_sample_interval_seconds,
        diff_threshold=settings.frame_diff_threshold,
        frames=frames,
    )
    _manifest_path(video_id).write_text(frame_set.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "Extracted %d frames for %s (%d candidates checked, interval=%.1fs)",
        saved_count,
        video_id,
        candidates_checked,
        settings.frame_sample_interval_seconds,
    )
    return frame_set


def get_frame_set(video_id: str) -> Optional[FrameSet]:
    path = _manifest_path(video_id)
    if not path.exists():
        return None
    return FrameSet.model_validate_json(path.read_text(encoding="utf-8"))


def get_frame_image_path(video_id: str, frame_id: str) -> Path:
    """Resolve a frame_id to its image file on disk, or raise 404."""
    frame_set = get_frame_set(video_id)
    if frame_set is None:
        raise HTTPException(status_code=404, detail=f"No frames found for video '{video_id}'.")
    for frame in frame_set.frames:
        if frame.frame_id == frame_id:
            path = _frame_dir(video_id) / frame.image_path
            if not path.exists():
                raise HTTPException(
                    status_code=404, detail=f"Frame image file missing on disk for '{frame_id}'."
                )
            return path
    raise HTTPException(status_code=404, detail=f"Frame '{frame_id}' not found for video '{video_id}'.")
