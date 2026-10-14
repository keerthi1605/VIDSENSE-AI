"""
Frame extraction: video -> a small set of representative frames.

Sampling strategy (deliberately NOT "extract every frame"):
  1. Fixed-interval cadence (settings.frame_sample_interval_seconds) sets
     a ceiling on how often we even LOOK at a frame -- lecture visuals
     rarely change faster than a few seconds, so sampling more often
     would mostly just find duplicates of the same slide.
  2. Near-duplicate skip: each candidate is compared to the last SAVED
     frame on a small downsampled grayscale copy. If it still looks like
     the same scene it is skipped; otherwise it is saved AND marks a scene
     change. "Looks like" is settings.frame_diff_method:
       - "ssim" (default): structural similarity (Wang et al. 2004). Unlike
         raw pixel difference it compares local luminance, contrast and
         STRUCTURE, so a brightness/exposure shift or a small overlay (a
         name caption) does not register as a new scene. New scene when
         SSIM < settings.frame_ssim_threshold.
       - "mad": mean absolute pixel difference, the original heuristic,
         kept as a baseline for benchmarking. New scene when
         MAD >= settings.frame_diff_threshold.
  The saved frames therefore double as the video's scene boundaries
  (frames[1:] timestamps) -- chunking_service uses those to align text
  chunks to visual scenes.

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


_SSIM_WINDOW = (11, 11)
_SSIM_SIGMA = 1.5
_SSIM_C1 = (0.01 * 255) ** 2
_SSIM_C2 = (0.03 * 255) ** 2


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean structural similarity between two equal-size grayscale images,
    in [-1, 1] (1 = identical).

    Standard Gaussian-window SSIM (11x11, sigma 1.5, K1=0.01, K2=0.03,
    L=255) built on cv2.GaussianBlur so no extra dependency is needed.
    The 5px border, where the window hangs off the image, is excluded --
    the same "valid region" convention scikit-image uses. Checked against
    skimage.metrics.structural_similarity: max difference ~1e-15 on 40
    real frame pairs (tests/test_frame_service.py pins known values).
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, _SSIM_WINDOW, _SSIM_SIGMA)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    covariance = blur(a * b) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + _SSIM_C1) * (2 * covariance + _SSIM_C2)) / (
        (mu_a**2 + mu_b**2 + _SSIM_C1) * (var_a + var_b + _SSIM_C2)
    )
    pad = _SSIM_WINDOW[0] // 2
    return float(ssim_map[pad:-pad, pad:-pad].mean())


def _is_flat(gray: np.ndarray) -> bool:
    return float(gray.std()) < settings.frame_flat_std_threshold


def _is_new_scene(candidate: np.ndarray, last_saved: np.ndarray) -> bool:
    """Does this candidate frame look like a different scene than the last
    saved one, under the configured metric?"""
    if settings.frame_diff_method == "ssim":
        if _is_flat(candidate) and _is_flat(last_saved):
            # Nothing structural to compare (SSIM would be dominated by its
            # constants) -- luminance is the only meaningful signal.
            luminance_gap = abs(float(candidate.mean()) - float(last_saved.mean()))
            return luminance_gap >= settings.frame_flat_luminance_threshold
        return _ssim(candidate, last_saved) < settings.frame_ssim_threshold
    if settings.frame_diff_method == "mad":
        return _mean_abs_diff(candidate, last_saved) >= settings.frame_diff_threshold
    raise ValueError(
        f"Unknown frame_diff_method '{settings.frame_diff_method}'. Supported: 'ssim', 'mad'."
    )


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

                    is_new_scene = last_saved_gray is None or _is_new_scene(
                        candidate_gray, last_saved_gray
                    )

                    if is_new_scene:
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
        diff_threshold=(
            settings.frame_ssim_threshold
            if settings.frame_diff_method == "ssim"
            else settings.frame_diff_threshold
        ),
        diff_method=settings.frame_diff_method,
        frames=frames,
    )
    _manifest_path(video_id).write_text(frame_set.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "Extracted %d frames for %s (%d candidates checked, interval=%.1fs, method=%s)",
        saved_count,
        video_id,
        candidates_checked,
        settings.frame_sample_interval_seconds,
        settings.frame_diff_method,
    )
    return frame_set


def get_frame_set(video_id: str) -> Optional[FrameSet]:
    path = _manifest_path(video_id)
    if not path.exists():
        return None
    return FrameSet.model_validate_json(path.read_text(encoding="utf-8"))


def get_scene_boundaries(video_id: str) -> Optional[List[float]]:
    """
    Timestamps (seconds) at which a new visual scene begins, or None if
    this video has no extracted frames.

    Every saved frame after the first marks a scene change: it is the
    first sampled frame that no longer looked like the previous saved one.
    The true change happened somewhere in the preceding sample interval,
    so a boundary is accurate to settings.frame_sample_interval_seconds.
    The first frame (t ~ 0) is the start of the video, not a boundary.
    """
    frame_set = get_frame_set(video_id)
    if frame_set is None:
        return None
    return [f.timestamp for f in frame_set.frames[1:]]


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
