"""
Tests for app.services.frame_service.

Uses ffmpeg to generate small synthetic test videos with genuine visual
changes (solid color segments) -- this lets us verify the near-duplicate
skip logic actually discriminates "same content" from "different
content" on real pixel data, not just trust the implementation.
"""

import subprocess

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.services import frame_service


def _run_ffmpeg(args):
    subprocess.run([settings.ffmpeg_binary, "-y", *args], capture_output=True, check=True)


@pytest.fixture
def static_color_video(tmp_path):
    """10 seconds of solid red -- visually unchanging throughout."""
    video_path = tmp_path / "static.mp4"
    _run_ffmpeg(
        [
            "-f", "lavfi", "-i", "color=c=red:size=64x64:duration=10",
            "-r", "25",
            str(video_path),
        ]
    )
    return video_path


@pytest.fixture
def two_scene_video(tmp_path):
    """5s solid red, then 5s solid blue -- one real scene change at t=5."""
    video_path = tmp_path / "two_scene.mp4"
    _run_ffmpeg(
        [
            "-f", "lavfi", "-i", "color=c=red:size=64x64:duration=5",
            "-f", "lavfi", "-i", "color=c=blue:size=64x64:duration=5",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
            "-r", "25",
            str(video_path),
        ]
    )
    return video_path


@pytest.fixture(autouse=True)
def _small_interval(monkeypatch):
    monkeypatch.setattr(settings, "frame_sample_interval_seconds", 1.0)


@pytest.fixture(autouse=True)
def _cleanup(tmp_path, monkeypatch):
    # Redirect frames_dir into tmp_path so tests never touch real storage.
    monkeypatch.setattr(settings, "frames_dir", tmp_path / "frames_storage")
    yield


def test_static_video_produces_a_single_frame(static_color_video):
    """A visually unchanging video should save only the first frame --
    every later candidate is a duplicate of it."""
    frame_set = frame_service.extract_frames(static_color_video, "static_vid")

    assert len(frame_set.frames) == 1
    assert frame_set.frames[0].timestamp == 0.0
    assert frame_set.frames[0].frame_id == "static_vid_0000"


def test_scene_change_produces_two_frames(two_scene_video):
    """A real color change at t=5 should be caught -- the duplicate
    check must NOT collapse genuinely different content."""
    frame_set = frame_service.extract_frames(two_scene_video, "two_scene_vid")

    assert len(frame_set.frames) == 2
    assert frame_set.frames[0].timestamp == 0.0
    assert frame_set.frames[1].timestamp == pytest.approx(5.0, abs=0.5)


def test_frame_images_actually_saved_to_disk(two_scene_video):
    frame_set = frame_service.extract_frames(two_scene_video, "two_scene_vid")

    frame_dir = settings.frames_dir / "two_scene_vid"
    for frame in frame_set.frames:
        image_path = frame_dir / frame.image_path
        assert image_path.exists()
        assert image_path.stat().st_size > 0


def test_manifest_persists_and_reloads(two_scene_video):
    saved = frame_service.extract_frames(two_scene_video, "two_scene_vid")

    reloaded = frame_service.get_frame_set("two_scene_vid")
    assert reloaded is not None
    assert len(reloaded.frames) == len(saved.frames)
    assert reloaded.frames[0].timestamp == saved.frames[0].timestamp


def test_get_frame_set_returns_none_when_missing():
    assert frame_service.get_frame_set("nonexistent") is None


def test_get_frame_image_path_resolves_correctly(two_scene_video):
    frame_set = frame_service.extract_frames(two_scene_video, "two_scene_vid")
    first_frame = frame_set.frames[0]

    resolved = frame_service.get_frame_image_path("two_scene_vid", first_frame.frame_id)
    assert resolved.exists()
    assert resolved.name == first_frame.image_path


def test_get_frame_image_path_404s_for_unknown_frame(two_scene_video):
    frame_service.extract_frames(two_scene_video, "two_scene_vid")
    with pytest.raises(HTTPException) as exc_info:
        frame_service.get_frame_image_path("two_scene_vid", "two_scene_vid_9999")
    assert exc_info.value.status_code == 404


def test_get_frame_image_path_404s_for_unknown_video():
    with pytest.raises(HTTPException) as exc_info:
        frame_service.get_frame_image_path("no_such_video", "no_such_video_0000")
    assert exc_info.value.status_code == 404


def test_extract_frames_raises_on_invalid_video(tmp_path):
    bogus_path = tmp_path / "not_a_video.mp4"
    bogus_path.write_bytes(b"this is not a real video file")

    with pytest.raises(HTTPException) as exc_info:
        frame_service.extract_frames(bogus_path, "bogus_vid")
    assert exc_info.value.status_code == 500


def test_re_extraction_clears_previous_frames(two_scene_video, static_color_video):
    """Re-running extraction (e.g. after a settings change) shouldn't
    accumulate stale frames from the previous run."""
    frame_service.extract_frames(two_scene_video, "shared_id")
    second_result = frame_service.extract_frames(static_color_video, "shared_id")

    frame_dir = settings.frames_dir / "shared_id"
    on_disk_images = list(frame_dir.glob("frame_*.jpg"))
    assert len(on_disk_images) == len(second_result.frames) == 1


# ---------------------------------------------------------------------
# SSIM scene detection (and the MAD baseline it replaced as default)
# ---------------------------------------------------------------------

import numpy as np


def _textured(seed, size=64):
    """A deterministic, structured (non-flat) grayscale image."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (size, size)).astype(np.uint8)
    import cv2
    return cv2.GaussianBlur(noise, (5, 5), 1.0)


def test_ssim_of_identical_images_is_one():
    img = _textured(0)
    assert frame_service._ssim(img, img) == pytest.approx(1.0)


def test_ssim_is_symmetric_and_low_for_unrelated_images():
    a, b = _textured(0), _textured(1)
    assert frame_service._ssim(a, b) == pytest.approx(frame_service._ssim(b, a))
    assert frame_service._ssim(a, b) < 0.3


def test_brightness_shift_is_same_scene_under_ssim_but_new_scene_under_mad(monkeypatch):
    """The reason to prefer SSIM: a global exposure change moves every
    pixel (MAD sees a huge difference) but leaves the structure intact."""
    original = _textured(0)
    brighter = np.clip(original.astype(np.int16) + 25, 0, 255).astype(np.uint8)

    monkeypatch.setattr(settings, "frame_diff_method", "mad")
    assert frame_service._is_new_scene(brighter, original) is True

    monkeypatch.setattr(settings, "frame_diff_method", "ssim")
    assert frame_service._is_new_scene(brighter, original) is False


def test_small_caption_overlay_is_not_a_new_scene_under_ssim(monkeypatch):
    """A lower-third name caption appearing on the same shot (a real case
    in the CS101 lecture) must not be treated as a scene change."""
    original = _textured(0)
    captioned = original.copy()
    captioned[50:60, 2:30] = 255  # small bright box in one corner

    monkeypatch.setattr(settings, "frame_diff_method", "ssim")
    assert frame_service._is_new_scene(captioned, original) is False


def test_different_content_is_a_new_scene_under_both_methods(monkeypatch):
    a, b = _textured(0), _textured(1)
    for method in ("ssim", "mad"):
        monkeypatch.setattr(settings, "frame_diff_method", method)
        assert frame_service._is_new_scene(b, a) is True


def test_cut_between_flat_frames_is_detected_via_luminance_fallback(monkeypatch):
    """SSIM alone scores a red->blue (gray 76 -> 29) solid cut at ~0.67 and
    would miss it; two flat frames fall back to a luminance comparison."""
    monkeypatch.setattr(settings, "frame_diff_method", "ssim")
    dark = np.full((64, 64), 29, dtype=np.uint8)
    mid = np.full((64, 64), 76, dtype=np.uint8)
    assert frame_service._ssim(dark, mid) > settings.frame_ssim_threshold  # SSIM alone misses it
    assert frame_service._is_new_scene(mid, dark) is True
    # Identical flat frames and a tiny luminance wobble are still "same".
    assert frame_service._is_new_scene(dark, dark) is False
    assert frame_service._is_new_scene(np.full((64, 64), 35, dtype=np.uint8), dark) is False


def test_flat_black_frame_to_structured_frame_is_a_new_scene(monkeypatch):
    monkeypatch.setattr(settings, "frame_diff_method", "ssim")
    black = np.zeros((64, 64), dtype=np.uint8)
    assert frame_service._is_new_scene(_textured(0), black) is True


def test_unknown_diff_method_raises(monkeypatch):
    monkeypatch.setattr(settings, "frame_diff_method", "bogus")
    with pytest.raises(ValueError, match="Unknown frame_diff_method"):
        frame_service._is_new_scene(_textured(0), _textured(1))


def test_manifest_records_diff_method_and_matching_threshold(two_scene_video, monkeypatch):
    monkeypatch.setattr(settings, "frame_diff_method", "ssim")
    frame_set = frame_service.extract_frames(two_scene_video, "two_scene_vid")
    assert frame_set.diff_method == "ssim"
    assert frame_set.diff_threshold == settings.frame_ssim_threshold

    monkeypatch.setattr(settings, "frame_diff_method", "mad")
    frame_set = frame_service.extract_frames(two_scene_video, "two_scene_vid")
    assert frame_set.diff_method == "mad"
    assert frame_set.diff_threshold == settings.frame_diff_threshold


def test_legacy_manifest_without_diff_method_defaults_to_mad(two_scene_video):
    import json
    frame_service.extract_frames(two_scene_video, "two_scene_vid")
    manifest = settings.frames_dir / "two_scene_vid" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    del data["diff_method"]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    assert frame_service.get_frame_set("two_scene_vid").diff_method == "mad"


def test_scene_boundaries_are_timestamps_of_frames_after_the_first(two_scene_video):
    frame_set = frame_service.extract_frames(two_scene_video, "two_scene_vid")
    boundaries = frame_service.get_scene_boundaries("two_scene_vid")
    assert boundaries == [f.timestamp for f in frame_set.frames[1:]]
    assert len(boundaries) == 1 and boundaries[0] == pytest.approx(5.0, abs=0.5)


def test_scene_boundaries_empty_for_single_scene_and_none_when_not_extracted(static_color_video):
    frame_service.extract_frames(static_color_video, "static_vid")
    assert frame_service.get_scene_boundaries("static_vid") == []
    assert frame_service.get_scene_boundaries("never_extracted") is None
