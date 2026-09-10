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
