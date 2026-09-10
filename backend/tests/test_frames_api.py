"""
Tests for POST/GET /api/videos/{video_id}/frames and the frame image
endpoint. Uploads a real tiny synthetic video through the actual
upload endpoint first, since frame extraction needs a real video_id
resolvable to a file on disk (via video_service).
"""

import subprocess

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


def _make_test_video(tmp_path):
    video_path = tmp_path / "test.mp4"
    subprocess.run(
        [
            settings.ffmpeg_binary, "-y",
            "-f", "lavfi", "-i", "color=c=green:size=64x64:duration=3",
            "-r", "10",
            str(video_path),
        ],
        capture_output=True,
        check=True,
    )
    return video_path


@pytest.fixture
def uploaded_video_id(tmp_path):
    video_path = _make_test_video(tmp_path)
    with open(video_path, "rb") as f:
        response = client.post(
            "/api/videos/upload",
            files={"file": ("test.mp4", f, "video/mp4")},
        )
    video_id = response.json()["video_id"]
    yield video_id

    # Cleanup: uploaded video file + metadata + any extracted frames.
    for path in settings.videos_dir.glob(f"{video_id}*"):
        path.unlink(missing_ok=True)
    import shutil
    frame_dir = settings.frames_dir / video_id
    if frame_dir.exists():
        shutil.rmtree(frame_dir)


def test_frames_endpoint_requires_existing_video():
    response = client.post("/api/videos/no_such_video/frames")
    assert response.status_code == 404


def test_get_frames_before_extraction_returns_404(uploaded_video_id):
    response = client.get(f"/api/videos/{uploaded_video_id}/frames")
    assert response.status_code == 404


def test_extract_then_get_frames_roundtrip(uploaded_video_id):
    post_response = client.post(f"/api/videos/{uploaded_video_id}/frames")
    assert post_response.status_code == 200
    posted = post_response.json()

    assert posted["video_id"] == uploaded_video_id
    assert len(posted["frames"]) >= 1

    get_response = client.get(f"/api/videos/{uploaded_video_id}/frames")
    assert get_response.status_code == 200
    assert get_response.json() == posted


def test_get_frame_image_returns_jpeg(uploaded_video_id):
    post_response = client.post(f"/api/videos/{uploaded_video_id}/frames")
    frame_id = post_response.json()["frames"][0]["frame_id"]

    image_response = client.get(f"/api/videos/{uploaded_video_id}/frames/{frame_id}/image")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"
    assert len(image_response.content) > 0


def test_get_frame_image_404s_for_unknown_frame(uploaded_video_id):
    client.post(f"/api/videos/{uploaded_video_id}/frames")
    response = client.get(f"/api/videos/{uploaded_video_id}/frames/{uploaded_video_id}_9999/image")
    assert response.status_code == 404
