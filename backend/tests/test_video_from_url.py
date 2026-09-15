"""
Tests for app.services.video_service.save_video_from_url.

yt_dlp.YoutubeDL is entirely mocked -- these tests never touch the
network. A fake implementation simulates yt-dlp's two-call contract
(extract_info for probing, download for the actual fetch) with
configurable info/failure modes, so each validation branch (duration
cap, live-stream rejection, network failure, missing output) can be
exercised deterministically. Live network verification is done
separately via a manual/real-server check, not in the automated suite.
"""

from pathlib import Path

import pytest
import yt_dlp
from fastapi import HTTPException

from app.core.config import settings
from app.services import video_service


def _make_fake_ytdl(info: dict, download_error: bool = False, download_writes_no_file: bool = False):
    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            return info

        def download(self, urls):
            if download_writes_no_file:
                return  # simulate yt-dlp silently not producing the expected file
            output_path = Path(self.opts["outtmpl"].replace("%(ext)s", "mp4"))
            if download_error:
                # A real partial download leaves a file behind before
                # failing -- write one so the cleanup path is genuinely
                # exercised, not just trivially true.
                output_path.write_bytes(b"partial garbage")
                raise yt_dlp.utils.DownloadError("simulated network failure")
            output_path.write_bytes(b"fake mp4 bytes")

    return FakeYoutubeDL


@pytest.fixture(autouse=True)
def _cleanup():
    before = set(settings.videos_dir.glob("*"))
    yield
    after = set(settings.videos_dir.glob("*"))
    for path in after - before:
        path.unlink(missing_ok=True)


def test_downloads_and_saves_metadata(monkeypatch):
    fake_info = {"title": "A Great Lecture", "duration": 600, "is_live": False}
    monkeypatch.setattr(video_service.yt_dlp, "YoutubeDL", _make_fake_ytdl(fake_info))

    metadata = video_service.save_video_from_url("https://youtube.com/watch?v=abc123")

    assert metadata.original_filename == "A Great Lecture.mp4"
    assert metadata.extension == ".mp4"
    assert metadata.content_type == "video/mp4"
    assert metadata.source_url == "https://youtube.com/watch?v=abc123"
    assert metadata.status == "uploaded"

    stored_path = settings.videos_dir / metadata.stored_filename
    assert stored_path.exists()
    assert stored_path.read_bytes() == b"fake mp4 bytes"
    assert (settings.videos_dir / f"{metadata.video_id}.json").exists()


def test_rejects_video_exceeding_duration_limit(monkeypatch):
    too_long = settings.max_youtube_duration_seconds + 1
    fake_info = {"title": "Very Long Video", "duration": too_long, "is_live": False}
    monkeypatch.setattr(video_service.yt_dlp, "YoutubeDL", _make_fake_ytdl(fake_info))

    with pytest.raises(HTTPException) as exc_info:
        video_service.save_video_from_url("https://youtube.com/watch?v=long")
    assert exc_info.value.status_code == 400
    assert "minutes long" in exc_info.value.detail


def test_rejects_live_streams(monkeypatch):
    fake_info = {"title": "Live Stream", "duration": None, "is_live": True}
    monkeypatch.setattr(video_service.yt_dlp, "YoutubeDL", _make_fake_ytdl(fake_info))

    with pytest.raises(HTTPException) as exc_info:
        video_service.save_video_from_url("https://youtube.com/watch?v=live123")
    assert exc_info.value.status_code == 400
    assert "Live streams" in exc_info.value.detail


def test_raises_400_when_video_info_cannot_be_fetched(monkeypatch):
    class FailingProbeYoutubeDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            raise yt_dlp.utils.DownloadError("Unsupported URL")

    monkeypatch.setattr(video_service.yt_dlp, "YoutubeDL", FailingProbeYoutubeDL)

    with pytest.raises(HTTPException) as exc_info:
        video_service.save_video_from_url("not-a-real-url")
    assert exc_info.value.status_code == 400


def test_raises_502_on_download_failure_and_cleans_up_partial_file(monkeypatch):
    fake_info = {"title": "Flaky Video", "duration": 60, "is_live": False}
    monkeypatch.setattr(
        video_service.yt_dlp, "YoutubeDL", _make_fake_ytdl(fake_info, download_error=True)
    )

    before = set(settings.videos_dir.glob("*"))
    with pytest.raises(HTTPException) as exc_info:
        video_service.save_video_from_url("https://youtube.com/watch?v=flaky")
    assert exc_info.value.status_code == 502

    # The partial file the fake download wrote before failing must be gone.
    after = set(settings.videos_dir.glob("*"))
    assert after == before


def test_raises_500_when_output_file_missing_after_download(monkeypatch):
    fake_info = {"title": "Ghost Video", "duration": 60, "is_live": False}
    monkeypatch.setattr(
        video_service.yt_dlp, "YoutubeDL", _make_fake_ytdl(fake_info, download_writes_no_file=True)
    )

    with pytest.raises(HTTPException) as exc_info:
        video_service.save_video_from_url("https://youtube.com/watch?v=ghost")
    assert exc_info.value.status_code == 500


def test_never_trusts_remote_title_as_a_filesystem_path(monkeypatch):
    """A malicious/weird title must never influence the stored path --
    only the generated video_id does."""
    fake_info = {"title": "../../evil", "duration": 60, "is_live": False}
    monkeypatch.setattr(video_service.yt_dlp, "YoutubeDL", _make_fake_ytdl(fake_info))

    metadata = video_service.save_video_from_url("https://youtube.com/watch?v=weird")

    assert metadata.stored_filename == f"{metadata.video_id}.mp4"
    assert (settings.videos_dir / metadata.stored_filename).exists()
    # The literal weird title is preserved only as a display field.
    assert metadata.original_filename == "../../evil.mp4"
