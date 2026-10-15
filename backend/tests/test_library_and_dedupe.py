"""
Tests for deleting a video (library_service + DELETE /api/videos/{id}) and
for recognising an already-ingested URL (video_service.canonical_source_key,
POST /api/videos/from-url idempotence).

All storage directories are redirected into tmp_path, so nothing here can
touch real data, and the vector store is faked at the one function that
deletes -- what is under test is WHICH ids get deleted and which files go,
not ChromaDB itself (that has its own tests).
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.schemas import VideoMetadata
from app.services import vector_store_service, video_service

client = TestClient(app)


@pytest.fixture
def library(tmp_path, monkeypatch):
    for name in ("videos", "audio", "frames", "transcripts", "embeddings", "chunks"):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(settings, f"{name}_dir", d)
    deleted = []
    monkeypatch.setattr(vector_store_service, "delete_video", lambda collection_name, video_id: deleted.append((collection_name, video_id)))
    return deleted


def _make_video(video_id, source_url=None):
    """A video with a file, metadata and one artifact of every kind."""
    metadata = VideoMetadata(
        video_id=video_id, original_filename=f"{video_id}.mp4", stored_filename=f"{video_id}.mp4", extension=".mp4",
        content_type="video/mp4", size_bytes=5, uploaded_at=datetime.now(timezone.utc), source_url=source_url,
    )
    (settings.videos_dir / f"{video_id}.mp4").write_bytes(b"video")
    (settings.videos_dir / f"{video_id}.json").write_text(metadata.model_dump_json(), encoding="utf-8")
    (settings.audio_dir / f"{video_id}.wav").write_bytes(b"audio")
    (settings.transcripts_dir / f"{video_id}.json").write_text("{}", encoding="utf-8")
    (settings.chunks_dir / f"{video_id}.json").write_text("{}", encoding="utf-8")
    for suffix in (".npy", ".json", "_frames.npy", "_frames.json"):
        (settings.embeddings_dir / f"{video_id}{suffix}").write_bytes(b"x")
    frame_dir = settings.frames_dir / video_id
    frame_dir.mkdir()
    (frame_dir / "frame_0000.jpg").write_bytes(b"img")
    (frame_dir / "manifest.json").write_text("{}", encoding="utf-8")


def _files_for(video_id):
    found = []
    for d in (settings.videos_dir, settings.audio_dir, settings.transcripts_dir, settings.chunks_dir, settings.embeddings_dir, settings.frames_dir):
        found += [p for p in d.iterdir() if p.name == video_id or p.name.startswith(f"{video_id}.") or p.name.startswith(f"{video_id}_")]
    return found


# ------------------------------------------------------------------ deletion

def test_delete_removes_every_artifact_and_both_vector_collections(library):
    _make_video("vidaaa")
    assert len(_files_for("vidaaa")) == 10

    response = client.delete("/api/videos/vidaaa")

    assert response.status_code == 204
    assert _files_for("vidaaa") == []
    assert (settings.chroma_collection_name, "vidaaa") in library
    assert (settings.chroma_frames_collection_name, "vidaaa") in library
    assert client.get("/api/videos/vidaaa").status_code == 404


def test_deleting_one_video_never_touches_another_whose_id_shares_a_prefix(library):
    _make_video("vid1")
    _make_video("vid10")

    assert client.delete("/api/videos/vid1").status_code == 204

    assert _files_for("vid1") == []
    assert len(_files_for("vid10")) == 10  # every file of the look-alike id survives
    assert client.get("/api/videos/vid10").status_code == 200
    assert all(video_id == "vid1" for _, video_id in library)


def test_delete_unknown_video_is_404(library):
    assert client.delete("/api/videos/no_such_video").status_code == 404


def test_delete_rejects_an_id_that_is_not_a_plain_identifier(library):
    assert client.delete("/api/videos/bad.id").status_code == 400


# ------------------------------------------------------------ URL identity

@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=CNFK86hJRfE",
    "https://m.youtube.com/watch?v=CNFK86hJRfE&pp=ugUEEgJlbg%3D%3D&ra=m",
    "https://www.youtube.com/watch?time_continue=13&v=CNFK86hJRfE&embeds_referring_euri=https%3A%2F%2Fwww.bing.com",
    "https://youtu.be/CNFK86hJRfE?t=6",
    "https://www.youtube.com/watch?v=CNFK86hJRfE&t=1shttps://www.youtube.com/watch?v=CNFK86hJRfE&t=1s",  # pasted twice
])
def test_every_spelling_of_a_youtube_link_has_the_same_key(url):
    assert video_service.canonical_source_key(url) == "youtube:CNFK86hJRfE"


def test_different_videos_and_non_youtube_urls_get_distinct_stable_keys():
    a = video_service.canonical_source_key("https://www.youtube.com/watch?v=AAAAAAAAAAA")
    b = video_service.canonical_source_key("https://www.youtube.com/watch?v=BBBBBBBBBBB")
    assert a != b
    assert video_service.canonical_source_key("https://example.com/v/clip.mp4/#t=3") == "https://example.com/v/clip.mp4"


def test_adding_a_url_already_in_the_library_returns_it_without_downloading(library, monkeypatch):
    _make_video("existing1", source_url="https://www.youtube.com/watch?v=CNFK86hJRfE")

    def must_not_download(url):
        raise AssertionError("a duplicate URL must not trigger a download")

    monkeypatch.setattr(video_service, "save_video_from_url", must_not_download)
    response = client.post("/api/videos/from-url", json={"url": "https://youtu.be/CNFK86hJRfE?t=30"})

    assert response.status_code == 200  # existing video, not 201 Created
    assert response.json()["video_id"] == "existing1"


def test_a_new_url_is_still_downloaded_and_created(library, monkeypatch):
    _make_video("existing1", source_url="https://www.youtube.com/watch?v=CNFK86hJRfE")
    fresh = VideoMetadata(
        video_id="fresh1", original_filename="f.mp4", stored_filename="fresh1.mp4", extension=".mp4",
        size_bytes=1, uploaded_at=datetime.now(timezone.utc), source_url="https://www.youtube.com/watch?v=ZZZZZZZZZZZ",
    )
    monkeypatch.setattr(video_service, "save_video_from_url", lambda url: fresh)
    response = client.post("/api/videos/from-url", json={"url": "https://www.youtube.com/watch?v=ZZZZZZZZZZZ"})
    assert response.status_code == 201 and response.json()["video_id"] == "fresh1"
