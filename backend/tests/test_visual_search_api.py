"""
Full-pipeline integration test for visual search: real ChromaDB (temp
dir) + real frame_service/vision_service/retrieval_service code paths,
with only the CLIP *model* faked (deterministic, controlled vectors) so
the test is fast and exactly predictable -- while everything downstream
of "image/text -> vector" is genuinely exercised, including the actual
HTTP endpoints.

Mirrors test_search_api.py's structure for text search; this is the
test that proves Phase 5's frame -> embed -> index -> search pipeline
composes correctly end-to-end, using a SEPARATE Chroma collection from
text chunks.
"""

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app
from app.models.schemas import Frame, FrameSet
from app.services import vector_store_service, vision_service

client = TestClient(app)

# Deterministic 2D "CLIP embeddings": red images/text -> [1,0],
# blue images/text -> [0,1]. Lets us construct queries with predictable,
# checkable rankings without a real CLIP model.
RED = (220, 20, 20)
BLUE = (20, 20, 220)


class FakeClipModel:
    def encode(self, inputs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for item in inputs:
            if isinstance(item, Image.Image):
                pixel = item.getpixel((0, 0))
                vectors.append([1.0, 0.0] if pixel == RED else [0.0, 1.0])
            else:
                vectors.append([1.0, 0.0] if "red" in item else [0.0, 1.0])
        return np.array(vectors, dtype=np.float32)


def _index_video_frame(video_id: str, color: tuple, monkeypatch) -> None:
    """Run the real extract-frames-manifest -> embed -> index-frames
    pipeline for a single single-frame video, via the actual service
    functions and HTTP endpoints."""
    frame_dir = settings.frames_dir / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(frame_dir / "frame_0000.jpg")

    frame = Frame(
        frame_id=f"{video_id}_0000", video_id=video_id, frame_index=0,
        timestamp=0.0, image_path="frame_0000.jpg",
    )
    frame_set = FrameSet(
        video_id=video_id, created_at=datetime.now(timezone.utc),
        sample_interval_seconds=5.0, diff_threshold=10.0, frames=[frame],
    )
    (frame_dir / "manifest.json").write_text(frame_set.model_dump_json(indent=2), encoding="utf-8")

    embed_response = client.post(f"/api/videos/{video_id}/frame-embeddings")
    assert embed_response.status_code == 200, embed_response.text

    index_response = client.post(f"/api/videos/{video_id}/index-frames")
    assert index_response.status_code == 200, index_response.text


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_service, "get_clip_model", lambda: FakeClipModel())
    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / "chroma_db")
    monkeypatch.setattr(vector_store_service, "_client", None)
    monkeypatch.setattr(vector_store_service, "_collections", {})

    yield

    import shutil
    for video_id in ("visual_test_v1", "visual_test_v2"):
        frame_dir = settings.frames_dir / video_id
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        (settings.embeddings_dir / f"{video_id}_frames.json").unlink(missing_ok=True)
        (settings.embeddings_dir / f"{video_id}_frames.npy").unlink(missing_ok=True)
    monkeypatch.setattr(vector_store_service, "_client", None)
    monkeypatch.setattr(vector_store_service, "_collections", {})


def test_index_frames_endpoint_requires_frames_and_embeddings():
    response = client.post("/api/videos/no_such_video/index-frames")
    assert response.status_code == 404


def test_visual_search_post_ranks_matching_video_first(monkeypatch):
    _index_video_frame("visual_test_v1", RED, monkeypatch)
    _index_video_frame("visual_test_v2", BLUE, monkeypatch)

    response = client.post("/api/search/visual", json={"query": "a red square", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "a red square"
    assert len(body["results"]) == 2
    assert body["results"][0]["video_id"] == "visual_test_v1"
    assert body["results"][0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert body["results"][0]["timestamp"] == 0.0
    assert body["results"][0]["image_path"] == "frame_0000.jpg"
    assert body["results"][1]["video_id"] == "visual_test_v2"
    assert body["results"][0]["score"] > body["results"][1]["score"]


def test_visual_search_get_matches_post(monkeypatch):
    _index_video_frame("visual_test_v1", RED, monkeypatch)
    _index_video_frame("visual_test_v2", BLUE, monkeypatch)

    response = client.get("/api/search/visual", params={"query": "a blue square", "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["video_id"] == "visual_test_v2"


def test_visual_search_video_id_filter_restricts_results(monkeypatch):
    _index_video_frame("visual_test_v1", RED, monkeypatch)
    _index_video_frame("visual_test_v2", BLUE, monkeypatch)

    # Query semantically closer to v1's content, but filtered to v2 --
    # should still only return v2's frame despite the weaker match.
    response = client.post(
        "/api/search/visual", json={"query": "a red square", "video_id": "visual_test_v2"}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["video_id"] == "visual_test_v2"


def test_text_and_visual_search_use_independent_collections(monkeypatch):
    """The core proof of Phase 5 Step 3's design: indexing frames must
    NOT show up in (or interfere with) the text-chunk collection."""
    _index_video_frame("visual_test_v1", RED, monkeypatch)

    text_count = vector_store_service.count(settings.chroma_collection_name)
    frame_count = vector_store_service.count(settings.chroma_frames_collection_name)
    assert frame_count == 1
    assert text_count == 0
