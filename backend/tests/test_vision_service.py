"""
Tests for app.services.vision_service.

Split the same way as test_embedding_service.py:
  - Fast tests using a fake model (persistence, edge cases)
  - One real-model test that actually loads CLIP and checks the ONE
    property that justifies using CLIP at all: images and text land in
    a shared space where a matching pair scores higher than a
    mismatched one.
"""

from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.models.schemas import Frame, FrameSet
from app.services import vision_service


# ---------------------------------------------------------------------
# embed_and_save_frames / persistence: fake model, fast and deterministic
# ---------------------------------------------------------------------

class FakeClipModel:
    """Returns a distinct, deterministic 4-dim vector per input --
    based on the object's identity/length -- enough to verify shape
    and ordering without needing the real CLIP model."""

    def encode(self, inputs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for item in inputs:
            if isinstance(item, Image.Image):
                seed = sum(item.size)
            else:
                seed = len(item)
            vectors.append([float(seed), 0.0, 0.0, 1.0])
        return np.array(vectors, dtype=np.float32)


def _frame_set(video_id="vid1", n=3, tmp_path=None):
    frames = []
    frame_dir = tmp_path / "frames_storage" / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        image_filename = f"frame_{i:04d}.jpg"
        Image.new("RGB", (10 + i, 10 + i), color=(255, 0, 0)).save(frame_dir / image_filename)
        frames.append(
            Frame(
                frame_id=f"{video_id}_{i:04d}",
                video_id=video_id,
                frame_index=i,
                timestamp=float(i * 5),
                image_path=image_filename,
            )
        )
    return FrameSet(
        video_id=video_id,
        created_at=datetime.now(timezone.utc),
        sample_interval_seconds=5.0,
        diff_threshold=10.0,
        frames=frames,
    )


@pytest.fixture(autouse=True)
def _fake_model_and_tmp_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(vision_service, "get_clip_model", lambda: FakeClipModel())
    monkeypatch.setattr(settings, "frames_dir", tmp_path / "frames_storage")
    monkeypatch.setattr(settings, "embeddings_dir", tmp_path)
    monkeypatch.setattr(settings, "clip_model_name", "fake-clip-for-tests")


def test_embed_and_save_frames_persists_correct_shape_and_order(tmp_path):
    frame_set = _frame_set(n=3, tmp_path=tmp_path)
    metadata = vision_service.embed_and_save_frames(frame_set)

    assert metadata.video_id == "vid1"
    assert metadata.frame_count == 3
    assert metadata.dimension == 4
    assert metadata.frame_ids == ["vid1_0000", "vid1_0001", "vid1_0002"]
    assert metadata.model_name == "fake-clip-for-tests"

    vectors = vision_service.load_frame_embeddings("vid1")
    assert vectors.shape == (3, 4)


def test_embed_and_save_frames_handles_empty_frame_set(tmp_path):
    empty = _frame_set(n=0, tmp_path=tmp_path)
    metadata = vision_service.embed_and_save_frames(empty)

    assert metadata.frame_count == 0
    assert metadata.dimension == 0
    assert metadata.frame_ids == []

    vectors = vision_service.load_frame_embeddings("vid1")
    assert vectors.shape == (0, 0)


def test_get_frame_embedding_metadata_returns_none_when_missing():
    assert vision_service.get_frame_embedding_metadata("nonexistent") is None


def test_load_frame_embeddings_returns_none_when_missing():
    assert vision_service.load_frame_embeddings("nonexistent") is None


# ---------------------------------------------------------------------
# Real model: proves the actual joint image-text space claim
# ---------------------------------------------------------------------

def test_real_clip_matches_image_to_its_correct_text_description(monkeypatch, tmp_path):
    # Undo the autouse fake-model patch -- same technique as Phase 2's
    # real-model test: this test's `monkeypatch` fixture is the SAME
    # instance the autouse fixture used (function-scoped caching), so
    # undoing it here reverts get_clip_model to the real implementation.
    monkeypatch.undo()

    red_image_path = tmp_path / "red.jpg"
    blue_image_path = tmp_path / "blue.jpg"
    Image.new("RGB", (224, 224), color=(220, 20, 20)).save(red_image_path)
    Image.new("RGB", (224, 224), color=(20, 20, 220)).save(blue_image_path)

    image_vectors = vision_service.embed_images([red_image_path, blue_image_path])
    red_vec, blue_vec = image_vectors[0], image_vectors[1]

    red_text_vec = vision_service.embed_text_for_visual("a solid red image")
    blue_text_vec = vision_service.embed_text_for_visual("a solid blue image")

    # Both vector families are unit-normalized (cosine similarity ==
    # dot product), same convention as embedding_service.
    for v in (red_vec, blue_vec, red_text_vec, blue_text_vec):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-3)

    # The core claim: the red image should match "a solid red image"
    # better than it matches "a solid blue image" -- proof the image
    # and text embeddings genuinely share a comparable space.
    assert np.dot(red_vec, red_text_vec) > np.dot(red_vec, blue_text_vec)
    assert np.dot(blue_vec, blue_text_vec) > np.dot(blue_vec, red_text_vec)
