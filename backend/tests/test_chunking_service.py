"""
Tests for app.services.chunking_service.

Pure-function tests against hand-built Transcript objects -- no need
for real Whisper output. `chunk_target_words`/`chunk_overlap_segments`
are monkeypatched to small values so test data stays small and the
chunk boundaries are easy to reason about by hand.
"""

from datetime import datetime, timezone

import pytest

from app.core.config import settings
from app.models.schemas import Transcript, TranscriptSegment
from app.services import chunking_service


def _segment(i: int, words: int = 4) -> TranscriptSegment:
    """A 4-second segment starting at i*5s, with `words` unique words
    (so we can later check via substring whether a given segment's
    text ended up inside a chunk)."""
    text = " ".join(f"seg{i}word{w}" for w in range(words))
    return TranscriptSegment(
        segment_id=i, start_time=float(i * 5), end_time=float(i * 5 + 4), text=text
    )


def _transcript(segments):
    return Transcript(
        video_id="vid1",
        language="en",
        duration_seconds=segments[-1].end_time if segments else 0.0,
        model_size="base",
        created_at=datetime.now(timezone.utc),
        segments=segments,
    )


@pytest.fixture(autouse=True)
def _small_chunk_settings(monkeypatch):
    monkeypatch.setattr(settings, "chunk_target_words", 10)
    monkeypatch.setattr(settings, "chunk_overlap_segments", 1)


def test_empty_transcript_produces_no_chunks():
    assert chunking_service.chunk_transcript(_transcript([])) == []


def test_short_transcript_produces_a_single_chunk():
    # 3 segments x 4 words = 12 words, but it's also the last segment,
    # so it should all land in one chunk regardless of the 10-word target.
    segments = [_segment(i) for i in range(3)]
    chunks = chunking_service.chunk_transcript(_transcript(segments))

    assert len(chunks) == 1
    assert chunks[0].start_time == segments[0].start_time
    assert chunks[0].end_time == segments[-1].end_time
    assert chunks[0].chunk_id == "vid1_0000"
    assert chunks[0].word_count == 12


def test_chunking_splits_at_word_target_with_overlap():
    # 5 segments x 4 words, target=10, overlap=1 segment.
    # Expect: chunk0 = segs[0,1,2] (12 words >= 10 -> finalize)
    #         chunk1 = segs[2,3,4] (seg2 carried over as overlap)
    segments = [_segment(i) for i in range(5)]
    chunks = chunking_service.chunk_transcript(_transcript(segments))

    assert len(chunks) == 2

    chunk0, chunk1 = chunks
    assert chunk0.chunk_index == 0
    assert chunk0.start_time == segments[0].start_time
    assert chunk0.end_time == segments[2].end_time

    assert chunk1.chunk_index == 1
    # Overlap: chunk1 starts at segment 2's time, not segment 3's --
    # segment 2 was carried over as context.
    assert chunk1.start_time == segments[2].start_time
    assert chunk1.end_time == segments[4].end_time
    assert "seg2word0" in chunk1.text  # the overlapping segment's text
    assert "seg2word0" in chunk0.text  # ...and it's still in chunk0 too

    # chunk_ids are unique and video-prefixed (future vector DB IDs).
    assert chunk0.chunk_id == "vid1_0000"
    assert chunk1.chunk_id == "vid1_0001"


def test_zero_overlap_means_no_shared_segments(monkeypatch):
    monkeypatch.setattr(settings, "chunk_overlap_segments", 0)
    segments = [_segment(i) for i in range(5)]
    chunks = chunking_service.chunk_transcript(_transcript(segments))

    assert len(chunks) == 2
    assert "seg2word0" not in chunks[1].text


def test_chunk_and_save_persists_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chunks_dir", tmp_path)
    segments = [_segment(i) for i in range(5)]
    transcript = _transcript(segments)

    saved = chunking_service.chunk_and_save(transcript)
    assert saved.video_id == "vid1"
    assert len(saved.chunks) == 2

    reloaded = chunking_service.get_chunk_set("vid1")
    assert reloaded is not None
    assert reloaded.chunks[0].start_time == saved.chunks[0].start_time
    assert reloaded.chunk_target_words == 10


def test_get_chunk_set_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chunks_dir", tmp_path)
    assert chunking_service.get_chunk_set("nonexistent") is None
