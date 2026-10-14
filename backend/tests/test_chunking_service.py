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


# ---------------------------------------------------------------------
# Content-aligned chunking (visual scene boundaries steer chunk cuts)
# ---------------------------------------------------------------------

@pytest.fixture
def scene_settings(monkeypatch):
    """Word ceiling out of the way, 8-word floor, 5s sampling interval:
    segment i starts at 5*i, so a boundary b maps to the segment whose
    start is nearest (b - 2.5)."""
    monkeypatch.setattr(settings, "chunk_target_words", 1000)
    monkeypatch.setattr(settings, "chunk_min_words", 8)
    monkeypatch.setattr(settings, "frame_sample_interval_seconds", 5.0)


def test_no_boundaries_is_identical_to_plain_word_count_chunking():
    segments = [_segment(i) for i in range(12)]
    plain = chunking_service.chunk_transcript(_transcript(segments))
    assert chunking_service.chunk_transcript(_transcript(segments), scene_boundaries=[]) == plain
    assert chunking_service.chunk_transcript(_transcript(segments), scene_boundaries=None) == plain


def test_scene_boundary_cuts_chunk_at_nearest_segment_start_without_overlap(scene_settings):
    segments = [_segment(i) for i in range(10)]  # 4 words each, starts 0,5,...,45
    # Boundary 27.5 -> estimate 25.0 -> segment 5 (start 25).
    chunks = chunking_service.chunk_transcript(_transcript(segments), scene_boundaries=[27.5])

    assert len(chunks) == 2
    assert chunks[0].end_time == segments[4].end_time
    assert chunks[1].start_time == segments[5].start_time
    # A scene change is a deliberate boundary: no overlap carried across it.
    assert "seg4word0" not in chunks[1].text
    assert "seg5word0" not in chunks[0].text


def test_boundary_snaps_to_the_nearest_segment_start_never_interpolating(scene_settings):
    segments = [_segment(i) for i in range(10)]
    # Estimate 23.5 is 1.5s from start 25 and 3.5s from start 20 -> segment 5.
    chunks = chunking_service.chunk_transcript(_transcript(segments), scene_boundaries=[26.0])
    assert chunks[1].start_time == segments[5].start_time
    # Every timestamp in every chunk is a real segment boundary.
    real_starts = {s.start_time for s in segments}
    real_ends = {s.end_time for s in segments}
    for chunk in chunks:
        assert chunk.start_time in real_starts and chunk.end_time in real_ends


def test_word_floor_stops_rapid_cuts_from_shredding_the_transcript(scene_settings, monkeypatch):
    monkeypatch.setattr(settings, "chunk_min_words", 12)  # >= 3 segments
    segments = [_segment(i) for i in range(20)]
    every_segment = [5.0 * i + 2.5 for i in range(1, 20)]  # a "scene change" at every segment
    chunks = chunking_service.chunk_transcript(_transcript(segments), scene_boundaries=every_segment)

    assert len(chunks) > 1
    assert all(c.word_count >= 12 for c in chunks)
    assert sum(c.word_count for c in chunks) == 20 * 4  # nothing dropped or duplicated


def test_word_ceiling_still_splits_a_long_scene(monkeypatch):
    monkeypatch.setattr(settings, "chunk_target_words", 10)
    monkeypatch.setattr(settings, "chunk_min_words", 8)
    monkeypatch.setattr(settings, "frame_sample_interval_seconds", 5.0)
    segments = [_segment(i) for i in range(20)]
    # One scene boundary far away; the long scene before it must still split.
    chunks = chunking_service.chunk_transcript(_transcript(segments), scene_boundaries=[92.5])
    assert len(chunks) > 2
    # Ceiling cuts keep the original overlap behaviour.
    assert "seg2word0" in chunks[0].text and "seg2word0" in chunks[1].text


def test_scene_cut_right_after_a_ceiling_cut_drops_the_carried_overlap(monkeypatch):
    monkeypatch.setattr(settings, "chunk_target_words", 10)
    monkeypatch.setattr(settings, "chunk_min_words", 4)
    monkeypatch.setattr(settings, "frame_sample_interval_seconds", 5.0)
    segments = [_segment(i) for i in range(6)]
    # Ceiling hits after seg2 (12 words). Scene boundary maps to segment 3,
    # so seg2 (before the change) must NOT be carried into the new scene.
    chunks = chunking_service.chunk_transcript(_transcript(segments), scene_boundaries=[17.5])
    assert chunks[0].end_time == segments[2].end_time
    assert chunks[1].start_time == segments[3].start_time
    assert "seg2word0" not in chunks[1].text


def test_boundary_too_close_to_the_end_is_ignored(scene_settings):
    segments = [_segment(i) for i in range(10)]
    # Maps to the last segment: only 4 words remain (< 8 floor) -> no sliver chunk.
    chunks = chunking_service.chunk_transcript(_transcript(segments), scene_boundaries=[47.5])
    assert len(chunks) == 1


def test_boundary_at_the_very_start_is_ignored(scene_settings):
    segments = [_segment(i) for i in range(10)]
    chunks = chunking_service.chunk_transcript(_transcript(segments), scene_boundaries=[1.0])
    assert len(chunks) == 1


def test_chunk_and_save_records_alignment_metadata(tmp_path, monkeypatch, scene_settings):
    monkeypatch.setattr(settings, "chunks_dir", tmp_path)
    segments = [_segment(i) for i in range(10)]
    t = _transcript(segments)

    aligned = chunking_service.chunk_and_save(t, scene_boundaries=[27.5, 1.0])
    assert aligned.alignment == "scene_aligned"
    assert aligned.scene_boundaries_available == 2
    assert aligned.scene_cuts_applied == 1  # the boundary at t=1.0 maps to index 0 -> unused
    assert chunking_service.get_chunk_set("vid1").alignment == "scene_aligned"

    plain = chunking_service.chunk_and_save(t)
    assert plain.alignment == "word_count"
    assert plain.scene_cuts_applied == 0
