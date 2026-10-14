"""
Transcript chunking: group consecutive Whisper segments into
embedding-sized chunks, with overlap, while keeping every timestamp
exact (never interpolated).

See docs/architecture.md and this module's design note in the Phase 2
completion report for why this merges real segments instead of cutting
raw text at a fixed character offset.

CONTENT-ALIGNED CHUNKING: plain word-count chunking is blind to what is
on screen, so a chunk can straddle two slides (its embedding then blends
two topics, and its citation points at an arbitrary cutoff). When the
caller supplies the video's visual scene boundaries (from frame_service),
chunks are additionally cut at those boundaries -- snapped to the nearest
transcript segment start so timestamps stay exact -- subject to a word
floor (so rapid cuts cannot shred the transcript) and the original word
ceiling (so a long scene is still split).
"""

from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import Chunk, ChunkSet, Transcript, TranscriptSegment

logger = get_logger(__name__)


def _word_count(text: str) -> int:
    return len(text.split())


def _build_chunk(video_id: str, chunk_index: int, segments: List[TranscriptSegment]) -> Chunk:
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return Chunk(
        chunk_id=f"{video_id}_{chunk_index:04d}",
        video_id=video_id,
        chunk_index=chunk_index,
        start_time=segments[0].start_time,
        end_time=segments[-1].end_time,
        text=text,
        word_count=_word_count(text),
    )


def _scene_cut_indexes(
    segments: List[TranscriptSegment], scene_boundaries: List[float]
) -> set[int]:
    """
    Map each scene boundary to the index of the segment a chunk should
    START at.

    A boundary timestamp is the first SAMPLED frame of the new scene, so
    the real change happened somewhere in the preceding sample interval;
    its midpoint is the best estimate. The chunk then starts at the
    segment whose start_time is nearest that estimate -- always a real
    segment start, so no timestamp is ever interpolated. Index 0 is
    never a cut (a chunk can't start before the transcript does).
    """
    if not scene_boundaries or not segments:
        return set()
    starts = [s.start_time for s in segments]
    half_interval = settings.frame_sample_interval_seconds / 2
    cuts: set[int] = set()
    for boundary in scene_boundaries:
        estimate = boundary - half_interval
        pos = bisect_left(starts, estimate)
        candidates = [p for p in (pos - 1, pos) if 0 <= p < len(starts)]
        nearest = min(candidates, key=lambda p: abs(starts[p] - estimate))
        if nearest > 0:
            cuts.add(nearest)
    return cuts


def _chunk_with_stats(
    transcript: Transcript, scene_boundaries: Optional[List[float]] = None
) -> tuple[List[Chunk], int]:
    segments = transcript.segments
    if not segments:
        return [], 0

    scene_cuts = _scene_cut_indexes(segments, scene_boundaries or [])
    # Words still ahead of segment i (inclusive), so we never cut so late
    # that the remainder would be a sliver of a chunk.
    remaining_words = [0] * (len(segments) + 1)
    for k in range(len(segments) - 1, -1, -1):
        remaining_words[k] = remaining_words[k + 1] + _word_count(segments[k].text)

    chunks: List[Chunk] = []
    current: List[TranscriptSegment] = []
    current_words = 0
    fresh_count = 0  # segments added since the last chunk was finalized
    fresh_words = 0
    chunk_index = 0
    cuts_applied = 0

    for i, segment in enumerate(segments):
        if i in scene_cuts:
            if fresh_count == 0:
                # A chunk just ended exactly here; the only thing in
                # `current` is carried-over overlap from BEFORE the scene
                # change -- drop it so the new scene starts clean.
                current, current_words = [], 0
                cuts_applied += 1
            elif (
                fresh_words >= settings.chunk_min_words
                and remaining_words[i] >= settings.chunk_min_words
            ):
                chunks.append(_build_chunk(transcript.video_id, chunk_index, current))
                chunk_index += 1
                current, current_words = [], 0  # no overlap across a real scene change
                fresh_count = fresh_words = 0
                cuts_applied += 1

        words = _word_count(segment.text)
        current.append(segment)
        current_words += words
        fresh_count += 1
        fresh_words += words

        is_last_segment = i == len(segments) - 1
        if current_words >= settings.chunk_target_words or is_last_segment:
            chunks.append(_build_chunk(transcript.video_id, chunk_index, current))
            chunk_index += 1

            overlap_count = min(settings.chunk_overlap_segments, len(current))
            current = current[-overlap_count:] if overlap_count > 0 else []
            current_words = sum(_word_count(seg.text) for seg in current)
            fresh_count = fresh_words = 0

    return chunks, cuts_applied


def chunk_transcript(
    transcript: Transcript, scene_boundaries: Optional[List[float]] = None
) -> List[Chunk]:
    """
    Merge consecutive transcript segments into chunks targeting
    `settings.chunk_target_words`, carrying the last
    `settings.chunk_overlap_segments` segments of each chunk into the
    start of the next one.

    A chunk is always finalized once the accumulated word count meets
    the target, OR once the final segment is reached (whichever comes
    first) -- so every segment ends up in at least one chunk, and no
    trailing partial chunk is ever silently dropped.

    With `scene_boundaries` (seconds, from frame_service), a chunk is
    ALSO cut where a visual scene changes, provided it already holds at
    least `settings.chunk_min_words` words and enough transcript remains
    to form another such chunk. A scene cut carries no overlap: overlap
    exists to preserve an idea that straddles an ARBITRARY cut, whereas a
    scene change is a deliberate content boundary. Without boundaries
    this is exactly the original word-count algorithm.
    """
    chunks, _ = _chunk_with_stats(transcript, scene_boundaries)
    logger.info(
        "Chunked transcript %s: %d segments -> %d chunks (target=%d words, overlap=%d segments, scene boundaries=%d)",
        transcript.video_id,
        len(transcript.segments),
        len(chunks),
        settings.chunk_target_words,
        settings.chunk_overlap_segments,
        len(scene_boundaries or []),
    )
    return chunks


def _chunkset_path(video_id: str) -> Path:
    return settings.chunks_dir / f"{video_id}.json"


def chunk_and_save(
    transcript: Transcript, scene_boundaries: Optional[List[float]] = None
) -> ChunkSet:
    chunks, cuts_applied = _chunk_with_stats(transcript, scene_boundaries)
    chunk_set = ChunkSet(
        video_id=transcript.video_id,
        created_at=datetime.now(timezone.utc),
        chunk_target_words=settings.chunk_target_words,
        chunk_overlap_segments=settings.chunk_overlap_segments,
        alignment="scene_aligned" if scene_boundaries else "word_count",
        scene_boundaries_available=len(scene_boundaries or []),
        scene_cuts_applied=cuts_applied,
        chunks=chunks,
    )
    _chunkset_path(transcript.video_id).write_text(
        chunk_set.model_dump_json(indent=2), encoding="utf-8"
    )
    return chunk_set


def get_chunk_set(video_id: str) -> Optional[ChunkSet]:
    path = _chunkset_path(video_id)
    if not path.exists():
        return None
    return ChunkSet.model_validate_json(path.read_text(encoding="utf-8"))
