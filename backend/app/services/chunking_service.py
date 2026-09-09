"""
Transcript chunking: group consecutive Whisper segments into
embedding-sized chunks, with overlap, while keeping every timestamp
exact (never interpolated).

See docs/architecture.md and this module's design note in the Phase 2
completion report for why this merges real segments instead of cutting
raw text at a fixed character offset.
"""

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


def chunk_transcript(transcript: Transcript) -> List[Chunk]:
    """
    Merge consecutive transcript segments into chunks targeting
    `settings.chunk_target_words`, carrying the last
    `settings.chunk_overlap_segments` segments of each chunk into the
    start of the next one.

    A chunk is always finalized once the accumulated word count meets
    the target, OR once the final segment is reached (whichever comes
    first) -- so every segment ends up in at least one chunk, and no
    trailing partial chunk is ever silently dropped.
    """
    segments = transcript.segments
    if not segments:
        return []

    chunks: List[Chunk] = []
    current: List[TranscriptSegment] = []
    current_words = 0
    chunk_index = 0

    for i, segment in enumerate(segments):
        current.append(segment)
        current_words += _word_count(segment.text)

        is_last_segment = i == len(segments) - 1
        if current_words >= settings.chunk_target_words or is_last_segment:
            chunks.append(_build_chunk(transcript.video_id, chunk_index, current))
            chunk_index += 1

            overlap_count = min(settings.chunk_overlap_segments, len(current))
            current = current[-overlap_count:] if overlap_count > 0 else []
            current_words = sum(_word_count(seg.text) for seg in current)

    logger.info(
        "Chunked transcript %s: %d segments -> %d chunks (target=%d words, overlap=%d segments)",
        transcript.video_id,
        len(segments),
        len(chunks),
        settings.chunk_target_words,
        settings.chunk_overlap_segments,
    )
    return chunks


def _chunkset_path(video_id: str) -> Path:
    return settings.chunks_dir / f"{video_id}.json"


def chunk_and_save(transcript: Transcript) -> ChunkSet:
    chunks = chunk_transcript(transcript)
    chunk_set = ChunkSet(
        video_id=transcript.video_id,
        created_at=datetime.now(timezone.utc),
        chunk_target_words=settings.chunk_target_words,
        chunk_overlap_segments=settings.chunk_overlap_segments,
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
