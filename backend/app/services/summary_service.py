"""
Video summary and key points.

Map-reduce over the transcript, sized for a small local model on CPU:

  1. Split the transcript into sections at segment boundaries.
  2. One LLM call per section writes a single-sentence key point.
  3. One more call writes a short overview from those key points.

As with RAG, every timestamp comes from the transcript, not from the model:
a key point's start/end are the real times of the first and last segment in
its section. The model only writes sentences, so it cannot invent a time.

The result is cached on disk because generating it takes minutes on CPU; a
summary is only recomputed when the caller asks for it.
"""

import math
import re
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import KeyPoint, Transcript, TranscriptSegment, VideoSummary
from app.services import llm_service, transcription_service, video_service

logger = get_logger(__name__)

# A section this short carries nothing worth a sentence (intro music, "thanks
# for watching"); skipping it saves a call and avoids a hallucinated point.
MIN_SECTION_WORDS = 15

POINT_PROMPT = """Below is one part of a video transcript. Write ONE sentence (at most 25 words) stating the main point made in this part.

Rules:
- Use only the transcript below. Do not add outside knowledge.
- Write a plain sentence: no bullet, no label, no quotation marks.

Transcript:
{text}

Main point:"""

OVERVIEW_PROMPT = """Below are the key points of a video, in order. Write a short overview of the whole video in 3 or 4 sentences.

Rules:
- Use only the key points below. Do not add outside knowledge.
- Write plain prose: no bullets, no headings.

Key points:
{points}

Overview:"""


def _summary_path(video_id: str):
    return settings.summaries_dir / f"{video_id}.json"


def get_summary(video_id: str) -> Optional[VideoSummary]:
    """A previously generated summary, or None."""
    path = _summary_path(video_id)
    if not path.exists():
        return None
    return VideoSummary.model_validate_json(path.read_text(encoding="utf-8"))


def _word_count(text: str) -> int:
    return len(text.split())


def split_sections(
    segments: List[TranscriptSegment], section_words: int, max_sections: int
) -> List[List[TranscriptSegment]]:
    """
    Group consecutive segments into sections of roughly `section_words`
    words, never cutting a segment in half. The target grows for long
    videos so there are never more than about `max_sections` sections.
    """
    total = sum(_word_count(s.text) for s in segments)
    if total == 0:
        return []
    target = max(section_words, math.ceil(total / max_sections))
    sections: List[List[TranscriptSegment]] = []
    current: List[TranscriptSegment] = []
    words = 0
    for seg in segments:
        current.append(seg)
        words += _word_count(seg.text)
        if words >= target:
            sections.append(current)
            current, words = [], 0
    if current:
        # A short tail is folded into the previous section instead of
        # becoming a section of its own.
        if sections and words < target / 2:
            sections[-1].extend(current)
        else:
            sections.append(current)
    return sections


def clean_sentence(text: str) -> str:
    """Tidy a model reply into one line: drop labels, bullets and wrapping
    quotes that small models add despite being told not to."""
    text = text.strip()
    text = re.sub(r"^(main point|key point|overview|summary)\s*[:\-]\s*", "", text, flags=re.I)
    text = re.sub(r"^[\-\*•\d.)\s]+", "", text)
    text = " ".join(text.split())
    return text.strip(" \"'")


def generate_summary(video_id: str) -> VideoSummary:
    """
    Build, cache and return the summary. Raises 409 if the video has no
    transcript yet, and RuntimeError (surfaced as 503 by the API) if the LLM
    is unreachable.
    """
    video_service.get_video_metadata(video_id)  # 404s if unknown
    transcript: Optional[Transcript] = transcription_service.get_transcript(video_id)
    if transcript is None or not transcript.segments:
        raise HTTPException(
            status_code=409,
            detail="This video has no transcript yet. Run the text pipeline first.",
        )

    provider = llm_service.get_llm_provider()
    sections = split_sections(
        transcript.segments, settings.summary_section_words, settings.summary_max_sections
    )

    points: List[KeyPoint] = []
    for section in sections:
        text = " ".join(s.text.strip() for s in section)
        if _word_count(text) < MIN_SECTION_WORDS:
            continue
        sentence = clean_sentence(provider.generate(POINT_PROMPT.format(text=text)))
        if not sentence:
            continue
        points.append(
            KeyPoint(start_time=section[0].start_time, end_time=section[-1].end_time, text=sentence)
        )

    if not points:
        raise HTTPException(
            status_code=409, detail="The transcript is too short or empty to summarise."
        )

    numbered = "\n".join(f"{i + 1}. {p.text}" for i, p in enumerate(points))
    overview = clean_sentence(provider.generate(OVERVIEW_PROMPT.format(points=numbered)))

    summary = VideoSummary(
        video_id=video_id,
        created_at=datetime.now(timezone.utc),
        model=settings.ollama_model,
        overview=overview,
        key_points=points,
    )
    settings.summaries_dir.mkdir(parents=True, exist_ok=True)
    _summary_path(video_id).write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Summarised video %s: %d key points", video_id, len(points))
    return summary
