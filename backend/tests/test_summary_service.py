"""
Tests for the video summary / key points feature.

The LLM is a stub that records its prompts, so what is under test is our own
logic: how the transcript is sectioned, that timestamps come from the
transcript rather than the model, how messy model replies are cleaned, and
caching / error behaviour -- not the quality of any real model.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.schemas import Transcript, TranscriptSegment, VideoMetadata
from app.services import llm_service, summary_service, transcription_service, video_service

client = TestClient(app)


def _seg(i, words, start):
    return TranscriptSegment(segment_id=i, start_time=start, end_time=start + 10.0, text=" ".join(["word"] * words))


def test_split_sections_groups_by_word_target_without_cutting_segments():
    segs = [_seg(i, 40, i * 10.0) for i in range(10)]  # 400 words
    sections = summary_service.split_sections(segs, section_words=100, max_sections=10)
    assert [len(s) for s in sections] == [3, 3, 4]  # the 1-segment tail (40 words < 50) is folded in
    assert sum(len(s) for s in sections) == 10  # every segment used exactly once


def test_split_sections_caps_the_number_of_sections_for_long_videos():
    segs = [_seg(i, 50, i * 10.0) for i in range(200)]  # 10,000 words
    sections = summary_service.split_sections(segs, section_words=100, max_sections=8)
    assert len(sections) <= 9  # the target widens instead of adding calls


def test_split_sections_folds_a_short_tail_into_the_previous_section():
    segs = [_seg(0, 100, 0.0), _seg(1, 100, 10.0), _seg(2, 5, 20.0)]
    sections = summary_service.split_sections(segs, section_words=100, max_sections=10)
    assert len(sections) == 2 and sections[-1][-1].segment_id == 2


def test_split_sections_empty_transcript():
    assert summary_service.split_sections([], 100, 10) == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Main point: The speaker explains X.", "The speaker explains X."),
        ('"The speaker explains X."', "The speaker explains X."),
        ("- The speaker explains X.", "The speaker explains X."),
        ("1. The speaker explains X.", "The speaker explains X."),
        ("  Overview:\n  Line one\n  line two  ", "Line one line two"),
    ],
)
def test_clean_sentence(raw, expected):
    assert summary_service.clean_sentence(raw) == expected


class StubLLM(llm_service.LLMProvider):
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "Overview: All about it." if "Key points:" in prompt else "Main point: It explains a thing."


@pytest.fixture
def stored(tmp_path, monkeypatch):
    for name in ("videos", "transcripts", "summaries"):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(settings, f"{name}_dir", d)
    monkeypatch.setattr(settings, "summary_section_words", 100)
    llm = StubLLM()
    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: llm)
    return llm


def _store_video(video_id, segments):
    meta = VideoMetadata(
        video_id=video_id, original_filename="v.mp4", stored_filename="v.mp4", extension=".mp4",
        size_bytes=1, uploaded_at=datetime.now(timezone.utc),
    )
    (settings.videos_dir / f"{video_id}.json").write_text(meta.model_dump_json(), encoding="utf-8")
    if segments is not None:
        transcript = Transcript(
            video_id=video_id, language="en", duration_seconds=100.0, model_size="base",
            created_at=datetime.now(timezone.utc), segments=segments,
        )
        (settings.transcripts_dir / f"{video_id}.json").write_text(transcript.model_dump_json(), encoding="utf-8")


def test_generate_takes_timestamps_from_the_transcript_not_the_model(stored):
    _store_video("vid1", [_seg(i, 50, 100.0 + i * 10.0) for i in range(4)])  # 200 words -> 2 sections
    summary = summary_service.generate_summary("vid1")
    assert [(p.start_time, p.end_time) for p in summary.key_points] == [(100.0, 120.0), (120.0, 140.0)]
    assert all(p.text == "It explains a thing." for p in summary.key_points)
    assert summary.overview == "All about it."
    # one call per section plus one for the overview
    assert len(stored.prompts) == 3


def test_generate_is_cached_on_disk(stored):
    _store_video("vid1", [_seg(i, 50, i * 10.0) for i in range(4)])
    created = summary_service.generate_summary("vid1")
    assert summary_service.get_summary("vid1").overview == created.overview


def test_generate_skips_sections_too_short_to_summarise(stored):
    _store_video("vid1", [_seg(0, 120, 0.0), _seg(1, 3, 10.0)])
    summary = summary_service.generate_summary("vid1")
    assert len(summary.key_points) == 1


def test_generate_without_transcript_is_409(stored):
    _store_video("vid1", None)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        summary_service.generate_summary("vid1")
    assert exc.value.status_code == 409


def test_api_get_404_before_generation_then_post_then_get(stored):
    _store_video("vid1", [_seg(i, 50, i * 10.0) for i in range(4)])
    assert client.get("/api/videos/vid1/summary").status_code == 404
    assert client.get("/api/videos/nope/summary").status_code == 404
    posted = client.post("/api/videos/vid1/summary")
    assert posted.status_code == 200 and posted.json()["key_points"]
    assert client.get("/api/videos/vid1/summary").json()["overview"] == posted.json()["overview"]


def test_api_llm_down_is_503(stored, monkeypatch):
    _store_video("vid1", [_seg(i, 50, i * 10.0) for i in range(4)])

    class Down(llm_service.LLMProvider):
        def generate(self, prompt):
            raise RuntimeError("Could not reach Ollama")

    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: Down())
    assert client.post("/api/videos/vid1/summary").status_code == 503
