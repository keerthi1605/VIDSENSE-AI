"""
Tests for app.services.transcription_service.

The real Whisper model is never loaded in these tests -- that would
mean downloading model weights and running real (slow) inference just
to check our own plumbing. Instead we monkeypatch `get_whisper_model`
with a fake that returns fixed segments, and verify that timestamps
and text survive the trip through our own schema and JSON persistence
unchanged. That's the actual thing worth testing here.
"""

from collections import namedtuple

import pytest

from app.core.config import settings
from app.services import transcription_service

FakeSegment = namedtuple("FakeSegment", ["start", "end", "text"])
FakeInfo = namedtuple("FakeInfo", ["language", "duration"])


class FakeWhisperModel:
    def __init__(self, segments, language="en", duration=12.5):
        self._segments = segments
        self._language = language
        self._duration = duration

    def transcribe(self, audio_path):
        return iter(self._segments), FakeInfo(self._language, self._duration)


@pytest.fixture(autouse=True)
def _cleanup_transcript_files():
    before = set(settings.transcripts_dir.glob("*"))
    yield
    after = set(settings.transcripts_dir.glob("*"))
    for path in after - before:
        path.unlink(missing_ok=True)


def test_transcribe_and_save_preserves_timestamps_and_text(monkeypatch, tmp_path):
    fake_segments = [
        FakeSegment(start=0.0, end=4.321, text="  Today we discuss binary search.  "),
        FakeSegment(start=4.321, end=9.8, text="It repeatedly halves the search space."),
    ]
    monkeypatch.setattr(
        transcription_service,
        "get_whisper_model",
        lambda: FakeWhisperModel(fake_segments, language="en", duration=9.8),
    )

    fake_audio_path = tmp_path / "fake.wav"
    fake_audio_path.write_bytes(b"")  # never actually read; model is faked

    transcript = transcription_service.transcribe_and_save("test_video_1", fake_audio_path)

    assert transcript.video_id == "test_video_1"
    assert transcript.language == "en"
    assert transcript.duration_seconds == 9.8
    assert len(transcript.segments) == 2

    first, second = transcript.segments
    assert first.segment_id == 0
    assert first.start_time == 0.0
    assert first.end_time == 4.32  # rounded to 2 decimals
    assert first.text == "Today we discuss binary search."  # whitespace stripped
    assert second.start_time == 4.32
    assert second.end_time == 9.8

    # Persisted to disk and re-loadable, with the same timestamps.
    reloaded = transcription_service.get_transcript("test_video_1")
    assert reloaded is not None
    assert reloaded.segments[0].start_time == first.start_time
    assert reloaded.segments[1].end_time == second.end_time


def test_get_transcript_returns_none_when_missing():
    assert transcription_service.get_transcript("nonexistent_video_id") is None
