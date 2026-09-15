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

FakeSegment = namedtuple("FakeSegment", ["start", "end", "text", "words"], defaults=[None])
FakeWord = namedtuple("FakeWord", ["start", "end", "word"])
FakeInfo = namedtuple("FakeInfo", ["language", "duration"])


class FakeWhisperModel:
    """Records the kwargs it was called with, so tests can assert we're
    actually requesting VAD/word-timestamps/etc., not just that plumbing
    doesn't crash."""

    def __init__(self, segments, language="en", duration=12.5):
        self._segments = segments
        self._language = language
        self._duration = duration
        self.last_call_kwargs = None

    def transcribe(self, audio_path, **kwargs):
        self.last_call_kwargs = kwargs
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


def test_transcribe_and_save_requests_vad_and_word_timestamps(monkeypatch, tmp_path):
    """We should actually be asking faster-whisper for the accuracy
    features (VAD, word timestamps, no cross-segment conditioning) --
    not just have them sit unused in settings."""
    fake_model = FakeWhisperModel([FakeSegment(start=0.0, end=1.0, text="Hi.")])
    monkeypatch.setattr(transcription_service, "get_whisper_model", lambda: fake_model)

    fake_audio_path = tmp_path / "fake.wav"
    fake_audio_path.write_bytes(b"")

    transcription_service.transcribe_and_save("test_video_2", fake_audio_path)

    kwargs = fake_model.last_call_kwargs
    assert kwargs["vad_filter"] == settings.whisper_vad_filter
    assert kwargs["vad_parameters"] == {"min_silence_duration_ms": settings.whisper_vad_min_silence_ms}
    assert kwargs["word_timestamps"] == settings.whisper_word_timestamps
    assert kwargs["condition_on_previous_text"] == settings.whisper_condition_on_previous_text


def test_transcribe_and_save_prefers_request_prompt_over_settings_default(monkeypatch, tmp_path):
    fake_model = FakeWhisperModel([FakeSegment(start=0.0, end=1.0, text="Hi.")])
    monkeypatch.setattr(transcription_service, "get_whisper_model", lambda: fake_model)
    monkeypatch.setattr(settings, "whisper_initial_prompt", "default course prompt")

    fake_audio_path = tmp_path / "fake.wav"
    fake_audio_path.write_bytes(b"")

    # Explicit per-request prompt wins...
    transcription_service.transcribe_and_save(
        "test_video_3", fake_audio_path, initial_prompt="cooking show, ingredients", hotwords="saute, julienne"
    )
    assert fake_model.last_call_kwargs["initial_prompt"] == "cooking show, ingredients"
    assert fake_model.last_call_kwargs["hotwords"] == "saute, julienne"

    # ...but omitting it falls back to the settings-level default.
    transcription_service.transcribe_and_save("test_video_4", fake_audio_path)
    assert fake_model.last_call_kwargs["initial_prompt"] == "default course prompt"


def test_transcribe_and_save_preserves_word_timestamps(monkeypatch, tmp_path):
    fake_segments = [
        FakeSegment(
            start=0.0,
            end=1.5,
            text="Hello world.",
            words=[
                FakeWord(start=0.0, end=0.5, word=" Hello"),
                FakeWord(start=0.5, end=1.5, word=" world."),
            ],
        ),
        FakeSegment(start=1.5, end=2.0, text="Ok.", words=None),
    ]
    monkeypatch.setattr(
        transcription_service,
        "get_whisper_model",
        lambda: FakeWhisperModel(fake_segments),
    )

    fake_audio_path = tmp_path / "fake.wav"
    fake_audio_path.write_bytes(b"")

    transcript = transcription_service.transcribe_and_save("test_video_5", fake_audio_path)

    first, second = transcript.segments
    assert first.words is not None
    assert [w.word for w in first.words] == ["Hello", "world."]
    assert first.words[0].start_time == 0.0
    assert first.words[1].end_time == 1.5
    # A segment faster-whisper reports with no word list stays None, not [].
    assert second.words is None

    reloaded = transcription_service.get_transcript("test_video_5")
    assert reloaded.segments[0].words[1].word == "world."
