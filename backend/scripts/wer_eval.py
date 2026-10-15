"""
Transcript accuracy: word error rate (WER) against human-written captions.

Run from backend/:

    HF_HUB_DISABLE_XET=1 venv/Scripts/python.exe scripts/wer_eval.py

Reference text = the English caption track a YouTube uploader published
(`yt-dlp --write-subs`, i.e. NOT auto-generated), downloaded once into
storage/transcripts/_reference/ (git-ignored -- not redistributed). Only
videos that actually have such a track can be scored here: of the test set,
the explainer talk (the lecture, vlog and news clip only have auto-generated
captions, and the cooking clip's captions are on-screen recipe text, not speech).

Configurations, all on the audio already extracted for each video:
  base / defaults      faster-whisper with NO options -- the pipeline before
                       the transcript-quality work
  base / production    VAD + no cross-segment conditioning (settings today)
  base / + prompt      production settings + a short domain prompt
  small / production   same, larger model
  small / + prompt

WER = (substitutions + deletions + insertions) / reference words, after
lower-casing and stripping punctuation, bracketed cues ([Music]) and
speaker markers. Two short clips (~460 reference words together) is a SMALL
sample: read differences of a point or two as noise, and a large gap as a
direction worth confirming on more audio.
"""

import html
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faster_whisper import WhisperModel  # noqa: E402

from app.core.config import settings  # noqa: E402

REF_DIR = settings.transcripts_dir / "_reference"

CLIPS = {
    # name: (local video_id, YouTube id, domain prompt used for the "+ prompt" runs)
    # The cooking clip is deliberately NOT here: its uploaded "captions" are the
    # on-screen recipe steps ("Instant coffee 1 tablespoon"), not a transcript of
    # the speech, so scoring against them gives a meaningless ~133% WER.
    "explainer": ("4eaeeb0e024346d2824c7eba72d2cb5c", "EMbZ72cCAzI", "A talk about how to explain complicated ideas clearly."),
}


def fetch_reference(name, youtube_id):
    REF_DIR.mkdir(parents=True, exist_ok=True)
    target = REF_DIR / f"{name}.en.vtt"
    if not target.exists():
        subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--skip-download", "--write-subs", "--sub-langs", "en",
             "--sub-format", "vtt", "-o", str(REF_DIR / name), f"https://www.youtube.com/watch?v={youtube_id}"],
            check=True, capture_output=True,
        )
    return target


_NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}


def normalise(text):
    text = html.unescape(text)                                    # &amp; -> &
    text = text.replace("’", "'").replace("‘", "'")      # curly apostrophes
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)          # [Music], (applause)
    text = re.sub(r">>+|-->", " ", text)                          # speaker markers
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    text = text.replace("'", "")
    return [_NUMBER_WORDS.get(w, w) for w in text.split()]        # "two" == "2"


def vtt_to_words(path):
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")) or "-->" in line or line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)  # inline timing/style tags
        if lines and lines[-1] == line:       # rolling-caption duplicate
            continue
        lines.append(line)
    return normalise(" ".join(lines))


def wer(reference, hypothesis):
    n, m = len(reference), len(hypothesis)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


_models = {}


def model(size):
    if size not in _models:
        _models[size] = WhisperModel(size, device=settings.whisper_device, compute_type=settings.whisper_compute_type)
    return _models[size]


def transcribe(size, audio, production, prompt):
    kwargs = {}
    if production:
        kwargs = dict(
            vad_filter=settings.whisper_vad_filter,
            vad_parameters={"min_silence_duration_ms": settings.whisper_vad_min_silence_ms},
            condition_on_previous_text=settings.whisper_condition_on_previous_text,
        )
    if prompt:
        kwargs["initial_prompt"] = prompt
    start = time.perf_counter()
    segments, _ = model(size).transcribe(str(audio), **kwargs)
    text = " ".join(s.text for s in segments)
    return normalise(text), time.perf_counter() - start


def main():
    configs = [
        ("base / defaults (before)", "base", False, False),
        ("base / production", "base", True, False),
        ("base / production + prompt", "base", True, True),
        ("small / production", "small", True, False),
        ("small / production + prompt", "small", True, True),
    ]
    refs = {name: vtt_to_words(fetch_reference(name, yt)) for name, (_, yt, _) in CLIPS.items()}
    print("reference words: " + ", ".join(f"{n}={len(w)}" for n, w in refs.items()) + "\n")

    header = f"{'configuration':32}" + "".join(f"{n:>10}" for n in CLIPS) + f"{'combined':>10}{'seconds':>9}"
    print(header)
    print("-" * len(header))
    for label, size, production, use_prompt in configs:
        errors, total, secs, cells = 0.0, 0, 0.0, []
        for name, (video_id, _, prompt) in CLIPS.items():
            hyp, took = transcribe(size, settings.audio_dir / f"{video_id}.wav", production, prompt if use_prompt else None)
            rate = wer(refs[name], hyp)
            cells.append(f"{rate:>10.1%}")
            errors += rate * len(refs[name])
            total += len(refs[name])
            secs += took
        print(f"{label:32}" + "".join(cells) + f"{errors / total:>10.1%}{secs:>9.0f}")


if __name__ == "__main__":
    main()
