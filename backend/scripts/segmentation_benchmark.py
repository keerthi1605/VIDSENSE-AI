"""
Segmentation benchmark: how do the two scene-detection metrics (MAD vs SSIM)
and the two chunking strategies (plain word-count vs scene-aligned) behave on
real, already-transcribed videos?

Run from backend/:

    venv/Scripts/python.exe scripts/segmentation_benchmark.py [video_id ...]

With no ids it uses every video that has a saved transcript. Needs no model
and no server -- only the stored video files and transcripts.

What it measures, per video:
  * scene boundaries found by each metric (same sampling interval and the
    same "compare against last saved frame" rule as frame_service);
  * for each chunking strategy: chunk count, mean words / duration, and two
    STRUCTURAL alignment metrics --
      alignment rate: share of scene boundaries that coincide with a chunk
        start (the nearest transcript segment start to the boundary);
      scene purity:   mean share of a chunk's duration that lies inside its
        single dominant scene (1.0 = every chunk covers one scene).

CAVEAT -- read before quoting numbers: alignment rate and purity are
measured against the SAME scene boundaries the aligned chunker is steered by,
so they confirm the chunker does what it was built to do; they are NOT an
independent measure of retrieval quality. That needs labelled queries
(see docs/architecture.md, Phase 10).

Frames are sampled by seeking (fast) rather than the sequential decode
frame_service uses, so boundary timestamps can differ from a production run
by a fraction of a second -- irrelevant at a 5 s sampling interval.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services import chunking_service, frame_service, transcription_service, video_service  # noqa: E402


def sample_frames(video_path: Path):
    """Downsampled grayscale frames every frame_sample_interval_seconds,
    exactly as frame_service prepares candidates for comparison."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    grays, timestamps = [], []
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, bgr = cap.read()
        if not ok:
            break
        grays.append(frame_service._downsample_gray(bgr))
        timestamps.append(round(t, 2))
        t += settings.frame_sample_interval_seconds
    cap.release()
    return grays, timestamps


def detect_boundaries(grays, timestamps, method: str):
    """frame_service's keep/skip rule, replayed on pre-sampled frames."""
    original = settings.frame_diff_method
    settings.frame_diff_method = method
    try:
        last = grays[0]
        boundaries = []
        for gray, ts in zip(grays[1:], timestamps[1:]):
            if frame_service._is_new_scene(gray, last):
                boundaries.append(ts)
                last = gray
        return boundaries
    finally:
        settings.frame_diff_method = original


def purity(chunks, boundaries, duration):
    """Mean share of each chunk's duration inside its dominant scene."""
    half = settings.frame_sample_interval_seconds / 2
    edges = [0.0] + [b - half for b in boundaries] + [duration]
    shares = []
    for chunk in chunks:
        span = chunk.end_time - chunk.start_time
        if span <= 0:
            continue
        best = max(
            max(0.0, min(chunk.end_time, edges[k + 1]) - max(chunk.start_time, edges[k]))
            for k in range(len(edges) - 1)
        )
        shares.append(best / span)
    return sum(shares) / len(shares) if shares else 0.0


def alignment_rate(chunks, boundaries, segments):
    if not boundaries:
        return float("nan")
    half = settings.frame_sample_interval_seconds / 2
    chunk_starts = {c.start_time for c in chunks}
    starts = [s.start_time for s in segments]
    hits = 0
    for b in boundaries:
        nearest = min(starts, key=lambda s: abs(s - (b - half)))
        hits += nearest in chunk_starts
    return hits / len(boundaries)


def chunk_stats(chunks):
    n = len(chunks)
    words = sum(c.word_count for c in chunks) / n
    seconds = sum(c.end_time - c.start_time for c in chunks) / n
    return n, words, seconds


def main(video_ids):
    if not video_ids:
        video_ids = sorted(p.stem for p in settings.transcripts_dir.glob("*.json"))
    print(
        f"sampling every {settings.frame_sample_interval_seconds:g}s | SSIM new scene < {settings.frame_ssim_threshold} "
        f"| MAD new scene >= {settings.frame_diff_threshold:g} | chunk ceiling {settings.chunk_target_words} words, "
        f"floor {settings.chunk_min_words}\n"
    )
    header = f"{'video':34} {'method':6} {'bounds':>6} | {'strategy':14} {'chunks':>6} {'words':>6} {'secs':>6} {'align':>6} {'purity':>6}"
    print(header)
    print("-" * len(header))
    for vid in video_ids:
        transcript = transcription_service.get_transcript(vid)
        if transcript is None:
            continue
        meta = video_service.get_video_metadata(vid)
        title = meta.original_filename.rsplit(".", 1)[0][:32]
        grays, timestamps = sample_frames(video_service.get_video_file_path(vid))
        duration = transcript.duration_seconds
        for method in ("mad", "ssim"):
            boundaries = detect_boundaries(grays, timestamps, method)
            for strategy, scene_arg in (("word-count", None), ("scene-aligned", boundaries)):
                chunks = chunking_service.chunk_transcript(transcript, scene_boundaries=scene_arg)
                n, words, seconds = chunk_stats(chunks)
                align = alignment_rate(chunks, boundaries, transcript.segments)
                pur = purity(chunks, boundaries, duration)
                print(
                    f"{title:34} {method:6} {len(boundaries):>6} | {strategy:14} {n:>6} {words:>6.0f} {seconds:>6.0f} "
                    f"{align:>6.0%} {pur:>6.2f}"
                )
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
