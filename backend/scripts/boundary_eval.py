"""
Boundary evaluation: which signals find where a lecture changes topic?

Run from backend/:

    HF_HUB_DISABLE_XET=1 venv/Scripts/python.exe scripts/boundary_eval.py

Ground truth = the chapter markers the VIDEO'S CREATOR published
(scripts/eval_data/chapters.json) -- human-made topic boundaries that no
algorithm in this repo produced, so scoring against them is not circular
(unlike the alignment/purity numbers in segmentation_benchmark.py).

Methods (each returns boundary timestamps in seconds)
  fixed-60s     a boundary every 60 s            (the fixed-window baseline)
  scene-mad     visual: mean-pixel-difference scene changes (old metric)
  scene-ssim    visual: SSIM scene changes       (production)
  pyscenedetect visual: PySceneDetect ContentDetector (third-party baseline)
  topic         transcript: embedding similarity dips between neighbouring
                windows of segments (TextTiling-style)
  pause         transcript: long silences between speech segments
  hybrid        weighted fusion of scene-ssim + topic + pause, thresholded,
                with minimum spacing. Its settings are chosen LEAVE-ONE-VIDEO-OUT
                (tuned on the other videos, scored on the held-out one) so it
                cannot simply be fitted to the answers it is scored on.
  random        same number of boundaries as `hybrid`, placed at random
                (averaged over many draws) -- the chance level to beat.

A predicted boundary is correct if it lies within TOLERANCE seconds of a not
yet matched chapter start (one-to-one matching). Chapter markers point at
roughly where the speaker moves on, so +/-15 s is a fair window; +/-30 s is
reported too. With only a handful of videos and ~40 chapter boundaries in
total, differences of a few points are noise.
"""

import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services import embedding_service, transcription_service, video_service  # noqa: E402
from segmentation_benchmark import detect_boundaries, sample_frames  # noqa: E402

TOLERANCES = (15.0, 30.0)
TOPIC_WINDOW = 6      # segments on each side of a gap when comparing topics
MIN_SPACING = 30.0    # hybrid: at most one boundary per 30 s


# ------------------------------------------------------------------ data

def load_videos():
    spec = json.loads((Path(__file__).parent / "eval_data" / "chapters.json").read_text(encoding="utf-8"))["videos"]
    by_youtube = {}
    for path in settings.videos_dir.glob("*.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        url = meta.get("source_url") or ""
        for key, info in spec.items():
            if f"v={info['youtube_id']}" in url:
                by_youtube[key] = meta["video_id"]
    videos = []
    for key, info in spec.items():
        vid = by_youtube.get(key)
        if vid and transcription_service.get_transcript(vid):
            truth = [c["start"] for c in info["chapters"] if c["start"] > 0]
            videos.append((key, vid, truth))
    return videos


# --------------------------------------------------------------- matching

def prf(predicted, truth, tolerance):
    """One-to-one greedy matching of predicted boundaries to chapter starts."""
    unmatched = list(truth)
    hits = 0
    for p in sorted(predicted):
        near = [t for t in unmatched if abs(t - p) <= tolerance]
        if near:
            unmatched.remove(min(near, key=lambda t: abs(t - p)))
            hits += 1
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------- signals

def topic_depths(segments):
    """Per-gap topic-shift depth (TextTiling-style). gap i sits just before
    segment i; depth is 0 where there is no room for a window."""
    texts = [s.text for s in segments]
    emb = embedding_service.embed_texts(texts)
    n, w = len(segments), TOPIC_WINDOW
    sims = np.full(n, np.nan)
    for i in range(w, n - w + 1):
        left, right = emb[i - w:i].mean(axis=0), emb[i:i + w].mean(axis=0)
        sims[i] = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
    depths = np.zeros(n)
    m = 5
    for i in range(w, n - w + 1):
        lo, hi = max(w, i - m), min(n - w + 1, i + m + 1)
        left_peak = np.nanmax(sims[lo:i]) if i > lo else sims[i]
        right_peak = np.nanmax(sims[i + 1:hi]) if hi > i + 1 else sims[i]
        depths[i] = (left_peak - sims[i]) + (right_peak - sims[i])
    return depths


def topic_boundaries(segments, depths, alpha=1.0):
    valid = depths[depths > 0]
    if len(valid) == 0:
        return []
    cutoff = valid.mean() + alpha * valid.std()
    out = []
    for i in range(1, len(segments) - 1):
        if depths[i] >= cutoff and depths[i] >= depths[max(0, i - 2):i + 3].max():
            out.append(segments[i].start_time)
    return out


def pause_gaps(segments):
    gaps = np.zeros(len(segments))
    for i in range(1, len(segments)):
        gaps[i] = max(0.0, segments[i].start_time - segments[i - 1].end_time)
    return gaps


def hybrid_scores(segments, visual_boundaries, depths, gaps):
    """Per-gap evidence in [0,1] for each of the three signals."""
    half = settings.frame_sample_interval_seconds / 2
    visual = np.zeros(len(segments))
    for b in visual_boundaries:
        estimate = b - half
        for i, s in enumerate(segments):
            visual[i] = max(visual[i], float(np.exp(-abs(s.start_time - estimate) / 5.0)))
    scale = np.percentile(depths[depths > 0], 95) if (depths > 0).any() else 1.0
    topic = np.clip(depths / (scale or 1.0), 0, 1)
    pause = np.clip(gaps / 2.0, 0, 1)
    return visual, topic, pause


def hybrid_boundaries(segments, signals, weights, threshold):
    visual, topic, pause = signals
    wv, wt, wp = weights
    score = (wv * visual + wt * topic + wp * pause) / (wv + wt + wp)
    chosen = []
    for i in np.argsort(-score):
        if score[i] < threshold:
            break
        t = segments[i].start_time
        if t <= 0 or any(abs(t - c) < MIN_SPACING for c in chosen):
            continue
        chosen.append(t)
    return sorted(chosen)


# ------------------------------------------------------------------- main

def pyscenedetect_boundaries(video_path):
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video_path), ContentDetector(), show_progress=False)
    return [start.get_seconds() for start, _ in scenes[1:]]


HYBRID_GRID = list(itertools.product(
    [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
    [0.3, 0.4, 0.5, 0.6, 0.7],
))


def main():
    videos = load_videos()
    print("videos:", ", ".join(f"{k} ({len(t)} chapter boundaries)" for k, _, t in videos), "\n")

    per_video = {}
    for key, vid, truth in videos:
        segs = transcription_service.get_transcript(vid).segments
        path = video_service.get_video_file_path(vid)
        grays, stamps = sample_frames(path)
        duration = max(s.end_time for s in segs)
        depths = topic_depths(segs)
        gaps = pause_gaps(segs)
        ssim = detect_boundaries(grays, stamps, "ssim")
        per_video[key] = {
            "truth": truth, "duration": duration, "segments": segs, "depths": depths, "gaps": gaps,
            "methods": {
                "fixed-60s": [float(t) for t in np.arange(60, duration, 60)],
                "scene-mad": detect_boundaries(grays, stamps, "mad"),
                "scene-ssim": ssim,
                "pyscenedetect": pyscenedetect_boundaries(path),
                "topic": topic_boundaries(segs, depths),
                "pause": [segs[i].start_time for i in range(len(segs)) if gaps[i] >= 1.5],
            },
            "signals": hybrid_scores(segs, ssim, depths, gaps),
        }

    # Hybrid: choose settings leave-one-video-out, score on the held-out video.
    keys = list(per_video)
    for held_out in keys:
        others = [k for k in keys if k != held_out]

        def mean_f1(cfg, which):
            return np.mean([
                prf(hybrid_boundaries(per_video[k]["segments"], per_video[k]["signals"], *cfg),
                    per_video[k]["truth"], TOLERANCES[0])[2] for k in which
            ])

        best = max(HYBRID_GRID, key=lambda cfg: mean_f1(cfg, others))
        per_video[held_out]["methods"]["hybrid"] = hybrid_boundaries(
            per_video[held_out]["segments"], per_video[held_out]["signals"], *best
        )
        per_video[held_out]["hybrid_cfg"] = best

    rng = random.Random(0)
    for key in keys:
        n = len(per_video[key]["methods"]["hybrid"])
        per_video[key]["random_n"] = n

    method_names = ["fixed-60s", "scene-mad", "scene-ssim", "pyscenedetect", "topic", "pause", "hybrid", "random"]
    for tol in TOLERANCES:
        print(f"=== tolerance +/-{tol:.0f}s  (precision / recall / F1) ===")
        print(f"{'method':14}" + "".join(f"{k:>26}" for k in keys) + f"{'macro F1':>11}")
        for name in method_names:
            cells, f1s = [], []
            for key in keys:
                pv = per_video[key]
                truth, dur = pv["truth"], pv["duration"]
                if name == "random":
                    runs = [prf([rng.uniform(0, dur) for _ in range(pv["random_n"])], truth, tol) for _ in range(300)]
                    p, r, f = (float(np.mean([x[j] for x in runs])) for j in range(3))
                    count = pv["random_n"]
                else:
                    pred = pv["methods"][name]
                    p, r, f = prf(pred, truth, tol)
                    count = len(pred)
                cells.append(f"{count:>3}b {p:>4.0%}/{r:>4.0%}/{f:>4.2f}".rjust(26))
                f1s.append(f)
            print(f"{name:14}" + "".join(cells) + f"{np.mean(f1s):>11.3f}")
        print()

    print("hybrid settings chosen when each video was held out (weights visual,topic,pause | threshold):")
    for key in keys:
        print(f"  {key:14} {per_video[key]['hybrid_cfg']}")
    print("\nchapter boundaries per video:", {k: len(per_video[k]['truth']) for k in keys})


if __name__ == "__main__":
    main()
