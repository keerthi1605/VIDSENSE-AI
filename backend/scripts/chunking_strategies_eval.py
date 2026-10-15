"""
Chunking strategies head to head: does visual alignment add anything beyond
text-only topic alignment, once chunk size is controlled?

Run from backend/:

    HF_HUB_DISABLE_XET=1 venv/Scripts/python.exe scripts/chunking_strategies_eval.py

Strategies (same transcript, same segment-start snapping, same word floor):
  wc150          plain word-count, ceiling 150 words (previous default)
  wc75           plain word-count, ceiling 75 words  (size control)
  scene-ssim     cut at SSIM visual scene changes (production alignment)
  topic          cut at transcript topic shifts (TextTiling-style, no vision)
  scene+topic    cut at either

Two independent measures:
  1. Chapter purity -- share of each chunk's duration inside ONE creator-published
     chapter (scripts/eval_data/chapters.json). Human-made ground truth, so it is
     not circular. Only videos that have chapters.
  2. Retrieval -- Hit@1/3 and MRR of MiniLM nearest-chunk search over the 57
     developer-labelled questions (scripts/eval_data/questions.json), plus the
     words the LLM would be sent (top-5). Dense retrieval only, so the effect
     of the chunking is not blurred by the reranker. Only videos with questions.

Caveats: few videos (4 with chapters, 4 with questions, partly different
sets), author-written labels, and a hit-coverage rule that favours larger
chunks -- hence the context-size column. Differences of 1-2 questions are noise.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services import chunking_service, embedding_service, transcription_service, video_service  # noqa: E402
from boundary_eval import load_videos, topic_boundaries, topic_depths  # noqa: E402
from segmentation_benchmark import detect_boundaries, sample_frames  # noqa: E402

HIT_COVERAGE = 0.5
STRATEGIES = ["wc150", "wc75", "scene-ssim", "topic", "scene+topic"]


def chunks_for(transcript, strategy, scene_b, topic_b):
    half = settings.frame_sample_interval_seconds / 2
    original = settings.chunk_target_words
    try:
        settings.chunk_target_words = 75 if strategy == "wc75" else 150
        if strategy in ("wc150", "wc75"):
            return chunking_service.chunk_transcript(transcript)
        # chunking_service subtracts half a sample interval from each boundary;
        # topic boundaries are already segment starts, so pre-add it.
        topic_shifted = [t + half for t in topic_b]
        boundaries = {"scene-ssim": scene_b, "topic": topic_shifted, "scene+topic": sorted(scene_b + topic_shifted)}[strategy]
        return chunking_service.chunk_transcript(transcript, scene_boundaries=boundaries)
    finally:
        settings.chunk_target_words = original


def purity(chunks, chapter_starts, duration):
    edges = [0.0] + sorted(chapter_starts) + [duration]
    shares = []
    for c in chunks:
        span = c.end_time - c.start_time
        if span <= 0:
            continue
        best = max(max(0.0, min(c.end_time, edges[k + 1]) - max(c.start_time, edges[k])) for k in range(len(edges) - 1))
        shares.append(best / span)
    return float(np.mean(shares))


def coverage(chunk, a_start, a_end):
    return max(0.0, min(chunk.end_time, a_end) - max(chunk.start_time, a_start)) / (a_end - a_start)


def main():
    spec = json.loads((Path(__file__).parent / "eval_data" / "questions.json").read_text(encoding="utf-8"))
    q_videos = spec["videos"]
    questions = [(q_videos[v], q, a, b) for v, q, a, b, _ in spec["questions"]]

    chapter_videos = {vid: (key + ("" if prov == "creator" else "*"), truth) for key, vid, truth, prov in load_videos()}
    all_ids = sorted(set(chapter_videos) | set(q_videos.values()))

    prepared = {}
    for vid in all_ids:
        tr = transcription_service.get_transcript(vid)
        grays, stamps = sample_frames(video_service.get_video_file_path(vid))
        scene_b = detect_boundaries(grays, stamps, "ssim")
        depths = topic_depths(tr.segments)
        prepared[vid] = (tr, scene_b, topic_boundaries(tr.segments, depths))

    print("strategy        ", "  ".join(f"{vid[:6]:>8}" for vid in all_ids), "\n")
    print("boundaries (scene-ssim / topic):", {vid[:6]: (len(prepared[vid][1]), len(prepared[vid][2])) for vid in all_ids}, "\n")

    # ---------------- 1. chapter purity
    print("=== chapter purity (higher = chunks stay inside one chapter; * = chapters unverified/auto-generated) ===")
    header = f"{'strategy':13}" + "".join(f"{chapter_videos[v][0]:>15}" for v in chapter_videos) + f"{'mean':>8}   chunks/video"
    print(header)
    for s in STRATEGIES:
        row, counts = [], []
        for vid, (key, truth) in chapter_videos.items():
            tr, scene_b, topic_b = prepared[vid]
            ch = chunks_for(tr, s, scene_b, topic_b)
            row.append(purity(ch, truth, tr.duration_seconds))
            counts.append(len(ch))
        print(f"{s:13}" + "".join(f"{x:>15.3f}" for x in row) + f"{np.mean(row):>8.3f}   {counts}")

    # ---------------- 2. retrieval
    print("\n=== retrieval (dense MiniLM, 57 questions) ===")
    print(f"{'strategy':13} {'chunks':>6} {'words/chunk':>11} {'Hit@1':>6} {'Hit@3':>6} {'MRR':>6} {'ctx words(top5)':>16}")
    for s in STRATEGIES:
        ranks, ctx, total_chunks, total_words = [], [], 0, 0
        by_video = {}
        per_video_chunks = {}
        for vid in q_videos.values():
            tr, scene_b, topic_b = prepared[vid]
            ch = chunks_for(tr, s, scene_b, topic_b)
            per_video_chunks[vid] = (ch, embedding_service.embed_texts([c.text for c in ch]))
            total_chunks += len(ch)
            total_words += sum(c.word_count for c in ch)
        for vid, q, a, b in questions:
            ch, emb = per_video_chunks[vid]
            qv = embedding_service.embed_texts([q])[0]
            order = np.argsort(-(emb @ qv))
            rank = next((r + 1 for r, i in enumerate(order) if coverage(ch[i], a, b) >= HIT_COVERAGE), None)
            ranks.append(rank)
            by_video.setdefault(vid, []).append(rank)
            ctx.append(sum(ch[i].word_count for i in order[:5]))
        n = len(questions)
        h = lambda k: sum(1 for r in ranks if r and r <= k) / n
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"{s:13} {total_chunks:>6} {total_words / total_chunks:>11.0f} {h(1):>6.0%} {h(3):>6.0%} {mrr:>6.3f} {np.mean(ctx):>16.0f}")
        breakdown = "  ".join(f"{vid[:6]}(n={len(rs)}): {sum(1 for r in rs if r == 1) / len(rs):.0%}/{sum(1 / r for r in rs if r) / len(rs):.2f}" for vid, rs in by_video.items())
        print(f"{'':13} per-video Hit@1/MRR -> {breakdown}")


if __name__ == "__main__":
    main()
