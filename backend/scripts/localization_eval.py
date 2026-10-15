"""
Citation localization: how far from the real answer does a click land?

Run from backend/:

    HF_HUB_DISABLE_XET=1 venv/Scripts/python.exe scripts/localization_eval.py

A cited chunk can be a minute or more long, so jumping to its START can leave
the viewer scrubbing for the answer. This measures whether a cheap second
stage -- re-scoring the individual transcript SEGMENTS inside the retrieved
chunk with the cross-encoder and jumping to the best one -- lands closer.

For each of the 57 labelled questions (scripts/eval_data/questions.json), using
the full retrieval stack (dense shortlist of 10 + cross-encoder, per video):

  chunk-start     jump to the start of the top-ranked chunk   (current behaviour)
  segment-best    jump to the start of the best-scoring segment inside it
  segment-ctx     same, scoring each segment with its neighbours as context

Reported: median / mean absolute error (seconds) between the jump target and
the start of the labelled answer, and the share of questions landing within
10 s / 20 s -- over ALL questions, and over those where the top chunk really
contains the answer (isolating localization from retrieval mistakes).

Caveat: answer spans were written by hand from the transcripts and are
accurate to a few seconds; 57 questions, one labeller.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.models.schemas import ChunkSet  # noqa: E402
from app.services import chunking_service, embedding_service, transcription_service  # noqa: E402

SHORTLIST = 10


def main():
    spec = json.loads((Path(__file__).parent / "eval_data" / "questions.json").read_text(encoding="utf-8"))
    videos = spec["videos"]
    questions = [(videos[v], q, a, b) for v, q, a, b, _ in spec["questions"]]
    ce = CrossEncoder(settings.reranker_model_name, device=settings.reranker_device)

    def evaluate(label, chunks_by_video):
        errors = {"chunk-start": [], "segment-best": [], "segment-ctx": []}
        in_chunk = []
        chunk_lengths = []
        cache = {}
        for vid in chunks_by_video:
            chunks = chunks_by_video[vid]
            cache[vid] = (chunks, embedding_service.embed_texts([c.text for c in chunks]),
                          transcription_service.get_transcript(vid).segments)
        for vid, q, a_start, a_end in questions:
            chunks, emb, segments = cache[vid]
            qv = embedding_service.embed_texts([q])[0]
            short = list(np.argsort(-(emb @ qv))[:SHORTLIST])
            scores = ce.predict([(q, chunks[i].text) for i in short], show_progress_bar=False)
            top = chunks[short[int(np.argmax(scores))]]
            chunk_lengths.append(top.end_time - top.start_time)
            inside = [k for k, s in enumerate(segments) if s.start_time >= top.start_time - 0.01 and s.end_time <= top.end_time + 0.01]
            if not inside:
                inside = [k for k, s in enumerate(segments) if top.start_time <= s.start_time < top.end_time]
            seg_scores = ce.predict([(q, segments[k].text) for k in inside], show_progress_bar=False)
            ctx_scores = ce.predict(
                [(q, " ".join(segments[j].text for j in range(max(0, k - 1), min(len(segments), k + 2)))) for k in inside],
                show_progress_bar=False,
            )
            targets = {
                "chunk-start": top.start_time,
                "segment-best": segments[inside[int(np.argmax(seg_scores))]].start_time,
                "segment-ctx": segments[inside[int(np.argmax(ctx_scores))]].start_time,
            }
            for name, t in targets.items():
                errors[name].append(abs(t - a_start))
            in_chunk.append(top.start_time <= a_start + 5 and top.end_time >= a_start + (a_end - a_start) * 0.5)

        mask = np.array(in_chunk)
        print(f"--- {label}: {len(questions)} questions, mean cited-chunk length {np.mean(chunk_lengths):.0f}s, "
              f"top chunk contains the answer for {mask.sum()}/{len(mask)} ---")
        print(f"{'jump target':14} | {'ALL: median':>11} {'mean':>6} {'<=10s':>6} {'<=20s':>6} | {'ANSWER-IN-CHUNK: median':>24} {'<=10s':>6} {'<=20s':>6}")
        for name, errs in errors.items():
            e = np.array(errs)
            print(f"{name:14} | {np.median(e):>10.1f}s {e.mean():>5.0f}s {np.mean(e <= 10):>6.0%} {np.mean(e <= 20):>6.0%} | "
                  f"{np.median(e[mask]):>23.1f}s {np.mean(e[mask] <= 10):>6.0%} {np.mean(e[mask] <= 20):>6.0%}")
        print()

    production = {}
    for vid in videos.values():
        production[vid] = ChunkSet.model_validate_json(
            (settings.chunks_dir / f"{vid}.json").read_text(encoding="utf-8")
        ).chunks
    evaluate("production chunks (scene-aligned where frames exist)", production)

    original = settings.chunk_target_words
    settings.chunk_target_words = 150
    try:
        plain = {vid: chunking_service.chunk_transcript(transcription_service.get_transcript(vid)) for vid in videos.values()}
    finally:
        settings.chunk_target_words = original
    evaluate("plain word-count chunks (ceiling 150)", plain)


if __name__ == "__main__":
    main()
