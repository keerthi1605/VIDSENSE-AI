"""
Reranker follow-up analysis (run AFTER retrieval_ablation.py, on an idle machine).

  1. Clean latency: warm cross-encoder cost to rescore a 20-chunk shortlist.
  2. Relevance gate: can the reranker's top score tell an ANSWERABLE question
     from an out-of-scope one? If so, "no relevant context" can be decided
     deterministically instead of leaving the small LLM to decline on its own.

Run from backend/:  HF_HUB_DISABLE_XET=1 venv/Scripts/python.exe scripts/rerank_analysis.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.models.schemas import ChunkSet  # noqa: E402
from app.services import embedding_service, rerank_service  # noqa: E402

import os

SHORTLIST = int(os.environ.get("SHORTLIST", settings.rerank_candidates))

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Who won the 2018 football world cup?",
    "How do vaccines work?",
    "What is the plot of Hamlet?",
    "How do I change a flat tire on a car?",
    "What is the population of Tokyo?",
    "Explain quantum entanglement.",
    "How do airplanes stay in the air?",
    "What is the stock price of Tesla?",
    "How do I learn to play the guitar?",
    "What is photosynthesis?",
    "Who painted the Mona Lisa?",
    "What is the tallest mountain in the world?",
    "How many moons does Jupiter have?",
    "What causes earthquakes?",
    "How do I file my income taxes?",
]


def auc(positives, negatives):
    """P(a random positive scores higher than a random negative)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def main():
    spec = json.loads((Path(__file__).parent / "eval_data" / "questions.json").read_text(encoding="utf-8"))
    in_scope = [q for _, q, *_ in spec["questions"]]

    chunks = []
    for path in sorted(settings.chunks_dir.glob("*.json")):
        chunks.extend(ChunkSet.model_validate_json(path.read_text(encoding="utf-8")).chunks)
    texts = [c.text for c in chunks]
    corpus = embedding_service.embed_texts(texts)

    def shortlist(query):
        sims = embedding_service.embed_texts([query])[0] @ corpus.T
        return [texts[i] for i in np.argsort(-sims)[:SHORTLIST]]

    rerank_service.rerank_scores("warm up", texts[:2])  # load the model outside the timed region

    def top_score(query):
        start = time.perf_counter()
        scores = rerank_service.rerank_scores(query, shortlist(query))
        return float(scores.max()), time.perf_counter() - start

    in_results = [top_score(q) for q in in_scope]
    out_results = [top_score(q) for q in OUT_OF_SCOPE]
    in_scores = np.array([s for s, _ in in_results])
    out_scores = np.array([s for s, _ in out_results])
    times = np.array([t for _, t in in_results + out_results]) * 1000

    print(f"corpus: {len(chunks)} chunks | shortlist {SHORTLIST} | model {settings.reranker_model_name}\n")
    print(f"latency (embed query + score {SHORTLIST} chunks, warm): median {np.median(times):.0f} ms, p95 {np.percentile(times, 95):.0f} ms\n")
    print(f"top rerank logit  in-scope (n={len(in_scores)}): min {in_scores.min():.2f}  p10 {np.percentile(in_scores, 10):.2f}  median {np.median(in_scores):.2f}")
    print(f"top rerank logit  out-of-scope (n={len(out_scores)}): median {np.median(out_scores):.2f}  p90 {np.percentile(out_scores, 90):.2f}  max {out_scores.max():.2f}")
    print(f"AUC (in-scope scores above out-of-scope): {auc(in_scores, out_scores):.3f}\n")
    print(f"{'threshold':>10} {'answerable kept':>16} {'out-of-scope refused':>21}")
    for t in (-8, -6, -5, -4, -3, -2, -1, 0, 1, 2):
        kept = (in_scores >= t).mean()
        refused = (out_scores < t).mean()
        print(f"{t:>10} {kept:>16.0%} {refused:>21.0%}")
    print("\nout-of-scope questions and their top logit:")
    for q, (s, _) in zip(OUT_OF_SCOPE, out_results):
        print(f"  {s:>6.2f}  {q}")
    print("\nlowest in-scope top logits (would be wrongly refused first):")
    for s, q in sorted(zip(in_scores, in_scope))[:6]:
        print(f"  {s:>6.2f}  {q}")


if __name__ == "__main__":
    main()
