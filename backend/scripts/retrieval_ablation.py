"""
Retrieval ablation: where can accuracy be improved?

Run from backend/:

    HF_HUB_DISABLE_XET=1 venv/Scripts/python.exe scripts/retrieval_ablation.py

Compares retrieval pipelines over the PRODUCTION chunks (storage/chunks/*.json
-- scene-aligned where frames exist) using the labelled questions in
scripts/eval_data/questions.json:

  dense        MiniLM cosine (what the app did before this work)
  bm25         lexical only
  hybrid-rrf   dense + BM25 fused with Reciprocal Rank Fusion
  hybrid-wsum  dense + BM25, each min-max normalised, equal-weight sum
  +rerank      any of the above, shortlist reranked by a cross-encoder

Two scopes: "per-video" (corpus = that question's video, like the app UI) and
"pooled" (every chunk of every video together -- harder, like a library-wide
question). A chunk hits when it is from the right video and covers >= 50% of
the labelled answer span.

Reading the numbers: with ~57 questions one question is ~1.7 points, so the
"vs dense" win/loss counts matter more than small differences in Hit@k.
Questions and answer spans were written by the developer from the transcripts.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.models.schemas import ChunkSet  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from app.services import embedding_service, lexical_service  # noqa: E402

HIT_COVERAGE = 0.5
# SHORTLIST=10 ABLATION_QUICK=1 -> only dense and dense+L6 at that shortlist size.
SHORTLIST = int(os.environ.get("SHORTLIST", 20))
QUICK = os.environ.get("ABLATION_QUICK") == "1"
RRF_K = 60


_encoders = {}


def cross_encoder(name):
    if name not in _encoders:
        _encoders[name] = CrossEncoder(name, device=settings.reranker_device)
    return _encoders[name]


def load_corpus():
    chunks = []
    for path in sorted(settings.chunks_dir.glob("*.json")):
        chunks.extend(ChunkSet.model_validate_json(path.read_text(encoding="utf-8")).chunks)
    return chunks


def hit(chunk, video_id, a_start, a_end):
    if chunk.video_id != video_id:
        return False
    overlap = max(0.0, min(chunk.end_time, a_end) - max(chunk.start_time, a_start))
    return overlap / (a_end - a_start) >= HIT_COVERAGE


def rrf(rank_lists, k=RRF_K):
    fused = {}
    for ranking in rank_lists:
        for pos, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(fused, key=lambda i: -fused[i])


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    return np.zeros_like(x) if span == 0 else (x - x.min()) / span


def main():
    spec = json.loads((Path(__file__).parent / "eval_data" / "questions.json").read_text(encoding="utf-8"))
    video_ids = spec["videos"]
    questions = [(video_ids[v], q, a, b, t) for v, q, a, b, t in spec["questions"]]

    corpus = load_corpus()
    texts = [c.text for c in corpus]
    by_video = {}
    for i, c in enumerate(corpus):
        by_video.setdefault(c.video_id, []).append(i)
    print(f"shortlist={SHORTLIST}")
    print(f"{len(corpus)} chunks across {len(by_video)} videos, {len(questions)} questions\n")

    dense_corpus = embedding_service.embed_texts(texts)
    dense_queries = embedding_service.embed_texts([q for _, q, _, _, _ in questions])
    bm25_pooled = lexical_service.BM25Index(texts)
    # Per-video scope mirrors the app: BM25 statistics come from that video's own chunks.
    bm25_video = {v: lexical_service.BM25Index([texts[i] for i in idx]) for v, idx in by_video.items()}

    rerankers = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"]

    def ranking_for(qi, scope, variant, reranker=None):
        video_id, query = questions[qi][0], questions[qi][1]
        pool = np.array(by_video[video_id] if scope == "per-video" else range(len(corpus)))
        dense = dense_queries[qi] @ dense_corpus[pool].T
        lexical = np.array(
            bm25_video[video_id].scores(query) if scope == "per-video" else bm25_pooled.scores(query)
        )
        if variant == "dense":
            order = list(np.argsort(-dense))
        elif variant == "bm25":
            order = list(np.argsort(-lexical))
        elif variant == "hybrid-rrf":
            order = rrf([list(np.argsort(-dense)), list(np.argsort(-lexical))])
        elif variant == "hybrid-wsum":
            order = list(np.argsort(-(minmax(dense) + minmax(lexical))))
        else:
            raise ValueError(variant)
        if reranker:
            shortlist = order[:SHORTLIST]
            scores = cross_encoder(reranker).predict(
                [(query, texts[pool[i]]) for i in shortlist], show_progress_bar=False
            )
            order = [shortlist[i] for i in np.argsort(-scores)] + order[SHORTLIST:]
        return [int(pool[i]) for i in order]

    def run(scope, variant, reranker=None):
        ranks, secs = [], 0.0
        for qi, (vid, q, a, b, t) in enumerate(questions):
            start = time.perf_counter()
            order = ranking_for(qi, scope, variant, reranker)
            secs += time.perf_counter() - start
            ranks.append(next((r + 1 for r, ci in enumerate(order) if hit(corpus[ci], vid, a, b)), None))
        return ranks, secs / len(questions) * 1000

    variants = [(v, None) for v in ("dense", "bm25", "hybrid-rrf", "hybrid-wsum")]
    variants += [(v, r) for r in rerankers for v in ("dense", "hybrid-rrf")]
    if QUICK:
        variants = [("dense", None), ("dense", rerankers[0])]

    for scope in ("per-video", "pooled"):
        print(f"=== scope: {scope} ===")
        print(f"{'pipeline':44} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'vs dense':>9} {'ms/q':>7}")
        base_ranks = None
        results = {}
        for variant, reranker in variants:
            ranks, ms = run(scope, variant, reranker)
            results[(variant, reranker)] = ranks
            if base_ranks is None:
                base_ranks = ranks
            n = len(ranks)
            h = lambda k: sum(1 for r in ranks if r and r <= k) / n
            mrr = sum(1 / r for r in ranks if r) / n
            better = sum(1 for r, b in zip(ranks, base_ranks) if (r or 99) < (b or 99))
            worse = sum(1 for r, b in zip(ranks, base_ranks) if (r or 99) > (b or 99))
            label = variant + (f" + {reranker.split('/')[-1]}" if reranker else "")
            print(f"{label:44} {h(1):>6.0%} {h(3):>6.0%} {h(5):>6.0%} {mrr:>6.3f} {f'+{better}/-{worse}':>9} {ms:>7.1f}")
        print()

        # MRR by question type for the interesting pipelines
        types = sorted({t for *_, t in questions})
        print(f"MRR by question type ({scope}):")
        print(f"{'pipeline':44} " + " ".join(f"{t:>10}" for t in types))
        for key in [("dense", None), ("bm25", None), ("hybrid-rrf", None), ("hybrid-rrf", rerankers[0]), ("dense", rerankers[0])]:
            if key not in results:
                continue
            ranks = results[key]
            cells = []
            for t in types:
                idx = [i for i, q in enumerate(questions) if q[4] == t]
                cells.append(sum(1 / ranks[i] for i in idx if ranks[i]) / len(idx))
            label = key[0] + (f" + {key[1].split('/')[-1]}" if key[1] else "")
            print(f"{label:44} " + " ".join(f"{c:>10.3f}" for c in cells))
        print()

        worst = [(questions[i][1], results[('dense', None)][i]) for i in range(len(questions)) if (results[('dense', None)][i] or 99) > 1]
        print(f"dense misses at rank 1 ({scope}): {len(worst)}")
        for q, r in worst:
            print(f"   rank {r!s:>4}  {q}")
        print()


if __name__ == "__main__":
    main()
