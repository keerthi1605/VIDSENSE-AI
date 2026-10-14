"""
Retrieval evaluation: does content-aligned chunking retrieve the right part
of a video better than plain word-count chunking?

Run from backend/:

    venv/Scripts/python.exe scripts/retrieval_eval.py

Method
  * QUESTIONS below are written from the CS101 transcript text, each with
    the time span (seconds) where the transcript actually answers it. The
    spans come from reading the transcript, NOT from chunk or scene
    boundaries, so they are independent of both chunking strategies.
  * Each strategy chunks the same transcript, embeds the chunks with the
    production MiniLM model, and ranks them by cosine similarity per
    question -- the same retrieval the app uses, minus ChromaDB.
  * A chunk "hits" a question when it covers at least HIT_COVERAGE of the
    answer span.

Strategies compared
  word-count@150  the original: ceiling 150 words (what was indexed before)
  word-count@75   SIZE-MATCHED control: same algorithm, ceiling 75 words, so
                  its chunks are about as small as the scene-aligned ones.
                  Without this, any gain could just be "smaller chunks".
  scene-aligned   ceiling 150 + cuts at SSIM scene changes (the new default)

Metrics: Hit@1/3/5 (5 = how many chunks the RAG prompt actually receives),
MRR, and the words the LLM must read (top-5 context size). A strategy that
only wins by returning bigger chunks would show up as a larger context.

LIMITS, stated plainly: 18 questions, one video, labels written by the
developer. Treat differences of a question or two as noise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services import chunking_service, embedding_service, transcription_service, video_service  # noqa: E402
from segmentation_benchmark import detect_boundaries, sample_frames  # noqa: E402

VIDEO_ID = "379884270e9545fc8fbaddc24bda7e30"  # CS101 (Treehouse)
HIT_COVERAGE = 0.5
TOP_K_CONTEXT = settings.rag_context_chunks  # chunks sent to the LLM

# (question, answer_start_s, answer_end_s)
QUESTIONS = [
    ("What movie is used to explain what a computer originally was?", 76, 100),
    ("What did Dijkstra say about computers and astronomy?", 106, 116),
    ("How would you define computer science?", 132, 146),
    ("Who is considered the father of the computer?", 178, 186),
    ("How big, heavy and expensive was the ENIAC?", 212, 230),
    ("Which computer is the best selling of all time?", 263, 280),
    ("How much faster is the iPhone than the ENIAC?", 297, 313),
    ("Who invented the world wide web and when?", 358, 367),
    ("What is an input device?", 445, 468),
    ("Why does RAM lose its data when the computer is turned off?", 507, 519),
    ("What is cloud storage and what are some examples?", 545, 566),
    ("What is the difference between system software and application software?", 583, 631),
    ("What is a bit?", 681, 706),
    ("What do the prefixes kilo, mega and giga mean?", 746, 800),
    ("How does the kitchen cabinet analogy explain data types?", 857, 900),
    ("Which programming languages require declaring a variable's type first?", 902, 928),
    ("What is a Boolean value?", 1005, 1035),
    ("How does the peanut butter and jelly sandwich explain an algorithm?", 1179, 1250),
]


def coverage(chunk, a_start, a_end):
    overlap = max(0.0, min(chunk.end_time, a_end) - max(chunk.start_time, a_start))
    return overlap / (a_end - a_start)


def evaluate(chunks, query_vecs):
    chunk_vecs = embedding_service.embed_texts([c.text for c in chunks])
    sims = query_vecs @ chunk_vecs.T  # vectors are unit-length -> cosine
    ranks, hits_at = [], {1: 0, 3: 0, 5: 0}
    context_words = []
    for qi, (_, a_start, a_end) in enumerate(QUESTIONS):
        order = np.argsort(-sims[qi])
        rank = next(
            (r + 1 for r, ci in enumerate(order) if coverage(chunks[ci], a_start, a_end) >= HIT_COVERAGE),
            None,
        )
        ranks.append(rank)
        for k in hits_at:
            hits_at[k] += rank is not None and rank <= k
        context_words.append(sum(chunks[ci].word_count for ci in order[:TOP_K_CONTEXT]))
    n = len(QUESTIONS)
    mrr = sum(1 / r for r in ranks if r) / n
    return {
        "chunks": len(chunks),
        "hit@1": hits_at[1] / n,
        "hit@3": hits_at[3] / n,
        "hit@5": hits_at[5] / n,
        "mrr": mrr,
        "ctx_words": sum(context_words) / n,
        "ranks": ranks,
    }


def main():
    transcript = transcription_service.get_transcript(VIDEO_ID)
    grays, timestamps = sample_frames(video_service.get_video_file_path(VIDEO_ID))
    boundaries = detect_boundaries(grays, timestamps, "ssim")

    query_vecs = embedding_service.embed_texts([q for q, _, _ in QUESTIONS])

    results = {}
    original_target = settings.chunk_target_words
    try:
        settings.chunk_target_words = 150
        results["word-count@150"] = evaluate(chunking_service.chunk_transcript(transcript), query_vecs)
        results["scene-aligned"] = evaluate(
            chunking_service.chunk_transcript(transcript, scene_boundaries=boundaries), query_vecs
        )
        settings.chunk_target_words = 75
        results["word-count@75"] = evaluate(chunking_service.chunk_transcript(transcript), query_vecs)
    finally:
        settings.chunk_target_words = original_target

    print(f"{len(QUESTIONS)} questions, {len(boundaries)} SSIM scene boundaries, hit = chunk covers >= {HIT_COVERAGE:.0%} of the answer span\n")
    print(f"{'strategy':16} {'chunks':>6} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'ctx words':>9}")
    for name in ("word-count@150", "word-count@75", "scene-aligned"):
        r = results[name]
        print(f"{name:16} {r['chunks']:>6} {r['hit@1']:>6.0%} {r['hit@3']:>6.0%} {r['hit@5']:>6.0%} {r['mrr']:>6.2f} {r['ctx_words']:>9.0f}")

    print("\nper-question rank of the first answer-covering chunk (None = never in the ranking):")
    for qi, (q, a, b) in enumerate(QUESTIONS):
        row = "  ".join(f"{results[n]['ranks'][qi]!s:>4}" for n in ("word-count@150", "word-count@75", "scene-aligned"))
        print(f"  {row}   [{a:>4}-{b:<4}] {q}")
    print("  (columns: word-count@150, word-count@75, scene-aligned)")


if __name__ == "__main__":
    main()
