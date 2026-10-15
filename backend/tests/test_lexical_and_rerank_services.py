"""
Tests for lexical_service (BM25) and rerank_service. The cross-encoder model
itself is never loaded: get_reranker is faked so only our own wiring --
pairing the query with each passage, empty input -- is under test.
"""

import numpy as np

from app.services import lexical_service, rerank_service


def test_tokenize_lowercases_drops_stopwords_and_stems_plurals():
    assert lexical_service.tokenize("The Computers and the Companies") == ["computer", "company"]


def test_bm25_ranks_the_document_with_the_rare_query_term_first():
    docs = [
        "the commodore 64 was the best selling computer of all time",
        "a computer has input and output devices",
        "computers are tools in computer science",
    ]
    scores = lexical_service.BM25Index(docs).scores("commodore 64 computer")
    assert int(np.argmax(scores)) == 0


def test_bm25_scores_zero_when_no_term_overlaps_and_handles_empty_input():
    index = lexical_service.BM25Index(["alpha beta", "gamma delta"])
    assert index.scores("zzz") == [0.0, 0.0]
    assert lexical_service.BM25Index([]).scores("anything") == []


def test_bm25_matches_plural_to_singular():
    scores = lexical_service.BM25Index(["file sizes are measured in bytes", "unrelated words here"]).scores("byte")
    assert scores[0] > 0 and scores[1] == 0


def test_bm25_prefers_a_shorter_document_for_the_same_term_count():
    docs = ["bit " + "filler " * 40, "bit short doc"]
    scores = lexical_service.BM25Index(docs).scores("bit")
    assert scores[1] > scores[0]


class FakeCrossEncoder:
    def __init__(self):
        self.pairs = None

    def predict(self, pairs, show_progress_bar=False):
        self.pairs = pairs
        return [float(len(p[1])) for p in pairs]  # longer passage = "more relevant"


def test_rerank_scores_pairs_the_query_with_every_passage(monkeypatch):
    fake = FakeCrossEncoder()
    monkeypatch.setattr(rerank_service, "get_reranker", lambda: fake)

    scores = rerank_service.rerank_scores("my question", ["aa", "bbbb"])

    assert fake.pairs == [("my question", "aa"), ("my question", "bbbb")]
    assert list(scores) == [2.0, 4.0]


def test_rerank_scores_empty_input_never_loads_the_model(monkeypatch):
    def boom():
        raise AssertionError("model must not load for empty input")

    monkeypatch.setattr(rerank_service, "get_reranker", boom)
    assert len(rerank_service.rerank_scores("q", [])) == 0
