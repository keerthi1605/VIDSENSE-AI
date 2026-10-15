"""
Lexical (keyword) scoring with Okapi BM25 -- the other half of hybrid search.

Dense embeddings (MiniLM) match MEANING but can blur exact terms: a question
about "the Commodore 64" or "$500,000" is about a specific token sequence,
which a bag of word-pieces squeezed into 384 numbers may rank below a vaguely
related chunk. BM25 scores by exact (lightly stemmed) term overlap, weighting
rare terms more -- strong exactly where embeddings are weak, and weak (no
synonyms, no paraphrase) exactly where they are strong. Hybrid search fuses
the two.

Implemented here in ~40 lines rather than adding a dependency: the corpus is
one video's chunks (tens to hundreds of short documents), so there is nothing
to optimise, and "no fake AI" applies to libraries too.
"""

import math
import re
from collections import Counter
from typing import List

# Small, conservative list: words so common they carry no ranking signal.
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i if in into is it its "
    "me my of on or our she so than that the their them then there these they this to "
    "was we were what when where which who why will with you your do does did how".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Strip the commonest plural endings so "computers" matches "computer".
    Deliberately crude: a real stemmer risks mangling proper nouns."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """Okapi BM25 over a fixed list of documents (k1=1.5, b=0.75 -- the
    standard defaults)."""

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._term_freqs = [Counter(tokenize(d)) for d in documents]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_length = (sum(self._lengths) / len(documents)) if documents else 0.0
        document_freq: Counter = Counter()
        for tf in self._term_freqs:
            document_freq.update(tf.keys())
        n = len(documents)
        # +1 inside the log keeps IDF positive even for a term in every document.
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for term, df in document_freq.items()
        }

    def scores(self, query: str) -> List[float]:
        """BM25 score of every document for the query (0.0 when no term
        overlaps), in document order."""
        query_terms = tokenize(query)
        out = []
        for tf, length in zip(self._term_freqs, self._lengths):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_length) if self._avg_length else 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            out.append(score)
        return out
