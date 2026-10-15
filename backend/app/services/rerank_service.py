"""
Cross-encoder reranking: a second, more careful look at the top candidates.

Retrieval (dense or hybrid) scores a query and a chunk INDEPENDENTLY -- each
is squeezed into a vector first, so fine interactions between the two are
lost. A cross-encoder reads the query and the chunk TOGETHER in one forward
pass and scores their relevance directly: much more accurate, but far too
slow to run over every chunk, so it only reorders the few candidates the
first stage already shortlisted.

Same lazy-singleton pattern as the other models: loading weights costs real
time, so it is paid once per process, never at import. `get_reranker()` is the
seam tests monkeypatch to avoid loading a real model.
"""

from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_model: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info(
            "Loading reranker '%s' (device=%s)...", settings.reranker_model_name, settings.reranker_device
        )
        _model = CrossEncoder(settings.reranker_model_name, device=settings.reranker_device)
        logger.info("Reranker loaded.")
    return _model


def rerank_scores(query: str, passages: List[str]) -> np.ndarray:
    """Relevance score of each passage for the query (higher = more
    relevant; raw model logits, only meaningful relative to each other)."""
    if not passages:
        return np.empty(0, dtype=np.float32)
    model = get_reranker()
    return np.asarray(
        model.predict([(query, p) for p in passages], show_progress_bar=False), dtype=np.float32
    )
