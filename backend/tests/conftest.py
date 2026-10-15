"""
Shared test fixtures.

Cross-encoder reranking is ON in production (settings.retrieval_rerank), but
the vast majority of tests exercise plumbing around retrieval and must not
load a real model. It is switched off for every test by default; the tests
that exercise reranking turn it back on explicitly and fake the model.
"""

import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def _rerank_off_by_default(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_rerank", False)
