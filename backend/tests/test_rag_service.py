"""
Tests for app.services.rag_service, with retrieval_service and
llm_service mocked -- this layer's job is orchestration (when to call
the LLM at all, how to build the prompt, how to attach sources), which
is exactly what's under test here.
"""

import pytest

from app.models.schemas import SearchResponse, SearchResult
from app.services import llm_service, rag_service, retrieval_service


class FakeProvider:
    def __init__(self, response_text="a synthesized answer"):
        self.response_text = response_text
        self.received_prompt = None

    def generate(self, prompt):
        self.received_prompt = prompt
        return self.response_text


def _search_result(video_id="v1", chunk_id="v1_0000", start=0.0, end=5.0, score=0.9, text="excerpt text"):
    return SearchResult(
        video_id=video_id, chunk_id=chunk_id, text=text,
        start_time=start, end_time=end, score=score,
    )


def test_build_prompt_formats_timestamps_and_numbers_excerpts():
    chunks = [
        _search_result(start=65.0, end=131.0, text="first excerpt"),
        _search_result(start=3725.0, end=3800.0, text="second excerpt"),
    ]
    prompt = rag_service.build_prompt("my question", chunks)

    assert "[Excerpt 1] (1:05-2:11): \"first excerpt\"" in prompt
    assert "[Excerpt 2] (1:02:05-1:03:20): \"second excerpt\"" in prompt
    assert "Question: my question" in prompt
    assert "ONLY the transcript excerpts" in prompt


def test_answer_question_skips_llm_when_no_results(monkeypatch):
    monkeypatch.setattr(
        retrieval_service, "search",
        lambda query, top_k=None, video_id=None: SearchResponse(query=query, results=[]),
    )

    fake_provider = FakeProvider()
    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: fake_provider)

    response = rag_service.answer_question("anything")

    assert response.has_sufficient_context is False
    assert response.sources == []
    assert response.answer == rag_service.NO_CONTEXT_ANSWER
    assert fake_provider.received_prompt is None  # LLM never called


def test_answer_question_calls_llm_and_attaches_sources(monkeypatch):
    result = _search_result(video_id="v1", chunk_id="v1_0000", start=10.0, end=20.0, score=0.85)
    monkeypatch.setattr(
        retrieval_service, "search",
        lambda query, top_k=None, video_id=None: SearchResponse(query=query, results=[result]),
    )

    fake_provider = FakeProvider(response_text="Binary search is a divide-and-conquer algorithm.")
    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: fake_provider)

    response = rag_service.answer_question("what is binary search?")

    assert response.has_sufficient_context is True
    assert response.answer == "Binary search is a divide-and-conquer algorithm."
    assert len(response.sources) == 1
    assert response.sources[0].video_id == "v1"
    assert response.sources[0].chunk_id == "v1_0000"
    assert response.sources[0].start_time == 10.0
    assert response.sources[0].end_time == 20.0
    assert response.sources[0].score == 0.85
    # The prompt actually sent to the LLM included the retrieved text.
    assert "excerpt text" in fake_provider.received_prompt
    assert "what is binary search?" in fake_provider.received_prompt


def test_answer_question_passes_video_id_and_top_k_through(monkeypatch):
    captured = {}

    def fake_search(query, top_k=None, video_id=None):
        captured["top_k"] = top_k
        captured["video_id"] = video_id
        return SearchResponse(query=query, results=[])

    monkeypatch.setattr(retrieval_service, "search", fake_search)
    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: FakeProvider())

    rag_service.answer_question("q", top_k=3, video_id="v42")

    assert captured["top_k"] == 3
    assert captured["video_id"] == "v42"


def test_answer_question_propagates_llm_errors(monkeypatch):
    result = _search_result()
    monkeypatch.setattr(
        retrieval_service, "search",
        lambda query, top_k=None, video_id=None: SearchResponse(query=query, results=[result]),
    )

    class FailingProvider:
        def generate(self, prompt):
            raise RuntimeError("Ollama is down")

    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: FailingProvider())

    with pytest.raises(RuntimeError, match="Ollama is down"):
        rag_service.answer_question("anything")


# ---------------------------------------------------------------------
# Relevance gate (cross-encoder score decides "out of scope" up front)
# ---------------------------------------------------------------------

def _reranked(rerank_score, **kwargs):
    result = _search_result(**kwargs)
    return result.model_copy(update={"rerank_score": rerank_score})


def _stub_search(monkeypatch, results):
    monkeypatch.setattr(
        retrieval_service, "search",
        lambda query, top_k=None, video_id=None: SearchResponse(query=query, results=results),
    )
    provider = FakeProvider()
    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: provider)
    return provider


def test_gate_refuses_without_calling_llm_when_best_rerank_score_is_too_low(monkeypatch):
    from app.core.config import settings

    provider = _stub_search(monkeypatch, [_reranked(settings.rag_min_relevance - 0.5)])
    response = rag_service.answer_question("what is the capital of France?")

    assert response.has_sufficient_context is False
    assert response.answer == rag_service.NO_CONTEXT_ANSWER
    assert response.sources == []
    assert provider.received_prompt is None  # LLM never called


def test_gate_lets_relevant_questions_through(monkeypatch):
    from app.core.config import settings

    provider = _stub_search(monkeypatch, [_reranked(settings.rag_min_relevance + 3.0)])
    response = rag_service.answer_question("a relevant question")

    assert response.has_sufficient_context is True
    assert provider.received_prompt is not None


def test_gate_only_looks_at_the_top_result(monkeypatch):
    """Results arrive best-first; a weak supporting chunk below a strong
    top chunk must not trigger a refusal."""
    from app.core.config import settings

    _stub_search(
        monkeypatch,
        [_reranked(4.0, chunk_id="v1_0000"), _reranked(settings.rag_min_relevance - 5, chunk_id="v1_0001")],
    )
    response = rag_service.answer_question("a relevant question")
    assert response.has_sufficient_context is True
    assert len(response.sources) == 2


def test_gate_is_skipped_when_results_were_not_reranked(monkeypatch):
    """Without the cross-encoder (rerank_score None) there is nothing to
    gate on -- behaviour is exactly the pre-gate behaviour."""
    provider = _stub_search(monkeypatch, [_search_result(score=0.05)])
    response = rag_service.answer_question("anything")
    assert response.has_sufficient_context is True
    assert provider.received_prompt is not None


def test_sources_carry_the_localized_jump_time(monkeypatch):
    result = _search_result(start=100.0, end=170.0).model_copy(update={"jump_time": 142.5})
    _stub_search(monkeypatch, [result])
    response = rag_service.answer_question("q")
    assert response.sources[0].jump_time == 142.5
    assert response.sources[0].start_time == 100.0  # the chunk range is still reported
