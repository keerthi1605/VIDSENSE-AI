"""
RAG (Retrieval-Augmented Generation) orchestration.

Pipeline: query -> retrieval_service.search() -> prompt construction ->
llm_service -> answer. This module owns the prompt template and the
"what counts as insufficient context" decision; it does NOT know which
LLM or vector store is behind it -- those stay abstracted behind
llm_service and retrieval_service respectively.

KEY DESIGN DECISION: sources are built directly from the retrieved
chunks, never parsed out of the LLM's generated text. A 3B local model
asked to also produce structured citations (e.g. JSON) is unreliable --
small models are notoriously bad at strict structured output. Instead,
the LLM's only job is the natural-language answer; the timestamps and
video IDs shown to the user are attached by our own code afterward,
from data we already trust (the same SearchResult objects passed into
the prompt). This guarantees every citation is exact, never invented.
"""

from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import ChatResponse, SearchResult, SourceRef
from app.services import llm_service, retrieval_service

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = (
    "I don't have enough information in the indexed video(s) to answer "
    "that question. Try uploading and indexing a relevant video first, "
    "or rephrase the question."
)

PROMPT_TEMPLATE = """You are an assistant that answers questions about video lecture content using ONLY the transcript excerpts provided below.

Rules:
- Answer using only the information in the excerpts below. Do not use outside knowledge.
- If the excerpts do not contain enough information to answer the question, say so explicitly instead of guessing.
- Be concise and factual. Do not repeat the excerpts verbatim; synthesize an answer.

Transcript excerpts:
{context}

Question: {query}

Answer:"""


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def build_prompt(query: str, chunks: List[SearchResult]) -> str:
    context = "\n\n".join(
        f"[Excerpt {i + 1}] "
        f"({_format_timestamp(c.start_time)}-{_format_timestamp(c.end_time)}): "
        f'"{c.text}"'
        for i, c in enumerate(chunks)
    )
    return PROMPT_TEMPLATE.format(context=context, query=query)


def answer_question(
    query: str, top_k: Optional[int] = None, video_id: Optional[str] = None
) -> ChatResponse:
    """
    Answer a question about indexed video content.

    has_sufficient_context reflects a deterministic check -- did
    retrieval find ANY chunks at all -- not whether the LLM's own
    answer text indicates it could actually answer. The prompt
    instructs the model to say so itself when the excerpts are
    insufficient; that's a real behavior, just not (yet) parsed back
    into a boolean, since reliably parsing free text from a small
    local model is its own source of bugs.
    """
    resolved_top_k = top_k or settings.rag_context_chunks
    search_response = retrieval_service.search(query, top_k=resolved_top_k, video_id=video_id)

    if not search_response.results:
        logger.info("No chunks retrieved for query %r -- skipping LLM call.", query)
        return ChatResponse(
            query=query, answer=NO_CONTEXT_ANSWER, sources=[], has_sufficient_context=False
        )

    prompt = build_prompt(query, search_response.results)
    provider = llm_service.get_llm_provider()
    answer_text = provider.generate(prompt)

    sources = [
        SourceRef(
            video_id=r.video_id,
            chunk_id=r.chunk_id,
            start_time=r.start_time,
            end_time=r.end_time,
            score=r.score,
        )
        for r in search_response.results
    ]
    return ChatResponse(query=query, answer=answer_text, sources=sources, has_sufficient_context=True)
