"""
RAG question-answering endpoint.

Single-turn Q&A (no conversation history/memory) -- the master
roadmap's Phase 4 scope is retrieval-grounded answering with
citations, not a stateful chat feature. Conversation memory would be
a real feature to add later, deliberately not built here to avoid
scope creep this phase.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services import rag_service

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer a natural-language question using retrieval-augmented
    generation: relevant chunks are retrieved first, then an LLM
    answers using only that retrieved text as context. Every answer
    comes with exact source timestamps -- see SourceRef.
    """
    try:
        return rag_service.answer_question(
            query=request.query, top_k=request.top_k, video_id=request.video_id
        )
    except RuntimeError as exc:
        # LLM provider unreachable/timed out -- a real operational
        # failure, not a client error, but not a generic 500 either.
        raise HTTPException(status_code=503, detail=str(exc))
