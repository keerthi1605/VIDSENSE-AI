"""
Semantic search over indexed video chunks.

No LLM here -- this is the "standalone semantic-search API" the
roadmap calls for before an LLM gets involved (Phase 4). Every result
is directly traceable to a video_id, an exact timestamp range, and a
similarity score -- nothing is generated or paraphrased.
"""

from typing import Optional

from fastapi import APIRouter, Query

from app.models.schemas import SearchRequest, SearchResponse
from app.services import retrieval_service

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search_post(request: SearchRequest) -> SearchResponse:
    """Semantic search via a JSON body -- the primary form, since a
    search request has enough optional fields (top_k, video_id) that
    query-string encoding gets awkward."""
    return retrieval_service.search(
        query=request.query, top_k=request.top_k, video_id=request.video_id
    )


@router.get("/search", response_model=SearchResponse)
def search_get(
    query: str = Query(..., description="Natural-language search query"),
    top_k: Optional[int] = Query(None, ge=1, le=50),
    video_id: Optional[str] = Query(None, description="Restrict search to one video"),
) -> SearchResponse:
    """Same search, as a GET with query params -- convenient for a
    simple search box or quick manual testing (curl/browser)."""
    return retrieval_service.search(query=query, top_k=top_k, video_id=video_id)
