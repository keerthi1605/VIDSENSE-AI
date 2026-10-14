"""
Semantic search over indexed video chunks and frames.

No LLM here -- this is the "standalone semantic-search API" the
roadmap calls for before an LLM gets involved (Phase 4). Every result
is directly traceable to a video_id, an exact timestamp, and a
similarity score -- nothing is generated or paraphrased.

/search (text) and /search/visual (frames) are deliberately separate,
unfused endpoints -- same reasoning as building text search standalone
before Phase 4 added the LLM: prove each retrieval channel works on its
own; Phase 6 adds the fused /search/multimodal endpoint on top of them.
"""

from typing import Optional

from fastapi import APIRouter, Query

from app.models.schemas import (
    MultimodalSearchRequest,
    MultimodalSearchResponse,
    SearchRequest,
    SearchResponse,
    VisualSearchRequest,
    VisualSearchResponse,
)
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


@router.post("/search/visual", response_model=VisualSearchResponse)
def search_visual_post(request: VisualSearchRequest) -> VisualSearchResponse:
    """
    Visual search via a JSON body: the query is embedded through
    CLIP's text encoder (not MiniLM) and matched against indexed frame
    vectors -- a completely separate search from /search.
    """
    return retrieval_service.search_frames(
        query=request.query, top_k=request.top_k, video_id=request.video_id
    )


@router.get("/search/visual", response_model=VisualSearchResponse)
def search_visual_get(
    query: str = Query(..., description="Natural-language visual search query"),
    top_k: Optional[int] = Query(None, ge=1, le=50),
    video_id: Optional[str] = Query(None, description="Restrict search to one video"),
) -> VisualSearchResponse:
    """Same visual search, as a GET with query params."""
    return retrieval_service.search_frames(query=query, top_k=top_k, video_id=video_id)


@router.post("/search/multimodal", response_model=MultimodalSearchResponse)
def search_multimodal_post(request: MultimodalSearchRequest) -> MultimodalSearchResponse:
    """
    Fused search (Phase 6): transcript chunks and video frames matched
    for the same query, aligned by timestamp and ranked together. Each
    result says whether it matched by text, visually, or both.
    """
    return retrieval_service.search_multimodal(
        query=request.query,
        top_k=request.top_k,
        video_id=request.video_id,
        text_weight=request.text_weight,
        visual_weight=request.visual_weight,
    )


@router.get("/search/multimodal", response_model=MultimodalSearchResponse)
def search_multimodal_get(
    query: str = Query(..., description="Natural-language query"),
    top_k: Optional[int] = Query(None, ge=1, le=50),
    video_id: Optional[str] = Query(None, description="Restrict search to one video"),
    text_weight: Optional[float] = Query(None, ge=0),
    visual_weight: Optional[float] = Query(None, ge=0),
) -> MultimodalSearchResponse:
    """Same fused search, as a GET with query params."""
    return retrieval_service.search_multimodal(
        query=query,
        top_k=top_k,
        video_id=video_id,
        text_weight=text_weight,
        visual_weight=visual_weight,
    )
