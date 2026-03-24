"""POST /search endpoint."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from src.api.models import SearchResponse, SearchResult

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    request: Request,
    file: UploadFile = File(...),
    k: int = Query(default=10, ge=1, le=50),
    mode: str = Query(default="visual", description="visual | reranked | full"),
):
    """Upload a query image and retrieve the top-k visually similar images."""
    state = request.app.state

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "query.jpg").suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        candidate_k = max(k * 5, state.config["retrieval"]["candidate_pool"])
        raw_results = state.searcher.search(tmp_path, k=candidate_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if mode == "reranked" and hasattr(state, "metadata_reranker"):
        reranked = state.metadata_reranker.rerank(raw_results, top_k=k)
    elif mode == "full" and hasattr(state, "multimodal_reranker"):
        reranked = state.multimodal_reranker.rerank(raw_results, top_k=k)
    else:
        reranked = raw_results[:k]

    results = []
    for r in reranked:
        entity_tags = r.get("entity_tags")
        if isinstance(entity_tags, str):
            entity_tags = [t.strip() for t in entity_tags.split(",") if t.strip()]

        topic_labels = r.get("topic_labels")
        if isinstance(topic_labels, str):
            topic_labels = [t.strip() for t in topic_labels.split(",") if t.strip()]

        results.append(
            SearchResult(
                image_id=int(r["image_id"]),
                score=float(r.get("score", 0.0)),
                path=r.get("path"),
                category=r.get("category"),
                sub_category=r.get("sub_category"),
                article_type=r.get("article_type"),
                color=r.get("color"),
                season=r.get("season"),
                usage=r.get("usage"),
                product_name=r.get("product_name"),
                caption=r.get("caption"),
                ocr_text=r.get("ocr_text"),
                landmark=r.get("landmark"),
                city=r.get("city"),
                country=r.get("country"),
                entity_type=r.get("entity_type"),
                entity_tags=entity_tags,
                topic_labels=topic_labels,
                explanation=r.get("explanation"),
            )
        )

    return SearchResponse(
        query_image=file.filename or "upload",
        results=results,
        mode=mode,
        total=len(results),
    )
