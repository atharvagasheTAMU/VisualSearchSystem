"""GET /images/{id} endpoint - serve image file and metadata."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from src.api.models import ImageMetadata

router = APIRouter()


@router.get("/images/{image_id}/metadata", response_model=ImageMetadata)
async def get_image_metadata(image_id: int, request: Request):
    """Return metadata for a given image ID."""
    state = request.app.state
    meta_index = state.searcher.meta_index

    if image_id not in meta_index.index:
        raise HTTPException(status_code=404, detail=f"Image ID {image_id} not found.")

    row = meta_index.loc[image_id].to_dict()

    entity_tags = row.get("entity_tags")
    if isinstance(entity_tags, str):
        entity_tags = entity_tags

    return ImageMetadata(
        image_id=image_id,
        path=row.get("path"),
        category=row.get("category"),
        sub_category=row.get("sub_category"),
        article_type=row.get("article_type"),
        color=row.get("color"),
        season=row.get("season"),
        usage=row.get("usage"),
        product_name=row.get("product_name"),
        caption=row.get("caption"),
        ocr_text=row.get("ocr_text"),
        landmark=row.get("landmark"),
        city=row.get("city"),
        country=row.get("country"),
        entity_type=row.get("entity_type"),
        entity_description=row.get("entity_description"),
    )


@router.get("/images/{image_id}")
async def get_image_file(image_id: int, request: Request):
    """Serve the actual image file for a given image ID."""
    state = request.app.state
    meta_index = state.searcher.meta_index

    if image_id not in meta_index.index:
        raise HTTPException(status_code=404, detail=f"Image ID {image_id} not found.")

    path = meta_index.loc[image_id].get("path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"Image file not found for ID {image_id}.")

    return FileResponse(path, media_type="image/jpeg")
