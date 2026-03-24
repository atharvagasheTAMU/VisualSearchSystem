"""Pydantic request/response schemas for the search API."""

from typing import Any

from pydantic import BaseModel


class SearchResult(BaseModel):
    image_id: int
    score: float
    path: str | None = None
    category: str | None = None
    sub_category: str | None = None
    article_type: str | None = None
    color: str | None = None
    season: str | None = None
    usage: str | None = None
    product_name: str | None = None
    caption: str | None = None
    ocr_text: str | None = None
    landmark: str | None = None
    city: str | None = None
    country: str | None = None
    entity_type: str | None = None
    entity_tags: list[str] | None = None
    topic_labels: list[str] | None = None
    explanation: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    query_image: str
    results: list[SearchResult]
    mode: str = "visual"
    total: int


class ImageMetadata(BaseModel):
    image_id: int
    path: str | None = None
    category: str | None = None
    sub_category: str | None = None
    article_type: str | None = None
    color: str | None = None
    season: str | None = None
    usage: str | None = None
    product_name: str | None = None
    caption: str | None = None
    ocr_text: str | None = None
    landmark: str | None = None
    city: str | None = None
    country: str | None = None
    entity_type: str | None = None
    entity_description: str | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    index_size: int
    device: str
