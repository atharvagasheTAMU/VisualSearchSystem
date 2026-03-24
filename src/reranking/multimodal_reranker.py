"""
Multimodal reranker combining visual, caption, metadata, geo, entity, and topic signals.

Scoring formula:
    final_score =
        0.55 * visual_similarity +
        0.15 * caption_similarity +
        0.10 * metadata_match +
        0.10 * geo_proximity +
        0.05 * entity_tag_match +
        0.05 * topic_trend_score
"""

from pathlib import Path

import numpy as np
import yaml


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _tag_overlap(tags_a: list[str] | str | None, tags_b: list[str] | str | None) -> float:
    """Jaccard-style overlap between two tag lists."""
    def to_set(tags) -> set[str]:
        if not tags:
            return set()
        if isinstance(tags, str):
            return {t.strip().lower() for t in tags.split(",") if t.strip()}
        return {str(t).lower() for t in tags}

    a, b = to_set(tags_a), to_set(tags_b)
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _geo_score(query_row: dict, candidate: dict) -> float:
    """Simple geographic proximity score."""
    if (
        query_row.get("country")
        and candidate.get("country")
        and query_row["country"] == candidate["country"]
    ):
        if query_row.get("city") and candidate.get("city") and query_row["city"] == candidate["city"]:
            return 1.0
        return 0.5
    return 0.0


class MultimodalReranker:
    def __init__(self, config: dict, clip_encoder=None):
        weights = config["reranking"]["weights"]
        self.w_visual = weights.get("visual_similarity", 0.55)
        self.w_caption = weights.get("caption_similarity", 0.15)
        self.w_metadata = weights.get("metadata_match", 0.10)
        self.w_geo = weights.get("geo_proximity", 0.10)
        self.w_entity = weights.get("entity_tag_match", 0.05)
        self.w_topic = weights.get("topic_trend", 0.05)

        self.encoder = clip_encoder

        # Load topic index if available
        self.topic_index = None
        try:
            from src.enrichment.topic_index import TopicIndex
            ti = TopicIndex()
            ti.load()
            self.topic_index = ti
        except Exception:
            pass

        self.metadata_reranker_weights = config["reranking"].get("metadata_weights", {})

    def _caption_similarity(self, query_caption: str | None, candidate_caption: str | None) -> float:
        """Compute cosine similarity between two captions using CLIP text encoder."""
        if not self.encoder or not query_caption or not candidate_caption:
            return 0.0
        try:
            embs = self.encoder.encode_text([query_caption, candidate_caption])
            return float(np.dot(embs[0], embs[1]))
        except Exception:
            return 0.0

    def _metadata_match(self, query_row: dict, candidate: dict) -> float:
        score = 0.0
        cat_w = self.metadata_reranker_weights.get("category", 0.15)
        col_w = self.metadata_reranker_weights.get("color", 0.05)

        if (
            query_row.get("category")
            and candidate.get("category")
            and query_row["category"] == candidate["category"]
        ):
            score += cat_w

        if (
            query_row.get("color")
            and candidate.get("color")
            and str(query_row["color"]).lower() == str(candidate["color"]).lower()
        ):
            score += col_w

        return min(score, 1.0)

    def _topic_score(self, caption: str | None) -> float:
        """Match caption to topic index; return best topic similarity score."""
        if not self.topic_index or not self.encoder or not caption:
            return 0.0
        try:
            caption_emb = self.encoder.encode_text([caption])[0]
            hits = self.topic_index.search(caption_emb, top_k=1)
            if hits:
                return float(hits[0].get("topic_score", 0.0))
        except Exception:
            pass
        return 0.0

    def rerank(
        self,
        results: list[dict],
        query_meta: dict | None = None,
        query_caption: str | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Rerank candidates using all available signals.

        Args:
            results:        raw retrieval results from Searcher
            query_meta:     metadata dict for the query image
            query_caption:  caption string for the query image
            top_k:          number of results to return
        """
        if not results:
            return results

        if query_meta is None:
            query_meta = {}

        # Infer query caption from metadata if not provided
        if not query_caption:
            query_caption = str(query_meta.get("caption", "") or "")

        # Pre-compute topic score for query (same signal for all candidates)
        query_topic_score = self._topic_score(query_caption)

        scored = []
        for r in results:
            visual_score = _safe_float(r.get("score"))
            caption_sim = self._caption_similarity(query_caption, r.get("caption"))
            meta_match = self._metadata_match(query_meta, r)
            geo = _geo_score(query_meta, r)
            entity_overlap = _tag_overlap(
                query_meta.get("entity_tags"), r.get("entity_tags")
            )
            topic_score = query_topic_score  # same for all candidates in this query

            final_score = (
                self.w_visual * visual_score
                + self.w_caption * caption_sim
                + self.w_metadata * meta_match
                + self.w_geo * geo
                + self.w_entity * entity_overlap
                + self.w_topic * topic_score
            )

            r_copy = dict(r)
            r_copy["score"] = final_score
            r_copy["explanation"] = {
                "visual_similarity": round(visual_score, 4),
                "caption_similarity": round(caption_sim, 4),
                "metadata_match": round(meta_match, 4),
                "geo_proximity": round(geo, 4),
                "entity_tag_match": round(entity_overlap, 4),
                "topic_trend_score": round(topic_score, 4),
            }
            scored.append(r_copy)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
