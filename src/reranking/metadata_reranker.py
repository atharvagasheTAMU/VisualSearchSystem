"""Metadata-aware reranker using category, color, and usage signals."""

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class MetadataReranker:
    def __init__(self, config: dict):
        weights = config["reranking"]["metadata_weights"]
        self.article_type_weight = weights.get("article_type", 0.20)
        self.sub_category_weight = weights.get("sub_category", 0.10)
        self.category_weight = weights.get("category", 0.05)
        self.color_weight = weights.get("color", 0.05)
        self.visual_weight = config["reranking"]["weights"].get("visual_similarity", 0.80)

    def _metadata_score(self, query_meta: dict, candidate: dict) -> float:
        """Compute metadata match score between query metadata and a candidate."""
        score = 0.0

        # article_type is the most specific signal (e.g. "Tshirts", "Casual Shoes")
        if (
            query_meta.get("article_type")
            and candidate.get("article_type")
            and query_meta["article_type"] == candidate["article_type"]
        ):
            score += self.article_type_weight

        # sub_category is a medium-specificity signal (e.g. "Topwear", "Shoes")
        if (
            query_meta.get("sub_category")
            and candidate.get("sub_category")
            and query_meta["sub_category"] == candidate["sub_category"]
        ):
            score += self.sub_category_weight

        # masterCategory is a broad signal — only add if neither above matched
        if (
            score == 0.0
            and query_meta.get("category")
            and candidate.get("category")
            and query_meta["category"] == candidate["category"]
        ):
            score += self.category_weight

        if (
            query_meta.get("color")
            and candidate.get("color")
            and query_meta["color"].lower() == candidate["color"].lower()
        ):
            score += self.color_weight

        return score

    def rerank(
        self,
        results: list[dict],
        query_meta: dict | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Rerank retrieval results using metadata signals.

        Args:
            results:    list of result dicts from Searcher.search() (already have visual score)
            query_meta: metadata dict for the query image (if available)
                        If None, uses the first result as a proxy for query context.
            top_k:      number of results to return

        Returns:
            Reranked and trimmed list of result dicts with updated 'score'.
        """
        if not results:
            return results

        # If query metadata is unknown, use a pseudo-query from top result's context
        if query_meta is None:
            query_meta = results[0] if results else {}

        scored = []
        for r in results:
            visual_score = float(r.get("score", 0.0))
            meta_score = self._metadata_score(query_meta, r)
            final_score = self.visual_weight * visual_score + meta_score

            r_copy = dict(r)
            r_copy["visual_score"] = visual_score
            r_copy["metadata_score"] = meta_score
            r_copy["score"] = final_score
            r_copy["explanation"] = {
                "visual_similarity": visual_score,
                "metadata_match": meta_score,
            }
            scored.append(r_copy)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
