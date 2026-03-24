"""Metadata-aware reranker using category, color, and usage signals."""

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class MetadataReranker:
    def __init__(self, config: dict):
        weights = config["reranking"]["metadata_weights"]
        self.category_weight = weights.get("category", 0.15)
        self.color_weight = weights.get("color", 0.05)
        self.visual_weight = config["reranking"]["weights"].get("visual_similarity", 0.80)

    def _metadata_score(self, query_result: dict, candidate: dict) -> float:
        """Compute metadata match score between query metadata and a candidate."""
        score = 0.0
        if (
            query_result.get("category")
            and candidate.get("category")
            and query_result["category"] == candidate["category"]
        ):
            score += self.category_weight

        if (
            query_result.get("color")
            and candidate.get("color")
            and query_result["color"].lower() == candidate["color"].lower()
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
