"""Retrieval evaluation metrics: Recall@K, Precision@K, mAP, nDCG."""

import numpy as np


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Precision@K: fraction of top-k retrieved that are relevant."""
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Recall@K: fraction of all relevant items that appear in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list[int], relevant: set[int]) -> float:
    """Average Precision for a single query."""
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant) if relevant else 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """nDCG@K for a single query (binary relevance)."""
    top_k = retrieved[:k]

    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    queries: list[dict],
    k_values: list[int] = (5, 10, 20),
) -> dict:
    """
    Compute mean metrics over a list of query evaluations.

    Each entry in `queries` is a dict:
        {
          "retrieved": [image_id, ...],  # ranked list
          "relevant":  {image_id, ...},  # set of relevant image IDs
        }

    Returns a dict of metric_name -> float.
    """
    results: dict[str, list[float]] = {}

    for k in k_values:
        results[f"Precision@{k}"] = []
        results[f"Recall@{k}"] = []
        results[f"nDCG@{k}"] = []

    results["mAP"] = []

    for q in queries:
        retrieved = q["retrieved"]
        relevant = q["relevant"]

        for k in k_values:
            results[f"Precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            results[f"Recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            results[f"nDCG@{k}"].append(ndcg_at_k(retrieved, relevant, k))

        results["mAP"].append(average_precision(retrieved, relevant))

    return {metric: float(np.mean(vals)) for metric, vals in results.items()}


def print_metrics_table(metrics: dict[str, float], label: str = "") -> None:
    header = f"  {'Metric':<20} {'Value':>8}"
    if label:
        print(f"\n{'='*40}")
        print(f"  {label}")
        print(f"{'='*40}")
    else:
        print(f"\n{'-'*35}")
    print(header)
    print(f"  {'-'*30}")
    for metric, value in sorted(metrics.items()):
        print(f"  {metric:<20} {value:>8.4f}")
