"""
Run offline evaluation for all retrieval modes and print comparison table.

Modes evaluated:
  1. visual-only
  2. visual + metadata reranking
  3. visual + context (caption/geo/entity) - Week 2
  4. visual + full context + topic reranking - Week 2

Usage:
    python scripts/run_evaluation.py
"""

import csv
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.eval_set import build_eval_set, load_eval_set
from src.evaluation.metrics import compute_metrics, print_metrics_table
from src.retrieval.search import Searcher


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_evaluation(config: dict) -> None:
    k_values = config["evaluation"]["k_values"]
    candidate_pool = config["retrieval"]["candidate_pool"]
    eval_set_size = config["evaluation"]["eval_set_size"]
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build or load eval set
    eval_path = Path(config["paths"]["eval_dir"]) / "eval_queries.csv"
    if not eval_path.exists():
        print("Building eval set ...")
        build_eval_set(config, n_queries=eval_set_size)
    eval_queries = load_eval_set(config)
    print(f"Loaded {len(eval_queries)} eval queries.")

    searcher = Searcher(config)
    metadata_df = pd.read_csv(config["paths"]["metadata_csv"])
    metadata_df["image_id"] = metadata_df["image_id"].astype(int)
    meta_index = metadata_df.set_index("image_id")

    # Load rerankers if available
    metadata_reranker = None
    try:
        from src.reranking.metadata_reranker import MetadataReranker
        metadata_reranker = MetadataReranker(config)
    except Exception as e:
        print(f"MetadataReranker unavailable: {e}")

    multimodal_reranker = None
    try:
        from src.reranking.multimodal_reranker import MultimodalReranker
        multimodal_reranker = MultimodalReranker(config, searcher.encoder)
    except Exception as e:
        print(f"MultimodalReranker unavailable: {e}")

    modes = {"visual_only": None}
    if metadata_reranker:
        modes["metadata_reranked"] = metadata_reranker
    if multimodal_reranker:
        modes["context_reranked"] = multimodal_reranker

    all_results: dict[str, dict] = {}

    for mode_name, reranker in modes.items():
        print(f"\nEvaluating mode: {mode_name} ...")
        query_evals = []

        for q in tqdm(eval_queries, desc=mode_name):
            qid = q["query_image_id"]
            relevant = q["relevant"]

            if qid not in meta_index.index:
                continue
            query_path = meta_index.loc[qid].get("path")
            if not query_path or not Path(query_path).exists():
                continue

            try:
                raw = searcher.search(query_path, k=candidate_pool)
            except Exception as e:
                print(f"  Search failed for {qid}: {e}")
                continue

            if reranker is not None:
                query_meta = meta_index.loc[qid].to_dict() if qid in meta_index.index else {}
                reranked = reranker.rerank(raw, query_meta=query_meta, top_k=max(k_values))
            else:
                reranked = raw[:max(k_values)]

            retrieved_ids = [r["image_id"] for r in reranked]
            query_evals.append({"retrieved": retrieved_ids, "relevant": relevant})

        if query_evals:
            metrics = compute_metrics(query_evals, k_values=k_values)
            all_results[mode_name] = metrics
            print_metrics_table(metrics, label=mode_name)

    # Save results to CSV
    output_csv = results_dir / "ablation.csv"
    fieldnames = ["mode"] + sorted(list(next(iter(all_results.values())).keys()))
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode_name, metrics in all_results.items():
            writer.writerow({"mode": mode_name, **metrics})

    print(f"\nAblation results saved to {output_csv}")
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for mode_name, metrics in all_results.items():
        print_metrics_table(metrics, label=mode_name)


if __name__ == "__main__":
    cfg = load_config()
    run_evaluation(cfg)
