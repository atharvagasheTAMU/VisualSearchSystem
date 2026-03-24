"""
Build and manage the evaluation set.

Relevance definition (Phase 1 baseline):
  - An image is relevant to a query if it shares the same `category`
    (masterCategory in the Kaggle fashion dataset).

The eval set is stored as data/eval/eval_queries.csv with columns:
  query_image_id, relevant_image_ids (pipe-separated)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_eval_set(
    config: dict,
    n_queries: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n_queries images and define relevant set for each.

    Saves to data/eval/eval_queries.csv.
    Returns the eval DataFrame.
    """
    metadata_csv = Path(config["paths"]["metadata_csv"])
    eval_dir = Path(config["paths"]["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / "eval_queries.csv"

    df = pd.read_csv(metadata_csv)
    df["image_id"] = df["image_id"].astype(int)

    # Sample query images, stratified by category
    rng = np.random.default_rng(seed)
    categories = df["category"].dropna().unique()
    queries = []

    per_cat = max(1, n_queries // len(categories))
    for cat in categories:
        cat_df = df[df["category"] == cat]
        sample_n = min(per_cat, len(cat_df))
        sampled = cat_df.sample(n=sample_n, random_state=int(rng.integers(0, 1_000_000)))
        queries.append(sampled)

    query_df = pd.concat(queries).sample(n=min(n_queries, len(pd.concat(queries))), random_state=seed).reset_index(drop=True)

    # Build relevance sets
    cat_to_ids: dict[str, list[int]] = {}
    for cat in categories:
        cat_to_ids[cat] = df[df["category"] == cat]["image_id"].tolist()

    rows = []
    for _, row in query_df.iterrows():
        qid = int(row["image_id"])
        cat = row.get("category")
        if cat and cat in cat_to_ids:
            relevant = [iid for iid in cat_to_ids[cat] if iid != qid]
        else:
            relevant = []
        rows.append({
            "query_image_id": qid,
            "category": cat,
            "relevant_image_ids": "|".join(str(i) for i in relevant),
            "n_relevant": len(relevant),
        })

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(output_path, index=False)
    print(f"Eval set saved: {len(eval_df)} queries -> {output_path}")
    return eval_df


def load_eval_set(config: dict) -> list[dict]:
    """
    Load the eval set and return a list of dicts:
      {"query_image_id": int, "relevant": set[int], "category": str}
    """
    eval_path = Path(config["paths"]["eval_dir"]) / "eval_queries.csv"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Eval set not found at {eval_path}. Run build_eval_set() first."
        )

    df = pd.read_csv(eval_path)
    result = []
    for _, row in df.iterrows():
        rel_ids = row["relevant_image_ids"]
        if pd.isna(rel_ids) or rel_ids == "":
            relevant = set()
        else:
            relevant = {int(x) for x in str(rel_ids).split("|") if x}
        result.append({
            "query_image_id": int(row["query_image_id"]),
            "category": row.get("category"),
            "relevant": relevant,
        })
    return result


if __name__ == "__main__":
    cfg = load_config()
    build_eval_set(cfg)
