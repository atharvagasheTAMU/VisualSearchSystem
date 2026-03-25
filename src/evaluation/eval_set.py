"""
Build and manage the evaluation set.

Relevance definition:
  - An image is relevant to a query if it shares the same `article_type`
    (e.g. "Tshirts", "Casual Shoes", "Watches") from the Kaggle fashion dataset.
  - article_type is much more specific than masterCategory and produces
    relevant sets of 50-300 items, making Recall/mAP/nDCG meaningful.
  - Relevant set is capped at MAX_RELEVANT to keep recall denominators tractable.

The eval set is stored as data/eval/eval_queries.csv with columns:
  query_image_id, relevant_image_ids (pipe-separated), article_type, n_relevant
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MAX_RELEVANT = 200  # cap relevant set size so recall denominators stay tractable


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_eval_set(
    config: dict,
    n_queries: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n_queries images and define relevant set for each using article_type.

    Saves to data/eval/eval_queries.csv.
    Returns the eval DataFrame.
    """
    metadata_csv = Path(config["paths"]["metadata_csv"])
    eval_dir = Path(config["paths"]["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / "eval_queries.csv"

    df = pd.read_csv(metadata_csv)
    df["image_id"] = df["image_id"].astype(int)

    # Use article_type as relevance (specific: "Tshirts", "Casual Shoes", etc.)
    # Fall back to sub_category, then category if article_type is absent
    if "article_type" in df.columns and df["article_type"].notna().sum() > 0:
        relevance_col = "article_type"
    elif "sub_category" in df.columns and df["sub_category"].notna().sum() > 0:
        relevance_col = "sub_category"
    else:
        relevance_col = "category"

    print(f"Using '{relevance_col}' as relevance column.")

    rng = np.random.default_rng(seed)
    groups = df[relevance_col].dropna().unique()

    # Only use groups with at least 5 items so there are meaningful relevant sets
    valid_groups = [g for g in groups if len(df[df[relevance_col] == g]) >= 5]
    queries = []

    per_group = max(1, n_queries // len(valid_groups))
    for grp in valid_groups:
        grp_df = df[df[relevance_col] == grp]
        sample_n = min(per_group, len(grp_df))
        sampled = grp_df.sample(n=sample_n, random_state=int(rng.integers(0, 1_000_000)))
        queries.append(sampled)

    query_df = pd.concat(queries).sample(
        n=min(n_queries, len(pd.concat(queries))), random_state=seed
    ).reset_index(drop=True)

    # Build relevance sets — cap at MAX_RELEVANT to keep recall denominators tractable
    grp_to_ids: dict[str, list[int]] = {}
    for grp in valid_groups:
        all_ids = df[df[relevance_col] == grp]["image_id"].tolist()
        grp_to_ids[grp] = all_ids

    rows = []
    for _, row in query_df.iterrows():
        qid = int(row["image_id"])
        grp = row.get(relevance_col)
        if grp and grp in grp_to_ids:
            candidates = [iid for iid in grp_to_ids[grp] if iid != qid]
            rng2 = np.random.default_rng(seed + qid)
            if len(candidates) > MAX_RELEVANT:
                candidates = rng2.choice(candidates, MAX_RELEVANT, replace=False).tolist()
        else:
            candidates = []
        rows.append({
            "query_image_id": qid,
            "category": row.get("category", ""),
            "article_type": grp,
            "relevant_image_ids": "|".join(str(i) for i in candidates),
            "n_relevant": len(candidates),
        })

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(output_path, index=False)
    print(f"Eval set saved: {len(eval_df)} queries, relevance='{relevance_col}' -> {output_path}")
    print(f"  Avg relevant per query: {eval_df['n_relevant'].mean():.1f}")
    print(f"  Min/Max relevant: {eval_df['n_relevant'].min()} / {eval_df['n_relevant'].max()}")
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
            "article_type": row.get("article_type"),
            "relevant": relevant,
        })
    return result


if __name__ == "__main__":
    cfg = load_config()
    build_eval_set(cfg)
