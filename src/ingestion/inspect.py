"""Inspect the combined dataset: counts, schema, per-dataset breakdown."""

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def inspect_dataset(config: dict) -> None:
    metadata_csv = Path(config["paths"]["metadata_csv"])

    if not metadata_csv.exists():
        print(
            f"images.csv not found at {metadata_csv}.\n"
            "Run clean_all_datasets() first (Day 1 notebook, Step 7)."
        )
        return

    df = pd.read_csv(metadata_csv)

    print(f"\n{'='*60}")
    print(f"Combined dataset: {len(df):,} rows  x  {df.shape[1]} columns")
    print(f"{'='*60}")
    print(f"Columns: {list(df.columns)}")

    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  None")
    else:
        print(missing.to_string())

    if "dataset" in df.columns:
        print(f"\nPer-dataset breakdown:")
        for ds_name, grp in df.groupby("dataset"):
            cats = grp["category"].nunique() if "category" in grp else "?"
            art_types = grp["article_type"].nunique() if "article_type" in grp else "?"
            print(f"  {ds_name:14} {len(grp):>7,} images  |  "
                  f"{cats} categories  |  {art_types} article types")

    print(f"\nCategory distribution (top 15):")
    if "category" in df.columns:
        print(df["category"].value_counts().head(15).to_string())

    print(f"\nArticle type distribution (top 20):")
    if "article_type" in df.columns:
        print(df["article_type"].value_counts().head(20).to_string())

    print(f"\nSample rows:")
    print(df.sample(min(5, len(df))).to_string())

    print(f"\n{'='*60}")
    print("Inspection complete.")


if __name__ == "__main__":
    cfg = load_config()
    inspect_dataset(cfg)
