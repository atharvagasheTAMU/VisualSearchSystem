"""
Enrich images that have detected landmarks/places with Wikidata and Wikipedia data.

Appends to data/metadata/images.csv:
  entity_type, entity_description, entity_tags, wikidata_id, wikipedia_url

Usage:
    python scripts/enrich_entities.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrichment.wikidata import enrich_entity
from src.enrichment.wikipedia import enrich_entity_wikipedia


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def enrich_entities(config: dict, save_every: int = 50) -> None:
    metadata_csv = Path(config["paths"]["metadata_csv"])
    df = pd.read_csv(metadata_csv)
    print(f"Loaded {len(df):,} rows.")

    for col in ["entity_type", "entity_description", "entity_tags", "wikidata_id", "wikipedia_url"]:
        if col not in df.columns:
            df[col] = None

    # Only enrich rows that have a detected landmark and haven't been enriched yet
    has_landmark = df["landmark"].notna() & (df["landmark"] != "") & (df["landmark"] != "None")
    not_enriched = df["entity_type"].isna() | (df["entity_type"] == "")
    todo_mask = has_landmark & not_enriched
    todo_df = df[todo_mask]

    print(f"Images with landmarks to enrich: {len(todo_df):,}")

    if todo_df.empty:
        print("No new entities to enrich.")
        return

    for i, (idx, row) in enumerate(tqdm(todo_df.iterrows(), total=len(todo_df), desc="Enriching")):
        entity_name = str(row.get("landmark", "") or "").strip()
        if not entity_name:
            continue

        # Try Wikidata first
        wiki_data = enrich_entity(entity_name)

        # Complement with Wikipedia
        wiki_pedia = enrich_entity_wikipedia(entity_name)

        entity_type = wiki_data.get("entity_type") or wiki_pedia.get("entity_type")
        entity_description = wiki_data.get("entity_description") or wiki_pedia.get("entity_description")
        entity_tags = list(set(
            (wiki_data.get("entity_tags") or []) +
            (wiki_pedia.get("entity_tags") or [])
        ))[:10]

        df.at[idx, "entity_type"] = entity_type
        df.at[idx, "entity_description"] = entity_description
        df.at[idx, "entity_tags"] = ",".join(entity_tags) if entity_tags else None
        df.at[idx, "wikidata_id"] = wiki_data.get("wikidata_id")
        df.at[idx, "wikipedia_url"] = wiki_pedia.get("wikipedia_url")

        # Be polite to APIs
        time.sleep(0.3)

        if (i + 1) % save_every == 0:
            df.to_csv(metadata_csv, index=False)
            print(f"  Progress saved at {i+1} entities.")

    df.to_csv(metadata_csv, index=False)
    print(f"\nEntity enrichment complete. Updated {metadata_csv}")


if __name__ == "__main__":
    cfg = load_config()
    enrich_entities(cfg)
