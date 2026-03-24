"""
Enrich all images with landmark detection and GeoNames location resolution.

Appends columns to data/metadata/images.csv:
  landmark, landmark_confidence, city, country, country_code, region, lat, lng

Usage:
    python scripts/enrich_locations.py
"""

import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrichment.geonames import GeoNamesResolver, resolve_place_safe
from src.enrichment.place_extractor import best_place_candidate
from src.models.clip_encoder import CLIPEncoder
from src.models.landmark_detector import LandmarkDetector


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def enrich_locations(config: dict, save_every: int = 100) -> None:
    metadata_csv = Path(config["paths"]["metadata_csv"])
    df = pd.read_csv(metadata_csv)
    print(f"Loaded {len(df):,} rows.")

    for col in ["landmark", "landmark_confidence", "city", "country", "country_code", "region", "lat", "lng"]:
        if col not in df.columns:
            df[col] = None

    todo_mask = df["landmark"].isna() | (df["landmark"] == "")
    todo_df = df[todo_mask]
    print(f"Images to process: {len(todo_df):,}")

    if todo_df.empty:
        print("All images already enriched with location data.")
        return

    encoder = CLIPEncoder(
        model_name=config["model"]["clip_model"],
        device=config["model"]["device"],
    )
    detector = LandmarkDetector(encoder)

    try:
        resolver = GeoNamesResolver()
        use_geonames = True
    except EnvironmentError as e:
        print(f"GeoNames disabled: {e}")
        use_geonames = False
        resolver = None

    for i, (idx, row) in enumerate(tqdm(todo_df.iterrows(), total=len(todo_df), desc="Landmarks")):
        path = row.get("path")
        if not path or not Path(path).exists():
            continue

        try:
            analysis = detector.analyze(path)
            df.at[idx, "landmark"] = analysis["landmark"] or ""
            df.at[idx, "landmark_confidence"] = round(analysis["landmark_confidence"], 4)
        except Exception as e:
            df.at[idx, "landmark"] = ""
            df.at[idx, "landmark_confidence"] = 0.0

        if use_geonames and resolver:
            caption = str(row.get("caption", "") or "")
            ocr_text = str(row.get("ocr_text", "") or "")
            landmark = df.at[idx, "landmark"]
            place_candidate = best_place_candidate(caption, ocr_text, landmark or None)

            if place_candidate:
                geo = resolve_place_safe(resolver, place_candidate)
                df.at[idx, "city"] = geo.get("city")
                df.at[idx, "country"] = geo.get("country")
                df.at[idx, "country_code"] = geo.get("country_code")
                df.at[idx, "region"] = geo.get("region")
                df.at[idx, "lat"] = geo.get("lat")
                df.at[idx, "lng"] = geo.get("lng")

        if (i + 1) % save_every == 0:
            df.to_csv(metadata_csv, index=False)
            print(f"  Progress saved at {i+1} images.")

    df.to_csv(metadata_csv, index=False)
    print(f"\nLocation enrichment complete. Updated {metadata_csv}")


if __name__ == "__main__":
    cfg = load_config()
    enrich_locations(cfg)
