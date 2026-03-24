"""Inspect the raw dataset: counts, schema, sample rows, missing values."""

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_styles_csv(raw_dir: Path) -> Path | None:
    for candidate in raw_dir.rglob("styles.csv"):
        return candidate
    return None


def inspect_dataset(config: dict) -> None:
    raw_dir = Path(config["paths"]["raw_images"])
    styles_csv = find_styles_csv(raw_dir)

    if styles_csv is None:
        print(
            "styles.csv not found. Have you run download.py?\n"
            f"Expected somewhere under: {raw_dir}"
        )
        return

    print(f"Found styles.csv at: {styles_csv}")
    df = pd.read_csv(styles_csv, on_bad_lines="skip")

    print(f"\n{'='*60}")
    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"{'='*60}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")

    print("\nDtype summary:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  None")
    else:
        print(missing.to_string())

    print("\nCategory distribution (top 10):")
    if "masterCategory" in df.columns:
        print(df["masterCategory"].value_counts().head(10).to_string())

    print("\nSubcategory distribution (top 10):")
    if "subCategory" in df.columns:
        print(df["subCategory"].value_counts().head(10).to_string())

    print("\nSample rows:")
    print(df.head(5).to_string())

    images_dir = raw_dir / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        print(f"\nImage files found: {len(image_files):,}")
        ids_with_images = {int(f.stem) for f in image_files if f.stem.isdigit()}
        ids_in_csv = set(df["id"].dropna().astype(int))
        missing_images = ids_in_csv - ids_with_images
        extra_images = ids_with_images - ids_in_csv
        print(f"IDs in CSV with matching image: {len(ids_in_csv & ids_with_images):,}")
        print(f"IDs in CSV without image: {len(missing_images):,}")
        print(f"Image files without CSV row: {len(extra_images):,}")
    else:
        print(f"\nImages directory not found at {images_dir}")

    print(f"\n{'='*60}")
    print("Inspection complete.")


if __name__ == "__main__":
    cfg = load_config()
    inspect_dataset(cfg)
