"""Clean raw dataset: filter corrupt/missing images, produce data/metadata/images.csv."""

from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_styles_csv(raw_dir: Path) -> Path | None:
    for candidate in raw_dir.rglob("styles.csv"):
        return candidate
    return None


def find_images_dir(raw_dir: Path) -> Path | None:
    candidate = raw_dir / "images"
    if candidate.exists():
        return candidate
    for d in raw_dir.rglob("images"):
        if d.is_dir():
            return d
    return None


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, Exception):
        return False


def clean_dataset(config: dict) -> pd.DataFrame:
    raw_dir = Path(config["paths"]["raw_images"])
    metadata_dir = Path(config["paths"]["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(config["paths"]["metadata_csv"])

    styles_csv = find_styles_csv(raw_dir)
    if styles_csv is None:
        raise FileNotFoundError(
            f"styles.csv not found under {raw_dir}. Run download.py first."
        )

    images_dir = find_images_dir(raw_dir)
    if images_dir is None:
        raise FileNotFoundError(
            f"images/ directory not found under {raw_dir}. Run download.py first."
        )

    print(f"Loading metadata from {styles_csv} ...")
    df = pd.read_csv(styles_csv, on_bad_lines="skip")

    column_map = {
        "id": "image_id",
        "masterCategory": "category",
        "subCategory": "sub_category",
        "articleType": "article_type",
        "baseColour": "color",
        "season": "season",
        "usage": "usage",
        "productDisplayName": "product_name",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    df["image_id"] = df["image_id"].astype(int)

    print(f"Checking {len(df):,} entries against image files ...")
    valid_rows = []
    skipped_missing = 0
    skipped_corrupt = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        image_path = images_dir / f"{row['image_id']}.jpg"
        if not image_path.exists():
            skipped_missing += 1
            continue
        if not is_valid_image(image_path):
            skipped_corrupt += 1
            continue
        row_dict = row.to_dict()
        row_dict["path"] = str(image_path)
        valid_rows.append(row_dict)

    clean_df = pd.DataFrame(valid_rows)

    keep_cols = [
        "image_id",
        "path",
        "category",
        "sub_category",
        "article_type",
        "color",
        "season",
        "usage",
        "product_name",
    ]
    keep_cols = [c for c in keep_cols if c in clean_df.columns]
    clean_df = clean_df[keep_cols].reset_index(drop=True)

    clean_df.to_csv(output_csv, index=False)

    print(f"\nCleaning summary:")
    print(f"  Total rows in CSV:     {len(df):,}")
    print(f"  Skipped (no image):    {skipped_missing:,}")
    print(f"  Skipped (corrupt):     {skipped_corrupt:,}")
    print(f"  Valid images saved:    {len(clean_df):,}")
    print(f"  Output CSV:            {output_csv}")

    return clean_df


if __name__ == "__main__":
    cfg = load_config()
    clean_dataset(cfg)
