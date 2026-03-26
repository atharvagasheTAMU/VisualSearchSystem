"""Clean raw datasets and produce a unified data/metadata/images.csv.

Supports multiple dataset formats:
  - fashion: Kaggle fashion-product-images-small (styles.csv + images/)
  - folder:  class-folder layout (category/image.jpg) used by landmarks, food, etc.

All loaders return a DataFrame with the same canonical columns:
  image_id, path, category, sub_category, article_type,
  color, season, usage, product_name
"""

from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import yaml


CANONICAL_COLS = [
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


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, Exception):
        return False


# ── Fashion dataset loader ────────────────────────────────────────────────────

def _find_styles_csv(raw_dir: Path) -> Path | None:
    for candidate in raw_dir.rglob("styles.csv"):
        return candidate
    return None


def _find_images_dir(raw_dir: Path) -> Path | None:
    candidate = raw_dir / "images"
    if candidate.exists():
        return candidate
    for d in raw_dir.rglob("images"):
        if d.is_dir():
            return d
    return None


def clean_fashion_dataset(raw_dir: Path) -> pd.DataFrame:
    """Load the Kaggle fashion-product-images dataset and return canonical DataFrame."""
    styles_csv = _find_styles_csv(raw_dir)
    if styles_csv is None:
        raise FileNotFoundError(f"styles.csv not found under {raw_dir}")

    images_dir = _find_images_dir(raw_dir)
    if images_dir is None:
        raise FileNotFoundError(f"images/ directory not found under {raw_dir}")

    print(f"  [fashion] Loading {styles_csv} ...")
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

    valid_rows = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  [fashion] Validating"):
        image_path = images_dir / f"{row['image_id']}.jpg"
        if not image_path.exists():
            skipped += 1
            continue
        if not is_valid_image(image_path):
            skipped += 1
            continue
        row_dict = row.to_dict()
        row_dict["path"] = str(image_path)
        valid_rows.append(row_dict)

    print(f"  [fashion] {len(valid_rows):,} valid  |  {skipped:,} skipped")
    return pd.DataFrame(valid_rows)


# ── Folder-based dataset loader ───────────────────────────────────────────────

def clean_folder_dataset(raw_dir: Path, category: str) -> pd.DataFrame:
    """Load a class-folder dataset of any nesting depth.

    Finds every image recursively via rglob; uses each image's immediate
    parent directory name as the class (article_type).  Works for:
      - <class>/<image>.jpg          (1 level, e.g. Wonders of the World)
      - root/images/<class>/<img>    (2 levels, e.g. Food-101)
      - root/root/<class>/<img>      (2 levels, e.g. Intel Image Classification)
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    all_images = [
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if not all_images:
        print(f"  [{category.lower()}] No images found under {raw_dir}")
        return pd.DataFrame()

    valid_rows = []
    skipped = 0

    for img_path in tqdm(all_images, desc=f"  [{category.lower()}] Validating"):
        if not is_valid_image(img_path):
            skipped += 1
            continue
        class_name = img_path.parent.name
        valid_rows.append({
            "path":         str(img_path),
            "category":     category,
            "article_type": class_name,
            "product_name": class_name.replace("_", " ").replace("-", " ").title(),
        })

    print(f"  [{category.lower()}] {len(valid_rows):,} valid  |  {skipped:,} skipped")
    return pd.DataFrame(valid_rows)


# ── Master multi-dataset cleaner ─────────────────────────────────────────────

def clean_all_datasets(config: dict) -> pd.DataFrame:
    """Clean all configured datasets and write a unified images.csv.

    Reads config["datasets"] list. Each entry needs:
      raw_dir  — path relative to project root (or absolute)
      loader   — "fashion" or "folder"
      category — top-level label for folder-based datasets

    Assigns sequential global image_ids (0, 1, 2, ...) across all datasets.
    """
    metadata_dir = Path(config["paths"]["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(config["paths"]["metadata_csv"])

    dataset_configs = config.get("datasets", [])
    if not dataset_configs:
        raise ValueError("No datasets defined in config['datasets'].")

    all_frames: list[pd.DataFrame] = []

    for ds in dataset_configs:
        name = ds["name"]
        raw_dir = Path(ds["raw_dir"])
        loader = ds.get("loader", "folder")
        category = ds.get("category", name.capitalize())

        if not raw_dir.is_absolute():
            # Resolve relative to the project root inferred from metadata_dir
            project_root = metadata_dir.parent.parent
            raw_dir = project_root / raw_dir

        if not raw_dir.exists():
            print(f"  [{name}] raw_dir not found: {raw_dir} — skipping")
            continue

        print(f"\nLoading dataset: {name} ({loader}) from {raw_dir}")

        try:
            if loader == "fashion":
                df = clean_fashion_dataset(raw_dir)
            elif loader == "folder":
                df = clean_folder_dataset(raw_dir, category)
            else:
                print(f"  [{name}] Unknown loader '{loader}' — skipping")
                continue
        except Exception as e:
            print(f"  [{name}] Failed: {e} — skipping")
            continue

        df["dataset"] = name
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No datasets were loaded successfully.")

    combined = pd.concat(all_frames, ignore_index=True)

    # Assign global sequential image_id (drop any per-dataset IDs)
    combined["image_id"] = range(len(combined))

    # Keep only canonical columns (plus dataset tag)
    keep = [c for c in CANONICAL_COLS if c in combined.columns] + ["dataset"]
    combined = combined[keep].reset_index(drop=True)

    combined.to_csv(output_csv, index=False)

    print(f"\nCombined dataset summary:")
    print(f"  Total images: {len(combined):,}")
    print(f"  Columns:      {list(combined.columns)}")
    print(f"  Per dataset:")
    for ds_name, group in combined.groupby("dataset"):
        print(f"    {ds_name}: {len(group):,} images  |  "
              f"{group['category'].nunique()} categories  |  "
              f"{group['article_type'].nunique() if 'article_type' in group else '?'} article types")
    print(f"  Saved to: {output_csv}")

    return combined


# ── Legacy single-dataset entry point (backward compat) ──────────────────────

def clean_dataset(config: dict) -> pd.DataFrame:
    """Backward-compatible wrapper — cleans just the fashion dataset.

    Uses config["paths"]["raw_images"] like before.
    Prefer clean_all_datasets() for multi-dataset setups.
    """
    raw_dir = Path(config["paths"]["raw_images"])
    metadata_dir = Path(config["paths"]["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(config["paths"]["metadata_csv"])

    df = clean_fashion_dataset(raw_dir)

    keep = [c for c in CANONICAL_COLS if c in df.columns]
    df = df[keep].reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    print(f"\nCleaning summary: {len(df):,} valid images -> {output_csv}")
    return df


if __name__ == "__main__":
    cfg = load_config()
    clean_all_datasets(cfg)
