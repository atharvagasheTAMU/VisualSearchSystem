"""
Batch-generate BLIP captions and EasyOCR text for all images.

Appends `caption` and `ocr_text` columns to data/metadata/images.csv.
Saves progress periodically to avoid restarting from scratch on failure.

Usage:
    python scripts/enrich_captions.py
"""

import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.captioner import ImageCaptioner
from src.models.ocr import OCRExtractor


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def enrich_captions(config: dict, batch_size: int = 16, save_every: int = 200) -> None:
    metadata_csv = Path(config["paths"]["metadata_csv"])
    device = config["model"]["device"]

    df = pd.read_csv(metadata_csv)
    print(f"Loaded {len(df):,} rows from {metadata_csv}")

    if "caption" not in df.columns:
        df["caption"] = ""
    if "ocr_text" not in df.columns:
        df["ocr_text"] = ""

    df["caption"] = df["caption"].fillna("")
    df["ocr_text"] = df["ocr_text"].fillna("")

    # Only process rows that haven't been captioned yet
    todo_mask = df["caption"] == ""
    todo_df = df[todo_mask]
    print(f"Images to caption: {len(todo_df):,} (skipping {(~todo_mask).sum():,} already done)")

    if todo_df.empty:
        print("All images already captioned.")
        return

    use_gpu = device == "cuda"
    captioner = ImageCaptioner(model_name=config["model"]["blip_model"], device=device)
    ocr = OCRExtractor(gpu=use_gpu)

    paths = todo_df["path"].tolist()
    indices = todo_df.index.tolist()

    for batch_start in tqdm(range(0, len(paths), batch_size), desc="Captioning"):
        batch_paths = paths[batch_start : batch_start + batch_size]
        batch_indices = indices[batch_start : batch_start + batch_size]

        # Caption
        valid_paths, valid_indices = [], []
        for p, idx in zip(batch_paths, batch_indices):
            if Path(p).exists():
                valid_paths.append(p)
                valid_indices.append(idx)

        if not valid_paths:
            continue

        try:
            captions = captioner.caption_batch(valid_paths)
        except Exception as e:
            print(f"  Batch caption failed: {e}. Falling back to single ...")
            captions = []
            for p in valid_paths:
                try:
                    captions.append(captioner.caption_image(p))
                except Exception:
                    captions.append("")

        for p, idx, cap in zip(valid_paths, valid_indices, captions):
            df.at[idx, "caption"] = cap
            try:
                ocr_text = ocr.extract_text(p)
            except Exception:
                ocr_text = ""
            df.at[idx, "ocr_text"] = ocr_text

        if (batch_start // batch_size + 1) % (save_every // batch_size) == 0:
            df.to_csv(metadata_csv, index=False)
            print(f"  Progress saved at {batch_start + batch_size} images.")

    df.to_csv(metadata_csv, index=False)
    print(f"\nEnrichment complete. Updated {metadata_csv}")
    print(f"Sample captions:")
    for _, row in df[df["caption"] != ""].head(5).iterrows():
        print(f"  [{row['image_id']}] {row['caption']}")


if __name__ == "__main__":
    cfg = load_config()
    enrich_captions(cfg)
