"""
Build CLIP embeddings for all images in images.csv.

Outputs:
  data/embeddings/embeddings.npy  - shape (N, 512), float32, L2-normalized
  data/embeddings/id_map.json     - {str(index): image_id}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.clip_encoder import CLIPEncoder


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_embeddings(config: dict) -> None:
    metadata_csv = Path(config["paths"]["metadata_csv"])
    embeddings_dir = Path(config["paths"]["embeddings_dir"])
    embeddings_path = Path(config["paths"]["embeddings_path"])
    id_map_path = Path(config["paths"]["id_map_path"])
    batch_size = config["model"]["batch_size"]
    model_name = config["model"]["clip_model"]
    device = config["model"]["device"]

    embeddings_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    print(f"Loaded {len(df):,} images from {metadata_csv}")

    encoder = CLIPEncoder(model_name=model_name, device=device)

    all_embeddings = []
    id_map = {}
    failed = []
    global_idx = 0  # running count of successfully encoded images

    paths = df["path"].tolist()
    image_ids = df["image_id"].tolist()

    for batch_start in tqdm(range(0, len(paths), batch_size), desc="Encoding batches"):
        batch_paths = paths[batch_start : batch_start + batch_size]
        batch_ids = image_ids[batch_start : batch_start + batch_size]

        valid_paths, valid_ids = [], []
        for p, img_id in zip(batch_paths, batch_ids):
            if Path(p).exists():
                valid_paths.append(p)
                valid_ids.append(img_id)
            else:
                failed.append(img_id)

        if not valid_paths:
            continue

        try:
            embeddings = encoder.encode_batch(valid_paths)
        except Exception as e:
            print(f"  Batch failed ({e}), falling back to single-image encoding ...")
            embeddings = []
            for p, img_id in zip(valid_paths, list(valid_ids)):
                try:
                    emb = encoder.encode_image(p)
                    embeddings.append(emb)
                except Exception as e2:
                    print(f"    Skipping image {img_id}: {e2}")
                    failed.append(img_id)
                    valid_ids = [i for i in valid_ids if i != img_id]
            if not embeddings:
                continue
            embeddings = np.stack(embeddings)

        for i, img_id in enumerate(valid_ids):
            id_map[str(global_idx + i)] = int(img_id)

        global_idx += len(valid_ids)
        all_embeddings.append(embeddings)

    final_embeddings = np.vstack(all_embeddings).astype(np.float32)
    np.save(embeddings_path, final_embeddings)

    with open(id_map_path, "w") as f:
        json.dump(id_map, f)

    print(f"\nEmbedding generation complete.")
    print(f"  Shape:          {final_embeddings.shape}")
    print(f"  Saved to:       {embeddings_path}")
    print(f"  ID map saved:   {id_map_path}")
    if failed:
        print(f"  Failed images:  {len(failed)}")


if __name__ == "__main__":
    cfg = load_config()
    build_embeddings(cfg)
