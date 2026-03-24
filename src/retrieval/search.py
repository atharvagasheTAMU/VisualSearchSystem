"""
Core search module.

Usage from terminal:
    python src/retrieval/search.py --image data/raw/images/1234.jpg --k 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.indexing.faiss_index import FaissIndex
from src.models.clip_encoder import CLIPEncoder


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class Searcher:
    def __init__(self, config: dict):
        self.config = config
        self.encoder = CLIPEncoder(
            model_name=config["model"]["clip_model"],
            device=config["model"]["device"],
        )
        self.index = FaissIndex()
        self.index.load(config["paths"]["index_path"])

        with open(config["paths"]["id_map_path"]) as f:
            raw = json.load(f)
        # id_map: str(faiss_index) -> image_id
        self.id_map: dict[int, int] = {int(k): int(v) for k, v in raw.items()}

        self.metadata = pd.read_csv(config["paths"]["metadata_csv"])
        self.metadata["image_id"] = self.metadata["image_id"].astype(int)
        self.meta_index = self.metadata.set_index("image_id")

    def search(
        self, image_path: str | Path, k: int | None = None
    ) -> list[dict]:
        """
        Search for top-k similar images.

        Returns a list of dicts with keys:
            image_id, score, path, category, color, product_name, ...
        """
        if k is None:
            k = self.config["retrieval"]["top_k"]

        embedding = self.encoder.encode_image(image_path)
        scores, faiss_indices = self.index.search(embedding, k=k)

        results = []
        for score, fi in zip(scores, faiss_indices):
            if fi < 0:
                continue
            image_id = self.id_map.get(fi)
            if image_id is None:
                continue
            row = {"image_id": image_id, "score": float(score)}
            if image_id in self.meta_index.index:
                meta = self.meta_index.loc[image_id].to_dict()
                row.update(meta)
            results.append(row)

        return results

    def search_by_embedding(
        self, embedding: np.ndarray, k: int | None = None
    ) -> list[dict]:
        """Search using a pre-computed embedding (already normalized)."""
        if k is None:
            k = self.config["retrieval"]["top_k"]

        scores, faiss_indices = self.index.search(embedding, k=k)

        results = []
        for score, fi in zip(scores, faiss_indices):
            if fi < 0:
                continue
            image_id = self.id_map.get(fi)
            if image_id is None:
                continue
            row = {"image_id": image_id, "score": float(score)}
            if image_id in self.meta_index.index:
                meta = self.meta_index.loc[image_id].to_dict()
                row.update(meta)
            results.append(row)

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual similarity search")
    parser.add_argument("--image", required=True, help="Path to query image")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    searcher = Searcher(config)
    results = searcher.search(args.image, k=args.k)

    print(f"\nTop {args.k} results for: {args.image}\n")
    print(f"{'Rank':<6} {'Score':<8} {'Image ID':<12} {'Category':<20} {'Product'}")
    print("-" * 70)
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:<6} {r['score']:<8.4f} {r['image_id']:<12} "
            f"{str(r.get('category', 'N/A')):<20} {r.get('product_name', 'N/A')}"
        )


if __name__ == "__main__":
    main()
