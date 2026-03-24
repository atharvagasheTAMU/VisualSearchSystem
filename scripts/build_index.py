"""Build and save FAISS index from saved embeddings."""

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.indexing.faiss_index import FaissIndex


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_index(config: dict) -> None:
    embeddings_path = Path(config["paths"]["embeddings_path"])
    index_path = Path(config["paths"]["index_path"])
    index_type = config["retrieval"]["index_type"]
    nlist = config["retrieval"]["ivf_nlist"]

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {embeddings_path}. Run build_embeddings.py first."
        )

    print(f"Loading embeddings from {embeddings_path} ...")
    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings: shape={embeddings.shape}")

    dim = embeddings.shape[1]
    fi = FaissIndex(dim=dim, index_type=index_type, nlist=nlist)
    fi.build(embeddings)
    fi.save(index_path)

    print(f"\nIndex build complete.")
    print(f"  Vectors indexed: {fi.index.ntotal:,}")
    print(f"  Index saved at:  {index_path}")


if __name__ == "__main__":
    cfg = load_config()
    build_index(cfg)
