"""FAISS index builder, saver, and loader."""

from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, dim: int = 512, index_type: str = "flat", nlist: int = 100):
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.index: faiss.Index | None = None

    def build(self, embeddings: np.ndarray) -> None:
        """Build a FAISS index from a (N, D) float32 embedding matrix."""
        assert embeddings.dtype == np.float32, "Embeddings must be float32"
        n, d = embeddings.shape
        assert d == self.dim, f"Embedding dim mismatch: expected {self.dim}, got {d}"

        if self.index_type == "flat":
            # Exact inner-product search (cosine on normalized vectors)
            self.index = faiss.IndexFlatIP(d)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
            print(f"Training IVF index with nlist={self.nlist} on {n} vectors ...")
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        self.index.add(embeddings)
        print(f"FAISS index built: type={self.index_type}, vectors={self.index.ntotal:,}, dim={d}")

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Index is not built yet. Call build() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        print(f"Index saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        self.index = faiss.read_index(str(path))
        print(f"Index loaded from {path} ({self.index.ntotal:,} vectors)")

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Search top-k nearest neighbors.

        Args:
            query: (D,) or (1, D) float32 normalized vector
            k:     number of results

        Returns:
            scores: (k,) float32
            indices: (k,) int64
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call build() or load() first.")
        query = query.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, k)
        return scores[0], indices[0]
