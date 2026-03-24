"""
Build and query a FAISS index over Reddit/topic text embeddings.

Used at query time to match image caption/context to trending topics.
"""

import json
from pathlib import Path

import faiss
import numpy as np


TOPIC_INDEX_PATH = "data/embeddings/topic_index.faiss"
TOPIC_RECORDS_PATH = "data/embeddings/topic_records.json"


class TopicIndex:
    def __init__(self):
        self.index: faiss.Index | None = None
        self.records: list[dict] = []

    def build(self, records: list[dict], text_encoder) -> None:
        """
        Build a FAISS index from a list of topic records.

        Each record must have a 'text' field.
        text_encoder: TextEncoder instance
        """
        texts = [r["text"] for r in records]
        print(f"Encoding {len(texts)} topic texts ...")
        embeddings = text_encoder.encode(texts).astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.records = records
        print(f"Topic index built: {self.index.ntotal} topics.")

    def save(
        self,
        index_path: str = TOPIC_INDEX_PATH,
        records_path: str = TOPIC_RECORDS_PATH,
    ) -> None:
        if self.index is None:
            raise RuntimeError("Index not built.")
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(records_path, "w") as f:
            json.dump(self.records, f)
        print(f"Topic index saved to {index_path}")

    def load(
        self,
        index_path: str = TOPIC_INDEX_PATH,
        records_path: str = TOPIC_RECORDS_PATH,
    ) -> None:
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Topic index not found at {index_path}.")
        self.index = faiss.read_index(str(index_path))
        with open(records_path) as f:
            self.records = json.load(f)
        print(f"Topic index loaded: {self.index.ntotal} topics.")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[dict]:
        """
        Find top-k matching topics for a query text embedding.

        Returns list of dicts: {subreddit, title, text, score}
        """
        if self.index is None:
            raise RuntimeError("Topic index not loaded.")

        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = dict(self.records[idx])
            record["topic_score"] = float(score)
            results.append(record)

        return results

    def get_topic_labels(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[str]:
        """Return just the subreddit names of the top matching topics."""
        results = self.search(query_embedding, top_k=top_k)
        seen = set()
        labels = []
        for r in results:
            sub = r["subreddit"]
            if sub not in seen:
                seen.add(sub)
                labels.append(sub)
        return labels
