"""CLIP text encoder wrapper for encoding captions and topic text."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.clip_encoder import CLIPEncoder


class TextEncoder:
    """
    Thin wrapper around CLIPEncoder for text-only encoding tasks.
    Reuses the same underlying model to avoid loading it twice.
    """

    def __init__(self, clip_encoder: CLIPEncoder | None = None, **kwargs):
        if clip_encoder is not None:
            self.encoder = clip_encoder
        else:
            from src.models.clip_encoder import CLIPEncoder
            self.encoder = CLIPEncoder(**kwargs)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings. Returns (N, 512) normalized float32 array."""
        return self.encoder.encode_text(texts)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string. Returns (512,) normalized float32 array."""
        return self.encode([text])[0]
