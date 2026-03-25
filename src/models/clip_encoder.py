"""CLIP image encoder wrapper for embedding generation and query encoding."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model '{model_name}' on {self.device} ...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("CLIP model loaded.")

    def _extract_features(self, raw) -> torch.Tensor:
        """Ensure we always get a plain tensor back from get_image/text_features."""
        if isinstance(raw, torch.Tensor):
            return raw
        # Newer transformers versions may return a dataclass/ModelOutput
        if hasattr(raw, "image_embeds"):
            return raw.image_embeds
        if hasattr(raw, "pooler_output"):
            return raw.pooler_output
        if hasattr(raw, "last_hidden_state"):
            return raw.last_hidden_state[:, 0]
        raise TypeError(f"Unexpected features type: {type(raw)}")

    @torch.no_grad()
    def encode_image(self, image_path: str | Path) -> np.ndarray:
        """Encode a single image. Returns a normalized (512,) float32 array."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        raw = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        features = F.normalize(self._extract_features(raw), dim=-1)
        return features.cpu().numpy().squeeze().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, image_paths: list[str | Path]) -> np.ndarray:
        """Encode a batch of images. Returns a normalized (N, 512) float32 array."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        raw = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        features = F.normalize(self._extract_features(raw), dim=-1)
        return features.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings. Returns a normalized (N, 512) float32 array."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        raw = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        features = F.normalize(self._extract_features(raw), dim=-1)
        return features.cpu().numpy().astype(np.float32)
