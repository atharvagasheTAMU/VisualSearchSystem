"""BLIP image captioning wrapper."""

from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class ImageCaptioner:
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "auto",
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading BLIP captioning model '{model_name}' on {self.device} ...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("BLIP model loaded.")

    @torch.no_grad()
    def caption_image(self, image_path: str | Path) -> str:
        """Generate a caption for a single image."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()

    @torch.no_grad()
    def caption_batch(self, image_paths: list[str | Path]) -> list[str]:
        """Generate captions for a batch of images."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=50)
        captions = [
            self.processor.decode(o, skip_special_tokens=True).strip() for o in out
        ]
        return captions
