"""EasyOCR wrapper for extracting visible text from images."""

from pathlib import Path


class OCRExtractor:
    def __init__(self, languages: list[str] | None = None, gpu: bool = False):
        import easyocr

        self.languages = languages or ["en"]
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

    def extract_text(self, image_path: str | Path) -> str:
        """
        Extract visible text from an image.

        Returns a single string of all detected text, space-joined.
        Returns empty string if no text detected.
        """
        try:
            results = self.reader.readtext(str(image_path), detail=0, paragraph=True)
            return " ".join(results).strip()
        except Exception:
            return ""

    def extract_text_with_confidence(
        self, image_path: str | Path, min_confidence: float = 0.4
    ) -> list[dict]:
        """
        Extract text with bounding boxes and confidence scores.

        Returns list of {text, confidence, bbox}.
        """
        try:
            raw = self.reader.readtext(str(image_path))
            results = []
            for bbox, text, conf in raw:
                if conf >= min_confidence:
                    results.append({"text": text, "confidence": conf, "bbox": bbox})
            return results
        except Exception:
            return []
