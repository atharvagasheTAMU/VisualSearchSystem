"""
Zero-shot landmark and scene detection using CLIP.

Uses a curated list of landmark/scene labels and scores them against
the query image using CLIP image-text similarity.
"""

from pathlib import Path

import numpy as np

LANDMARK_LABELS = [
    # World landmarks
    "Eiffel Tower", "Statue of Liberty", "Big Ben", "Colosseum",
    "Taj Mahal", "Great Wall of China", "Sydney Opera House",
    "Empire State Building", "Burj Khalifa", "Sagrada Familia",
    "Louvre Museum", "Acropolis of Athens", "Machu Picchu",
    "Christ the Redeemer", "Tower Bridge", "Golden Gate Bridge",
    "Niagara Falls", "Mount Fuji", "Stonehenge", "Vatican",
    # Scenes
    "beach", "mountain", "city skyline", "forest", "desert",
    "airport", "train station", "stadium", "shopping mall",
    "restaurant", "cafe", "park", "museum", "university campus",
    "concert hall", "festival grounds", "marketplace",
    # Fashion / product scenes
    "fashion runway", "clothing store", "outdoor street style",
    "studio photography", "product catalog",
]

SCENE_LABELS = [
    "indoor", "outdoor", "urban", "rural", "night", "day",
    "sunny", "rainy", "snowy", "crowded", "empty",
    "travel", "fashion", "food", "sports", "music", "art",
    "nature", "architecture", "street photography",
]


class LandmarkDetector:
    def __init__(self, clip_encoder, confidence_threshold: float = 0.20):
        self.encoder = clip_encoder
        self.threshold = confidence_threshold

        # Pre-compute text embeddings for all labels
        self._landmark_embeddings = self.encoder.encode_text(LANDMARK_LABELS)
        self._scene_embeddings = self.encoder.encode_text(SCENE_LABELS)

    def detect_landmark(self, image_path: str | Path) -> dict:
        """
        Detect the most likely landmark in an image.

        Returns:
            {
              "label": str or None,
              "confidence": float,
              "is_landmark": bool,
            }
        """
        img_emb = self.encoder.encode_image(image_path)  # (512,)
        scores = img_emb @ self._landmark_embeddings.T  # (N_labels,)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_label = LANDMARK_LABELS[best_idx]

        return {
            "label": best_label if best_score >= self.threshold else None,
            "confidence": best_score,
            "is_landmark": best_score >= self.threshold,
        }

    def detect_scene(self, image_path: str | Path) -> dict:
        """
        Detect the top scene tags for an image.

        Returns:
            {"tags": list[str], "scores": list[float]}
        """
        img_emb = self.encoder.encode_image(image_path)
        scores = img_emb @ self._scene_embeddings.T

        top_indices = np.argsort(scores)[::-1][:5]
        return {
            "tags": [SCENE_LABELS[i] for i in top_indices],
            "scores": [float(scores[i]) for i in top_indices],
        }

    def analyze(self, image_path: str | Path) -> dict:
        """Run full landmark + scene analysis on an image."""
        landmark = self.detect_landmark(image_path)
        scene = self.detect_scene(image_path)
        return {
            "landmark": landmark["label"],
            "landmark_confidence": landmark["confidence"],
            "scene_tags": scene["tags"],
            "scene_scores": scene["scores"],
        }
