"""
Extract place names from image captions and OCR text.

Uses simple NLP heuristics and a curated place name list
to identify candidate place names for GeoNames resolution.
"""

import re

# Common place/location indicator words
LOCATION_KEYWORDS = [
    "in", "at", "near", "by", "from", "around", "overlooking",
    "downtown", "city", "town", "village", "district",
]

# Regex: Capitalized word sequences (crude NER for proper nouns)
PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def extract_place_candidates(caption: str, ocr_text: str) -> list[str]:
    """
    Extract candidate place names from caption and OCR text.

    Returns a deduplicated list of candidate strings for GeoNames resolution.
    Priority order: OCR text first (more precise), then caption.
    """
    candidates: list[str] = []

    for text in [ocr_text, caption]:
        if not text or not isinstance(text, str):
            continue
        matches = PROPER_NOUN_RE.findall(text)
        candidates.extend(matches)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for c in candidates:
        c_clean = c.strip()
        if c_clean and c_clean not in seen and len(c_clean) > 2:
            seen.add(c_clean)
            unique.append(c_clean)

    return unique


def best_place_candidate(
    caption: str,
    ocr_text: str,
    landmark: str | None = None,
) -> str | None:
    """
    Return the single best place name candidate for GeoNames resolution.

    Priority:
    1. Landmark name if detected
    2. First proper noun from OCR
    3. First proper noun from caption
    """
    if landmark:
        return landmark

    candidates = extract_place_candidates(caption, ocr_text)
    return candidates[0] if candidates else None
