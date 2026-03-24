"""
Wikipedia summary enrichment for landmarks and places.

Falls back gracefully if wikipedia package is unavailable or entity not found.
"""

import time


def get_wikipedia_summary(entity_name: str, sentences: int = 2) -> dict:
    """
    Fetch a Wikipedia summary for an entity.

    Returns:
        {"summary": str, "url": str, "categories": list[str]}
    Returns empty dict on any failure.
    """
    if not entity_name or not isinstance(entity_name, str):
        return {}

    try:
        import wikipedia

        wikipedia.set_lang("en")
        try:
            page = wikipedia.page(entity_name, auto_suggest=True, redirect=True)
            summary = wikipedia.summary(entity_name, sentences=sentences, auto_suggest=True)
            categories = page.categories[:10] if hasattr(page, "categories") else []
            return {
                "summary": summary,
                "url": page.url,
                "categories": categories,
            }
        except wikipedia.exceptions.DisambiguationError as e:
            # Try first option
            if e.options:
                try:
                    page = wikipedia.page(e.options[0])
                    summary = wikipedia.summary(e.options[0], sentences=sentences)
                    return {"summary": summary, "url": page.url, "categories": []}
                except Exception:
                    return {}
        except wikipedia.exceptions.PageError:
            return {}
        except Exception:
            return {}

    except ImportError:
        return {}


def enrich_entity_wikipedia(entity_name: str) -> dict:
    """
    Combined enrichment: Wikipedia summary + tag extraction.

    Returns dict with:
        entity_description (from Wikipedia summary)
        entity_tags (from Wikipedia categories, cleaned)
        wikipedia_url
    """
    result = get_wikipedia_summary(entity_name)
    if not result:
        return {}

    raw_categories = result.get("categories", [])
    tags = []
    skip_prefixes = ("articles", "wikipedia", "cs1", "pages", "all ", "use ")
    for cat in raw_categories:
        cat_lower = cat.lower()
        if not any(cat_lower.startswith(p) for p in skip_prefixes):
            tags.append(cat.replace("_", " ").strip())

    return {
        "entity_description": result.get("summary"),
        "entity_tags": tags[:8],
        "wikipedia_url": result.get("url"),
    }
