"""
Fetch Reddit posts and build a FAISS topic index from their embeddings.

Outputs:
    data/embeddings/topic_index.faiss
    data/embeddings/topic_records.json

Usage:
    python scripts/build_topic_index.py
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrichment.reddit import RedditTopicFetcher, build_topic_records
from src.enrichment.topic_index import TopicIndex
from src.models.clip_encoder import CLIPEncoder
from src.models.text_encoder import TextEncoder


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    subreddits = config["reddit"]["subreddits"]
    posts_per_sub = config["reddit"]["posts_per_subreddit"]

    print("Fetching Reddit posts ...")
    try:
        fetcher = RedditTopicFetcher()
        posts = fetcher.fetch_all_subreddits(subreddits, posts_per_subreddit=posts_per_sub)
    except EnvironmentError as e:
        print(f"Reddit API not configured: {e}")
        print("Using fallback static topic list instead.")
        posts = []

    if not posts:
        # Fallback: static topic list for testing without Reddit credentials
        from src.enrichment.reddit import TopicPost
        static_topics = [
            ("fashion", "Best summer outfits 2025 - streetwear edition"),
            ("fashion", "How to style wide-leg trousers for work"),
            ("streetwear", "New Adidas collection drops this week"),
            ("travel", "Hidden gems in Southeast Asia you need to visit"),
            ("travel", "Visiting Eiffel Tower - tips and photos"),
            ("travel", "Best cafes in Tokyo - a complete guide"),
            ("architecture", "Most stunning buildings in the world ranked"),
            ("food", "Street food tour of Istanbul"),
            ("sports", "Best sports events of 2025 so far"),
            ("malefashionadvice", "Business casual for tech office - what works"),
            ("femalefashionadvice", "Transitional outfits from summer to fall"),
        ]
        posts = [
            TopicPost(subreddit=sub, title=title, score=1000, url="")
            for sub, title in static_topics
        ]

    records = build_topic_records(posts)
    print(f"Building topic index from {len(records)} records ...")

    encoder = CLIPEncoder(
        model_name=config["model"]["clip_model"],
        device=config["model"]["device"],
    )
    text_encoder = TextEncoder(clip_encoder=encoder)

    topic_index = TopicIndex()
    topic_index.build(records, text_encoder)
    topic_index.save()

    print(f"\nTopic index built successfully with {topic_index.index.ntotal} topics.")


if __name__ == "__main__":
    main()
