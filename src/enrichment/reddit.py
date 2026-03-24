"""
Reddit topic fetcher using PRAW.

Collects recent post titles from configured subreddits to build
a topic index for trend-aware reranking.

Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env
"""

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class TopicPost:
    subreddit: str
    title: str
    score: int
    url: str


class RedditTopicFetcher:
    def __init__(self):
        import praw

        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "PinterestDemo/1.0")

        if not client_id or not client_secret:
            raise EnvironmentError(
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in .env. "
                "Create a Reddit app at: https://www.reddit.com/prefs/apps"
            )

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    def fetch_subreddit_posts(
        self,
        subreddit_name: str,
        limit: int = 100,
        sort: str = "hot",
    ) -> list[TopicPost]:
        """Fetch top posts from a subreddit."""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            if sort == "hot":
                posts = subreddit.hot(limit=limit)
            elif sort == "top":
                posts = subreddit.top(limit=limit, time_filter="week")
            else:
                posts = subreddit.new(limit=limit)

            results = []
            for post in posts:
                results.append(
                    TopicPost(
                        subreddit=subreddit_name,
                        title=post.title,
                        score=post.score,
                        url=post.url,
                    )
                )
            return results
        except Exception as e:
            print(f"  Error fetching r/{subreddit_name}: {e}")
            return []

    def fetch_all_subreddits(
        self,
        subreddits: list[str],
        posts_per_subreddit: int = 100,
    ) -> list[TopicPost]:
        """Fetch posts from multiple subreddits."""
        all_posts = []
        for sub in subreddits:
            print(f"  Fetching r/{sub} ...")
            posts = self.fetch_subreddit_posts(sub, limit=posts_per_subreddit)
            all_posts.extend(posts)
            time.sleep(1.0)  # rate limit courtesy
        print(f"Fetched {len(all_posts)} posts from {len(subreddits)} subreddits.")
        return all_posts


def build_topic_records(posts: list[TopicPost]) -> list[dict]:
    """Convert raw posts into topic records for embedding."""
    return [
        {
            "subreddit": p.subreddit,
            "title": p.title,
            "score": p.score,
            "text": f"[r/{p.subreddit}] {p.title}",
        }
        for p in posts
    ]
