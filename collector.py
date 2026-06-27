# ============================================================
# collector.py — Data collection from all 4 sources
# ============================================================
# Sources:
#   1. NewsAPI        — mainstream tech headlines
#   2. HackerNews     — developer community top stories
#   3. Dev.to         — developer-written articles
#   4. Google News RSS — broad publisher coverage
#
# Run directly to test:
#   python collector.py

import hashlib
import os
from datetime import datetime, timezone, timedelta

import feedparser
import pandas as pd
import requests

from config import (
    DATA_DIR,
    DEVTO_API_URL,
    DEVTO_ARTICLES_PER_TAG,
    DEVTO_TAGS,
    HN_TOP_STORIES_COUNT,
    MAX_ARTICLES_PER_SOURCE,
    NEWS_API_KEY,
    NEWSAPI_QUERY,
    RAW_ARTICLE_COLUMNS,
    RAW_ARTICLES_PATH,
    RSS_QUERY,
    SECTOR_KEYWORDS,
    DEFAULT_SECTOR,
    ensure_data_dir,
)


# ============================================================
# Utility helpers
# ============================================================

def make_article_id(url: str) -> str:
    """Generate a unique ID for an article by hashing its URL.
    This is how we deduplicate — same URL always gives same ID."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def get_current_time() -> str:
    """Return current UTC time as a string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def detect_sector(text: str) -> str:
    """
    Tag an article with a tech sector based on keyword matching.
    Checks title + description text against SECTOR_KEYWORDS in config.
    Returns the first matching sector, or DEFAULT_SECTOR if none match.
    """
    text_lower = text.lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return sector
    return DEFAULT_SECTOR


def clean_text(text: str) -> str:
    """Basic text cleaning — strip whitespace, handle None."""
    if not text or not isinstance(text, str):
        return ""
    return text.strip()


def make_article(title, description, content, url, source,
                 published_at, data_source) -> dict:
    """
    Build a clean article dict with all required columns.
    Detects sector automatically from title + description.
    """
    title       = clean_text(title)
    description = clean_text(description)
    content     = clean_text(content)
    url         = clean_text(url)

    sector_text = f"{title} {description}"
    sector      = detect_sector(sector_text)

    return {
        "article_id":  make_article_id(url),
        "title":       title,
        "description": description,
        "content":     content,
        "url":         url,
        "source":      clean_text(source),
        "published_at": clean_text(published_at),
        "collected_at": get_current_time(),
        "sector":      sector,
        "data_source": data_source,
    }


# ============================================================
# Source 1 — NewsAPI
# ============================================================

def fetch_newsapi() -> list[dict]:
    """
    Fetch tech headlines from NewsAPI.
    Uses the everything endpoint with our tech query.
    Free tier returns title + description (no full content — that's fine).
    """
    print("  📰 Fetching from NewsAPI...")
    articles = []

    if not NEWS_API_KEY:
        print("  ⚠️  NEWS_API_KEY not set — skipping NewsAPI")
        return articles

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q":        NEWSAPI_QUERY,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 100,                  # max allowed — cast a wider net
            "apiKey":   NEWS_API_KEY,
            "from":     (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            print(f"  ❌ NewsAPI error: {data.get('message', 'unknown error')}")
            return articles

        raw_articles = data.get("articles", [])

        for item in raw_articles:
            # Skip articles with no title or URL
            if not item.get("title") or not item.get("url"):
                continue
            # Skip removed articles
            if item["title"] == "[Removed]":
                continue

            article = make_article(
                title       = item.get("title", ""),
                description = item.get("description", ""),
                content     = item.get("description", ""),  # free tier: use description as content
                url         = item.get("url", ""),
                source      = item.get("source", {}).get("name", "NewsAPI"),
                published_at = item.get("publishedAt", ""),
                data_source = "newsapi",
            )
            articles.append(article)

        print(f"  ✅ NewsAPI: {len(articles)} articles fetched")

    except requests.exceptions.Timeout:
        print("  ❌ NewsAPI: request timed out")
    except requests.exceptions.RequestException as e:
        print(f"  ❌ NewsAPI request error: {e}")
    except Exception as e:
        print(f"  ❌ NewsAPI unexpected error: {e}")

    return articles


# ============================================================
# Source 2 — HackerNews
# ============================================================

def fetch_hackernews() -> list[dict]:
    """
    Fetch top stories from HackerNews.
    Uses the official Firebase API — free, no key needed.
    Pulls top story IDs then fetches each story's details.
    """
    print("  🟠 Fetching from HackerNews...")
    articles = []

    try:
        # Step 1: Get top story IDs
        ids_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        response = requests.get(ids_url, timeout=10)
        response.raise_for_status()
        story_ids = response.json()[:100]   # fetch top 100, dedupe handles the rest

        # Step 2: Fetch each story's details
        fetched = 0
        for story_id in story_ids:
            if fetched >= MAX_ARTICLES_PER_SOURCE:
                break
            try:
                story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                story_response = requests.get(story_url, timeout=5)
                story_response.raise_for_status()
                story = story_response.json()

                # Skip non-story types (Ask HN, polls etc) and stories without URLs
                if not story or story.get("type") != "story":
                    continue
                if not story.get("url") or not story.get("title"):
                    continue

                # Build HN story URL as fallback
                hn_url = story.get("url", f"https://news.ycombinator.com/item?id={story_id}")

                # Use title as both title and description (HN stories have no description)
                title = story.get("title", "")
                score = story.get("score", 0)
                description = f"HackerNews story with {score} points and {story.get('descendants', 0)} comments."

                # Convert Unix timestamp to datetime string
                published_at = ""
                if story.get("time"):
                    published_at = datetime.fromtimestamp(
                        story["time"], tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S")

                article = make_article(
                    title        = title,
                    description  = description,
                    content      = title,   # HN: title is the main signal
                    url          = hn_url,
                    source       = "HackerNews",
                    published_at = published_at,
                    data_source  = "hackernews",
                )
                articles.append(article)
                fetched += 1

            except Exception:
                # Skip individual story fetch errors silently
                continue

        print(f"  ✅ HackerNews: {len(articles)} articles fetched")

    except requests.exceptions.Timeout:
        print("  ❌ HackerNews: request timed out")
    except requests.exceptions.RequestException as e:
        print(f"  ❌ HackerNews request error: {e}")
    except Exception as e:
        print(f"  ❌ HackerNews unexpected error: {e}")

    return articles


# ============================================================
# Source 3 — Dev.to
# ============================================================

def fetch_devto() -> list[dict]:
    """
    Fetch developer-written articles from Dev.to API.
    Free, no key needed. Fetches by tag for each tech sector tag.
    """
    print("  💻 Fetching from Dev.to...")
    articles = []
    seen_ids = set()  # Avoid duplicates across tags

    try:
        for tag in DEVTO_TAGS:
            try:
                response = requests.get(
                    DEVTO_API_URL,
                    params={
                        "tag":      tag,
                        "per_page": DEVTO_ARTICLES_PER_TAG,
                        "state":    "fresh",  # recent articles only
                    },
                    timeout=10,
                )
                response.raise_for_status()
                items = response.json()

                for item in items:
                    article_id = item.get("id")
                    if article_id in seen_ids:
                        continue
                    seen_ids.add(article_id)

                    url = item.get("url", "")
                    if not url:
                        continue

                    title       = item.get("title", "")
                    description = item.get("description", "") or title
                    published_at = item.get("published_at", "")

                    # Clean up Dev.to timestamp format
                    if published_at:
                        try:
                            published_at = datetime.fromisoformat(
                                published_at.replace("Z", "+00:00")
                            ).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            pass

                    article = make_article(
                        title        = title,
                        description  = description,
                        content      = description,
                        url          = url,
                        source       = item.get("user", {}).get("name", "Dev.to"),
                        published_at = published_at,
                        data_source  = "devto",
                    )
                    articles.append(article)

                    if len(articles) >= MAX_ARTICLES_PER_SOURCE:
                        break

            except Exception as e:
                print(f"  ⚠️  Dev.to tag '{tag}' error: {e}")
                continue

            if len(articles) >= MAX_ARTICLES_PER_SOURCE:
                break

        print(f"  ✅ Dev.to: {len(articles)} articles fetched")

    except Exception as e:
        print(f"  ❌ Dev.to unexpected error: {e}")

    return articles


# ============================================================
# Source 4 — Google News RSS
# ============================================================

def fetch_google_news_rss() -> list[dict]:
    """
    Fetch articles from Google News RSS feed.
    Free, no key needed. Uses feedparser to parse RSS.
    Pulls from multiple topic URLs for better coverage.
    """
    print("  🔵 Fetching from Google News RSS...")
    articles = []
    seen_urls = set()

    # Multiple RSS feeds for broader tech coverage
    rss_urls = [
        f"https://news.google.com/rss/search?q={RSS_QUERY}&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en",  # Technology topic
    ]

    for rss_url in rss_urls:
        try:
            feed = feedparser.parse(rss_url)

            for entry in feed.entries:
                url = entry.get("link", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                title       = entry.get("title", "")
                description = entry.get("summary", "") or title

                # Clean Google News title format — remove source suffix
                # e.g. "AI news - TechCrunch" → "AI news"
                if " - " in title:
                    title = title.rsplit(" - ", 1)[0].strip()

                # Parse published date
                published_at = ""
                if entry.get("published"):
                    try:
                        from email.utils import parsedate_to_datetime
                        published_at = parsedate_to_datetime(
                            entry["published"]
                        ).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        published_at = entry.get("published", "")

                # Get source name from Google News feed tags
                source = "Google News"
                if entry.get("source"):
                    source = entry.source.get("title", "Google News")

                article = make_article(
                    title        = title,
                    description  = description,
                    content      = description,
                    url          = url,
                    source       = source,
                    published_at = published_at,
                    data_source  = "googlenews",
                )
                articles.append(article)

                if len(articles) >= MAX_ARTICLES_PER_SOURCE:
                    break

        except Exception as e:
            print(f"  ⚠️  Google News RSS error: {e}")
            continue

        if len(articles) >= MAX_ARTICLES_PER_SOURCE:
            break

    print(f"  ✅ Google News RSS: {len(articles)} articles fetched")
    return articles


# ============================================================
# Deduplication
# ============================================================

def deduplicate(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove articles that already exist in the CSV.
    Comparison is done on article_id (MD5 hash of URL).
    Returns only the new articles that are not already saved.
    """
    if existing_df.empty:
        return new_df

    existing_ids = set(existing_df["article_id"].values)
    mask = ~new_df["article_id"].isin(existing_ids)
    new_only = new_df[mask].copy()
    duplicates_removed = len(new_df) - len(new_only)

    if duplicates_removed > 0:
        print(f"  🔁 Deduplication: removed {duplicates_removed} already-seen articles")

    return new_only


# ============================================================
# Save to CSV
# ============================================================

def load_existing_articles() -> pd.DataFrame:
    """Load existing raw_articles.csv if it exists, else return empty DataFrame."""
    if os.path.exists(RAW_ARTICLES_PATH):
        try:
            return pd.read_csv(RAW_ARTICLES_PATH)
        except Exception as e:
            print(f"  ⚠️  Could not read existing CSV: {e}")
    return pd.DataFrame(columns=RAW_ARTICLE_COLUMNS)


def save_articles(df: pd.DataFrame):
    """Append new articles to raw_articles.csv."""
    ensure_data_dir()

    if df.empty:
        print("  ⚠️  No new articles to save")
        return

    # Ensure correct column order
    df = df[RAW_ARTICLE_COLUMNS]

    if os.path.exists(RAW_ARTICLES_PATH):
        # Append without writing header again
        df.to_csv(RAW_ARTICLES_PATH, mode="a", header=False, index=False)
    else:
        # First time — write with header
        df.to_csv(RAW_ARTICLES_PATH, mode="w", header=True, index=False)

    print(f"  💾 Saved {len(df)} new articles to {RAW_ARTICLES_PATH}")


# ============================================================
# Main collect function
# ============================================================

def collect_all() -> pd.DataFrame:
    """
    Run all 4 collectors, merge results, deduplicate,
    tag sectors, and save to raw_articles.csv.
    Returns the DataFrame of newly added articles.
    """
    print("\n" + "=" * 50)
    print("  Collecting from all sources...")
    print("=" * 50)

    # Run all collectors
    all_articles = []
    all_articles.extend(fetch_newsapi())
    all_articles.extend(fetch_hackernews())
    all_articles.extend(fetch_devto())
    all_articles.extend(fetch_google_news_rss())

    if not all_articles:
        print("\n  ❌ No articles collected from any source")
        return pd.DataFrame(columns=RAW_ARTICLE_COLUMNS)

    # Convert to DataFrame
    new_df = pd.DataFrame(all_articles, columns=RAW_ARTICLE_COLUMNS)

    # Remove duplicates within this batch (same URL appearing in multiple sources)
    before = len(new_df)
    new_df = new_df.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    if len(new_df) < before:
        print(f"  🔁 Removed {before - len(new_df)} cross-source duplicates")

    # Remove articles already in the CSV from previous runs
    existing_df = load_existing_articles()
    new_df = deduplicate(new_df, existing_df)

    # Save new articles
    save_articles(new_df)

    # Write a timestamp marker so the dashboard can show "Last collected at X"
    marker_path = os.path.join(DATA_DIR, "last_collected.txt")
    with open(marker_path, "w") as f:
        f.write(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    # Print summary
    print("\n" + "=" * 50)
    print("  Collection summary")
    print("=" * 50)
    print(f"  Already in CSV     : {len(existing_df)}")
    print(f"  Fetched this run   : {len(pd.DataFrame(all_articles)) if all_articles else 0}")
    print(f"  Truly new articles : {len(new_df)}")

    if not new_df.empty:
        print(f"\n  By source:")
        source_counts = new_df["data_source"].value_counts()
        for source, count in source_counts.items():
            print(f"    {source:<15} {count} articles")

        print(f"\n  By sector:")
        sector_counts = new_df["sector"].value_counts()
        for sector, count in sector_counts.items():
            print(f"    {sector:<20} {count} articles")

    print("=" * 50 + "\n")

    return new_df


# ============================================================
# Run directly to test
# ============================================================

if __name__ == "__main__":
    df = collect_all()
    if not df.empty:
        print(f"✅ collector.py working correctly")
        print(f"   Sample titles:")
        for title in df["title"].head(5).values:
            print(f"   • {title[:80]}")
    else:
        print("❌ No articles collected — check your API keys and internet connection")