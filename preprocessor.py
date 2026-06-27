# ============================================================
# preprocessor.py — Clean and prepare raw articles
# ============================================================
# Reads  : data/raw_articles.csv
# Writes : data/preprocessed_articles.csv
#
# What it does:
#   1. Drops rows with no title and no description
#   2. Removes HTML tags from all text fields
#   3. Removes URLs, mentions, special characters
#   4. Detects and keeps only English articles
#   5. Removes articles with text too short to analyze
#   6. Deduplicates again on cleaned title (catches near-duplicates)
#   7. Builds a single combined_text column for the sentiment model
#   8. Saves clean data to preprocessed_articles.csv
#
# Run directly to test:
#   python preprocessor.py

import os
import re
import pandas as pd
from datetime import datetime, timezone

from config import (
    RAW_ARTICLES_PATH,
    DATA_DIR,
    ensure_data_dir,
)

# Output path — separate from raw so you can always re-run preprocessing
PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed_articles.csv")

# Minimum character length for combined_text to be worth analyzing
MIN_TEXT_LENGTH = 30


# ============================================================
# Text cleaning helpers
# ============================================================

def remove_html_tags(text: str) -> str:
    """Strip HTML tags like <b>, <p>, <a href=...> etc."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove http/https URLs."""
    if not text:
        return ""
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_special_characters(text: str) -> str:
    """
    Keep only letters, numbers, spaces, and basic punctuation.
    Removes emojis, symbols, and non-printable characters.
    """
    if not text:
        return ""
    # Keep letters, digits, spaces, and . , ! ? ' -
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_field(text) -> str:
    """
    Full cleaning pipeline for a single text field.
    Applies all cleaning steps in order.
    """
    if not text or not isinstance(text, str):
        return ""
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    return text.strip()


def build_combined_text(row) -> str:
    """
    Build a single text string for sentiment analysis.
    Priority: title + description (what we have on free NewsAPI tier).
    Falls back gracefully if any field is empty.
    """
    parts = []

    title = str(row.get("title", "")).strip()
    description = str(row.get("description", "")).strip()
    content = str(row.get("content", "")).strip()

    if title:
        parts.append(title)

    # Add description if it's different from title
    if description and description.lower() != title.lower():
        parts.append(description)

    # Add content only if it adds new information
    if content and content.lower() not in (title.lower(), description.lower()):
        # Take first 300 chars of content to stay within model token limits
        parts.append(content[:300])

    combined = " ".join(parts)
    return combined[:512]  # Hard cap at 512 chars — RoBERTa's max token window


# ============================================================
# Language detection
# ============================================================

def is_english(text: str) -> bool:
    """
    Detect if text is English.
    Uses langdetect library. Returns True if English, False otherwise.
    Falls back to True if detection fails (don't drop article on error).
    """
    if not text or len(text) < 20:
        return True  # Too short to detect — keep it

    try:
        from langdetect import detect, LangDetectException
        return detect(text) == "en"
    except Exception:
        return True  # Keep article if detection fails


# ============================================================
# Main preprocessing function
# ============================================================

def preprocess() -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Reads raw_articles.csv, cleans it, saves preprocessed_articles.csv.
    Returns the clean DataFrame.
    """
    print("\n" + "=" * 50)
    print("  Preprocessing articles...")
    print("=" * 50)

    # ── Step 1: Load raw data ──────────────────────────────
    if not os.path.exists(RAW_ARTICLES_PATH):
        print(f"  ❌ {RAW_ARTICLES_PATH} not found")
        print("  👉 Run: python main.py collect  first")
        return pd.DataFrame()

    df = pd.read_csv(RAW_ARTICLES_PATH)
    initial_count = len(df)
    print(f"  📥 Loaded {initial_count} raw articles")

    if df.empty:
        print("  ❌ raw_articles.csv is empty — run collector first")
        return pd.DataFrame()

    # ── Step 2: Drop rows missing both title and description ─
    before = len(df)
    df = df.dropna(subset=["title", "description"], how="all")
    dropped = before - len(df)
    if dropped:
        print(f"  🗑️  Dropped {dropped} rows with no title and no description")

    # Fill remaining NaN text fields with empty string
    for col in ["title", "description", "content"]:
        df[col] = df[col].fillna("")

    # ── Step 3: Clean all text fields ─────────────────────
    print("  🧹 Cleaning text fields...")
    df["title"]       = df["title"].apply(clean_field)
    df["description"] = df["description"].apply(clean_field)
    df["content"]     = df["content"].apply(clean_field)

    # ── Step 4: Drop rows where title is now empty after cleaning ─
    before = len(df)
    df = df[df["title"].str.len() > 0]
    dropped = before - len(df)
    if dropped:
        print(f"  🗑️  Dropped {dropped} rows with empty title after cleaning")

    # ── Step 5: Build combined_text for sentiment model ───
    print("  🔗 Building combined text for sentiment analysis...")
    df["combined_text"] = df.apply(build_combined_text, axis=1)

    # ── Step 6: Drop rows where combined_text is too short ─
    before = len(df)
    df = df[df["combined_text"].str.len() >= MIN_TEXT_LENGTH]
    dropped = before - len(df)
    if dropped:
        print(f"  🗑️  Dropped {dropped} rows with text too short to analyze (<{MIN_TEXT_LENGTH} chars)")

    # ── Step 7: Language detection (keep English only) ────
    print("  🌐 Detecting language (keeping English only)...")
    df["is_english"] = df["combined_text"].apply(is_english)
    before = len(df)
    df = df[df["is_english"] == True].copy()
    df = df.drop(columns=["is_english"])
    dropped = before - len(df)
    if dropped:
        print(f"  🗑️  Dropped {dropped} non-English articles")

    # ── Step 8: Clean near-duplicates on title ────────────
    # Some sources republish the same headline with minor wording changes.
    # Lowercase + strip the title and drop exact matches.
    before = len(df)
    df["title_lower"] = df["title"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["title_lower"]).copy()
    df = df.drop(columns=["title_lower"])
    dropped = before - len(df)
    if dropped:
        print(f"  🔁 Removed {dropped} near-duplicate titles")

    # ── Step 9: Standardize published_at format ───────────
    print("  📅 Standardizing dates...")

    def parse_any_date(val):
        if not val or (isinstance(val, float)):
            return None
        s = str(val).strip().replace("Z", "+00:00")
        formats = [
            "%Y-%m-%dT%H:%M:%S+00:00",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return pd.to_datetime(s, format=fmt, utc=True).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
        try:
            return pd.to_datetime(s, utc=True).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    df["published_at"] = df["published_at"].apply(parse_any_date)
    dropped_dates = df["published_at"].isna().sum()
    if dropped_dates:
        print(f"  ⚠️  {dropped_dates} articles had unparseable dates — dropped")
    df = df.dropna(subset=["published_at"])

    # ── Step 10: Add preprocessed_at timestamp ────────────
    df["preprocessed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # ── Step 11: Reset index ──────────────────────────────
    df = df.reset_index(drop=True)

    # ── Step 12: Save to CSV ──────────────────────────────
    ensure_data_dir()
    df.to_csv(PREPROCESSED_PATH, index=False)

    # ── Summary ───────────────────────────────────────────
    final_count = len(df)
    removed_total = initial_count - final_count

    print("\n" + "=" * 50)
    print("  Preprocessing summary")
    print("=" * 50)
    print(f"  Input articles  : {initial_count}")
    print(f"  Removed total   : {removed_total}")
    print(f"  Clean articles  : {final_count}")

    if not df.empty:
        print(f"\n  By source:")
        for source, count in df["data_source"].value_counts().items():
            print(f"    {source:<15} {count} articles")

        print(f"\n  By sector:")
        for sector, count in df["sector"].value_counts().items():
            print(f"    {sector:<20} {count} articles")

        avg_len = int(df["combined_text"].str.len().mean())
        print(f"\n  Avg combined_text length : {avg_len} chars")

    print(f"\n  💾 Saved to: {PREPROCESSED_PATH}")
    print("=" * 50 + "\n")

    return df


# ============================================================
# Quick inspection helper
# ============================================================

def inspect_preprocessed():
    """
    Print a sample of the preprocessed data so you can
    visually verify the cleaning worked correctly.
    """
    if not os.path.exists(PREPROCESSED_PATH):
        print("❌ preprocessed_articles.csv not found — run preprocess() first")
        return

    df = pd.read_csv(PREPROCESSED_PATH)
    print(f"\n📊 preprocessed_articles.csv — {len(df)} articles")
    print(f"   Columns: {list(df.columns)}\n")

    print("Sample articles:")
    print("-" * 60)
    for _, row in df.head(5).iterrows():
        print(f"  Title   : {row['title'][:70]}")
        print(f"  Sector  : {row['sector']}")
        print(f"  Source  : {row['data_source']}")
        print(f"  Text len: {len(str(row['combined_text']))} chars")
        print(f"  Combined: {str(row['combined_text'])[:100]}...")
        print()


# ============================================================
# Run directly to test
# ============================================================

if __name__ == "__main__":
    df = preprocess()
    if not df.empty:
        print("✅ preprocessor.py working correctly\n")
        inspect_preprocessed()
    else:
        print("❌ Preprocessing failed — check raw_articles.csv exists")