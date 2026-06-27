# ============================================================
# analyzer.py — Sentiment analysis + keyword extraction
# ============================================================
# Reads  : data/preprocessed_articles.csv
# Writes : data/sentiment_results.csv
#
# What it does:
#   1. Loads RoBERTa sentiment model (downloads once, cached after)
#   2. Processes articles in batches of 16 for efficiency
#   3. Maps model output to a numeric score (-1.0 to +1.0)
#   4. Extracts top 5 keyphrases per article using YAKE
#   5. Saves results to sentiment_results.csv
#
# Sentiment score scale:
#   +1.0  strongly positive
#    0.0  neutral
#   -1.0  strongly negative
#
# Run directly to test:
#   python analyzer.py

import csv
import os
import warnings
from datetime import datetime, timezone

import pandas as pd
import yake

warnings.filterwarnings("ignore")

from config import (
    DATA_DIR,
    SENTIMENT_COLUMNS,
    SENTIMENT_RESULTS_PATH,
    ensure_data_dir,
)

PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed_articles.csv")

# RoBERTa model — trained on tweets and news, handles social media well
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Batch size for inference — 16 is safe for CPU
BATCH_SIZE = 16

# YAKE settings
YAKE_MAX_NGRAM    = 2   # Extract up to 2-word phrases e.g. "machine learning"
YAKE_TOP_N        = 5   # Top 5 keyphrases per article
YAKE_DEDUP_THRESH = 0.7 # Avoid near-duplicate keyphrases


# ============================================================
# Model loading
# ============================================================

def load_model():
    """
    Load RoBERTa sentiment pipeline from HuggingFace.
    Downloads model on first run (~500MB), cached after that.
    Returns the pipeline object or None if loading fails.
    """
    try:
        from transformers import pipeline as hf_pipeline
        print("  🤖 Loading RoBERTa sentiment model...")
        print("     (First run downloads ~500MB — subsequent runs use cache)")

        sentiment_pipeline = hf_pipeline(
            task            = "sentiment-analysis",
            model           = MODEL_NAME,
            tokenizer       = MODEL_NAME,
            truncation      = True,    # Truncate text exceeding 512 tokens
            max_length      = 512,     # RoBERTa's maximum token window
            padding         = True,    # Pad shorter sequences in a batch
            batch_size      = BATCH_SIZE,
        )
        print("  ✅ Model loaded successfully")
        return sentiment_pipeline

    except Exception as e:
        print(f"  ❌ Failed to load RoBERTa model: {e}")
        print("  💡 Make sure transformers and torch are installed:")
        print("     pip install transformers torch")
        return None


# ============================================================
# Sentiment scoring
# ============================================================

def label_to_score(label: str, confidence: float) -> float:
    """
    Convert RoBERTa label + confidence to a numeric sentiment score.

    RoBERTa outputs:
      "positive" with confidence e.g. 0.92
      "negative" with confidence e.g. 0.87
      "neutral"  with confidence e.g. 0.65

    We map this to a single float:
      positive → +confidence  (e.g. +0.92)
      negative → -confidence  (e.g. -0.87)
      neutral  →  0.0

    This gives an intuitive -1 to +1 scale.
    """
    label = label.lower().strip()
    if label == "positive":
        return round(confidence, 4)
    elif label == "negative":
        return round(-confidence, 4)
    else:
        return 0.0


def analyze_batch(texts: list[str], pipeline) -> list[dict]:
    """
    Run a batch of texts through the sentiment pipeline.
    Returns a list of dicts with label, score, confidence.
    Falls back to neutral on any error.
    """
    try:
        results = pipeline(texts)
        output = []
        for result in results:
            label      = result["label"].lower()
            confidence = round(result["score"], 4)
            score      = label_to_score(label, confidence)
            output.append({
                "sentiment_label":      label,
                "sentiment_score":      score,
                "sentiment_confidence": confidence,
            })
        return output

    except Exception as e:
        print(f"  ⚠️  Batch error: {e} — marking batch as neutral")
        return [
            {"sentiment_label": "neutral",
             "sentiment_score": 0.0,
             "sentiment_confidence": 0.0}
        ] * len(texts)


# ============================================================
# Keyword extraction
# ============================================================

def extract_keywords(text: str) -> str:
    """
    Extract top keyphrases from text using YAKE.
    Returns a comma-separated string of keyphrases.
    e.g. "machine learning, openai model, neural network, gpt-5, language model"

    YAKE is unsupervised — no model download, no API, works offline.
    Lower YAKE score = more important keyphrase (counter-intuitive but correct).
    """
    if not text or len(text.strip()) < 10:
        return ""

    try:
        extractor = yake.KeywordExtractor(
            lan         = "en",
            n           = YAKE_MAX_NGRAM,
            dedupLim    = YAKE_DEDUP_THRESH,
            top         = YAKE_TOP_N,
            features    = None,
        )
        keywords = extractor.extract_keywords(text)
        # keywords is a list of (keyphrase, score) tuples
        # Sort by score ascending (lower = more important)
        keywords.sort(key=lambda x: x[1])
        # Use " | " separator — commas inside keywords corrupt CSV columns
        return " | ".join([kw for kw, score in keywords])

    except Exception as e:
        print(f"  ⚠️  YAKE error: {e}")
        return ""


# ============================================================
# Main analysis function
# ============================================================

def analyze() -> pd.DataFrame:
    """
    Full analysis pipeline.
    Reads preprocessed_articles.csv, runs sentiment + keywords,
    saves results to sentiment_results.csv.
    Returns the results DataFrame.
    """
    print("\n" + "=" * 50)
    print("  Running sentiment analysis...")
    print("=" * 50)

    # ── Step 1: Load preprocessed data ────────────────────
    if not os.path.exists(PREPROCESSED_PATH):
        print(f"  ❌ {PREPROCESSED_PATH} not found")
        print("  👉 Run: python main.py preprocess  first")
        return pd.DataFrame()

    df = pd.read_csv(PREPROCESSED_PATH)
    print(f"  📥 Loaded {len(df)} preprocessed articles")

    if df.empty:
        print("  ❌ No articles to analyze")
        return pd.DataFrame()

    # ── Step 2: Skip already-analyzed articles ─────────────
    # If sentiment_results.csv exists, only analyze new articles.
    # This avoids re-running the model on articles we've already processed.
    already_analyzed_ids = set()
    if os.path.exists(SENTIMENT_RESULTS_PATH):
        try:
            existing = pd.read_csv(SENTIMENT_RESULTS_PATH, on_bad_lines="skip")
            already_analyzed_ids = set(existing["article_id"].values)
            print(f"  ⏭️  Skipping {len(already_analyzed_ids)} already-analyzed articles")
        except Exception:
            pass

    new_df = df[~df["article_id"].isin(already_analyzed_ids)].copy()
    print(f"  🆕 Articles to analyze: {len(new_df)}")

    if new_df.empty:
        print("  ✅ All articles already analyzed — nothing new to process")
        # Return existing results
        if os.path.exists(SENTIMENT_RESULTS_PATH):
            return pd.read_csv(SENTIMENT_RESULTS_PATH, on_bad_lines="skip")
        return pd.DataFrame()

    # ── Step 3: Load model ────────────────────────────────
    pipeline = load_model()
    if pipeline is None:
        print("  ❌ Cannot proceed without sentiment model")
        return pd.DataFrame()

    # ── Step 4: Run sentiment analysis in batches ─────────
    print(f"\n  🔍 Analyzing sentiment in batches of {BATCH_SIZE}...")

    texts = new_df["combined_text"].fillna("").tolist()
    all_sentiments = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch_num  = i // BATCH_SIZE + 1
        batch      = texts[i : i + BATCH_SIZE]
        sentiments = analyze_batch(batch, pipeline)
        all_sentiments.extend(sentiments)

        # Progress update every 5 batches
        if batch_num % 5 == 0 or batch_num == total_batches:
            done = min(i + BATCH_SIZE, len(texts))
            print(f"  📊 Progress: {done}/{len(texts)} articles "
                  f"({batch_num}/{total_batches} batches)")

    # ── Step 5: Extract keywords ──────────────────────────
    print(f"\n  🔑 Extracting keywords with YAKE...")
    keywords_list = []
    for idx, text in enumerate(texts):
        keywords = extract_keywords(text)
        keywords_list.append(keywords)
        if (idx + 1) % 50 == 0:
            print(f"  🔑 Keywords: {idx + 1}/{len(texts)} done")

    # ── Step 6: Build results DataFrame ──────────────────
    sentiment_df = pd.DataFrame(all_sentiments)
    new_df = new_df.reset_index(drop=True)
    sentiment_df = sentiment_df.reset_index(drop=True)

    results_df = pd.DataFrame({
        "article_id":           new_df["article_id"].values,
        "title":                new_df["title"].values,
        "url":                  new_df["url"].values,
        "source":               new_df["source"].values,
        "published_at":         new_df["published_at"].values,
        "sector":               new_df["sector"].values,
        "data_source":          new_df["data_source"].values,
        "sentiment_label":      sentiment_df["sentiment_label"].values,
        "sentiment_score":      sentiment_df["sentiment_score"].values,
        "sentiment_confidence": sentiment_df["sentiment_confidence"].values,
        "keywords":             keywords_list,
        "analyzed_at":          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    })
    

    # ── Step 7: Append to existing results ───────────────
    ensure_data_dir()

    if os.path.exists(SENTIMENT_RESULTS_PATH) and already_analyzed_ids:
        results_df.to_csv(
            SENTIMENT_RESULTS_PATH, mode="a", header=False, index=False,
            quoting=csv.QUOTE_ALL,
        )
    else:
        results_df.to_csv(SENTIMENT_RESULTS_PATH, index=False, quoting=csv.QUOTE_ALL)

    # ── Step 8: Load full results for summary ─────────────
    full_results = pd.read_csv(SENTIMENT_RESULTS_PATH, on_bad_lines="skip")

    # ── Summary ───────────────────────────────────────────
    total      = len(results_df)
    pos_count  = (results_df["sentiment_label"] == "positive").sum()
    neg_count  = (results_df["sentiment_label"] == "negative").sum()
    neu_count  = (results_df["sentiment_label"] == "neutral").sum()
    avg_score  = results_df["sentiment_score"].mean()
    avg_conf   = results_df["sentiment_confidence"].mean()

    print("\n" + "=" * 50)
    print("  Sentiment analysis summary")
    print("=" * 50)
    print(f"  Articles analyzed  : {total}")
    print(f"  Positive           : {pos_count} ({pos_count/total*100:.1f}%)")
    print(f"  Negative           : {neg_count} ({neg_count/total*100:.1f}%)")
    print(f"  Neutral            : {neu_count} ({neu_count/total*100:.1f}%)")
    print(f"  Average score      : {avg_score:+.3f}")
    print(f"  Avg confidence     : {avg_conf:.3f}")

    print(f"\n  By sector:")
    sector_scores = (
        results_df.groupby("sector")["sentiment_score"]
        .mean()
        .sort_values(ascending=False)
    )
    for sector, score in sector_scores.items():
        bar = "█" * int(abs(score) * 10)
        sign = "+" if score >= 0 else "-"
        print(f"    {sector:<20} {sign}{abs(score):.3f}  {bar}")

    print(f"\n  Sample results:")
    print(f"  {'Title':<45} {'Label':<10} {'Score':>6}")
    print(f"  {'-'*45} {'-'*10} {'-'*6}")
    for _, row in results_df.head(5).iterrows():
        title = row["title"][:44]
        print(f"  {title:<45} {row['sentiment_label']:<10} {row['sentiment_score']:>+.3f}")

    print(f"\n  💾 Saved to: {SENTIMENT_RESULTS_PATH}")
    print(f"  📊 Total in file: {len(full_results)} articles")
    print("=" * 50 + "\n")

    return full_results


# ============================================================
# Utility — inspect results
# ============================================================

def inspect_results():
    """Print a summary of sentiment_results.csv for quick verification."""
    if not os.path.exists(SENTIMENT_RESULTS_PATH):
        print("❌ sentiment_results.csv not found — run analyze() first")
        return

    df = pd.read_csv(SENTIMENT_RESULTS_PATH)
    print(f"\n📊 sentiment_results.csv — {len(df)} articles")
    print(f"   Columns: {list(df.columns)}\n")

    print("Sentiment distribution:")
    for label, count in df["sentiment_label"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {label:<10} {count:>4} ({pct:.1f}%)")

    print(f"\nAverage score by sector:")
    sector_scores = (
        df.groupby("sector")["sentiment_score"]
        .mean()
        .sort_values(ascending=False)
    )
    for sector, score in sector_scores.items():
        print(f"  {sector:<20} {score:+.3f}")

    print(f"\nSample keywords:")
    for _, row in df.head(3).iterrows():
        print(f"  {row['title'][:50]}")
        print(f"    → {row['keywords']}")
        print()


# ============================================================
# Run directly to test
# ============================================================

if __name__ == "__main__":
    df = analyze()
    if not df.empty:
        print("✅ analyzer.py working correctly\n")
        inspect_results()
    else:
        print("❌ Analysis failed — check preprocessed_articles.csv exists")