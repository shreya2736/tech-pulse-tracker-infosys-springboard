# ============================================================
# config.py — Central configuration for Tech Pulse Tracker
# ============================================================
# Every other module imports from here.
# Run this file directly to test your configuration:
#   python config.py

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# ============================================================
# API Keys
# ============================================================

NEWS_API_KEY       = os.getenv("NEWS_API_KEY", "")
GMAIL_SENDER       = os.getenv("GMAIL_SENDER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
GMAIL_RECEIVER     = os.getenv("GMAIL_RECEIVER", "")


# ============================================================
# Pipeline settings
# ============================================================

REFRESH_INTERVAL_HOURS  = int(os.getenv("REFRESH_INTERVAL_HOURS", 2))
MAX_ARTICLES_PER_SOURCE = int(os.getenv("MAX_ARTICLES_PER_SOURCE", 50))


# ============================================================
# Search queries per source
# ============================================================

NEWSAPI_QUERY = (
    "artificial intelligence OR machine learning OR "
    "tech layoffs OR startup OR cloud computing OR "
    "cybersecurity OR semiconductor OR OpenAI OR "
    "Google OR Microsoft OR Apple OR Nvidia"
)

RSS_QUERY = "technology+AI+startup+cloud+cybersecurity+semiconductor"

# HackerNews
HN_TOP_STORIES_COUNT = 50

# Dev.to
DEVTO_API_URL          = "https://dev.to/api/articles"
DEVTO_TAGS             = ["ai", "machinelearning", "cloud", "security", "startup", "programming"]
DEVTO_ARTICLES_PER_TAG = 10


# ============================================================
# Sector definitions
# ============================================================

SECTOR_KEYWORDS = {
    "AI/ML": [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "llm", "large language model", "openai",
        "chatgpt", "gpt", "gemini", "claude", "generative ai",
        "transformer", "diffusion model", "computer vision", "nlp",
        "reinforcement learning", "ai model", "foundation model",
    ],
    "Cloud": [
        "cloud computing", "aws", "amazon web services", "azure",
        "google cloud", "gcp", "saas", "paas", "iaas", "kubernetes",
        "docker", "serverless", "cloud storage", "data center",
        "cloud migration", "multicloud", "hybrid cloud",
    ],
    "Cybersecurity": [
        "cybersecurity", "data breach", "ransomware", "malware",
        "hacking", "vulnerability", "zero day", "phishing",
        "cyber attack", "security flaw", "encryption", "firewall",
        "threat intelligence", "endpoint security", "infosec",
    ],
    "Startups": [
        "startup", "venture capital", "vc funding", "series a",
        "series b", "seed round", "unicorn", "yc", "y combinator",
        "techcrunch", "funding round", "valuation", "pitch",
        "founder", "bootstrapped", "accelerator", "incubator",
    ],
    "Semiconductors": [
        "semiconductor", "chip", "nvidia", "amd", "intel", "tsmc",
        "gpu", "cpu", "processor", "wafer", "fab", "chipmaker",
        "microchip", "silicon", "arm", "qualcomm", "transistor",
    ],
    "Big Tech": [
        "google", "microsoft", "apple", "meta", "amazon", "alphabet",
        "big tech", "faang", "antitrust", "tech giant", "regulation",
        "privacy", "gdpr", "monopoly", "platform", "app store",
    ],
}

DEFAULT_SECTOR = "General Tech"


# ============================================================
# Alert thresholds
# ============================================================

ALERT_NEGATIVE_THRESHOLD   = -0.20 
ALERT_POSITIVE_THRESHOLD   =  0.50
ALERT_VOLATILITY_THRESHOLD =  0.15
ALERT_TREND_CHANGE_MIN     =  0.10


# ============================================================
# File paths
# ============================================================

DATA_DIR               = "data"
RAW_ARTICLES_PATH      = os.path.join(DATA_DIR, "raw_articles.csv")
SENTIMENT_RESULTS_PATH = os.path.join(DATA_DIR, "sentiment_results.csv")
FORECAST_RESULTS_PATH  = os.path.join(DATA_DIR, "forecast_results.csv")
ALERT_HISTORY_PATH     = os.path.join(DATA_DIR, "alert_history.csv")


# ============================================================
# Column definitions
# ============================================================

RAW_ARTICLE_COLUMNS = [
    "article_id",
    "title",
    "description",
    "content",
    "url",
    "source",
    "published_at",
    "collected_at",
    "sector",
    "data_source",
]

SENTIMENT_COLUMNS = [
    "article_id",
    "title",
    "url",
    "source",
    "published_at",
    "sector",
    "data_source",
    "sentiment_label",
    "sentiment_score",
    "sentiment_confidence",
    "keywords",
    "analyzed_at",
]


# ============================================================
# Validation
# ============================================================

def validate_config() -> bool:
    print("\n" + "=" * 50)
    print("  Tech Pulse Tracker — Config Validation")
    print("=" * 50)

    all_ok = True

    required = {
        "NEWS_API_KEY": NEWS_API_KEY,
    }

    optional = {
        "GMAIL_SENDER":       GMAIL_SENDER,
        "GMAIL_APP_PASSWORD": GMAIL_APP_PASSWORD,
        "GMAIL_RECEIVER":     GMAIL_RECEIVER,
    }

    print("\n  Required keys:")
    for name, value in required.items():
        if not value or value.startswith("your_"):
            print(f"  ❌  {name} — NOT SET")
            all_ok = False
        else:
            masked = value[:6] + "*" * (len(value) - 6)
            print(f"  ✅  {name} — {masked}")

    print("\n  Optional keys (email alerts):")
    for name, value in optional.items():
        if not value or value.startswith("your_"):
            print(f"  ⚠️   {name} — not set (email alerts disabled)")
        else:
            masked = value[:4] + "*" * (len(value) - 4)
            print(f"  ✅  {name} — {masked}")

    print("\n  Sources that need no key:")
    print("  ✅  HackerNews API — free, no key needed")
    print("  ✅  Google News RSS — free, no key needed")
    print("  ✅  Dev.to API     — free, no key needed")

    print("\n  Pipeline settings:")
    print(f"  🔄  Refresh interval : every {REFRESH_INTERVAL_HOURS} hour(s)")
    print(f"  📰  Max articles/src : {MAX_ARTICLES_PER_SOURCE}")
    print(f"  📁  Data directory   : {os.path.abspath(DATA_DIR)}")

    print()
    if all_ok:
        print("  ✅  Config OK — ready to run the pipeline.")
    else:
        print("  ❌  Fix the missing keys above before running the pipeline.")
        print("  👉  Copy .env.example to .env and fill in your actual keys.")
    print("=" * 50 + "\n")

    return all_ok


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


if __name__ == "__main__":
    ensure_data_dir()
    ok = validate_config()
    if not ok:
        exit(1)