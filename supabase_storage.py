# supabase_storage.py
# ============================================================
# Downloads the latest CSVs from Supabase Storage.
# Used by app.py when running on Streamlit Cloud (no local data/).
# Falls back to local data/ folder if Supabase is not configured
# — so your local dev workflow is unchanged.
# ============================================================

import io
import os

import pandas as pd

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
BUCKET       = "techpulse"

_client = None


def _get_client():
    """Lazy-load Supabase client so import doesn't fail if not installed."""
    global _client
    if _client is None:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def supabase_configured() -> bool:
    """Returns True if Supabase env vars are set."""
    return bool(SUPABASE_URL and SUPABASE_KEY)


def read_csv_from_supabase(filename: str) -> pd.DataFrame:
    """
    Download a CSV file from Supabase Storage and return as DataFrame.
    filename: just the filename e.g. "sentiment_results.csv"
    """
    try:
        client   = _get_client()
        response = client.storage.from_(BUCKET).download(filename)
        return pd.read_csv(io.BytesIO(response), on_bad_lines="skip")
    except Exception as e:
        print(f"  ⚠️  Could not download {filename} from Supabase: {e}")
        return pd.DataFrame()


def read_text_from_supabase(filename: str) -> str:
    """Download a plain text file from Supabase Storage."""
    try:
        client   = _get_client()
        response = client.storage.from_(BUCKET).download(filename)
        return response.decode("utf-8").strip()
    except Exception:
        return ""