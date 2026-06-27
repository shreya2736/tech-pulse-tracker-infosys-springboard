# upload_to_supabase.py
# ============================================================
# Uploads the 3 result CSVs to Supabase Storage.
# Run automatically by GitHub Actions after the pipeline.
# You can also run it manually: python upload_to_supabase.py
# ============================================================

import os
import sys
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
BUCKET       = "techpulse"       # must match the bucket you create in Supabase

FILES_TO_UPLOAD = [
    "data/sentiment_results.csv",
    "data/forecast_results.csv",
    "data/alert_history.csv",
    "data/last_collected.txt",
]

def upload():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ SUPABASE_URL or SUPABASE_KEY not set")
        sys.exit(1)

    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"\n📤 Uploading files to Supabase bucket: {BUCKET}")

    for path in FILES_TO_UPLOAD:
        if not os.path.exists(path):
            print(f"  ⚠️  Skipping {path} — file not found")
            continue

        filename = os.path.basename(path)

        with open(path, "rb") as f:
            data = f.read()

        try:
            # upsert=True overwrites if the file already exists
            client.storage.from_(BUCKET).upload(
                path=filename,
                file=data,
                file_options={"upsert": "true"},
            )
            size_kb = len(data) // 1024
            print(f"  ✅ {filename} ({size_kb} KB)")
        except Exception as e:
            print(f"  ❌ Failed to upload {filename}: {e}")
            sys.exit(1)

    print("\n✅ All files uploaded to Supabase\n")


if __name__ == "__main__":
    upload()