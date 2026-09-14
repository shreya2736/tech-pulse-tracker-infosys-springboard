# upload_to_supabase.py
# ============================================================
# Uploads the 3 result CSVs to Supabase Storage.
# Run automatically by GitHub Actions after the pipeline.
# You can also run it manually: python upload_to_supabase.py
# ============================================================

import os
import sys
from supabase import create_client
import time

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

            for attempt in range(3):         # try up to 3 times
                try:
                    client.storage.from_(BUCKET).upload(
                        path=filename,
                        file=data,
                        file_options={"upsert": "true"},
                    )
                    break                     # success — stop retrying
                except Exception as e:
                    if attempt < 2:
                        print(f"  ⚠️ Attempt {attempt+1} failed: {e} — retrying in 5s")
                        time.sleep(5)
                    else:
                        print(f"  ❌ Failed to upload {filename} after 3 attempts: {e}")
                        sys.exit(1)

        print("\n✅ All files uploaded to Supabase\n")


if __name__ == "__main__":
    upload()