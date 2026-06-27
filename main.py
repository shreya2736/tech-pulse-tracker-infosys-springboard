# ============================================================
# main.py — CLI entry point for Tech Pulse Tracker
# ============================================================
# Usage:
#   python main.py config      → validate API keys and settings
#   python main.py collect     → run data collection only
#   python main.py preprocess  → run preprocessing only
#   python main.py analyze     → run sentiment analysis only
#   python main.py forecast    → run forecasting only
#   python main.py alerts      → check and send alerts only
#   python main.py full        → run complete pipeline
#   python main.py dashboard   → launch Streamlit dashboard

import sys
import os
from config import validate_config, ensure_data_dir


def run_config():
    """Validate configuration and API keys."""
    ensure_data_dir()
    ok = validate_config()
    if not ok:
        sys.exit(1)


def run_collect():
    """Collect articles from all data sources."""
    from collector import collect_all
    print("\n📡 Starting data collection...")
    df = collect_all()
    print(f"✅ Collection complete — {len(df)} articles saved to raw_articles.csv")


def run_preprocess():
    """Clean and preprocess raw articles."""
    from preprocessor import preprocess
    print("\n🧹 Starting preprocessing...")
    df = preprocess()
    print(f"✅ Preprocessing complete — {len(df)} clean articles ready")


def run_analyze():
    """Run sentiment analysis and keyword extraction."""
    from analyzer import analyze
    print("\n🧠 Starting sentiment analysis...")
    df = analyze()
    print(f"✅ Analysis complete — {len(df)} articles analyzed")


def run_forecast():
    """Generate 7-day sentiment forecast."""
    from forecasting import forecast
    print("\n🔮 Starting forecast generation...")
    forecast_df = forecast()
    if forecast_df is not None:
        print(f"✅ Forecast complete — {len(forecast_df)} days forecasted")
    else:
        print("⚠️  Forecast skipped — not enough data yet (need 7+ days)")


def run_alerts():
    """Check thresholds and send alerts."""
    from alerts import check_and_send_alerts
    print("\n🚨 Checking for alerts...")
    triggered = check_and_send_alerts()
    if triggered:
        print(f"✅ {len(triggered)} alert(s) triggered and logged")
    else:
        print("✅ No alerts triggered — all clear")


def run_full_pipeline():
    """Run the complete pipeline end to end."""
    print("\n" + "=" * 50)
    print("  Tech Pulse Tracker — Full Pipeline")
    print("=" * 50)

    run_config()
    run_collect()
    run_preprocess()
    run_analyze()
    run_forecast()
    run_alerts()

    print("\n" + "=" * 50)
    print("  ✅  Pipeline complete!")
    print("  👉  Run: streamlit run app.py  to view dashboard")
    print("=" * 50 + "\n")


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    print("\n🚀 Launching dashboard at http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def show_help():
    print("""
  Tech Pulse Tracker — Commands

  python main.py config      Validate API keys and settings
  python main.py collect     Collect articles from all sources
  python main.py preprocess  Clean and tag raw articles
  python main.py analyze     Run FinBERT sentiment + YAKE keywords
  python main.py forecast    Generate 7-day Prophet forecast
  python main.py alerts      Check thresholds, send email if triggered
  python main.py full        Run complete pipeline end to end
  python main.py dashboard   Launch Streamlit dashboard
    """)


# ============================================================
# Entry point
# ============================================================

COMMANDS = {
    "config":     run_config,
    "collect":    run_collect,
    "preprocess": run_preprocess,
    "analyze":    run_analyze,
    "forecast":   run_forecast,
    "alerts":     run_alerts,
    "full":       run_full_pipeline,
    "dashboard":  run_dashboard,
    "help":       show_help,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: run full pipeline
        run_full_pipeline()
    else:
        command = sys.argv[1].lower()
        if command in COMMANDS:
            COMMANDS[command]()
        else:
            print(f"\n❌  Unknown command: '{command}'")
            show_help()
            sys.exit(1)