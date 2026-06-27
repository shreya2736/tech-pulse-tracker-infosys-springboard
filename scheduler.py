# ============================================================
# scheduler.py — Auto-refresh pipeline every N hours
# ============================================================
# Runs as a background thread inside the Streamlit app.
# Every REFRESH_INTERVAL_HOURS it runs the full pipeline:
#   collect → preprocess → analyze → forecast → alerts
#
# Import and call start_scheduler() from app.py once.

import logging
import threading
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from config import REFRESH_INTERVAL_HOURS

# Suppress APScheduler's noisy logs
logging.getLogger("apscheduler").setLevel(logging.WARNING)

# Single global scheduler instance
_scheduler = None
_lock      = threading.Lock()


# ============================================================
# Pipeline job
# ============================================================

def run_pipeline_job():
    """
    The job APScheduler calls on every tick.
    Runs each pipeline step independently so one failure
    doesn't block the rest.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n🔄 [{now}] Scheduled pipeline starting...")

    steps = [
        ("Collect",    _run_collect),
        ("Preprocess", _run_preprocess),
        ("Analyze",    _run_analyze),
        ("Forecast",   _run_forecast),
        ("Alerts",     _run_alerts),
    ]

    for name, fn in steps:
        try:
            fn()
            print(f"  ✅ {name} complete")
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            # Continue with next step even if this one fails

    print(f"🏁 Scheduled pipeline finished at "
          f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}\n")


def _run_collect():
    from collector import collect_all
    collect_all()


def _run_preprocess():
    from preprocessor import preprocess
    preprocess()


def _run_analyze():
    from analyzer import analyze
    analyze()


def _run_forecast():
    from forecasting import forecast
    forecast()


def _run_alerts():
    from alerts import check_and_send_alerts
    check_and_send_alerts()


# ============================================================
# Scheduler lifecycle
# ============================================================

def _on_job_event(event):
    """Log job execution and errors."""
    if event.exception:
        print(f"  ⚠️  Scheduler job error: {event.exception}")


def start_scheduler():
    """
    Start the background scheduler.
    Safe to call multiple times — only starts once.
    Call this once from app.py at startup.
    """
    global _scheduler

    with _lock:
        if _scheduler is not None and _scheduler.running:
            return  # Already running

        _scheduler = BackgroundScheduler(timezone="UTC")
        _scheduler.add_listener(_on_job_event, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

        #scheduler should not run the moment the dashboard opens.

        # Then run on interval
        _scheduler.add_job(
            func             = run_pipeline_job,
            trigger          = "interval",
            hours            = REFRESH_INTERVAL_HOURS,
            id               = "pipeline_interval",
            name             = f"Pipeline every {REFRESH_INTERVAL_HOURS}h",
            misfire_grace_time = 300,
        )

        _scheduler.start()
        print(f"⏰ Scheduler started — pipeline runs every "
              f"{REFRESH_INTERVAL_HOURS} hour(s)")


def stop_scheduler():
    """Stop the scheduler gracefully. Called on app shutdown."""
    global _scheduler
    with _lock:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            print("⏹️  Scheduler stopped")


def get_next_run_time() -> str:
    """
    Return the next scheduled pipeline run time as a string.
    Used by the dashboard to show 'Next refresh at HH:MM UTC'.
    """
    global _scheduler
    if not _scheduler or not _scheduler.running:
        return "Scheduler not running"

    try:
        job = _scheduler.get_job("pipeline_interval")
        if job and job.next_run_time:
            return job.next_run_time.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    return "Unknown"


def is_running() -> bool:
    """Return True if the scheduler is active."""
    global _scheduler
    return _scheduler is not None and _scheduler.running


# ============================================================
# Run directly to test
# ============================================================

if __name__ == "__main__":
    import time
    print("Testing scheduler — will run pipeline once then stop...")
    start_scheduler()
    time.sleep(5)
    print(f"Next run: {get_next_run_time()}")
    print(f"Running : {is_running()}")
    stop_scheduler()