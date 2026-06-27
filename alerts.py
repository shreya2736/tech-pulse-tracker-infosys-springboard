# ============================================================
# alerts.py — Threshold detection, logging, and email alerts
# ============================================================
# Reads  : data/sentiment_results.csv
# Writes : data/alert_history.csv
#
# What it does:
#   1. Loads latest sentiment data
#   2. Checks 4 alert types against configurable thresholds
#   3. Logs every triggered alert to alert_history.csv
#   4. Sends Gmail email for HIGH severity alerts
#   5. Returns alert list for Streamlit in-app banners
#
# Alert types:
#   negative_sentiment  — avg score below ALERT_NEGATIVE_THRESHOLD
#   positive_surge      — avg score above ALERT_POSITIVE_THRESHOLD
#   volatility          — 7-day std dev above ALERT_VOLATILITY_THRESHOLD
#   trend_reversal      — 3-day direction change above ALERT_TREND_CHANGE_MIN
#
# Run directly to test:
#   python alerts.py

import os
import smtplib
import warnings
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    ALERT_HISTORY_PATH,
    ALERT_NEGATIVE_THRESHOLD,
    ALERT_POSITIVE_THRESHOLD,
    ALERT_TREND_CHANGE_MIN,
    ALERT_VOLATILITY_THRESHOLD,
    DATA_DIR,
    GMAIL_APP_PASSWORD,
    GMAIL_RECEIVER,
    GMAIL_SENDER,
    SENTIMENT_RESULTS_PATH,
    ensure_data_dir,
)


# ============================================================
# Alert data structure
# ============================================================

def make_alert(
    alert_type: str,
    sector: str,
    severity: str,
    value: float,
    threshold: float,
    message: str,
) -> dict:
    """
    Build a clean alert dictionary.
    This is the single format used everywhere —
    in the CSV log, in emails, and in Streamlit banners.

    severity: HIGH / MEDIUM / LOW
    alert_type: negative_sentiment / positive_surge /
                volatility / trend_reversal
    """
    return {
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "alert_type": alert_type,
        "sector":     sector,
        "severity":   severity,
        "value":      round(value, 4),
        "threshold":  round(threshold, 4),
        "message":    message,
    }


# ============================================================
# Threshold checks
# ============================================================

def check_negative_sentiment(daily_df: pd.DataFrame, sector: str) -> list[dict]:
    """
    Alert when the latest day's average sentiment is below threshold.
    Severity scales with how far below the threshold:
      HIGH   — score <= -0.40
      MEDIUM — score <= -0.30
      LOW    — score <= ALERT_NEGATIVE_THRESHOLD (-0.20)
    """
    alerts = []
    if daily_df.empty:
        return alerts

    latest_score = daily_df["avg_score"].iloc[-1]
    latest_date  = str(daily_df["date"].iloc[-1])

    if latest_score <= ALERT_NEGATIVE_THRESHOLD:
        if latest_score <= -0.40:
            severity = "HIGH"
        elif latest_score <= -0.30:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        alerts.append(make_alert(
            alert_type = "negative_sentiment",
            sector     = sector,
            severity   = severity,
            value      = latest_score,
            threshold  = ALERT_NEGATIVE_THRESHOLD,
            message    = (
                f"🚨 Negative sentiment detected in {sector}. "
                f"Score: {latest_score:+.3f} on {latest_date}. "
                f"Threshold: {ALERT_NEGATIVE_THRESHOLD:+.3f}."
            ),
        ))

    return alerts


def check_positive_surge(daily_df: pd.DataFrame, sector: str) -> list[dict]:
    """
    Alert when the latest day's average sentiment is above threshold.
    Severity scales with how far above the threshold:
      HIGH   — score >= 0.70
      MEDIUM — score >= 0.60
      LOW    — score >= ALERT_POSITIVE_THRESHOLD (0.50)
    """
    alerts = []
    if daily_df.empty:
        return alerts

    latest_score = daily_df["avg_score"].iloc[-1]
    latest_date  = str(daily_df["date"].iloc[-1])

    if latest_score >= ALERT_POSITIVE_THRESHOLD:
        if latest_score >= 0.70:
            severity = "HIGH"
        elif latest_score >= 0.60:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        alerts.append(make_alert(
            alert_type = "positive_surge",
            sector     = sector,
            severity   = severity,
            value      = latest_score,
            threshold  = ALERT_POSITIVE_THRESHOLD,
            message    = (
                f"🚀 Positive sentiment surge in {sector}. "
                f"Score: {latest_score:+.3f} on {latest_date}. "
                f"Threshold: {ALERT_POSITIVE_THRESHOLD:+.3f}."
            ),
        ))

    return alerts


def check_volatility(daily_df: pd.DataFrame, sector: str) -> list[dict]:
    """
    Alert when the 7-day rolling standard deviation of sentiment
    exceeds the volatility threshold — indicates unstable sentiment.
    Severity scales with volatility level:
      HIGH   — std > 0.30
      MEDIUM — std > 0.20
      LOW    — std > ALERT_VOLATILITY_THRESHOLD (0.15)
    """
    alerts = []

    # Need at least 3 days to calculate meaningful std
    if len(daily_df) < 3:
        return alerts

    # Use last 7 days (or all available if fewer)
    window = daily_df["avg_score"].tail(7)
    volatility = window.std(ddof=0)  # ddof=0 for population std

    if pd.isna(volatility):
        return alerts

    if volatility > ALERT_VOLATILITY_THRESHOLD:
        if volatility > 0.30:
            severity = "HIGH"
        elif volatility > 0.20:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        alerts.append(make_alert(
            alert_type = "volatility",
            sector     = sector,
            severity   = severity,
            value      = volatility,
            threshold  = ALERT_VOLATILITY_THRESHOLD,
            message    = (
                f"⚡ High sentiment volatility in {sector}. "
                f"7-day std dev: {volatility:.3f}. "
                f"Threshold: {ALERT_VOLATILITY_THRESHOLD:.3f}."
            ),
        ))

    return alerts


def check_trend_reversal(daily_df: pd.DataFrame, sector: str) -> list[dict]:
    """
    Alert when sentiment direction changes significantly.
    Compares the mean of the last 3 days vs the 3 days before that.
    A shift above ALERT_TREND_CHANGE_MIN indicates a reversal.
    Severity:
      HIGH   — change > 0.25
      MEDIUM — change > 0.15
      LOW    — change > ALERT_TREND_CHANGE_MIN (0.10)
    """
    alerts = []

    # Need at least 6 days for a meaningful comparison
    if len(daily_df) < 6:
        return alerts

    recent_mean = daily_df["avg_score"].iloc[-3:].mean()
    prior_mean  = daily_df["avg_score"].iloc[-6:-3].mean()
    change      = recent_mean - prior_mean

    if abs(change) > ALERT_TREND_CHANGE_MIN:
        if abs(change) > 0.25:
            severity = "HIGH"
        elif abs(change) > 0.15:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        direction = "improving 📈" if change > 0 else "declining 📉"

        alerts.append(make_alert(
            alert_type = "trend_reversal",
            sector     = sector,
            severity   = severity,
            value      = change,
            threshold  = ALERT_TREND_CHANGE_MIN,
            message    = (
                f"📊 Trend reversal in {sector}: sentiment is {direction}. "
                f"Change: {change:+.3f} over last 3 days. "
                f"Recent avg: {recent_mean:+.3f}, Prior avg: {prior_mean:+.3f}."
            ),
        ))

    return alerts


# ============================================================
# Daily aggregation helper
# ============================================================

def get_daily_sentiment(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """
    Aggregate sentiment_results.csv to daily average scores
    for a given sector (or 'Overall' for all sectors).
    Returns DataFrame with columns: date, avg_score, article_count.
    """
    if sector != "Overall":
        df = df[df["sector"] == sector]

    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at"])
    df["date"] = df["published_at"].dt.date

    daily = (
        df.groupby("date")
        .agg(
            avg_score     = ("sentiment_score", "mean"),
            article_count = ("sentiment_score", "count"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily


# ============================================================
# CSV logging
# ============================================================

def load_alert_history() -> pd.DataFrame:
    """Load existing alert_history.csv or return empty DataFrame."""
    if os.path.exists(ALERT_HISTORY_PATH):
        try:
            return pd.read_csv(ALERT_HISTORY_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "timestamp", "alert_type", "sector",
        "severity", "value", "threshold", "message"
    ])


def save_alerts_to_csv(alerts: list[dict]):
    """
    Append newly triggered alerts to alert_history.csv.
    Each run only adds alerts that fired this time.
    """
    if not alerts:
        return

    ensure_data_dir()
    new_df = pd.DataFrame(alerts)

    if os.path.exists(ALERT_HISTORY_PATH):
        new_df.to_csv(ALERT_HISTORY_PATH, mode="a", header=False, index=False)
    else:
        new_df.to_csv(ALERT_HISTORY_PATH, mode="w", header=True, index=False)

    print(f"  💾 Logged {len(alerts)} alert(s) to {ALERT_HISTORY_PATH}")


# ============================================================
# Gmail email alert
# ============================================================

def send_email_alert(alerts: list[dict]):
    """
    Send a Gmail email summarising all HIGH severity alerts.
    Only fires if GMAIL_SENDER, GMAIL_APP_PASSWORD, GMAIL_RECEIVER
    are all configured in .env.

    Uses Python's built-in smtplib — no external library needed.
    """
    # Only send for HIGH severity alerts
    high_alerts = [a for a in alerts if a["severity"] == "HIGH"]
    if not high_alerts:
        return

    # Check credentials are configured
    if not all([GMAIL_SENDER, GMAIL_APP_PASSWORD, GMAIL_RECEIVER]):
        print("  ⚠️  Gmail not configured — skipping email alert")
        print("  👉  Set GMAIL_SENDER, GMAIL_APP_PASSWORD, GMAIL_RECEIVER in .env")
        return

    try:
        # ── Build email content ──────────────────────────
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subject = f"[Tech Pulse] 🚨 {len(high_alerts)} HIGH Alert(s) — {now_str}"

        # Plain text body
        body_lines = [
            "Tech Industry Pulse Tracker — Alert Notification",
            "=" * 50,
            f"Time     : {now_str}",
            f"Alerts   : {len(high_alerts)} HIGH severity",
            "",
        ]
        for i, alert in enumerate(high_alerts, 1):
            body_lines += [
                f"Alert {i}: {alert['alert_type'].replace('_', ' ').title()}",
                f"  Sector   : {alert['sector']}",
                f"  Severity : {alert['severity']}",
                f"  Value    : {alert['value']:+.4f}",
                f"  Threshold: {alert['threshold']:+.4f}",
                f"  Message  : {alert['message']}",
                "",
            ]
        body_lines += [
            "=" * 50,
            "Open your dashboard to see full details.",
            "Tech Industry Pulse Tracker",
        ]
        plain_body = "\n".join(body_lines)

        # HTML body — cleaner look in email clients
        html_rows = ""
        for alert in high_alerts:
            color = "#dc3545" if alert["severity"] == "HIGH" else "#fd7e14"
            html_rows += f"""
            <tr>
              <td style='padding:8px;border:1px solid #dee2e6'>{alert['sector']}</td>
              <td style='padding:8px;border:1px solid #dee2e6'>
                {alert['alert_type'].replace('_', ' ').title()}
              </td>
              <td style='padding:8px;border:1px solid #dee2e6;color:{color};font-weight:bold'>
                {alert['severity']}
              </td>
              <td style='padding:8px;border:1px solid #dee2e6'>{alert['value']:+.4f}</td>
              <td style='padding:8px;border:1px solid #dee2e6'>{alert['message']}</td>
            </tr>"""

        html_body = f"""
        <html><body style='font-family:Arial,sans-serif;max-width:700px;margin:auto'>
          <h2 style='color:#dc3545'>🚨 Tech Pulse Alert — {len(high_alerts)} HIGH Severity</h2>
          <p style='color:#6c757d'>{now_str}</p>
          <table style='border-collapse:collapse;width:100%;margin-top:16px'>
            <thead>
              <tr style='background:#f8f9fa'>
                <th style='padding:8px;border:1px solid #dee2e6;text-align:left'>Sector</th>
                <th style='padding:8px;border:1px solid #dee2e6;text-align:left'>Type</th>
                <th style='padding:8px;border:1px solid #dee2e6;text-align:left'>Severity</th>
                <th style='padding:8px;border:1px solid #dee2e6;text-align:left'>Value</th>
                <th style='padding:8px;border:1px solid #dee2e6;text-align:left'>Details</th>
              </tr>
            </thead>
            <tbody>{html_rows}</tbody>
          </table>
          <p style='margin-top:24px;color:#6c757d;font-size:12px'>
            Tech Industry Pulse Tracker — Open your dashboard for full details.
          </p>
        </body></html>"""

        # ── Build MIME message ───────────────────────────
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_SENDER
        msg["To"]      = GMAIL_RECEIVER
        msg.attach(MIMEText(plain_body, "plain"))
        msg.attach(MIMEText(html_body,  "html"))

        # ── Send via Gmail SMTP ──────────────────────────
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            # Gmail App Password may have spaces — strip them
            password = GMAIL_APP_PASSWORD.replace(" ", "")
            server.login(GMAIL_SENDER, password)
            server.sendmail(GMAIL_SENDER, GMAIL_RECEIVER, msg.as_string())

        print(f"  📧 Email alert sent to {GMAIL_RECEIVER}")

    except smtplib.SMTPAuthenticationError:
        print("  ❌ Gmail authentication failed")
        print("  👉 Check GMAIL_SENDER and GMAIL_APP_PASSWORD in .env")
        print("  👉 Make sure you used an App Password, not your real Gmail password")
    except smtplib.SMTPException as e:
        print(f"  ❌ Gmail SMTP error: {e}")
    except Exception as e:
        print(f"  ❌ Email sending error: {e}")


# ============================================================
# Main alert function
# ============================================================

def check_and_send_alerts() -> list[dict]:
    """
    Run all threshold checks across all sectors.
    Log triggered alerts to CSV.
    Send email for HIGH severity alerts.
    Return full alert list for Streamlit banners.
    """
    print("\n" + "=" * 50)
    print("  Checking alert thresholds...")
    print("=" * 50)

    # ── Load sentiment data ────────────────────────────
    if not os.path.exists(SENTIMENT_RESULTS_PATH):
        print(f"  ❌ {SENTIMENT_RESULTS_PATH} not found")
        print("  👉 Run: python main.py analyze  first")
        return []

    df = pd.read_csv(SENTIMENT_RESULTS_PATH, on_bad_lines="skip")
    if df.empty:
        print("  ❌ No sentiment data found")
        return []

    print(f"  📥 Loaded {len(df)} articles for alert checking")

    # ── Check all sectors ──────────────────────────────
    from config import SECTOR_KEYWORDS
    sectors_to_check = ["Overall"] + list(SECTOR_KEYWORDS.keys())

    all_alerts = []

    for sector in sectors_to_check:
        daily_df = get_daily_sentiment(df, sector)

        if daily_df.empty:
            continue

        # Run all 4 checks
        all_alerts.extend(check_negative_sentiment(daily_df, sector))
        all_alerts.extend(check_positive_surge(daily_df, sector))
        all_alerts.extend(check_volatility(daily_df, sector))
        all_alerts.extend(check_trend_reversal(daily_df, sector))

    # ── Results ────────────────────────────────────────
    if all_alerts:
        print(f"\n  🚨 {len(all_alerts)} alert(s) triggered:")
        for alert in all_alerts:
            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(alert["severity"], "⚪")
            print(f"  {icon} [{alert['severity']:<6}] {alert['alert_type']:<22} "
                  f"sector={alert['sector']}")

        # Log to CSV
        save_alerts_to_csv(all_alerts)

        # Email HIGH severity alerts
        send_email_alert(all_alerts)

    else:
        print("\n  ✅ No alerts triggered — all sectors within thresholds")

        # Log a status entry so the dashboard can show "last checked" time
        status_entry = make_alert(
            alert_type = "status_ok",
            sector     = "Overall",
            severity   = "LOW",
            value      = 0.0,
            threshold  = 0.0,
            message    = "✅ All sectors within normal thresholds.",
        )
        save_alerts_to_csv([status_entry])

    print("\n" + "=" * 50)
    print(f"  Alert check complete — {len(all_alerts)} alert(s) triggered")
    print("=" * 50 + "\n")

    return all_alerts


# ============================================================
# Dashboard helpers
# ============================================================

def get_alert_history(last_n_days: int = 30) -> pd.DataFrame:
    """
    Load alert_history.csv and return alerts from the last N days.
    Used by the Streamlit Alerts tab to show history table.
    """
    if not os.path.exists(ALERT_HISTORY_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(ALERT_HISTORY_PATH)
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])

        cutoff = datetime.now(timezone.utc) - timedelta(days=last_n_days)
        df = df[df["timestamp"] >= cutoff]

        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    except Exception as e:
        print(f"  ⚠️  Could not load alert history: {e}")
        return pd.DataFrame()


def get_active_alerts() -> list[dict]:
    """
    Return alerts triggered in the last 24 hours.
    Used by Streamlit Overview tab to show active banners.
    """
    df = get_alert_history(last_n_days=1)
    if df.empty:
        return []

    # Exclude status_ok entries from banners
    df = df[df["alert_type"] != "status_ok"]
    return df.to_dict("records")


def get_severity_color(severity: str) -> str:
    """Return a hex color for each severity level — used in dashboard."""
    return {
        "HIGH":   "#dc3545",  # red
        "MEDIUM": "#fd7e14",  # orange
        "LOW":    "#28a745",  # green
    }.get(severity, "#6c757d")


# ============================================================
# Run directly to test
# ============================================================

if __name__ == "__main__":
    triggered = check_and_send_alerts()

    print(f"Triggered alerts: {len(triggered)}")
    if triggered:
        print("\nAlert details:")
        for a in triggered:
            print(f"  [{a['severity']}] {a['alert_type']} — {a['sector']}")
            print(f"    {a['message']}")
            print()

    # Show recent history
    history = get_alert_history(last_n_days=7)
    if not history.empty:
        print(f"\nAlert history (last 7 days): {len(history)} entries")