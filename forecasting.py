# ============================================================
# forecasting.py — 7-day sentiment forecast using Prophet
# ============================================================
# Reads  : data/sentiment_results.csv
# Writes : data/forecast_results.csv
#
# What it does:
#   1. Loads sentiment_results.csv
#   2. Aggregates daily average sentiment score per sector
#   3. Trains a Prophet model on historical daily sentiment
#   4. Forecasts the next 7 days with confidence intervals
#   5. Saves forecast to forecast_results.csv
#
# Minimum data requirement:
#   Prophet needs at least 7 days of history to forecast.
#   If less data exists, a simple moving average fallback is used.
#
# Run directly to test:
#   python forecasting.py

import os
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    DATA_DIR,
    FORECAST_RESULTS_PATH,
    SENTIMENT_RESULTS_PATH,
    ensure_data_dir,
)

# Minimum days of history required for Prophet
MIN_DAYS_FOR_PROPHET = 7

# Number of days to forecast ahead
FORECAST_DAYS = 7

# Sectors to forecast individually + overall
from config import SECTOR_KEYWORDS, DEFAULT_SECTOR
ALL_SECTORS = list(SECTOR_KEYWORDS.keys()) + [DEFAULT_SECTOR, "Overall"]


# ============================================================
# Data preparation
# ============================================================

def load_and_aggregate(sector: str = "Overall") -> pd.DataFrame:
    if not os.path.exists(SENTIMENT_RESULTS_PATH):
        print(f"  ❌ {SENTIMENT_RESULTS_PATH} not found")
        return pd.DataFrame()

    df = pd.read_csv(SENTIMENT_RESULTS_PATH, on_bad_lines="skip")
    if df.empty:
        return pd.DataFrame()

    # Robust date parsing — handles Z suffix, +00:00, plain dates, all formats
    df["published_at"] = pd.to_datetime(
        df["published_at"]
        .astype(str)
        .str.replace("Z", "+00:00", regex=False)
        .str.strip(),
        errors="coerce",
        utc=True,
    )

    # Drop rows where date completely failed to parse
    before = len(df)
    df = df.dropna(subset=["published_at"])
    if len(df) < before:
        print(f"  ⚠️  Dropped {before - len(df)} rows with unparseable dates")

    df["date"] = df["published_at"].dt.date

    if sector != "Overall":
        df = df[df["sector"] == sector]
        if df.empty:
            return pd.DataFrame()

    daily = (
        df.groupby("date")
        .agg(
            y             = ("sentiment_score", "mean"),
            article_count = ("sentiment_score", "count"),
            std           = ("sentiment_score", "std"),
        )
        .reset_index()
    )

    daily = daily.rename(columns={"date": "ds"})
    daily["ds"]  = pd.to_datetime(daily["ds"])
    daily["std"] = daily["std"].fillna(0.0)
    daily = daily.sort_values("ds").reset_index(drop=True)

    return daily


# ============================================================
# Prophet forecasting
# ============================================================

def forecast_with_prophet(daily_df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """
    Train Prophet on daily sentiment data and forecast 7 days ahead.

    Prophet expects:
        ds — datetime column
        y  — value to forecast

    Returns a DataFrame with columns:
        ds, forecast_date, sentiment_forecast,
        lower_bound, upper_bound, sector, method
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("  ❌ prophet not installed — run: pip install prophet")
        return pd.DataFrame()

    try:
        print(f"  🤖 Training Prophet for sector: {sector}")

        # Prophet model configuration
        model = Prophet(
            daily_seasonality   = False,  # Not enough data for daily patterns
            weekly_seasonality  = True,   # Tech news has weekly patterns
            yearly_seasonality  = False,  # Not enough data for yearly
            changepoint_prior_scale   = 0.05,  # Flexibility of trend changes
            seasonality_prior_scale   = 10.0,  # Strength of seasonality
            interval_width            = 0.80,  # 80% confidence interval
            uncertainty_samples       = 200,   # Faster than default 1000
        )

        # Fit on historical data
        prophet_df = daily_df[["ds", "y"]].copy()
        model.fit(prophet_df)

        # Make future dataframe — historical + 7 days ahead
        future = model.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
        forecast = model.predict(future)

        # Extract only the future 7 days (not the historical fitted values)
        last_historical_date = daily_df["ds"].max()
        future_only = forecast[forecast["ds"] > last_historical_date].copy()

        if future_only.empty:
            print(f"  ⚠️  No future dates generated for {sector}")
            return pd.DataFrame()

        # Build clean results DataFrame
        result = pd.DataFrame({
            "forecast_date":      future_only["ds"].dt.strftime("%Y-%m-%d"),
            "sentiment_forecast": future_only["yhat"].round(4),
            "lower_bound":        future_only["yhat_lower"].round(4),
            "upper_bound":        future_only["yhat_upper"].round(4),
            "sector":             sector,
            "method":             "prophet",
            "generated_at":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Clip forecast values to valid sentiment range [-1, 1]
        result["sentiment_forecast"] = result["sentiment_forecast"].clip(-1.0, 1.0)
        result["lower_bound"]        = result["lower_bound"].clip(-1.0, 1.0)
        result["upper_bound"]        = result["upper_bound"].clip(-1.0, 1.0)

        print(f"  ✅ Prophet forecast complete for {sector}")
        return result

    except Exception as e:
        print(f"  ❌ Prophet error for {sector}: {e}")
        return pd.DataFrame()


# ============================================================
# Fallback: simple moving average forecast
# ============================================================

def forecast_with_moving_average(daily_df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """
    Simple fallback when insufficient data for Prophet.
    Uses the mean of available data as flat forecast.
    Confidence interval = ± 1 standard deviation.

    Used when fewer than 7 days of history exist.
    """
    print(f"  📈 Using moving average fallback for {sector}")

    if daily_df.empty:
        return pd.DataFrame()

    # Calculate baseline from available data
    mean_sentiment = daily_df["y"].mean()
    std_sentiment  = daily_df["y"].std() if len(daily_df) > 1 else 0.15
    std_sentiment  = std_sentiment if not np.isnan(std_sentiment) else 0.15

    last_date = daily_df["ds"].max()
    rows = []

    for i in range(1, FORECAST_DAYS + 1):
        forecast_date = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
        rows.append({
            "forecast_date":      forecast_date,
            "sentiment_forecast": round(float(mean_sentiment), 4),
            "lower_bound":        round(float(mean_sentiment - std_sentiment), 4),
            "upper_bound":        round(float(mean_sentiment + std_sentiment), 4),
            "sector":             sector,
            "method":             "moving_average",
            "generated_at":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })

    result = pd.DataFrame(rows)

    # Clip to valid range
    for col in ["sentiment_forecast", "lower_bound", "upper_bound"]:
        result[col] = result[col].clip(-1.0, 1.0)

    return result


# ============================================================
# Main forecast function
# ============================================================

def forecast() -> pd.DataFrame:
    """
    Generate 7-day sentiment forecasts for Overall + each sector.
    Saves all forecasts to forecast_results.csv.
    Returns the complete forecast DataFrame.
    """
    print("\n" + "=" * 50)
    print("  Generating sentiment forecasts...")
    print("=" * 50)

    ensure_data_dir()
    all_forecasts = []

    # Forecast for Overall + each individual sector
    sectors_to_forecast = ["Overall"] + list(SECTOR_KEYWORDS.keys())

    for sector in sectors_to_forecast:
        print(f"\n  📊 Processing: {sector}")

        # Load and aggregate daily data for this sector
        daily_df = load_and_aggregate(sector)

        if daily_df.empty:
            print(f"  ⚠️  No data found for sector: {sector} — skipping")
            continue

        days_available = len(daily_df)
        print(f"  📅 Days of history: {days_available}")

        # Choose forecasting method based on data availability
        if days_available >= MIN_DAYS_FOR_PROPHET:
            forecast_df = forecast_with_prophet(daily_df, sector)

            # Fall back to moving average if Prophet fails
            if forecast_df.empty:
                forecast_df = forecast_with_moving_average(daily_df, sector)
        else:
            print(f"  ⚠️  Only {days_available} day(s) of data "
                  f"(need {MIN_DAYS_FOR_PROPHET} for Prophet) — using fallback")
            forecast_df = forecast_with_moving_average(daily_df, sector)

        if not forecast_df.empty:
            all_forecasts.append(forecast_df)

    # Combine all sector forecasts
    if not all_forecasts:
        print("\n  ❌ No forecasts generated")
        return pd.DataFrame()

    combined = pd.concat(all_forecasts, ignore_index=True)

    # Save to CSV
    combined.to_csv(FORECAST_RESULTS_PATH, index=False)

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Forecast summary")
    print("=" * 50)

    overall = combined[combined["sector"] == "Overall"]
    if not overall.empty:
        print(f"\n  7-day Overall sentiment forecast:")
        print(f"  {'Date':<14} {'Forecast':>9} {'Lower':>8} {'Upper':>8}  {'Method'}")
        print(f"  {'-'*14} {'-'*9} {'-'*8} {'-'*8}  {'-'*15}")
        for _, row in overall.iterrows():
            trend = "📈" if row["sentiment_forecast"] > 0 else "📉"
            print(
                f"  {row['forecast_date']:<14} "
                f"{row['sentiment_forecast']:>+.4f}   "
                f"{row['lower_bound']:>+.4f}  "
                f"{row['upper_bound']:>+.4f}  "
                f"{row['method']}  {trend}"
            )

        avg_forecast = overall["sentiment_forecast"].mean()
        direction = "improving 📈" if avg_forecast > 0 else "declining 📉"
        print(f"\n  Overall 7-day outlook : {direction}")
        print(f"  Average forecast score: {avg_forecast:+.4f}")

    print(f"\n  Sectors forecasted : {combined['sector'].nunique()}")
    print(f"  Total rows saved   : {len(combined)}")
    print(f"  💾 Saved to        : {FORECAST_RESULTS_PATH}")
    print("=" * 50 + "\n")

    return combined


# ============================================================
# Utility — get latest forecast for dashboard
# ============================================================

def get_latest_forecast(sector: str = "Overall") -> pd.DataFrame:
    """
    Load forecast_results.csv and return the forecast for a given sector.
    Used by the Streamlit dashboard to display forecast charts.
    """
    if not os.path.exists(FORECAST_RESULTS_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(FORECAST_RESULTS_PATH)
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        return df[df["sector"] == sector].copy()
    except Exception as e:
        print(f"  ⚠️  Could not load forecast: {e}")
        return pd.DataFrame()


def get_forecast_interpretation(sector: str = "Overall") -> str:
    """
    Return a human-readable interpretation of the forecast.
    Used in the dashboard below the forecast chart.
    """
    df = get_latest_forecast(sector)
    if df.empty:
        return "No forecast available yet."

    avg   = df["sentiment_forecast"].mean()
    first = df["sentiment_forecast"].iloc[0]
    last  = df["sentiment_forecast"].iloc[-1]
    trend = last - first

    if avg > 0.3:
        mood = "strongly positive"
    elif avg > 0.1:
        mood = "mildly positive"
    elif avg > -0.1:
        mood = "roughly neutral"
    elif avg > -0.3:
        mood = "mildly negative"
    else:
        mood = "strongly negative"

    if abs(trend) < 0.05:
        direction = "remaining stable"
    elif trend > 0:
        direction = "gradually improving"
    else:
        direction = "gradually declining"

    method = df["method"].iloc[0]
    method_note = (
        "based on Prophet time-series forecasting"
        if method == "prophet"
        else "based on moving average (more data needed for Prophet)"
    )

    return (
        f"Sentiment for {sector} is forecast to be {mood} "
        f"over the next 7 days, {direction}. "
        f"Forecast is {method_note}."
    )


# ============================================================
# Run directly to test
# ============================================================

if __name__ == "__main__":
    df = forecast()
    if not df.empty:
        print("✅ forecasting.py working correctly\n")
        print("Forecast interpretation:")
        print(f"  {get_forecast_interpretation('Overall')}")
    else:
        print("❌ Forecasting failed — check sentiment_results.csv exists")