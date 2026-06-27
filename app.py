# ============================================================
# app.py — Tech Industry Pulse Tracker Dashboard
# ============================================================
# Launch with: streamlit run app.py

import os
import warnings
from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

from config import (
    ALERT_HISTORY_PATH,
    DATA_DIR,
    FORECAST_RESULTS_PATH,
    SECTOR_KEYWORDS,
    SENTIMENT_RESULTS_PATH,
    ensure_data_dir,
)

# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Tech Industry Pulse Tracker",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 12px;
}
.stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; }
#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sector → stock ticker mapping
# One representative ticker per sector — simple and clean
# ============================================================

SECTOR_TICKERS = {
    "AI/ML":          ("NVDA",  "Nvidia"),
    "Cloud":          ("MSFT",  "Microsoft"),
    "Cybersecurity":  ("HACK",  "HACK ETF"),
    "Startups":       ("QQQ",   "NASDAQ 100"),
    "Semiconductors": ("SOXX",  "SOXX ETF"),
    "Big Tech":       ("QQQ",   "NASDAQ 100"),
    "Overall":        ("^IXIC", "NASDAQ"),
}


# ============================================================
# Data loaders — cached
# ============================================================

# Supabase storage helper — used when running on Streamlit Cloud.
# Falls back to local files automatically when not configured.
from supabase_storage import supabase_configured, read_csv_from_supabase, read_text_from_supabase

@st.cache_data(ttl=30)
def load_sentiment() -> pd.DataFrame:
    try:
        if supabase_configured():
            df = read_csv_from_supabase("sentiment_results.csv")
        else:
            if not os.path.exists(SENTIMENT_RESULTS_PATH):
                return pd.DataFrame()
            df = pd.read_csv(SENTIMENT_RESULTS_PATH, on_bad_lines="skip")
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df = df.dropna(subset=["published_at"])
        df["date"] = df["published_at"].dt.date
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_forecast() -> pd.DataFrame:
    try:
        if supabase_configured():
            df = read_csv_from_supabase("forecast_results.csv")
        else:
            if not os.path.exists(FORECAST_RESULTS_PATH):
                return pd.DataFrame()
            df = pd.read_csv(FORECAST_RESULTS_PATH)
        df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_alerts() -> pd.DataFrame:
    try:
        if supabase_configured():
            df = read_csv_from_supabase("alert_history.csv")
        else:
            if not os.path.exists(ALERT_HISTORY_PATH):
                return pd.DataFrame()
            df = pd.read_csv(ALERT_HISTORY_PATH, on_bad_lines="skip")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df.dropna(subset=["timestamp"]).sort_values(
            "timestamp", ascending=False
        ).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_stock(ticker: str) -> pd.DataFrame:
    """
    Load 30-day price history for a stock ticker using yfinance.
    Returns DataFrame with columns: date, close, pct_change.
    Cached 10 minutes — stock prices don't need constant refreshing.
    """
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="60d").reset_index()
        if hist.empty:
            return pd.DataFrame()
        hist["Date"] = pd.to_datetime(hist["Date"], utc=True).dt.date
        hist = hist[["Date", "Close"]].rename(
            columns={"Date": "date", "Close": "close"}
        )
        # Normalize to % change from first day — makes overlay readable
        # regardless of whether the stock is $10 or $500
        first = hist["close"].iloc[0]
        hist["pct_change"] = ((hist["close"] - first) / first * 100).round(2)
        return hist
    except Exception:
        return pd.DataFrame()


# ============================================================
# Helpers
# ============================================================

def get_daily(df: pd.DataFrame, sector: str = "Overall") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    filtered = df if sector == "Overall" else df[df["sector"] == sector]
    if filtered.empty:
        return pd.DataFrame()
    daily = (
        filtered.groupby("date")
        .agg(
            avg_score    =("sentiment_score", "mean"),
            article_count=("sentiment_score", "count"),
            positive_pct =("sentiment_label", lambda x: (x=="positive").mean()*100),
            negative_pct =("sentiment_label", lambda x: (x=="negative").mean()*100),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def score_label(score: float) -> str:
    if score >=  0.20: return "Positive 😊"
    if score >=  0.05: return "Mildly Positive 🙂"
    if score >= -0.05: return "Neutral 😐"
    if score >= -0.20: return "Mildly Negative 🙁"
    return "Negative 😟"


def score_color(score: float) -> str:
    if score >=  0.20: return "#28a745"
    if score >=  0.0:  return "#85c785"
    if score >= -0.20: return "#fd7e14"
    return "#dc3545"


CHART_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
)

SECTOR_COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c",
    "#d62728","#9467bd","#8c564b",
]


def chart_caption(text: str):
    """Render a styled interpretation caption below a chart."""
    st.markdown(
        f'<div style="background:#f0f4ff;border-left:3px solid #4a7edd;'
        f'padding:8px 12px;border-radius:4px;font-size:0.85rem;color:#444;margin-top:-8px">'
        f'ℹ️ {text}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Pipeline runner — called when Refresh Data is clicked
# ============================================================

def run_pipeline():
    """
    Runs the full pipeline.
    Uses a clean container so progress renders without conflicting with
    Streamlit's rerun mechanism. All steps are wrapped so a failure in
    one step (e.g. forecast needs 7 days, alerts needs email) never
    blocks the rerun that loads the dashboard.
    """
    st.cache_data.clear()

    st.title("📡 Tech Industry Pulse Tracker")
    st.subheader("⏳ Running pipeline — please wait...")
    st.caption("Do not close this tab. This takes 2–5 minutes on first run.")

    status = st.status("Starting pipeline...", expanded=True)

    def run_step(label, fn):
        try:
            status.write(f"▶ {label}")
            fn()
            status.write(f"✅ {label}")
            return True
        except Exception as e:
            status.write(f"⚠️ {label} — {e}")
            return False

    run_step("📡 Collecting articles from all sources", _collect)
    run_step("🧹 Cleaning and preprocessing articles",  _preprocess)
    run_step("🧠 Running sentiment analysis",           _analyze)
    run_step("🔮 Generating 7-day forecast",            _forecast)
    run_step("🚨 Checking alert thresholds",            _alerts)

    status.update(label="✅ Pipeline complete — loading dashboard...", state="complete")

    # Show a note explaining that "no change" is normal when APIs return same stories
    st.info(
        "ℹ️ If the dashboard looks unchanged, it means the news APIs returned "
        "the same articles as last time (no new stories published since your last update). "
        "This is normal — try again in a few hours when fresh articles are available."
    )

    # Reset date widget state so sidebar picks up new date range
    for key in ["date_from", "date_to"]:
        if key in st.session_state:
            del st.session_state[key]

    # Clear cache AFTER writing so loaders get fresh CSVs
    st.cache_data.clear()

    import time
    time.sleep(2)
    st.rerun()


def _collect():
    from collector import collect_all
    collect_all()

def _preprocess():
    from preprocessor import preprocess
    preprocess()

def _analyze():
    from analyzer import analyze
    analyze()

def _forecast():
    from forecasting import forecast
    forecast()

def _alerts():
    from alerts import check_and_send_alerts
    check_and_send_alerts()


# ============================================================
# Report generator
# ============================================================

def generate_report(df: pd.DataFrame, date_range, sector: str) -> bytes:
    """
    Build a plain-text summary report and return as UTF-8 bytes for download.
    Covers: summary stats, sector breakdown, top articles, forecast outlook.
    """
    from io import StringIO
    d_from, d_to = date_range
    mask = (
        (pd.to_datetime(df["date"]).dt.date >= d_from) &
        (pd.to_datetime(df["date"]).dt.date <= d_to)
    )
    fdf = df[mask].copy()

    buf = StringIO()
    w = buf.write

    w("=" * 60 + "\n")
    w("  TECH INDUSTRY PULSE TRACKER — SENTIMENT REPORT\n")
    w("=" * 60 + "\n")
    w(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    w(f"  Period    : {d_from} to {d_to}\n")
    w(f"  Sector    : {sector}\n")
    w("=" * 60 + "\n\n")

    if fdf.empty:
        w("No data available for this period.\n")
        return buf.getvalue().encode("utf-8")

    # ── Overall summary ──
    total     = len(fdf)
    pos       = (fdf["sentiment_label"] == "positive").sum()
    neg       = (fdf["sentiment_label"] == "negative").sum()
    neu       = (fdf["sentiment_label"] == "neutral").sum()
    avg_score = fdf["sentiment_score"].mean()

    w("SUMMARY\n")
    w("-" * 40 + "\n")
    w(f"  Total articles analysed : {total}\n")
    w(f"  Positive                : {pos} ({pos/total*100:.1f}%)\n")
    w(f"  Negative                : {neg} ({neg/total*100:.1f}%)\n")
    w(f"  Neutral                 : {neu} ({neu/total*100:.1f}%)\n")
    w(f"  Average sentiment score : {avg_score:+.4f}  "
      f"({'Positive' if avg_score > 0.05 else 'Negative' if avg_score < -0.05 else 'Neutral'})\n\n")

    # ── Sector breakdown ──
    w("SENTIMENT BY SECTOR\n")
    w("-" * 40 + "\n")
    sector_stats = (
        fdf.groupby("sector")
        .agg(count=("sentiment_score","count"), avg=("sentiment_score","mean"))
        .sort_values("avg", ascending=False)
    )
    for sec, row in sector_stats.iterrows():
        bar = "█" * int(abs(row["avg"]) * 10)
        sign = "+" if row["avg"] >= 0 else ""
        w(f"  {sec:<20} {sign}{row['avg']:.4f}  {bar}  ({int(row['count'])} articles)\n")
    w("\n")

    # ── Top 5 positive articles ──
    w("TOP 5 MOST POSITIVE ARTICLES\n")
    w("-" * 40 + "\n")
    top_pos = fdf.nlargest(5, "sentiment_score")
    for _, r in top_pos.iterrows():
        w(f"  [{r['sentiment_score']:+.3f}] {r['title'][:70]}\n")
        w(f"           Source: {r['source']} | {str(r['published_at'])[:10]}\n")
    w("\n")

    # ── Top 5 negative articles ──
    w("TOP 5 MOST NEGATIVE ARTICLES\n")
    w("-" * 40 + "\n")
    top_neg = fdf.nsmallest(5, "sentiment_score")
    for _, r in top_neg.iterrows():
        w(f"  [{r['sentiment_score']:+.3f}] {r['title'][:70]}\n")
        w(f"           Source: {r['source']} | {str(r['published_at'])[:10]}\n")
    w("\n")

    # ── Forecast ──
    if os.path.exists(FORECAST_RESULTS_PATH):
        try:
            fc = pd.read_csv(FORECAST_RESULTS_PATH)
            fc_sector = "Overall" if sector == "Overall" else sector
            fc_filt = fc[fc["sector"] == fc_sector].head(7)
            if not fc_filt.empty:
                w("7-DAY FORECAST\n")
                w("-" * 40 + "\n")
                method = fc_filt["method"].iloc[0]
                w(f"  Method: {method}\n")
                for _, r in fc_filt.iterrows():
                    direction = "↑" if r["sentiment_forecast"] > 0 else "↓"
                    w(f"  {r['forecast_date']}  {r['sentiment_forecast']:+.4f} "
                      f"  [{r['lower_bound']:+.4f}, {r['upper_bound']:+.4f}]  {direction}\n")
                w("\n")
        except Exception:
            pass

    w("=" * 60 + "\n")
    w("  END OF REPORT\n")
    w("=" * 60 + "\n")

    return buf.getvalue().encode("utf-8")


# ============================================================
# Sidebar
# ============================================================

def render_sidebar(df: pd.DataFrame):
    st.sidebar.title("📡 Tech Pulse Tracker")
    st.sidebar.caption("Real-time tech industry sentiment")
    st.sidebar.divider()

    # Last updated indicator
    if supabase_configured():
        last_run = read_text_from_supabase("last_collected.txt") or "never"
    else:
        marker = os.path.join(DATA_DIR, "last_collected.txt")
        last_run = open(marker).read().strip() if os.path.exists(marker) else "never"
    st.sidebar.caption(f"🕐 Last collected: {last_run}")

    st.sidebar.divider()

    # Refresh button — label depends on whether data already exists
    has_data = os.path.exists(SENTIMENT_RESULTS_PATH)
    btn_label = "🔄 Update Data" if has_data else "🚀 Collect & Analyze"
    if st.sidebar.button(btn_label, use_container_width=True, type="primary"):
        st.session_state["refresh"] = True
        st.rerun()

    st.sidebar.divider()

    # Date range
    st.sidebar.subheader("📅 Date range")
    if not df.empty:
        min_d = pd.to_datetime(df["date"]).min().date()
        max_d = pd.to_datetime(df["date"]).max().date()
    else:
        max_d = datetime.now().date()
        min_d = max_d - timedelta(days=30)

    default_from = max(min_d, max_d - timedelta(days=30))

    date_from = st.sidebar.date_input(
        "From", value=default_from, min_value=min_d, max_value=max_d, key="date_from"
    )
    date_to = st.sidebar.date_input(
        "To", value=max_d, min_value=min_d, max_value=max_d, key="date_to"
    )
    st.sidebar.caption(
        "📅 **Date Range** filters all charts and metrics to show only articles "
        "published within the selected window. "
        "Narrow it to see recent trends; widen it to spot long-term patterns. "
        "Does not affect the 7-day forecast or alert history."
    )

    # Sector
    st.sidebar.subheader("🏷️ Sector")
    sectors = ["Overall"] + list(SECTOR_KEYWORDS.keys())
    sector  = st.sidebar.selectbox("Select sector", sectors)

    st.sidebar.divider()

    # ── Download report ──
    st.sidebar.subheader("📥 Download Report")
    if not df.empty:
        report_bytes = generate_report(df, (date_from, date_to), sector)
        filename = (
            f"tech_pulse_report_{sector.lower().replace('/','_').replace(' ','_')}"
            f"_{date_from}_{date_to}.txt"
        )
        st.sidebar.download_button(
            label="📄 Download Summary Report",
            data=report_bytes,
            file_name=filename,
            mime="text/plain",
            use_container_width=True,
        )
        st.sidebar.caption(
            "Downloads a plain-text report covering sentiment summary, "
            "sector breakdown, top articles, and 7-day forecast."
        )
    else:
        st.sidebar.info("Run the pipeline first to enable report download.")

    st.sidebar.divider()

    # File status
    st.sidebar.subheader("📁 Data files")
    for label, path in [
        ("raw_articles.csv",      "data/raw_articles.csv"),
        ("sentiment_results.csv", SENTIMENT_RESULTS_PATH),
        ("forecast_results.csv",  FORECAST_RESULTS_PATH),
        ("alert_history.csv",     ALERT_HISTORY_PATH),
    ]:
        if os.path.exists(path):
            kb = os.path.getsize(path) // 1024
            st.sidebar.markdown(f"✅ `{label}` ({kb} KB)")
        else:
            st.sidebar.markdown(f"❌ `{label}`")

    st.sidebar.divider()
    st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    return (date_from, date_to), sector


# ============================================================
# Tab 1 — Overview
# ============================================================

def tab_overview(df: pd.DataFrame, date_range, sector: str):
    if df.empty:
        st.warning("No data yet. Click **🔄 Refresh Data** in the sidebar.")
        return

    d_from, d_to = date_range
    mask = (
        (pd.to_datetime(df["date"]).dt.date >= d_from) &
        (pd.to_datetime(df["date"]).dt.date <= d_to)
    )
    fdf = df[mask].copy()
    if sector != "Overall":
        fdf = fdf[fdf["sector"] == sector]
    if fdf.empty:
        st.info("No articles in selected date range / sector.")
        return

    # Active alert banner — slim summary only, full detail in Alerts tab
    alert_df = load_alerts()
    if not alert_df.empty:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        active = alert_df[
            (alert_df["timestamp"] >= cutoff) &
            (alert_df["alert_type"] != "status_ok")
        ]
        if not active.empty:
            high   = (active["severity"] == "HIGH").sum()
            medium = (active["severity"] == "MEDIUM").sum()
            low    = (active["severity"] == "LOW").sum()
            parts  = []
            if high:   parts.append(f"🔴 {high} HIGH")
            if medium: parts.append(f"🟡 {medium} MEDIUM")
            if low:    parts.append(f"🟢 {low} LOW")
            st.warning(
                f"**🚨 {len(active)} active alert(s) in the last 24 hours:** "
                f"{' · '.join(parts)} — see the **Alerts tab** for details."
            )
            st.divider()

    # KPI cards
    avg   = fdf["sentiment_score"].mean()
    total = len(fdf)
    pos   = (fdf["sentiment_label"] == "positive").mean() * 100
    neg   = (fdf["sentiment_label"] == "negative").mean() * 100

    today     = datetime.now().date()
    yesterday = today - timedelta(days=1)
    t_avg = fdf[pd.to_datetime(fdf["date"]).dt.date == today]["sentiment_score"].mean()
    y_avg = fdf[pd.to_datetime(fdf["date"]).dt.date == yesterday]["sentiment_score"].mean()
    delta = f"{t_avg - y_avg:+.3f} vs yesterday" if pd.notna(t_avg) and pd.notna(y_avg) else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Sentiment",    f"{avg:+.3f}",  delta=score_label(avg))
    c2.metric("Articles Analyzed",    f"{total:,}",   delta=f"{fdf['date'].nunique()} days")
    c3.metric("Positive %",           f"{pos:.1f}%",  delta=f"-{neg:.1f}% negative", delta_color="inverse")
    c4.metric("Today vs Yesterday",   f"{t_avg:+.3f}" if pd.notna(t_avg) else "—", delta=delta)

    st.divider()

    # Sector bar + pie
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Sentiment by Sector")
        sec_s = (
            fdf.groupby("sector")["sentiment_score"]
            .mean().sort_values().reset_index()
        )
        sec_s.columns = ["Sector", "Score"]
        fig = go.Figure(go.Bar(
            x=sec_s["Score"], y=sec_s["Sector"], orientation="h",
            marker_color=[score_color(s) for s in sec_s["Score"]],
            text=sec_s["Score"].apply(lambda x: f"{x:+.3f}"),
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(**CHART_LAYOUT, height=320,
                          xaxis_range=[-1.1, 1.1],
                          xaxis_title="Average Sentiment Score")
        st.plotly_chart(fig, use_container_width=True, key="overview_sector_bar")
        chart_caption(
            "Each bar shows the average sentiment score for that sector, "
            "ranging from -1 (very negative) to +1 (very positive). "
            "Green bars indicate positive market mood; red bars signal concern or pessimism. "
            "Scores near 0 are neutral. Use this to quickly spot which sectors are thriving or struggling in the news."
        )

    with col_r:
        st.subheader("Label Distribution")
        lc = fdf["sentiment_label"].value_counts().reset_index()
        lc.columns = ["Label", "Count"]
        fig2 = px.pie(lc, names="Label", values="Count", hole=0.45,
                      color="Label",
                      color_discrete_map={
                          "positive": "#28a745",
                          "neutral":  "#6c757d",
                          "negative": "#dc3545",
                      })
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True, key="overview_label_pie")
        chart_caption(
            "This donut chart shows the overall split of article sentiment across your selected date range. "
            "A dominant green (positive) slice suggests bullish news coverage; "
            "a growing red (negative) slice may signal market headwinds or controversy worth watching."
        )

    st.divider()

    # Article search + table
    st.subheader("📰 Recent Articles")
    search = st.text_input("🔍 Search by keyword", placeholder="e.g. OpenAI, layoffs, chip")
    recent = fdf.sort_values("published_at", ascending=False)
    if search.strip():
        mask_s = (
            recent["title"].str.contains(search, case=False, na=False) |
            recent["keywords"].str.contains(search, case=False, na=False)
        )
        recent = recent[mask_s]

    show_cols = [c for c in
                 ["published_at","title","source","sector",
                  "sentiment_label","sentiment_score","keywords"]
                 if c in recent.columns]

    def color_row(row):
        c = {"positive":"#d4edda","negative":"#f8d7da","neutral":"#e2e3e5"}.get(
            row.get("sentiment_label",""), "")
        return [f"background-color:{c}" if col == "sentiment_label" else "" for col in row.index]

    st.dataframe(
        recent[show_cols].head(25)
        .style.apply(color_row, axis=1)
        .format({"sentiment_score": "{:+.3f}"}),
        use_container_width=True, height=380, hide_index=True,
    )


# ============================================================
# Tab 2 — Trends  (with stock overlay)
# ============================================================

def tab_trends(df: pd.DataFrame, date_range, sector: str):
    if df.empty:
        st.warning("No data yet — click Refresh Data.")
        return

    d_from, d_to = date_range
    mask = (
        (pd.to_datetime(df["date"]).dt.date >= d_from) &
        (pd.to_datetime(df["date"]).dt.date <= d_to)
    )
    fdf = df[mask].copy()

    # ── Sentiment time-series ──────────────────────────────
    st.subheader("Daily Sentiment Over Time")
    fig = go.Figure()
    to_plot = list(SECTOR_KEYWORDS.keys()) if sector == "Overall" else [sector]

    for i, sec in enumerate(to_plot):
        daily = get_daily(fdf, sec)
        if daily.empty:
            continue
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["avg_score"],
            mode="lines+markers", name=sec,
            line=dict(color=SECTOR_COLORS[i % len(SECTOR_COLORS)], width=2),
            marker=dict(size=5),
            customdata=daily["article_count"],
            hovertemplate=(
                f"<b>{sec}</b><br>Date: %{{x}}<br>"
                "Score: %{y:+.3f}<br>Articles: %{customdata}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  opacity=0.4, annotation_text="Neutral")
    fig.update_layout(**CHART_LAYOUT, height=400,
                      yaxis_range=[-1, 1],
                      yaxis_title="Sentiment Score",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True, key="trends_daily_sentiment")
    chart_caption(
        "This chart shows how the average daily sentiment for each sector changes over time. "
        "A rising line means news coverage is becoming more positive; a falling line signals growing negativity. "
        "Sharp dips often correlate with major news events (layoffs, breaches, regulatory action). "
        "The dashed grey line at 0 is the neutral baseline — anything above it is net-positive coverage."
    )

    st.divider()

    # ── Stock price overlay ────────────────────────────────
    st.subheader("📈 Sentiment vs Stock Price")

    # Sector selector for stock — defaults to current sidebar sector
    ticker_sector = st.selectbox(
        "Select sector for stock comparison",
        list(SECTOR_TICKERS.keys()),
        index=list(SECTOR_TICKERS.keys()).index(sector)
              if sector in SECTOR_TICKERS else 0,
        key="stock_sector_select",
    )

    ticker, ticker_name = SECTOR_TICKERS[ticker_sector]
    daily_sentiment     = get_daily(fdf, ticker_sector)
    stock_df            = load_stock(ticker)

    if not daily_sentiment.empty and not stock_df.empty:
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        merged = pd.merge(daily_sentiment, stock_df, on="date", how="inner")

        if not merged.empty:
            fig2 = go.Figure()

            # Sentiment — left Y axis (absolute score)
            fig2.add_trace(go.Scatter(
                x=merged["date"], y=merged["avg_score"],
                name=f"{ticker_sector} Sentiment",
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                yaxis="y1",
                hovertemplate="Date: %{x}<br>Sentiment: %{y:+.3f}<extra></extra>",
            ))

            # Stock % change — right Y axis (normalized so it overlays cleanly)
            fig2.add_trace(go.Scatter(
                x=merged["date"], y=merged["pct_change"],
                name=f"{ticker_name} ({ticker}) % change",
                mode="lines",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
                yaxis="y2",
                hovertemplate="Date: %{x}<br>Stock: %{y:+.2f}%<extra></extra>",
            ))

            fig2.update_layout(
                **CHART_LAYOUT, height=400,
                yaxis=dict(
                    title=dict(text="Sentiment Score", font=dict(color="#1f77b4")),
                    range=[-1, 1],
                    tickfont=dict(color="#1f77b4"),
                ),
                yaxis2=dict(
                    title=dict(text=f"{ticker} Price Change (%)", font=dict(color="#ff7f0e")),
                    overlaying="y", side="right",
                    tickfont=dict(color="#ff7f0e"),
                    ticksuffix="%",
                ),
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig2, use_container_width=True, key="stock_overlay_chart")
            chart_caption(
                f"Blue line = daily {ticker_sector} sentiment score (left axis, -1 to +1). "
                f"Orange dotted = {ticker_name} ({ticker}) price % change from 60 days ago (right axis). "
                "When both lines move together, news sentiment may be influencing price — or reacting to the same events. "
                "Divergence (sentiment rising while price falls, or vice versa) can be an early signal worth watching."
            )

            # Stock / index definition box
            definitions = {
                "^IXIC": ("NASDAQ Composite Index",
                          "Tracks over 3,000 stocks listed on the NASDAQ exchange, "
                          "heavily weighted toward technology and growth companies. "
                          "It is one of the most widely followed indicators of tech sector health. "
                          "A rising NASDAQ generally signals investor confidence in tech; "
                          "a falling NASDAQ often reflects risk-off sentiment or macro headwinds."),
                "QQQ":   ("NASDAQ 100 ETF (QQQ)",
                          "Tracks the 100 largest non-financial companies on NASDAQ, "
                          "including Apple, Microsoft, Nvidia, Google, and Meta. "
                          "QQQ is widely used as a proxy for large-cap tech performance. "
                          "It tends to move more sharply than the broader market during tech-driven rallies or sell-offs."),
                "HACK":  ("HACK ETF — Cybersecurity Index",
                          "Tracks a basket of companies in the cybersecurity industry: "
                          "firewall vendors, endpoint security, threat intelligence, and cloud security firms. "
                          "It rises when cybersecurity spending increases (often after major breaches or new regulations) "
                          "and falls when the broader tech sector pulls back."),
                "SOXX":  ("SOXX ETF — Semiconductor Index",
                          "Tracks 30 of the largest US-listed semiconductor companies including Nvidia, AMD, Intel, TSMC, and Qualcomm. "
                          "Semiconductors are highly cyclical — they boom during AI/data center buildouts and fall sharply during inventory corrections. "
                          "SOXX is one of the most volatile tech ETFs and closely follows GPU and chip demand trends."),
            }
            defn = definitions.get(ticker, None)
            if defn:
                title, body = defn
                st.markdown(
                    f'<div style="background:#f0f4ff;border-left:3px solid #4a7edd;'
                    f'padding:10px 14px;border-radius:6px;margin-top:8px">'
                    f'<strong>📖 What is {title}?</strong><br>'
                    f'<span style="font-size:0.88rem;color:#444">{body}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Not enough overlapping dates between sentiment and stock data yet.")
    else:
        if stock_df.empty:
            st.info(f"Could not load stock data for {ticker} — check internet connection.")
        else:
            st.info("Not enough sentiment data yet.")

    st.divider()

    # ── Next-day stock price prediction ───────────────────
    st.subheader("🔭 Next-Day Stock Price Prediction")
    st.caption("Uses linear regression on the last 30 days of closing prices + today's sentiment score.")

    pred_sector = st.selectbox(
        "Select sector for prediction",
        list(SECTOR_TICKERS.keys()),
        index=list(SECTOR_TICKERS.keys()).index(sector) if sector in SECTOR_TICKERS else 0,
        key="pred_sector_select",
    )
    pred_ticker, pred_ticker_name = SECTOR_TICKERS[pred_sector]
    pred_stock = load_stock(pred_ticker)

    if pred_stock.empty or len(pred_stock) < 10:
        st.info(f"Not enough stock data for {pred_ticker} to make a prediction. Need at least 10 trading days.")
    else:
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np

            # Use last 30 days (or all available)
            pred_data = pred_stock.tail(30).copy().reset_index(drop=True)
            X = np.arange(len(pred_data)).reshape(-1, 1)
            y = pred_data["close"].values

            model = LinearRegression()
            model.fit(X, y)

            # Predict next trading day
            next_x = np.array([[len(pred_data)]])
            next_price = model.predict(next_x)[0]

            # Residual std for confidence interval
            residuals = y - model.predict(X).flatten()
            std_err = residuals.std()

            last_price  = pred_data["close"].iloc[-1]
            last_date   = pd.to_datetime(pred_data["date"].iloc[-1])
            next_date   = last_date + timedelta(days=1)
            # Skip weekends
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

            price_change   = next_price - last_price
            pct_chg        = price_change / last_price * 100
            direction      = "📈 Up" if price_change > 0 else "📉 Down"
            direction_color = "#28a745" if price_change > 0 else "#dc3545"

            # Current sector sentiment (latest available)
            today_sentiment = None
            if not fdf.empty and pred_sector in fdf["sector"].values:
                today_sentiment = fdf[fdf["sector"] == pred_sector]["sentiment_score"].tail(20).mean()

            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Last Close",        f"${last_price:.2f}",  f"{last_date.strftime('%b %d')}")
            pc2.metric("Predicted Price",   f"${next_price:.2f}",  f"{pct_chg:+.2f}%",
                       delta_color="normal" if price_change > 0 else "inverse")
            pc3.metric("Confidence Range",
                       f"${next_price - std_err:.2f} – ${next_price + std_err:.2f}",
                       "±1 std dev")
            pc4.metric("Direction Signal",  direction,
                       f"Sentiment: {today_sentiment:+.3f}" if today_sentiment is not None else "—")

            # Mini prediction chart
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=pred_data["date"].astype(str), y=pred_data["close"],
                mode="lines+markers", name="Historical Close",
                line=dict(color="#1f77b4", width=2), marker=dict(size=4),
            ))
            # Trend line
            trend_y = model.predict(X).flatten()
            fig_pred.add_trace(go.Scatter(
                x=pred_data["date"].astype(str), y=trend_y,
                mode="lines", name="Trend",
                line=dict(color="#aaa", width=1, dash="dot"),
            ))
            # Prediction point
            fig_pred.add_trace(go.Scatter(
                x=[next_date.strftime("%Y-%m-%d")], y=[next_price],
                mode="markers", name=f"Predicted ({next_date.strftime('%b %d')})",
                marker=dict(size=14, color=direction_color, symbol="star"),
                error_y=dict(type="data", array=[std_err], visible=True, color=direction_color),
            ))
            fig_pred.update_layout(
                **CHART_LAYOUT, height=320,
                yaxis_title="Price (USD)",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_pred, use_container_width=True, key="trends_stock_prediction")
            chart_caption(
                f"The star (⭐) marks the predicted closing price for {pred_ticker_name} ({pred_ticker}) "
                f"on {next_date.strftime('%A, %b %d')}. "
                "The prediction is based on a linear trend fitted to the last 30 trading days of closing prices. "
                "The error bar shows ±1 standard deviation of recent residuals — the wider it is, the more volatile the stock has been. "
                "⚠️ This is a statistical estimate, not financial advice. Always consider broader market conditions."
            )

        except Exception as e:
            st.warning(f"Prediction unavailable: {e}")

    st.info(
        "**Model: Ordinary Least Squares Linear Regression (scikit-learn)**\n\n"
        "Fits a straight trend line through the last 30 days of closing prices and extends it one trading day forward. "
        "It captures the overall price direction (uptrend or downtrend) but assumes the trend continues linearly — "
        "it does not account for earnings releases, macro events, or sudden volatility. "
        "The confidence range (±1 std dev) widens when recent prices have been erratic. "
        "Use this as a directional signal only, not a precise price target. "
        "For real trading decisions, combine with fundamental analysis and broader market context."
    )
    st.divider()

    # ── Trending keywords treemap ──────────────────────────
    st.subheader("🔑 Trending Keywords")
    if "keywords" in fdf.columns:
        kws = (
            fdf["keywords"].dropna()
            .str.split(", ").explode()
            .str.strip().str.lower()
        )
        kws = kws[kws.str.len() > 2].value_counts().head(30).reset_index()
        kws.columns = ["keyword", "count"]
        if not kws.empty:
            fig3 = px.treemap(kws, path=["keyword"], values="count",
                              color="count", color_continuous_scale="Blues")
            fig3.update_layout(height=320, margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(fig3, use_container_width=True, key="trends_keywords_treemap")
            chart_caption(
                "Larger tiles = keywords that appear most frequently across analyzed articles. "
                "Darker blue = even higher frequency. "
                "These are the topics dominating tech news right now — useful for spotting emerging themes "
                "or narratives building around a company, technology, or event."
            )


# ============================================================
# Tab 3 — Forecast
# ============================================================

def tab_forecast(df: pd.DataFrame, date_range, sector: str):
    forecast_df = load_forecast()

    if forecast_df.empty:
        st.warning("No forecast yet — click Refresh Data.")
        return

    available = forecast_df["sector"].unique().tolist()
    choice    = st.selectbox(
        "Sector to forecast",
        available,
        index=available.index(sector) if sector in available else 0,
    )

    fc = forecast_df[forecast_df["sector"] == choice].copy()
    if fc.empty:
        st.info(f"No forecast for {choice}")
        return

    # Interpretation banner
    from forecasting import get_forecast_interpretation
    text  = get_forecast_interpretation(choice)
    avg   = fc["sentiment_forecast"].mean()
    color = score_color(avg)
    st.markdown(
        f'<div style="background:{color}22;border-left:4px solid {color};'
        f'padding:12px;border-radius:4px;margin-bottom:16px">'
        f'<b>📋 {choice} outlook</b><br>{text}</div>',
        unsafe_allow_html=True,
    )

    # Historical + forecast on one chart
    st.subheader(f"Historical + 7-Day Forecast — {choice}")

    d_from, d_to = date_range
    mask = (
        (pd.to_datetime(df["date"]).dt.date >= d_from) &
        (pd.to_datetime(df["date"]).dt.date <= d_to)
    )
    hist = get_daily(df[mask], choice) if not df.empty else pd.DataFrame()

    fig = go.Figure()

    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["avg_score"],
            mode="lines+markers", name="Historical",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
            hovertemplate="Date: %{x}<br>Actual: %{y:+.3f}<extra></extra>",
        ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc["forecast_date"], fc["forecast_date"].iloc[::-1]]),
        y=pd.concat([fc["upper_bound"],   fc["lower_bound"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(255,127,14,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Confidence Interval",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc["forecast_date"], y=fc["sentiment_forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#ff7f0e", width=3, dash="dash"),
        marker=dict(size=8),
        hovertemplate="Date: %{x}<br>Forecast: %{y:+.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray",
                  opacity=0.4, annotation_text="Neutral")
    fig.update_layout(**CHART_LAYOUT, height=420,
                      yaxis_range=[-1, 1],
                      yaxis_title="Sentiment Score",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True, key="forecast_main_chart")
    chart_caption(
        "Blue line = actual historical sentiment. Orange dashed line = 7-day forecast. "
        "The shaded orange band is the 80% confidence interval — the model expects the true value to fall inside this band 8 out of 10 times. "
        "A wider band means more uncertainty; a narrower band means the model is more confident. "
        "Forecasts are generated by Facebook Prophet (or a moving average if fewer than 7 days of data exist)."
    )

    # Forecast table
    st.subheader("Forecast values")
    tbl = fc[["forecast_date","sentiment_forecast","lower_bound","upper_bound","method"]].copy()
    tbl["forecast_date"] = tbl["forecast_date"].dt.strftime("%Y-%m-%d")
    tbl["outlook"] = tbl["sentiment_forecast"].apply(
        lambda x: "📈 Positive" if x > 0.05 else ("📉 Negative" if x < -0.05 else "➡️ Neutral")
    )
    tbl.columns = ["Date","Forecast","Lower","Upper","Method","Outlook"]
    st.dataframe(
        tbl.style.format({"Forecast":"{:+.4f}","Lower":"{:+.4f}","Upper":"{:+.4f}"}),
        use_container_width=True, hide_index=True,
    )
    chart_caption(
        "Forecast = the model's best estimate of next-day sentiment. "
        "Lower / Upper = the 80% confidence bounds. "
        "Outlook icons: 📈 score above +0.05 (positive), 📉 below -0.05 (negative), ➡️ in between (neutral). "
        "Prophet method = time-series model with weekly seasonality. Moving average = simpler fallback used when data is sparse."
    )


# ============================================================
# Tab 4 — Alerts
# ============================================================

def tab_alerts():
    alert_df = load_alerts()

    if alert_df.empty:
        st.info("No alert history yet — click Refresh Data.")
        return

    real = alert_df[alert_df["alert_type"] != "status_ok"]

    # ── Summary metrics — always visible at top ──────────
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    active = real[real["timestamp"] >= cutoff]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active (last 24h)",    len(active))
    c2.metric("Total Alerts (all)",   len(real))
    c3.metric("HIGH Severity",        (real["severity"] == "HIGH").sum())
    c4.metric("Most Alerted Sector",
              real["sector"].value_counts().idxmax() if not real.empty else "—")

    st.divider()

    # ── Active alert banners — in an expander ────────────
    active_label = (
        f"🚨 Active Alerts — last 24 hours ({len(active)} triggered)"
        if not active.empty
        else "✅ Active Alerts — all clear in last 24 hours"
    )
    with st.expander(active_label, expanded=not active.empty):
        if active.empty:
            st.success("All sectors within normal thresholds in the last 24 hours.")
        else:
            for _, row in active.iterrows():
                sev = row.get("severity", "LOW")
                if   sev == "HIGH":   st.error(  f"🔴 **[{sev}]** {row['message']}")
                elif sev == "MEDIUM": st.warning(f"🟡 **[{sev}]** {row['message']}")
                else:                 st.info(   f"🟢 **[{sev}]** {row['message']}")

    st.divider()

    st.caption(
        "🔔 **Active Alerts** show threshold breaches from the last 24 hours only — "
        "these are actionable signals worth investigating right now. "
        "HIGH = significant anomaly, MEDIUM = moderate shift, LOW = minor deviation."
    )

    if real.empty:
        st.info("No alerts logged yet.")
        return

    # ── Alert frequency chart — always visible ───────────
    st.subheader("Alert frequency by type")
    tc = real["alert_type"].value_counts().reset_index()
    tc.columns = ["Type", "Count"]
    tc["Type"] = tc["Type"].str.replace("_", " ").str.title()
    fig = px.bar(tc, x="Type", y="Count", color="Type",
                 color_discrete_sequence=["#dc3545", "#fd7e14", "#ffc107", "#28a745"])
    fig.update_layout(**CHART_LAYOUT, height=280, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="alerts_frequency_bar")
    chart_caption(
        "Shows how many times each alert type has fired historically. "
        "High 'Negative Spike' counts suggest recurring bad-news cycles in one or more sectors. "
        "Frequent 'Trend Change' alerts indicate volatile sentiment — the market narrative is shifting often. "
        "'Positive Surge' alerts are rare and typically signal strong bullish momentum in the news."
    )

    st.divider()

    # ── Alert history — in an expander ───────────────────
    with st.expander("📋 Alert History Log", expanded=False):
        days = st.slider("Show last N days", 1, 90, 30, key="alert_days_slider")
        hist = real[
            real["timestamp"] >= datetime.now(timezone.utc) - timedelta(days=days)
        ].copy()

        if hist.empty:
            st.info(f"No alerts in the last {days} days.")
        else:
            hist["timestamp"]  = hist["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
            hist["alert_type"] = hist["alert_type"].str.replace("_", " ").str.title()

            def sev_color(row):
                bg = {"HIGH": "#f8d7da", "MEDIUM": "#fff3cd", "LOW": "#d4edda"}.get(
                    row.get("severity", ""), "")
                return [f"background-color:{bg}"] * len(row)

            st.dataframe(
                hist[["timestamp", "alert_type", "sector", "severity", "value", "message"]]
                .style.apply(sev_color, axis=1),
                use_container_width=True, height=320, hide_index=True,
            )
        st.caption(
            "📋 **Alert History Log** is the full audit trail of every threshold breach detected across all pipeline runs. "
            "Use it to spot recurring patterns — e.g. if Cybersecurity triggers HIGH alerts every Monday, "
            "that's a weekly news cycle worth tracking, not a one-off event."
        )


# ============================================================
# Landing page — shown when no data exists yet
# ============================================================

def render_landing():
    """
    Full-screen welcome shown the very first time the app is opened
    (before main.py full has ever been run).
    A single button triggers the full pipeline and then loads the dashboard.
    """
    st.markdown("""
    <style>
    .landing-box {
        max-width: 640px;
        margin: 80px auto 0 auto;
        text-align: center;
        padding: 48px 40px;
        background: #f8f9ff;
        border: 1px solid #dee2f5;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    }
    .landing-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 8px; }
    .landing-sub   { font-size: 1.05rem; color: #555; margin-bottom: 32px; }
    .step-row      { display: flex; justify-content: center; gap: 24px;
                     margin-bottom: 32px; flex-wrap: wrap; }
    .step          { background: white; border: 1px solid #e0e4f0;
                     border-radius: 10px; padding: 14px 18px; width: 130px;
                     font-size: 0.85rem; color: #333; }
    .step .icon    { font-size: 1.5rem; margin-bottom: 4px; }
    .eta           { font-size: 0.8rem; color: #888; margin-top: 12px; }
    </style>

    <div class="landing-box">
      <div class="landing-title">📡 Tech Pulse Tracker</div>
      <div class="landing-sub">
        No data found yet. Click the button below to collect articles,
        run sentiment analysis, generate forecasts, and launch the dashboard —
        all in one go.
      </div>
      <div class="step-row">
        <div class="step"><div class="icon">📰</div>Collect<br>articles</div>
        <div class="step"><div class="icon">🧹</div>Clean &<br>preprocess</div>
        <div class="step"><div class="icon">🧠</div>Sentiment<br>analysis</div>
        <div class="step"><div class="icon">🔮</div>Forecast<br>7 days</div>
        <div class="step"><div class="icon">📊</div>Load<br>dashboard</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Centre the button using columns
    _, mid, _ = st.columns([2, 2, 2])
    with mid:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if st.button(
            "🚀 Collect Data & Launch Dashboard",
            use_container_width=True,
            type="primary",
        ):
            st.session_state["refresh"] = True
            st.rerun()

    st.markdown(
        "<p style='text-align:center;color:#aaa;font-size:0.8rem;margin-top:12px'>"
        "⏱️ First run takes 3–6 minutes (downloads ~500 MB model once, cached after)"
        "</p>",
        unsafe_allow_html=True,
    )


# ============================================================
# Main
# ============================================================

def main():
    ensure_data_dir()

    # ── Handle pipeline run FIRST before anything else renders ──
    if st.session_state.get("refresh", False):
        st.session_state["refresh"] = False
        run_pipeline()
        return   # run_pipeline() calls st.rerun() at the end

    # Start background scheduler
    try:
        from scheduler import start_scheduler
        start_scheduler()
    except Exception:
        pass

    # Load data
    df = load_sentiment()

    # ── Decide: landing page or dashboard ──
    # Use file existence as the gate, not df.empty — df can be empty
    # due to a date-parse issue even when the file exists and has rows.
    # We only show the landing page when sentiment_results.csv is truly absent.
    sentiment_file_exists = os.path.exists(SENTIMENT_RESULTS_PATH)

    if not sentiment_file_exists:
        render_landing()
        return

    # ── Data exists → render full dashboard ──
    date_range, sector = render_sidebar(df)

    st.title("📡 Tech Industry Pulse Tracker")
    st.caption(
        "Tracking AI/ML · Cloud · Cybersecurity · Startups · "
        "Semiconductors · Big Tech  |  "
        f"Loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    st.divider()

    if df.empty:
        st.warning(
            "⚠️ sentiment_results.csv exists but could not be loaded — "
            "it may be empty or have a formatting issue. "
            "Click **🔄 Update Data** in the sidebar to re-run the pipeline."
        )
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "📈 Trends",
        "🔮 Forecast",
        "🚨 Alerts",
    ])

    with tab1: tab_overview(df, date_range, sector)
    with tab2: tab_trends(df, date_range, sector)
    with tab3: tab_forecast(df, date_range, sector)
    with tab4: tab_alerts()


if __name__ == "__main__":
    main()