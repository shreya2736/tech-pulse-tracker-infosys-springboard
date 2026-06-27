# 📡 Tech Industry Pulse Tracker

> Real-time sentiment analysis, forecasting, and stock correlation dashboard for the tech industry — powered by RoBERTa, Prophet, and Streamlit.

🔗 **Live Demo:** [tech-pulse-tracker-infosys-springboard.streamlit.app](https://tech-pulse-tracker-infosys-springboard-b8hydz5pffxsmvafsdp7s4.streamlit.app/)

---

## 📌 What It Does

Tech Pulse Tracker automatically collects tech news from multiple sources, runs NLP sentiment analysis on every article, forecasts sentiment trends for the next 7 days, and visualizes everything in an interactive dashboard — updated every 6 hours automatically via GitHub Actions.

---

## 🖥️ Dashboard Preview

| Tab | What You See |
|-----|-------------|
| 📊 Overview | KPI cards, sector sentiment bar chart, label distribution pie, active alerts |
| 📈 Trends | Daily sentiment trends, stock price overlay, next-day stock prediction, keyword treemap |
| 🔮 Forecast | 7-day Prophet sentiment forecast with confidence intervals per sector |
| 🚨 Alerts | Active alert banners, alert frequency chart, full alert history log |

---

## 🏗️ Architecture

```
News Sources          Pipeline                    Storage          Dashboard
──────────────        ──────────────────────────  ───────────────  ─────────────────
NewsAPI           →   collect → preprocess    →   Supabase     →   Streamlit Cloud
HackerNews        →   analyze → forecast      →   Storage          (reads CSVs)
Dev.to            →   alerts                  →   (3 CSVs)
Google News RSS   →
                       Runs every 6 hours
                       via GitHub Actions
                       (laptop-independent)
```

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Collection | NewsAPI, HackerNews API, Dev.to API, Google News RSS |
| NLP / Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` (HuggingFace) |
| Keyword Extraction | YAKE (unsupervised, offline) |
| Forecasting | Facebook Prophet (time-series, weekly seasonality) |
| Stock Data | yfinance (NASDAQ, QQQ, HACK ETF, SOXX ETF) |
| Stock Prediction | Scikit-learn Linear Regression |
| Dashboard | Streamlit + Plotly |
| Cloud Storage | Supabase Storage |
| Automation | GitHub Actions (cron every 6 hours) |
| Deployment | Streamlit Community Cloud |

---

## 📂 Project Structure

```
tech_pulse_tracker/
│
├── app.py                  # Streamlit dashboard
├── main.py                 # CLI entry point
├── collector.py            # Data collection from 4 sources
├── preprocessor.py         # Text cleaning and deduplication
├── analyzer.py             # RoBERTa sentiment + YAKE keywords
├── forecasting.py          # Prophet 7-day forecast
├── alerts.py               # Threshold-based alert system
├── scheduler.py            # Background auto-refresh scheduler
├── config.py               # Central configuration
├── supabase_storage.py     # Cloud CSV read/write helper
├── upload_to_supabase.py   # Uploads CSVs to Supabase after pipeline
├── repair_csv.py           # One-time CSV corruption repair utility
├── requirements.txt
├── .env                    # API keys (never committed)
├── .gitignore
│
└── .github/
    └── workflows/
        └── pipeline.yml    # GitHub Actions automation
```

---

## ⚙️ How It Works

### 1. Data Collection
Every 6 hours GitHub Actions triggers the pipeline. Articles are fetched from:
- **NewsAPI** — last 7 days, 100 articles, filtered by tech keywords
- **HackerNews** — top 100 stories
- **Dev.to** — latest posts tagged `ai`, `cloud`, `security`, `startup`
- **Google News RSS** — technology feed

Articles are deduplicated by URL hash so no article is processed twice.

### 2. Preprocessing
Raw articles are cleaned (HTML stripped, URLs removed, non-English filtered), deduplicated on title, and a `combined_text` field is built from title + description + content for the sentiment model.

### 3. Sentiment Analysis
Each article is scored by `cardiffnlp/twitter-roberta-base-sentiment-latest` — a RoBERTa model trained on tweets and news. Output is a score from **-1.0 (very negative)** to **+1.0 (very positive)**. Top 5 keyphrases are extracted per article using YAKE.

### 4. Forecasting
Daily sentiment scores are aggregated per sector and fed into **Facebook Prophet**. Prophet forecasts the next 7 days with an 80% confidence interval. Weekly seasonality is enabled when 14+ days of data are available. Falls back to moving average when fewer than 7 days exist.

### 5. Alerts
After each pipeline run, sentiment scores are checked against thresholds:
- 🔴 **HIGH** — score drops below -0.20 (negative spike)
- 🟡 **MEDIUM** — volatility exceeds 0.15 (unstable sentiment)
- 🟢 **LOW** — positive surge above 0.50

### 6. Storage & Dashboard
Results are uploaded to **Supabase Storage** as CSVs. The Streamlit Cloud dashboard reads from Supabase — so it always shows fresh data regardless of whether your local machine is running.

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- NewsAPI key (free at [newsapi.org](https://newsapi.org))

### Installation

```bash
git clone https://github.com/shreya2736/tech-pulse-tracker-infosys-springboard.git
cd tech-pulse-tracker-infosys-springboard

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
NEWS_API_KEY=your_newsapi_key_here

# Optional — only needed for cloud deployment
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key_here

# Optional — email alerts
GMAIL_SENDER=your@gmail.com
GMAIL_APP_PASSWORD=your_app_password
GMAIL_RECEIVER=notify@gmail.com
```

### Run Locally

```bash
# Run full pipeline then launch dashboard
python main.py full
streamlit run app.py

# Or run steps individually
python main.py collect      # fetch articles
python main.py preprocess   # clean text
python main.py analyze      # sentiment analysis
python main.py forecast     # 7-day forecast
python main.py alerts       # check thresholds
```

On first run the dashboard shows a landing page with a **"🚀 Collect Data & Launch Dashboard"** button — clicking it runs the full pipeline automatically.

---

## ☁️ Deployment

The app is deployed using a **split architecture** to handle the heavy ML pipeline:

| Component | Where It Runs |
|-----------|--------------|
| Data collection + RoBERTa + Prophet | GitHub Actions (free, every 6h) |
| CSV storage | Supabase Storage (free tier) |
| Dashboard | Streamlit Community Cloud (free) |

This means the Streamlit server never runs the heavy ML model — it only reads lightweight CSVs from Supabase. The dashboard stays fast and within free tier memory limits.

### Deploy Your Own

1. Fork this repo
2. Create a [Supabase](https://supabase.com) project → Storage → create bucket `techpulse` (public) → add RLS policy allowing anon all operations
3. Add GitHub secrets: `NEWS_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`
4. Deploy to [Streamlit Cloud](https://share.streamlit.io) → add same three keys as secrets in TOML format
5. Trigger the GitHub Actions workflow manually to populate Supabase on first run

---

## 📊 Sectors Tracked

| Sector | Key Terms |
|--------|----------|
| AI/ML | OpenAI, ChatGPT, LLM, Gemini, neural network, generative AI |
| Cloud | AWS, Azure, GCP, Kubernetes, serverless, SaaS |
| Cybersecurity | data breach, ransomware, zero day, phishing, malware |
| Startups | VC funding, Series A/B, unicorn, Y Combinator |
| Semiconductors | Nvidia, AMD, Intel, TSMC, GPU, chip shortage |
| Big Tech | Google, Microsoft, Apple, Meta, Amazon, antitrust |

---

## 📈 Tracked Stocks

| Sector | Ticker | Index |
|--------|--------|-------|
| Overall | `^IXIC` | NASDAQ Composite |
| AI/ML | `QQQ` | NASDAQ 100 ETF |
| Cybersecurity | `HACK` | Cybersecurity ETF |
| Semiconductors | `SOXX` | Semiconductor ETF |
| Big Tech | `QQQ` | NASDAQ 100 ETF |

---

## ⚠️ Disclaimers

- **Not financial advice.** Stock price predictions use simple linear regression on 30 days of historical prices. Do not use for real trading decisions.
- **Sentiment is news-based**, not a reflection of actual company performance or stock fundamentals.
- **Free API tier limits apply.** NewsAPI free tier restricts to 100 articles/request and a 1-month lookback window.

---

## 📄 License

MIT License — free to use, modify, and distribute.