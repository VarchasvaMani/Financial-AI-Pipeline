# 📈 Financial AI Pipeline

> A production-grade, end-to-end AI-driven financial data pipeline — from raw market data to Claude-powered signals, backtesting, paper trading, portfolio optimisation, live streaming, and multi-channel alerts.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![LightGBM](https://img.shields.io/badge/ML-LightGBM-orange?style=flat-square)
![Claude](https://img.shields.io/badge/AI-Claude%20API-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square)
![SQLite](https://img.shields.io/badge/DB-SQLite-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Environment variables](#environment-variables)
- [Pipeline stages](#pipeline-stages)
- [Features computed](#features-computed)
- [AI agents](#ai-agents)
- [Extension modules](#extension-modules)
- [Dashboard pages](#dashboard-pages)
- [Running tests](#running-tests)
- [API cost estimate](#api-cost-estimate)
- [Production checklist](#production-checklist)
- [Disclaimer](#disclaimer)

---

## Overview

This project builds a complete AI-driven financial data pipeline that:

1. **Ingests** real market data (OHLCV + news) from free APIs — no paid subscriptions required to get started
2. **Computes** 13 technical indicators with point-in-time correctness
3. **Scores** news sentiment using Claude (falls back to keyword matching without an API key)
4. **Generates** directional signals using LightGBM + SHAP explainability
5. **Explains** every signal in plain English using Claude narratives
6. **Detects** anomalies in volume, price action, and volatility regimes
7. **Backtests** strategies on historical data with full performance metrics
8. **Simulates** live trading in a virtual portfolio (paper trading)
9. **Optimises** portfolio weights using Markowitz mean-variance optimisation
10. **Streams** live prices via WebSocket or polling fallback
11. **Alerts** you via Telegram and Slack when signals, anomalies, or risk breaches occur
12. **Visualises** everything in a 6-page interactive Streamlit dashboard

Works **fully offline** without any API keys (using mock signals and keyword sentiment). Add your `ANTHROPIC_API_KEY` to unlock Claude-powered narratives and sentiment scoring.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                        │
│   yfinance (OHLCV) │ Alpha Vantage (news) │ Polygon (WS)     │
│   SEC EDGAR (filings) │ Earnings calendar                    │
└───────────────────────────┬──────────────────────────────────┘
                            │  raw OHLCV + news → SQLite
┌───────────────────────────▼──────────────────────────────────┐
│                    FEATURE ENGINEERING                        │
│   RSI · MACD · Bollinger Bands · ATR · Volume Z-score        │
│   Realised Volatility · Momentum 1d/5d/20d · VWAP           │
└───────────────────────────┬──────────────────────────────────┘
                            │  13 features per asset per bar
┌───────────────────────────▼──────────────────────────────────┐
│                      AI / ML LAYER                            │
│   Claude Sentiment Agent  │  LightGBM Signal Model           │
│   SHAP Explainability     │  Anomaly Detector                │
│   Claude Narrative Agent  │  NL Query Interface              │
└───────────────────────────┬──────────────────────────────────┘
                            │  signals + narratives + anomalies
┌───────────────────────────▼──────────────────────────────────┐
│                   EXECUTION & DELIVERY                        │
│   Backtester  │  Paper Trader  │  Portfolio Optimiser        │
│   Telegram/Slack Alerts  │  Live Price Stream               │
│   6-Page Streamlit Dashboard                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Project structure

```
financial-ai-pipeline/
│
├── core.py                        # SQLAlchemy ORM tables + Pydantic schemas
├── pipeline.py                    # Main orchestrator (runs all 6 stages)
├── requirements.txt               # All dependencies
├── .env.example                   # Environment variable template
├── .gitignore
│
├── ingestion/
│   ├── market_feed.py             # OHLCV from yfinance — dedup, incremental
│   └── news_feed.py               # Headlines from yfinance + Alpha Vantage
│
├── features/
│   └── technical.py               # 13 indicators: RSI, MACD, BB, ATR, momentum…
│
├── agents/
│   ├── sentiment_agent.py         # Claude news scoring (bullish/bearish/neutral)
│   └── narrative_agent.py         # Claude plain-English signal explanations
│
├── models/
│   ├── signal_model.py            # LightGBM + SHAP + mock signals
│   └── anomaly_detector.py        # Volume spikes, price moves, vol regimes
│
├── backtest/
│   └── engine.py                  # Vectorised backtester — Sharpe, Sortino, Calmar
│
├── trading/
│   └── paper_trader.py            # Virtual portfolio — SQLite-persisted positions
│
├── alerts/
│   └── alert_bot.py               # Telegram + Slack — 3-tier alert routing
│
├── datasources/
│   ├── extended.py                # SEC EDGAR filings + earnings calendar
│   └── live_stream.py             # WebSocket prices (Polygon) + polling fallback
│
├── portfolio/
│   └── optimiser.py               # Markowitz: Max Sharpe, Min Vol, Risk Parity
│
├── dashboard/
│   └── app.py                     # Streamlit — 6 interactive pages
│
└── tests/
    └── test_pipeline.py           # 18 tests — all pass without API keys
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/financial-ai-pipeline.git
cd financial-ai-pipeline
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and add your keys. At minimum, set `ANTHROPIC_API_KEY` to enable Claude features. Everything else is optional — the pipeline runs with fallbacks if keys are missing.

### 3. Run the pipeline

```bash
# Demo mode — 5 symbols, fast, no training required
python pipeline.py --demo

# Full pipeline — all watchlist symbols
python pipeline.py

# Full pipeline with model retraining
python pipeline.py --train

# Custom symbols
python pipeline.py --symbols AAPL MSFT NVDA TSLA SPY
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Recommended | — | Claude sentiment + narrative agents |
| `ALPHA_VANTAGE_API_KEY` | Optional | — | News feed (25 req/day free at alphavantage.co) |
| `POLYGON_API_KEY` | Optional | — | Live WebSocket price stream (polygon.io) |
| `TELEGRAM_BOT_TOKEN` | Optional | — | Alert bot (get from @BotFather on Telegram) |
| `TELEGRAM_CHAT_ID` | Optional | — | Your Telegram chat or group ID |
| `SLACK_WEBHOOK_URL` | Optional | — | Slack incoming webhook URL |
| `DEFAULT_WATCHLIST` | Optional | `AAPL,MSFT,GOOGL,NVDA,TSLA,AMZN,META,JPM,GS,SPY` | Comma-separated tickers |
| `DB_PATH` | Optional | `data/pipeline.db` | SQLite database path |
| `LOG_LEVEL` | Optional | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `PAPER_INITIAL_CAPITAL` | Optional | `100000` | Paper trading starting capital (USD) |
| `PAPER_POSITION_SIZE` | Optional | `0.10` | Fraction of capital per position |
| `BACKTEST_START` | Optional | `2024-01-01` | Default backtest start date |
| `BACKTEST_END` | Optional | `2024-12-31` | Default backtest end date |

---

## Pipeline stages

Running `python pipeline.py` executes all 6 stages in sequence:

| Stage | What happens |
|---|---|
| **1a. Market ingestion** | Fetches OHLCV bars from yfinance for all watchlist symbols. Saves to SQLite, skipping duplicates. |
| **1b. News ingestion** | Fetches recent headlines from yfinance and Alpha Vantage. Deduplicates by headline text. |
| **2. Feature engineering** | Computes 13 technical indicators per asset per bar. Saves to feature store. |
| **3. Sentiment scoring** | Runs Claude (or keyword fallback) on all unscored headlines. Updates sentiment, magnitude, theme, confidence fields. |
| **4. Signal generation** | Loads features → runs LightGBM → generates directional signals with SHAP top features → Claude writes narratives → saves to DB. |
| **5. Anomaly detection** | Scans for volume spikes, abnormal price moves, and volatility regime shifts. Claude explains each anomaly found. |

---

## Features computed

All features are computed in `features/technical.py` with no lookahead bias.

| Feature | Description | Period |
|---|---|---|
| `rsi` | Relative Strength Index | 14 bars |
| `macd` | MACD line (EMA12 − EMA26) | 12/26 bars |
| `macd_signal` | MACD signal line (EMA9 of MACD) | 9 bars |
| `bb_upper` | Bollinger upper band | 20 bars, 2σ |
| `bb_mid` | Bollinger middle band (SMA) | 20 bars |
| `bb_lower` | Bollinger lower band | 20 bars, 2σ |
| `atr` | Average True Range | 14 bars |
| `volume_zscore` | Volume vs 20-bar rolling mean (σ units) | 20 bars |
| `realized_vol` | Rolling std of log returns, annualised | 20 bars |
| `momentum_1d` | 1-bar price return | 1 bar |
| `momentum_5d` | 5-bar price return | 5 bars |
| `momentum_20d` | 20-bar price return | 20 bars |
| `vwap` | Volume-weighted average price | session |

---

## AI agents

### Sentiment agent (`agents/sentiment_agent.py`)

Scores every news headline using Claude:

```json
{
  "sentiment": "bullish",
  "magnitude": 4,
  "theme": "earnings beat",
  "confidence": 0.91
}
```

Falls back to keyword matching when `ANTHROPIC_API_KEY` is not set. Processes headlines in batches with configurable rate limiting.

### Narrative agent (`agents/narrative_agent.py`)

Generates a 2–3 sentence plain-English explanation for every signal:

> *"Going long NVDA with 82% confidence over a 4-hour horizon. Data centre momentum is driving the 5-day return of +3.1%, and RSI recovery from 38 signals buyer re-entry. Three analyst upgrades published this morning with a consensus PT raise to $920. Key risk: macro risk-off if tomorrow's CPI print beats estimates."*

Also powers:
- **Anomaly explanations** — explains what caused each detected anomaly
- **NL query interface** — answers natural language questions about current signals in the dashboard

---

## Extension modules

### Backtesting (`backtest/engine.py`)

Vectorised, point-in-time correct backtester. Simulates trading signals on historical OHLCV data with stop-loss, take-profit, and time-based exits.

```python
from backtest.engine import BacktestEngine

bt = BacktestEngine(
    symbols=["AAPL", "MSFT", "NVDA"],
    start="2024-01-01",
    end="2024-12-31",
    position_size=0.10,           # 10% of capital per position
    confidence_threshold=0.62,    # minimum signal confidence to trade
    stop_loss_pct=0.03,           # 3% stop-loss
    take_profit_pct=0.06,         # 6% take-profit
    hold_bars=4,                  # max 4 bars per trade
)
results = bt.run()
bt.report(results)
```

**Metrics output:**
- Total & annualised return
- Sharpe ratio, Sortino ratio, Calmar ratio
- Max drawdown, annualised volatility
- Win rate, profit factor
- Per-trade log with entry/exit prices, PnL, exit reason

---

### Paper trading (`trading/paper_trader.py`)

Virtual portfolio simulator persisted in SQLite. Survives process restarts.

```python
from trading.paper_trader import execute_signal, get_portfolio_value, get_positions

# Execute a signal (same dict format as signal_model output)
execute_signal({"symbol": "AAPL", "direction": "long", "confidence": 0.78})

# Check portfolio
portfolio = get_portfolio_value()
print(f"Cash: ${portfolio['cash']:,.2f}")
print(f"Positions value: ${portfolio['positions_value']:,.2f}")
print(f"Total: ${portfolio['total_value']:,.2f}")
print(f"PnL: {portfolio['total_pnl_pct']:+.2f}%")
```

---

### Portfolio optimiser (`portfolio/optimiser.py`)

Markowitz mean-variance optimisation using `scipy`. Computes 4 portfolio types from the same set of assets.

```python
from portfolio.optimiser import run_optimisation

results = run_optimisation(["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"])

# Max Sharpe portfolio
print(results["max_sharpe"]["weights"])
print(f"Expected Sharpe: {results['max_sharpe']['sharpe']:.2f}")
print(f"Expected return: {results['max_sharpe']['expected_return']:.1%}")
print(f"Expected vol:    {results['max_sharpe']['volatility']:.1%}")

# Also available: min_volatility, risk_parity, equal_weight
```

---

### Alerts (`alerts/alert_bot.py`)

Three-tier alert routing to Telegram and/or Slack. Falls back to console logging when no keys are configured.

```python
from alerts.alert_bot import send_alert, alert_signal, alert_anomaly, AlertLevel

# Tier 1 — INFO: signal generated
alert_signal({"symbol": "NVDA", "direction": "long", "confidence": 0.84,
              "narrative": "Strong momentum post earnings..."})

# Tier 2 — WARNING: anomaly detected
alert_anomaly("TSLA", "volume_spike", zscore=3.2,
              explanation="Volume 3.2× normal — no clear news catalyst")

# Tier 3 — CRITICAL: risk breach
send_alert("VaR limit breached on portfolio", level=AlertLevel.CRITICAL)
```

**Alert tiers:**

| Tier | Level | Channels | When |
|---|---|---|---|
| 1 | INFO | Slack | New signal generated |
| 2 | WARNING | Slack + Telegram | Anomaly detected, risk at 80% |
| 3 | CRITICAL | Slack + Telegram + PagerDuty-style | VaR breach, sanctions match |

---

### Extended data sources (`datasources/extended.py`)

```python
from datasources.extended import fetch_sec_filings, get_upcoming_earnings

# Fetch and Claude-analyse recent SEC filings
filings = fetch_sec_filings("AAPL", form_types=["8-K", "10-Q"])
for f in filings:
    print(f["date"], f["title"])
    print(f["claude_analysis"])   # Claude's plain-English summary

# Earnings calendar
upcoming = get_upcoming_earnings(["AAPL", "MSFT", "NVDA"], days_ahead=14)
for e in upcoming:
    print(f"{e['symbol']} reports on {e['date']} — EPS est: {e['eps_estimate']}")
```

---

### Live price stream (`datasources/live_stream.py`)

WebSocket streaming via Polygon API with automatic fallback to yfinance polling when no API key is set.

```python
from datasources.live_stream import start_stream, subscribe, get_live_prices

# Subscribe to ticks
def on_tick(tick):
    print(f"{tick['symbol']}: ${tick['price']:.2f}  vol={tick['volume']}")

subscribe(on_tick)
streamer = start_stream(["AAPL", "MSFT", "NVDA"], interval_sec=60)

# Or just get latest snapshot
prices = get_live_prices(["AAPL", "MSFT"])
```

---

## Dashboard pages

Start with `streamlit run dashboard/app.py`, then open [http://localhost:8501](http://localhost:8501).

| Page | What you see |
|---|---|
| **🏠 Live Signal Feed** | Signal cards with direction, confidence, price, sentiment score, Claude narrative, SHAP bar chart. Filter by direction/confidence/symbol. Natural language query box powered by Claude. |
| **🔍 Asset Deep Dive** | Candlestick chart with Bollinger Bands and VWAP overlays. Volume and RSI subplots. Feature snapshot panel. Recent news with sentiment labels. |
| **📊 Backtesting** | Run a backtest directly from the UI. Equity curve, drawdown chart, full metrics table, trade-by-trade log. |
| **📋 Paper Trading** | Live virtual portfolio. Open positions table, trade history, cumulative PnL chart, cash balance. |
| **⚖️ Portfolio Optimiser** | Efficient frontier visualisation. Max Sharpe, Min Volatility, Risk Parity, Equal Weight weight tables with expected return and vol. |
| **⚠️ Risk & Anomalies** | Signal confidence heatmap, directional exposure pie chart, live anomaly scanner, full signal history table. |

---

## Running tests

```bash
# Run all 18 tests (no API keys required)
python -m pytest tests/ -v

# Specific test
python -m pytest tests/test_pipeline.py::test_compute_features_shape -v

# With coverage
python -m pytest tests/ -v --tb=short
```

All tests use synthetic data and deterministic mocks — they pass in CI without any API keys or network access.

---

## API cost estimate

Using `claude-sonnet-4-20250514` (the default model):

| Use case | Calls/month | Estimated cost |
|---|---|---|
| Personal / portfolio project (10 symbols, 2× daily) | ~1,500 | **< $0.01/month** |
| Small production (100 symbols, hourly) | ~75,000 | **~$2–4/month** |
| Large production (500 symbols, hourly) | ~375,000 | **~$10–20/month** |

**Cost tips:**
- Switch `sentiment_agent.py` to `claude-haiku-4-5-20251001` for 3× cheaper sentiment scoring
- Use the Anthropic Batch API (50% discount) for non-real-time sentiment jobs
- Enable prompt caching on the sentiment system prompt (up to 90% input token savings)

---

## Production checklist

- [ ] Swap SQLite → TimescaleDB for time-series performance at scale
- [ ] Add Apache Kafka for real-time tick ingestion
- [ ] Enable Polygon WebSocket by setting `POLYGON_API_KEY`
- [ ] Configure Telegram alerts (`TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`)
- [ ] Connect Slack webhook (`SLACK_WEBHOOK_URL`)
- [ ] Schedule pipeline with `cron` or `APScheduler` (e.g. every hour on market days)
- [ ] Add Grafana + InfluxDB for system health monitoring
- [ ] Implement 7-year audit log retention policy (MiFID II compliance)
- [ ] Add authentication to Streamlit dashboard for multi-user access
- [ ] Containerise with Docker + deploy on Kubernetes for production scale

---

## Disclaimer

This is an educational portfolio project. Nothing in this codebase constitutes financial advice. All signals, backtests, and portfolio optimisations are for learning and demonstration purposes only. Do not trade real capital based on the outputs of this system.

---
