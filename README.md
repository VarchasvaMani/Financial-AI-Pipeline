# 📈 Financial AI Pipeline — Extended Edition

A production-grade, AI-driven financial data pipeline with full backtesting,
paper trading, portfolio optimisation, live streaming, multi-channel alerts,
and extended data sources. Powered by Claude, LightGBM, and Streamlit.

---

## What's included

### Original modules
| Module | Description |
|---|---|
| `core.py` | SQLAlchemy ORM + Pydantic schemas |
| `pipeline.py` | Main orchestrator |
| `ingestion/market_feed.py` | OHLCV from yfinance |
| `ingestion/news_feed.py` | News from yfinance + Alpha Vantage |
| `features/technical.py` | 13 technical indicators |
| `agents/sentiment_agent.py` | Claude news sentiment scoring |
| `agents/narrative_agent.py` | Claude signal explanations |
| `models/signal_model.py` | LightGBM + SHAP |
| `models/anomaly_detector.py` | Volume / price / vol regime detection |
| `dashboard/app.py` | 6-page Streamlit dashboard |

### Extension modules
| Module | Description |
|---|---|
| `backtest/engine.py` | Vectorised backtesting — Sharpe, Sortino, Calmar, max DD, win rate |
| `trading/paper_trader.py` | Virtual portfolio — tracks positions, PnL, trade history in SQLite |
| `alerts/alert_bot.py` | Telegram + Slack alerts — 3-tier routing (INFO / WARNING / CRITICAL) |
| `datasources/extended.py` | SEC filings (EDGAR), earnings calendar, insider trades |
| `datasources/live_stream.py` | WebSocket live prices (Polygon) + yfinance polling fallback |
| `portfolio/optimiser.py` | Markowitz: Max Sharpe, Min Vol, Risk Parity, Equal Weight |

---

## Architecture

```
┌─────────────────────── INGESTION ───────────────────────┐
│  yfinance · Alpha Vantage · SEC EDGAR · Polygon WS       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              FEATURE ENGINEERING (13 features)           │
│  RSI · MACD · BB · ATR · Vol Z-score · Momentum · VWAP  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   AI / ML LAYER                          │
│  Claude Sentiment · LightGBM+SHAP · Anomaly Detection    │
│  Claude Narratives · Portfolio Optimiser · Backtester    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│             EXECUTION & DELIVERY                         │
│  Paper Trader · Telegram/Slack Alerts · Live Stream      │
│  6-Page Streamlit Dashboard                              │
└─────────────────────────────────────────────────────────┘
```

---

## Quickstart

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add API keys

# 3. Run the full pipeline
python pipeline.py --demo

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Recommended | Claude sentiment + narratives |
| `ALPHA_VANTAGE_API_KEY` | Optional | News (25 req/day free tier) |
| `POLYGON_API_KEY` | Optional | Live WebSocket price stream |
| `TELEGRAM_BOT_TOKEN` | Optional | Telegram alert bot |
| `TELEGRAM_CHAT_ID` | Optional | Your Telegram chat ID |
| `SLACK_WEBHOOK_URL` | Optional | Slack incoming webhook |
| `DEFAULT_WATCHLIST` | Optional | Comma-separated tickers |
| `DB_PATH` | Optional | SQLite path (default: data/pipeline.db) |

---

## Dashboard pages

| Page | What you see |
|---|---|
| 🏠 Live Signal Feed | Signal cards with Claude narratives, SHAP charts, NL query |
| 🔍 Asset Deep Dive | Candlestick + BB/VWAP, RSI, feature snapshot, news feed |
| 📊 Backtesting | Equity curve, drawdown, full metrics table, trade log |
| 📋 Paper Trading | Virtual portfolio, open positions, trade history, PnL chart |
| ⚖️ Portfolio Optimiser | Efficient frontier, Max Sharpe/Min Vol/Risk Parity weights |
| ⚠️ Risk & Anomalies | Heatmap, anomaly scanner, signal history, alert log |

---

## Extension module usage

### Backtesting
```python
from backtest.engine import BacktestEngine
bt = BacktestEngine(
    symbols=["AAPL", "MSFT", "NVDA"],
    start="2024-01-01",
    end="2024-12-31",
    position_size=0.10,        # 10% per position
    confidence_threshold=0.62,
    stop_loss_pct=0.03,
    take_profit_pct=0.06,
    hold_bars=4,
)
results = bt.run()
bt.report(results)
# → Sharpe, Sortino, Calmar, Max DD, Win Rate, Profit Factor
```

### Paper Trading
```python
from trading.paper_trader import execute_signal, get_portfolio_value

# Feed it a signal dict (same format as signal_model output)
result = execute_signal({"symbol": "AAPL", "direction": "long", "confidence": 0.78})
portfolio = get_portfolio_value()
print(f"Total value: ${portfolio['total_value']:,.2f}")
```

### Portfolio Optimisation
```python
from portfolio.optimiser import run_optimisation

results = run_optimisation(["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"])
# Returns: max_sharpe, min_volatility, risk_parity, equal_weight weights
print(results["max_sharpe"]["weights"])
print(f"Expected Sharpe: {results['max_sharpe']['sharpe']:.2f}")
```

### Alerts (Telegram + Slack)
```python
from alerts.alert_bot import send_alert, alert_signal, AlertLevel

# Manual alert
send_alert("VaR breach on TSLA", level=AlertLevel.CRITICAL, symbol="TSLA")

# Auto-format and send a signal alert
alert_signal({"symbol": "NVDA", "direction": "long", "confidence": 0.84,
              "narrative": "Strong momentum post earnings..."})
```

### SEC Filings + Earnings Calendar
```python
from datasources.extended import fetch_sec_filings, get_upcoming_earnings

# Fetch and Claude-analyse recent 8-K filings
filings = fetch_sec_filings("AAPL", form_types=["8-K"])

# Get earnings dates for watchlist
upcoming = get_upcoming_earnings(["AAPL","MSFT","NVDA"], days_ahead=14)
```

### Live Price Streaming
```python
from datasources.live_stream import start_stream, get_live_prices, subscribe

# Subscribe to price ticks
def on_tick(tick):
    print(f"{tick['symbol']}: ${tick['price']}")

subscribe(on_tick)
streamer = start_stream(["AAPL", "MSFT", "NVDA"])
```

---

## Running tests

```bash
# All 18 tests — no API keys needed
python -m pytest tests/ -v
```

---

## Production checklist

- [ ] Swap SQLite → TimescaleDB
- [ ] Add Kafka for real-time ingestion
- [ ] Enable Polygon WebSocket (set `POLYGON_API_KEY`)
- [ ] Configure Telegram bot (`TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`)
- [ ] Connect Slack webhook (`SLACK_WEBHOOK_URL`)
- [ ] Schedule pipeline with cron or APScheduler
- [ ] Add Grafana for system health monitoring
- [ ] Implement 7-year audit log retention (MiFID II)

---

## Disclaimer
Educational portfolio project only. Not financial advice.
Do not trade real capital based on model outputs.
