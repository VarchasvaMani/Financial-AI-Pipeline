"""
Stage 1 — Market Data Ingestion
Fetches OHLCV data from yfinance and stores in SQLite.
Supports historical backfill and incremental updates.
"""

from __future__ import annotations
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import select, func
from dotenv import load_dotenv

from core import engine, init_db, OHLCVRecord, MarketEvent, get_session

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

DEFAULT_WATCHLIST = os.getenv(
    "DEFAULT_WATCHLIST",
    "AAPL,MSFT,GOOGL,NVDA,TSLA,AMZN,META,JPM,GS,SPY"
).split(",")


def fetch_ohlcv(
    symbol: str,
    period: str = "90d",
    interval: str = "1h",
) -> list[MarketEvent]:
    """Fetch OHLCV bars from yfinance and return as validated Pydantic models."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            log.warning(f"[{symbol}] No data returned from yfinance")
            return []

        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

        events = []
        for ts, row in df.iterrows():
            try:
                events.append(MarketEvent(
                    symbol=symbol,
                    timestamp=ts.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    source="yfinance",
                ))
            except Exception as e:
                log.debug(f"[{symbol}] Skipping row {ts}: {e}")
                continue

        log.info(f"[{symbol}] Fetched {len(events)} bars ({interval}, {period})")
        return events

    except Exception as e:
        log.error(f"[{symbol}] Fetch failed: {e}")
        return []


def get_last_timestamp(symbol: str) -> Optional[datetime]:
    """Get the most recent stored timestamp for a symbol."""
    with get_session() as session:
        result = session.execute(
            select(func.max(OHLCVRecord.timestamp))
            .where(OHLCVRecord.symbol == symbol)
        ).scalar()
        return result


def save_ohlcv(events: list[MarketEvent], symbol: str) -> int:
    """Upsert OHLCV records — skip duplicates by timestamp."""
    if not events:
        return 0

    with get_session() as session:
        # Get existing timestamps to avoid duplicates
        existing = set(
            row[0] for row in session.execute(
                select(OHLCVRecord.timestamp)
                .where(OHLCVRecord.symbol == symbol)
            ).fetchall()
        )

        new_records = []
        for e in events:
            if e.timestamp not in existing:
                new_records.append(OHLCVRecord(
                    symbol=e.symbol,
                    timestamp=e.timestamp,
                    open=e.open,
                    high=e.high,
                    low=e.low,
                    close=e.close,
                    volume=e.volume,
                    source=e.source,
                ))

        session.add_all(new_records)
        session.commit()
        log.info(f"[{symbol}] Saved {len(new_records)} new bars")
        return len(new_records)


def load_ohlcv(
    symbol: str,
    days: int = 90,
) -> pd.DataFrame:
    """Load OHLCV from DB into a DataFrame sorted by timestamp."""
    since = datetime.utcnow() - timedelta(days=days)
    with get_session() as session:
        rows = session.execute(
            select(OHLCVRecord)
            .where(OHLCVRecord.symbol == symbol)
            .where(OHLCVRecord.timestamp >= since)
            .order_by(OHLCVRecord.timestamp)
        ).scalars().all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "timestamp": r.timestamp,
        "open": r.open,
        "high": r.high,
        "low": r.low,
        "close": r.close,
        "volume": r.volume,
    } for r in rows])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def ingest_watchlist(
    symbols: list[str] = DEFAULT_WATCHLIST,
    period: str = "90d",
    interval: str = "1h",
    delay: float = 1.0,
) -> dict[str, int]:
    """Ingest all symbols in the watchlist. Returns {symbol: rows_saved}."""
    results = {}
    for symbol in symbols:
        events = fetch_ohlcv(symbol, period=period, interval=interval)
        saved = save_ohlcv(events, symbol)
        results[symbol] = saved
        time.sleep(delay)  # be polite to the API
    return results


def get_latest_prices(symbols: list[str] = DEFAULT_WATCHLIST) -> dict[str, float]:
    """Quick fetch of latest close prices for all symbols."""
    prices = {}
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="2d", interval="1d")
            if not hist.empty:
                prices[sym] = round(float(hist["Close"].iloc[-1]), 2)
        except Exception:
            pass
    return prices


if __name__ == "__main__":
    init_db()
    print("Starting ingestion for watchlist:", DEFAULT_WATCHLIST)
    results = ingest_watchlist(period="90d", interval="1h")
    print("\nIngestion complete:")
    for sym, count in results.items():
        print(f"  {sym}: {count} new bars")
