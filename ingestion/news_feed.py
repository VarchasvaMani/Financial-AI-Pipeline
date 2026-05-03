"""
Stage 1b — News & Alternative Data Ingestion
Fetches news headlines from yfinance (free) and Alpha Vantage (free tier).
"""

from __future__ import annotations
import os
import logging
import time
import requests
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select
from dotenv import load_dotenv

from core import get_session, init_db, NewsRecord

load_dotenv()
log = logging.getLogger(__name__)

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")


def fetch_news_yfinance(symbol: str) -> list[dict]:
    """Fetch recent news via yfinance (no API key needed)."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        results = []
        for item in news[:20]:
            results.append({
                "symbol": symbol,
                "timestamp": datetime.fromtimestamp(item.get("providerPublishTime", 0)),
                "headline": item.get("title", ""),
                "source": item.get("publisher", ""),
                "url": item.get("link", ""),
            })
        log.info(f"[{symbol}] Fetched {len(results)} headlines from yfinance")
        return results
    except Exception as e:
        log.error(f"[{symbol}] yfinance news failed: {e}")
        return []


def fetch_news_alphavantage(symbol: str) -> list[dict]:
    """Fetch news sentiment from Alpha Vantage (free tier: 25 req/day)."""
    if ALPHA_VANTAGE_KEY == "demo":
        log.warning("No Alpha Vantage key set — skipping AV news fetch")
        return []

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "limit": 50,
        "apikey": ALPHA_VANTAGE_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("feed", []):
            ts_str = item.get("time_published", "")
            try:
                ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%S")
            except Exception:
                ts = datetime.utcnow()

            results.append({
                "symbol": symbol,
                "timestamp": ts,
                "headline": item.get("title", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
            })

        log.info(f"[{symbol}] Fetched {len(results)} headlines from Alpha Vantage")
        return results
    except Exception as e:
        log.error(f"[{symbol}] Alpha Vantage news failed: {e}")
        return []


def save_news(articles: list[dict]) -> int:
    """Save news articles to DB, skip duplicates by headline+symbol."""
    if not articles:
        return 0

    with get_session() as session:
        # Load existing headlines to avoid duplicates
        existing = set(
            row[0] for row in session.execute(
                select(NewsRecord.headline)
            ).fetchall()
        )

        new_records = []
        for a in articles:
            headline = a.get("headline", "").strip()
            if headline and headline not in existing:
                new_records.append(NewsRecord(
                    symbol=a["symbol"],
                    timestamp=a.get("timestamp", datetime.utcnow()),
                    headline=headline,
                    source=a.get("source", ""),
                    url=a.get("url", ""),
                    processed=False,
                ))
                existing.add(headline)

        session.add_all(new_records)
        session.commit()
        log.info(f"Saved {len(new_records)} new articles")
        return len(new_records)


def load_unprocessed_news(limit: int = 50) -> list[NewsRecord]:
    """Fetch news items not yet sentiment-scored."""
    with get_session() as session:
        rows = session.execute(
            select(NewsRecord)
            .where(NewsRecord.processed == False)
            .order_by(NewsRecord.timestamp.desc())
            .limit(limit)
        ).scalars().all()
        session.expunge_all()
        return rows


def load_recent_news(symbol: str, days: int = 3) -> list[NewsRecord]:
    """Fetch recent scored news for a symbol."""
    since = datetime.utcnow() - timedelta(days=days)
    with get_session() as session:
        rows = session.execute(
            select(NewsRecord)
            .where(NewsRecord.symbol == symbol)
            .where(NewsRecord.timestamp >= since)
            .where(NewsRecord.processed == True)
            .order_by(NewsRecord.timestamp.desc())
            .limit(10)
        ).scalars().all()
        session.expunge_all()
        return rows


def ingest_news(symbols: list[str], delay: float = 0.5) -> int:
    """Ingest news for all symbols from all sources."""
    total = 0
    for symbol in symbols:
        articles = fetch_news_yfinance(symbol)
        articles += fetch_news_alphavantage(symbol)
        total += save_news(articles)
        time.sleep(delay)
    return total


if __name__ == "__main__":
    init_db()
    from ingestion.market_feed import DEFAULT_WATCHLIST
    count = ingest_news(DEFAULT_WATCHLIST)
    print(f"Ingested {count} new articles")
