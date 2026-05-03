"""
Stage 3a — AI Sentiment Agent
Uses Claude to score news headlines with financial context.
Falls back to keyword-based scoring when API key is not set.
"""

from __future__ import annotations
import os
import json
import logging
import time
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import select

from core import get_session, NewsRecord, SentimentResult, init_db

load_dotenv()
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ─────────────────────── Claude sentiment ───────────────────────────

SENTIMENT_SYSTEM = """You are a financial news analyst. 
Analyse headlines and return ONLY valid JSON. No markdown, no explanation.
JSON schema: {"sentiment": "bullish|bearish|neutral", "magnitude": 1-5, "theme": "string", "confidence": 0.0-1.0}
magnitude: 1=minor, 5=major market-moving. theme: short label like "earnings beat", "rate hike", "M&A", etc."""


def analyse_with_claude(headline: str, symbol: str) -> SentimentResult:
    """Score a headline using Claude. Returns SentimentResult."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"Ticker: {symbol}\nHeadline: {headline}"

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            system=SENTIMENT_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        return SentimentResult(
            sentiment=data.get("sentiment", "neutral"),
            magnitude=int(data.get("magnitude", 2)),
            theme=data.get("theme", "general"),
            confidence=float(data.get("confidence", 0.5)),
        )
    except Exception as e:
        log.error(f"Claude sentiment error: {e}")
        return _keyword_fallback(headline)


def _keyword_fallback(headline: str) -> SentimentResult:
    """Simple keyword-based fallback when Claude is unavailable."""
    text = headline.lower()
    bullish_words = ["beat", "surge", "rally", "upgrade", "record", "profit",
                     "growth", "buyback", "dividend", "above", "strong", "gain"]
    bearish_words = ["miss", "drop", "fall", "downgrade", "loss", "decline",
                     "below", "weak", "cut", "warn", "lawsuit", "probe"]

    bull_score = sum(1 for w in bullish_words if w in text)
    bear_score = sum(1 for w in bearish_words if w in text)

    if bull_score > bear_score:
        return SentimentResult(sentiment="bullish", magnitude=min(bull_score, 5),
                               theme="general positive", confidence=0.4)
    elif bear_score > bull_score:
        return SentimentResult(sentiment="bearish", magnitude=min(bear_score, 5),
                               theme="general negative", confidence=0.4)
    return SentimentResult(sentiment="neutral", magnitude=1,
                           theme="no clear signal", confidence=0.3)


# ─────────────────────── Batch processing ───────────────────────────

def process_unscored_news(limit: int = 30, delay: float = 0.5) -> int:
    """Score all unprocessed news headlines and update the DB."""
    with get_session() as session:
        rows = session.execute(
            select(NewsRecord)
            .where(NewsRecord.processed == False)
            .order_by(NewsRecord.timestamp.desc())
            .limit(limit)
        ).scalars().all()

        if not rows:
            log.info("No unprocessed news found")
            return 0

        use_claude = bool(ANTHROPIC_API_KEY)
        log.info(f"Scoring {len(rows)} headlines ({'Claude' if use_claude else 'keyword fallback'})")

        processed = 0
        for row in rows:
            if use_claude:
                result = analyse_with_claude(row.headline, row.symbol)
                time.sleep(delay)
            else:
                result = _keyword_fallback(row.headline)

            row.sentiment = result.sentiment
            row.magnitude = result.magnitude
            row.theme = result.theme
            row.confidence = result.confidence
            row.processed = True
            processed += 1

        session.commit()
        log.info(f"Scored {processed} headlines")
        return processed


def get_sentiment_summary(symbol: str, days: int = 2) -> dict:
    """Aggregate recent sentiment scores for a symbol."""
    from ingestion.news_feed import load_recent_news
    news = load_recent_news(symbol, days=days)

    if not news:
        return {"score": 0.0, "count": 0, "dominant": "neutral", "headlines": []}

    scores = []
    for n in news:
        if n.sentiment == "bullish":
            scores.append((n.magnitude or 1) * (n.confidence or 0.5))
        elif n.sentiment == "bearish":
            scores.append(-1 * (n.magnitude or 1) * (n.confidence or 0.5))
        else:
            scores.append(0.0)

    net = sum(scores)
    dominant = "neutral"
    if net > 1: dominant = "bullish"
    elif net < -1: dominant = "bearish"

    headlines = [
        {"headline": n.headline, "sentiment": n.sentiment,
         "magnitude": n.magnitude, "source": n.source}
        for n in news[:5]
    ]

    return {
        "score": round(net, 2),
        "count": len(news),
        "dominant": dominant,
        "headlines": headlines,
    }


if __name__ == "__main__":
    init_db()
    from ingestion.news_feed import ingest_news
    from ingestion.market_feed import DEFAULT_WATCHLIST

    print("Ingesting news...")
    ingest_news(DEFAULT_WATCHLIST)

    print("Scoring with sentiment agent...")
    count = process_unscored_news(limit=50)
    print(f"Scored {count} headlines")

    for sym in DEFAULT_WATCHLIST[:3]:
        summary = get_sentiment_summary(sym)
        print(f"\n{sym}: score={summary['score']}, dominant={summary['dominant']}, count={summary['count']}")
