"""
Stage 3b — Narrative Agent
Uses Claude to generate plain-English explanations for every signal.
This is the key differentiator — traders see WHY, not just WHAT.
"""

from __future__ import annotations
import os
import logging
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

NARRATIVE_SYSTEM = """You are a senior quantitative analyst explaining trading signals to portfolio managers.
Write a 2-3 sentence explanation that covers:
1. The signal direction and the top technical reasons for it
2. Relevant recent news context if available
3. The key risk to watch

Be specific with numbers. Be direct. No fluff. Write in present tense."""


def generate_narrative(
    symbol: str,
    direction: str,
    confidence: float,
    horizon: str,
    top_features: list[tuple[str, float]],
    sentiment_summary: dict,
    current_price: Optional[float] = None,
) -> str:
    """
    Generate a Claude-powered plain-English explanation for a signal.
    Falls back to a template if API key not set.
    """
    if not ANTHROPIC_API_KEY:
        return _template_fallback(symbol, direction, confidence, top_features, sentiment_summary)

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build context for Claude
    feature_text = "\n".join(
        f"  - {name}: {value:+.4f} (SHAP contribution)"
        for name, value in top_features[:5]
    )

    news_text = "No recent news available."
    if sentiment_summary.get("headlines"):
        headlines = sentiment_summary["headlines"][:3]
        news_text = "\n".join(
            f"  [{h['sentiment'].upper()} mag={h['magnitude']}] {h['headline']}"
            for h in headlines
        )

    price_text = f"Current price: ${current_price:.2f}" if current_price else ""

    prompt = f"""Generate a trading signal explanation.

Symbol: {symbol}
Signal: {direction.upper()} | Confidence: {confidence:.0%} | Horizon: {horizon}
{price_text}

Top model features driving this signal (SHAP values):
{feature_text}

Recent news sentiment: {sentiment_summary.get('dominant', 'neutral')} (score: {sentiment_summary.get('score', 0):.1f})
Recent headlines:
{news_text}

Write the 2-3 sentence analyst explanation now:"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=NARRATIVE_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        log.error(f"Narrative generation failed: {e}")
        return _template_fallback(symbol, direction, confidence, top_features, sentiment_summary)


def _template_fallback(
    symbol: str,
    direction: str,
    confidence: float,
    top_features: list[tuple[str, float]],
    sentiment_summary: dict,
) -> str:
    """Template-based fallback when Claude is unavailable."""
    top_feat = top_features[0][0].replace("_", " ") if top_features else "technical indicators"
    sentiment = sentiment_summary.get("dominant", "neutral")
    score = sentiment_summary.get("score", 0)

    arrow = "↑" if direction == "long" else "↓" if direction == "short" else "→"

    return (
        f"{arrow} {direction.upper()} signal on {symbol} with {confidence:.0%} confidence "
        f"over {top_features[0][0] if top_features else 'next period'}. "
        f"Primary driver: {top_feat}. "
        f"News sentiment is {sentiment} (score: {score:+.1f}). "
        f"Monitor for reversal if confidence drops below 60%."
    )


def generate_anomaly_narrative(
    symbol: str,
    anomaly_type: str,
    zscore: float,
    recent_news: list[dict],
) -> str:
    """Generate an explanation for a detected anomaly."""
    if not ANTHROPIC_API_KEY:
        return (
            f"Anomaly detected on {symbol}: {anomaly_type} "
            f"(z-score: {zscore:.1f}x normal). "
            f"{'Recent news may be a catalyst.' if recent_news else 'No clear news catalyst found.'}"
        )

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    news_text = "\n".join(f"  - {n.get('headline', '')}" for n in recent_news[:3]) or "None found"

    prompt = f"""Anomaly detected:
Symbol: {symbol}
Type: {anomaly_type}
Z-score: {zscore:.1f}x normal baseline

Recent news:
{news_text}

Write one sentence explaining what likely caused this anomaly and whether it warrants action:"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        log.error(f"Anomaly narrative failed: {e}")
        return f"Anomaly on {symbol}: {anomaly_type} at {zscore:.1f}x normal. Review recent news."


def answer_nl_query(query: str, signals_context: str) -> str:
    """
    Natural language query interface for the dashboard.
    Allows analysts to ask questions like 'show me all bullish signals above 0.7'
    """
    if not ANTHROPIC_API_KEY:
        return "Natural language queries require an Anthropic API key."

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""You are a financial data assistant. 
The user is asking about signals in our pipeline.

Available signals data:
{signals_context}

User query: {query}

Answer concisely and directly, referencing specific data where available:"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Query failed: {e}"


if __name__ == "__main__":
    # Test narrative generation
    narrative = generate_narrative(
        symbol="NVDA",
        direction="long",
        confidence=0.82,
        horizon="4h",
        top_features=[
            ("momentum_5d", 0.031),
            ("rsi", 0.018),
            ("volume_zscore", 0.015),
            ("macd", 0.012),
        ],
        sentiment_summary={
            "score": 3.2,
            "dominant": "bullish",
            "headlines": [
                {"headline": "NVIDIA data centre revenue hits record $18B", "sentiment": "bullish", "magnitude": 5, "source": "Reuters"},
                {"headline": "MS upgrades NVDA to overweight, raises PT to $900", "sentiment": "bullish", "magnitude": 3, "source": "Bloomberg"},
            ]
        },
        current_price=875.40,
    )
    print("Generated narrative:\n")
    print(narrative)
