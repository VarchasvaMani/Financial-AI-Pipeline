"""
Test suite for the Financial AI Pipeline.
Tests run without API keys using mock data.
Run: python -m pytest tests/ -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────── DB / Core tests ────────────────────────────────────

def test_db_init():
    from core import init_db, engine
    init_db()
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "ohlcv" in tables
    assert "features" in tables
    assert "news" in tables
    assert "signals" in tables


def test_market_event_validation():
    from core import MarketEvent
    e = MarketEvent(
        symbol="aapl",
        timestamp=datetime.utcnow(),
        open=150.0, high=155.0, low=149.0, close=153.0, volume=1e7,
    )
    assert e.symbol == "AAPL"  # auto-uppercased


def test_sentiment_result_validation():
    from core import SentimentResult
    s = SentimentResult(sentiment="bullish", magnitude=3, theme="earnings", confidence=0.85)
    assert s.sentiment == "bullish"
    assert 0 <= s.confidence <= 1


# ─────────────── Ingestion tests ────────────────────────────────────

def test_fetch_ohlcv_returns_list():
    from ingestion.market_feed import fetch_ohlcv
    events = fetch_ohlcv("AAPL", period="5d", interval="1d")
    assert isinstance(events, list)
    if events:
        assert events[0].symbol == "AAPL"
        assert events[0].close > 0


def test_save_and_load_ohlcv_synthetic():
    from ingestion.market_feed import save_ohlcv, load_ohlcv
    from core import init_db, MarketEvent
    init_db()
    events = [
        MarketEvent(
            symbol="SYNTHETIC",
            timestamp=datetime.utcnow() - timedelta(hours=i),
            open=100.0+i, high=101.0+i, low=99.0+i,
            close=100.5+i, volume=1_000_000.0,
        )
        for i in range(10)
    ]
    saved = save_ohlcv(events, "SYNTHETIC")
    assert saved >= 0
    df = load_ohlcv("SYNTHETIC", days=1)
    assert not df.empty
    assert "close" in df.columns


def test_news_save_dedup():
    from ingestion.news_feed import save_news
    from core import init_db
    import uuid
    init_db()
    unique = f"Unique dedup test headline {uuid.uuid4().hex}"
    articles = [
        {"symbol": "TEST", "headline": unique,
         "timestamp": datetime.utcnow(), "source": "test", "url": ""},
        {"symbol": "TEST", "headline": unique,  # duplicate
         "timestamp": datetime.utcnow(), "source": "test", "url": ""},
    ]
    saved = save_news(articles)
    assert saved == 1  # only one saved, duplicate skipped


# ─────────────── Feature tests ──────────────────────────────────────

def _make_sample_ohlcv(n: int = 100) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.utcnow(), periods=n, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high  = close + np.abs(np.random.randn(n) * 0.3)
    low   = close - np.abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1e6, 1e7, n).astype(float),
    }, index=dates)


def test_compute_features_shape():
    from features.technical import compute_features
    df = _make_sample_ohlcv(100)
    feat = compute_features(df)
    assert not feat.empty
    assert "rsi" in feat.columns
    assert "macd" in feat.columns
    assert "momentum_5d" in feat.columns


def test_rsi_bounds():
    from features.technical import compute_rsi
    close = pd.Series(100 + np.cumsum(np.random.randn(200)))
    rsi = compute_rsi(close).dropna()
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def test_features_no_inf():
    from features.technical import compute_features
    df = _make_sample_ohlcv(100)
    feat = compute_features(df)
    assert not feat.isin([float("inf"), float("-inf")]).any().any()


def test_save_and_load_features():
    from features.technical import compute_features, save_features, load_features
    from ingestion.market_feed import save_ohlcv
    from core import init_db, MarketEvent
    init_db()
    df = _make_sample_ohlcv(100)
    feat = compute_features(df)
    saved = save_features("TEST_SYM", feat)
    assert saved > 0
    loaded = load_features("TEST_SYM")
    assert not loaded.empty


# ─────────────── Sentiment agent tests ──────────────────────────────

def test_keyword_fallback_bullish():
    from agents.sentiment_agent import _keyword_fallback
    result = _keyword_fallback("AAPL earnings beat consensus by 15%, stock surges")
    assert result.sentiment == "bullish"
    assert result.magnitude >= 1


def test_keyword_fallback_bearish():
    from agents.sentiment_agent import _keyword_fallback
    result = _keyword_fallback("Company warns of revenue miss, stock drops on weak guidance")
    assert result.sentiment == "bearish"


def test_keyword_fallback_neutral():
    from agents.sentiment_agent import _keyword_fallback
    result = _keyword_fallback("Company holds annual shareholder meeting")
    assert result.sentiment == "neutral"


# ─────────────── Narrative agent tests ──────────────────────────────

def test_template_fallback_narrative():
    from agents.narrative_agent import _template_fallback
    narrative = _template_fallback(
        symbol="NVDA",
        direction="long",
        confidence=0.82,
        top_features=[("momentum_5d", 0.03), ("rsi", 0.02)],
        sentiment_summary={"score": 2.5, "dominant": "bullish"},
    )
    assert "NVDA" in narrative
    assert len(narrative) > 20


# ─────────────── Signal model tests ─────────────────────────────────

def test_mock_signal_structure():
    from models.signal_model import _mock_signal
    sig = _mock_signal("AAPL")
    assert sig["symbol"] == "AAPL"
    assert sig["direction"] in ["long", "short", "neutral"]
    assert 0 <= sig["confidence"] <= 1
    assert len(sig["top_features"]) > 0


def test_mock_signal_deterministic():
    from models.signal_model import _mock_signal
    # Same symbol + date should give same result
    s1 = _mock_signal("TSLA")
    s2 = _mock_signal("TSLA")
    assert s1["direction"] == s2["direction"]


# ─────────────── Anomaly detector tests ─────────────────────────────

def test_volume_anomaly_detection():
    from models.anomaly_detector import detect_volume_anomaly
    df = _make_sample_ohlcv(50)
    # Inject a clear volume spike
    df.iloc[-1, df.columns.get_loc("volume")] = df["volume"].mean() * 10
    anomalies = detect_volume_anomaly(df, threshold=2.0)
    assert len(anomalies) > 0
    assert anomalies[0]["type"] == "volume_spike"


def test_no_anomaly_on_flat_data():
    from models.anomaly_detector import detect_volume_anomaly
    dates = pd.date_range(end=datetime.utcnow(), periods=50, freq="1h")
    df = pd.DataFrame({
        "volume": [1_000_000.0] * 50,
        "close": [100.0] * 50,
        "high": [101.0] * 50,
        "low": [99.0] * 50,
    }, index=dates)
    anomalies = detect_volume_anomaly(df)
    assert len(anomalies) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
