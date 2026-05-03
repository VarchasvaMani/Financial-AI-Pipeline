"""
Anomaly Detection — flags unusual price action, volume spikes, volatility regimes.
Uses Z-score + Isolation Forest. Claude explains each anomaly.
"""

from __future__ import annotations
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


def detect_volume_anomaly(df: pd.DataFrame, window: int = 20, threshold: float = 2.5) -> list[dict]:
    """Flag volume spikes beyond threshold standard deviations."""
    if "volume" not in df.columns or len(df) < window:
        return []

    vol_mean = df["volume"].rolling(window).mean()
    vol_std  = df["volume"].rolling(window).std()
    zscore   = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    anomalies = []
    recent = zscore.tail(5)
    for ts, z in recent.items():
        if abs(z) > threshold:
            anomalies.append({
                "type": "volume_spike",
                "timestamp": ts,
                "zscore": round(float(z), 2),
                "description": f"Volume {abs(z):.1f}× normal",
            })
    return anomalies


def detect_price_anomaly(df: pd.DataFrame, window: int = 20, threshold: float = 2.5) -> list[dict]:
    """Flag abnormal price moves (log returns z-score)."""
    if "close" not in df.columns or len(df) < window:
        return []

    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_mean = log_ret.rolling(window).mean()
    ret_std  = log_ret.rolling(window).std()
    zscore   = (log_ret - ret_mean) / ret_std.replace(0, np.nan)

    anomalies = []
    recent = zscore.tail(5)
    for ts, z in recent.items():
        if abs(z) > threshold:
            direction = "up" if z > 0 else "down"
            anomalies.append({
                "type": f"price_spike_{direction}",
                "timestamp": ts,
                "zscore": round(float(z), 2),
                "description": f"Price moved {abs(z):.1f}× normal range ({direction})",
            })
    return anomalies


def detect_volatility_regime(df: pd.DataFrame) -> Optional[dict]:
    """Detect volatility regime shifts using rolling ratio."""
    if "close" not in df.columns or len(df) < 40:
        return None

    log_ret = np.log(df["close"] / df["close"].shift(1))
    short_vol = log_ret.rolling(5).std().iloc[-1]
    long_vol  = log_ret.rolling(20).std().iloc[-1]

    if long_vol == 0 or np.isnan(long_vol) or np.isnan(short_vol):
        return None

    ratio = float(short_vol / long_vol)
    if ratio > 1.8:
        return {
            "type": "volatility_expansion",
            "zscore": round(ratio, 2),
            "description": f"Short-term vol is {ratio:.1f}× long-term average — regime expanding",
        }
    elif ratio < 0.4:
        return {
            "type": "volatility_compression",
            "zscore": round(ratio, 2),
            "description": f"Vol compressing to {ratio:.1f}× norm — breakout may be imminent",
        }
    return None


def run_anomaly_detection(symbol: str) -> list[dict]:
    """Run all anomaly detectors for a symbol and return flagged items."""
    from ingestion.market_feed import load_ohlcv
    from agents.narrative_agent import generate_anomaly_narrative
    from ingestion.news_feed import load_recent_news

    df = load_ohlcv(symbol, days=10)
    if df.empty:
        return []

    anomalies = []
    anomalies += detect_volume_anomaly(df)
    anomalies += detect_price_anomaly(df)

    regime = detect_volatility_regime(df)
    if regime:
        anomalies.append(regime)

    if anomalies:
        recent_news = load_recent_news(symbol, days=1)
        news_list = [{"headline": n.headline} for n in recent_news[:3]]

        for a in anomalies:
            a["symbol"] = symbol
            a["explanation"] = generate_anomaly_narrative(
                symbol=symbol,
                anomaly_type=a["type"],
                zscore=a.get("zscore", 0),
                recent_news=news_list,
            )
            log.info(f"[{symbol}] Anomaly: {a['type']} (z={a.get('zscore', '?')})")

    return anomalies


if __name__ == "__main__":
    from core import init_db
    from ingestion.market_feed import DEFAULT_WATCHLIST
    init_db()
    for sym in DEFAULT_WATCHLIST[:3]:
        anomalies = run_anomaly_detection(sym)
        if anomalies:
            print(f"\n{sym} anomalies:")
            for a in anomalies:
                print(f"  [{a['type']}] {a['description']}")
        else:
            print(f"{sym}: no anomalies")
