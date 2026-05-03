"""
Stage 2 — Feature Engineering
Computes technical indicators and stores enriched features per asset.
Uses pandas-ta for all indicator math — no hand-rolling.
"""

from __future__ import annotations
import os
import logging
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logging.warning("pandas-ta not installed — using manual indicator fallbacks")

from sqlalchemy import select, delete
from core import get_session, FeatureRecord

log = logging.getLogger(__name__)


# ─────────────────────── Indicator computation ──────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def compute_bollinger(close: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cumvol = df["volume"].cumsum()
    cumtpvol = (tp * df["volume"]).cumsum()
    return cumtpvol / cumvol.replace(0, np.nan)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw OHLCV DataFrame and returns a feature DataFrame.
    Expects columns: open, high, low, close, volume
    """
    if df.empty or len(df) < 30:
        log.warning("Not enough data to compute features (need 30+ bars)")
        return pd.DataFrame()

    feat = df.copy()

    # RSI
    feat["rsi"] = compute_rsi(feat["close"])

    # MACD
    feat["macd"], feat["macd_signal"] = compute_macd(feat["close"])

    # Bollinger Bands
    feat["bb_upper"], feat["bb_mid"], feat["bb_lower"] = compute_bollinger(feat["close"])

    # ATR
    feat["atr"] = compute_atr(feat["high"], feat["low"], feat["close"])

    # Volume Z-score (rolling 20-bar)
    vol_mean = feat["volume"].rolling(20).mean()
    vol_std = feat["volume"].rolling(20).std()
    feat["volume_zscore"] = (feat["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # Realised volatility (20-bar rolling std of log returns)
    log_ret = np.log(feat["close"] / feat["close"].shift(1))
    feat["realized_vol"] = log_ret.rolling(20).std() * np.sqrt(252)

    # Momentum
    feat["momentum_1d"]  = feat["close"].pct_change(1)
    feat["momentum_5d"]  = feat["close"].pct_change(5)
    feat["momentum_20d"] = feat["close"].pct_change(20)

    # VWAP
    feat["vwap"] = compute_vwap(feat)

    # Drop rows with all-NaN features
    feature_cols = [
        "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "bb_mid",
        "atr", "volume_zscore", "realized_vol",
        "momentum_1d", "momentum_5d", "momentum_20d", "vwap"
    ]
    feat = feat.dropna(subset=feature_cols, how="all")

    return feat[feature_cols]


# ─────────────────────── Storage ────────────────────────────────────

def save_features(symbol: str, feat_df: pd.DataFrame) -> int:
    """Upsert feature rows for a symbol."""
    if feat_df.empty:
        return 0

    with get_session() as session:
        # Delete existing features for symbol to refresh
        session.execute(delete(FeatureRecord).where(FeatureRecord.symbol == symbol))

        records = []
        for ts, row in feat_df.iterrows():
            records.append(FeatureRecord(
                symbol=symbol,
                timestamp=ts if isinstance(ts, datetime) else ts.to_pydatetime(),
                rsi=_safe(row.get("rsi")),
                macd=_safe(row.get("macd")),
                macd_signal=_safe(row.get("macd_signal")),
                bb_upper=_safe(row.get("bb_upper")),
                bb_lower=_safe(row.get("bb_lower")),
                bb_mid=_safe(row.get("bb_mid")),
                atr=_safe(row.get("atr")),
                volume_zscore=_safe(row.get("volume_zscore")),
                realized_vol=_safe(row.get("realized_vol")),
                momentum_1d=_safe(row.get("momentum_1d")),
                momentum_5d=_safe(row.get("momentum_5d")),
                momentum_20d=_safe(row.get("momentum_20d")),
                vwap=_safe(row.get("vwap")),
            ))

        session.add_all(records)
        session.commit()
        log.info(f"[{symbol}] Saved {len(records)} feature rows")
        return len(records)


def load_features(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Load feature rows as a DataFrame."""
    with get_session() as session:
        rows = session.execute(
            select(FeatureRecord)
            .where(FeatureRecord.symbol == symbol)
            .order_by(FeatureRecord.timestamp.desc())
            .limit(limit)
        ).scalars().all()

    if not rows:
        return pd.DataFrame()

    data = [{
        "timestamp": r.timestamp,
        "rsi": r.rsi,
        "macd": r.macd,
        "macd_signal": r.macd_signal,
        "bb_upper": r.bb_upper,
        "bb_lower": r.bb_lower,
        "bb_mid": r.bb_mid,
        "atr": r.atr,
        "volume_zscore": r.volume_zscore,
        "realized_vol": r.realized_vol,
        "momentum_1d": r.momentum_1d,
        "momentum_5d": r.momentum_5d,
        "momentum_20d": r.momentum_20d,
        "vwap": r.vwap,
    } for r in rows]

    df = pd.DataFrame(data).set_index("timestamp").sort_index()
    return df


def get_latest_features(symbol: str) -> Optional[dict]:
    """Get the most recent feature snapshot for a symbol as a dict."""
    df = load_features(symbol, limit=1)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def _safe(val) -> Optional[float]:
    """Convert NaN/inf to None for DB storage."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except Exception:
        return None


def run_feature_pipeline(symbols: list[str]) -> dict[str, int]:
    """Compute and save features for all symbols."""
    from ingestion.market_feed import load_ohlcv
    results = {}
    for symbol in symbols:
        df = load_ohlcv(symbol, days=90)
        if df.empty:
            log.warning(f"[{symbol}] No OHLCV data — skipping features")
            results[symbol] = 0
            continue
        feat = compute_features(df)
        saved = save_features(symbol, feat)
        results[symbol] = saved
    return results


if __name__ == "__main__":
    from core import init_db
    from ingestion.market_feed import DEFAULT_WATCHLIST
    init_db()
    print("Computing features for:", DEFAULT_WATCHLIST)
    results = run_feature_pipeline(DEFAULT_WATCHLIST)
    for sym, count in results.items():
        print(f"  {sym}: {count} feature rows")
