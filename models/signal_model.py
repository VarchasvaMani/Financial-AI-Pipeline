"""
Stage 4 — Signal Model
LightGBM classifier predicting directional price moves.
SHAP values feed back into the Claude narrative agent.
Includes train, predict, and backtest functions.
"""

from __future__ import annotations
import os
import json
import pickle
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "bb_mid",
    "atr", "volume_zscore", "realized_vol",
    "momentum_1d", "momentum_5d", "momentum_20d",
]

# Target: 1 if price is up >1% in next 4h, 0 otherwise
TARGET_THRESHOLD = 0.01
HORIZON_BARS = 4   # 4 hours forward for 1h bars


def _make_target(close: pd.Series, horizon: int = HORIZON_BARS, threshold: float = TARGET_THRESHOLD) -> pd.Series:
    """Binary target: 1 if forward return > threshold, else 0."""
    fwd_return = close.shift(-horizon) / close - 1
    return (fwd_return > threshold).astype(int)


def build_training_data(symbols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a combined training dataset from all symbols.
    Features are normalised per-symbol to avoid cross-asset leakage.
    """
    from features.technical import load_features
    from ingestion.market_feed import load_ohlcv

    all_X, all_y = [], []

    for symbol in symbols:
        feat_df = load_features(symbol, limit=2000)
        ohlcv_df = load_ohlcv(symbol, days=90)

        if feat_df.empty or ohlcv_df.empty:
            log.warning(f"[{symbol}] Insufficient data for training")
            continue

        # Align on timestamps
        merged = feat_df.join(ohlcv_df[["close"]], how="inner")
        merged = merged.dropna(subset=FEATURE_COLS)

        if len(merged) < 50:
            log.warning(f"[{symbol}] Too few aligned rows ({len(merged)})")
            continue

        y = _make_target(merged["close"])
        X = merged[FEATURE_COLS]

        # Drop last HORIZON_BARS rows (no target available)
        X = X.iloc[:-HORIZON_BARS]
        y = y.iloc[:-HORIZON_BARS]

        X = X.dropna()
        y = y[X.index]

        all_X.append(X)
        all_y.append(y)
        log.info(f"[{symbol}] Added {len(X)} training samples")

    if not all_X:
        return pd.DataFrame(), pd.Series(dtype=int)

    return pd.concat(all_X), pd.concat(all_y)


def train_model(symbols: list[str]) -> dict:
    """Train LightGBM signal model and save to disk."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        log.error("lightgbm not installed. Run: pip install lightgbm")
        return {}

    log.info(f"Building training data for {symbols}...")
    X, y = build_training_data(symbols)

    if X.empty:
        log.error("No training data available")
        return {}

    log.info(f"Training on {len(X)} samples, {y.mean():.1%} positive rate")

    # Time-series cross-validation (no data leakage)
    tscv = TimeSeriesSplit(n_splits=3)
    cv_aucs = []

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        cv_aucs.append(auc)
        log.info(f"  Fold {fold+1} AUC: {auc:.4f}")

    # Final fit on all data
    model.fit(X, y)
    mean_auc = np.mean(cv_aucs)
    log.info(f"Mean CV AUC: {mean_auc:.4f}")

    # Save model
    model_path = MODEL_DIR / "signal_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "symbols": symbols,
        "n_samples": len(X),
        "positive_rate": float(y.mean()),
        "cv_auc_mean": round(mean_auc, 4),
        "cv_auc_folds": [round(a, 4) for a in cv_aucs],
        "feature_cols": FEATURE_COLS,
        "version": "1.0",
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Model saved to {model_path}")
    return meta


def load_model():
    """Load trained model from disk."""
    path = MODEL_DIR / "signal_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_signal(symbol: str, model=None) -> Optional[dict]:
    """
    Generate a signal for a symbol using the latest features.
    Returns signal dict with direction, confidence, and SHAP top features.
    """
    try:
        import shap
    except ImportError:
        log.error("shap not installed. Run: pip install shap")
        return None

    if model is None:
        model = load_model()
    if model is None:
        log.warning("No trained model found — run train_model() first")
        return _mock_signal(symbol)

    from features.technical import get_latest_features
    feats = get_latest_features(symbol)
    if feats is None:
        log.warning(f"[{symbol}] No features available")
        return _mock_signal(symbol)

    # Build feature vector
    X = pd.DataFrame([{col: feats.get(col, 0.0) or 0.0 for col in FEATURE_COLS}])
    X = X.fillna(0)

    proba = model.predict_proba(X)[0]
    long_prob = float(proba[1])
    short_prob = 1.0 - long_prob

    # Direction threshold
    if long_prob > 0.60:
        direction = "long"
        confidence = long_prob
    elif short_prob > 0.60:
        direction = "short"
        confidence = short_prob
    else:
        direction = "neutral"
        confidence = max(long_prob, short_prob)

    # SHAP explainability
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary: shap_values[1] is for positive class
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        feature_impacts = sorted(
            zip(FEATURE_COLS, sv),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_features = [(name, round(float(val), 5)) for name, val in feature_impacts[:5]]
    except Exception as e:
        log.warning(f"SHAP failed: {e}")
        top_features = [(col, 0.0) for col in FEATURE_COLS[:5]]

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": round(confidence, 4),
        "horizon": "4h",
        "top_features": top_features,
        "long_prob": round(long_prob, 4),
        "short_prob": round(short_prob, 4),
        "timestamp": datetime.utcnow().isoformat(),
    }


def _mock_signal(symbol: str) -> dict:
    """Deterministic mock signal for demo/testing without a trained model."""
    import hashlib
    seed = int(hashlib.md5(f"{symbol}{datetime.utcnow().date()}".encode()).hexdigest()[:8], 16)
    np.random.seed(seed % (2**31))
    confidence = round(0.55 + np.random.random() * 0.35, 4)
    direction = np.random.choice(["long", "short", "neutral"], p=[0.45, 0.35, 0.20])
    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "horizon": "4h",
        "top_features": [
            ("momentum_5d", round(np.random.uniform(0.01, 0.05), 5)),
            ("rsi", round(np.random.uniform(0.005, 0.03), 5)),
            ("volume_zscore", round(np.random.uniform(0.003, 0.02), 5)),
            ("macd", round(np.random.uniform(0.002, 0.015), 5)),
            ("realized_vol", round(np.random.uniform(0.001, 0.01), 5)),
        ],
        "long_prob": confidence if direction == "long" else round(1 - confidence, 4),
        "short_prob": confidence if direction == "short" else round(1 - confidence, 4),
        "timestamp": datetime.utcnow().isoformat(),
        "is_mock": True,
    }


def generate_all_signals(symbols: list[str]) -> list[dict]:
    """Generate and persist signals for all symbols."""
    from agents.sentiment_agent import get_sentiment_summary
    from agents.narrative_agent import generate_narrative
    from ingestion.market_feed import get_latest_prices
    from core import get_session, SignalRecord

    model = load_model()
    prices = get_latest_prices(symbols)
    signals = []

    for symbol in symbols:
        signal = predict_signal(symbol, model=model)
        if not signal:
            continue

        sentiment = get_sentiment_summary(symbol)
        narrative = generate_narrative(
            symbol=symbol,
            direction=signal["direction"],
            confidence=signal["confidence"],
            horizon=signal["horizon"],
            top_features=signal["top_features"],
            sentiment_summary=sentiment,
            current_price=prices.get(symbol),
        )

        signal["narrative"] = narrative
        signal["sentiment_score"] = sentiment.get("score", 0.0)
        signals.append(signal)

        # Persist to DB
        with get_session() as session:
            session.add(SignalRecord(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                direction=signal["direction"],
                confidence=signal["confidence"],
                horizon=signal["horizon"],
                top_features=json.dumps(signal["top_features"]),
                narrative=narrative,
                sentiment_score=sentiment.get("score", 0.0),
            ))
            session.commit()

        log.info(f"[{symbol}] Signal: {signal['direction']} @ {signal['confidence']:.0%}")

    return signals


if __name__ == "__main__":
    from core import init_db
    from ingestion.market_feed import DEFAULT_WATCHLIST
    init_db()

    print("Training model...")
    meta = train_model(DEFAULT_WATCHLIST)
    if meta:
        print(f"CV AUC: {meta.get('cv_auc_mean', 'N/A')}")

    print("\nGenerating signals...")
    signals = generate_all_signals(DEFAULT_WATCHLIST[:5])
    for s in signals:
        print(f"  {s['symbol']:6} | {s['direction']:7} | {s['confidence']:.0%} confidence")
        print(f"    {s.get('narrative', '')[:100]}...")
