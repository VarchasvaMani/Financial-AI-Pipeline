"""
Pipeline Orchestrator
Runs all stages in sequence: ingest → features → sentiment → signals.
Can be run once or scheduled (e.g. via cron or APScheduler).
"""

from __future__ import annotations
import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_WATCHLIST = os.getenv(
    "DEFAULT_WATCHLIST",
    "AAPL,MSFT,GOOGL,NVDA,TSLA,AMZN,META,JPM,GS,SPY"
).split(",")


def run_full_pipeline(symbols: list[str] = DEFAULT_WATCHLIST, skip_train: bool = True) -> dict:
    """Run the complete pipeline end-to-end."""
    results = {}
    start = time.time()

    # ── 0. DB init ──────────────────────────────────────────────────
    from core import init_db
    init_db()
    log.info("=" * 55)
    log.info(f"PIPELINE START — {len(symbols)} symbols")
    log.info("=" * 55)

    # ── 1. Ingest market data ────────────────────────────────────────
    log.info("Stage 1a: Market data ingestion")
    from ingestion.market_feed import ingest_watchlist
    ohlcv_results = ingest_watchlist(symbols, period="90d", interval="1h")
    results["ohlcv_bars"] = sum(ohlcv_results.values())
    log.info(f"  → {results['ohlcv_bars']} new OHLCV bars ingested")

    # ── 1b. Ingest news ──────────────────────────────────────────────
    log.info("Stage 1b: News ingestion")
    from ingestion.news_feed import ingest_news
    news_count = ingest_news(symbols)
    results["news_articles"] = news_count
    log.info(f"  → {news_count} new articles ingested")

    # ── 2. Feature engineering ───────────────────────────────────────
    log.info("Stage 2: Feature engineering")
    from features.technical import run_feature_pipeline
    feat_results = run_feature_pipeline(symbols)
    results["feature_rows"] = sum(feat_results.values())
    log.info(f"  → {results['feature_rows']} feature rows computed")

    # ── 3a. Sentiment scoring ────────────────────────────────────────
    log.info("Stage 3a: Sentiment scoring")
    from agents.sentiment_agent import process_unscored_news
    scored = process_unscored_news(limit=100)
    results["news_scored"] = scored
    log.info(f"  → {scored} headlines scored")

    # ── 4. Train model (optional) ────────────────────────────────────
    if not skip_train:
        log.info("Stage 4: Model training")
        from models.signal_model import train_model
        meta = train_model(symbols)
        results["model_auc"] = meta.get("cv_auc_mean", 0)
        log.info(f"  → Model AUC: {results['model_auc']:.4f}")

    # ── 5. Signal generation ─────────────────────────────────────────
    log.info("Stage 5: Signal generation")
    from models.signal_model import generate_all_signals
    signals = generate_all_signals(symbols)
    results["signals_generated"] = len(signals)
    log.info(f"  → {len(signals)} signals generated")

    # ── 6. Anomaly detection ─────────────────────────────────────────
    log.info("Stage 6: Anomaly detection")
    from models.anomaly_detector import run_anomaly_detection
    all_anomalies = []
    for sym in symbols:
        all_anomalies += run_anomaly_detection(sym)
    results["anomalies"] = len(all_anomalies)
    log.info(f"  → {len(all_anomalies)} anomalies detected")

    elapsed = round(time.time() - start, 1)
    results["elapsed_seconds"] = elapsed

    log.info("=" * 55)
    log.info(f"PIPELINE COMPLETE in {elapsed}s")
    log.info("=" * 55)

    # Print signal summary
    if signals:
        log.info("\nSignal Summary:")
        for s in sorted(signals, key=lambda x: x["confidence"], reverse=True):
            flag = "🔺" if s["direction"] == "long" else "🔻" if s["direction"] == "short" else "⬜"
            mock = " [mock]" if s.get("is_mock") else ""
            log.info(
                f"  {flag} {s['symbol']:6} {s['direction']:7} "
                f"{s['confidence']:.0%} conf{mock}"
            )

    return results


def run_demo_mode(symbols: list[str] = None) -> None:
    """Run pipeline with mock data for demo/testing (no API keys needed)."""
    symbols = symbols or ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
    log.info("Running in DEMO mode (mock signals, no API keys required)")
    run_full_pipeline(symbols, skip_train=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial AI Pipeline")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_WATCHLIST)
    parser.add_argument("--train", action="store_true", help="Retrain model")
    parser.add_argument("--demo", action="store_true", help="Demo mode with small symbol list")
    args = parser.parse_args()

    if args.demo:
        run_demo_mode()
    else:
        results = run_full_pipeline(args.symbols, skip_train=not args.train)
        print("\nResults:", json.dumps(results, indent=2))
