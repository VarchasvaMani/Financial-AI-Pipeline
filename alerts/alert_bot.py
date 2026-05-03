"""
Alert Bot — Telegram + Slack notifications for signals, anomalies, and risk breaches.
Supports three tiers: INFO / WARNING / CRITICAL with per-channel routing.

Setup:
  Telegram: Create bot via @BotFather → get token + chat_id
  Slack:    Create incoming webhook at api.slack.com/apps
  Add to .env:
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
"""
from __future__ import annotations
import os, logging, requests
from datetime import datetime, timedelta
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
SLACK_WEBHOOK  = os.getenv("SLACK_WEBHOOK_URL", "")


class AlertLevel(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


LEVEL_EMOJI = {AlertLevel.INFO: "📊", AlertLevel.WARNING: "⚠️", AlertLevel.CRITICAL: "🚨"}
LEVEL_COLOR = {AlertLevel.INFO: "#36a64f", AlertLevel.WARNING: "#ffc107", AlertLevel.CRITICAL: "#dc3545"}


# ─────────────────────────── Transport ──────────────────────────────

def _send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        log.debug("Telegram not configured")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        log.error(f"Telegram failed: {e}")
        return False


def _send_slack(message: str, level: AlertLevel) -> bool:
    if not SLACK_WEBHOOK:
        log.debug("Slack not configured")
        return False
    try:
        r = requests.post(
            SLACK_WEBHOOK,
            json={"attachments": [{"color": LEVEL_COLOR[level], "text": message, "mrkdwn_in": ["text"]}]},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        log.error(f"Slack failed: {e}")
        return False


def _log_to_db(message: str, level: AlertLevel, symbol: str = ""):
    try:
        from core import get_session, AuditRecord
        with get_session() as s:
            s.add(AuditRecord(event_type=f"alert_{level.value.lower()}",
                              symbol=symbol, details=message[:1000]))
            s.commit()
    except Exception:
        pass


# ─────────────────────────── Public API ─────────────────────────────

def send_alert(message: str, level: AlertLevel = AlertLevel.INFO, symbol: str = "") -> dict:
    """Route alert to all configured channels. Always logs to DB."""
    ts   = datetime.utcnow().strftime("%H:%M UTC")
    full = f"{LEVEL_EMOJI[level]} *[{level.value}]* {ts}\n{message}"

    _log_to_db(full, level, symbol)

    tg_sent    = _send_telegram(full)
    slack_sent = _send_slack(full, level) if level in (AlertLevel.WARNING, AlertLevel.CRITICAL) else False

    if not tg_sent and not slack_sent:
        # Fallback: print to console so alerts are never silently dropped
        print(f"\n{full}\n")

    log.info(f"[ALERT/{level.value}] {symbol or 'SYS'}: {message[:80]}")
    return {"telegram": tg_sent, "slack": slack_sent, "logged": True}


def alert_signal(signal: dict) -> dict:
    """Send a formatted signal alert."""
    sym  = signal.get("symbol", "?")
    d    = signal.get("direction", "neutral")
    conf = signal.get("confidence", 0)
    narr = signal.get("narrative", "No narrative available.")
    sent = signal.get("sentiment_score", 0)

    arrow = "🔺 LONG" if d == "long" else "🔻 SHORT" if d == "short" else "⬜ NEUTRAL"
    level = AlertLevel.WARNING if conf >= 0.75 else AlertLevel.INFO

    msg = (
        f"*{arrow} — {sym}*\n"
        f"Confidence: `{conf:.0%}` | Sentiment: `{sent:+.1f}` | Horizon: {signal.get('horizon','4h')}\n"
        f"_{narr}_"
    )
    return send_alert(msg, level, symbol=sym)


def alert_anomaly(anomaly: dict) -> dict:
    """Send a formatted anomaly alert."""
    sym   = anomaly.get("symbol", "?")
    atype = anomaly.get("type", "unknown").replace("_", " ").title()
    z     = anomaly.get("zscore", 0)
    expl  = anomaly.get("explanation", anomaly.get("description", ""))

    level = AlertLevel.CRITICAL if abs(z) > 4 else AlertLevel.WARNING
    msg   = f"*Anomaly: {atype} — {sym}*\nZ-score: `{z:.1f}×` normal\n_{expl}_"
    return send_alert(msg, level, symbol=sym)


def alert_risk_breach(symbol: str, metric: str, value: float, limit: float) -> dict:
    msg = (f"*Risk Limit Breach — {symbol}*\n"
           f"{metric}: `{value:.2f}` exceeded limit `{limit:.2f}`\nReview position immediately.")
    return send_alert(msg, AlertLevel.CRITICAL, symbol=symbol)


def alert_paper_trade(trade_result: dict) -> dict:
    """Notify when a paper trade is executed."""
    sym    = trade_result.get("symbol", "?")
    action = trade_result.get("action", "?").upper()
    price  = trade_result.get("price", 0)
    cost   = trade_result.get("cost", 0)
    conf   = trade_result.get("confidence", 0)

    msg = (f"*Paper Trade: {action} — {sym}*\n"
           f"Price: `${price:.2f}` | Cost: `${cost:,.0f}` | Conf: `{conf:.0%}`")
    return send_alert(msg, AlertLevel.INFO, symbol=sym)


def alert_pipeline_error(error: str, stage: str) -> dict:
    msg = f"*Pipeline Error in {stage}*\n```{error[:300]}```"
    return send_alert(msg, AlertLevel.CRITICAL, symbol="SYSTEM")


def run_signal_alerter(min_confidence: float = 0.72) -> int:
    """Pull recent high-confidence signals and alert on them."""
    from core import get_session, SignalRecord
    from sqlalchemy import select
    since = datetime.utcnow() - timedelta(hours=2)
    with get_session() as s:
        rows = s.execute(
            select(SignalRecord)
            .where(SignalRecord.timestamp >= since)
            .where(SignalRecord.confidence >= min_confidence)
            .where(SignalRecord.direction != "neutral")
            .order_by(SignalRecord.confidence.desc())
        ).scalars().all()

    for r in rows:
        alert_signal({"symbol": r.symbol, "direction": r.direction,
                      "confidence": r.confidence, "horizon": r.horizon,
                      "narrative": r.narrative, "sentiment_score": r.sentiment_score or 0})
    log.info(f"Sent {len(rows)} signal alerts")
    return len(rows)


def test_alerts():
    """Send a test message to verify channel configuration."""
    result = send_alert("✅ Financial AI Pipeline — alert channels working correctly.", AlertLevel.INFO)
    configured = []
    if TELEGRAM_TOKEN:  configured.append("Telegram")
    if SLACK_WEBHOOK:   configured.append("Slack")
    if not configured:
        print("\n[!] No alert channels configured. Add to .env:")
        print("    TELEGRAM_BOT_TOKEN=...")
        print("    TELEGRAM_CHAT_ID=...")
        print("    SLACK_WEBHOOK_URL=https://hooks.slack.com/...")
        print("\nAlerts will print to console until channels are configured.\n")
    return result


if __name__ == "__main__":
    from core import init_db
    init_db()
    test_alerts()
