"""
Live Price WebSocket Stream
Connects to Polygon.io WebSocket for real-time tick data.
Falls back to yfinance polling when no API key is set.
Publishes price updates to an in-memory queue for downstream consumers.
"""
from __future__ import annotations
import os, json, logging, asyncio, time, threading
from datetime import datetime
from typing import Callable, Optional
from collections import defaultdict, deque
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_WS  = "wss://socket.polygon.io/stocks"

# ─────────────── In-memory price cache ──────────────────────────────

_price_cache: dict[str, dict] = {}          # latest tick per symbol
_price_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
_subscribers: list[Callable] = []           # callback functions


def get_cached_price(symbol: str) -> Optional[dict]:
    return _price_cache.get(symbol.upper())


def get_price_history(symbol: str, n: int = 50) -> list[dict]:
    return list(_price_history[symbol.upper()])[-n:]


def subscribe(callback: Callable[[dict], None]):
    """Register a callback to receive price updates."""
    _subscribers.append(callback)


def _publish(tick: dict):
    """Push a tick to cache, history, and all subscribers."""
    sym = tick["symbol"]
    _price_cache[sym] = tick
    _price_history[sym].append(tick)
    for cb in _subscribers:
        try:
            cb(tick)
        except Exception as e:
            log.debug(f"Subscriber error: {e}")


# ─────────────── Polygon WebSocket (real-time) ──────────────────────

class PolygonStream:
    """
    Real-time WebSocket stream from Polygon.io.
    Requires POLYGON_API_KEY (free tier: delayed data, paid: real-time).
    """
    def __init__(self, symbols: list[str]):
        self.symbols = [s.upper() for s in symbols]
        self._running = False

    def start(self):
        """Run in a background thread."""
        self._running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        log.info(f"Polygon stream started for {self.symbols}")
        return t

    def stop(self):
        self._running = False

    def _run(self):
        try:
            import websocket
        except ImportError:
            log.error("websocket-client not installed: pip install websocket-client")
            return

        def on_open(ws):
            ws.send(json.dumps({"action": "auth", "params": POLYGON_KEY}))

        def on_message(ws, message):
            events = json.loads(message)
            for ev in events:
                if ev.get("ev") == "A":   # aggregate per second
                    tick = {
                        "symbol":    ev.get("sym", ""),
                        "price":     ev.get("c", ev.get("a", 0)),   # close or ask
                        "volume":    ev.get("av", 0),
                        "timestamp": datetime.utcnow().isoformat(),
                        "source":    "polygon_ws",
                    }
                    _publish(tick)
                elif ev.get("ev") == "status" and ev.get("status") == "auth_success":
                    # Subscribe to aggregate feed
                    subs = [f"A.{s}" for s in self.symbols]
                    ws.send(json.dumps({"action": "subscribe", "params": ",".join(subs)}))
                    log.info(f"Polygon subscribed to {subs}")

        def on_error(ws, error):
            log.error(f"Polygon WS error: {error}")

        def on_close(ws, *args):
            log.info("Polygon WS closed")

        ws = websocket.WebSocketApp(
            POLYGON_WS,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        while self._running:
            try:
                ws.run_forever(ping_interval=30)
            except Exception as e:
                log.error(f"WS reconnect after error: {e}")
                time.sleep(5)


# ─────────────── yfinance polling fallback ──────────────────────────

class YFinancePoller:
    """
    Polls yfinance for latest prices every N seconds.
    Free, no API key. Use when Polygon key isn't set.
    """
    def __init__(self, symbols: list[str], interval_sec: int = 60):
        self.symbols = [s.upper() for s in symbols]
        self.interval = interval_sec
        self._running = False

    def start(self) -> threading.Thread:
        self._running = True
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()
        log.info(f"yfinance poller started ({self.interval}s interval)")
        return t

    def stop(self):
        self._running = False

    def _poll_loop(self):
        import yfinance as yf
        while self._running:
            try:
                for sym in self.symbols:
                    ticker = yf.Ticker(sym)
                    hist   = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        tick = {
                            "symbol":    sym,
                            "price":     round(float(latest["Close"]), 4),
                            "volume":    int(latest["Volume"]),
                            "timestamp": datetime.utcnow().isoformat(),
                            "source":    "yfinance_poll",
                        }
                        _publish(tick)
                time.sleep(self.interval)
            except Exception as e:
                log.error(f"Poller error: {e}")
                time.sleep(10)


# ─────────────── Auto-select stream ─────────────────────────────────

def start_stream(symbols: list[str], interval_sec: int = 60) -> object:
    """
    Start the best available price stream.
    Uses Polygon WebSocket if API key is set, else yfinance polling.
    """
    if POLYGON_KEY and POLYGON_KEY != "your_polygon_key_here":
        stream = PolygonStream(symbols)
        stream.start()
        log.info("Using Polygon.io WebSocket (real-time)")
    else:
        stream = YFinancePoller(symbols, interval_sec=interval_sec)
        stream.start()
        log.info(f"Using yfinance poller ({interval_sec}s interval) — add POLYGON_API_KEY for real-time")
    return stream


def get_live_prices(symbols: list[str]) -> dict[str, float]:
    """Get latest cached prices for all symbols."""
    prices = {}
    for sym in symbols:
        tick = get_cached_price(sym)
        if tick:
            prices[sym] = tick["price"]
    # Fill missing from yfinance as fallback
    missing = [s for s in symbols if s not in prices]
    if missing:
        from ingestion.market_feed import get_latest_prices
        prices.update(get_latest_prices(missing))
    return prices


# ─────────────── Price alert triggers ────────────────────────────────

class PriceAlertMonitor:
    """
    Monitors live prices and fires alerts when thresholds are crossed.
    Register alerts with add_alert(), then call start_monitoring().
    """
    def __init__(self):
        self._alerts: list[dict] = []

    def add_alert(self, symbol: str, condition: str, threshold: float, message: str = ""):
        """
        condition: 'above' | 'below' | 'change_pct'
        Example: add_alert('AAPL', 'above', 200.0, 'AAPL crossed $200!')
        """
        self._alerts.append({
            "symbol": symbol.upper(),
            "condition": condition,
            "threshold": threshold,
            "message": message,
            "triggered": False,
        })

    def _check_tick(self, tick: dict):
        sym   = tick["symbol"]
        price = tick["price"]
        for alert in self._alerts:
            if alert["symbol"] != sym or alert["triggered"]:
                continue
            fired = False
            if alert["condition"] == "above" and price > alert["threshold"]:
                fired = True
            elif alert["condition"] == "below" and price < alert["threshold"]:
                fired = True
            if fired:
                from alerts.alert_bot import send_alert, AlertLevel
                msg = alert["message"] or f"{sym} @ ${price:.2f} ({alert['condition']} {alert['threshold']})"
                send_alert(msg, AlertLevel.WARNING, symbol=sym)
                alert["triggered"] = True
                log.info(f"Price alert triggered: {msg}")

    def start_monitoring(self):
        subscribe(self._check_tick)
        log.info(f"Price alert monitor active — {len(self._alerts)} alerts registered")


if __name__ == "__main__":
    import time as _time

    print("Starting live price stream (60s polling via yfinance)...")
    print("Press Ctrl+C to stop\n")

    symbols = ["AAPL", "MSFT", "NVDA", "SPY"]

    # Register a simple print subscriber
    def print_tick(tick):
        print(f"  [{tick['source']}] {tick['symbol']:6} ${tick['price']:.2f}  vol={tick['volume']:,}")

    subscribe(print_tick)

    stream = start_stream(symbols, interval_sec=60)

    # Add a price alert example
    monitor = PriceAlertMonitor()
    monitor.add_alert("AAPL", "above", 999999, "AAPL crossed $999,999!")  # won't fire
    monitor.start_monitoring()

    try:
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        stream.stop()
        print("\nStream stopped.")
