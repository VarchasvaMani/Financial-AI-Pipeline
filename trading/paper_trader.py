"""
Paper Trading Simulator — virtual portfolio with real prices.
Tracks positions, cash, PnL, and trade history without risking real money.
Persists state to SQLite so it survives restarts.
"""
from __future__ import annotations
import json, logging, os
from datetime import datetime
from typing import Optional
import pandas as pd
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, select, delete
from core import Base, engine, get_session, init_db, AuditRecord

log = logging.getLogger(__name__)
INITIAL_CAPITAL = float(os.getenv("PAPER_CAPITAL", "100000"))


class PaperPosition(Base):
    __tablename__ = "paper_positions"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol      = Column(String(10), unique=True, nullable=False)
    direction   = Column(String(10))
    shares      = Column(Float, default=0)
    avg_cost    = Column(Float, default=0)
    entry_date  = Column(DateTime)
    signal_conf = Column(Float, default=0)


class PaperTrade(Base):
    __tablename__ = "paper_trades"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime, default=datetime.utcnow)
    symbol      = Column(String(10))
    action      = Column(String(10))   # buy / sell / short / cover
    shares      = Column(Float)
    price       = Column(Float)
    pnl         = Column(Float, default=0)
    reason      = Column(Text)


class PaperAccount(Base):
    __tablename__ = "paper_account"
    id      = Column(Integer, primary_key=True, default=1)
    cash    = Column(Float, default=INITIAL_CAPITAL)
    created = Column(DateTime, default=datetime.utcnow)


def _init_tables():
    Base.metadata.create_all(engine)
    with get_session() as s:
        acct = s.execute(select(PaperAccount)).scalar_one_or_none()
        if not acct:
            s.add(PaperAccount(cash=INITIAL_CAPITAL))
            s.commit()


def get_cash() -> float:
    _init_tables()
    with get_session() as s:
        acct = s.execute(select(PaperAccount)).scalar_one_or_none()
        return float(acct.cash) if acct else INITIAL_CAPITAL


def _set_cash(cash: float):
    with get_session() as s:
        acct = s.execute(select(PaperAccount)).scalar_one_or_none()
        if acct:
            acct.cash = cash
            s.commit()


def get_positions() -> list[dict]:
    _init_tables()
    with get_session() as s:
        rows = s.execute(select(PaperPosition)).scalars().all()
        return [{"symbol": r.symbol, "direction": r.direction, "shares": r.shares,
                 "avg_cost": r.avg_cost, "entry_date": r.entry_date,
                 "signal_conf": r.signal_conf} for r in rows]


def get_portfolio_value(prices: Optional[dict] = None) -> dict:
    """Compute total portfolio value including open positions."""
    if prices is None:
        from ingestion.market_feed import get_latest_prices
        syms = [p["symbol"] for p in get_positions()]
        prices = get_latest_prices(syms) if syms else {}

    cash = get_cash()
    positions = get_positions()
    position_value = 0.0
    pos_detail = []

    for p in positions:
        price = prices.get(p["symbol"], p["avg_cost"])
        market_val = p["shares"] * price
        unrealised = (price - p["avg_cost"]) * p["shares"]
        if p["direction"] == "short":
            unrealised = -unrealised
        position_value += market_val
        pos_detail.append({
            **p,
            "current_price": price,
            "market_value": round(market_val, 2),
            "unrealised_pnl": round(unrealised, 2),
            "unrealised_pct": round(unrealised / (p["avg_cost"] * p["shares"]) * 100, 2) if p["avg_cost"] else 0,
        })

    total = cash + position_value
    return {
        "cash": round(cash, 2),
        "position_value": round(position_value, 2),
        "total_value": round(total, 2),
        "total_pnl": round(total - INITIAL_CAPITAL, 2),
        "total_pnl_pct": round((total / INITIAL_CAPITAL - 1) * 100, 2),
        "positions": pos_detail,
        "n_positions": len(pos_detail),
        "as_of": datetime.utcnow().isoformat(),
    }


def execute_signal(signal: dict, prices: Optional[dict] = None) -> dict:
    """
    Execute a trading signal in the paper account.
    signal = {symbol, direction, confidence, horizon, narrative}
    Returns trade result dict.
    """
    _init_tables()
    symbol    = signal["symbol"]
    direction = signal["direction"]
    confidence = signal.get("confidence", 0.0)

    if direction == "neutral":
        return {"status": "skipped", "reason": "neutral signal"}

    if prices is None:
        from ingestion.market_feed import get_latest_prices
        prices = get_latest_prices([symbol])

    price = prices.get(symbol)
    if not price:
        return {"status": "error", "reason": f"No price for {symbol}"}

    cash = get_cash()

    with get_session() as s:
        existing = s.execute(
            select(PaperPosition).where(PaperPosition.symbol == symbol)
        ).scalar_one_or_none()

        # ── Close opposing position ──────────────────────────────
        if existing and existing.direction != direction:
            exit_price = price
            pnl = (exit_price - existing.avg_cost) * existing.shares
            if existing.direction == "short":
                pnl = -pnl
            action = "cover" if existing.direction == "short" else "sell"
            new_cash = cash + existing.shares * exit_price
            _set_cash(new_cash)
            cash = new_cash

            s.add(PaperTrade(
                symbol=symbol, action=action, shares=existing.shares,
                price=exit_price, pnl=round(pnl, 2),
                reason=f"Reversed to {direction} on new signal (conf={confidence:.0%})",
            ))
            s.delete(existing)
            s.commit()
            existing = None
            log.info(f"[{symbol}] Closed {action} @ ${exit_price:.2f}  PnL=${pnl:+.2f}")

        # ── Open new position ────────────────────────────────────
        if existing is None:
            alloc  = cash * 0.10  # 10% position size
            shares = alloc / price
            cost   = shares * price

            if cost > cash:
                return {"status": "skipped", "reason": "insufficient cash"}

            _set_cash(cash - cost)
            action = "buy" if direction == "long" else "short"

            s.add(PaperPosition(
                symbol=symbol, direction=direction, shares=shares,
                avg_cost=price, entry_date=datetime.utcnow(),
                signal_conf=confidence,
            ))
            s.add(PaperTrade(
                symbol=symbol, action=action, shares=shares,
                price=price, pnl=0,
                reason=signal.get("narrative", "Signal-driven entry"),
            ))
            s.commit()
            log.info(f"[{symbol}] {action.upper()} {shares:.2f} shares @ ${price:.2f}")

            return {
                "status": "executed", "action": action, "symbol": symbol,
                "shares": round(shares, 4), "price": price,
                "cost": round(cost, 2), "confidence": confidence,
            }

        return {"status": "skipped", "reason": "position already open in same direction"}


def close_position(symbol: str, reason: str = "manual") -> dict:
    """Manually close a position."""
    _init_tables()
    from ingestion.market_feed import get_latest_prices
    prices = get_latest_prices([symbol])
    price = prices.get(symbol)
    if not price:
        return {"status": "error", "reason": "no price"}

    with get_session() as s:
        pos = s.execute(select(PaperPosition).where(PaperPosition.symbol==symbol)).scalar_one_or_none()
        if not pos:
            return {"status": "error", "reason": "no open position"}

        pnl = (price - pos.avg_cost) * pos.shares * (1 if pos.direction=="long" else -1)
        cash = get_cash() + pos.shares * price
        _set_cash(cash)

        s.add(PaperTrade(symbol=symbol, action="sell", shares=pos.shares,
                         price=price, pnl=round(pnl,2), reason=reason))
        s.delete(pos)
        s.commit()
        log.info(f"[{symbol}] Closed @ ${price:.2f}  PnL=${pnl:+.2f}  ({reason})")
        return {"status": "closed", "symbol": symbol, "price": price,
                "pnl": round(pnl, 2), "reason": reason}


def get_trade_history(limit: int = 100) -> pd.DataFrame:
    _init_tables()
    with get_session() as s:
        rows = s.execute(
            select(PaperTrade).order_by(PaperTrade.timestamp.desc()).limit(limit)
        ).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp": r.timestamp, "symbol": r.symbol, "action": r.action,
            "shares": r.shares, "price": r.price, "pnl": r.pnl, "reason": r.reason,
        } for r in rows])


def reset_account():
    """Reset paper account to starting capital."""
    with get_session() as s:
        s.execute(delete(PaperPosition))
        s.execute(delete(PaperTrade))
        acct = s.execute(select(PaperAccount)).scalar_one_or_none()
        if acct:
            acct.cash = INITIAL_CAPITAL
        s.commit()
    log.info(f"Paper account reset to ${INITIAL_CAPITAL:,.0f}")


if __name__ == "__main__":
    init_db()
    _init_tables()
    print(f"Paper account value: ${get_portfolio_value()['total_value']:,.2f}")
    print(f"Cash: ${get_cash():,.2f}")
    positions = get_positions()
    print(f"Open positions: {len(positions)}")
