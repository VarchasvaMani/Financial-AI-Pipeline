"""
Backtesting Engine — vectorised, point-in-time correct.
Tests signal model on historical OHLCV data. Computes full suite of
performance metrics: Sharpe, Sortino, Calmar, max drawdown, win rate,
profit factor, per-trade PnL, long/short breakdown.
"""
from __future__ import annotations
import json, logging, warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

TRADING_DAYS      = 252
RISK_FREE_RATE    = 0.05
TRANSACTION_COST  = 0.001
INITIAL_CAPITAL   = 100_000.0


@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    direction: str
    entry_price: float
    exit_price: Optional[float]
    shares: float
    pnl: float = 0.0
    return_pct: float = 0.0
    confidence: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResults:
    equity_curve: pd.Series
    trades: list
    signals_df: pd.DataFrame
    metrics: dict
    drawdown_series: pd.Series


def _mock_signals_from_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Generate deterministic signals from feature data (no model needed)."""
    rows = []
    for ts, row in feat_df.iterrows():
        rsi  = float(row.get("rsi", 50) or 50)
        mom  = float(row.get("momentum_5d", 0) or 0)
        macd = float(row.get("macd", 0) or 0)
        score = (50 - rsi) / 100 + mom * 3 + (1 if macd > 0 else -1) * 0.05
        long_prob = max(0.3, min(0.85, 0.5 + score))
        if long_prob > 0.62:
            direction, confidence = "long", long_prob
        elif long_prob < 0.38:
            direction, confidence = "short", 1 - long_prob
        else:
            direction, confidence = "neutral", max(long_prob, 1-long_prob)
        rows.append({"timestamp": ts, "direction": direction, "confidence": confidence})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index("timestamp").sort_index()


class BacktestEngine:
    def __init__(self, symbols, start="2024-01-01", end="2024-12-31",
                 initial_capital=INITIAL_CAPITAL, position_size=0.10,
                 confidence_threshold=0.62, stop_loss_pct=0.03,
                 take_profit_pct=0.06, hold_bars=4,
                 transaction_cost=TRANSACTION_COST):
        self.symbols = symbols
        self.start = pd.Timestamp(start)
        self.end   = pd.Timestamp(end)
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.confidence_threshold = confidence_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.hold_bars = hold_bars
        self.transaction_cost = transaction_cost

    def _load_data(self, symbol):
        from ingestion.market_feed import load_ohlcv
        from features.technical import load_features, compute_features
        ohlcv = load_ohlcv(symbol, days=500)
        if ohlcv.empty:
            return pd.DataFrame(), pd.DataFrame()
        ohlcv.index = pd.to_datetime(ohlcv.index)
        ohlcv = ohlcv[(ohlcv.index >= self.start) & (ohlcv.index <= self.end)]
        feat  = load_features(symbol, limit=5000)
        if feat.empty:
            feat = compute_features(ohlcv)
        feat.index = pd.to_datetime(feat.index)
        feat = feat[(feat.index >= self.start) & (feat.index <= self.end)]
        return ohlcv, feat

    def _simulate(self, symbol, ohlcv, signals, capital):
        trades, cash = [], capital
        position, daily_pnl = None, pd.Series(0.0, index=ohlcv.index)
        for i, (ts, row) in enumerate(ohlcv.iterrows()):
            price = float(row["close"])
            sig   = signals.loc[ts].to_dict() if ts in signals.index else {}
            if position is not None:
                bars_held = i - position._bar
                pnl_pct   = (price - position.entry_price) / position.entry_price
                if position.direction == "short":
                    pnl_pct = -pnl_pct
                reason = None
                if pnl_pct <= -self.stop_loss_pct:   reason = "stop_loss"
                elif pnl_pct >= self.take_profit_pct: reason = "take_profit"
                elif bars_held >= self.hold_bars:      reason = "time_exit"
                if reason:
                    cost = position.shares * price * self.transaction_cost
                    gross = position.shares * (price - position.entry_price) * (1 if position.direction=="long" else -1)
                    cash += position.shares * price - cost
                    position.exit_date, position.exit_price = ts, price
                    position.pnl = round(gross - cost, 2)
                    position.return_pct = round(pnl_pct * 100, 3)
                    position.exit_reason = reason
                    trades.append(position)
                    daily_pnl[ts] = gross - cost
                    position = None
            if position is None and sig:
                d, c = sig.get("direction","neutral"), sig.get("confidence",0.0)
                if d in ("long","short") and c >= self.confidence_threshold:
                    alloc  = cash * self.position_size
                    shares = alloc / price
                    cost   = shares * price * self.transaction_cost
                    cash  -= shares * price + cost
                    position = Trade(symbol=symbol, entry_date=ts, exit_date=None,
                                     direction=d, entry_price=price, exit_price=None,
                                     shares=shares, confidence=c)
                    position._bar = i
        if position is not None and not ohlcv.empty:
            lp = float(ohlcv["close"].iloc[-1])
            lt = ohlcv.index[-1]
            pp = (lp - position.entry_price)/position.entry_price * (1 if position.direction=="long" else -1)
            position.exit_date, position.exit_price = lt, lp
            position.pnl = round(position.shares*(lp-position.entry_price)*(1 if position.direction=="long" else -1), 2)
            position.return_pct = round(pp*100, 3)
            position.exit_reason = "end_of_period"
            trades.append(position)
        return daily_pnl, trades

    def run(self) -> BacktestResults:
        log.info(f"Backtest {self.start.date()} → {self.end.date()}, {len(self.symbols)} symbols")
        all_pnl, all_trades, all_sigs = [], [], []
        for sym in self.symbols:
            ohlcv, feat = self._load_data(sym)
            if ohlcv.empty:
                continue
            try:
                from models.signal_model import load_model, FEATURE_COLS
                model = load_model()
                if model:
                    X_list = []
                    for ts, row in feat.iterrows():
                        X_list.append({c: float(row.get(c,0) or 0) for c in FEATURE_COLS})
                    X = pd.DataFrame(X_list, index=feat.index)
                    probs = model.predict_proba(X)[:,1]
                    sigs  = pd.DataFrame({
                        "direction":  ["long" if p>0.62 else "short" if p<0.38 else "neutral" for p in probs],
                        "confidence": [max(p,1-p) for p in probs],
                    }, index=feat.index)
                else:
                    sigs = _mock_signals_from_features(feat)
            except Exception:
                sigs = _mock_signals_from_features(feat)
            sigs_aligned = sigs.reindex(ohlcv.index, method="ffill")
            all_sigs.append(sigs_aligned.assign(symbol=sym))
            cap_alloc = self.initial_capital / len(self.symbols)
            pnl, trades = self._simulate(sym, ohlcv, sigs_aligned, cap_alloc)
            all_pnl.append(pnl)
            all_trades.extend(trades)
            sym_pnl = sum(t.pnl for t in trades)
            log.info(f"  [{sym}] {len(trades)} trades  PnL ${sym_pnl:+,.0f}")
        if not all_pnl:
            return BacktestResults(pd.Series(), [], pd.DataFrame(), {}, pd.Series())
        combined = pd.concat(all_pnl, axis=1).sum(axis=1).sort_index()
        equity   = self.initial_capital + combined.cumsum()
        dd       = (equity - equity.cummax()) / equity.cummax() * 100
        metrics  = self._metrics(equity, all_trades)
        log.info(f"Done — Sharpe {metrics['sharpe']:.2f}  MaxDD {metrics['max_drawdown_pct']:.1f}%  Return {metrics['total_return_pct']:.1f}%")
        return BacktestResults(equity, all_trades, pd.concat(all_sigs) if all_sigs else pd.DataFrame(), metrics, dd)

    def _metrics(self, equity, trades):
        if equity.empty or len(equity) < 2:
            return {}
        dr = equity.pct_change().dropna()
        rf = RISK_FREE_RATE / TRADING_DAYS
        ex = dr - rf
        sharpe  = ex.mean()/ex.std()*np.sqrt(TRADING_DAYS) if ex.std()>0 else 0
        ds      = dr[dr < rf]
        sortino = ex.mean()/ds.std()*np.sqrt(TRADING_DAYS) if len(ds)>0 and ds.std()>0 else 0
        ann_ret = ((1+dr.mean())**TRADING_DAYS - 1)*100
        ann_vol = dr.std()*np.sqrt(TRADING_DAYS)*100
        max_dd  = float(((equity - equity.cummax())/equity.cummax()*100).min())
        calmar  = ann_ret/abs(max_dd) if max_dd != 0 else 0
        done    = [t for t in trades if t.exit_price]
        wins    = [t for t in done if t.pnl > 0]
        losses  = [t for t in done if t.pnl <= 0]
        pf      = sum(t.pnl for t in wins)/abs(sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses)!=0 else float("inf")
        return {
            "total_return_pct":   round((equity.iloc[-1]/equity.iloc[0]-1)*100, 2),
            "ann_return_pct":     round(ann_ret, 2),
            "ann_volatility_pct": round(ann_vol, 2),
            "sharpe":             round(sharpe, 3),
            "sortino":            round(sortino, 3),
            "calmar":             round(calmar, 3),
            "max_drawdown_pct":   round(max_dd, 2),
            "win_rate_pct":       round(len(wins)/len(done)*100 if done else 0, 2),
            "profit_factor":      round(pf, 3),
            "total_trades":       len(done),
            "winning_trades":     len(wins),
            "losing_trades":      len(losses),
            "long_trades":        len([t for t in done if t.direction=="long"]),
            "short_trades":       len([t for t in done if t.direction=="short"]),
            "avg_win_usd":        round(np.mean([t.pnl for t in wins]) if wins else 0, 2),
            "avg_loss_usd":       round(np.mean([t.pnl for t in losses]) if losses else 0, 2),
            "final_equity":       round(float(equity.iloc[-1]), 2),
            "initial_capital":    self.initial_capital,
        }

    def report(self, r: BacktestResults):
        m = r.metrics
        if not m:
            print("No results."); return
        print(f"\n{'='*50}\n  BACKTEST REPORT  {self.start.date()} → {self.end.date()}\n{'='*50}")
        print(f"  Total return:    {m['total_return_pct']:+.2f}%  (ann: {m['ann_return_pct']:+.2f}%)")
        print(f"  Sharpe / Sortino / Calmar: {m['sharpe']:.2f} / {m['sortino']:.2f} / {m['calmar']:.2f}")
        print(f"  Max drawdown:    {m['max_drawdown_pct']:.2f}%   Vol: {m['ann_volatility_pct']:.2f}%")
        print(f"  Trades:          {m['total_trades']}  Win rate: {m['win_rate_pct']:.1f}%  PF: {m['profit_factor']:.2f}")
        print(f"  Avg win/loss:    ${m['avg_win_usd']:+,.0f} / ${m['avg_loss_usd']:+,.0f}")
        print(f"  Final equity:    ${m['final_equity']:,.2f}\n{'='*50}\n")


if __name__ == "__main__":
    from core import init_db
    from ingestion.market_feed import DEFAULT_WATCHLIST
    init_db()
    bt = BacktestEngine(DEFAULT_WATCHLIST[:5], start="2024-01-01", end="2024-12-31")
    results = bt.run()
    bt.report(results)
