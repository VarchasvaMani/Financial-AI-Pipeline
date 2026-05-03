"""
Portfolio Optimisation — Markowitz mean-variance optimisation.
Computes optimal weights using expected returns from signals + historical covariance.
Includes: Max Sharpe, Min Volatility, Risk Parity, and Equal Weight portfolios.
Uses scipy for optimisation (no cvxpy dependency needed).
"""
from __future__ import annotations
import logging, warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

RISK_FREE_RATE = 0.05
TRADING_PERIODS = 252   # annualisation factor for hourly → annual


# ─────────────────────── Return & covariance ────────────────────────

def build_return_matrix(symbols: list[str], days: int = 90) -> pd.DataFrame:
    """Load hourly close prices and compute log returns per symbol."""
    from ingestion.market_feed import load_ohlcv
    frames = {}
    for sym in symbols:
        df = load_ohlcv(sym, days=days)
        if not df.empty:
            frames[sym] = df["close"]

    if not frames:
        return pd.DataFrame()

    prices = pd.DataFrame(frames).dropna(how="all").ffill()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def compute_covariance(returns: pd.DataFrame, annualise: bool = True) -> pd.DataFrame:
    """Compute shrinkage-adjusted covariance matrix (Ledoit-Wolf style)."""
    cov = returns.cov()
    if annualise:
        cov = cov * TRADING_PERIODS
    # Simple shrinkage toward diagonal (improves conditioning)
    n = len(cov)
    target = np.diag(np.diag(cov.values))
    shrink = 0.1
    cov_shrunk = (1 - shrink) * cov.values + shrink * target
    return pd.DataFrame(cov_shrunk, index=cov.index, columns=cov.columns)


def get_expected_returns(symbols: list[str], returns: pd.DataFrame) -> pd.Series:
    """
    Blend signal-based expected returns with historical mean returns.
    Signal confidence boosts/dampens the historical mean.
    """
    from core import get_session, SignalRecord
    from sqlalchemy import select
    since = datetime.utcnow() - timedelta(hours=6)

    # Historical annualised mean
    hist_returns = returns.mean() * TRADING_PERIODS

    # Signal adjustment
    adjustments = {}
    with get_session() as s:
        for sym in symbols:
            row = s.execute(
                select(SignalRecord)
                .where(SignalRecord.symbol == sym)
                .where(SignalRecord.timestamp >= since)
                .order_by(SignalRecord.timestamp.desc())
            ).scalar()
            if row:
                conf = row.confidence or 0.5
                boost = (conf - 0.5) * 0.2   # ±10% adjustment at max confidence
                if row.direction == "short":
                    boost = -boost
                adjustments[sym] = boost
            else:
                adjustments[sym] = 0.0

    expected = hist_returns.copy()
    for sym, adj in adjustments.items():
        if sym in expected.index:
            expected[sym] = expected[sym] + adj

    return expected


# ─────────────────────── Portfolio optimisation ──────────────────────

def _portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float, float]:
    """Return (return, volatility, sharpe) for a weight vector."""
    port_return = float(weights @ mu)
    port_vol    = float(np.sqrt(weights @ cov @ weights))
    sharpe      = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe


def max_sharpe_portfolio(mu: pd.Series, cov: pd.DataFrame, allow_short: bool = False) -> dict:
    """Maximise Sharpe ratio via scipy minimisation."""
    n = len(mu)
    bounds = [(-0.30, 1.0) if allow_short else (0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def neg_sharpe(w):
        r, v, _ = _portfolio_stats(w, mu.values, cov.values)
        return -(r - RISK_FREE_RATE) / v if v > 1e-8 else 999

    best = None
    for _ in range(10):  # multi-start for robustness
        w0 = np.random.dirichlet(np.ones(n))
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res

    weights = best.x / best.x.sum()
    r, v, s = _portfolio_stats(weights, mu.values, cov.values)
    return {"weights": dict(zip(mu.index, weights.round(4))),
            "expected_return": round(r * 100, 2),
            "volatility": round(v * 100, 2),
            "sharpe": round(s, 3),
            "method": "max_sharpe"}


def min_volatility_portfolio(mu: pd.Series, cov: pd.DataFrame) -> dict:
    """Minimise portfolio volatility (Global Minimum Variance)."""
    n = len(mu)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def port_vol(w):
        return float(np.sqrt(w @ cov.values @ w))

    w0  = np.ones(n) / n
    res = minimize(port_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = res.x / res.x.sum()
    r, v, s = _portfolio_stats(weights, mu.values, cov.values)
    return {"weights": dict(zip(mu.index, weights.round(4))),
            "expected_return": round(r * 100, 2),
            "volatility": round(v * 100, 2),
            "sharpe": round(s, 3),
            "method": "min_volatility"}


def risk_parity_portfolio(mu: pd.Series, cov: pd.DataFrame) -> dict:
    """Risk parity — equalise each asset's contribution to total portfolio risk."""
    n = len(mu)

    def risk_contributions_sq_diff(w):
        var  = w @ cov.values @ w
        mrc  = cov.values @ w         # marginal risk contribution
        rc   = w * mrc / var          # % risk contribution
        target = 1.0 / n
        return sum((rc[i] - target)**2 for i in range(n))

    bounds      = [(0.001, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0  = np.ones(n) / n
    res = minimize(risk_contributions_sq_diff, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints, options={"maxiter": 5000})
    weights = res.x / res.x.sum()
    r, v, s = _portfolio_stats(weights, mu.values, cov.values)
    return {"weights": dict(zip(mu.index, weights.round(4))),
            "expected_return": round(r * 100, 2),
            "volatility": round(v * 100, 2),
            "sharpe": round(s, 3),
            "method": "risk_parity"}


def equal_weight_portfolio(mu: pd.Series, cov: pd.DataFrame) -> dict:
    """Naive 1/N equal weight benchmark."""
    n = len(mu)
    w = np.ones(n) / n
    r, v, s = _portfolio_stats(w, mu.values, cov.values)
    return {"weights": dict(zip(mu.index, w.round(4))),
            "expected_return": round(r * 100, 2),
            "volatility": round(v * 100, 2),
            "sharpe": round(s, 3),
            "method": "equal_weight"}


# ─────────────────────── Main optimiser ─────────────────────────────

def run_optimisation(symbols: list[str], days: int = 90) -> dict:
    """
    Run all four optimisation methods and return comparison.
    Returns the recommended allocation (max Sharpe) plus all alternatives.
    """
    log.info(f"Running portfolio optimisation for {len(symbols)} assets")

    returns = build_return_matrix(symbols, days=days)
    if returns.empty or len(returns) < 30:
        log.error("Insufficient return data for optimisation")
        return {}

    # Filter to symbols with enough data
    valid = [s for s in symbols if s in returns.columns and returns[s].notna().sum() > 30]
    if len(valid) < 2:
        log.error("Need at least 2 symbols with data")
        return {}

    returns = returns[valid].dropna()
    mu  = get_expected_returns(valid, returns)
    cov = compute_covariance(returns)

    # Align mu and cov
    mu  = mu[valid]
    cov = cov.loc[valid, valid]

    portfolios = {
        "max_sharpe":    max_sharpe_portfolio(mu, cov),
        "min_volatility": min_volatility_portfolio(mu, cov),
        "risk_parity":   risk_parity_portfolio(mu, cov),
        "equal_weight":  equal_weight_portfolio(mu, cov),
    }

    # Efficient frontier (20 points)
    frontier = _efficient_frontier(mu, cov, n_points=20)

    recommended = portfolios["max_sharpe"]
    log.info(f"Max Sharpe portfolio: return={recommended['expected_return']:.1f}%, "
             f"vol={recommended['volatility']:.1f}%, sharpe={recommended['sharpe']:.2f}")

    return {
        "symbols": valid,
        "portfolios": portfolios,
        "recommended": recommended,
        "frontier": frontier,
        "as_of": datetime.utcnow().isoformat(),
        "days_history": days,
    }


def _efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 20) -> list[dict]:
    """Compute efficient frontier by sweeping target returns."""
    n = len(mu)
    min_ret = float(mu.min()) * 0.9
    max_ret = float(mu.max()) * 1.1
    target_returns = np.linspace(min_ret, max_ret, n_points)
    frontier = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu.values) - t},
        ]
        bounds = [(0.0, 1.0)] * n
        res = minimize(lambda w: float(np.sqrt(w @ cov.values @ w)),
                       np.ones(n)/n, method="SLSQP",
                       bounds=bounds, constraints=constraints)
        if res.success:
            r, v, s = _portfolio_stats(res.x, mu.values, cov.values)
            frontier.append({"return_pct": round(r*100,2), "vol_pct": round(v*100,2), "sharpe": round(s,3)})

    return frontier


def format_allocation_report(result: dict) -> str:
    """Pretty-print optimisation results."""
    if not result:
        return "No optimisation results available."

    lines = ["\n" + "="*55, "  PORTFOLIO OPTIMISATION REPORT", "="*55]
    rec = result["recommended"]
    lines.append(f"\n  RECOMMENDED: {rec['method'].replace('_',' ').title()}")
    lines.append(f"  Expected return:  {rec['expected_return']:+.1f}% ann.")
    lines.append(f"  Volatility:       {rec['volatility']:.1f}% ann.")
    lines.append(f"  Sharpe ratio:     {rec['sharpe']:.2f}")
    lines.append(f"\n  Weights:")
    for sym, w in sorted(rec["weights"].items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 30)
        lines.append(f"    {sym:8} {w:6.1%}  {bar}")
    lines.append(f"\n  Comparison:")
    lines.append(f"  {'Method':<20} {'Return':>8} {'Vol':>8} {'Sharpe':>8}")
    lines.append("  " + "-"*44)
    for name, p in result["portfolios"].items():
        lines.append(f"  {name:<20} {p['expected_return']:>7.1f}% {p['volatility']:>7.1f}% {p['sharpe']:>8.2f}")
    lines.append("="*55 + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    from core import init_db
    from ingestion.market_feed import DEFAULT_WATCHLIST
    init_db()
    symbols = DEFAULT_WATCHLIST[:8]
    print(f"Optimising portfolio: {symbols}")
    result = run_optimisation(symbols)
    print(format_allocation_report(result))
