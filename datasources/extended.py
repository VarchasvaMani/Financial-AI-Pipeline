"""
Extended Data Sources
- SEC EDGAR filings (8-K, 10-K, 10-Q) — fully free
- Earnings calendar via yfinance — free
- Insider trades via SEC Form 4 — free
- Options flow (open interest, put/call ratio) via yfinance — free
- Economic calendar (FRED) — free with API key
"""
from __future__ import annotations
import os, logging, requests, time
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
FRED_KEY = os.getenv("FRED_API_KEY", "")
SEC_UA   = os.getenv("SEC_USER_AGENT", "financial-ai-pipeline research@example.com")


# ─────────────────────── SEC EDGAR ──────────────────────────────────

def fetch_sec_filings(ticker: str, form_types: list[str] = ["8-K", "10-K", "10-Q"],
                      limit: int = 10) -> list[dict]:
    """
    Fetch recent SEC filings for a ticker via the free EDGAR full-text search API.
    No API key required — just a valid User-Agent header.
    """
    headers = {"User-Agent": SEC_UA, "Accept": "application/json"}

    # Step 1: Resolve ticker to CIK
    try:
        cik_resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index?q=%22{}%22&dateRange=custom&startdt=2020-01-01&forms=10-K".format(ticker),
            headers=headers, timeout=10
        )
        # Use company tickers JSON endpoint instead (more reliable)
        tickers_resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers, timeout=15
        )
        tickers_resp.raise_for_status()
        tickers_data = tickers_resp.json()

        cik = None
        for _, company in tickers_data.items():
            if company.get("ticker", "").upper() == ticker.upper():
                cik = str(company["cik_str"]).zfill(10)
                break

        if not cik:
            log.warning(f"[{ticker}] CIK not found in EDGAR")
            return []

        # Step 2: Fetch submissions (filings list)
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        sub_resp = requests.get(sub_url, headers=headers, timeout=15)
        sub_resp.raise_for_status()
        sub_data = sub_resp.json()

        filings = sub_data.get("filings", {}).get("recent", {})
        forms   = filings.get("form", [])
        dates   = filings.get("filingDate", [])
        accs    = filings.get("accessionNumber", [])
        descs   = filings.get("primaryDocument", [])

        results = []
        for form, date, acc, doc in zip(forms, dates, accs, descs):
            if form in form_types:
                acc_clean = acc.replace("-", "")
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
                results.append({
                    "ticker": ticker,
                    "form_type": form,
                    "filing_date": date,
                    "accession": acc,
                    "url": url,
                    "source": "SEC EDGAR",
                })
                if len(results) >= limit:
                    break

        log.info(f"[{ticker}] Found {len(results)} SEC filings")
        return results

    except Exception as e:
        log.error(f"[{ticker}] SEC EDGAR error: {e}")
        return []


def fetch_filing_text(url: str, max_chars: int = 5000) -> str:
    """Fetch and truncate SEC filing text for AI analysis."""
    headers = {"User-Agent": SEC_UA}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        text = resp.text
        # Strip HTML tags roughly
        import re
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        log.error(f"Filing fetch failed: {e}")
        return ""


def analyse_filing_with_claude(ticker: str, filing: dict) -> dict:
    """Use Claude to extract key facts from an SEC filing."""
    import anthropic, os
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"summary": "Claude API key not set", "sentiment": "neutral", "key_facts": []}

    text = fetch_filing_text(filing["url"])
    if not text:
        return {"summary": "Could not fetch filing", "sentiment": "neutral", "key_facts": []}

    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content":
                f"""Analyse this {filing['form_type']} SEC filing for {ticker} (filed {filing['filing_date']}).
Extract: 1) 2-sentence summary 2) sentiment (bullish/bearish/neutral) 3) top 3 key facts as bullet points.
Return JSON only: {{"summary": "...", "sentiment": "bullish|bearish|neutral", "key_facts": ["...", "...", "..."]}}

Filing text:
{text[:3000]}"""}]
        )
        import json
        raw = resp.content[0].text.strip().replace("```json","").replace("```","")
        return json.loads(raw)
    except Exception as e:
        log.error(f"Claude filing analysis failed: {e}")
        return {"summary": "Analysis failed", "sentiment": "neutral", "key_facts": []}


# ─────────────────────── Earnings calendar ──────────────────────────

def get_earnings_calendar(symbols: list[str]) -> pd.DataFrame:
    """Fetch upcoming earnings dates via yfinance."""
    import yfinance as yf
    rows = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                # calendar is a DataFrame with dates as columns
                for col in cal.columns:
                    rows.append({
                        "symbol": sym,
                        "event": "Earnings",
                        "date": pd.to_datetime(col).date(),
                        "eps_estimate": cal.get("Earnings Average", {}).get(col),
                        "revenue_estimate": cal.get("Revenue Average", {}).get(col),
                    })
        except Exception as e:
            log.debug(f"[{sym}] Earnings calendar error: {e}")
        time.sleep(0.3)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date")
    return df


def get_upcoming_earnings(symbols: list[str], days_ahead: int = 14) -> list[dict]:
    """Return earnings events in the next N days."""
    df = get_earnings_calendar(symbols)
    if df.empty:
        return []
    cutoff = (datetime.utcnow() + timedelta(days=days_ahead)).date()
    today  = datetime.utcnow().date()
    upcoming = df[(df["date"] >= today) & (df["date"] <= cutoff)]
    return upcoming.to_dict("records")


# ─────────────────────── Insider trades ─────────────────────────────

def fetch_insider_trades(ticker: str, limit: int = 20) -> list[dict]:
    """
    Fetch insider trades (Form 4) from SEC EDGAR.
    Free, no API key required.
    """
    headers = {"User-Agent": SEC_UA, "Accept": "application/json"}
    try:
        url = (f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
               f"&dateRange=custom&startdt={(datetime.utcnow()-timedelta(days=90)).strftime('%Y-%m-%d')}"
               f"&forms=4&hits.hits._source=period_of_report,entity_name,file_date,form_type")
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for hit in data.get("hits", {}).get("hits", [])[:limit]:
            src = hit.get("_source", {})
            results.append({
                "ticker": ticker,
                "insider": src.get("entity_name", "Unknown"),
                "date": src.get("period_of_report", ""),
                "filed": src.get("file_date", ""),
                "form": "Form 4",
                "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=4&dateb=&owner=include&count=40",
            })
        log.info(f"[{ticker}] Found {len(results)} insider trades")
        return results
    except Exception as e:
        log.error(f"[{ticker}] Insider trades error: {e}")
        return []


def get_insider_sentiment(ticker: str) -> dict:
    """Summarise insider activity as a sentiment signal."""
    trades = fetch_insider_trades(ticker, limit=10)
    if not trades:
        return {"signal": "neutral", "count": 0, "details": "No recent insider activity"}
    return {
        "signal": "bullish" if len(trades) > 3 else "neutral",
        "count": len(trades),
        "details": f"{len(trades)} insider transactions in last 90 days",
        "trades": trades[:5],
    }


# ─────────────────────── Options flow ───────────────────────────────

def get_options_flow(ticker: str) -> dict:
    """
    Fetch options data via yfinance — put/call ratio, open interest, IV.
    Free, no API key required.
    """
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return {"error": "No options data available"}

        # Use nearest expiry
        exp = exps[0]
        chain = t.option_chain(exp)
        calls = chain.calls
        puts  = chain.puts

        total_call_oi = float(calls["openInterest"].sum()) if not calls.empty else 0
        total_put_oi  = float(puts["openInterest"].sum())  if not puts.empty else 0
        pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

        atm_iv_call = None
        atm_iv_put  = None
        price_resp = t.history(period="1d")
        if not price_resp.empty:
            spot = float(price_resp["Close"].iloc[-1])
            # Find ATM strike
            if not calls.empty:
                idx = (calls["strike"] - spot).abs().idxmin()
                atm_iv_call = float(calls.loc[idx, "impliedVolatility"]) * 100
            if not puts.empty:
                idx = (puts["strike"] - spot).abs().idxmin()
                atm_iv_put = float(puts.loc[idx, "impliedVolatility"]) * 100

        # Interpretation
        if pc_ratio > 1.2:
            sentiment = "bearish"
        elif pc_ratio < 0.8:
            sentiment = "bullish"
        else:
            sentiment = "neutral"

        result = {
            "ticker": ticker,
            "expiry": exp,
            "put_call_ratio": round(pc_ratio, 3),
            "total_call_oi": int(total_call_oi),
            "total_put_oi": int(total_put_oi),
            "atm_iv_call_pct": round(atm_iv_call, 1) if atm_iv_call else None,
            "atm_iv_put_pct": round(atm_iv_put, 1) if atm_iv_put else None,
            "options_sentiment": sentiment,
            "interpretation": (
                f"P/C ratio {pc_ratio:.2f} suggests {sentiment} options positioning. "
                f"ATM IV: {atm_iv_call:.0f}%" if atm_iv_call else ""
            ),
        }
        log.info(f"[{ticker}] Options: P/C={pc_ratio:.2f}, sentiment={sentiment}")
        return result

    except Exception as e:
        log.error(f"[{ticker}] Options flow error: {e}")
        return {"ticker": ticker, "error": str(e)}


# ─────────────────────── FRED macro data ────────────────────────────

FRED_SERIES = {
    "fed_funds_rate":  "FEDFUNDS",
    "cpi_yoy":         "CPIAUCSL",
    "unemployment":    "UNRATE",
    "gdp_growth":      "A191RL1Q225SBEA",
    "10y_yield":       "DGS10",
    "2y_yield":        "DGS2",
    "yield_curve":     "T10Y2Y",
    "vix":             "VIXCLS",
    "credit_spread":   "BAMLH0A0HYM2",
}


def fetch_fred_series(series_id: str, limit: int = 10) -> pd.Series:
    """Fetch a FRED time series. Free with API key (get at fred.stlouisfed.org)."""
    if not FRED_KEY:
        log.debug("No FRED API key — skipping macro data")
        return pd.Series(dtype=float)
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": series_id, "api_key": FRED_KEY,
                  "file_type": "json", "limit": limit, "sort_order": "desc"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("observations", [])
        vals = {d["date"]: float(d["value"]) for d in data if d["value"] != "."}
        return pd.Series(vals).sort_index()
    except Exception as e:
        log.error(f"FRED {series_id}: {e}")
        return pd.Series(dtype=float)


def get_macro_snapshot() -> dict:
    """Fetch all key macro indicators as a single snapshot dict."""
    snapshot = {}
    for name, series_id in FRED_SERIES.items():
        s = fetch_fred_series(series_id, limit=1)
        if not s.empty:
            snapshot[name] = {"value": round(s.iloc[-1], 3), "date": s.index[-1]}
        time.sleep(0.2)

    # Yield curve inversion check
    if "10y_yield" in snapshot and "2y_yield" in snapshot:
        spread = snapshot["10y_yield"]["value"] - snapshot["2y_yield"]["value"]
        snapshot["yield_curve_inverted"] = spread < 0
        snapshot["yield_curve_spread"] = round(spread, 3)

    return snapshot


# ─────────────────────── Unified data enrichment ────────────────────

def enrich_symbol(ticker: str) -> dict:
    """
    Run all data sources for a single ticker and return a unified enrichment dict.
    Designed to be called before signal generation for maximum context.
    """
    log.info(f"[{ticker}] Running full data enrichment...")
    enrichment = {"ticker": ticker, "as_of": datetime.utcnow().isoformat()}

    # Options flow (fast)
    enrichment["options"] = get_options_flow(ticker)

    # Insider sentiment
    enrichment["insider"] = get_insider_sentiment(ticker)

    # Recent SEC filings (just metadata, not full text)
    filings = fetch_sec_filings(ticker, form_types=["8-K"], limit=3)
    enrichment["recent_filings"] = filings

    return enrichment


if __name__ == "__main__":
    from core import init_db
    init_db()
    print("\n=== SEC Filings ===")
    filings = fetch_sec_filings("AAPL", limit=3)
    for f in filings:
        print(f"  {f['form_type']} — {f['filing_date']}  {f['url'][:60]}...")

    print("\n=== Options Flow ===")
    opts = get_options_flow("AAPL")
    print(f"  P/C ratio: {opts.get('put_call_ratio')}  Sentiment: {opts.get('options_sentiment')}")

    print("\n=== Macro Snapshot ===")
    if FRED_KEY:
        macro = get_macro_snapshot()
        for k, v in macro.items():
            if isinstance(v, dict):
                print(f"  {k}: {v['value']} ({v['date']})")
    else:
        print("  Add FRED_API_KEY to .env for macro data (free at fred.stlouisfed.org)")
