"""
Financial AI Pipeline — Extended Streamlit Dashboard v2
6 pages: Live Signals | Asset Deep Dive | Backtesting | Paper Trading | Portfolio Optimiser | Risk & Anomalies
"""
from __future__ import annotations
import os, sys, json, warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Financial AI Pipeline", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.main .block-container{padding-top:1.2rem}
.signal-long{background:#d4edda;border-left:4px solid #28a745;padding:.7rem 1rem;border-radius:4px;margin:4px 0}
.signal-short{background:#f8d7da;border-left:4px solid #dc3545;padding:.7rem 1rem;border-radius:4px;margin:4px 0}
.signal-neutral{background:#e2e3e5;border-left:4px solid #6c757d;padding:.7rem 1rem;border-radius:4px;margin:4px 0}
.narrative-box{background:#fff3cd;border-left:4px solid #ffc107;padding:.75rem;border-radius:4px;font-size:.9em;margin:4px 0}
.anomaly-box{background:#f8d7da;border-left:4px solid #dc3545;padding:.75rem;border-radius:4px;font-size:.9em;margin:4px 0}
</style>""", unsafe_allow_html=True)

WATCHLIST = os.getenv("DEFAULT_WATCHLIST","AAPL,MSFT,GOOGL,NVDA,TSLA,AMZN,META,JPM,GS,SPY").split(",")

@st.cache_data(ttl=300)
def get_signals(limit=100):
    from core import get_session, SignalRecord, init_db
    from sqlalchemy import select
    init_db()
    with get_session() as s:
        rows = s.execute(select(SignalRecord).order_by(SignalRecord.timestamp.desc()).limit(limit)).scalars().all()
        if not rows: return pd.DataFrame()
        return pd.DataFrame([{"symbol":r.symbol,"timestamp":r.timestamp,"direction":r.direction,
            "confidence":r.confidence,"horizon":r.horizon,"narrative":r.narrative,
            "sentiment_score":r.sentiment_score,"top_features":r.top_features} for r in rows])

@st.cache_data(ttl=60)
def cached_ohlcv(sym):
    from ingestion.market_feed import load_ohlcv
    return load_ohlcv(sym, days=60)

@st.cache_data(ttl=120)
def cached_feat(sym):
    from features.technical import load_features
    return load_features(sym, limit=300)

@st.cache_data(ttl=600)
def cached_prices():
    from ingestion.market_feed import get_latest_prices
    return get_latest_prices(WATCHLIST)

def dir_emoji(d): return {"long":"🔺","short":"🔻","neutral":"⬜"}.get(d,"⬜")
def conf_color(c): return "🟢" if c>=0.75 else "🟡" if c>=0.62 else "🔴"

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Financial AI Pipeline")
    st.markdown("---")
    page = st.radio("", ["🏠 Live Signals","🔍 Asset Deep Dive","📊 Backtesting",
                          "📋 Paper Trading","⚖️ Portfolio Optimiser","⚠️ Risk & Anomalies"],
                    label_visibility="collapsed")
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("▶ Run Pipeline", use_container_width=True):
            with st.spinner("Running..."):
                try:
                    from pipeline import run_full_pipeline
                    r = run_full_pipeline(WATCHLIST[:5], skip_train=True)
                    st.success(f"{r.get('signals_generated',0)} signals"); st.cache_data.clear()
                except Exception as e: st.error(str(e)[:120])
    with c2:
        if st.button("🎯 Quick Demo", use_container_width=True):
            with st.spinner("Loading..."):
                try:
                    from core import init_db
                    from ingestion.market_feed import ingest_watchlist
                    from features.technical import run_feature_pipeline
                    from models.signal_model import generate_all_signals
                    syms = ["AAPL","MSFT","NVDA","TSLA","SPY"]
                    init_db(); ingest_watchlist(syms,"60d"); run_feature_pipeline(syms); generate_all_signals(syms)
                    st.cache_data.clear(); st.rerun()
                except Exception as e: st.error(str(e)[:120])
    st.markdown("---")
    api_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    st.markdown(f"**Claude:** {'✅' if api_ok else '⚠️ Not set'}")
    st.markdown(f"**Polygon:** {'✅' if os.getenv('POLYGON_API_KEY') else '⚠️ yfinance fallback'}")
    tg = bool(os.getenv("TELEGRAM_BOT_TOKEN")); sl = bool(os.getenv("SLACK_WEBHOOK_URL"))
    st.markdown(f"**Alerts:** {'Telegram ✅ ' if tg else ''}{'Slack ✅' if sl else ''}{'Console only ⚠️' if not tg and not sl else ''}")

# ── PAGE 1: LIVE SIGNALS ─────────────────────────────────────────────
if page == "🏠 Live Signals":
    st.title("🏠 Live Signal Feed")
    df = get_signals(100)
    if df.empty: st.info("No signals — click Quick Demo in sidebar.", icon="💡"); st.stop()
    latest = df.drop_duplicates("symbol", keep="first")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔺 Long",   (latest["direction"]=="long").sum())
    c2.metric("🔻 Short",  (latest["direction"]=="short").sum())
    c3.metric("⬜ Neutral",(latest["direction"]=="neutral").sum())
    c4.metric("Avg confidence", f"{latest['confidence'].mean():.0%}")
    st.markdown("---")
    cf1,cf2,cf3 = st.columns(3)
    dir_f = cf1.multiselect("Direction",["long","short","neutral"],default=["long","short","neutral"])
    min_c = cf2.slider("Min confidence",0.5,1.0,0.55,0.05)
    sym_f = cf3.multiselect("Symbols",sorted(latest["symbol"].unique()),default=list(latest["symbol"].unique()))
    filt  = latest[latest["direction"].isin(dir_f)&(latest["confidence"]>=min_c)&latest["symbol"].isin(sym_f)].sort_values("confidence",ascending=False)
    prices = cached_prices()
    for _, row in filt.iterrows():
        price_str = f"${prices.get(row['symbol'],0):.2f}" if prices.get(row['symbol']) else "N/A"
        ts = row['timestamp'].strftime("%b %d %H:%M") if pd.notna(row['timestamp']) else "—"
        st.markdown(f'<div class="signal-{row["direction"]}"><strong>{dir_emoji(row["direction"])} {row["symbol"]}</strong> '
                    f'{conf_color(row["confidence"])} {row["confidence"]:.0%} | {price_str} | Sentiment {row["sentiment_score"]:+.1f} | {ts}</div>',unsafe_allow_html=True)
        if pd.notna(row.get("narrative")) and row["narrative"]:
            st.markdown(f'<div class="narrative-box">💬 {row["narrative"]}</div>',unsafe_allow_html=True)
        if pd.notna(row.get("top_features")) and row["top_features"]:
            try:
                feats = json.loads(row["top_features"])
                if feats:
                    fig = go.Figure(go.Bar(x=[abs(f[1]) for f in feats],y=[f[0].replace("_"," ") for f in feats],
                        orientation="h",marker_color="#28a745" if row["direction"]=="long" else "#dc3545"))
                    fig.update_layout(height=110,margin=dict(l=0,r=0,t=0,b=0),font=dict(size=10),
                                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig,use_container_width=True)
            except: pass
        st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("---")
    q = st.text_input("💬 NL query (e.g. show bullish signals above 75% confidence)",label_visibility="collapsed")
    if q and st.button("Ask →"):
        from agents.narrative_agent import answer_nl_query
        st.info(answer_nl_query(q, filt[["symbol","direction","confidence","sentiment_score"]].to_string()))

# ── PAGE 2: ASSET DEEP DIVE ──────────────────────────────────────────
elif page == "🔍 Asset Deep Dive":
    st.title("🔍 Asset Deep Dive")
    symbol = st.selectbox("Asset", sorted(WATCHLIST))
    ohlcv = cached_ohlcv(symbol); feat = cached_feat(symbol)
    if ohlcv.empty: st.warning("No data. Run pipeline first."); st.stop()

    with st.expander("🔌 Extended data (SEC, Options, Insider trades)"):
        if st.button("Fetch extended data"):
            with st.spinner("Fetching..."):
                from datasources.extended import enrich_symbol
                st.session_state[f"enr_{symbol}"] = enrich_symbol(symbol)
                st.success("Done!")
        if f"enr_{symbol}" in st.session_state:
            e = st.session_state[f"enr_{symbol}"]
            ec1,ec2 = st.columns(2)
            with ec1:
                opts = e.get("options",{})
                if "put_call_ratio" in opts:
                    st.metric("P/C Ratio", opts["put_call_ratio"])
                    st.metric("Options Sentiment", opts.get("options_sentiment","—"))
            with ec2:
                ins = e.get("insider",{})
                st.metric("Insider Signal", ins.get("signal","—").title())
                st.write(ins.get("details",""))
            for f2 in e.get("recent_filings",[]):
                st.markdown(f"• [{f2['form_type']} — {f2['filing_date']}]({f2['url']})")

    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.55,0.25,0.20],
        subplot_titles=["Price + Bollinger","Volume","RSI"],vertical_spacing=0.06)
    fig.add_trace(go.Candlestick(x=ohlcv.index,open=ohlcv["open"],high=ohlcv["high"],
        low=ohlcv["low"],close=ohlcv["close"],increasing_line_color="#26a69a",decreasing_line_color="#ef5350"),row=1,col=1)
    if not feat.empty:
        al=feat.reindex(ohlcv.index,method="nearest")
        if "bb_upper" in al.columns:
            fig.add_trace(go.Scatter(x=al.index,y=al["bb_upper"],line=dict(color="rgba(100,149,237,0.4)",dash="dot"),name="BB+"),row=1,col=1)
            fig.add_trace(go.Scatter(x=al.index,y=al["bb_lower"],line=dict(color="rgba(100,149,237,0.4)",dash="dot"),name="BB-",fill="tonexty",fillcolor="rgba(100,149,237,0.05)"),row=1,col=1)
        if "vwap" in al.columns:
            fig.add_trace(go.Scatter(x=al.index,y=al["vwap"],line=dict(color="#ff9800",width=1.5),name="VWAP"),row=1,col=1)
    colors=["#26a69a" if c>=o else "#ef5350" for c,o in zip(ohlcv["close"],ohlcv["open"])]
    fig.add_trace(go.Bar(x=ohlcv.index,y=ohlcv["volume"],marker_color=colors,showlegend=False),row=2,col=1)
    if not feat.empty and "rsi" in feat.columns:
        ra=feat["rsi"].reindex(ohlcv.index,method="nearest")
        fig.add_trace(go.Scatter(x=ra.index,y=ra,line=dict(color="#9c27b0",width=1.5),name="RSI"),row=3,col=1)
        fig.add_hline(y=70,line_dash="dot",line_color="red",row=3,col=1)
        fig.add_hline(y=30,line_dash="dot",line_color="green",row=3,col=1)
    fig.update_layout(height=560,xaxis_rangeslider_visible=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    if not feat.empty:
        lf=feat.iloc[-1]; c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("RSI",f"{lf.get('rsi',0):.1f}"); c2.metric("MACD",f"{lf.get('macd',0):.4f}")
        c3.metric("Vol Z",f"{lf.get('volume_zscore',0):.2f}×"); c4.metric("RealVol",f"{lf.get('realized_vol',0)*100:.1f}%")
        c5.metric("Mom 5d",f"{lf.get('momentum_5d',0)*100:+.2f}%")

# ── PAGE 3: BACKTESTING ───────────────────────────────────────────────
elif page == "📊 Backtesting":
    st.title("📊 Backtesting Engine")
    c1,c2,c3=st.columns(3)
    bt_syms  = c1.multiselect("Symbols",WATCHLIST,default=WATCHLIST[:5])
    bt_start = c2.date_input("Start",value=datetime(2024,1,1))
    bt_end   = c3.date_input("End",  value=datetime(2024,12,31))
    c4,c5,c6=st.columns(3)
    ps=c4.slider("Position size %",5,25,10)/100
    sl=c5.slider("Stop loss %",1,10,3)/100
    tp=c6.slider("Take profit %",2,20,6)/100
    if st.button("▶ Run Backtest",type="primary",use_container_width=True):
        if not bt_syms: st.warning("Select symbols"); st.stop()
        with st.spinner("Backtesting..."):
            try:
                from backtest.engine import BacktestEngine
                bt=BacktestEngine(bt_syms,start=str(bt_start),end=str(bt_end),position_size=ps,stop_loss_pct=sl,take_profit_pct=tp)
                results=bt.run(); st.session_state["bt_r"]=results; st.session_state["bt_e"]=bt
            except Exception as e: st.error(f"Failed: {e}")
    if "bt_r" in st.session_state:
        r=st.session_state["bt_r"]; m=r.metrics
        if not m: st.warning("No trades generated. Adjust settings."); st.stop()
        mc1,mc2,mc3,mc4,mc5,mc6=st.columns(6)
        mc1.metric("Total Return",f"{m['total_return_pct']:+.1f}%"); mc2.metric("Ann Return",f"{m['ann_return_pct']:+.1f}%")
        mc3.metric("Sharpe",f"{m['sharpe']:.2f}"); mc4.metric("Max DD",f"{m['max_drawdown_pct']:.1f}%")
        mc5.metric("Win Rate",f"{m['win_rate_pct']:.0f}%"); mc6.metric("Trades",m['total_trades'])
        eq=r.equity_curve
        fig_eq=go.Figure(go.Scatter(x=eq.index,y=eq.values,fill="tozeroy",line=dict(color="#26a69a",width=2)))
        fig_eq.add_hline(y=m["initial_capital"],line_dash="dot",line_color="gray")
        fig_eq.update_layout(title="Equity Curve",height=300,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_eq,use_container_width=True)
        dd=r.drawdown_series
        fig_dd=go.Figure(go.Scatter(x=dd.index,y=dd.values,fill="tozeroy",line=dict(color="#ef5350",width=1.5)))
        fig_dd.update_layout(title="Drawdown (%)",height=180,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dd,use_container_width=True)
        if r.trades:
            sym_pnl={} 
            for t in r.trades: sym_pnl[t.symbol]=sym_pnl.get(t.symbol,0)+t.pnl
            fig_b=px.bar(x=list(sym_pnl.keys()),y=list(sym_pnl.values()),
                color=[v>=0 for v in sym_pnl.values()],color_discrete_map={True:"#26a69a",False:"#ef5350"},
                title="PnL by Symbol"); fig_b.update_layout(showlegend=False,height=260,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_b,use_container_width=True)
            td=pd.DataFrame([{"Symbol":t.symbol,"Dir":t.direction,"Entry $":f"${t.entry_price:.2f}",
                "Exit $":f"${t.exit_price:.2f}" if t.exit_price else "—","PnL":f"${t.pnl:+,.0f}",
                "Return":f"{t.return_pct:+.2f}%","Exit":t.exit_reason} for t in r.trades])
            st.dataframe(td,use_container_width=True,hide_index=True)

# ── PAGE 4: PAPER TRADING ─────────────────────────────────────────────
elif page == "📋 Paper Trading":
    st.title("📋 Paper Trading Simulator")
    from trading.paper_trader import get_portfolio_value,get_trade_history,execute_signal,close_position,reset_account,_init_tables,get_positions
    from core import init_db; init_db(); _init_tables()
    prices=cached_prices(); portfolio=get_portfolio_value(prices)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Value",f"${portfolio['total_value']:,.2f}"); c2.metric("Cash",f"${portfolio['cash']:,.2f}")
    c3.metric("Positions",  f"${portfolio['position_value']:,.2f}")
    c4.metric("Total PnL",  f"${portfolio['total_pnl']:+,.2f}",f"{portfolio['total_pnl_pct']:+.2f}%")
    st.markdown("---")
    col_a,col_b=st.columns([2,1])
    with col_a:
        st.subheader("Execute from signals")
        sigs_df=get_signals(20)
        if not sigs_df.empty:
            hc=sigs_df.drop_duplicates("symbol").query("confidence>=0.65").sort_values("confidence",ascending=False)
            for _,row in hc.head(5).iterrows():
                sc1,sc2=st.columns([3,1])
                sc1.markdown(f"{dir_emoji(row['direction'])} **{row['symbol']}** {row['direction'].upper()} @ {row['confidence']:.0%}")
                with sc2:
                    if st.button("Execute",key=f"ex_{row['symbol']}"):
                        res=execute_signal({"symbol":row["symbol"],"direction":row["direction"],"confidence":row["confidence"],"narrative":row.get("narrative","")},prices)
                        if res.get("status")=="executed":
                            st.success(f"✅ {res['action'].upper()} {row['symbol']} @ ${res['price']:.2f}")
                            from alerts.alert_bot import alert_paper_trade; alert_paper_trade(res); st.rerun()
                        else: st.warning(res.get("reason","Skipped"))
    with col_b:
        st.subheader("Manual close")
        pos=get_positions()
        if pos:
            csym=st.selectbox("Close",[ p["symbol"] for p in pos])
            if st.button("Close",type="secondary"):
                r=close_position(csym,"manual"); st.success(f"Closed {csym} PnL ${r.get('pnl',0):+.2f}"); st.rerun()
        if st.button("🔄 Reset account"):
            reset_account(); st.success("Reset to $100,000"); st.rerun()
    st.markdown("---")
    positions=portfolio.get("positions",[])
    if positions:
        pdf=pd.DataFrame(positions)
        pdf["unrealised_pnl"]=pdf["unrealised_pnl"].apply(lambda x:f"${x:+,.2f}")
        pdf["unrealised_pct"]=pdf["unrealised_pct"].apply(lambda x:f"{x:+.2f}%")
        st.dataframe(pdf[["symbol","direction","shares","avg_cost","unrealised_pnl","unrealised_pct"]],use_container_width=True,hide_index=True)
    else: st.caption("No open positions")
    hist=get_trade_history(50)
    if not hist.empty:
        hist["timestamp"]=pd.to_datetime(hist["timestamp"]).dt.strftime("%b %d %H:%M")
        hist["pnl"]=hist["pnl"].apply(lambda x:f"${x:+,.2f}")
        st.dataframe(hist[["timestamp","symbol","action","shares","price","pnl"]],use_container_width=True,hide_index=True)

# ── PAGE 5: PORTFOLIO OPTIMISER ───────────────────────────────────────
elif page == "⚖️ Portfolio Optimiser":
    st.title("⚖️ Portfolio Optimisation")
    opt_syms=st.multiselect("Symbols",WATCHLIST,default=WATCHLIST[:6]); days_h=st.slider("History (days)",30,180,90)
    if st.button("▶ Run Optimisation",type="primary",use_container_width=True):
        if len(opt_syms)<2: st.warning("Need at least 2 symbols"); st.stop()
        with st.spinner("Optimising..."):
            try:
                from portfolio.optimiser import run_optimisation
                result=run_optimisation(opt_syms,days=days_h); st.session_state["opt_r"]=result
            except Exception as e: st.error(f"Failed: {e}")
    if "opt_r" in st.session_state:
        result=st.session_state["opt_r"]
        if not result: st.warning("Insufficient data."); st.stop()
        rec=result["recommended"]; ports=result["portfolios"]
        st.subheader(f"✅ Recommended: {rec['method'].replace('_',' ').title()}")
        r1,r2,r3=st.columns(3)
        r1.metric("Expected Return",f"{rec['expected_return']:+.1f}%"); r2.metric("Volatility",f"{rec['volatility']:.1f}%"); r3.metric("Sharpe",f"{rec['sharpe']:.2f}")
        fig_pie=go.Figure(go.Pie(labels=list(rec["weights"].keys()),values=[round(v*100,1) for v in rec["weights"].values()],hole=0.45))
        fig_pie.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",title="Recommended Weights")
        st.plotly_chart(fig_pie,use_container_width=True)
        comp=pd.DataFrame([{"Method":p["method"].replace("_"," ").title(),"Return":f"{p['expected_return']:+.1f}%",
            "Vol":f"{p['volatility']:.1f}%","Sharpe":f"{p['sharpe']:.2f}"} for p in ports.values()])
        st.dataframe(comp,use_container_width=True,hide_index=True)
        frontier=result.get("frontier",[])
        if frontier:
            ff=pd.DataFrame(frontier)
            fig_ef=go.Figure(go.Scatter(x=ff["vol_pct"],y=ff["return_pct"],mode="lines+markers",line=dict(color="#185FA5",width=2)))
            for nm,p in ports.items():
                fig_ef.add_trace(go.Scatter(x=[p["volatility"]],y=[p["expected_return"]],mode="markers+text",
                    text=[nm.replace("_"," ").title()],textposition="top center",marker=dict(size=12)))
            fig_ef.update_layout(title="Efficient Frontier",height=360,xaxis_title="Volatility (%)",yaxis_title="Return (%)",
                                  paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_ef,use_container_width=True)

# ── PAGE 6: RISK & ANOMALIES ─────────────────────────────────────────
elif page == "⚠️ Risk & Anomalies":
    st.title("⚠️ Risk Monitor")
    df=get_signals(200)
    if df.empty: st.info("Run the pipeline first.",icon="💡"); st.stop()
    latest=df.drop_duplicates("symbol",keep="first")
    fig_heat=go.Figure(go.Heatmap(x=latest["symbol"],y=["Signal"],z=[latest["confidence"].tolist()],
        colorscale="RdYlGn",zmin=0.5,zmax=1.0,
        text=[[f"{d} {c:.0%}" for d,c in zip(latest["direction"],latest["confidence"])]],texttemplate="%{text}"))
    fig_heat.update_layout(height=110,margin=dict(t=5,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_heat,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        dc=latest["direction"].value_counts()
        fig_pie=go.Figure(go.Pie(labels=dc.index,values=dc.values,marker_colors=["#28a745","#6c757d","#dc3545"],hole=0.5))
        fig_pie.update_layout(height=240,paper_bgcolor="rgba(0,0,0,0)",title="Directional Exposure"); st.plotly_chart(fig_pie,use_container_width=True)
    with c2:
        fig_hist=px.histogram(latest,x="confidence",nbins=10,color="direction",
            color_discrete_map={"long":"#28a745","short":"#dc3545","neutral":"#6c757d"},title="Confidence Distribution")
        fig_hist.update_layout(height=240,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(fig_hist,use_container_width=True)
    st.subheader("🔔 Alert Channels")
    al1,al2=st.columns(2)
    with al1:
        if st.button("Send test alert"):
            from alerts.alert_bot import test_alerts
            r=test_alerts(); st.success(f"Telegram:{r['telegram']} Slack:{r['slack']} Logged:{r['logged']}")
    with al2:
        mn=st.slider("Alert threshold",0.60,0.95,0.72,0.01)
        if st.button("Alert high-conf signals"):
            from alerts.alert_bot import run_signal_alerter
            n=run_signal_alerter(min_confidence=mn); st.success(f"Sent {n} alerts")
    st.subheader("🚨 Anomaly Scanner")
    scan_s=st.multiselect("Scan symbols",sorted(latest["symbol"].unique()),default=list(latest["symbol"].unique())[:5])
    if st.button("Scan",use_container_width=True):
        from models.anomaly_detector import run_anomaly_detection
        all_a=[]; prog=st.progress(0)
        for i,sym in enumerate(scan_s):
            all_a+=run_anomaly_detection(sym); prog.progress((i+1)/len(scan_s))
        if all_a:
            for a in all_a:
                st.markdown(f'<div class="anomaly-box">🚨 <strong>{a.get("symbol","?")} — {a["type"].replace("_"," ").title()}</strong> '
                            f'(z={a.get("zscore","?")}×)<br>{a["description"]}<br><em>{a.get("explanation","")}</em></div>',unsafe_allow_html=True)
        else: st.success("✅ No anomalies detected")
    hist2=df[["timestamp","symbol","direction","confidence","sentiment_score"]].copy()
    hist2["confidence"]=(hist2["confidence"]*100).round(1).astype(str)+"%"
    hist2["timestamp"]=pd.to_datetime(hist2["timestamp"]).dt.strftime("%b %d %H:%M")
    st.subheader("Signal History"); st.dataframe(hist2.head(50),use_container_width=True,hide_index=True)
