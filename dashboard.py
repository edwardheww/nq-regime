# dashboard.py

import time
import pytz
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from bias import compute_bias
from macro import compute_macro_indicators
from config import TICKERS

API_URL = "http://localhost:8000"

REGIME_META = {
    "trending_low_vol":  {"label": "TRENDING  \u00b7  LOW VOL",  "color": "#00D26A", "advice": "Smooth trend \u2014 trend-following conditions"},
    "trending_high_vol": {"label": "TRENDING  \u00b7  HIGH VOL", "color": "#FFB800", "advice": "Volatile trend \u2014 breakout / news-driven, widen stops"},
    "ranging_low_vol":   {"label": "RANGING   \u00b7  LOW VOL",  "color": "#3D9BE9", "advice": "Quiet range \u2014 mean-reversion conditions"},
    "ranging_high_vol":  {"label": "RANGING   \u00b7  HIGH VOL", "color": "#FF4455", "advice": "Choppy market \u2014 reduce size or sit out"},
    "extreme_vol":       {"label": "EXTREME VOL  \u00b7  EVENT", "color": "#FF00FF", "advice": "Event-driven spike \u2014 do not trade"},
    "extreme_vol_2": {"label": "EXTREME VOL 2  ·  EVENT", "color": "#CC00CC", "advice": "Secondary event spike — do not trade"},
    "initialising":      {"label": "INITIALISING...",              "color": "#AAAABB", "advice": "Model is warming up"},
}

DIRECTION_META = {
    "long":       {"label": "LONG  \u25b2",    "color": "#00D26A"},
    "short":      {"label": "SHORT  \u25bc",   "color": "#FF4455"},
    "neutral":    {"label": "NEUTRAL  \u25c6", "color": "#FF9500"},
    "suppressed": {"label": "NO TRADE  \u2715","color": "#5C5C5C"},
}

REGIME_COLORS = {
    "trending_low_vol":  "rgba(0, 210, 106, 0.12)",
    "trending_high_vol": "rgba(255, 184, 0, 0.12)",
    "ranging_low_vol":   "rgba(61, 155, 233, 0.10)",
    "ranging_high_vol":  "rgba(255, 68, 85, 0.12)",
    "extreme_vol":       "rgba(255, 0, 255, 0.15)",
    "extreme_vol_2":     "rgba(204, 0, 204, 0.15)",
    "unknown":           "rgba(100, 100, 100, 0.05)",
    "initialising":      "rgba(100, 100, 100, 0.05)",
}

st.set_page_config(page_title="Futures Regime", page_icon="\U0001f4ca", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace !important; background-color: #0a0a0f !important; color: #c8c8d0 !important; }
.stApp, .stApp > header, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color: #0a0a0f !important; }
.main .block-container { background-color: #0a0a0f; padding: 2rem 2.5rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }
h1 { font-family: 'IBM Plex Mono', monospace !important; font-weight: 600 !important; font-size: 1.1rem !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; color: #e8e8f0 !important; border-bottom: 1px solid #1e1e2e !important; padding-bottom: 1rem !important; margin-bottom: 1.5rem !important; }
h2, h3 { font-family: 'IBM Plex Mono', monospace !important; font-weight: 500 !important; font-size: 0.7rem !important; letter-spacing: 0.2em !important; text-transform: uppercase !important; color: #AAAABB !important; margin-bottom: 0.75rem !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stToggle"] * { color: #AAAABB !important; }
[data-testid="stPlotlyChart"] { background-color: #0f0f1a !important; border-radius: 4px; }
hr { border-color: #1e1e2e !important; margin: 1.5rem 0 !important; }
[data-testid="stAlert"] { border-radius: 4px !important; font-size: 0.72rem !important; border-left-width: 3px !important; }
::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #0a0a0f; } ::-webkit-scrollbar-thumb { background: #1e1e2e; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def fetch_regime(asset: str = "NQ"):
    try:
        r    = requests.get(f"{API_URL}/regime", params={"asset": asset}, timeout=5)
        data = r.json()
        for tf in ["15m", "1h"]:
            raw_post = data[tf].get("posteriors", {})
            data[tf]["posteriors"] = {int(k): float(v) for k, v in raw_post.items()}
        return data
    except Exception:
        empty = {"regime": "initialising", "confidence": 0.0, "posteriors": {}, "updated_at": None, "regime_sequence": []}
        return {"15m": empty, "1h": empty}


def fetch_price_data(ticker: str = "NQ=F"):
    import yfinance as yf
    for _ in range(3):
        try:
            df = yf.download(ticker, interval="15m", period="5d", progress=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.dropna(inplace=True)
            df.index = pd.to_datetime(df.index)
            df.index.name = "Datetime"
            if not df.empty:
                return df.tail(100)
        except Exception:
            continue
    return pd.DataFrame()


def divider():
    st.markdown("<hr style='border-color:#1e1e2e; margin:1.5rem 0;'>", unsafe_allow_html=True)


def regime_card(data: dict, label: str, tz: str = "Asia/Singapore"):
    regime       = data.get("regime", "initialising")
    conf         = data.get("confidence", 0.0)
    post         = data.get("posteriors", {})
    updated      = data.get("updated_at", "\u2014")
    regime_bars  = data.get("regime_bars", 0)
    regime_since = data.get("regime_since")
    meta         = REGIME_META.get(regime, REGIME_META["initialising"])
    adj_conf     = data.get("adjusted_confidence", conf)
    self_trans   = data.get("self_transition", 0.0)
    sticky_label = "STICKY" if self_trans >= 0.90 else "MODERATE" if self_trans >= 0.80 else "UNSTABLE"
    sticky_color = "#00D26A" if self_trans >= 0.90 else "#FFB800" if self_trans >= 0.80 else "#FF4455"

    if regime_since:
        since_dt = pd.Timestamp(regime_since)
        now_dt   = pd.Timestamp.now(tz=since_dt.tzinfo)
        mins     = int((now_dt - since_dt).seconds / 60)
        duration = f"{mins // 60}H {mins % 60}M" if mins >= 60 else f"{mins}M"
    else:
        duration = "\u2014"

    updated_str = pd.Timestamp(updated).tz_convert(pytz.timezone(tz)).strftime("%Y-%m-%d %H:%M") if updated and updated != "\u2014" else "\u2014"

    bars_html = ""
    if post:
        values  = [post[k] for k in sorted(post.keys())]
        max_val = max(values)
        for k in sorted(post.keys()):
            v     = post[k]
            width = int(v * 100)
            color = meta["color"] if v == max_val else "#2a2a3d"
            bars_html += (
                f'<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">'
                f'<div style="font-size:0.55rem; color:#AAAABB; width:1.5rem;">S{k}</div>'
                f'<div style="flex:1; background:#111120; border-radius:2px; height:6px;">'
                f'<div style="width:{width}%; background:{color}; height:100%; border-radius:2px;"></div>'
                f'</div>'
                f'<div style="font-size:0.55rem; color:#AAAABB; width:2.5rem; text-align:right;">{v:.2f}</div>'
                f'</div>'
            )

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#13131f 0%,#0f0f1a 100%);
        border:1px solid {meta['color']}33; border-left:3px solid {meta['color']};
        border-radius:4px; padding:1.25rem 1.5rem; margin-bottom:0.5rem; position:relative; overflow:hidden;">
        <div style="position:absolute; top:0; right:0; width:120px; height:120px;
            background:radial-gradient(circle,{meta['color']}08 0%,transparent 70%);"></div>
        <div style="font-size:0.55rem; letter-spacing:0.2em; color:#AAAABB; margin-bottom:0.5rem; text-transform:uppercase;">{label}</div>
        <div style="font-size:1.1rem; font-weight:600; color:{meta['color']}; letter-spacing:0.08em; margin-bottom:0.4rem;">{meta['label']}</div>
        <div style="font-size:0.65rem; color:#AAAABB; margin-bottom:1rem;">{meta['advice']}</div>
        <div style="display:flex; gap:2rem; align-items:center;">
            <div>
                <div style="font-size:0.5rem; letter-spacing:0.15em; color:#AAAABB; text-transform:uppercase;">Confidence</div>
                <div style="font-size:1.4rem; font-weight:600; color:#e8e8f0;">{conf*100:.1f}<span style="font-size:0.7rem; color:#AAAABB;">%</span></div>
                <div style="font-size:0.55rem; color:{sticky_color}; margin-top:0.2rem;">ADJ: {adj_conf*100:.1f}%  \u00b7  {sticky_label}</div>
            </div>
            <div style="flex:1; height:1px; background:linear-gradient(90deg,#1e1e2e,transparent);"></div>
            <div style="font-size:0.55rem; color:#AAAABB; text-align:right;">{updated_str}</div>
        </div>
        <div style="display:flex; gap:2rem; margin-top:0.75rem;">
            <div>
                <div style="font-size:0.5rem; letter-spacing:0.15em; color:#AAAABB; text-transform:uppercase;">Regime Age</div>
                <div style="font-size:0.9rem; font-weight:600; color:#e8e8f0;">{duration}</div>
            </div>
            <div>
                <div style="font-size:0.5rem; letter-spacing:0.15em; color:#AAAABB; text-transform:uppercase;">Bars Active</div>
                <div style="font-size:0.9rem; font-weight:600; color:#e8e8f0;">{regime_bars}</div>
            </div>
        </div>
        <div style="margin-top:0.75rem;">{bars_html}</div>
    </div>
    """, unsafe_allow_html=True)


def strategy_card(all_data, show_mtf, price_df, macro, asset="NQ", ticker="NQ=F"):
    regime_15m     = all_data["15m"].get("regime", "initialising")
    regime_1h      = all_data["1h"].get("regime", "initialising")
    conf_15m       = all_data["15m"].get("confidence", 0.0)
    conf_1h        = all_data["1h"].get("confidence", 0.0)
    adj_conf_15m   = all_data["15m"].get("adjusted_confidence", conf_15m)
    adj_conf_1h    = all_data["1h"].get("adjusted_confidence", conf_1h)
    self_trans_15m = all_data["15m"].get("self_transition", 0.0)
    macro_dir      = macro.get("direction", "neutral")
    events         = macro.get("events", [])
    imminent       = any(e["days_away"] <= 1 and e["impact"] == "HIGH" for e in events)
    mtf_conflict   = show_mtf and (regime_15m != regime_1h)

    raw_ok   = conf_15m >= 0.70
    adj_ok   = adj_conf_15m >= 0.65
    sticky   = self_trans_15m >= 0.90
    moderate = self_trans_15m >= 0.80
    mtf_raw_ok = conf_1h >= 0.70 if show_mtf else True
    mtf_adj_ok = adj_conf_1h >= 0.65 if show_mtf else True

    if raw_ok and adj_ok and sticky and mtf_raw_ok and mtf_adj_ok:
        conviction_tier, conviction_color, conviction_label = "FULL",     "#00D26A", "FULL CONVICTION"
    elif raw_ok and adj_ok and moderate and mtf_raw_ok and mtf_adj_ok:
        conviction_tier, conviction_color, conviction_label = "MODERATE", "#00D26A", "MODERATE CONVICTION"
    elif raw_ok and not adj_ok and mtf_raw_ok:
        conviction_tier, conviction_color, conviction_label = "FRAGILE",  "#FFB800", "REGIME FRAGILE \u2014 reduce size"
    else:
        conviction_tier, conviction_color, conviction_label = "WAIT",     "#FF4455", "INSUFFICIENT CONFIDENCE \u2014 no trade"

    action, color, approach, size = "WAIT", "#888899", "Unrecognised regime. No trades.", "0%"

    if regime_15m == "initialising":
        action, color, approach, size = "WAIT", "#888899", "Model is initialising. No trades.", "0%"
    elif conviction_tier == "WAIT":
        action, color = "WAIT", "#FF4455"
        approach = f"Raw: {conf_15m*100:.0f}%  \u00b7  Adj: {adj_conf_15m*100:.0f}% \u2014 below threshold. Wait for confidence."
        size = "0%"
    elif conviction_tier == "FRAGILE":
        action, color = "CAUTION", "#FFB800"
        approach = f"Regime identified but self-transition low ({self_trans_15m:.2f}) \u2014 may flip. No new entries."
        size = "25%"
    elif regime_15m == "ranging_high_vol":
        action, color, approach, size = "SIT OUT", "#FF4455", "Choppy high vol \u2014 both sides get stopped out. Stay flat.", "0%"
    elif regime_15m in ("extreme_vol", "extreme_vol_2"):
        action, color = "SIT OUT", "#FF00FF"
        approach = "Extreme volatility event — no trades until vol normalises."
        size = "0%"
    elif imminent:
        action, color = "REDUCE", "#FFB800"
        event_name = next(e["name"] for e in events if e["days_away"] <= 1 and e["impact"] == "HIGH")
        approach = f"High-impact event imminent ({event_name}). Hold existing only, no new entries."
        size = "25%"
    elif mtf_conflict:
        action, color = "CAUTION", "#FFB800"
        approach = f"Timeframe conflict \u2014 15m: {regime_15m.replace('_',' ')} vs 1H: {regime_1h.replace('_',' ')}. Wait for alignment."
        size = "0%"
    elif regime_15m == "trending_low_vol":
        if macro_dir == "bearish":
            action, color = "CAUTION", "#FFB800"
            approach = "Trend regime but macro bearish. Trade direction aligned with macro, tight risk."
            size = "50%" if conviction_tier == "MODERATE" else "75%"
        else:
            action, color = "TRADE", "#00D26A"
            approach = "Wait for 3-5 bar pullback to 9 EMA or VWAP. Enter on reclaim. Stop below pullback low. Target prior swing or 2:1 R:R."
            size = "75%" if conviction_tier == "MODERATE" else "100%"
    elif regime_15m == "trending_high_vol":
        action, color = "TRADE", "#FFB800"
        approach = "Breakouts only \u2014 no pullback entries. Stop 1.5-2x ATR. Take partials quickly."
        size = "35%" if conviction_tier == "MODERATE" else "50%"
    elif regime_15m == "ranging_low_vol":
        action, color = "TRADE", "#3D9BE9"
        approach = "Fade extremes at VWAP 2\u03c3. Stop at 3\u03c3. Target VWAP."
        size = "75%" if conviction_tier == "MODERATE" else "100%"

    bias       = compute_bias(price_df, regime=regime_15m, ticker=ticker)
    bias_dir   = bias.get("direction", "neutral")
    bias_conf  = bias.get("conviction", 0.0)
    bias_str   = bias.get("strength", "")
    bias_color = "#00D26A" if bias_dir == "long" else "#FF4455" if bias_dir == "short" else "#888899"
    bias_icon  = "\u25b2" if bias_dir == "long" else "\u25bc" if bias_dir == "short" else "\u25c6"
    conf_pct   = int(conf_15m * 100)
    conf_color = "#00D26A" if conf_15m >= 0.80 else "#FFB800" if conf_15m >= 0.60 else "#FF4455"
    mc         = "#00D26A" if macro_dir == "bullish" else "#FF4455" if macro_dir == "bearish" else "#888899"
    mi         = "\u25b2" if macro_dir == "bullish" else "\u25bc" if macro_dir == "bearish" else "\u25c6"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#13131f 0%,#0f0f1a 100%);
        border:1px solid {color}33; border-left:4px solid {color};
        border-radius:4px; padding:1.25rem 1.5rem; margin-bottom:0.5rem; position:relative; overflow:hidden;">
        <div style="position:absolute; top:0; right:0; width:180px; height:180px; background:radial-gradient(circle,{color}06 0%,transparent 70%);"></div>
        <div style="font-size:0.55rem; letter-spacing:0.2em; color:#444455; margin-bottom:0.5rem; text-transform:uppercase;">Strategy</div>
        <div style="display:flex; align-items:baseline; gap:1rem; margin-bottom:0.5rem;">
            <div style="font-size:1.4rem; font-weight:600; color:{color}; letter-spacing:0.1em;">{action}</div>
            <div style="font-size:0.65rem; color:#555566;">SIZE: <span style="color:#e8e8f0; font-weight:600;">{size}</span></div>
        </div>
        <div style="font-size:0.6rem; color:{conviction_color}; letter-spacing:0.08em; margin-bottom:0.75rem;">\u25cf {conviction_label}</div>
        <div style="font-size:0.68rem; color:#aaaabc; line-height:1.6; margin-bottom:1rem; border-left:2px solid {color}44; padding-left:0.75rem;">{approach}</div>
        <div style="display:flex; gap:2rem; margin-top:0.5rem; flex-wrap:wrap;">
            <div>
                <div style="font-size:0.5rem; letter-spacing:0.15em; color:#444455; text-transform:uppercase; margin-bottom:0.2rem;">Directional Bias</div>
                <div style="font-size:0.7rem; font-weight:600; color:{bias_color};">{bias_icon} {bias_str.upper()} {bias_dir.upper()} <span style="font-size:0.55rem; color:#555566; font-weight:400;">\u00b7 {bias_conf*100:.0f}% conviction</span></div>
            </div>
            <div>
                <div style="font-size:0.5rem; letter-spacing:0.15em; color:#444455; text-transform:uppercase; margin-bottom:0.2rem;">Model Confidence</div>
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <div style="width:80px; background:#111120; border-radius:2px; height:4px;"><div style="width:{conf_pct}%; background:{conf_color}; height:100%; border-radius:2px;"></div></div>
                    <div style="font-size:0.7rem; font-weight:600; color:{conf_color};">{conf_pct}%</div>
                </div>
            </div>
            <div>
                <div style="font-size:0.5rem; letter-spacing:0.15em; color:#444455; text-transform:uppercase; margin-bottom:0.2rem;">Macro</div>
                <div style="font-size:0.7rem; font-weight:600; color:{mc};">{mi} {macro_dir.upper()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def bias_card(bias: dict):
    direction = bias.get("direction", "neutral")
    strength  = bias.get("strength", "Neutral")
    bull      = bias.get("bull_votes", 0)
    bear      = bias.get("bear_votes", 0)
    neutral   = bias.get("neutral_votes", 0)
    total     = bias.get("total_votes", 1)
    reasons   = bias.get("reasons", {})
    meta      = DIRECTION_META.get(direction, DIRECTION_META["neutral"])
    bull_pct  = int((bull / total) * 100) if total else 0
    bear_pct  = int((bear / total) * 100) if total else 0

    bull_words = ['above', 'bull', 'hh', 'expanding on up', 'bullish', 'outperform', 'gap up']
    bear_words = ['below', 'bear', 'lh', 'expanding on down', 'bearish', 'underperform', 'gap down']

    reason_rows = ""
    for k, r in reasons.items():
        if "error" in k:
            continue
        r_lower = r.lower()
        if any(x in r_lower for x in bull_words):
            color, icon = "#00D26A", "\u25b2"
        elif any(x in r_lower for x in bear_words):
            color, icon = "#FF4455", "\u25bc"
        else:
            color, icon = "#888899", "\u25c6"
        reason_rows += f'<div style="font-size:0.62rem; color:{color}; padding:0.3rem 0; border-bottom:1px solid #111120;">{icon} {r}</div>'

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#13131f 0%,#0f0f1a 100%);
        border:1px solid {meta['color']}33; border-left:3px solid {meta['color']};
        border-radius:4px; padding:1.25rem 1.5rem; margin-bottom:1rem;">
        <div style="font-size:0.55rem; letter-spacing:0.2em; color:#888899; margin-bottom:0.5rem; text-transform:uppercase;">Structural Bias</div>
        <div style="font-size:1.1rem; font-weight:600; color:{meta['color']}; letter-spacing:0.08em; margin-bottom:0.3rem;">{strength.upper()}  {meta['label']}</div>
        <div style="font-size:0.65rem; color:#AAAABB; margin-bottom:1rem;">{bull} bull \u00b7 {bear} bear \u00b7 {neutral} neutral of {total} factors</div>
        <div style="display:flex; height:3px; border-radius:2px; overflow:hidden; gap:1px; margin-bottom:1rem;">
            <div style="width:{bull_pct}%; background:#00D26A;"></div>
            <div style="width:{bear_pct}%; background:#FF4455;"></div>
            <div style="flex:1; background:#1e1e2e;"></div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.4rem;">{reason_rows}</div>
    </div>
    """, unsafe_allow_html=True)


def macro_card(macro: dict):
    direction   = macro.get("direction", "neutral")
    strength    = macro.get("strength", "Neutral")
    bull        = macro.get("bull", 0)
    bear        = macro.get("bear", 0)
    neutral     = macro.get("neutral", 0)
    indicators  = macro.get("indicators", {})
    yield_curve = macro.get("yield_curve", {})

    color = "#00D26A" if direction == "bullish" else "#FF4455" if direction == "bearish" else "#888899"
    direction_label = "\u25b2 BULLISH" if direction == "bullish" else "\u25bc BEARISH" if direction == "bearish" else "\u25c6 NEUTRAL"

    all_rows = ""
    for ticker, ind in indicators.items():
        data         = ind.get("data") or {}
        chg_1d       = data.get("chg_1d", 0.0)
        chg_1h       = data.get("chg_1h", 0.0)
        icolor       = ind["color"]
        icon         = "\u25b2" if ind["vote"] == 1 else "\u25bc" if ind["vote"] == -1 else "\u25c6"
        chg_1d_color = "#00D26A" if chg_1d >= 0 else "#FF4455"
        all_rows += (
            f'<div style="display:flex; align-items:center; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid #111120;">'
            f'<div style="display:flex; align-items:center; gap:0.75rem;">'
            f'<div style="font-size:0.6rem; color:{icolor};">{icon}</div>'
            f'<div>'
            f'<div style="font-size:0.7rem; font-weight:600; color:#e8e8f0;">{ind["name"]}<span style="font-size:0.55rem; color:#555566; margin-left:0.3rem;">{ind["desc"]}</span></div>'
            f'<div style="font-size:0.6rem; color:{icolor}; margin-top:0.1rem;">{ind["label"]}</div>'
            f'</div></div>'
            f'<div style="text-align:right;">'
            f'<div style="font-size:0.6rem; color:{chg_1d_color};">1D: {chg_1d:+.2f}%</div>'
            f'<div style="font-size:0.55rem; color:#888899;">1H: {chg_1h:+.2f}%</div>'
            f'</div></div>'
        )

    yc_label    = yield_curve.get("label", "")
    yc_inverted = yield_curve.get("inverted", False)
    yc_color    = "#FF4455" if yc_inverted else "#888899"
    yc_html     = (
        f'<div style="margin-top:0.75rem; padding:0.5rem 0; border-top:1px solid #111120;">'
        f'<div style="font-size:0.5rem; letter-spacing:0.15em; color:#444455; text-transform:uppercase; margin-bottom:0.2rem;">Yield Curve</div>'
        f'<div style="font-size:0.6rem; color:{yc_color};">{yc_label}</div>'
        f'</div>'
    ) if yc_label else ""

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#13131f 0%,#0f0f1a 100%);
        border:1px solid {color}33; border-left:3px solid {color};
        border-radius:4px; padding:1.25rem 1.5rem; margin-bottom:1rem;">
        <div style="font-size:0.55rem; letter-spacing:0.2em; color:#444455; margin-bottom:0.5rem; text-transform:uppercase;">Macro Indicators</div>
        <div style="font-size:1.1rem; font-weight:600; color:{color}; letter-spacing:0.08em; margin-bottom:0.3rem;">{strength.upper()}  {direction_label}</div>
        <div style="font-size:0.65rem; color:#888899; margin-bottom:1rem;">{bull} bullish \u00b7 {bear} bearish \u00b7 {neutral} neutral</div>
        <div>{all_rows}</div>
        {yc_html}
    </div>
    """, unsafe_allow_html=True)


def agreement_banner(data_15m, data_1h):
    r15   = data_15m.get("regime", "")
    r1h   = data_1h.get("regime", "")
    agree = r15 == r1h
    if agree:
        st.markdown(f'''
        <div style="background:#001a0d; border:1px solid #00D26A33; border-left:3px solid #00D26A;
            border-radius:4px; padding:0.75rem 1.25rem; font-size:0.65rem; color:#00D26A;
            letter-spacing:0.08em; margin-bottom:1rem;">
            \u2713 BOTH TIMEFRAMES ALIGNED  \u00b7  {r15.replace('_',' ').upper()}  \u00b7  HIGH CONVICTION
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div style="background:#1a0a00; border:1px solid #FFB80033; border-left:3px solid #FFB800;
            border-radius:4px; padding:0.75rem 1.25rem; font-size:0.65rem; color:#FFB800;
            letter-spacing:0.08em; margin-bottom:1rem;">
            \u26a0 TIMEFRAME CONFLICT  \u00b7  15M: {r15.replace('_',' ').upper()}  \u00b7  1H: {r1h.replace('_',' ').upper()}  \u00b7  STAND ASIDE
        </div>
        ''', unsafe_allow_html=True)


# ── layout ────────────────────────────────────────────────────────────────────
st.title("FUTURES  //  HMM MARKET REGIME")

col_asset, col_toggle, col_tz, _ = st.columns([1, 1, 2, 3])
with col_asset:
    asset_options  = {k: v["name"] for k, v in TICKERS.items()}
    selected_asset = st.selectbox("ASSET", options=list(asset_options.keys()), format_func=lambda x: asset_options[x])
with col_toggle:
    show_mtf = st.toggle("MULTI-TIMEFRAME", value=False)
with col_tz:
    tz_options  = {"GMT+8 (Singapore/HK)": "Asia/Singapore", "GMT+0 (London)": "Europe/London", "GMT-5 (New York)": "America/New_York", "GMT-6 (Chicago)": "America/Chicago"}
    selected_tz = st.selectbox("TIMEZONE", options=list(tz_options.keys()), index=0)

tz     = tz_options[selected_tz]
ticker = TICKERS[selected_asset]["futures"]

divider()

all_data = fetch_regime(selected_asset)
price_df = fetch_price_data(ticker)
macro    = compute_macro_indicators(selected_asset)

if price_df.empty:
    st.warning("\u26a0 PRICE DATA UNAVAILABLE \u2014 yfinance fetch failed. Retrying on next refresh.")
    time.sleep(5)
    st.rerun()

last_bar_time  = price_df.index[-1]
now            = pd.Timestamp.now(tz=pytz.timezone(tz))
last_bar_local = last_bar_time.tz_convert(pytz.timezone(tz))
minutes_stale  = int((now - last_bar_local).seconds / 60)
if minutes_stale > 30:
    st.warning(f"\u26a0 DATA MAY BE STALE \u2014 last bar: {last_bar_local.strftime('%H:%M')} ({minutes_stale}min ago)")

if show_mtf:
    agreement_banner(all_data["15m"], all_data["1h"])
    col_15, col_1h = st.columns(2)
    with col_15:
        st.subheader("15 Minute")
        regime_card(all_data["15m"], "15M", tz=tz)
    with col_1h:
        st.subheader("1 Hour")
        regime_card(all_data["1h"], "1H", tz=tz)
else:
    st.subheader("15 Minute")
    regime_card(all_data["15m"], "15M", tz=tz)

divider()

st.subheader("Strategy")
strategy_card(all_data, show_mtf, price_df, macro, asset=selected_asset, ticker=ticker)

divider()

col_bias, col_macro = st.columns(2)
with col_bias:
    st.subheader("Structural Bias")
    bias_card(compute_bias(price_df, regime=all_data["15m"].get("regime", "ranging_low_vol"), ticker=ticker))
with col_macro:
    st.subheader("Macro Bias")
    macro_card(macro)

divider()

asset_name = TICKERS[selected_asset]["name"]
st.subheader(f"{asset_name}  \u00b7  15M  \u00b7  LAST 100 BARS")

chart_df             = price_df[["close"]].copy().reset_index()
chart_df["close"]    = pd.to_numeric(chart_df["close"], errors="coerce")
chart_df["Datetime"] = pd.to_datetime(chart_df["Datetime"], utc=True).dt.tz_convert(tz).dt.strftime("%m-%d %H:%M")

y_min = float(chart_df["close"].min()) * 0.9995
y_max = float(chart_df["close"].max()) * 1.0005

regime_seq      = all_data["15m"].get("regime_sequence", [])
n_bars          = len(chart_df)
n_regimes       = len(regime_seq)

if n_regimes >= n_bars:
    aligned_regimes = regime_seq[-n_bars:]
elif n_regimes > 0:
    aligned_regimes = ["unknown"] * (n_bars - n_regimes) + regime_seq
else:
    aligned_regimes = ["unknown"] * n_bars

fig = go.Figure()
for i, regime in enumerate(aligned_regimes):
    fig.add_shape(type="rect", x0=i-0.5, x1=i+0.5, y0=y_min, y1=y_max,
                  fillcolor=REGIME_COLORS.get(regime, REGIME_COLORS["unknown"]), line_width=0, layer="below")

fig.add_trace(go.Scatter(
    x=list(range(n_bars)), y=list(chart_df["close"]),
    mode="lines", line=dict(color="#3D9BE9", width=1.5),
    fill="tozeroy", fillcolor="rgba(61,155,233,0.04)",
    hovertemplate="%{text}<br>%{y:,.2f}<extra></extra>",
    text=list(chart_df["Datetime"]),
))

legend_items = {
    "trending_low_vol":  ("Trending Low Vol",  "#00D26A"),
    "trending_high_vol": ("Trending High Vol", "#FFB800"),
    "ranging_low_vol":   ("Ranging Low Vol",   "#3D9BE9"),
    "ranging_high_vol":  ("Ranging High Vol",  "#FF4455"),
    "extreme_vol":       ("Extreme Vol",       "#FF00FF"),
    "extreme_vol_2":     ("Extreme Vol 2", "#CC00CC"),
}

seen_regimes = set(aligned_regimes)
annotations, x_pos = [], 0.01
for regime, (lbl, clr) in legend_items.items():
    if regime in seen_regimes:
        annotations.append(dict(x=x_pos, y=1.02, xref="paper", yref="paper",
            text=f"<span style='color:{clr}'>\u25a0</span> {lbl}",
            showarrow=False, font=dict(size=8, color="#888899", family="IBM Plex Mono"), xanchor="left"))
        x_pos += 0.22

fig.update_layout(
    height=320, margin=dict(l=0, r=0, t=24, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0f",
    font=dict(color="#888899", family="IBM Plex Mono", size=10),
    annotations=annotations,
    xaxis=dict(gridcolor="#111120", tickangle=45, tickfont=dict(size=9),
               zeroline=False, showline=False, tickmode="array",
               tickvals=list(range(0, n_bars, max(1, n_bars // 12))),
               ticktext=[chart_df["Datetime"].iloc[i] for i in range(0, n_bars, max(1, n_bars // 12))]),
    yaxis=dict(gridcolor="#111120", range=[y_min, y_max], tickfont=dict(size=9),
               zeroline=False, tickformat=",.2f", showline=False),
)

chart_placeholder = st.empty()
with chart_placeholder:
    st.plotly_chart(fig, width="stretch")

time.sleep(5)
st.rerun()
