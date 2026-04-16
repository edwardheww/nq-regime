# macro.py

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# ── economic calendar ─────────────────────────────────────────────────────────
# NQ events — update monthly
NQ_EVENTS = [
    {"name": "CPI",     "date": "2026-04-15", "impact": "HIGH"},
    {"name": "FOMC",    "date": "2026-04-29", "impact": "HIGH"},
    {"name": "NFP",     "date": "2026-05-01", "impact": "HIGH"},
    {"name": "GDP",     "date": "2026-04-30", "impact": "HIGH"},
    {"name": "PCE",     "date": "2026-04-30", "impact": "HIGH"},
    {"name": "RETAIL",  "date": "2026-04-16", "impact": "MEDIUM"},
    {"name": "JOBLESS", "date": "2026-04-17", "impact": "MEDIUM"},
]

# GC/CL events — oil inventory report is key for CL
CL_EVENTS = [
    {"name": "EIA OIL",  "date": "2026-04-16", "impact": "HIGH"},
    {"name": "EIA OIL",  "date": "2026-04-23", "impact": "HIGH"},
    {"name": "EIA OIL",  "date": "2026-04-30", "impact": "HIGH"},
    {"name": "CPI",      "date": "2026-04-15", "impact": "HIGH"},
    {"name": "FOMC",     "date": "2026-04-29", "impact": "HIGH"},
]

GC_EVENTS = [
    {"name": "CPI",   "date": "2026-04-15", "impact": "HIGH"},
    {"name": "FOMC",  "date": "2026-04-29", "impact": "HIGH"},
    {"name": "NFP",   "date": "2026-05-01", "impact": "HIGH"},
]

ASSET_EVENTS = {"NQ": NQ_EVENTS, "GC": GC_EVENTS, "CL": CL_EVENTS}


# ── asset-specific macro configs ──────────────────────────────────────────────
# NQ: tech-focused — VIX, QQQ/SPY, yields, DXY
# GC: gold — VIX, DXY (most important for gold), yields, real rates
# CL: oil — VIX, DXY, supply/demand proxies

ASSET_WEIGHTS = {
    "NQ": {
        "^VIX":     0.28,
        "QQQ/SPY":  0.28,
        "^TNX":     0.20,
        "^IRX":     0.14,
        "DX-Y.NYB": 0.10,
    },
    "GC": {
        "^VIX":     0.20,
        "DX-Y.NYB": 0.40,   # gold most sensitive to dollar
        "^TNX":     0.25,   # real rates matter a lot for gold
        "^IRX":     0.15,
    },
    "CL": {
        "^VIX":     0.20,
        "DX-Y.NYB": 0.30,   # dollar affects oil pricing
        "^TNX":     0.15,
        "^IRX":     0.10,
        "CL_SPREAD": 0.25,  # WTI/Brent spread as supply signal (approximated)
    },
}


# ── data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_indicator(ticker: str) -> dict | None:
    try:
        df = yf.download(ticker, period="5d", interval="1h", progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]

        current  = float(df["close"].iloc[-1])
        prev     = float(df["close"].iloc[-2])
        day_ago  = float(df["close"].iloc[-8]) if len(df) >= 8 else prev
        week_ago = float(df["close"].iloc[0])

        return {
            "current": round(current, 4),
            "chg_1h":  round((current - prev) / prev * 100, 3),
            "chg_1d":  round((current - day_ago) / day_ago * 100, 3),
            "chg_1w":  round((current - week_ago) / week_ago * 100, 3),
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_qqq_spy_ratio() -> dict | None:
    try:
        qqq = fetch_indicator("QQQ")
        spy = fetch_indicator("SPY")
        if not qqq or not spy:
            return None
        return {
            "current": round(qqq["current"] / spy["current"], 4),
            "chg_1d":  round(qqq["chg_1d"] - spy["chg_1d"], 3),
            "chg_1h":  round(qqq["chg_1h"] - spy["chg_1h"], 3),
        }
    except Exception:
        return None


# ── signal interpretation ─────────────────────────────────────────────────────
def interpret_vix(data: dict) -> tuple:
    if not data:
        return 0, "VIX — unavailable", "#666677"
    current, chg_1d = data["current"], data["chg_1d"]

    level_vote = 1 if current <= 20 else -1
    level_label = f"{current:.1f} CALM" if current <= 20 else \
                  f"{current:.1f} ELEVATED" if current <= 30 else f"{current:.1f} EXTREME FEAR"

    if chg_1d > 5.0:
        direction_label = "↑ SPIKING"
        if level_vote == 1: level_vote = -1
    elif chg_1d > 2.0:
        direction_label = "↑ RISING"
        if level_vote == 1: level_vote = 0
    elif chg_1d < -5.0:
        direction_label = "↓ COLLAPSING"
        if level_vote == -1: level_vote = 1
    elif chg_1d < -2.0:
        direction_label = "↓ FALLING"
        if level_vote == -1: level_vote = 0
    else:
        direction_label = "→ STABLE"

    color = "#00D26A" if level_vote == 1 else "#FF4455" if level_vote == -1 else "#FFB800"
    return level_vote, f"{level_label}  {direction_label}", color


def interpret_10y(data: dict) -> tuple:
    if not data:
        return 0, "10Y — unavailable", "#666677"
    current, chg_1d = data["current"], data["chg_1d"]
    if chg_1d > 0.08:   return -1, f"{current:.2f}%  RISING SHARPLY",  "#FF4455"
    elif chg_1d > 0.03: return -1, f"{current:.2f}%  RISING",          "#FFB800"
    elif chg_1d < -0.08:return  1, f"{current:.2f}%  FALLING SHARPLY", "#00D26A"
    elif chg_1d < -0.03:return  1, f"{current:.2f}%  FALLING",         "#00D26A"
    else:               return  0, f"{current:.2f}%  STABLE",          "#888899"


def interpret_2y(data: dict) -> tuple:
    if not data:
        return 0, "2Y — unavailable", "#666677"
    current, chg_1d = data["current"], data["chg_1d"]
    if chg_1d > 0.08:   return -1, f"{current:.2f}%  RISING SHARPLY — hawkish",  "#FF4455"
    elif chg_1d > 0.03: return -1, f"{current:.2f}%  RISING — hawkish bias",     "#FFB800"
    elif chg_1d < -0.08:return  1, f"{current:.2f}%  FALLING SHARPLY — dovish",  "#00D26A"
    elif chg_1d < -0.03:return  1, f"{current:.2f}%  FALLING — dovish bias",     "#00D26A"
    else:               return  0, f"{current:.2f}%  STABLE",                    "#888899"


def interpret_dxy(data: dict, asset: str = "NQ") -> tuple:
    if not data:
        return 0, "DXY — unavailable", "#666677"
    current, chg_1d = data["current"], data["chg_1d"]

    # for gold and oil, dollar relationship is inverted and stronger
    threshold = 0.3 if asset in ("GC", "CL") else 0.5

    if chg_1d > threshold:
        vote  = -1
        label = f"{current:.1f}  STRENGTHENING"
        color = "#FF4455"
    elif chg_1d < -threshold:
        vote  = 1
        label = f"{current:.1f}  WEAKENING"
        color = "#00D26A"
    else:
        vote  = 0
        label = f"{current:.1f}  STABLE"
        color = "#888899"

    return vote, label, color


def interpret_qqq_spy(data: dict) -> tuple:
    if not data:
        return 0, "QQQ/SPY — unavailable", "#666677"
    ratio, chg_1d = data["current"], data["chg_1d"]
    if chg_1d > 0.5:    return  1, f"{ratio:.3f}  NQ OUTPERFORMING — tech bid",           "#00D26A"
    elif chg_1d > 0.2:  return  1, f"{ratio:.3f}  NQ MILD OUTPERFORM",                    "#00D26A"
    elif chg_1d < -0.5: return -1, f"{ratio:.3f}  NQ UNDERPERFORMING — rotation out",     "#FF4455"
    elif chg_1d < -0.2: return -1, f"{ratio:.3f}  NQ MILD UNDERPERFORM",                  "#FFB800"
    else:               return  0, f"{ratio:.3f}  IN LINE WITH BROAD MARKET",              "#888899"


def interpret_cl_spread(wti_data: dict, brent_data: dict) -> tuple:
    """WTI/Brent spread — negative spread = supply tightness = bullish CL"""
    if not wti_data or not brent_data:
        return 0, "WTI/Brent spread — unavailable", "#666677"
    spread = wti_data["current"] - brent_data["current"]
    if spread > 1.0:
        return  1, f"WTI premium ${spread:.1f} — US supply tight, bullish", "#00D26A"
    elif spread < -1.0:
        return -1, f"Brent premium ${abs(spread):.1f} — global supply tight, neutral CL", "#FFB800"
    else:
        return  0, f"WTI/Brent spread ${spread:.1f} — balanced", "#888899"


def compute_yield_curve(data_10y: dict, data_2y: dict) -> dict:
    if not data_10y or not data_2y:
        return {"spread": None, "inverted": False, "label": "Yield curve — unavailable"}
    spread   = data_10y["current"] - data_2y["current"]
    inverted = spread < 0
    if inverted:
        label = f"INVERTED ({spread:+.2f}%) — recession signal, structurally bearish"
    elif spread < 0.3:
        label = f"FLAT ({spread:+.2f}%) — caution"
    else:
        label = f"NORMAL ({spread:+.2f}%) — no recession signal"
    return {"spread": round(spread, 3), "inverted": inverted, "label": label}


def get_upcoming_events(asset: str = "NQ", within_days: int = 7) -> list:
    now    = datetime.now(timezone.utc).date()
    events = ASSET_EVENTS.get(asset, NQ_EVENTS)
    result = []
    for e in events:
        event_date = datetime.strptime(e["date"], "%Y-%m-%d").date()
        days_away  = (event_date - now).days
        if 0 <= days_away <= within_days:
            result.append({**e, "days_away": days_away})
    return sorted(result, key=lambda x: x["days_away"])


# ── main computation ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def compute_macro_indicators(asset: str = "NQ") -> dict:
    weights = ASSET_WEIGHTS.get(asset, ASSET_WEIGHTS["NQ"])

    # fetch common indicators
    data_vix = fetch_indicator("^VIX")
    data_10y = fetch_indicator("^TNX")
    data_2y  = fetch_indicator("^IRX")
    data_dxy = fetch_indicator("DX-Y.NYB")

    vix_vote,  vix_label,  vix_color  = interpret_vix(data_vix)
    teny_vote, teny_label, teny_color = interpret_10y(data_10y)
    twoy_vote, twoy_label, twoy_color = interpret_2y(data_2y)
    dxy_vote,  dxy_label,  dxy_color  = interpret_dxy(data_dxy, asset)

    yield_curve = compute_yield_curve(data_10y, data_2y)
    inversion_penalty = -0.15 if yield_curve["inverted"] else 0.0

    indicators = {
        "^VIX": {"name": "VIX", "desc": "Fear Index",
                 "data": data_vix, "vote": vix_vote, "label": vix_label, "color": vix_color},
        "^TNX": {"name": "10Y", "desc": "10Y Yield",
                 "data": data_10y, "vote": teny_vote, "label": teny_label, "color": teny_color},
        "^IRX": {"name": "2Y",  "desc": "2Y Yield",
                 "data": data_2y, "vote": twoy_vote, "label": twoy_label, "color": twoy_color},
        "DX-Y.NYB": {"name": "DXY", "desc": "US Dollar",
                     "data": data_dxy, "vote": dxy_vote, "label": dxy_label, "color": dxy_color},
    }

    votes = {
        "^VIX": vix_vote, "^TNX": teny_vote,
        "^IRX": twoy_vote, "DX-Y.NYB": dxy_vote,
    }

    # asset-specific additional indicators
    if asset == "NQ":
        data_ratio = fetch_qqq_spy_ratio()
        ratio_vote, ratio_label, ratio_color = interpret_qqq_spy(data_ratio)
        indicators["QQQ/SPY"] = {
            "name": "QQQ/SPY", "desc": "Tech vs Broad",
            "data": data_ratio, "vote": ratio_vote,
            "label": ratio_label, "color": ratio_color,
        }
        votes["QQQ/SPY"] = ratio_vote

    elif asset == "CL":
        data_brent = fetch_indicator("BZ=F")
        data_wti   = fetch_indicator("CL=F")
        cl_vote, cl_label, cl_color = interpret_cl_spread(data_wti, data_brent)
        indicators["CL_SPREAD"] = {
            "name": "WTI/Brent", "desc": "Supply Signal",
            "data": data_wti, "vote": cl_vote,
            "label": cl_label, "color": cl_color,
        }
        votes["CL_SPREAD"] = cl_vote

    # weighted score
    raw_score   = sum(votes.get(k, 0) * w for k, w in weights.items())
    final_score = raw_score + inversion_penalty

    bull    = sum(1 for v in votes.values() if v ==  1)
    bear    = sum(1 for v in votes.values() if v == -1)
    neutral = sum(1 for v in votes.values() if v ==  0)

    if final_score > 0.15:
        direction = "bullish"
        strength  = "Strong" if final_score > 0.25 else "Lean"
    elif final_score < -0.15:
        direction = "bearish"
        strength  = "Strong" if final_score < -0.25 else "Lean"
    else:
        direction = "neutral"
        strength  = "Neutral"

    return {
        "direction":   direction,
        "strength":    strength,
        "score":       round(final_score, 4),
        "bull":        bull,
        "bear":        bear,
        "neutral":     neutral,
        "indicators":  indicators,
        "yield_curve": yield_curve,
        "events":      get_upcoming_events(asset),
    }
