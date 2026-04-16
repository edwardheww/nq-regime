# bias.py

import numpy as np
import pandas as pd
import yfinance as yf


# ── helpers ───────────────────────────────────────────────────────────────────
def _fetch_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    return df.dropna()


# ── signal 1: overnight gap ───────────────────────────────────────────────────
def overnight_gap(df: pd.DataFrame, ticker: str) -> dict:
    try:
        daily        = _fetch_daily(ticker)
        prior_close  = float(daily["close"].iloc[-2])
        current_open = float(df["open"].iloc[0])
        gap_pct      = (current_open - prior_close) / prior_close * 100

        if abs(gap_pct) < 0.2:
            return {"vote": 0,  "label": f"Gap flat ({gap_pct:+.2f}%) — no bias", "gap_pct": gap_pct}
        elif gap_pct > 0:
            return {"vote": 1,  "label": f"Gap up {gap_pct:+.2f}% — bullish continuation bias", "gap_pct": gap_pct}
        else:
            return {"vote": -1, "label": f"Gap down {gap_pct:+.2f}% — bearish continuation bias", "gap_pct": gap_pct}
    except Exception:
        return {"vote": 0, "label": "Gap — data unavailable", "gap_pct": 0.0}


# ── signal 2: prior day levels ────────────────────────────────────────────────
def prior_day_levels(df: pd.DataFrame, ticker: str) -> dict:
    try:
        daily       = _fetch_daily(ticker)
        pdh         = float(daily["high"].iloc[-2])
        pdl         = float(daily["low"].iloc[-2])
        pdc         = float(daily["close"].iloc[-2])
        price       = float(df["close"].iloc[-1])
        pd_midpoint = pdl + (pdh - pdl) / 2

        if price > pdh:
            return {"vote":  1, "label": f"Above PDH ({pdh:,.1f}) — bullish breakout"}
        elif price > pdc:
            return {"vote":  1, "label": f"Above PDC ({pdc:,.1f}), below PDH — mild bullish"}
        elif price > pd_midpoint:
            return {"vote":  0, "label": f"Upper half prior range — neutral"}
        elif price > pdl:
            return {"vote": -1, "label": f"Lower half prior range — mild bearish"}
        else:
            return {"vote": -1, "label": f"Below PDL ({pdl:,.1f}) — bearish breakdown"}
    except Exception:
        return {"vote": 0, "label": "Prior day levels — data unavailable"}


# ── signal 3: vwap z-score ────────────────────────────────────────────────────
def vwap_zscore(df: pd.DataFrame) -> dict:
    try:
        today = df[df.index.date == df.index[-1].date()].copy()
        if len(today) < 4:
            return {"vote": 0, "label": "VWAP — insufficient session bars", "z_score": 0.0}

        typical  = (today["high"] + today["low"] + today["close"]) / 3
        vol      = today["volume"]
        vwap     = (typical * vol).cumsum() / vol.cumsum()
        variance = ((typical - vwap) ** 2 * vol).cumsum() / vol.cumsum()
        std      = variance ** 0.5

        price    = float(today["close"].iloc[-1])
        vwap_val = float(vwap.iloc[-1])
        std_val  = float(std.iloc[-1])
        z        = (price - vwap_val) / std_val if std_val > 0 else 0.0

        if z > 2.0:
            return {"vote": -1, "label": f"VWAP +{z:.2f}σ — extended above, mean reversion bias", "z_score": round(z, 3)}
        elif z > 1.0:
            return {"vote":  0, "label": f"VWAP +{z:.2f}σ — above fair value, neutral", "z_score": round(z, 3)}
        elif z < -2.0:
            return {"vote":  1, "label": f"VWAP {z:.2f}σ — extended below, mean reversion bias", "z_score": round(z, 3)}
        elif z < -1.0:
            return {"vote":  0, "label": f"VWAP {z:.2f}σ — below fair value, neutral", "z_score": round(z, 3)}
        else:
            return {"vote":  0, "label": f"VWAP {z:.2f}σ — near fair value, neutral", "z_score": round(z, 3)}
    except Exception:
        return {"vote": 0, "label": "VWAP — computation error", "z_score": 0.0}


# ── signal 4: relative volume ─────────────────────────────────────────────────
def relative_volume(df: pd.DataFrame) -> dict:
    try:
        today      = df[df.index.date == df.index[-1].date()].copy()
        prior_days = df[df.index.date != df.index[-1].date()].copy()

        if today.empty or prior_days.empty:
            return {"vote": 0, "label": "RVOL — insufficient data", "rvol": 1.0}

        current_time = today.index[-1].time()
        current_vol  = float(today["volume"].iloc[-1])
        same_time    = prior_days[prior_days.index.time == current_time]["volume"]

        if same_time.empty:
            return {"vote": 0, "label": "RVOL — no historical comparison", "rvol": 1.0}

        avg_vol = float(same_time.mean())
        rvol    = current_vol / avg_vol if avg_vol > 0 else 1.0

        if rvol > 2.0:
            label = f"RVOL {rvol:.1f}x — high participation, signals reliable"
        elif rvol > 1.2:
            label = f"RVOL {rvol:.1f}x — above average participation"
        elif rvol < 0.5:
            label = f"RVOL {rvol:.1f}x — very low participation, signals unreliable"
        else:
            label = f"RVOL {rvol:.1f}x — normal participation"

        return {"vote": 0, "label": label, "rvol": round(rvol, 2)}
    except Exception:
        return {"vote": 0, "label": "RVOL — computation error", "rvol": 1.0}


# ── signal 5: swing structure ─────────────────────────────────────────────────
def swing_structure(df: pd.DataFrame, lookback: int = 20) -> dict:
    try:
        recent = df.tail(lookback + 1).iloc[:-1].copy()
        highs  = recent["high"].values
        lows   = recent["low"].values

        swing_highs, swing_lows = [], []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"vote": 0, "label": "Structure — insufficient pivots"}

        hh = swing_highs[-1] > swing_highs[-2]
        hl = swing_lows[-1]  > swing_lows[-2]
        lh = swing_highs[-1] < swing_highs[-2]
        ll = swing_lows[-1]  < swing_lows[-2]

        if hh and hl:
            return {"vote":  1, "label": "Structure HH/HL — bullish"}
        elif lh and ll:
            return {"vote": -1, "label": "Structure LH/LL — bearish"}
        elif hh and ll:
            return {"vote":  0, "label": "Structure expanding — neutral"}
        elif lh and hl:
            return {"vote":  0, "label": "Structure contracting — neutral"}
        else:
            return {"vote":  0, "label": "Structure mixed — no bias"}
    except Exception:
        return {"vote": 0, "label": "Structure — computation error"}


# ── aggregation ───────────────────────────────────────────────────────────────
def compute_bias(df: pd.DataFrame, regime: str = "ranging_low_vol", ticker: str = "NQ=F") -> dict:
    if df.empty:
        return {
            "direction": "neutral", "strength": "No Data",
            "conviction": 0.0, "score": 0.0, "rvol": 1.0,
            "bull_votes": 0, "bear_votes": 0, "neutral_votes": 0,
            "total_votes": 0, "reasons": {},
        }

    gap       = overnight_gap(df, ticker)
    pdl       = prior_day_levels(df, ticker)
    vwap      = vwap_zscore(df)
    rvol      = relative_volume(df)
    structure = swing_structure(df)

    signals = {
        "Overnight Gap":   gap,
        "Prior Day Levels": pdl,
        "VWAP Z-Score":    vwap,
        "Swing Structure": structure,
    }

    if regime in ("trending_low_vol", "trending_high_vol"):
        weights = {"Overnight Gap": 0.30, "Prior Day Levels": 0.20,
                   "VWAP Z-Score": 0.15, "Swing Structure": 0.35}
    else:
        weights = {"Overnight Gap": 0.15, "Prior Day Levels": 0.30,
                   "VWAP Z-Score": 0.40, "Swing Structure": 0.15}

    rvol_val  = rvol.get("rvol", 1.0)
    rvol_mult = min(rvol_val / 1.5, 1.0)

    raw_score   = sum(signals[k]["vote"] * weights[k] for k in weights)
    final_score = raw_score * rvol_mult

    if final_score >= 0.25:
        direction, strength = "long",    "Strong"
    elif final_score >= 0.10:
        direction, strength = "long",    "Lean"
    elif final_score <= -0.25:
        direction, strength = "short",   "Strong"
    elif final_score <= -0.10:
        direction, strength = "short",   "Lean"
    else:
        direction, strength = "neutral", "Neutral"

    reasons = {
        "gap":       gap["label"],
        "pdl":       pdl["label"],
        "vwap":      vwap["label"],
        "structure": structure["label"],
        "rvol":      rvol["label"],
    }

    return {
        "direction":     direction,
        "strength":      strength,
        "conviction":    round(abs(final_score), 4),
        "score":         round(final_score, 4),
        "rvol":          round(rvol_val, 2),
        "bull_votes":    sum(1 for k in signals if signals[k]["vote"] ==  1),
        "bear_votes":    sum(1 for k in signals if signals[k]["vote"] == -1),
        "neutral_votes": sum(1 for k in signals if signals[k]["vote"] ==  0),
        "total_votes":   len(signals),
        "reasons":       reasons,
    }
