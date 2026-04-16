# main.py

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from config import TICKERS, RETRAIN_MINS, N_STATES
from data import fetch_features
from hmm_engine import decode, get_current_regime, label_states, train

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ── state ─────────────────────────────────────────────────────────────────────
def _empty_tf():
    return {
        "result":          {"regime": "initialising", "confidence": 0.0},
        "updated_at":      None,
        "label_map":       {},
        "regime_since":    None,
        "last_regime":     None,
        "regime_sequence": [],
    }


state = {
    asset: {"15m": _empty_tf(), "1h": _empty_tf()}
    for asset in TICKERS
}


# ── retraining ────────────────────────────────────────────────────────────────
def retrain(asset: str, ticker: str, interval: str, lookback: str, tf: str):
    cfg      = TICKERS[asset]
    n_states = cfg.get("n_states", N_STATES)

    log.info(f"Retraining HMM [{asset} {tf}] (n_states={n_states})...")
    try:
        X, df         = fetch_features(ticker, interval, lookback)
        model, scaler = train(X, n_states=n_states)
        label_map     = label_states(model, scaler, X, df, asset=asset, n_states=n_states)
        result        = get_current_regime(model, scaler, X, label_map)

        # full sequence for chart overlay
        states_seq = decode(model, scaler, X)
        regime_seq = [label_map.get(int(s), "unknown") for s in states_seq]

        # regime duration tracking
        prev_regime = state[asset][tf].get("last_regime")
        if prev_regime != result["regime"]:
            state[asset][tf]["regime_since"] = datetime.now(timezone.utc).isoformat()
        state[asset][tf]["last_regime"] = result["regime"]

        state[asset][tf]["result"]          = result
        state[asset][tf]["updated_at"]      = datetime.now(timezone.utc).isoformat()
        state[asset][tf]["label_map"]       = label_map
        state[asset][tf]["regime_sequence"] = regime_seq[-100:]

        log.info(
            f"[{asset} {tf}] Regime: {result['regime']} | "
            f"Conf: {result['confidence']:.3f} | "
            f"Adj: {result['adjusted_confidence']:.3f}"
        )
    except Exception as e:
        log.error(f"Retrain [{asset} {tf}] failed: {e}")


def retrain_all():
    for asset, cfg in TICKERS.items():
        retrain(asset, cfg["futures"], cfg["interval_15m"], cfg["lookback_15m"], "15m")
        retrain(asset, cfg["futures"], cfg["interval_1h"],  cfg["lookback_1h"],  "1h")


# ── lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    retrain_all()
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_all, "interval", minutes=RETRAIN_MINS)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)


# ── endpoints ─────────────────────────────────────────────────────────────────
def _build_tf_response(asset: str, tf: str, interval_mins: int) -> dict:
    s = state[asset][tf]

    def bars_from_since(since_str):
        if not since_str:
            return 0
        since        = datetime.fromisoformat(since_str)
        now          = datetime.now(timezone.utc)
        elapsed_mins = (now - since).total_seconds() / 60
        return max(1, int(elapsed_mins / interval_mins))

    return {
        **s["result"],
        "updated_at":      s["updated_at"],
        "label_map":       {str(k): v for k, v in s.get("label_map", {}).items()},
        "regime_bars":     bars_from_since(s.get("regime_since")),
        "regime_since":    s.get("regime_since"),
        "regime_sequence": s.get("regime_sequence", []),
    }


@app.get("/regime")
def get_regime(asset: str = "NQ"):
    if asset not in state:
        return {"error": f"Unknown asset: {asset}. Valid: {list(TICKERS.keys())}"}
    return {
        "15m": _build_tf_response(asset, "15m", 15),
        "1h":  _build_tf_response(asset, "1h",  60),
    }


@app.get("/assets")
def get_assets():
    return {
        asset: {"name": cfg["name"], "futures": cfg["futures"]}
        for asset, cfg in TICKERS.items()
    }


@app.get("/health")
def health():
    return {"status": "ok"}
