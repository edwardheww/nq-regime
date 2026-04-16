# NQ HMM Regime Detection

A Hidden Markov Model (HMM) system for detecting market regimes across NQ, Gold, and Crude Oil futures. Provides real-time regime signals via a FastAPI backend and a live Streamlit dashboard.

## What It Does

- Trains a Gaussian HMM on price features (returns, range, rolling volatility) for three futures contracts across two timeframes (15m, 1h)
- Classifies market state into regimes: trending/ranging × low/normal/high volatility
- Retrains every 15 minutes on the latest data
- Displays regime confidence, stickiness, multi-timeframe alignment, macro context, and strategy recommendations

## Project Structure

```
main.py          # FastAPI backend — HMM retraining scheduler, /regime endpoint
dashboard.py     # Streamlit frontend — live regime dashboard
hmm_engine.py    # Core HMM logic — training, decoding, state labeling
data.py          # Feature engineering — OHLCV fetch and preprocessing
bias.py          # Structural bias — gaps, VWAP, swing structure
macro.py         # Macro indicators — VIX, yields, DXY, QQQ/SPY ratio
config.py        # Configuration — state counts, lookback windows, tickers
bic_analysis.py  # Model selection — BIC/AIC sweep for optimal state count
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Start the backend (retrains HMM on startup and every 15 minutes):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Start the dashboard (auto-refreshes every 5 seconds):

```bash
streamlit run dashboard.py --server.port 8501
```

## Configuration

Key settings in `config.py`:

| Setting | Value |
|---|---|
| State counts | NQ: 5, GC: 5, CL: 6 |
| Lookback (15m) | 60 days |
| Lookback (1h) | 180 days |
| Retrain interval | 15 minutes |
| Confidence threshold | ≥70% raw, ≥65% adjusted |
| Stickiness threshold | ≥90% self-transition |

## Assets

| Asset | Ticker | Timeframes |
|---|---|---|
| Nasdaq Futures | NQ=F | 15m, 1h |
| Gold Futures | GC=F | 15m, 1h |
| Crude Oil Futures | CL=F | 15m, 1h |
