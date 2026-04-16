# config.py

N_STATES     = 5
N_ITER       = 1000
RETRAIN_MINS = 15

TICKERS = {
    "NQ": {
        "futures":      "NQ=F",
        "name":         "Nasdaq Futures",
        "interval_15m": "15m",
        "interval_1h":  "1h",
        "lookback_15m": "60d",
        "lookback_1h":  "180d",
        "n_states":     5,      # validated by BIC analysis
    },
    "GC": {
        "futures":      "GC=F",
        "name":         "Gold Futures",
        "interval_15m": "15m",
        "interval_1h":  "1h",
        "lookback_15m": "60d",
        "lookback_1h":  "180d",
        "n_states":     5,      # gold is slower, fewer distinct regimes
    },
    "CL": {
        "futures":      "CL=F",
        "name":         "Crude Oil Futures",
        "interval_15m": "15m",
        "interval_1h":  "1h",
        "lookback_15m": "60d",
        "lookback_1h":  "180d",
        "n_states":     6,      # placeholder — run BIC to confirm
    },
}

N_STATES     = 5   # default fallback
N_ITER       = 1000
RETRAIN_MINS = 15
