# data.py

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_features(ticker: str, interval: str, lookback: str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Fetch OHLCV data for a given ticker and engineer HMM features.
    Retries up to 3 times to handle yfinance intermittent failures.
    """
    last_error = None

    for attempt in range(3):
        try:
            df = yf.download(ticker, interval=interval, period=lookback, progress=False)

            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]

            df["returns"]   = df["close"].pct_change()
            df["log_range"] = np.log((df["high"] - df["low"]).clip(lower=1e-8))
            df["vol_5"]     = df["returns"].shift(1).rolling(5).std()
            df.dropna(inplace=True)

            if df.empty or len(df) < 100:
                continue

            X = df[["returns", "log_range", "vol_5"]].values
            return X, df

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"fetch_features failed for {ticker} after 3 attempts: {last_error}")