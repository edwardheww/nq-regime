# bic_analysis.py
# Run this once to determine the optimal number of HMM states for your data.
# Usage: python bic_analysis.py

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from config import TICKERS, N_STATES as DEFAULT_N_STATES


# ── config ────────────────────────────────────────────────────────────────────
TICKER        = "CL=F"
INTERVAL      = "15m"
PERIOD        = "60d"
MAX_STATES    = 7       # test 2 through 7
N_SEEDS       = 10      # random initialisations per n_states
N_ITER        = 1000


# ── data ──────────────────────────────────────────────────────────────────────
def fetch_features():
    df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    df["returns"]   = df["close"].pct_change()
    df["log_range"] = np.log((df["high"] - df["low"]).clip(lower=1e-8))
    df["vol_5"]     = df["returns"].shift(1).rolling(5).std()
    df.dropna(inplace=True)

    return df[["returns", "log_range", "vol_5"]].values


# ── stationarity check ────────────────────────────────────────────────────────
def check_stationarity(X: np.ndarray, feature_names: list):
    print("\n── Stationarity Check (ADF Test) ────────────────────────────────")
    print(f"{'Feature':<15} {'ADF Stat':>10} {'p-value':>10} {'Stationary':>12}")
    print("-" * 50)
    for i, name in enumerate(feature_names):
        result = adfuller(X[:, i])
        stat, pval = result[0], result[1]
        stationary = "YES" if pval < 0.05 else "NO — consider differencing"
        print(f"{name:<15} {stat:>10.3f} {pval:>10.4f} {stationary:>12}")


# ── bic calculation ───────────────────────────────────────────────────────────
def compute_bic(model: GaussianHMM, X: np.ndarray, n_states: int) -> float:
    n_features = X.shape[1]
    n_samples  = X.shape[0]

    # number of free parameters
    # transition matrix: n_states * (n_states - 1)
    # means: n_states * n_features
    # covariances (full): n_states * n_features * (n_features + 1) / 2
    # initial probabilities: n_states - 1
    n_params = (
        n_states * (n_states - 1) +
        n_states * n_features +
        n_states * n_features * (n_features + 1) // 2 +
        (n_states - 1)
    )

    log_likelihood = model.score(X) * n_samples
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    aic = -2 * log_likelihood + 2 * n_params

    return bic, aic, log_likelihood


# ── main ──────────────────────────────────────────────────────────────────────
def run():
    print("Fetching NQ data...")
    X_raw = fetch_features()
    print(f"Loaded {len(X_raw)} bars\n")

    # stationarity check
    check_stationarity(X_raw, ["returns", "log_range", "vol_5"])

    # scale
    scaler  = StandardScaler()
    X       = scaler.fit_transform(X_raw)

    results = []

    print("\n── BIC Analysis ─────────────────────────────────────────────────")
    print(f"{'n_states':<10} {'BIC':>12} {'AIC':>12} {'Log-Lik':>12} {'Best Seed':>10}")
    print("-" * 58)

    for n in range(2, MAX_STATES + 1):
        best_model = None
        best_ll    = -np.inf
        best_seed  = None

        for seed in range(N_SEEDS):
            try:
                model = GaussianHMM(
                    n_components=n,
                    covariance_type="full",
                    n_iter=N_ITER,
                    random_state=seed,
                    tol=1e-4,
                )
                model.fit(X)
                ll = model.score(X) * len(X)

                if ll > best_ll:
                    best_ll    = ll
                    best_model = model
                    best_seed  = seed

            except Exception:
                continue

        if best_model is None:
            continue

        bic, aic, ll = compute_bic(best_model, X, n)
        results.append({
            "n_states":  n,
            "bic":       bic,
            "aic":       aic,
            "log_lik":   ll,
            "best_seed": best_seed,
            "model":     best_model,
        })

        print(f"{n:<10} {bic:>12.1f} {aic:>12.1f} {ll:>12.1f} {best_seed:>10}")

    # ── results ───────────────────────────────────────────────────────────────
    best_bic = min(results, key=lambda x: x["bic"])
    best_aic = min(results, key=lambda x: x["aic"])

    print("\n── Recommendation ───────────────────────────────────────────────")
    print(f"Optimal n_states by BIC: {best_bic['n_states']}")
    print(f"Optimal n_states by AIC: {best_aic['n_states']}")

    if best_bic["n_states"] == best_aic["n_states"]:
        optimal = best_bic["n_states"]
        print(f"\nBoth criteria agree: use n_states = {optimal}")
    else:
        print(f"\nCriteria disagree — BIC favours fewer states (less overfit)")
        print(f"Recommendation: use BIC result → n_states = {best_bic['n_states']}")
        optimal = best_bic["n_states"]

    # ── transition matrix of best model ──────────────────────────────────────
    print(f"\n── Transition Matrix (n_states={optimal}) ───────────────────────")
    best_model = best_bic["model"]
    tm         = best_model.transmat_

    header = "        " + "".join(f"  S{j:<5}" for j in range(optimal))
    print(header)
    for i in range(optimal):
        row = f"  S{i}   " + "".join(f"  {tm[i,j]:.3f}" for j in range(optimal))
        print(row)

    print("\nDiagonal values = self-transition probability (regime stickiness)")
    print("High diagonal (>0.90) = sticky regime = high confidence signals")
    print("Low diagonal (<0.80)  = unstable regime = reduce signal confidence")

    for i in range(optimal):
        self_prob = tm[i, i]
        sticky    = "STICKY" if self_prob > 0.90 else "UNSTABLE" if self_prob < 0.80 else "MODERATE"
        print(f"  S{i}: {self_prob:.3f} → {sticky}")

    # ── state characteristics ─────────────────────────────────────────────────
    print(f"\n── State Characteristics ────────────────────────────────────────")
    states = best_model.predict(X)
    df_raw = pd.DataFrame(X_raw, columns=["returns", "log_range", "vol_5"])
    df_raw["state"] = states

    print(f"{'State':<8} {'Mean Ret':>10} {'Mean Vol':>10} {'Count':>8} {'% Time':>8}")
    print("-" * 48)
    for s in range(optimal):
        sub      = df_raw[df_raw["state"] == s]
        mean_ret = sub["returns"].mean() * 100
        mean_vol = sub["vol_5"].mean() * 100
        count    = len(sub)
        pct      = count / len(df_raw) * 100
        print(f"  S{s}    {mean_ret:>10.4f} {mean_vol:>10.4f} {count:>8} {pct:>7.1f}%")

    asset_key       = next((k for k, v in TICKERS.items() if v["futures"] == TICKER), None)
    current_n_states = TICKERS[asset_key]["n_states"] if asset_key else DEFAULT_N_STATES

    print(f"\n── Action Required ──────────────────────────────────────────────")
    if optimal != current_n_states:
        print(f"Your current config uses n_states={current_n_states}.")
        print(f"BIC suggests n_states={optimal} is more appropriate for your data.")
        print(f"Update config.py: n_states = {optimal} for {asset_key or TICKER}")
        print(f"Then restart uvicorn to retrain with the new setting.")
    else:
        print(f"n_states={current_n_states} is confirmed as optimal. No changes needed.")


if __name__ == "__main__":
    run()