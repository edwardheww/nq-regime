# hmm_engine.py

import json
import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config import N_STATES, N_ITER


# ── training ──────────────────────────────────────────────────────────────────
def train(X: np.ndarray, n_states: int = N_STATES) -> tuple[GaussianHMM, StandardScaler]:
    """
    Train GaussianHMM with multiple random seeds, select best by log-likelihood.
    n_states is per-asset configurable.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_model = None
    best_score = -np.inf

    for seed in range(5):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=N_ITER,
                random_state=seed,
                tol=1e-4,
            )
            model.fit(X_scaled)
            score = model.score(X_scaled)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("All HMM training seeds failed.")

    return best_model, scaler


# ── decoding ──────────────────────────────────────────────────────────────────
def decode(model: GaussianHMM, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return model.predict(scaler.transform(X))


# ── regime classification ─────────────────────────────────────────────────────
def get_current_regime(
    model: GaussianHMM,
    scaler: StandardScaler,
    X: np.ndarray,
    label_map: dict,
) -> dict:
    _, posteriors   = model.score_samples(scaler.transform(X))
    last_posteriors = posteriors[-1]
    current_state   = int(np.argmax(last_posteriors))
    confidence      = float(last_posteriors[current_state])
    self_transition = float(model.transmat_[current_state, current_state])

    return {
        "state":               current_state,
        "regime":              label_map.get(current_state, "unknown"),
        "confidence":          round(confidence, 4),
        "adjusted_confidence": round(confidence * self_transition, 4),
        "self_transition":     round(self_transition, 4),
        "posteriors":          {i: round(float(p), 4) for i, p in enumerate(last_posteriors)},
    }


# ── state anchoring ───────────────────────────────────────────────────────────
def get_anchor_file(asset: str) -> str:
    return f"state_anchors_{asset}.json"


def save_anchors(stats: dict, asset: str):
    with open(get_anchor_file(asset), "w") as f:
        json.dump(stats, f)


def load_anchors(asset: str) -> dict:
    path = get_anchor_file(asset)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def match_states_to_anchors(stats: dict, anchors: dict) -> dict:
    # Normalize both dimensions by their range across current states + anchors
    all_returns  = [stats[s]["mean_return"] for s in stats] + [a["mean_return"] for a in anchors.values()]
    all_vols     = [stats[s]["mean_vol"]    for s in stats] + [a["mean_vol"]    for a in anchors.values()]
    return_range = max(all_returns) - min(all_returns) or 1.0
    vol_range    = max(all_vols)    - min(all_vols)    or 1.0

    mapping      = {}
    used_anchors = set()
    for state, stat in stats.items():
        best_label, best_dist = None, float("inf")
        for anchor_label, anchor_stat in anchors.items():
            if anchor_label in used_anchors:
                continue
            dist = (
                abs(stat["mean_return"] - anchor_stat["mean_return"]) / return_range +
                abs(stat["mean_vol"]    - anchor_stat["mean_vol"])    / vol_range
            )
            if dist < best_dist:
                best_dist, best_label = dist, anchor_label
        mapping[state] = best_label
        used_anchors.add(best_label)
    return mapping


def _label_fresh(stats: dict) -> dict:
    """
    Label states based on return and vol characteristics.
    Handles any number of states (4, 5, or 6) cleanly.

    Core framework:
    - extreme_vol:        highest vol state (always 1)
    - extreme_vol_2:      second highest vol if n_states >= 6
    - ranging_low_vol:    low abs return, low vol
    - trending_low_vol:   high abs return, low vol
    - ranging_high_vol:   low abs return, high vol
    - trending_high_vol:  high abs return, high vol
    """
    n = len(stats)

    # sort all states by vol ascending
    by_vol = sorted(stats, key=lambda s: stats[s]["mean_vol"])

    if n == 4:
        # simple 2x2: split by vol then by abs return within each group
        low_vol_list  = sorted(by_vol[:2], key=lambda s: abs(stats[s]["mean_return"]))
        high_vol_list = sorted(by_vol[2:], key=lambda s: abs(stats[s]["mean_return"]))
        return {
            low_vol_list[0]:  "ranging_low_vol",
            low_vol_list[1]:  "trending_low_vol",
            high_vol_list[0]: "ranging_high_vol",
            high_vol_list[1]: "trending_high_vol",
        }

    elif n == 5:
        # 1 extreme + 2x2
        extreme_state = by_vol[-1]
        remaining     = by_vol[:-1]
        low_vol_list  = sorted(remaining[:2], key=lambda s: abs(stats[s]["mean_return"]))
        high_vol_list = sorted(remaining[2:], key=lambda s: abs(stats[s]["mean_return"]))
        return {
            extreme_state:    "extreme_vol",
            low_vol_list[0]:  "ranging_low_vol",
            low_vol_list[1]:  "trending_low_vol",
            high_vol_list[0]: "ranging_high_vol",
            high_vol_list[1]: "trending_high_vol",
        }

    elif n == 6:
        # 2 extreme + 2x2
        # top 2 by vol = extreme states
        extreme_1     = by_vol[-1]   # highest vol
        extreme_2     = by_vol[-2]   # second highest vol
        remaining     = by_vol[:-2]  # 4 remaining states
        low_vol_list  = sorted(remaining[:2], key=lambda s: abs(stats[s]["mean_return"]))
        high_vol_list = sorted(remaining[2:], key=lambda s: abs(stats[s]["mean_return"]))
        return {
            extreme_1:        "extreme_vol",
            extreme_2:        "extreme_vol_2",
            low_vol_list[0]:  "ranging_low_vol",
            low_vol_list[1]:  "trending_low_vol",
            high_vol_list[0]: "ranging_high_vol",
            high_vol_list[1]: "trending_high_vol",
        }

    else:
        # fallback for any other n: treat top (n-4) states as extreme vol tiers
        n_extreme     = n - 4
        extreme_states = by_vol[-n_extreme:]
        remaining      = by_vol[:-n_extreme]
        low_vol_list   = sorted(remaining[:2], key=lambda s: abs(stats[s]["mean_return"]))
        high_vol_list  = sorted(remaining[2:], key=lambda s: abs(stats[s]["mean_return"]))

        mapping = {
            low_vol_list[0]:  "ranging_low_vol",
            low_vol_list[1]:  "trending_low_vol",
            high_vol_list[0]: "ranging_high_vol",
            high_vol_list[1]: "trending_high_vol",
        }
        for i, s in enumerate(reversed(extreme_states)):
            label = "extreme_vol" if i == 0 else f"extreme_vol_{i+1}"
            mapping[s] = label
        return mapping


def label_states(
    model: GaussianHMM,
    scaler: StandardScaler,
    X: np.ndarray,
    df: pd.DataFrame,
    asset: str = "NQ",
    n_states: int = N_STATES,
) -> dict:
    states      = decode(model, scaler, X)
    df          = df.copy()
    df["state"] = states

    stats = {}
    for s in range(n_states):
        sub = df[df["state"] == s]
        stats[s] = {
            "mean_return": float(sub["returns"].mean()),
            "mean_vol":    float(sub["vol_5"].mean()),
            "count":       len(sub),
        }

    anchors = load_anchors(asset)
    if not anchors:
        mapping      = _label_fresh(stats)
        anchor_stats = {mapping[s]: stats[s] for s in range(n_states)}
        save_anchors(anchor_stats, asset)
        return mapping

    return match_states_to_anchors(stats, anchors)
