"""Drift monitoring with the Population Stability Index (PSI).

PSI < 0.10 stable · 0.10–0.25 moderate shift (investigate) · > 0.25 significant shift (retrain)
"""
from __future__ import annotations

import numpy as np

EPS = 1e-4
# Calendar features are cyclic: a short online window (e.g. 2,000 tx = a few hours) can never match the
# training distribution of hour/day-of-week, so monitoring them produces permanent false alarms.
ONLINE_EXCLUDE = {"hour", "dow", "is_weekend", "is_night"}


def _edges(ref: np.ndarray, bins: int) -> np.ndarray:
    ref = ref[np.isfinite(ref)]
    uniq = np.unique(ref)
    if len(uniq) <= bins:  # discrete feature -> one bin per value
        mids = (uniq[:-1] + uniq[1:]) / 2
        return np.r_[-np.inf, mids, np.inf]
    e = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.r_[-np.inf, e, np.inf]


def _dist(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    c = np.histogram(x[np.isfinite(x)], bins=edges)[0].astype(float)
    return np.clip(c / max(1.0, c.sum()), EPS, None)


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    e = _edges(np.asarray(ref, float), bins)
    p, q = _dist(np.asarray(ref, float), e), _dist(np.asarray(cur, float), e)
    return float(np.sum((q - p) * np.log(q / p)))


def build_profile(X: np.ndarray, names: list[str], bins: int = 10) -> dict:
    """Reference distribution stored with the model; online PSI only needs these numbers."""
    prof = {}
    for j, n in enumerate(names):
        e = _edges(X[:, j].astype(float), bins)
        prof[n] = {"edges": [float(v) for v in e[1:-1]], "p": _dist(X[:, j].astype(float), e).tolist()}
    return prof


def psi_from_profile(entry: dict, cur: np.ndarray) -> float:
    e = np.r_[-np.inf, entry["edges"], np.inf]
    p, q = np.asarray(entry["p"]), _dist(np.asarray(cur, float), e)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_level(v: float) -> str:
    return "stable" if v < 0.10 else ("moderate" if v < 0.25 else "significant")
