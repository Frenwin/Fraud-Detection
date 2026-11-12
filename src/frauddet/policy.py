"""Decision policy: probability calibration + cost-optimal 3-tier thresholds.

APPROVE  : p <  t_review
REVIEW   : t_review <= p < t_block   -> queued for a fraud analyst
BLOCK    : p >= t_block              -> auto-decline (only where precision >= block_min_precision)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .metrics import business_value, threshold_sweep


def fit_calibrator(p_raw: np.ndarray, y: np.ndarray) -> dict:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw, y)
    # stored as two arrays -> calibration at serving time is a single np.interp (sub-µs)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


TIE_BREAK = 1e-6


def calibrate(p_raw, cal: dict) -> np.ndarray:
    """Isotonic is a step function -> it creates ties and destroys ranking resolution (lower AUPRC).
    A 1e-6 blend with the raw score keeps the mapping *strictly* monotone: probabilities stay calibrated
    (error <= 1e-6) and the ranking is exactly the model's ranking."""
    p_raw = np.asarray(p_raw, dtype=float)
    return (1 - TIE_BREAK) * np.interp(p_raw, cal["x"], cal["y"]) + TIE_BREAK * p_raw


def fit_policy(p_cal: np.ndarray, df: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    y, amt = df["is_fraud"].to_numpy(), df["amount"].to_numpy()
    rc, fc = cfg["cost"]["review_cost"], cfg["cost"]["friction_cost"]
    sweep = threshold_sweep(y, p_cal, amt, rc, fc)
    best = sweep.loc[sweep.net_savings.idxmax()]
    t_review = float(best.threshold)
    ok = sweep[(sweep.precision >= cfg["alerting"]["block_min_precision"]) & (sweep.threshold >= t_review)
               & (sweep.tp >= 20)]
    t_block = float(ok.threshold.min()) if len(ok) else float(max(t_review, sweep.threshold.max()))
    policy = {"t_review": t_review, "t_block": t_block,
              "valid_at_review": business_value(y, p_cal, amt, t_review, rc, fc),
              "valid_at_block": business_value(y, p_cal, amt, t_block, rc, fc)}
    return policy, sweep


def decide(p: float, policy: dict) -> str:
    if p >= policy["t_block"]:
        return "BLOCK"
    if p >= policy["t_review"]:
        return "REVIEW"
    return "APPROVE"
