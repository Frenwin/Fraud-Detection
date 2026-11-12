"""Fraud-specific evaluation: ranking quality, alert-budget metrics and business cost (IDR)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def recall_at_fpr(y, p, fpr_target=0.005) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def precision_recall_at_k_daily(df: pd.DataFrame, p: np.ndarray, k: int) -> tuple[float, float]:
    """Analysts can only review k alerts/day: precision & recall of the daily top-k."""
    d = pd.DataFrame({"day": df["day"].to_numpy(), "y": df["is_fraud"].to_numpy(), "p": p})
    d["rank"] = d.groupby("day")["p"].rank(ascending=False, method="first")
    top = d[d["rank"] <= k]
    prec = top.groupby("day")["y"].mean().mean()
    rec = top["y"].sum() / max(1, d["y"].sum())
    return float(prec), float(rec)


def business_value(y, p, amount, threshold, review_cost, friction_cost) -> dict:
    """Net value of alerting at `threshold` vs. doing nothing (all fraud becomes loss)."""
    y, amount = np.asarray(y), np.asarray(amount)
    alert = np.asarray(p) >= threshold
    tp, fp = alert & (y == 1), alert & (y == 0)
    fraud_total = amount[y == 1].sum()
    prevented = amount[tp].sum()
    cost = review_cost * alert.sum() + friction_cost * fp.sum()
    return {
        "threshold": float(threshold), "alerts": int(alert.sum()), "tp": int(tp.sum()), "fp": int(fp.sum()),
        "precision": float(tp.sum() / max(1, alert.sum())), "recall": float(tp.sum() / max(1, (y == 1).sum())),
        "fraud_amount_total": float(fraud_total), "fraud_amount_prevented": float(prevented),
        "loss_prevented_pct": float(prevented / max(1.0, fraud_total)),
        "ops_cost": float(cost), "net_savings": float(prevented - cost),
    }


def threshold_sweep(y, p, amount, review_cost, friction_cost, n=400) -> pd.DataFrame:
    qs = np.unique(np.quantile(p, np.linspace(0.90, 0.99995, n)))
    return pd.DataFrame([business_value(y, p, amount, t, review_cost, friction_cost) for t in qs])


def evaluate(df: pd.DataFrame, p: np.ndarray, cfg: dict, threshold: float | None = None) -> dict:
    y = df["is_fraud"].to_numpy()
    k = cfg["alerting"]["daily_alert_budget"]
    pk, rk = precision_recall_at_k_daily(df, p, k)
    out = {
        "auprc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "recall@0.5%fpr": recall_at_fpr(y, p, 0.005),
        f"precision@{k}/day": pk, f"recall@{k}/day": rk,
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }
    if threshold is not None:
        bv = business_value(y, p, df["amount"].to_numpy(), threshold,
                            cfg["cost"]["review_cost"], cfg["cost"]["friction_cost"])
        out.update({f"biz_{k_}": v for k_, v in bv.items()})
    return out
