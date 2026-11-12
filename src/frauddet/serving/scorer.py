"""Production scoring path: online features -> model -> calibration -> 3-tier decision -> reason codes.

The latency benchmark (frauddet.optimize) measures *this* class, so reported numbers are the
numbers the API actually pays.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ..config import get_paths
from ..features import FEATURE_LABELS, FEATURES, FeatureEngine
from ..simulator import CATEGORIES
from ..policy import calibrate as _calibrate, decide


_SWAP = str.maketrans({",": ".", ".": ","})


def _id(s: str) -> str:  # Indonesian number format: 1.234.567,8
    return s.translate(_SWAP)


def _fmt(feature: str, v: float) -> str:
    return _id(_fmt_raw(feature, v))


def _fmt_raw(feature: str, v: float) -> str:
    if feature == "amount_log":
        amt = math.expm1(v)  # features are float32 -> round back to the data's Rp 1.000 granularity
        return f"Rp {round(amt, -3) if amt >= 10_000 else amt:,.0f}"
    if feature in ("dist_home_km", "dist_prev_km"):
        return f"{v:,.0f} km"
    if feature == "speed_prev_kmh":
        return f"{v:,.0f} km/jam"
    if feature == "c_amt_ratio_30d":
        return f"{v:.1f}x rata-rata" if v >= 0 else "tanpa riwayat"
    if feature == "category_code":
        return CATEGORIES[int(round(v))]
    if feature == "channel_code":
        return ("POS", "ONLINE", "ATM")[int(round(v))]
    if feature == "hour":
        return f"jam {int(v):02d}"
    if feature == "t_risk_delayed":
        return f"{v:.0%}"
    if feature in ("d_new_for_customer", "c_new_terminal", "c_new_city", "c_new_category", "is_night"):
        return "ya" if v >= 0.5 else "tidak"
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.2f}"


class Scorer:
    def __init__(self, bundle_dir: Path | None = None, engine: FeatureEngine | None = None,
                 explain: str = "on_alert"):
        paths = get_paths()
        self.bundle = Path(bundle_dir or paths.production)
        self.meta = json.loads((self.bundle / "metadata.json").read_text())
        self.features = self.meta["features"]
        self.idx = np.array([FEATURES.index(f) for f in self.features])
        self.version = self.meta["model_version"]
        self.policy = self.meta["policy"]
        self.booster = lgb.Booster(model_file=str(self.bundle / "model.txt"))
        self.runtime = self.meta["runtime"]
        if self.runtime == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(str(self.bundle / "model.onnx"), so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            self.out = self.meta.get("onnx_prob_output", self.sess.get_outputs()[1].name)
        self.explain_mode = explain  # "on_alert" | "always" | "never"
        self.engine = engine or FeatureEngine.load(paths.artifacts / "feature_state.pkl")
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ primitives
    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        if self.runtime == "onnx":
            return self.sess.run([self.out], {self.inp: x})[0][:, 1]
        return self.booster.predict(x, num_threads=1)

    def calibrate(self, p):
        return _calibrate(p, self.meta["calibration"])

    def explain(self, x: np.ndarray, k: int = 3) -> list[dict]:
        contrib = self.booster.predict(x, pred_contrib=True, num_threads=1)[0, :-1]
        top = np.argsort(-contrib)[:k]
        return [{"feature": self.features[j], "label": FEATURE_LABELS[self.features[j]],
                 "value": _fmt(self.features[j], float(x[0, j])), "contribution": round(float(contrib[j]), 3)}
                for j in top if contrib[j] > 0]

    # ------------------------------------------------------------------ public
    def score(self, tx: dict) -> dict:
        t0 = time.perf_counter()
        ts_s = tx["ts_s"] if "ts_s" in tx else self.engine.to_seconds(tx["ts"])
        with self.lock:  # feature state is shared & ordered
            full = self.engine.transform(ts_s, int(tx["customer_id"]), int(tx["terminal_id"]), tx["channel"],
                                         tx["category"], float(tx["amount"]), float(tx["lat"]), float(tx["lon"]),
                                         int(tx.get("device_id", -1)))
        x = np.asarray(full, dtype=np.float32)[self.idx].reshape(1, -1)
        t1 = time.perf_counter()
        raw = float(self.predict_raw(x)[0])
        p = float(self.calibrate(raw))
        decision = decide(p, self.policy)
        t2 = time.perf_counter()
        reasons = []
        if self.explain_mode == "always" or (self.explain_mode == "on_alert" and decision != "APPROVE"):
            reasons = self.explain(x)
        t3 = time.perf_counter()
        return {"tx_id": int(tx["tx_id"]), "fraud_score": round(p, 6), "raw_score": round(raw, 6),
                "decision": decision, "reasons": reasons, "model_version": self.version,
                "latency_ms": {"features": (t1 - t0) * 1e3, "model": (t2 - t1) * 1e3,
                               "explain": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3},
                "_x": x, "_raw": raw}

    def feedback(self, terminal_id: int, ts_s: float, is_fraud: int) -> None:
        if is_fraud:
            with self.lock:
                self.engine.add_label(int(terminal_id), float(ts_s))
