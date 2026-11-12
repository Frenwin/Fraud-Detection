"""Integration tests — require `make data research optimize` artifacts (skipped otherwise)."""
import os

import numpy as np
import pandas as pd
import pytest

from frauddet.config import get_paths, load_config
from frauddet.features import FEATURES, FeatureEngine

P = get_paths()
HAS_DATA = (P.artifacts / "feature_state.pkl").exists() and (P.processed / "features.parquet").exists()
HAS_BUNDLE = (P.production / "metadata.json").exists()


@pytest.mark.skipif(not HAS_DATA, reason="run `python -m frauddet.data` first")
def test_training_serving_parity():
    """Online engine (restored snapshot) must reproduce the offline training features exactly."""
    cfg = load_config()
    raw = pd.read_parquet(P.raw / "transactions.parquet")
    stream = raw[raw.day >= cfg["split"]["test"][0]].head(1500)
    off = pd.read_parquet(P.processed / "features.parquet").set_index("tx_id").loc[stream.tx_id, FEATURES]
    eng = FeatureEngine.load(P.artifacts / "feature_state.pkl")
    on = np.array([eng.transform(r.ts_s, r.customer_id, r.terminal_id, r.channel, r.category, r.amount,
                                 r.tx_lat, r.tx_lon, r.device_id) for r in stream.itertuples()], dtype=np.float32)
    np.testing.assert_allclose(on, off.to_numpy(np.float32), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not HAS_BUNDLE, reason="run `python -m frauddet.optimize` first")
def test_api_score_and_feedback():
    os.environ["FRAUDDET_SINK"] = "off"
    from fastapi.testclient import TestClient

    from frauddet.serving.api import app
    tx = {"tx_id": 10**9, "ts": "2026-08-06T10:00:00", "customer_id": 1, "terminal_id": 5, "device_id": 9_999_999,
          "channel": "ONLINE", "category": "ewallet_topup", "amount": 9_000_000, "lat": 1.35, "lon": 103.8}
    with TestClient(app) as c:
        r = c.post("/score", json=tx)
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["fraud_score"] <= 1 and body["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
        assert body["latency_ms"]["total"] < 100
        if body["decision"] != "APPROVE":
            assert len(body["reasons"]) >= 1
        assert c.post("/feedback", json=[{"tx_id": tx["tx_id"], "terminal_id": 5, "ts": tx["ts"],
                                          "is_fraud": 1}]).json() == {"received": 1}
        assert c.get("/health").json()["status"] == "ok"
        assert c.post("/score", json={**tx, "channel": "FAX"}).status_code == 422
