"""Real-time scoring API (FastAPI).

POST /score      score one authorisation request (synchronous decision, like a card switch)
POST /feedback   delayed chargeback labels -> updates merchant-risk state + DB outcome
GET  /health     liveness + model version
GET  /metrics    rolling latency percentiles, decision mix, sink stats
GET  /model      production model card (metadata.json)
GET  /drift      latest online PSI per monitored feature

Run:  uvicorn frauddet.serving.api:app --port 8010      (single worker: feature state is in-process)
Env:  FRAUDDET_SINK=off   disable the PostgreSQL sink (pure latency / load testing)
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..config import get_paths, load_config
from ..monitoring import ONLINE_EXCLUDE, psi_from_profile, psi_level
from .scorer import Scorer

CFG = load_config()


class Transaction(BaseModel):
    tx_id: int
    ts: datetime = Field(description="event timestamp (ISO-8601)")
    customer_id: int
    terminal_id: int
    device_id: int = -1
    channel: str = Field(pattern="^(POS|ONLINE|ATM)$")
    category: str
    amount: float = Field(gt=0)
    lat: float
    lon: float
    city: str | None = None


class Feedback(BaseModel):
    tx_id: int
    terminal_id: int
    ts: datetime
    is_fraud: int
    scenario: str | None = None


class State:
    scorer: Scorer
    sink = None
    lat = deque(maxlen=5000)
    decisions = Counter()
    buf_x: deque
    buf_s: deque
    n = 0
    lock = threading.Lock()
    drift_latest: dict = {}


S = State()


def _drift_job(xs: np.ndarray, scores: np.ndarray):
    prof = json.loads((get_paths().production / "drift_profile.json").read_text())
    now = datetime.now(timezone.utc)
    rows = []
    for j, f in enumerate(S.scorer.features):
        if f in ONLINE_EXCLUDE:
            continue
        v = psi_from_profile(prof[f], xs[:, j])
        rows.append((now, len(xs), f, v, psi_level(v)))
    v = psi_from_profile(prof["__score__"], scores)  # raw (uncalibrated, unrounded) model score
    rows.append((now, len(xs), "model_score", v, psi_level(v)))
    S.drift_latest = {r[2]: {"psi": round(r[3], 4), "level": r[4]} for r in rows}
    if S.sink:
        S.sink.put_drift(rows)


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.scorer = Scorer()
    n = CFG["serving"]["drift_every_n"]
    S.buf_x, S.buf_s = deque(maxlen=n), deque(maxlen=n)
    if os.environ.get("FRAUDDET_SINK", "on") != "off":
        try:
            from ..db import Sink
            S.sink = Sink(CFG["db_url"], CFG["serving"]["sink_flush_ms"])
            S.sink.register_model(S.scorer.meta)
            S.sink.start()
            print("[api] PostgreSQL sink connected")
        except Exception as e:
            S.sink = None
            print(f"[api] DB unavailable, running without sink: {e}")
    print(f"[api] model {S.scorer.version} ready ({S.scorer.runtime}, {len(S.scorer.features)} features)")
    yield
    if S.sink:
        S.sink.stop()


app = FastAPI(title="Fraud Detection Scoring API", version="1.0.0", lifespan=lifespan)


@app.post("/score")
def score(tx: Transaction):
    """Sync endpoint (FastAPI threadpool). E11 showed it is marginally faster than the async variant."""
    return _score(tx)


@app.post("/score/async")
async def score_async(tx: Transaction):
    """Same logic on the event loop (no threadpool hop) - kept for the E11 comparison."""
    return _score(tx)


def _score(tx: Transaction):
    d = tx.model_dump()
    r = S.scorer.score(d)
    x = r.pop("_x")
    lat = r["latency_ms"]
    with S.lock:
        S.lat.append(lat["total"])
        S.decisions[r["decision"]] += 1
        S.buf_x.append(x[0])
        S.buf_s.append(r.pop("_raw"))
        S.n += 1
        run_drift = S.n % S.buf_x.maxlen == 0
    if run_drift:
        threading.Thread(target=_drift_job, args=(np.array(S.buf_x), np.array(S.buf_s)), daemon=True).start()
    if S.sink:
        rs = [f"{x_['label']} ({x_['value']})" for x_ in r["reasons"]] + [None] * 3
        S.sink.put_tx((d["tx_id"], d["ts"].replace(tzinfo=None), datetime.now(timezone.utc), d["customer_id"],
                       d["terminal_id"], d["device_id"], d["channel"], d["category"], d.get("city"), d["lat"],
                       d["lon"], d["amount"], r["fraud_score"], r["decision"], r["model_version"], rs[0], rs[1],
                       rs[2], lat["total"], lat["features"], lat["model"], lat["explain"]))
    r["latency_ms"] = {k: round(v, 4) for k, v in lat.items()}
    return r


@app.post("/feedback")
def feedback(items: list[Feedback]):
    rows = []
    for it in items:
        S.scorer.feedback(it.terminal_id, S.scorer.engine.to_seconds(it.ts), it.is_fraud)
        rows.append((it.tx_id, it.is_fraud, it.scenario))
    if S.sink:
        S.sink.put_labels(rows)
    return {"received": len(rows)}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": S.scorer.version, "runtime": S.scorer.runtime,
            "sink": S.sink is not None, "scored": S.n}


@app.get("/metrics")
def metrics():
    lat = np.array(S.lat) if S.lat else np.zeros(1)
    total = max(1, sum(S.decisions.values()))
    return {"scored": S.n, "latency_ms": {q: round(float(np.percentile(lat, p)), 4)
                                          for q, p in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))},
            "decision_mix": {k: v / total for k, v in S.decisions.items()},
            "sink": S.sink.stats if S.sink else None}


@app.post("/metrics/reset")
def metrics_reset():
    S.lat.clear()
    return {"ok": True}


@app.get("/model")
def model():
    m = dict(S.scorer.meta)
    m.pop("calibration", None)
    return m


@app.get("/drift")
def drift():
    return S.drift_latest


def main():
    import uvicorn
    s = CFG["serving"]
    uvicorn.run(app, host=os.environ.get("FRAUDDET_HOST", s["host"]), port=int(os.environ.get("FRAUDDET_PORT", s["port"])),
                log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
