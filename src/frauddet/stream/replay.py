"""Stream the held-out test period into the scoring API at a fixed rate, with delayed labels.

    python -m frauddet.stream.replay --tps 40 --label-delay 20

* transactions are sent in event-time order (the feature state depends on order)
* every transaction's ground truth is sent back to /feedback after `--label-delay` seconds,
  emulating chargebacks / analyst confirmations arriving later than the decision
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import httpx
import pandas as pd

from ..config import get_paths, load_config


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=f"http://{cfg['serving']['host']}:{cfg['serving']['port']}")
    ap.add_argument("--tps", type=float, default=40.0, help="transactions per second")
    ap.add_argument("--label-delay", type=float, default=20.0, help="seconds until the label is fed back")
    ap.add_argument("--limit", type=int, default=0, help="stop after N transactions (0 = whole test period)")
    ap.add_argument("--start-day", type=int, default=cfg["split"]["test"][0])
    a = ap.parse_args()

    raw = pd.read_parquet(get_paths(cfg).raw / "transactions.parquet")
    df = raw[raw.day >= a.start_day].sort_values("ts_s")
    if a.limit:
        df = df.head(a.limit)
    print(f"[replay] {len(df):,} tx from day {a.start_day} at {a.tps} TPS -> {a.api} "
          f"(~{len(df) / a.tps / 60:.0f} min), label delay {a.label_delay}s")

    pending: deque = deque()
    client = httpx.Client(base_url=a.api, timeout=10.0)
    period = 1.0 / a.tps
    t_next = time.perf_counter()
    sent = alerts = 0
    t_report = time.perf_counter()
    lat = []

    def flush_labels(force=False):
        now = time.perf_counter()
        batch = []
        while pending and (force or pending[0][0] <= now):
            batch.append(pending.popleft()[1])
        if batch:
            client.post("/feedback", json=batch)

    try:
        for r in df.itertuples(index=False):
            payload = {"tx_id": int(r.tx_id), "ts": r.ts.strftime("%Y-%m-%dT%H:%M:%S.%f"), "customer_id": int(r.customer_id),
                       "terminal_id": int(r.terminal_id), "device_id": int(r.device_id), "channel": r.channel,
                       "category": r.category, "amount": float(r.amount), "lat": float(r.tx_lat),
                       "lon": float(r.tx_lon), "city": r.tx_city}
            t0 = time.perf_counter()
            res = client.post("/score", json=payload).json()
            lat.append((time.perf_counter() - t0) * 1e3)
            sent += 1
            alerts += res["decision"] != "APPROVE"
            pending.append((time.perf_counter() + a.label_delay,
                            {"tx_id": int(r.tx_id), "terminal_id": int(r.terminal_id), "ts": r.ts.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                             "is_fraud": int(r.is_fraud), "scenario": r.scenario if r.is_fraud else None}))
            flush_labels()
            if time.perf_counter() - t_report > 10:
                s = pd.Series(lat)
                print(f"[replay] sent={sent:,} alerts={alerts} ({alerts / sent:.2%}) | client RTT "
                      f"p50={s.quantile(.5):.2f} p99={s.quantile(.99):.2f} ms | event time {r.ts:%Y-%m-%d %H:%M}")
                lat.clear()
                t_report = time.perf_counter()
            t_next += period
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()  # fell behind: don't burst
    except KeyboardInterrupt:
        print("[replay] interrupted")
    finally:
        flush_labels(force=True)
        print(f"[replay] done: {sent:,} sent, {alerts} alerts")


if __name__ == "__main__":
    main()
