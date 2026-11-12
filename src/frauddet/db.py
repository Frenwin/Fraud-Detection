"""PostgreSQL access: schema bootstrap, async batch sink (off the latency path), research loader."""
from __future__ import annotations

import argparse
import queue
import threading
import time
from datetime import datetime, timezone

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from .config import ROOT, get_paths, load_config

TX_COLS = ["tx_id", "event_ts", "scored_at", "customer_id", "terminal_id", "device_id", "channel", "category",
           "city", "lat", "lon", "amount", "fraud_score", "decision", "model_version", "reason_1", "reason_2",
           "reason_3", "latency_ms", "feature_ms", "model_ms", "explain_ms"]


def connect(db_url: str | None = None):
    u = make_url(db_url or load_config()["db_url"])
    return psycopg2.connect(host=u.host, port=u.port or 5432, user=u.username, password=u.password,
                            dbname=u.database, connect_timeout=3)


def bootstrap(conn) -> None:
    with conn.cursor() as cur:
        cur.execute((ROOT / "sql" / "schema.sql").read_text(encoding="utf-8"))
        cur.execute((ROOT / "sql" / "views.sql").read_text(encoding="utf-8"))
    conn.commit()


def reset(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE scored_transactions, drift_metrics")
    conn.commit()


class Sink(threading.Thread):
    """Collects scored rows / labels / drift rows and flushes them in batches every `flush_ms`."""

    def __init__(self, db_url: str, flush_ms: int = 500):
        super().__init__(daemon=True, name="db-sink")
        self.conn = connect(db_url)
        bootstrap(self.conn)
        self.q: queue.Queue = queue.Queue()
        self.flush_s = flush_ms / 1000
        self.stats = {"rows": 0, "labels": 0, "flushes": 0, "errors": 0, "last_flush_ms": 0.0}
        self._stop = threading.Event()

    def put_tx(self, row: tuple):
        self.q.put(("tx", row))

    def put_labels(self, rows: list[tuple]):
        self.q.put(("label", rows))

    def put_drift(self, rows: list[tuple]):
        self.q.put(("drift", rows))

    def register_model(self, meta: dict):
        with self.conn.cursor() as cur:
            cur.execute("""INSERT INTO model_registry VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (model_version) DO UPDATE SET deployed_at = EXCLUDED.deployed_at""",
                        (meta["model_version"], datetime.now(timezone.utc), meta["variant"], meta["runtime"],
                         len(meta["features"]), meta["n_trees"], meta["policy"]["t_review"], meta["policy"]["t_block"],
                         meta["valid_auprc"], meta["test"]["auprc"], meta["latency_model_ms"]["p99_ms"],
                         meta.get("latency_e2e_ms", {}).get("p99")))
        self.conn.commit()

    def run(self):
        while not self._stop.is_set():
            time.sleep(self.flush_s)
            self.flush()

    def flush(self):
        tx, labels, drift = [], [], []
        while True:
            try:
                kind, item = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "tx":
                tx.append(item)
            elif kind == "label":
                labels.extend(item)
            else:
                drift.extend(item)
        if not (tx or labels or drift):
            return
        t0 = time.perf_counter()
        try:
            with self.conn.cursor() as cur:
                if tx:
                    execute_values(cur, f"INSERT INTO scored_transactions ({','.join(TX_COLS)}) VALUES %s "
                                        "ON CONFLICT (tx_id) DO NOTHING", tx, page_size=1000)
                if labels:
                    execute_values(cur, """UPDATE scored_transactions t SET is_fraud = v.is_fraud,
                                           scenario = v.scenario, labeled_at = now()
                                           FROM (VALUES %s) AS v(tx_id, is_fraud, scenario)
                                           WHERE t.tx_id = v.tx_id""", labels, page_size=1000)
                if drift:
                    execute_values(cur, "INSERT INTO drift_metrics (measured_at, window_n, feature, psi, level) "
                                        "VALUES %s", drift)
            self.conn.commit()
            self.stats["rows"] += len(tx)
            self.stats["labels"] += len(labels)
            self.stats["flushes"] += 1
        except Exception as e:  # never crash the scorer because of the DB
            self.conn.rollback()
            self.stats["errors"] += 1
            print(f"[sink] flush failed: {e}")
        self.stats["last_flush_ms"] = (time.perf_counter() - t0) * 1e3

    def stop(self):
        self._stop.set()
        self.flush()


def load_research(db_url: str | None = None) -> None:
    """Push research CSVs into r_* tables so the BI dashboard has a 'Research' page."""
    cfg = load_config()
    eng = create_engine(db_url or cfg["db_url"])
    res = get_paths(cfg).results
    with eng.begin() as c:  # plot views depend on r_* tables -> drop before replacing them
        c.exec_driver_sql("DROP VIEW IF EXISTS v_e9_plot, v_e11_plot")
    for f in sorted(res.glob("*.csv")):
        name = "r_" + f.stem
        pd.read_csv(f).to_sql(name, eng, if_exists="replace", index=False)
        print(f"[db] loaded {f.name} -> {name}")
    with eng.begin() as c:  # BI-friendly shapes: log axis precomputed, ordinal concurrency labels
        c.exec_driver_sql("""CREATE VIEW v_e9_plot AS SELECT variant, family, runtime, valid_auprc, test_auprc, p99_ms,
                             log10(p99_ms) AS log10_p99_ms, size_kb, pareto FROM r_e9_variants""")
        c.exec_driver_sql("""CREATE VIEW v_e11_plot AS SELECT endpoint, 'c=' || lpad(concurrency::text, 2, '0') AS clients,
                             throughput_rps, p99_ms, server_p99_ms FROM r_e11_loadtest""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "reset", "load-research"])
    a = ap.parse_args()
    if a.cmd == "load-research":
        return load_research()
    conn = connect()
    bootstrap(conn)
    if a.cmd == "reset":
        reset(conn)
    print(f"[db] {a.cmd} ok")


if __name__ == "__main__":
    main()
