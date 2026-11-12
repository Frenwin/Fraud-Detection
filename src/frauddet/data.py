"""Build the research dataset: simulate -> replay feature engine -> parquet + online state snapshot."""
from __future__ import annotations

import time

import pandas as pd

from .config import get_paths, load_config
from .features import FEATURES, FeatureEngine
from .simulator import simulate

META = ["tx_id", "ts", "ts_s", "day", "customer_id", "terminal_id", "device_id", "channel", "category",
        "amount", "tx_lat", "tx_lon", "tx_city", "is_fraud", "scenario"]


def build(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    t0 = time.perf_counter()
    tx, customers, terminals = simulate(cfg)
    tx.to_parquet(paths.raw / "transactions.parquet", index=False)
    customers.to_parquet(paths.raw / "customers.parquet", index=False)
    terminals.to_parquet(paths.raw / "terminals.parquet", index=False)
    print(f"[data] simulated {len(tx):,} tx | fraud rate {tx.is_fraud.mean():.3%} | {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    engine = FeatureEngine(customers, cfg["simulation"]["start_date"], cfg["label_delay_days"])
    test_start_s = cfg["split"]["test"][0] * 86400.0
    X = engine.process_frame(tx, use_labels=True, snapshot_at_s=test_start_s,
                             snapshot_path=paths.artifacts / "feature_state.pkl")
    dt = time.perf_counter() - t0
    print(f"[data] features: {len(tx):,} tx in {dt:.1f}s ({dt / len(tx) * 1e6:.1f} us/tx) "
          f"| state snapshot saved at day {cfg['split']['test'][0]}")
    df = pd.concat([tx[META].reset_index(drop=True), pd.DataFrame(X, columns=FEATURES)], axis=1)
    df.to_parquet(paths.processed / "features.parquet", index=False)
    return df


def load_features(cfg: dict | None = None) -> pd.DataFrame:
    return pd.read_parquet(get_paths(cfg).processed / "features.parquet")


def split(df: pd.DataFrame, cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    s = cfg["split"]
    return {k: df[df.day.between(*s[k])].reset_index(drop=True) for k in ("train", "valid", "test")}


if __name__ == "__main__":
    build()
