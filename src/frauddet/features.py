"""Stateful, point-in-time-correct feature engine — single source of truth.

The *same* `FeatureEngine.transform()` is used:
  * offline  -> replaying history to build the training matrix
  * online   -> inside the FastAPI scorer for every incoming transaction
so training/serving skew is impossible by construction (verified in tests/test_parity.py).

Every feature only looks at events strictly *before* the current transaction.
Merchant risk uses chargeback labels with a `label_delay` (default 7 days): a fraud
label for a transaction at time t only becomes usable at t + delay, exactly like in
production where chargebacks arrive weeks later.

Complexity: O(log n) per window via prefix sums + bisect, ~10-20 µs per transaction in
pure Python, so ~900k transactions are replayed in well under a minute.
"""
from __future__ import annotations

import math
import pickle
from bisect import bisect_left, insort
from pathlib import Path

import numpy as np
import pandas as pd

from .simulator import CATEGORIES

CHANNEL_CODE = {"POS": 0, "ONLINE": 1, "ATM": 2}
CATEGORY_CODE = {c: i for i, c in enumerate(CATEGORIES)}
H, D = 3600.0, 86400.0
SMALL_AMOUNT = 20_000.0

FEATURES = [
    # transaction
    "amount_log", "hour", "dow", "is_night", "is_weekend", "channel_code", "category_code",
    # customer behaviour (velocity / spending profile)
    "c_cnt_1h", "c_cnt_24h", "c_cnt_7d", "c_sum_24h_log", "c_avg_amt_30d_log", "c_amt_ratio_30d",
    "c_amt_z_30d", "c_secs_since_last_log", "c_small_cnt_1h",
    "c_new_terminal", "c_new_city", "c_new_category",
    # geo
    "dist_home_km", "dist_prev_km", "speed_prev_kmh",
    # merchant / terminal (delayed-label risk)
    "t_cnt_24h", "t_cnt_7d", "t_risk_delayed", "t_n_delayed",
    # device graph
    "d_new_for_customer", "d_n_customers_30d", "d_cnt_1h",
]
FEATURE_GROUPS = {
    "transaction": FEATURES[0:7],
    "customer_velocity": FEATURES[7:19],
    "geo": FEATURES[19:22],
    "terminal_risk": FEATURES[22:26],
    "device_graph": FEATURES[26:29],
}
# human-readable reason codes for the dashboard (Bahasa Indonesia)
FEATURE_LABELS = {
    "amount_log": "Nominal transaksi", "hour": "Jam transaksi", "dow": "Hari dalam minggu",
    "is_night": "Transaksi dini hari", "is_weekend": "Akhir pekan", "channel_code": "Kanal transaksi",
    "category_code": "Kategori merchant", "c_cnt_1h": "Frekuensi 1 jam terakhir",
    "c_cnt_24h": "Frekuensi 24 jam terakhir", "c_cnt_7d": "Frekuensi 7 hari terakhir",
    "c_sum_24h_log": "Total belanja 24 jam", "c_avg_amt_30d_log": "Rata-rata belanja 30 hari",
    "c_amt_ratio_30d": "Nominal jauh di atas kebiasaan", "c_amt_z_30d": "Nominal menyimpang (z-score)",
    "c_secs_since_last_log": "Jeda sejak transaksi terakhir", "c_small_cnt_1h": "Burst nominal kecil (card testing)",
    "c_new_terminal": "Merchant baru bagi nasabah", "c_new_city": "Kota baru bagi nasabah",
    "c_new_category": "Kategori baru bagi nasabah", "dist_home_km": "Jarak dari domisili",
    "dist_prev_km": "Jarak dari transaksi sebelumnya", "speed_prev_kmh": "Kecepatan perpindahan mustahil",
    "t_cnt_24h": "Keramaian merchant 24 jam", "t_cnt_7d": "Keramaian merchant 7 hari",
    "t_risk_delayed": "Riwayat chargeback merchant", "t_n_delayed": "Volume historis merchant",
    "d_new_for_customer": "Perangkat baru", "d_n_customers_30d": "Perangkat dipakai banyak kartu",
    "d_cnt_1h": "Aktivitas perangkat 1 jam",
}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 12742.0 * math.asin(min(1.0, math.sqrt(a)))


class _Customer:
    __slots__ = ("ts", "cum_amt", "cum_la", "cum_la2", "cum_small", "last_ts", "last_lat", "last_lon",
                 "terminals", "cities", "categories", "devices", "home_lat", "home_lon")

    def __init__(self, home_lat, home_lon):
        self.ts: list[float] = []
        self.cum_amt, self.cum_la, self.cum_la2, self.cum_small = [0.0], [0.0], [0.0], [0.0]
        self.last_ts = None
        self.last_lat = self.last_lon = 0.0
        self.terminals, self.cities, self.categories, self.devices = set(), set(), set(), set()
        self.home_lat, self.home_lon = home_lat, home_lon


class _Terminal:
    __slots__ = ("ts", "fraud_ts")

    def __init__(self):
        self.ts: list[float] = []
        self.fraud_ts: list[float] = []


class _Device:
    __slots__ = ("ts", "customers")

    def __init__(self):
        self.ts: list[float] = []
        self.customers: dict[int, float] = {}


class FeatureEngine:
    def __init__(self, customers: pd.DataFrame, start_date: str, label_delay_days: float = 7.0):
        self.start = pd.Timestamp(start_date)
        self.start_dow = self.start.dayofweek
        self.delay = label_delay_days * D
        self.risk_window = 7 * D
        self.profiles = {int(r.customer_id): (float(r.home_lat), float(r.home_lon))
                         for r in customers[["customer_id", "home_lat", "home_lon"]].itertuples()}
        self.cust: dict[int, _Customer] = {}
        self.term: dict[int, _Terminal] = {}
        self.dev: dict[int, _Device] = {}
        self.last_ts_s = -math.inf
        self.n_seen = 0

    # ------------------------------------------------------------------ helpers
    def to_seconds(self, ts) -> float:
        return (pd.Timestamp(ts) - self.start).total_seconds()

    def _customer(self, cid, lat, lon) -> _Customer:
        c = self.cust.get(cid)
        if c is None:
            home = self.profiles.get(cid, (lat, lon))  # unknown customer: first location = home
            c = self.cust[cid] = _Customer(*home)
        return c

    # ------------------------------------------------------------------ core
    def transform(self, ts_s: float, customer_id: int, terminal_id: int, channel: str, category: str,
                  amount: float, lat: float, lon: float, device_id: int) -> list[float]:
        """Compute features for one transaction (using only the past), then update state."""
        c = self._customer(customer_id, lat, lon)
        la = math.log1p(amount)
        # --- time
        day = int(ts_s // D)
        hour = int((ts_s % D) // H)
        dow = (self.start_dow + day) % 7
        # --- customer windows.  n = number of events strictly BEFORE ts_s: equals len(ts) for in-order
        # streams, and stays correct when concurrent requests / late events arrive out of order.
        ts = c.ts
        n = bisect_left(ts, ts_s)
        i1h, i24h = bisect_left(ts, ts_s - H), bisect_left(ts, ts_s - D)
        i7d, i30d = bisect_left(ts, ts_s - 7 * D), bisect_left(ts, ts_s - 30 * D)
        n30 = n - i30d
        if n30 > 0:
            mean_la = (c.cum_la[n] - c.cum_la[i30d]) / n30
            var = max(0.0, (c.cum_la2[n] - c.cum_la2[i30d]) / n30 - mean_la * mean_la)
            ratio = math.exp(la - mean_la)
            z = (la - mean_la) / (math.sqrt(var) + 0.25)
        else:
            mean_la, ratio, z = 0.0, -1.0, 0.0
        if n == 0:
            since, dprev, speed = 30 * D, 0.0, 0.0
        else:
            since = ts_s - ts[n - 1]
            dprev = haversine_km(c.last_lat, c.last_lon, lat, lon)
            speed = dprev / max(since / H, 1 / 60)
        # --- terminal
        t = self.term.get(terminal_id)
        if t is None:
            t = self.term[terminal_id] = _Terminal()
        tts = t.ts
        nt = bisect_left(tts, ts_s)
        lo, hi = ts_s - self.delay - self.risk_window, ts_s - self.delay
        n_del = bisect_left(tts, hi) - bisect_left(tts, lo)
        f_del = bisect_left(t.fraud_ts, hi) - bisect_left(t.fraud_ts, lo)
        # --- device graph
        card_present = device_id < 0
        if card_present:
            d_new, d_ncust, d_1h = 0.0, 0.0, 0.0
        else:
            d = self.dev.get(device_id)
            if d is None:
                d = self.dev[device_id] = _Device()
            d_new = 0.0 if device_id in c.devices else 1.0
            cut = ts_s - 30 * D
            d_ncust = float(sum(1 for k, v in d.customers.items() if v >= cut and k != customer_id))
            d_1h = float(bisect_left(d.ts, ts_s) - bisect_left(d.ts, ts_s - H))

        feats = [
            la, hour, dow, 1.0 if hour < 5 else 0.0, 1.0 if dow >= 5 else 0.0,
            CHANNEL_CODE[channel], CATEGORY_CODE[category],
            n - i1h, n - i24h, n - i7d, math.log1p(c.cum_amt[n] - c.cum_amt[i24h]), mean_la, ratio, z,
            math.log1p(since), c.cum_small[n] - c.cum_small[i1h],
            0.0 if terminal_id in c.terminals else 1.0,
            0.0 if (round(lat, 0), round(lon, 0)) in c.cities else 1.0,
            0.0 if category in c.categories else 1.0,
            haversine_km(c.home_lat, c.home_lon, lat, lon), dprev, speed,
            nt - bisect_left(tts, ts_s - D), nt - bisect_left(tts, ts_s - 7 * D),
            f_del / n_del if n_del else 0.0, n_del,
            d_new, d_ncust, d_1h,
        ]

        # --- update state (after computing features => no leakage of the current tx)
        small = 1.0 if amount < SMALL_AMOUNT else 0.0
        if n == len(ts):  # in-order fast path
            ts.append(ts_s)
            c.cum_amt.append(c.cum_amt[-1] + amount)
            c.cum_la.append(c.cum_la[-1] + la)
            c.cum_la2.append(c.cum_la2[-1] + la * la)
            c.cum_small.append(c.cum_small[-1] + small)
        else:  # late event: insert at its sorted position, shift the prefix sums after it
            ts.insert(n, ts_s)
            for lst, v in ((c.cum_amt, amount), (c.cum_la, la), (c.cum_la2, la * la), (c.cum_small, small)):
                lst.insert(n + 1, lst[n] + v)
                for k in range(n + 2, len(lst)):
                    lst[k] += v
        if c.last_ts is None or ts_s >= c.last_ts:
            c.last_ts, c.last_lat, c.last_lon = ts_s, lat, lon
        c.terminals.add(terminal_id)
        c.cities.add((round(lat, 0), round(lon, 0)))
        c.categories.add(category)
        insort(tts, ts_s)
        if not card_present:
            c.devices.add(device_id)
            insort(d.ts, ts_s)
            d.customers[customer_id] = max(ts_s, d.customers.get(customer_id, ts_s))
        self._prune(c, t, ts_s)
        self.last_ts_s = max(self.last_ts_s, ts_s)
        self.n_seen += 1
        return feats

    def add_label(self, terminal_id: int, ts_s: float) -> None:
        """Register a confirmed fraud (chargeback). Usable only after `label_delay`."""
        t = self.term.get(terminal_id)
        if t is None:
            t = self.term[terminal_id] = _Terminal()
        insort(t.fraud_ts, ts_s)

    @staticmethod
    def _prune(c: _Customer, t: _Terminal, now: float) -> None:
        if len(c.ts) > 256 and c.ts[0] < now - 31 * D:
            k = bisect_left(c.ts, now - 31 * D)
            for lst in (c.ts, c.cum_amt, c.cum_la, c.cum_la2, c.cum_small):
                del lst[:k]
        if len(t.ts) > 512 and t.ts[0] < now - 15 * D:
            del t.ts[:bisect_left(t.ts, now - 15 * D)]
            del t.fraud_ts[:bisect_left(t.fraud_ts, now - 15 * D)]

    # ------------------------------------------------------------------ batch replay
    def process_frame(self, df: pd.DataFrame, use_labels: bool = True,
                      snapshot_at_s: float | None = None, snapshot_path: Path | None = None) -> np.ndarray:
        """Replay a time-ordered frame. Optionally pickle the state right before `snapshot_at_s`."""
        out = np.empty((len(df), len(FEATURES)), dtype=np.float32)
        cols = [df[k].tolist() for k in ("ts_s", "customer_id", "terminal_id", "channel", "category",
                                         "amount", "tx_lat", "tx_lon", "device_id")]
        labels = df["is_fraud"].tolist() if use_labels else None
        snap_done = snapshot_at_s is None
        for i, row in enumerate(zip(*cols)):
            if not snap_done and row[0] >= snapshot_at_s:
                self.save(snapshot_path)
                snap_done = True
            out[i] = self.transform(*row)
            if labels is not None and labels[i]:
                self.add_label(row[2], row[0])
        if not snap_done:
            self.save(snapshot_path)
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "FeatureEngine":
        with open(path, "rb") as f:
            return pickle.load(f)
