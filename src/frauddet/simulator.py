"""Synthetic card-transaction simulator (Indonesia, IDR) with realistic fraud scenarios.

Why synthetic?  Public fraud datasets are either anonymised PCA features (no business
meaning, impossible to explain on a dashboard) or lack entity IDs / timestamps needed for
real-time behavioural features.  This generator is inspired by the *Reproducible Machine
Learning for Credit Card Fraud Detection* handbook (Le Borgne et al.) and extends it with
modern attack patterns:

==================== ============================================================== =========
scenario             signal a good model must learn                                 period
==================== ============================================================== =========
cnp_stolen_card      online, foreign/unknown device shared by a fraud ring, IP far   all
card_testing         burst of tiny amounts on digital goods, then a large purchase   all
account_takeover     brand-new device + other city + huge e-wallet top-up at night   all
card_cloning         card-present use in another city right after skimming           all
                     (impossible travel)
merchant_collusion   a merchant whose *delayed* chargeback rate spikes               all
social_engineering   victim's OWN device & location, unusual new payee, large amount test only
                     -> concept drift that the model has never seen
==================== ============================================================== =========
Legit noise is deliberate: customers travel, share household devices, and occasionally
make big purchases, so no single rule separates the classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CITIES = [  # name, lat, lon, population weight
    ("Jakarta", -6.2088, 106.8456, 0.30),
    ("Surabaya", -7.2575, 112.7521, 0.12),
    ("Bandung", -6.9175, 107.6191, 0.11),
    ("Medan", 3.5952, 98.6722, 0.09),
    ("Semarang", -6.9667, 110.4167, 0.07),
    ("Makassar", -5.1477, 119.4327, 0.07),
    ("Yogyakarta", -7.7956, 110.3695, 0.07),
    ("Denpasar", -8.6705, 115.2126, 0.07),
    ("Palembang", -2.9761, 104.7754, 0.05),
    ("Balikpapan", -1.2379, 116.8529, 0.05),
]
FOREIGN_IP = [("Singapore", 1.3521, 103.8198), ("Kuala Lumpur", 3.1390, 101.6869),
              ("Bangkok", 13.7563, 100.5018), ("Ho Chi Minh", 10.8231, 106.6297)]

CATEGORIES = ["grocery", "restaurant", "fuel", "fashion", "electronics",
              "ecommerce", "ewallet_topup", "digital_goods", "travel", "atm_cash"]
CAT_ID = {c: i for i, c in enumerate(CATEGORIES)}
BASE_AMOUNT = np.array([150e3, 85e3, 200e3, 350e3, 1.5e6, 250e3, 200e3, 75e3, 1.2e6, 500e3])
POS_CATS, POS_W = [0, 1, 2, 3, 4], [0.35, 0.30, 0.15, 0.13, 0.07]
ONL_CATS, ONL_W = [5, 6, 7, 8], [0.45, 0.25, 0.20, 0.10]
CHANNELS = ["POS", "ONLINE", "ATM"]
POS, ONLINE, ATM = 0, 1, 2

# hour-of-day profiles
LEGIT_HOURS = np.array([.5, .3, .2, .2, .3, .8, 2, 3.5, 4.5, 5, 5.5, 6, 7, 6.5, 5.5, 5, 5.5, 6.5, 7, 7, 6, 4.5, 3, 1.5])
FRAUD_HOURS = np.array([4, 4.5, 5, 5, 4.5, 3, 2, 2, 2.5, 3, 3, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4.5, 4.5, 4.5])
DAY_HOURS = np.array([0, 0, 0, 0, 0, 0, .5, 2, 4, 6, 7, 7, 6, 6, 6, 6, 5, 4, 3, 2, 1, .5, 0, 0], dtype=float)
COLUMNS = ["ts_s", "customer_id", "terminal_id", "channel", "category", "amount",
           "tx_lat", "tx_lon", "tx_city", "device_id", "is_fraud", "scenario"]
FRAUD_DEVICE_BASE = 10_000_000


class TransactionSimulator:
    def __init__(self, cfg: dict, seed: int = 42):
        self.c = cfg["simulation"]
        self.rng = np.random.default_rng(seed)
        self.n_days = int(self.c["n_days"])
        self.city_lat = np.array([c[1] for c in CITIES])
        self.city_lon = np.array([c[2] for c in CITIES])
        self.city_w = np.array([c[3] for c in CITIES])
        self.city_names = [c[0] for c in CITIES] + [f[0] for f in FOREIGN_IP]

    # ------------------------------------------------------------------ entities
    def _make_terminals(self) -> pd.DataFrame:
        r, c = self.rng, self.c
        parts = []
        for channel, n, cats, w in ((POS, c["n_pos_terminals"], POS_CATS, POS_W),
                                    (ATM, c["n_atm"], [CAT_ID["atm_cash"]], [1.0])):
            city = np.sort(r.choice(len(CITIES), n, p=self.city_w))
            parts.append(pd.DataFrame({
                "channel": channel, "category": r.choice(cats, n, p=w), "city": city,
                "lat": self.city_lat[city] + r.normal(0, 0.06, n),
                "lon": self.city_lon[city] + r.normal(0, 0.06, n)}))
        n = c["n_online_merchants"]
        parts.append(pd.DataFrame({"channel": ONLINE, "category": r.choice(ONL_CATS, n, p=ONL_W),
                                   "city": 0, "lat": self.city_lat[0], "lon": self.city_lon[0]}))
        t = pd.concat(parts, ignore_index=True)
        t.insert(0, "terminal_id", np.arange(len(t)))
        return t

    def _make_customers(self, terms: pd.DataFrame) -> pd.DataFrame:
        r, n = self.rng, int(self.c["n_customers"])
        city = r.choice(len(CITIES), n, p=self.city_w)
        cust = pd.DataFrame({
            "customer_id": np.arange(n), "home_city": city,
            "home_lat": self.city_lat[city] + r.normal(0, 0.06, n),
            "home_lon": self.city_lon[city] + r.normal(0, 0.06, n),
            "spend_scale": r.lognormal(0, 0.5, n),
            "tx_rate": r.gamma(4.0, self.c["mean_tx_per_day"] / 4.0, n),
            "online_share": r.beta(2, 5, n)})
        # devices: primary = customer_id; 35% own a 2nd device, ~4% share a household device
        second = np.full(n, -1)
        has2 = r.random(n) < 0.35
        second[has2] = n + np.arange(n)[has2]
        shared = r.random(n) < 0.04
        second[shared] = r.integers(0, n, shared.sum())
        cust["device2"] = second
        # preferred terminals near home
        pos = terms[terms.channel == POS]
        self._pref_pos = np.zeros((n, 15), dtype=np.int64)
        for ci in range(len(CITIES)):
            ids = pos.terminal_id[pos.city == ci].to_numpy()
            m = city == ci
            self._pref_pos[m] = r.choice(ids, (m.sum(), 15))
        onl = terms.terminal_id[terms.channel == ONLINE].to_numpy()
        self._pref_onl = r.choice(onl, (n, 8))
        return cust

    def _index_terminals(self, t: pd.DataFrame):
        self._t_lat, self._t_lon = t.lat.to_numpy(), t.lon.to_numpy()
        self._t_cat, self._t_ch = t.category.to_numpy(), t.channel.to_numpy()
        self._by = {}
        for ch in (POS, ATM):
            sub = t[t.channel == ch]
            self._by[ch] = [sub.terminal_id[sub.city == ci].to_numpy() for ci in range(len(CITIES))]
        self._online = t.terminal_id[t.channel == ONLINE].to_numpy()
        self._online_by_cat = {k: t.terminal_id[(t.channel == ONLINE) & (t.category == k)].to_numpy() for k in ONL_CATS}

    def _random_terminal(self, ch: int, city: np.ndarray) -> np.ndarray:
        out = np.empty(len(city), dtype=np.int64)
        for ci in np.unique(city):
            m = city == ci
            out[m] = self.rng.choice(self._by[ch][ci], m.sum())
        return out

    # ------------------------------------------------------------------ legit
    def _legit(self, cust: pd.DataFrame) -> pd.DataFrame:
        r, nd = self.rng, self.n_days
        n = len(cust)
        home = cust.home_city.to_numpy()
        # where each customer is on each day (legit travel creates geo noise)
        citymat = np.repeat(home[:, None], nd, axis=1)
        n_trips = r.poisson(self.c["trips_per_customer"], n)
        for i in np.nonzero(n_trips)[0]:
            for _ in range(n_trips[i]):
                s = r.integers(0, nd)
                dest = r.choice(len(CITIES), p=self.city_w)
                if dest != home[i]:
                    citymat[i, s:s + r.integers(2, 7)] = dest
        dow = (pd.Timestamp(self.c["start_date"]).dayofweek + np.arange(nd)) % 7
        day_factor = np.where(dow >= 5, 1.15, 1.0)
        counts = r.poisson(cust.tx_rate.to_numpy()[:, None] * day_factor[None, :])
        flat = np.repeat(np.arange(n * nd), counts.ravel())
        cid, day = flat // nd, flat % nd
        m = len(cid)
        hour = r.choice(24, m, p=LEGIT_HOURS / LEGIT_HOURS.sum())
        ts = day * 86400.0 + hour * 3600.0 + r.uniform(0, 3600, m)
        city = citymat[cid, day]

        u = r.random(m)
        channel = np.where(u < 0.05, ATM, np.where(u < 0.05 + cust.online_share.to_numpy()[cid], ONLINE, POS))
        term = np.empty(m, dtype=np.int64)
        # POS: 85% preferred terminal when at home, else random terminal in current city
        mp = channel == POS
        use_pref = mp & (city == home[cid]) & (r.random(m) < 0.85)
        term[use_pref] = self._pref_pos[cid[use_pref], r.integers(0, 15, use_pref.sum())]
        rnd = mp & ~use_pref
        term[rnd] = self._random_terminal(POS, city[rnd])
        ma = channel == ATM
        term[ma] = self._random_terminal(ATM, city[ma])
        mo = channel == ONLINE
        pref_o = mo & (r.random(m) < 0.8)
        term[pref_o] = self._pref_onl[cid[pref_o], r.integers(0, 8, pref_o.sum())]
        rnd_o = mo & ~pref_o
        term[rnd_o] = r.choice(self._online, rnd_o.sum())

        cat = self._t_cat[term]
        amount = BASE_AMOUNT[cat] * cust.spend_scale.to_numpy()[cid] * r.lognormal(0, 0.55, m)
        # online location = IP geolocation of current city; card-present = terminal location
        lat = np.where(mo, self.city_lat[city] + r.normal(0, 0.04, m), self._t_lat[term])
        lon = np.where(mo, self.city_lon[city] + r.normal(0, 0.04, m), self._t_lon[term])
        dev2 = cust.device2.to_numpy()[cid]
        device = np.where(mo, np.where((dev2 >= 0) & (r.random(m) < 0.3), dev2, cid), -1)
        return pd.DataFrame({"ts_s": ts, "customer_id": cid, "terminal_id": term, "channel": channel,
                             "category": cat, "amount": amount, "tx_lat": lat, "tx_lon": lon,
                             "tx_city": city, "device_id": device, "is_fraud": 0, "scenario": "legit"})

    # ------------------------------------------------------------------ fraud helpers
    def _fraud_device(self, k: int) -> np.ndarray:
        pool = int(self.c["fraud_device_pool"])
        w = 1.0 / np.arange(1, pool + 1) ** 0.8  # a few devices are reused by many victims
        return FRAUD_DEVICE_BASE + self.rng.choice(pool, k, p=w / w.sum())

    def _far_location(self, home_city: int, foreign_p: float):
        r = self.rng
        if r.random() < foreign_p:
            j = r.integers(len(FOREIGN_IP))
            return len(CITIES) + j, FOREIGN_IP[j][1], FOREIGN_IP[j][2]
        choices = [i for i in range(len(CITIES)) if i != home_city]
        ci = int(r.choice(choices))
        return ci, self.city_lat[ci] + r.normal(0, 0.04), self.city_lon[ci] + r.normal(0, 0.04)

    @staticmethod
    def _rows(**cols) -> pd.DataFrame:
        return pd.DataFrame(cols)[COLUMNS]

    def _online_tx(self, cats, k):
        cats = self.rng.choice(cats, k)
        return np.array([self.rng.choice(self._online_by_cat[c]) for c in cats]), cats

    # ------------------------------------------------------------------ fraud scenarios
    def _cnp(self, cust, n_victims):
        r, rows = self.rng, []
        for v, dev in zip(r.choice(len(cust), n_victims, replace=False), self._fraud_device(n_victims)):
            k, d0, span = r.integers(3, 11), r.integers(3, self.n_days - 3), r.integers(1, 4)
            ts = np.sort((d0 + r.integers(0, span, k)) * 86400.0 + r.choice(24, k, p=FRAUD_HOURS / FRAUD_HOURS.sum()) * 3600 + r.uniform(0, 3600, k))
            term, cat = self._online_tx([5, 7, 8], k)
            city, lat, lon = self._far_location(cust.home_city[v], 0.25) if r.random() < 0.65 else \
                (cust.home_city[v], cust.home_lat[v], cust.home_lon[v])
            amt = BASE_AMOUNT[cat] * cust.spend_scale[v] * r.lognormal(np.log(2.5), 0.6, k)
            rows.append(self._rows(ts_s=ts, customer_id=v, terminal_id=term, channel=ONLINE, category=cat,
                                   amount=amt, tx_lat=lat, tx_lon=lon, tx_city=city, device_id=dev,
                                   is_fraud=1, scenario="cnp_stolen_card"))
        return rows

    def _card_testing(self, cust, n_victims):
        r, rows = self.rng, []
        for v, dev in zip(r.choice(len(cust), n_victims, replace=False), self._fraud_device(n_victims)):
            k = r.integers(5, 16)
            t0 = r.integers(3, self.n_days - 1) * 86400.0 + r.uniform(0, 86400)
            ts_small = t0 + np.cumsum(r.exponential(90, k))
            merch = r.choice(self._online_by_cat[7], 2)
            n_big = r.integers(1, 3)
            ts_big = ts_small[-1] + np.sort(r.uniform(600, 3600, n_big))
            term_big, cat_big = self._online_tx([5, 8], n_big)
            city, lat, lon = self._far_location(cust.home_city[v], 0.5)
            rows.append(self._rows(
                ts_s=np.r_[ts_small, ts_big], customer_id=v,
                terminal_id=np.r_[r.choice(merch, k), term_big], channel=ONLINE,
                category=np.r_[np.full(k, 7), cat_big],
                amount=np.r_[r.uniform(1e3, 15e3, k), r.uniform(2e6, 8e6, n_big)],
                tx_lat=lat, tx_lon=lon, tx_city=city, device_id=dev, is_fraud=1, scenario="card_testing"))
        return rows

    def _ato(self, cust, n_victims):
        r, rows = self.rng, []
        for v, dev in zip(r.choice(len(cust), n_victims, replace=False), self._fraud_device(n_victims)):
            k = r.integers(1, 4)
            t0 = r.integers(3, self.n_days) * 86400.0 + r.choice(24, p=FRAUD_HOURS / FRAUD_HOURS.sum()) * 3600.0
            ts = t0 + np.sort(r.uniform(0, 7200, k))
            term, cat = self._online_tx([6, 6, 6, 5], k)
            city, lat, lon = self._far_location(cust.home_city[v], 0.2)
            amt = np.maximum(3e6, BASE_AMOUNT[cat] * cust.spend_scale[v] * r.lognormal(np.log(12), 0.5, k))
            rows.append(self._rows(ts_s=ts, customer_id=v, terminal_id=term, channel=ONLINE, category=cat,
                                   amount=amt, tx_lat=lat, tx_lon=lon, tx_city=city, device_id=dev,
                                   is_fraud=1, scenario="account_takeover"))
        return rows

    def _card_cloning(self, cust, legit, n_terminals):
        r, rows = self.rng, []
        pos_ids = np.concatenate(self._by[POS])
        for term in r.choice(pos_ids, n_terminals, replace=False):
            d0 = r.integers(5, self.n_days - 20)
            d1 = d0 + r.integers(10, 21)
            used = legit[(legit.terminal_id == term) & (legit.ts_s >= d0 * 86400) & (legit.ts_s < d1 * 86400)]
            for v, t_skim in used.groupby("customer_id").ts_s.min().items():
                if r.random() > 0.35:
                    continue
                k = r.integers(1, 4)
                ts = t_skim + r.uniform(1, 10) * 86400 + np.sort(r.uniform(0, 6 * 3600, k))
                ci = int(r.choice([i for i in range(len(CITIES)) if i != cust.home_city[v]]))
                is_atm = r.random(k) < 0.6
                t_atm = r.choice(self._by[ATM][ci], k)
                t_pos = r.choice(self._by[POS][ci], k)
                tid = np.where(is_atm, t_atm, t_pos)
                amt = np.where(is_atm, r.choice([1e6, 2e6, 2.5e6, 3e6, 5e6], k),
                               BASE_AMOUNT[self._t_cat[tid]] * r.lognormal(np.log(4), 0.5, k))
                rows.append(self._rows(ts_s=ts, customer_id=v, terminal_id=tid, channel=np.where(is_atm, ATM, POS),
                                       category=self._t_cat[tid], amount=amt, tx_lat=self._t_lat[tid],
                                       tx_lon=self._t_lon[tid], tx_city=ci, device_id=-1, is_fraud=1,
                                       scenario="card_cloning"))
        return rows

    def _merchant_collusion(self, df, n_merchants):
        r = self.rng
        cand = np.r_[np.concatenate(self._by[POS]), self._online]
        for term in r.choice(cand, n_merchants, replace=False):
            d0 = r.integers(10, self.n_days - 15)
            d1 = d0 + r.integers(15, 31)
            m = (df.terminal_id == term) & (df.ts_s >= d0 * 86400) & (df.ts_s < d1 * 86400) & (df.is_fraud == 0)
            hit = m & (r.random(len(df)) < 0.35)
            df.loc[hit, "amount"] *= r.uniform(1.5, 3.0, hit.sum())
            df.loc[hit, ["is_fraud", "scenario"]] = [1, "merchant_collusion"]
        return df

    def _social_engineering(self, cust, n_victims, start_day):
        r, rows = self.rng, []
        for v in r.choice(len(cust), n_victims, replace=False):
            k = r.integers(1, 3)
            t0 = r.integers(start_day, self.n_days) * 86400.0 + r.choice(24, p=DAY_HOURS / DAY_HOURS.sum()) * 3600.0
            ts = t0 + np.sort(r.uniform(0, 5400, k))
            term, cat = self._online_tx([6, 6, 5], k)
            amt = r.lognormal(np.log(3.0e6), 0.45, k) * np.sqrt(cust.spend_scale[v])
            rows.append(self._rows(ts_s=ts, customer_id=v, terminal_id=term, channel=ONLINE, category=cat,
                                   amount=amt, tx_lat=cust.home_lat[v] + r.normal(0, 0.02),
                                   tx_lon=cust.home_lon[v] + r.normal(0, 0.02), tx_city=cust.home_city[v],
                                   device_id=v, is_fraud=1, scenario="social_engineering"))
        return rows

    # ------------------------------------------------------------------ public
    def run(self):
        terms = self._make_terminals()
        self._index_terminals(terms)
        cust = self._make_customers(terms)
        legit = self._legit(cust)
        s = self.c["scenarios"]
        parts = [legit]
        parts += self._cnp(cust, s["cnp_stolen_card"])
        parts += self._card_testing(cust, s["card_testing"])
        parts += self._ato(cust, s["account_takeover"])
        parts += self._card_cloning(cust, legit, s["card_cloning"])
        parts += self._social_engineering(cust, s["social_engineering"], self.c["social_engineering_start_day"])
        df = pd.concat(parts, ignore_index=True)
        df = df[df.ts_s < self.n_days * 86400].copy()
        df = self._merchant_collusion(df, s["merchant_collusion"])

        df["amount"] = np.where(df.channel == ATM, np.maximum(50e3, np.round(df.amount / 50e3) * 50e3),
                                np.maximum(1e3, np.round(df.amount / 1e3) * 1e3))
        df = df.sort_values("ts_s", kind="stable").reset_index(drop=True)
        df.insert(0, "tx_id", np.arange(len(df), dtype=np.int64))
        df["ts"] = pd.Timestamp(self.c["start_date"]) + pd.to_timedelta(df.ts_s, unit="s")
        df["day"] = (df.ts_s // 86400).astype(np.int16)
        df["tx_city"] = np.array(self.city_names)[df.tx_city.astype(int)]
        df["channel"] = np.array(CHANNELS)[df.channel.astype(int)]
        df["category"] = np.array(CATEGORIES)[df.category.astype(int)]
        for col, t in (("customer_id", np.int32), ("terminal_id", np.int32), ("device_id", np.int64), ("is_fraud", np.int8)):
            df[col] = df[col].astype(t)

        cust_out = cust[["customer_id", "home_city", "home_lat", "home_lon", "spend_scale", "online_share", "device2"]].copy()
        cust_out["home_city"] = np.array(self.city_names)[cust_out.home_city]
        terms["city"] = np.array(self.city_names)[terms.city]
        terms["channel"] = np.array(CHANNELS)[terms.channel]
        terms["category"] = np.array(CATEGORIES)[terms.category]
        return df, cust_out, terms


def simulate(cfg: dict):
    return TransactionSimulator(cfg, seed=cfg["seed"]).run()
