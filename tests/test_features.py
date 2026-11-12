import numpy as np
import pandas as pd
import pytest

from frauddet.features import FEATURES, FeatureEngine

IDX = {f: i for i, f in enumerate(FEATURES)}


def _engine():
    cust = pd.DataFrame({"customer_id": [1, 2], "home_lat": [-6.2, -7.25], "home_lon": [106.8, 112.75]})
    return FeatureEngine(cust, "2026-05-01", label_delay_days=7)


def _tx(e, ts, cid=1, term=10, amount=100_000, lat=-6.2, lon=106.8, dev=-1, ch="POS", cat="grocery"):
    return e.transform(ts, cid, term, ch, cat, amount, lat, lon, dev)


def test_first_transaction_has_no_history():
    f = _tx(_engine(), 1000.0)
    assert f[IDX["c_cnt_24h"]] == 0 and f[IDX["c_amt_ratio_30d"]] == -1.0 and f[IDX["c_new_terminal"]] == 1.0


def test_current_transaction_is_excluded_no_leakage():
    e = _engine()
    _tx(e, 0.0, amount=50_000)
    f = _tx(e, 60.0, amount=5_000_000)
    # only the previous tx is in the window, and the ratio uses the past mean, not the current amount
    assert f[IDX["c_cnt_1h"]] == 1
    assert f[IDX["c_amt_ratio_30d"]] == pytest.approx((5_000_000 + 1) / (50_000 + 1), rel=1e-6)


def test_velocity_windows_match_bruteforce():
    rng = np.random.default_rng(0)
    ts = np.sort(rng.uniform(0, 40 * 86400, 400))
    amt = rng.lognormal(12, 1, 400)
    e = _engine()
    for i, (t, a) in enumerate(zip(ts, amt)):
        f = _tx(e, float(t), amount=float(a))
        past = ts[:i]
        assert f[IDX["c_cnt_1h"]] == np.sum(past >= t - 3600)
        assert f[IDX["c_cnt_24h"]] == np.sum(past >= t - 86400)
        assert f[IDX["c_cnt_7d"]] == np.sum(past >= t - 7 * 86400)
        assert f[IDX["c_sum_24h_log"]] == pytest.approx(np.log1p(amt[:i][past >= t - 86400].sum()), rel=1e-9)


def test_terminal_label_is_only_used_after_delay():
    e = _engine()
    d = 86400.0
    _tx(e, 0.0, term=7)
    e.add_label(7, 0.0)  # chargeback for the tx at t=0
    early = _tx(e, 3 * d, cid=2, term=7)  # 3 days later: label must NOT be visible yet
    late = _tx(e, 8 * d, cid=2, term=7)  # 8 days later: inside [t-14d, t-7d]
    assert early[IDX["t_risk_delayed"]] == 0.0
    assert late[IDX["t_risk_delayed"]] > 0.0


def test_impossible_travel_speed():
    e = _engine()
    _tx(e, 0.0, lat=-6.2, lon=106.8)  # Jakarta
    f = _tx(e, 1800.0, lat=3.59, lon=98.67)  # Medan, 30 minutes later
    assert f[IDX["dist_prev_km"]] > 1300 and f[IDX["speed_prev_kmh"]] > 2000


def test_device_shared_by_many_cards():
    e = _engine()
    for cid in range(100, 105):
        _tx(e, cid * 10.0, cid=cid, dev=999, ch="ONLINE", cat="ecommerce")
    f = _tx(e, 2000.0, cid=1, dev=999, ch="ONLINE", cat="ecommerce")
    assert f[IDX["d_n_customers_30d"]] == 5 and f[IDX["d_new_for_customer"]] == 1.0


def test_out_of_order_events_match_in_order_state():
    """Concurrent API requests can arrive out of event-time order; state must end up identical."""
    rng = np.random.default_rng(3)
    ts = np.sort(rng.uniform(0, 5 * 86400, 200))
    amt = rng.lognormal(11, 1, 200)
    order = np.arange(200)
    shuffled = order.copy()
    for i in range(0, 200, 4):  # local shuffles, like a small concurrency window
        rng.shuffle(shuffled[i:i + 4])
    e_in, e_out = _engine(), _engine()
    for i in order:
        _tx(e_in, float(ts[i]), amount=float(amt[i]))
    for i in shuffled:
        f = _tx(e_out, float(ts[i]), amount=float(amt[i]))
        assert f[IDX["c_secs_since_last_log"]] >= 0
    a, b = e_in.cust[1], e_out.cust[1]
    assert a.ts == b.ts
    np.testing.assert_allclose(a.cum_amt, b.cum_amt, rtol=1e-9)
    # a probe after all events sees exactly the same features in both engines
    np.testing.assert_allclose(_tx(e_in, 6 * 86400.0), _tx(e_out, 6 * 86400.0), rtol=1e-9)
