import numpy as np
import pandas as pd

from frauddet.metrics import business_value, precision_recall_at_k_daily
from frauddet.monitoring import psi
from frauddet.policy import calibrate, decide, fit_calibrator


def test_business_value_accounting():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.1, 0.0])
    amt = np.array([1_000_000, 50_000, 2_000_000, 10_000])
    bv = business_value(y, p, amt, 0.5, review_cost=10_000, friction_cost=20_000)
    assert bv["tp"] == 1 and bv["fp"] == 1 and bv["alerts"] == 2
    assert bv["fraud_amount_prevented"] == 1_000_000
    assert bv["net_savings"] == 1_000_000 - 2 * 10_000 - 20_000


def test_precision_at_k_daily():
    df = pd.DataFrame({"day": [0, 0, 0, 1, 1, 1], "is_fraud": [1, 0, 0, 0, 1, 0]})
    p = np.array([0.9, 0.5, 0.1, 0.8, 0.7, 0.1])
    prec, rec = precision_recall_at_k_daily(df, p, k=1)
    assert prec == 0.5 and rec == 0.5


def test_calibration_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p ** 3).astype(int)
    cal = fit_calibrator(p, y)
    out = calibrate(np.linspace(0, 1, 50), cal)
    assert np.all(np.diff(out) >= -1e-12) and out.min() >= 0 and out.max() <= 1


def test_decision_tiers():
    pol = {"t_review": 0.1, "t_block": 0.8}
    assert decide(0.05, pol) == "APPROVE" and decide(0.3, pol) == "REVIEW" and decide(0.9, pol) == "BLOCK"


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 20000)
    assert psi(a, rng.normal(0, 1, 20000)) < 0.02
    assert psi(a, rng.normal(1.0, 1, 20000)) > 0.25
