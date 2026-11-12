"""Research pipeline — experiments E1..E8 (results -> reports/results, figures -> reports/figures).

E1  Model zoo            rules vs. LogReg vs. RF vs. XGBoost vs. LightGBM vs. CatBoost
E2  Class imbalance      none / scale_pos_weight / is_unbalance / undersampling / SMOTE
E3  Feature ablation     contribution of each feature family (cumulative + leave-one-out)
E4  HPO                  Optuna TPE on LightGBM, objective = valid AUPRC
E5  Calibration          raw vs. Platt vs. isotonic (Brier, ECE, reliability)
E6  Decision policy      cost-optimal REVIEW threshold + high-precision BLOCK threshold
E7  Explainability       TreeSHAP global importance + "which feature catches which scenario"
E8  Robustness / drift   per-scenario recall, weekly performance, PSI drift on the test period
"""
from __future__ import annotations

import json
import os
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, precision_recall_curve  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler  # noqa: E402

from . import viz  # noqa: E402
from .config import get_paths, load_config  # noqa: E402
from .data import load_features, split  # noqa: E402
from .features import FEATURE_GROUPS, FEATURE_LABELS, FEATURES  # noqa: E402
from .metrics import evaluate  # noqa: E402
from .monitoring import psi  # noqa: E402
from .policy import calibrate, fit_calibrator, fit_policy  # noqa: E402
from .preprocessing import signed_log1p  # noqa: E402

CFG = load_config()
P = get_paths(CFG)
SEED = CFG["seed"]
MODELS = P.artifacts / "models"
MODELS.mkdir(parents=True, exist_ok=True)
LGB_BASE = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED)
CAT_COLS = ["channel_code", "category_code"]


def log(msg):
    print(f"[research] {msg}", flush=True)


def save_csv(df: pd.DataFrame, name: str):
    df.to_csv(P.results / name, index=False)


# ----------------------------------------------------------------------------- model helpers
def train_lgb(params, X, y, Xv, yv, weight=None, rounds=3000, features=None):
    feats = features or FEATURES
    dtr = lgb.Dataset(X[feats], y, weight=weight, params={"feature_pre_filter": False}, free_raw_data=False)
    dva = lgb.Dataset(Xv[feats], yv, reference=dtr)
    return lgb.train({**params, "metric": "average_precision"}, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def rule_score(d: pd.DataFrame) -> np.ndarray:
    """Hand-written rules a bank would start with (the baseline any ML model must beat)."""
    s = ((d.c_amt_ratio_30d > 4).astype(float) + ((d.d_new_for_customer == 1) & (d.channel_code == 1))
         + (d.dist_home_km > 300) + (d.speed_prev_kmh > 500) + (d.c_small_cnt_1h >= 3)
         + 0.5 * d.is_night + (d.amount_log > np.log1p(5e6)) + (d.d_n_customers_30d >= 2))
    return (s / 7.5).to_numpy()


def make_logreg():
    num = [f for f in FEATURES if f not in CAT_COLS]
    slog = FunctionTransformer(signed_log1p)
    pre = ColumnTransformer([("num", Pipeline([("log", slog), ("sc", StandardScaler())]), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=0.5, max_iter=3000))])


# ----------------------------------------------------------------------------- E1
def e1_model_zoo(tr, va, te):
    log("E1 model zoo")
    rows, curves = [], {}
    X, y, Xv, yv, Xt = tr[FEATURES], tr.is_fraud, va[FEATURES], va.is_fraud, te[FEATURES]

    days = te.day.to_numpy()
    uniq = np.unique(days)
    rng = np.random.default_rng(SEED)
    boot = [np.concatenate([np.nonzero(days == d)[0] for d in rng.choice(uniq, len(uniq))]) for _ in range(200)]
    yt = te.is_fraud.to_numpy()

    def ci(pt):  # day-block bootstrap: transactions within a day are correlated
        from sklearn.metrics import average_precision_score
        v = [average_precision_score(yt[i], pt[i]) for i in boot]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    def record(name, pv, pt, secs, n_trees=None, size=None):
        lo, hi = ci(np.asarray(pt))
        r = {"model": name, "train_time_s": round(secs, 1), "n_trees": n_trees, "model_size_kb": size,
             "test_auprc_ci_low": lo, "test_auprc_ci_high": hi}
        r.update({f"valid_{k}": v for k, v in evaluate(va, pv, CFG).items()})
        r.update({f"test_{k}": v for k, v in evaluate(te, pt, CFG).items()})
        rows.append(r)
        curves[name] = pt
        log(f"   {name:<12} valid AUPRC={r['valid_auprc']:.4f} test AUPRC={r['test_auprc']:.4f} "
            f"[95% CI {lo:.3f}-{hi:.3f}] ({secs:.0f}s)")

    record("Rules", rule_score(va), rule_score(te), 0.0)

    t = time.perf_counter(); lr = make_logreg().fit(X, y)
    record("LogReg", lr.predict_proba(Xv)[:, 1], lr.predict_proba(Xt)[:, 1], time.perf_counter() - t)
    joblib.dump(lr, MODELS / "logreg.joblib")

    t = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=3, n_jobs=-1,
                                random_state=SEED).fit(X, y)
    joblib.dump(rf, MODELS / "random_forest.joblib", compress=3)
    record("RandomForest", rf.predict_proba(Xv)[:, 1], rf.predict_proba(Xt)[:, 1], time.perf_counter() - t,
           300, (MODELS / "random_forest.joblib").stat().st_size // 1024)

    t = time.perf_counter()
    xg = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=6, subsample=0.8,
                           colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
                           early_stopping_rounds=100, n_jobs=os.cpu_count(), random_state=SEED)
    xg.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    xg.save_model(MODELS / "xgboost.json")
    record("XGBoost", xg.predict_proba(Xv)[:, 1], xg.predict_proba(Xt)[:, 1], time.perf_counter() - t,
           xg.best_iteration + 1, (MODELS / "xgboost.json").stat().st_size // 1024)

    t = time.perf_counter()
    bst = train_lgb(LGB_BASE, X, y, Xv, yv)
    bst.save_model(MODELS / "lgbm_default.txt", num_iteration=bst.best_iteration)
    record("LightGBM", bst.predict(Xv, num_iteration=bst.best_iteration), bst.predict(Xt, num_iteration=bst.best_iteration),
           time.perf_counter() - t, bst.best_iteration, (MODELS / "lgbm_default.txt").stat().st_size // 1024)

    t = time.perf_counter()
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.08, depth=6, eval_metric="PRAUC", od_type="Iter",
                            od_wait=100, use_best_model=True, random_seed=SEED, verbose=0, thread_count=-1)
    cb.fit(X, y, eval_set=(Xv, yv))
    cb.save_model(str(MODELS / "catboost.cbm"))
    record("CatBoost", cb.predict_proba(Xv)[:, 1], cb.predict_proba(Xt)[:, 1], time.perf_counter() - t,
           cb.get_best_iteration() + 1, (MODELS / "catboost.cbm").stat().st_size // 1024)

    res = pd.DataFrame(rows).sort_values("test_auprc", ascending=False)
    save_csv(res, "e1_model_comparison.csv")
    return res, curves


# ----------------------------------------------------------------------------- E2
def e2_imbalance(tr, va, te):
    log("E2 class imbalance strategies")
    y = tr.is_fraud.to_numpy()
    neg, pos = (y == 0).sum(), (y == 1).sum()
    rng = np.random.default_rng(SEED)
    rows = []
    strategies = {
        "none": (LGB_BASE, tr, None),
        "scale_pos_weight=sqrt": ({**LGB_BASE, "scale_pos_weight": float(np.sqrt(neg / pos))}, tr, None),
        "is_unbalance (=neg/pos)": ({**LGB_BASE, "is_unbalance": True}, tr, None),
    }
    keep = (y == 1) | (rng.random(len(y)) < (pos * 20) / neg)
    strategies["undersample 1:20"] = (LGB_BASE, tr[keep], None)
    Xs, ys = SMOTE(sampling_strategy=0.05, random_state=SEED, k_neighbors=5).fit_resample(tr[FEATURES], tr.is_fraud)
    strategies["SMOTE (5%)"] = (LGB_BASE, Xs.assign(is_fraud=ys), None)
    for name, (params, d, w) in strategies.items():
        t = time.perf_counter()
        b = train_lgb(params, d, d.is_fraud, va, va.is_fraud, weight=w)
        pv, pt = b.predict(va[FEATURES]), b.predict(te[FEATURES])
        ev, et = evaluate(va, pv, CFG), evaluate(te, pt, CFG)
        rows.append({"strategy": name, "train_rows": len(d), "valid_auprc": ev["auprc"], "test_auprc": et["auprc"],
                     "test_roc_auc": et["roc_auc"], "test_brier_raw": et["brier"],
                     "mean_pred_vs_base_rate": float(pt.mean() / te.is_fraud.mean()),
                     "train_time_s": round(time.perf_counter() - t, 1)})
        log(f"   {name:<24} test AUPRC={et['auprc']:.4f} brier={et['brier']:.5f}")
    res = pd.DataFrame(rows)
    save_csv(res, "e2_imbalance.csv")
    return res


# ----------------------------------------------------------------------------- E3
def e3_ablation(tr, va, te):
    log("E3 feature-family ablation")
    rows, used = [], []
    for g, feats in FEATURE_GROUPS.items():
        used = used + feats
        b = train_lgb(LGB_BASE, tr, tr.is_fraud, va, va.is_fraud, features=used)
        rows.append({"setting": f"+ {g}" if len(rows) else g, "mode": "cumulative", "n_features": len(used),
                     "valid_auprc": evaluate(va, b.predict(va[used]), CFG)["auprc"],
                     "test_auprc": evaluate(te, b.predict(te[used]), CFG)["auprc"]})
    for g in FEATURE_GROUPS:
        if g == "transaction":
            continue
        feats = [f for f in FEATURES if f not in FEATURE_GROUPS[g]]
        b = train_lgb(LGB_BASE, tr, tr.is_fraud, va, va.is_fraud, features=feats)
        rows.append({"setting": f"- {g}", "mode": "leave_one_out", "n_features": len(feats),
                     "valid_auprc": evaluate(va, b.predict(va[feats]), CFG)["auprc"],
                     "test_auprc": evaluate(te, b.predict(te[feats]), CFG)["auprc"]})
    res = pd.DataFrame(rows)
    for r in res.itertuples():
        log(f"   {r.setting:<22} test AUPRC={r.test_auprc:.4f}")
    save_csv(res, "e3_ablation.csv")
    return res


# ----------------------------------------------------------------------------- E4
def e4_optuna(tr, va):
    log(f"E4 Optuna HPO ({CFG['research']['optuna_trials']} trials)")
    cache = {}

    def datasets(max_bin):  # bin edges are fixed at construction -> one Dataset pair per max_bin
        if max_bin not in cache:
            dtr = lgb.Dataset(tr[FEATURES], tr.is_fraud, params={"feature_pre_filter": False, "max_bin": max_bin},
                              free_raw_data=False)
            cache[max_bin] = (dtr, lgb.Dataset(va[FEATURES], va.is_fraud, reference=dtr))
        return cache[max_bin]

    def objective(trial):
        params = {
            "objective": "binary", "metric": "average_precision", "verbose": -1, "seed": SEED,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0), "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True),
        }
        dtr, dva = datasets(params["max_bin"])
        b = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        trial.set_user_attr("best_iteration", b.best_iteration)
        return b.best_score["valid_0"]["average_precision"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "feature_fraction": 0.8,
                         "bagging_fraction": 0.8, "lambda_l1": 1e-3, "lambda_l2": 1e-3, "max_bin": 255,
                         "scale_pos_weight": 1.0})  # trial 0 = E1 default -> fair "before vs after"
    study.optimize(objective, n_trials=CFG["research"]["optuna_trials"], timeout=CFG["research"]["optuna_timeout_s"])
    trials = study.trials_dataframe()
    save_csv(trials, "e4_optuna_trials.csv")
    best = {**study.best_params, "best_iteration": study.best_trial.user_attrs["best_iteration"],
            "valid_auprc": study.best_value, "n_trials": len(study.trials)}
    (P.results / "e4_best_params.json").write_text(json.dumps(best, indent=2))
    log(f"   best valid AUPRC={study.best_value:.4f} (default trial={study.trials[0].value:.4f})")

    params = {**LGB_BASE, **{k: v for k, v in study.best_params.items()}}
    champ = train_lgb(params, tr, tr.is_fraud, va, va.is_fraud)
    champ.save_model(MODELS / "lgbm_champion.txt", num_iteration=champ.best_iteration)
    (MODELS / "lgbm_champion_params.json").write_text(json.dumps(params, indent=2))
    return champ, trials, best


# ----------------------------------------------------------------------------- E5 + E6
def ece(y, p, bins=None):
    bins = bins or [0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.0001]
    idx = np.digitize(p, bins) - 1
    out = 0.0
    for b in range(len(bins) - 1):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def e5_e6_calibration_policy(champ, va, te):
    log("E5 calibration + E6 decision policy")
    yv, yt = va.is_fraud.to_numpy(), te.is_fraud.to_numpy()
    pv, pt = champ.predict(va[FEATURES]), champ.predict(te[FEATURES])
    cal = fit_calibrator(pv, yv)
    platt = LogisticRegression().fit(np.log(pv / (1 - pv + 1e-12) + 1e-12).reshape(-1, 1), yv)
    pt_platt = platt.predict_proba(np.log(pt / (1 - pt + 1e-12) + 1e-12).reshape(-1, 1))[:, 1]
    pt_iso = calibrate(pt, cal)
    variants = {"raw": pt, "platt": pt_platt, "isotonic": pt_iso}
    cal_rows = [{"method": k, "test_brier": brier_score_loss(yt, v), "test_ece": ece(yt, v),
                 "test_auprc": evaluate(te, v, CFG)["auprc"]} for k, v in variants.items()]
    cal_df = pd.DataFrame(cal_rows)
    save_csv(cal_df, "e5_calibration.csv")

    pv_cal = calibrate(pv, cal)
    policy, sweep_v = fit_policy(pv_cal, va, CFG)
    from .metrics import business_value, threshold_sweep
    rc, fc = CFG["cost"]["review_cost"], CFG["cost"]["friction_cost"]
    sweep_t = threshold_sweep(yt, pt_iso, te.amount.to_numpy(), rc, fc)
    save_csv(sweep_v.assign(split="valid"), "e6_threshold_sweep_valid.csv")
    save_csv(sweep_t.assign(split="test"), "e6_threshold_sweep_test.csv")
    policy["test_at_review"] = business_value(yt, pt_iso, te.amount.to_numpy(), policy["t_review"], rc, fc)
    policy["test_at_block"] = business_value(yt, pt_iso, te.amount.to_numpy(), policy["t_block"], rc, fc)
    (P.results / "e6_policy.json").write_text(json.dumps(policy, indent=2))
    (MODELS / "lgbm_champion_calibration.json").write_text(json.dumps(cal))
    log(f"   t_review={policy['t_review']:.4f} t_block={policy['t_block']:.4f} | test recall="
        f"{policy['test_at_review']['recall']:.3f} precision={policy['test_at_review']['precision']:.3f} "
        f"loss prevented={policy['test_at_review']['loss_prevented_pct']:.1%}")
    return cal_df, variants, policy, sweep_v, sweep_t, pt_iso


# ----------------------------------------------------------------------------- E7
def e7_explain(champ, te):
    log("E7 TreeSHAP explainability")
    sample = pd.concat([te[te.is_fraud == 1], te[te.is_fraud == 0].sample(15000, random_state=SEED)])
    contrib = champ.predict(sample[FEATURES], pred_contrib=True)[:, :-1]  # native C++ TreeSHAP
    imp = pd.DataFrame({"feature": FEATURES, "label": [FEATURE_LABELS[f] for f in FEATURES],
                        "mean_abs_shap": np.abs(contrib).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    save_csv(imp, "e7_shap_importance.csv")
    fraud = sample.is_fraud.to_numpy() == 1
    rows = []
    for sc in sorted(sample.scenario[fraud].unique()):
        m = (sample.scenario == sc).to_numpy()
        mean_c = contrib[m].mean(0)
        top = np.argsort(-mean_c)[:3]
        rows.append({"scenario": sc, "n": int(m.sum()), **{f"top{i+1}": FEATURE_LABELS[FEATURES[j]] for i, j in enumerate(top)},
                     **{f"top{i+1}_shap": float(mean_c[j]) for i, j in enumerate(top)}})
    sc_df = pd.DataFrame(rows)
    save_csv(sc_df, "e7_scenario_drivers.csv")
    return imp, sc_df


# ----------------------------------------------------------------------------- E8
def e8_drift(tr, te, p_test, policy):
    log("E8 robustness: scenario recall, weekly performance, PSI drift")
    te = te.assign(p=p_test, alert=p_test >= policy["t_review"], week=(te.day - te.day.min()) // 7)
    sc = te[te.is_fraud == 1].groupby("scenario").agg(n=("alert", "size"), caught=("alert", "sum"),
                                                     amount=("amount", "sum"))
    sc["recall"] = sc.caught / sc.n
    sc = sc.reset_index().sort_values("recall")
    save_csv(sc, "e8_scenario_recall.csv")
    rows = []
    for w, g in te.groupby("week"):
        ev = evaluate(g, g.p.to_numpy(), CFG)
        row = {"week": int(w), "start_day": int(g.day.min()), "n_tx": len(g), "fraud_rate": g.is_fraud.mean(),
               "auprc": ev["auprc"], "recall_at_review": g[g.is_fraud == 1].alert.mean(),
               "share_social_engineering": (g.scenario == "social_engineering").sum() / max(1, g.is_fraud.sum())}
        for f in ["amount_log", "c_amt_ratio_30d", "dist_home_km", "d_new_for_customer", "category_code"]:
            row[f"psi_{f}"] = psi(tr[f].to_numpy(), g[f].to_numpy())
        rows.append(row)
    wk = pd.DataFrame(rows)
    save_csv(wk, "e8_weekly.csv")
    return sc, wk


# ----------------------------------------------------------------------------- figures
def figures(e1, curves, e2, e3, trials, cal_variants, te, sweep_v, sweep_t, policy, imp, sc, wk, champ_pt):
    log("rendering figures")
    F = P.figures
    # E1 ranked bars (champion emphasised)
    fig, ax = viz.plt.subplots(figsize=(7.4, 3.8))
    bars = viz.hbar(ax, e1.model.tolist(), e1.test_auprc.tolist(), highlight=e1.model.iloc[0])
    for t in ax.texts[-len(bars):]:
        t.remove()
    yy = np.arange(len(e1))
    ax.errorbar(e1.test_auprc, yy, xerr=[e1.test_auprc - e1.test_auprc_ci_low, e1.test_auprc_ci_high - e1.test_auprc],
                fmt="none", ecolor=viz.INK2, elinewidth=1, capsize=3)
    for y_, v, hi in zip(yy, e1.test_auprc, e1.test_auprc_ci_high):
        ax.text(hi, y_, f"  {v:.3f}", va="center", fontsize=9, color=viz.INK2)
    ax.set_xlim(0, 1.08)
    ax.set_title("E1 · Test AUPRC with 95% day-block bootstrap CI")
    ax.set_xlabel("AUPRC")
    viz.save(fig, F / "e1_model_comparison.png")

    # PR curves (<= 4 series, legend + end labels)
    fig, ax = viz.plt.subplots(figsize=(6.4, 4.6))
    y = te.is_fraud.to_numpy()
    show = ["Rules", "LogReg"] + [m for m in e1.model if m not in ("Rules", "LogReg")][:1]
    for i, name in enumerate(show):
        pr, rc, _ = precision_recall_curve(y, curves[name])
        ax.plot(rc, pr, color=viz.SERIES[i], lw=2, label=name)
    pr, rc, _ = precision_recall_curve(y, champ_pt)
    ax.plot(rc, pr, color=viz.SERIES[3], lw=2, label="LightGBM tuned + calibrated")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Precision–recall curves · test set")
    ax.legend(loc="lower left")
    viz.save(fig, F / "e1_pr_curves.png")

    # E3 ablation (cumulative)
    cum = e3[e3["mode"] == "cumulative"]
    fig, ax = viz.plt.subplots(figsize=(7, 3.2))
    viz.hbar(ax, cum.setting.tolist(), cum.test_auprc.tolist(), highlight=cum.setting.iloc[-1])
    ax.set_title("E3 · Each feature family adds signal (cumulative, test AUPRC)")
    viz.save(fig, F / "e3_ablation.png")

    # E4 optuna history
    fig, ax = viz.plt.subplots(figsize=(7, 3.6))
    v = trials.value.to_numpy()
    ax.scatter(trials.number, v, s=26, color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=1.5, label="trial", zorder=3)
    ax.plot(trials.number, np.maximum.accumulate(np.nan_to_num(v, nan=0)), color=viz.SERIES[1], lw=2, label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("valid AUPRC"); ax.legend(loc="lower right")
    ax.set_title("E4 · Optuna TPE search on LightGBM")
    viz.save(fig, F / "e4_optuna_history.png")

    # E5 reliability
    fig, ax = viz.plt.subplots(figsize=(5.6, 4.6))
    bins = np.array([0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.0001])
    for i, (name, p) in enumerate([("raw", cal_variants["raw"]), ("isotonic", cal_variants["isotonic"])]):
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() > 30:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, marker="o", color=viz.SERIES[i], label=name, markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    ax.plot([1e-4, 1], [1e-4, 1], color=viz.AXIS, lw=1, label="perfect")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("predicted probability"); ax.set_ylabel("observed fraud rate")
    ax.set_title("E5 · Reliability diagram (test)"); ax.legend(loc="upper left")
    viz.save(fig, F / "e5_calibration.png")

    # E6 cost curve
    fig, ax = viz.plt.subplots(figsize=(7, 3.8))
    for i, (name, s) in enumerate([("valid", sweep_v), ("test", sweep_t)]):
        ax.plot(s.threshold, s.net_savings / 1e6, color=viz.SERIES[i], label=name)
    ax.axvline(policy["t_review"], color=viz.INK2, lw=1)
    ax.text(policy["t_review"], ax.get_ylim()[1] * 0.95, "  REVIEW threshold", color=viz.INK2, fontsize=9, va="top")
    ax.set_xscale("log"); ax.set_xlabel("threshold (calibrated probability)"); ax.set_ylabel("net savings (Rp juta)")
    ax.set_title("E6 · Net business value vs. threshold"); ax.legend(loc="lower left")
    viz.save(fig, F / "e6_cost_curve.png")

    # E7 shap importance
    top = imp.head(15)
    fig, ax = viz.plt.subplots(figsize=(7, 4.8))
    viz.hbar(ax, top.label.tolist(), top.mean_abs_shap.tolist(), fmt="{:.2f}")
    ax.set_title("E7 · Global importance (mean |SHAP|, log-odds)")
    viz.save(fig, F / "e7_shap_importance.png")

    # E8 scenario recall (status colour only where it means something: the novel scenario)
    fig, ax = viz.plt.subplots(figsize=(7, 3.4))
    bars = viz.hbar(ax, sc.scenario.tolist(), sc.recall.tolist(), fmt="{:.0%}")
    for b, s in zip(bars, sc.scenario):
        if s == "social_engineering":
            b.set_color(viz.CRITICAL)
    ax.set_xlim(0, 1.12); ax.set_title("E8 · Recall per fraud scenario at the REVIEW threshold (test)")
    ax.text(0.0, -0.22, "red = pattern never seen in training (concept drift)", transform=ax.transAxes,
            fontsize=8.5, color=viz.INK2)
    viz.save(fig, F / "e8_scenario_recall.png")

    # E8 weekly — two separate panels, never dual-axis
    fig, (a1, a2) = viz.plt.subplots(1, 2, figsize=(10, 3.4))
    a1.plot(wk.week, wk.recall_at_review, marker="o", color=viz.SERIES[0], markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    a1.set_ylim(0, 1); a1.set_title("Weekly recall (test)"); a1.set_xlabel("test week")
    a2.plot(wk.week, wk.share_social_engineering, marker="o", color=viz.SERIES[1], markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    a2.set_ylim(0, 1); a2.set_title("Share of fraud that is the new scenario"); a2.set_xlabel("test week")
    viz.save(fig, F / "e8_weekly.png")


def run():
    t0 = time.perf_counter()
    s = split(load_features(CFG), CFG)
    tr, va, te = s["train"], s["valid"], s["test"]
    log(f"train={len(tr):,} ({tr.is_fraud.sum()} fraud) valid={len(va):,} ({va.is_fraud.sum()}) "
        f"test={len(te):,} ({te.is_fraud.sum()})")
    e1, curves = e1_model_zoo(tr, va, te)
    e2 = e2_imbalance(tr, va, te)
    e3 = e3_ablation(tr, va, te)
    champ, trials, best = e4_optuna(tr, va)
    cal_df, cal_variants, policy, sweep_v, sweep_t, pt_iso = e5_e6_calibration_policy(champ, va, te)
    champ_eval = evaluate(te, pt_iso, CFG, threshold=policy["t_review"])
    imp, sc_drivers = e7_explain(champ, te)
    sc, wk = e8_drift(tr, te, pt_iso, policy)
    figures(e1, curves, e2, e3, trials, cal_variants, te, sweep_v, sweep_t, policy, imp, sc, wk, pt_iso)
    summary = {"data": {k: {"rows": len(v), "fraud": int(v.is_fraud.sum())} for k, v in s.items()},
               "champion_test": champ_eval, "best_params": best, "policy": policy,
               "runtime_min": round((time.perf_counter() - t0) / 60, 1)}
    (P.results / "research_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done in {summary['runtime_min']} min")


if __name__ == "__main__":
    run()
