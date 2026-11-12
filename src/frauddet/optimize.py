"""Optimisation & latency research — experiments E9..E10, then builds the production bundle.

E9   Model-level trade-offs   accuracy vs. single-row p99 latency vs. throughput vs. size
       * model family (LogReg, RF, XGBoost, CatBoost, LightGBM)
       * runtime       (LightGBM native multi-thread vs. 1-thread vs. ONNX Runtime)
       * tree budget   (truncate the champion to N trees)
       * feature budget(retrain on SHAP top-k features)
     -> Pareto frontier + SLO-constrained selection of the production model
E10  End-to-end path          feature engine + inference + reason codes, comparing explanation
                              strategies (shap library vs. native TreeSHAP vs. explain-on-alert)
                              + training/serving parity check
"""
from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

from . import viz  # noqa: E402
from .config import get_paths, load_config  # noqa: E402
from .data import load_features, split  # noqa: E402
from .features import FEATURES, FeatureEngine  # noqa: E402
from .metrics import evaluate  # noqa: E402
from .monitoring import build_profile  # noqa: E402
from .policy import calibrate, fit_calibrator, fit_policy  # noqa: E402

CFG = load_config()
P = get_paths(CFG)
MODELS = P.artifacts / "models"
SEED = CFG["seed"]


def log(msg):
    print(f"[optimize] {msg}", flush=True)


# ----------------------------------------------------------------------------- benchmarking
def bench_single(fn, X: np.ndarray, n: int = 3000, warmup: int = 300, repeats: int = 3) -> dict:
    """Median over `repeats` runs of each percentile -> robust to OS scheduling noise on a laptop."""
    runs = [_bench_once(fn, X, n, warmup, seed=SEED + r) for r in range(repeats)]
    return {k: float(np.median([r[k] for r in runs])) for k in runs[0]}


def bench_explain(booster: lgb.Booster, X: np.ndarray, n: int = 300) -> dict:
    """Latency of native TreeSHAP reason codes for one row (scales with trees x leaves)."""
    t = []
    for i in range(n):
        x = np.ascontiguousarray(X[i:i + 1])
        t0 = time.perf_counter_ns()
        booster.predict(x, pred_contrib=True, num_threads=1)
        t.append(time.perf_counter_ns() - t0)
    t = np.array(t) / 1e6
    return {"explain_p50_ms": float(np.percentile(t, 50)), "explain_p99_ms": float(np.percentile(t, 99))}


def _bench_once(fn, X: np.ndarray, n: int, warmup: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = np.ascontiguousarray(X[rng.integers(0, len(X), n + warmup)], dtype=np.float32)
    for i in range(warmup):
        fn(rows[i:i + 1])
    t = np.empty(n)
    for i in range(n):
        x = rows[warmup + i:warmup + i + 1]
        t0 = time.perf_counter_ns()
        fn(x)
        t[i] = time.perf_counter_ns() - t0
    t /= 1e6
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99)), "mean_ms": float(t.mean())}


def bench_batch(fn, X: np.ndarray, batch: int = 20000, reps: int = 3) -> float:
    xb = np.ascontiguousarray(X[:batch], dtype=np.float32)
    fn(xb)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(xb)
    return batch * reps / (time.perf_counter() - t0)


def to_onnx(booster: lgb.Booster, n_features: int, path):
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    onx = convert_lightgbm(booster, initial_types=[("input", FloatTensorType([None, n_features]))],
                           zipmap=False, target_opset=15)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())


def onnx_session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0].name, sess.get_outputs()[1].name
    return lambda x: sess.run([out], {inp: x})[0][:, 1]


def kb(path) -> int:
    return int(os.path.getsize(path) // 1024)


# ----------------------------------------------------------------------------- E9
def e9_variants(tr, va, te):
    log("E9 building & benchmarking model variants (single-row latency, 3000 calls each)")
    Xv, Xt = va[FEATURES].to_numpy(np.float32), te[FEATURES].to_numpy(np.float32)
    rows, registry = [], {}

    def add(name, family, runtime, feats, fn_single, fn_batch, size, pv, pt, n_trees=None, extra=None, booster=None):
        idx = [FEATURES.index(f) for f in feats]
        lat = bench_single(fn_single, Xt[:, idx])
        if booster is not None:
            lat.update(bench_explain(booster, Xt[:, idx]))
        thr = bench_batch(fn_batch, Xt[:, idx])
        r = {"variant": name, "family": family, "runtime": runtime, "n_features": len(feats), "n_trees": n_trees,
             "size_kb": size, "valid_auprc": evaluate(va, pv, CFG)["auprc"], "test_auprc": evaluate(te, pt, CFG)["auprc"],
             **lat, "batch_rows_per_s": thr}
        rows.append(r)
        registry[name] = {"features": feats, **(extra or {})}
        log(f"   {name:<34} AUPRC v={r['valid_auprc']:.4f} t={r['test_auprc']:.4f} | p50={lat['p50_ms']:.3f} "
            f"p99={lat['p99_ms']:.3f} ms | {thr:,.0f} rows/s | {size} KB")

    # --- other families
    lr = joblib.load(MODELS / "logreg.joblib")
    lr_fn = lambda x: lr.predict_proba(pd.DataFrame(x, columns=FEATURES))[:, 1]  # noqa: E731
    add("LogReg (sklearn pipeline)", "LogReg", "sklearn", FEATURES, lr_fn, lr_fn, kb(MODELS / "logreg.joblib"),
        lr_fn(Xv), lr_fn(Xt))
    rf = joblib.load(MODELS / "random_forest.joblib")
    rf.n_jobs = 1
    rf_fn = lambda x: rf.predict_proba(x)[:, 1]  # noqa: E731
    add("RandomForest 300 (sklearn)", "RandomForest", "sklearn", FEATURES, rf_fn, rf_fn,
        kb(MODELS / "random_forest.joblib"), rf_fn(Xv), rf_fn(Xt), 300)
    del rf
    xb = xgb.Booster()
    xb.load_model(MODELS / "xgboost.json")
    xb.set_param({"nthread": 1})
    xg_fn = lambda x: xb.inplace_predict(x)  # noqa: E731
    add("XGBoost (native, 1 thread)", "XGBoost", "native", FEATURES, xg_fn, xg_fn, kb(MODELS / "xgboost.json"),
        xg_fn(Xv), xg_fn(Xt), xb.num_boosted_rounds())
    cb = CatBoostClassifier().load_model(str(MODELS / "catboost.cbm"))
    cb_fn = lambda x: cb.predict_proba(x, thread_count=1)[:, 1]  # noqa: E731
    add("CatBoost (native, 1 thread)", "CatBoost", "native", FEATURES, cb_fn, cb_fn, kb(MODELS / "catboost.cbm"),
        cb_fn(Xv), cb_fn(Xt), cb.tree_count_)

    # --- LightGBM champion: runtimes
    champ_path = MODELS / "lgbm_champion.txt"
    champ = lgb.Booster(model_file=str(champ_path))
    n_tree = champ.num_trees()
    pv_c, pt_c = champ.predict(Xv), champ.predict(Xt)
    add("LightGBM champion (native, all threads)", "LightGBM", "native-mt", FEATURES, lambda x: champ.predict(x),
        lambda x: champ.predict(x), kb(champ_path), pv_c, pt_c, n_tree, {"model": str(champ_path)}, booster=champ)
    add("LightGBM champion (native, 1 thread)", "LightGBM", "native", FEATURES,
        lambda x: champ.predict(x, num_threads=1), lambda x: champ.predict(x), kb(champ_path), pv_c, pt_c, n_tree,
        {"model": str(champ_path), "runtime": "lightgbm"}, booster=champ)
    onnx_path = MODELS / "lgbm_champion.onnx"
    to_onnx(champ, len(FEATURES), onnx_path)
    f1, fb = onnx_session(onnx_path, 1), onnx_session(onnx_path, os.cpu_count())
    add("LightGBM champion (ONNX Runtime)", "LightGBM", "onnx", FEATURES, f1, fb, kb(onnx_path), f1(Xv), f1(Xt),
        n_tree, {"model": str(champ_path), "onnx": str(onnx_path), "runtime": "onnx"}, booster=champ)
    onnx_parity = float(np.max(np.abs(f1(Xt) - pt_c)))

    # --- tree budget
    params = json.loads((MODELS / "lgbm_champion_params.json").read_text())
    for k in [b for b in CFG["research"]["tree_budgets"] if b < n_tree]:
        pth = MODELS / f"lgbm_trees{k}.txt"
        champ.save_model(pth, num_iteration=k)
        b = lgb.Booster(model_file=str(pth))
        op = MODELS / f"lgbm_trees{k}.onnx"
        to_onnx(b, len(FEATURES), op)
        f1 = onnx_session(op, 1)
        add(f"LightGBM {k} trees (ONNX)", "LightGBM", "onnx", FEATURES, f1, onnx_session(op, os.cpu_count()),
            kb(op), b.predict(Xv), b.predict(Xt), k, {"model": str(pth), "onnx": str(op), "runtime": "onnx"}, booster=b)

    # --- feature budget (SHAP top-k, retrained)
    imp = pd.read_csv(P.results / "e7_shap_importance.csv")
    for k in CFG["research"]["topk_features"]:
        feats = imp.feature.head(k).tolist()
        dtr = lgb.Dataset(tr[feats], tr.is_fraud, params={"feature_pre_filter": False})
        dva = lgb.Dataset(va[feats], va.is_fraud, reference=dtr)
        b = lgb.train({**params, "metric": "average_precision"}, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        pth = MODELS / f"lgbm_top{k}.txt"
        b.save_model(pth, num_iteration=b.best_iteration)
        b = lgb.Booster(model_file=str(pth))
        op = MODELS / f"lgbm_top{k}.onnx"
        to_onnx(b, k, op)
        f1 = onnx_session(op, 1)
        Xv_k, Xt_k = va[feats].to_numpy(np.float32), te[feats].to_numpy(np.float32)
        add(f"LightGBM top-{k} features (ONNX)", "LightGBM", "onnx", feats, f1, onnx_session(op, os.cpu_count()),
            kb(op), f1(Xv_k), f1(Xt_k), b.num_trees(), {"model": str(pth), "onnx": str(op), "runtime": "onnx"}, booster=b)

    res = pd.DataFrame(rows)
    # Pareto front: no other variant is both faster (p99) and more accurate (valid AUPRC)
    res["pareto"] = [not ((res.p99_ms <= r.p99_ms) & (res.valid_auprc >= r.valid_auprc) &
                          ((res.p99_ms < r.p99_ms) | (res.valid_auprc > r.valid_auprc))).any() for r in res.itertuples()]
    res.to_csv(P.results / "e9_variants.csv", index=False)
    return res, registry, onnx_parity


def select_production(res: pd.DataFrame) -> pd.Series:
    """Max valid AUPRC under the model-latency SLO; within 0.002 AUPRC prefer the faster one."""
    slo = CFG["slo"]["p99_model_ms"]
    ok = res[(res.p99_ms <= slo) & (res.family == "LightGBM") & (res.runtime.isin(["onnx", "native"]))]
    best = ok.valid_auprc.max()
    near = ok[ok.valid_auprc >= best - 0.002]
    return near.sort_values("p99_ms").iloc[0]


def build_bundle(choice: pd.Series, reg: dict, tr, va, te) -> dict:
    import shutil
    info = reg[choice.variant]
    feats = info["features"]
    out = P.production
    for f in out.glob("*"):
        f.unlink()
    shutil.copy(info["model"], out / "model.txt")
    runtime = info.get("runtime", "lightgbm")
    if runtime == "onnx":
        shutil.copy(info["onnx"], out / "model.onnx")
    b = lgb.Booster(model_file=str(out / "model.txt"))
    pv, pt = b.predict(va[feats]), b.predict(te[feats])
    cal = fit_calibrator(pv, va.is_fraud.to_numpy())
    pv_c, pt_c = calibrate(pv, cal), calibrate(pt, cal)
    policy, _ = fit_policy(pv_c, va, CFG)
    test_eval = evaluate(te, pt_c, CFG, threshold=policy["t_review"])
    prof = build_profile(tr[feats].to_numpy(np.float32), feats)
    # score drift on the RAW score: calibrated isotonic plateaus are ~1e-11 apart -> fragile bin edges
    prof["__score__"] = build_profile(pv.reshape(-1, 1), ["s"])["s"]
    version = f"lgbm-{len(feats)}f-{b.num_trees()}t-{runtime}-{datetime.now():%Y%m%d}"
    meta = {"model_version": version, "variant": choice.variant, "runtime": runtime, "features": feats,
            "n_trees": b.num_trees(), "calibration": cal, "policy": {k: v for k, v in policy.items()},
            "valid_auprc": float(choice.valid_auprc), "test": test_eval,
            "latency_model_ms": {k: float(choice[k]) for k in ("p50_ms", "p95_ms", "p99_ms")},
            "slo": CFG["slo"], "created_at": datetime.now().isoformat(timespec="seconds")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    (out / "drift_profile.json").write_text(json.dumps(prof))
    log(f"production bundle -> {version} | test AUPRC={test_eval['auprc']:.4f} recall@review="
        f"{test_eval['biz_recall']:.3f} precision@review={test_eval['biz_precision']:.3f}")
    return meta


# ----------------------------------------------------------------------------- E10
def e10_end_to_end(n: int = 4000):
    from .serving.scorer import Scorer
    log(f"E10 end-to-end latency over {n} streamed test transactions")
    raw = pd.read_parquet(P.raw / "transactions.parquet")
    stream = raw[raw.day >= CFG["split"]["test"][0]].head(n).rename(columns={"tx_lat": "lat", "tx_lon": "lon"})
    txs = stream.to_dict("records")
    feats_off = load_features(CFG).set_index("tx_id").loc[stream.tx_id, FEATURES].to_numpy(np.float32)

    import shap
    rows, parity = [], None
    for mode in ("always_shap_lib", "always_native", "on_alert_native"):
        sc = Scorer(explain="never" if mode == "always_shap_lib" else ("always" if mode == "always_native" else "on_alert"))
        explainer = shap.TreeExplainer(sc.booster) if mode == "always_shap_lib" else None
        lat, online = [], []
        for tx in txs:
            r = sc.score(tx)
            l = r["latency_ms"]
            if explainer is not None:
                t0 = time.perf_counter()
                explainer.shap_values(r["_x"])
                l["explain"] = (time.perf_counter() - t0) * 1e3
                l["total"] = l["features"] + l["model"] + l["explain"]
            lat.append(l)
            if mode == "on_alert_native":
                online.append(r)
        L = pd.DataFrame(lat)
        rows.append({"mode": mode, "runtime": sc.runtime, **{f"{c}_mean_ms": L[c].mean() for c in L},
                     "total_p50_ms": L.total.quantile(.5), "total_p95_ms": L.total.quantile(.95),
                     "total_p99_ms": L.total.quantile(.99), "alert_rate": None})
        log(f"   {mode:<18} total p50={rows[-1]['total_p50_ms']:.3f} p99={rows[-1]['total_p99_ms']:.3f} ms "
            f"(features {L.features.mean():.3f} | model {L.model.mean():.3f} | explain {L.explain.mean():.3f})")
        if mode == "on_alert_native":
            rows[-1]["alert_rate"] = float(np.mean([o["decision"] != "APPROVE" for o in online]))
    # parity: online engine (fresh snapshot) vs offline training matrix
    eng = FeatureEngine.load(P.artifacts / "feature_state.pkl")
    on = np.array([eng.transform(t["ts_s"], t["customer_id"], t["terminal_id"], t["channel"], t["category"],
                                 t["amount"], t["lat"], t["lon"], t["device_id"]) for t in txs], dtype=np.float32)
    parity = float(np.max(np.abs(on - feats_off)))
    res = pd.DataFrame(rows)
    res.to_csv(P.results / "e10_end_to_end.csv", index=False)
    log(f"   training/serving parity: max |online - offline| = {parity:.2e}")
    return res, parity


# ----------------------------------------------------------------------------- figures
def figures(res: pd.DataFrame, e2e: pd.DataFrame, choice: pd.Series):
    F = P.figures
    slo = CFG["slo"]["p99_model_ms"]
    # Pareto
    fig, ax = viz.plt.subplots(figsize=(8.4, 5.2))
    other = res[~res.pareto]
    ax.scatter(other.p99_ms, other.valid_auprc, s=46, color=viz.DE_EMPHASIS, edgecolor=viz.SURFACE, linewidth=2,
               label="dominated", zorder=3)
    front = res[res.pareto].sort_values("p99_ms")
    ax.plot(front.p99_ms, front.valid_auprc, color=viz.SERIES[0], lw=2, zorder=2)
    ax.scatter(front.p99_ms, front.valid_auprc, s=56, color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=2,
               label="Pareto frontier", zorder=4)
    ax.scatter([choice.p99_ms], [choice.valid_auprc], s=150, facecolor="none", edgecolor=viz.SERIES[1], linewidth=2,
               label="production choice", zorder=5)
    def short(v):
        return (v.replace("LightGBM ", "LGBM ").replace(" (ONNX Runtime)", " ONNX").replace(" (ONNX)", "")
                 .replace(" (native, 1 thread)", " 1-thr").replace(" (native, all threads)", " mt"))
    for i, r in enumerate(front.itertuples()):
        dy = 14 if i % 2 else -16  # alternate above/below so neighbours never collide
        ax.annotate(short(r.variant), (r.p99_ms, r.valid_auprc), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8, color=viz.INK2,
                    arrowprops=dict(arrowstyle="-", color=viz.AXIS, lw=0.6))
    ax.margins(y=0.08)
    ax.axvline(slo, color=viz.CRITICAL, lw=1)
    ax.text(slo, ax.get_ylim()[0], f"  SLO p99 ≤ {slo} ms", color=viz.CRITICAL, fontsize=8.5, va="bottom")
    ax.set_xscale("log"); ax.set_xlabel("single-row p99 latency (ms, log scale)"); ax.set_ylabel("valid AUPRC")
    ax.set_title("E9 · Accuracy vs. latency trade-off"); ax.legend(loc="lower right")
    viz.save(fig, F / "e9_pareto.png")

    # latency dot plot p50 / p99
    d = res.sort_values("p99_ms")
    fig, ax = viz.plt.subplots(figsize=(8.4, 0.34 * len(d) + 1.4))
    yy = np.arange(len(d))
    ax.hlines(yy, d.p50_ms, d.p99_ms, color=viz.GRID, lw=2)
    ax.scatter(d.p50_ms, yy, color=viz.SERIES[0], s=40, label="p50", zorder=3, edgecolor=viz.SURFACE, linewidth=1.5)
    ax.scatter(d.p99_ms, yy, color=viz.SERIES[1], s=40, label="p99", zorder=3, edgecolor=viz.SURFACE, linewidth=1.5)
    ax.set_yticks(yy, d.variant); ax.invert_yaxis(); ax.set_xscale("log"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("ms per single-row prediction (log scale)"); ax.legend(loc="lower right")
    ax.set_title("E9 · Single-row inference latency by variant")
    viz.save(fig, F / "e9_latency.png")

    # budgets: two separate panels per budget, never dual axis
    tb = res[res.variant.str.contains("trees|champion \\(ONNX", regex=True)].sort_values("n_trees")
    fk = res[res.variant.str.contains("top-|champion \\(ONNX", regex=True)].sort_values("n_features")
    fig, axs = viz.plt.subplots(2, 2, figsize=(9.6, 6))
    for (ax, x, yv, xl, yl) in [(axs[0, 0], tb.n_trees, tb.valid_auprc, "trees", "valid AUPRC"),
                                (axs[0, 1], tb.n_trees, tb.p99_ms, "trees", "p99 ms"),
                                (axs[1, 0], fk.n_features, fk.valid_auprc, "features", "valid AUPRC"),
                                (axs[1, 1], fk.n_features, fk.p99_ms, "features", "p99 ms")]:
        ax.plot(x, yv, marker="o", color=viz.SERIES[0], markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    axs[0, 0].set_title("Tree budget → accuracy"); axs[0, 1].set_title("Tree budget → latency")
    axs[1, 0].set_title("Feature budget → accuracy"); axs[1, 1].set_title("Feature budget → latency")
    viz.save(fig, F / "e9_budgets.png")

    # E10 stacked breakdown (mean ms), 2px surface gaps between segments
    fig, ax = viz.plt.subplots(figsize=(8.4, 2.8))
    left = np.zeros(len(e2e))
    for i, c in enumerate(["features_mean_ms", "model_mean_ms", "explain_mean_ms"]):
        ax.barh(e2e["mode"], e2e[c], left=left, color=viz.SERIES[i], height=0.55, edgecolor=viz.SURFACE,
                linewidth=2, label=c.replace("_mean_ms", ""))
        left += e2e[c].to_numpy()
    for y_, (tot, p99) in enumerate(zip(left, e2e.total_p99_ms)):
        ax.text(tot, y_, f"  mean {tot:.2f} ms · p99 {p99:.2f} ms", va="center", fontsize=8.5, color=viz.INK2)
    ax.set_xlim(0, left.max() * 1.6); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean ms per transaction"); ax.legend(loc="lower right", ncol=3)
    ax.set_title("E10 · End-to-end scoring cost by explanation strategy")
    viz.save(fig, F / "e10_end_to_end.png")


def run():
    s = split(load_features(CFG), CFG)
    tr, va, te = s["train"], s["valid"], s["test"]
    res, reg, onnx_parity = e9_variants(tr, va, te)
    choice = select_production(res)
    log(f"selected: {choice.variant} (valid AUPRC {choice.valid_auprc:.4f}, p99 {choice.p99_ms:.3f} ms)")
    meta = build_bundle(choice, reg, tr, va, te)
    e2e, parity = e10_end_to_end()
    figures(res, e2e, choice)
    prod = e2e[e2e["mode"] == "on_alert_native"].iloc[0]
    meta["latency_e2e_ms"] = {"p50": float(prod.total_p50_ms), "p95": float(prod.total_p95_ms),
                              "p99": float(prod.total_p99_ms)}
    meta["parity_max_abs_diff"] = parity
    meta["onnx_vs_native_max_abs_diff"] = onnx_parity
    (P.production / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    summary = {"selected": choice.to_dict(), "e2e_production": meta["latency_e2e_ms"],
               "slo": CFG["slo"], "slo_met": bool(prod.total_p99_ms <= CFG["slo"]["p99_e2e_ms"]),
               "parity_max_abs_diff": parity, "onnx_vs_native_max_abs_diff": onnx_parity}
    (P.results / "optimize_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"E2E p99 = {prod.total_p99_ms:.3f} ms (SLO {CFG['slo']['p99_e2e_ms']} ms) -> "
        f"{'MET' if summary['slo_met'] else 'MISSED'}")


if __name__ == "__main__":
    run()
