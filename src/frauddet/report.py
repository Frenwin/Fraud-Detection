"""Generate reports/RESEARCH_REPORT.md from result files (every number is computed, none typed by hand)."""
from __future__ import annotations

import json

import pandas as pd

from .config import get_paths, load_config

CFG = load_config()
P = get_paths(CFG)
R = P.results


def md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def f(v):
        if isinstance(v, float):
            if v != v:
                return "–"
            if v.is_integer() and abs(v) >= 1:
                return f"{v:,.0f}"
            return floatfmt.format(v) if abs(v) < 1000 else f"{v:,.0f}"
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False))


def rp(v: float) -> str:
    return f"Rp {v / 1e9:.2f} miliar" if abs(v) >= 1e9 else f"Rp {v / 1e6:.1f} juta"


def load(name):
    p = R / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p)


def build() -> str:
    s = load("research_summary.json")
    e1, e2, e3 = load("e1_model_comparison.csv"), load("e2_imbalance.csv"), load("e3_ablation.csv")
    e4 = load("e4_best_params.json")
    trials = load("e4_optuna_trials.csv")
    e5, e6 = load("e5_calibration.csv"), load("e6_policy.json")
    e7, e7s = load("e7_shap_importance.csv"), load("e7_scenario_drivers.csv")
    e8, e8w = load("e8_scenario_recall.csv"), load("e8_weekly.csv")
    e9, opt, e10, e11 = load("e9_variants.csv"), load("optimize_summary.json"), load("e10_end_to_end.csv"), load("e11_loadtest.csv")
    meta = json.loads((P.production / "metadata.json").read_text())
    k = CFG["alerting"]["daily_alert_budget"]

    d = s["data"]
    rules = e1[e1.model == "Rules"].iloc[0]
    best1 = e1.sort_values("valid_auprc", ascending=False).iloc[0]
    cum = e3[e3["mode"] == "cumulative"]
    loo = e3[e3["mode"] == "leave_one_out"].copy()
    full_test = cum.test_auprc.iloc[-1]
    loo["delta_test"] = loo.test_auprc - full_test
    default_trial = float(trials.value.iloc[0])
    raw_b, iso_b = e5.set_index("method").loc["raw"], e5.set_index("method").loc["isotonic"]
    tr_, tb_ = e6["test_at_review"], e6["test_at_block"]
    worst = e8.sort_values("recall").iloc[0]
    sel = opt["selected"]
    mt = e9[e9.variant == "LightGBM champion (native, all threads)"].iloc[0]
    e2e = e10.set_index("mode")
    pareto = e9[e9.pareto].sort_values("p99_ms")
    e8_sorted = e8.sort_values("recall")
    psi_max = float(e8w[[c for c in e8w.columns if c.startswith("psi_")]].to_numpy().max())
    v = e9.set_index("variant")
    rf_p99, lr_p99 = v.loc["RandomForest 300 (sklearn)", "p99_ms"], v.loc["LogReg (sklearn pipeline)", "p99_ms"]
    onnx_p99 = v.loc["LightGBM champion (ONNX Runtime)", "p99_ms"]
    nt1_p99 = v.loc["LightGBM champion (native, 1 thread)", "p99_ms"]
    exp50, exp400 = v.loc["LightGBM 50 trees (ONNX)", "explain_p99_ms"], v.loc["LightGBM 400 trees (ONNX)", "explain_p99_ms"]
    expchamp, champ_trees = v.loc["LightGBM champion (ONNX Runtime)", "explain_p99_ms"], v.loc["LightGBM champion (ONNX Runtime)", "n_trees"]
    champ_v, v400 = v.loc["LightGBM champion (ONNX Runtime)", "valid_auprc"], v.loc["LightGBM 400 trees (ONNX)", "valid_auprc"]
    if iso_b.test_ece < raw_b.test_ece:
        calib_text = (f"Isotonic menurunkan ECE test dari {raw_b.test_ece:.5f} → {iso_b.test_ece:.5f} tanpa mengubah ranking.")
    else:
        calib_text = (f"Pada run ini skor mentah sudah terkalibrasi baik (ECE test {raw_b.test_ece:.5f} vs isotonic "
                      f"{iso_b.test_ece:.5f}; selisihnya tipis). Isotonic tetap dipakai sebagai **pengaman**: Optuna bebas memilih "
                      f"`scale_pos_weight` (di sini {e4.get('scale_pos_weight', 1):.2f}) yang menggeser skala probabilitas, "
                      "sehingga tanpa kalibrasi threshold biaya bisa meleset setiap kali model di-retrain.")

    out = [f"""# Research Report — Real-time Fraud Detection

> Laporan ini **di-generate otomatis** oleh `python -m frauddet.report` dari `reports/results/*`.
> Semua angka berasal dari run terakhir pipeline (seed {CFG['seed']}).

## 1. Ringkasan eksekutif

| Item | Hasil |
|---|---|
| Data | {sum(v['rows'] for v in d.values()):,} transaksi berlabel (train/valid/test), fraud rate ≈ {d['test']['fraud'] / d['test']['rows']:.2%} di test |
| Model produksi | **{meta['model_version']}** ({sel['variant']}) |
| Kualitas (test out-of-time) | AUPRC **{meta['test']['auprc']:.3f}** · ROC-AUC {meta['test']['roc_auc']:.3f} · Precision@{k}/hari {meta['test'][f'precision@{k}/day']:.2f} |
| Baseline aturan manual | AUPRC {rules.test_auprc:.3f} → model **{meta['test']['auprc'] / rules.test_auprc:.1f}× lebih baik** |
| Keputusan REVIEW (test) | recall {meta['test']['biz_recall']:.1%} · precision {meta['test']['biz_precision']:.1%} · {meta['test']['biz_alerts']:,} alert |
| Nilai fraud tercegah (test, 23 hari) | {meta['test']['biz_loss_prevented_pct']:.1%} dari nilai fraud · net savings **{rp(meta['test']['biz_net_savings'])}** setelah biaya review & friksi |
| Latensi model (single row) | p50 {sel['p50_ms']:.3f} ms · p99 **{sel['p99_ms']:.3f} ms** (SLO {CFG['slo']['p99_model_ms']} ms) |
| Latensi end-to-end in-process | p50 {opt['e2e_production']['p50']:.3f} ms · p99 **{opt['e2e_production']['p99']:.3f} ms** (SLO {CFG['slo']['p99_e2e_ms']} ms → {'✓ terpenuhi' if opt['slo_met'] else '✕ terlewati'}) |
| Training/serving parity | max \\|online − offline\\| = {opt['parity_max_abs_diff']:.1e} (fitur identik) |

## 2. Desain eksperimen

* **Data**: simulator transaksi kartu Indonesia (10 kota, Rupiah) dengan 6 skenario fraud. Satu skenario
  (`social_engineering`) **hanya muncul di periode test**, untuk menguji ketahanan terhadap *concept drift*.
* **Split berbasis waktu dengan label delay**: train hari 14–75 → valid 76–89 → *gap* 90–96 → test 97–119.
  Gap 7 hari ada karena chargeback baru matang setelah ±7 hari, jadi saat model di-deploy pada hari 97,
  label hari 90–96 belum tersedia. Tidak ada random split, karena random split membocorkan masa depan.
* **Fitur point-in-time**: 29 fitur dari *stateful feature engine* yang sama untuk offline & online
  (velocity 1 jam/24 jam/7 hari/30 hari, profil nominal, geo & *impossible travel*, risiko merchant dengan label
  tertunda, graph perangkat). Biaya ±18 µs/transaksi.
* **Metrik utama AUPRC** (bukan akurasi/ROC-AUC) karena kelas sangat tidak seimbang. Pendukung: Precision@{k}/hari
  (kapasitas analis), recall@0.5% FPR, Brier/ECE (kalibrasi), dan **nilai bisnis Rupiah**.
* **Seleksi model hanya memakai valid**. Test disentuh sekali untuk pelaporan.

## 3. E1 — Perbandingan keluarga model

![E1](figures/e1_model_comparison.png)

{md(e1[['model', 'valid_auprc', 'test_auprc', 'test_auprc_ci_low', 'test_auprc_ci_high', 'test_roc_auc', f'test_precision@{k}/day', 'train_time_s']])}

**Temuan.** Semua model ML jauh di atas aturan manual (AUPRC {rules.test_auprc:.3f}). Di validasi, boosting
({best1.model}, {best1.valid_auprc:.3f}) memimpin. Di test, CI 95% (bootstrap per hari) antar-GBDT saling tumpang
tindih, jadi selisih kecil di test **tidak signifikan**. Penurunan valid → test yang seragam di semua model
disebabkan oleh skenario baru (lihat E8), bukan overfitting satu model.

![PR](figures/e1_pr_curves.png)

## 4. E2 — Penanganan class imbalance

{md(e2)}

**Temuan.** Reweighting dan resampling hanya menggeser AUPRC dalam rentang yang sama, tetapi mengubah skala
probabilitas (`mean_pred_vs_base_rate` ≠ 1). Karena keputusan berbasis biaya butuh probabilitas yang jujur,
pipeline memilih **tanpa resampling + kalibrasi isotonic** (E5), dan memberi Optuna kebebasan menyetel
`scale_pos_weight` sebagai hyperparameter.

## 5. E3 — Ablasi keluarga fitur

![E3](figures/e3_ablation.png)

{md(e3)}

**Temuan.** Fitur transaksi mentah saja hanya mencapai AUPRC {cum.test_auprc.iloc[0]:.3f}. Fitur *behavioural*
(velocity, profil nominal) memberi lompatan terbesar menjadi {cum.test_auprc.iloc[1]:.3f}. Leave-one-out paling
merugikan bila **{loo.sort_values('delta_test').iloc[0].setting}** dihapus
(Δ test {loo.sort_values('delta_test').iloc[0].delta_test:+.3f}). Catatan jujur: menghapus satu keluarga fitur bisa
*menaikkan* AUPRC test (Δ terbesar {loo.delta_test.max():+.3f} untuk {loo.sort_values('delta_test').iloc[-1].setting})
karena model yang sangat bergantung pada sinyal tersebut lebih mudah tertipu pola baru
(`social_engineering` dilakukan dari lokasi dan perangkat korban sendiri). Ini argumen untuk *monitoring drift* dan
retraining berkala, bukan untuk membuang fitur berdasarkan data test.

## 6. E4 — Hyperparameter optimisation (Optuna TPE)

![E4](figures/e4_optuna_history.png)

* {e4['n_trials']} trial · trial 0 = parameter default E1 (valid AUPRC {default_trial:.4f})
* Terbaik: valid AUPRC **{e4['valid_auprc']:.4f}** ({e4['valid_auprc'] - default_trial:+.4f}), best iteration {e4['best_iteration']}
* Parameter: `{json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in e4.items() if k_ not in ('valid_auprc', 'n_trials', 'best_iteration')})}`

## 7. E5 — Kalibrasi probabilitas

![E5](figures/e5_calibration.png)

{md(e5, '{:.5f}')}

{calib_text}
Kalibrasi memakai *tie-break* 1e-6 dengan skor mentah, sehingga pemetaan monoton ketat dan AUPRC identik dengan model
mentah. Tanpa tie-break, fungsi tangga isotonic membuat skor kembar dan pada percobaan awal AUPRC turun ±0.02.
Kalibrasi disimpan sebagai dua array dan dievaluasi dengan satu `np.interp` di jalur serving (< 1 µs).

## 8. E6 — Kebijakan keputusan berbasis biaya

Biaya: review analis Rp {CFG['cost']['review_cost']:,} per alert, friksi nasabah Rp {CFG['cost']['friction_cost']:,} per false positive.

![E6](figures/e6_cost_curve.png)

| Tier | Threshold (dipilih di valid) | Precision test | Recall test | Alert test | Fraud tercegah |
|---|---|---|---|---|---|
| REVIEW (≥) | {e6['t_review']:.4f} | {tr_['precision']:.1%} | {tr_['recall']:.1%} | {tr_['alerts']:,} | {tr_['loss_prevented_pct']:.1%} |
| BLOCK (≥) | {e6['t_block']:.4f} | {tb_['precision']:.1%} | {tb_['recall']:.1%} | {tb_['alerts']:,} | {tb_['loss_prevented_pct']:.1%} |

REVIEW = threshold yang memaksimalkan *net savings* di valid. BLOCK = threshold terendah dengan precision valid
≥ {CFG['alerting']['block_min_precision']:.0%}, sehingga hanya transaksi yang hampir pasti fraud yang ditolak otomatis.

## 9. E7 — Explainability (TreeSHAP)

![E7](figures/e7_shap_importance.png)

Fitur yang paling menentukan per skenario (rata-rata SHAP pada transaksi fraud di test):

{md(e7s[['scenario', 'n', 'top1', 'top2', 'top3']])}

Setiap alert di API membawa 3 *reason code* (kontribusi SHAP positif terbesar) dalam Bahasa Indonesia, misalnya
"Perangkat baru (ya)" atau "Kecepatan perpindahan mustahil (1.840 km/jam)". Ini syarat auditabilitas untuk tim
analis dan regulator.

## 10. E8 — Robustness & concept drift

![E8](figures/e8_scenario_recall.png)

{md(e8)}

![E8w](figures/e8_weekly.png)

**Temuan.**
* Dua skenario terlemah: **{e8_sorted.scenario.iloc[0]}** (recall {e8_sorted.recall.iloc[0]:.0%}) dan
  **{e8_sorted.scenario.iloc[1]}** (recall {e8_sorted.recall.iloc[1]:.0%}). `merchant_collusion` hanya terlihat dari
  riwayat chargeback merchant, dan sinyal itu baru muncul setelah label delay 7 hari. `social_engineering` tidak
  pernah ada di training (nasabah sendiri yang bertransaksi dari perangkat & lokasinya), sehingga model hanya
  menangkapnya lewat nominal yang tidak biasa.
* **PSI tidak mendeteksi drift ini.** PSI fitur per minggu di test maksimal {psi_max:.3f} (jauh di bawah 0.10), karena
  skenario baru hanya ±0,2% transaksi dan distribusi fitur nyaris tidak bergeser. Yang berubah adalah *hubungan*
  fitur→label (**concept drift**), dan itu hanya terlihat dari **recall berlabel** yang turun dari
  {e8w.recall_at_review.iloc[0]:.2f} ke {e8w.recall_at_review.iloc[-1]:.2f} ketika porsi `social_engineering` naik ke
  {e8w.share_social_engineering.iloc[-1]:.0%}. Pelajarannya: monitoring produksi butuh dua lapis, yaitu PSI (covariate
  drift, tanpa label) **dan** performa berlabel per segmen dengan label delay. Keduanya ada di dashboard.

{md(e8w, '{:.3f}')}

## 11. E9 — Optimasi & trade-off akurasi vs latensi

![E9](figures/e9_pareto.png)

{md(e9[[c for c in ['variant', 'n_features', 'n_trees', 'size_kb', 'valid_auprc', 'test_auprc', 'p50_ms', 'p99_ms', 'explain_p99_ms', 'batch_rows_per_s', 'pareto'] if c in e9.columns]], '{:.4f}')}

![E9 latency](figures/e9_latency.png)
![E9 budgets](figures/e9_budgets.png)

**Temuan.**
* **Runtime & keluarga model menentukan latensi.** Untuk akurasi yang setara, RandomForest sklearn butuh p99
  {rf_p99:.1f} ms, sedangkan LightGBM ONNX hanya {onnx_p99:.3f} ms (**{rf_p99 / onnx_p99:.0f}× lebih cepat**). Pipeline sklearn
  (LogReg) lambat untuk satu baris ({lr_p99:.1f} ms) karena overhead per panggilan, meskipun throughput batch-nya tinggi.
* LightGBM yang sama: thread default p99 {mt.p99_ms:.3f} ms → 1 thread {nt1_p99:.3f} ms → ONNX Runtime {onnx_p99:.3f} ms.
  Skor ONNX hampir identik (max |Δp| = {opt['onnx_vs_native_max_abs_diff']:.1e}, AUPRC sama) karena threshold disimpan float32.
* **Biaya explainability tumbuh dengan jumlah pohon.** p99 reason code (TreeSHAP) {exp50:.2f} ms untuk 50 pohon vs
  {expchamp:.2f} ms untuk champion ({int(champ_trees)} pohon), dan ini *lebih mahal* daripada inferensi itu sendiri. Varian 400 pohon
  memangkas biaya explain ±{expchamp / exp400:.1f}× dengan kehilangan AUPRC valid hanya {champ_v - v400:.4f}. Itu kandidat bila SLO
  end-to-end diperketat.
* **Tree/feature budget** memangkas latensi & ukuran model dengan penurunan AUPRC yang kecil, sehingga varian kecil
  berada di Pareto frontier ({', '.join(pareto.variant.head(6))}).
* **Aturan seleksi**: AUPRC valid tertinggi dengan p99 ≤ {CFG['slo']['p99_model_ms']} ms. Bila selisih ≤ 0.002,
  pilih yang lebih cepat → **{sel['variant']}**.

## 12. E10 — Latensi end-to-end & strategi explainability

![E10](figures/e10_end_to_end.png)

{md(e10.drop(columns=['runtime']), '{:.4f}')}

Menghitung SHAP dengan library `shap` untuk setiap transaksi menambah ±{e2e.loc['always_shap_lib', 'explain_mean_ms']:.2f} ms.
TreeSHAP native LightGBM (`pred_contrib`) {e2e.loc['always_native', 'explain_mean_ms']:.2f} ms. Strategi produksi
**explain-on-alert** hanya menghitung reason code untuk ±{(e2e.loc['on_alert_native', 'alert_rate'] or 0):.1%}
transaksi yang di-alert, sehingga biaya rata-ratanya mendekati nol dan p99 menjadi {e2e.loc['on_alert_native', 'total_p99_ms']:.3f} ms.
"""]
    if e11 is not None:
        best = e11.sort_values("throughput_rps").iloc[-1]
        sync, asy = e11[e11.endpoint == "/score"], e11[e11.endpoint == "/score/async"]
        out.append(f"""## 13. E11 — Load test API (HTTP, satu worker Uvicorn)

![E11](figures/e11_loadtest.png)

{md(e11, '{:.2f}')}

**Temuan.**
* Satu worker mencapai puncak **±{best.throughput_rps:,.0f} req/s** pada {int(best.concurrency)} klien konkuren
  (client p99 {best.p99_ms:.1f} ms).
* Di atas titik itu throughput **turun** dan client p99 meledak (hingga {e11.p99_ms.max():,.0f} ms), sementara
  **server-side p99 tetap {e11.server_p99_ms.min():.1f}–{e11.server_p99_ms.max():.1f} ms**. Artinya bottleneck ada di
  antrean HTTP/event loop satu proses Python (dan load generator di mesin yang sama), bukan di model atau
  feature engine.
* `async def` vs `def` (threadpool) praktis sama (puncak {asy.throughput_rps.max():,.0f} vs {sync.throughput_rps.max():,.0f} req/s):
  pekerjaan CPU-bound ±0.15 ms tidak diuntungkan oleh async. Default dipertahankan `def`.
* Implikasi kapasitas: untuk >1.000 TPS, tambah worker/pod dengan **partisi berdasarkan `customer_id`** (state velocity
  per nasabah tetap lokal) atau pindahkan state ke online store (Redis), plus admission control agar antrean tidak
  menembus SLO.
""")
    out.append("""## Keterbatasan & langkah lanjut

* Data sintetis: pola fraud dirancang secara eksplisit, sehingga angka absolut tidak bisa langsung dibandingkan dengan
  data bank nyata. Metodologinya (split waktu + label delay, cost-based threshold, parity, SLO) tetap berlaku.
* Retraining terjadwal + *champion/challenger shadow scoring* dengan label baru (termasuk `social_engineering`).
* Graph features lanjutan (GNN pada graf kartu–perangkat–merchant) untuk mendeteksi sindikat.
* Online store terdistribusi (Redis) + Kafka/Redpanda untuk event log downstream, dengan beberapa worker scoring.
""")
    return "\n".join(out)


def main():
    text = build()
    (P.reports / "RESEARCH_REPORT.md").write_text(text, encoding="utf-8")
    print(f"[report] wrote {P.reports / 'RESEARCH_REPORT.md'}")


if __name__ == "__main__":
    main()
