# Dashboard Tableau (alternatif Power BI)

Sumber data sama persis dengan Power BI: view PostgreSQL di `sql/views.sql`.

## Koneksi

1. **Connect → To a Server → PostgreSQL**
   Server `localhost`, Port `5434`, Database `frauddb`, Username/Password `fraud`/`fraud`.
2. Pilih koneksi **Live** (bukan Extract) agar data selalu terbaru.
3. Tarik setiap view ke data source terpisah (`v_kpi_current`, `v_kpi_minute`, `v_alert_queue`,
   `v_scenario_performance`, `v_city_risk`, `v_reason_frequency`, `v_drift_latest`, `v_live_transactions`).
   Tidak perlu join. Setiap view sudah teragregasi.

## Calculated fields (setara measure DAX)

```text
Alert Rate          = SUM([is_alert]) / COUNT([tx_id])
Fraud Caught        = SUM(IIF([is_fraud] = 1 AND [is_alert] = 1, 1, 0))
Precision           = [Fraud Caught] / SUM(IIF(NOT ISNULL([is_fraud]) AND [is_alert] = 1, 1, 0))
Recall              = [Fraud Caught] / SUM(IIF([is_fraud] = 1, 1, 0))
Intercepted (Rp)    = SUM(IIF([is_fraud] = 1 AND [is_alert] = 1, [amount_idr], 0))
Net Savings (Rp)    = [Intercepted (Rp)] - 15000 * SUM([is_alert])
                      - 25000 * SUM(IIF([is_fraud] = 0 AND [is_alert] = 1, 1, 0))
Latency Status      = IIF(MAX([p99_latency_ms_5m]) <= 15, "✓ SLO terpenuhi", "✕ SLO terlewati")
```

## Sheet → Dashboard

| Sheet | Mark | Rows / Columns |
|---|---|---|
| KPI tiles | Text | `v_kpi_current` fields (satu sheet per tile) |
| Tx per minute | Line | Columns `minute_wib` (exact minute), Rows `n_tx` |
| Alerts per minute | Bar (stacked) | `minute_wib` × `n_alert`, `n_block` |
| Alert queue | Text table | `v_alert_queue` |
| Scenario recall | Bar | `scenario` × `recall` |
| Map | Circle (map) | `lon`/`lat` dari `v_city_risk`, Size = `n_tx`, Color = `alert_rate` (sequential biru) |
| Drift | Bar | `feature` × `psi`, Color = `level` (stable hijau `#0ca30c` · moderate kuning `#fab219` · significant merah `#d03b3b`) |

Pakai palet kategorikal `#2a78d6, #eb6834, #1baf7a, #eda100` (urutan tetap) agar konsisten dengan Power BI/Streamlit.

## Realtime refresh

Koneksi *Live* menarik ulang data setiap kali view dirender (F5 / *Refresh data source*).
Untuk refresh otomatis berkala di **Tableau Server/Cloud**, gunakan dashboard extension *auto refresh* dari
Tableau Exchange (interval per detik), atau sematkan dashboard di portal dengan Tableau Embedding API dan
panggil `refreshDataAsync()` dengan timer. Tableau Desktop sendiri tidak punya timer refresh bawaan.
