# Dashboard Power BI — Real-time Fraud Operations

![Live Operations](../../docs/img/powerbi_live_operations.png)

| Business Impact | Model Health | Research & Trade-offs |
|---|---|---|
| ![](../../docs/img/powerbi_business_impact.png) | ![](../../docs/img/powerbi_model_health.png) | ![](../../docs/img/powerbi_research.png) |

## Cara tercepat: buka file yang sudah jadi

Dashboard ini disimpan sebagai **Power BI Project (PBIP)**: semantic model dalam TMDL dan report dalam PBIR
(format teks berbasis JSON schema resmi Microsoft). Semuanya di-generate dari kode oleh [`build_pbip.py`](build_pbip.py),
jadi isinya bisa di-review dan di-diff di Git.

1. Jalankan database + stream (lihat README utama: `.\scripts\run_live.ps1`)
2. Buka **`dashboard/powerbi/FraudOps.pbip`** di Power BI Desktop (versi Sept 2026 ke atas)
3. Saat diminta kredensial PostgreSQL: user `fraud`, password `fraud` → **Connect**. Ini hanya sekali.

| Halaman | Isi | Refresh |
|---|---|---|
| **Real-time Fraud Operations** | 7 KPI (TPS, alert/block rate, p99 vs SLO, fraud dicegah, precision, recall), tren per menit, antrean analis + reason code, recall per skenario | 3 dtk |
| **Business Impact** | nilai fraud terjadi/dicegah/lolos (Rp), net savings, false positives, nilai per skenario, top reason code, alert rate per kategori, top lokasi, kurva biaya riset | 5 dtk |
| **Model Health** | gauge p99 vs SLO, p50, status SLO, label coverage, versi model, tren latensi, histogram latensi, top-10 PSI drift, komposisi label per bucket skor (kalibrasi live) | 5 dtk |
| **Research & Trade-offs** | E1 model zoo, E9 Pareto akurasi–latensi, E10 biaya explainability, E11 load test | statis |

**Desain.** Header navy, kartu putih di atas kanvas abu-abu kebiruan, dan palet kategorikal yang sudah divalidasi aman
untuk buta warna. Warna status (hijau/kuning/merah) hanya dipakai bila nilainya punya arti, dan selalu disertai label
atau judul yang menjelaskan aturannya: p99 vs SLO 15 ms, recall < 50%, PSI 0.10/0.25, BLOCK/REVIEW. Nominal tampil
dalam format Rupiah ("Rp 755,4 jt"). Kanvas 1280×800 (16:10).

**Mengubah dashboard.** Edit `build_pbip.py`, lalu jalankan `python dashboard/powerbi/build_pbip.py` (butuh container
PostgreSQL karena tipe kolom dibaca dari database). Setelah itu buka ulang file `.pbip`.

> **Catatan teknis yang ditemukan saat membangun (Power BI Desktop 2.159):**
> - Auto page refresh di PBIR memakai `refreshType: 0D` dan **durasi ISO-8601** `duration: 'PT3S'`. Angka biasa
>   (`3D`) diterima tanpa error tetapi diabaikan. Format ini didapat dengan mengaktifkan refresh sekali di Desktop lalu
>   membaca `page.json` yang disimpan. Verifikasi: Postgres menerima query DirectQuery dari *Mashup Engine* tiap ±3–4 detik.
> - Theme kustom (RegisteredResources) yang di-generate di luar Desktop dibuang diam-diam, walaupun lolos
>   `reportThemeSchema-2.159`. Karena itu semua styling ditulis eksplisit per visual. `fraud_ops_theme.json` tetap
>   disediakan untuk impor manual (*View → Themes → Browse*) bila ingin menambah visual baru dengan gaya yang sama.
> - `axisScale: log` tidak diterapkan pada visual yang di-generate, sehingga sumbu log di-precompute di view
>   (`v_e9_plot.log10_p99_ms`). `MAX()` pada kolom boolean gagal di DAX, jadi diganti `SELECTEDVALUE`.
> - Desktop me-render dengan WebView2. Jendela yang tertutup jendela lain tidak di-repaint, jadi angka baru terlihat
>   saat jendela Power BI aktif.
