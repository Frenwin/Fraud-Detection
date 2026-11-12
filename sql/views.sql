-- =============================================================
-- BI semantic layer — one view per dashboard visual.
-- Power BI: connect in DirectQuery mode + Automatic page refresh.
-- All timestamps exposed as WIB (Asia/Jakarta) local time.
-- Numeric results are cast to double precision so BI tools get a stable data type.
-- =============================================================

DROP VIEW IF EXISTS v_live_transactions, v_kpi_minute, v_kpi_current, v_alert_queue, v_scenario_performance,
    v_city_risk, v_reason_frequency, v_latency_histogram, v_score_distribution, v_drift_latest, v_drift_history,
    v_model_registry, v_drift_top CASCADE;

-- Live fact table (rolling 3 hours keeps DirectQuery fast)
CREATE OR REPLACE VIEW v_live_transactions AS
SELECT tx_id,
       (scored_at AT TIME ZONE 'Asia/Jakarta')                    AS scored_at_wib,
       date_trunc('minute', scored_at AT TIME ZONE 'Asia/Jakarta') AS minute_wib,
       event_ts, customer_id, terminal_id, channel, category, city, lat, lon,
       amount::double precision                                     AS amount_idr,
       fraud_score, decision,
       (decision <> 'APPROVE')::int                                 AS is_alert,
       (decision = 'BLOCK')::int                                    AS is_block,
       reason_1, reason_2, reason_3, latency_ms, feature_ms, model_ms, explain_ms,
       is_fraud, scenario,
       CASE WHEN is_fraud IS NULL THEN 'Pending'
            WHEN is_fraud = 1 AND decision <> 'APPROVE' THEN 'True positive'
            WHEN is_fraud = 1 THEN 'Missed fraud'
            WHEN decision <> 'APPROVE' THEN 'False positive'
            ELSE 'True negative' END                                AS outcome,
       model_version
FROM scored_transactions
WHERE scored_at >= now() - interval '3 hours';

-- Per-minute KPIs for line charts
CREATE OR REPLACE VIEW v_kpi_minute AS
SELECT date_trunc('minute', scored_at AT TIME ZONE 'Asia/Jakarta')                AS minute_wib,
       count(*)                                                                    AS n_tx,
       count(*) / 60.0::double precision                                           AS tps,
       sum((decision <> 'APPROVE')::int)                                           AS n_alert,
       sum((decision = 'REVIEW')::int)                                             AS n_review,
       sum((decision = 'BLOCK')::int)                                              AS n_block,
       sum(amount)::double precision                                               AS amount_idr,
       sum(CASE WHEN decision = 'BLOCK' THEN amount ELSE 0 END)::double precision  AS amount_blocked_idr,
       avg(fraud_score)                                                            AS avg_score,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)                    AS p50_latency_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)                    AS p95_latency_ms,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms)                    AS p99_latency_ms,
       sum(COALESCE(is_fraud, 0))                                                  AS n_fraud_labeled,
       sum(CASE WHEN is_fraud = 1 AND decision <> 'APPROVE' THEN 1 ELSE 0 END)     AS n_fraud_caught
FROM scored_transactions
WHERE scored_at >= now() - interval '3 hours'
GROUP BY 1;

-- Single-row "headline" KPIs (cards). Window = last 5 minutes for rates, session for totals.
CREATE OR REPLACE VIEW v_kpi_current AS
WITH w AS (SELECT * FROM scored_transactions WHERE scored_at >= now() - interval '5 minutes'),
     s AS (SELECT * FROM scored_transactions WHERE scored_at >= now() - interval '3 hours')
SELECT (SELECT count(*) FROM w) / 300.0::double precision                                       AS tps_5m,
       (SELECT avg((decision <> 'APPROVE')::int)::double precision FROM w)                      AS alert_rate_5m,
       (SELECT avg((decision = 'BLOCK')::int)::double precision FROM w)                         AS block_rate_5m,
       (SELECT percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms) FROM w)                 AS p50_latency_ms_5m,
       (SELECT percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) FROM w)                 AS p99_latency_ms_5m,
       (SELECT count(*) FROM s)                                                                 AS n_tx,
       (SELECT sum(amount) FROM s)::double precision                                            AS amount_idr,
       (SELECT sum(CASE WHEN is_fraud = 1 AND decision <> 'APPROVE' THEN amount ELSE 0 END) FROM s)::double precision
                                                                                                AS fraud_amount_intercepted_idr,
       (SELECT sum(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) FROM s)::double precision     AS fraud_amount_total_idr,
       (SELECT sum(CASE WHEN is_fraud = 1 AND decision <> 'APPROVE' THEN 1 ELSE 0 END)::double precision
               / NULLIF(sum(CASE WHEN decision <> 'APPROVE' AND is_fraud IS NOT NULL THEN 1 ELSE 0 END), 0) FROM s)
                                                                                                AS precision_labeled,
       (SELECT sum(CASE WHEN is_fraud = 1 AND decision <> 'APPROVE' THEN 1 ELSE 0 END)::double precision
               / NULLIF(sum(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END), 0) FROM s)               AS recall_labeled,
       (SELECT max(model_version) FROM w)                                                       AS model_version;

-- Analyst queue: latest alerts with reason codes
CREATE OR REPLACE VIEW v_alert_queue AS
SELECT tx_id, (scored_at AT TIME ZONE 'Asia/Jakarta') AS scored_at_wib, customer_id, channel, category, city,
       amount::double precision AS amount_idr, fraud_score, decision, reason_1, reason_2, reason_3,
       CASE WHEN is_fraud IS NULL THEN 'Pending' WHEN is_fraud = 1 THEN 'Confirmed fraud' ELSE 'Legit' END AS label_status,
       scenario
FROM scored_transactions
WHERE decision <> 'APPROVE' AND scored_at >= now() - interval '3 hours'
ORDER BY scored_at DESC
LIMIT 300;

-- Detection quality per fraud scenario (uses delayed labels)
CREATE OR REPLACE VIEW v_scenario_performance AS
SELECT scenario,
       count(*)                                                   AS n_fraud,
       sum((decision <> 'APPROVE')::int)                          AS n_caught,
       avg((decision <> 'APPROVE')::int)::double precision        AS recall,
       sum(amount)::double precision                              AS amount_idr,
       sum(CASE WHEN decision <> 'APPROVE' THEN amount ELSE 0 END)::double precision AS amount_caught_idr
FROM scored_transactions
WHERE is_fraud = 1 AND scored_at >= now() - interval '3 hours'
GROUP BY scenario;

-- Geography: top 10 locations by alert volume
CREATE OR REPLACE VIEW v_city_risk AS
SELECT city, avg(lat) AS lat, avg(lon) AS lon, count(*) AS n_tx,
       sum((decision <> 'APPROVE')::int) AS n_alert,
       avg((decision <> 'APPROVE')::int)::double precision AS alert_rate,
       sum(COALESCE(is_fraud, 0)) AS n_fraud,
       sum(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END)::double precision AS fraud_amount_idr
FROM scored_transactions
WHERE scored_at >= now() - interval '3 hours'
GROUP BY city
ORDER BY n_alert DESC
LIMIT 10;

-- Which reasons drive alerts (top 8)
CREATE OR REPLACE VIEW v_reason_frequency AS
SELECT split_part(r, ' (', 1) AS reason, count(*) AS n_alerts
FROM (SELECT unnest(ARRAY[reason_1, reason_2, reason_3]) AS r
      FROM scored_transactions
      WHERE decision <> 'APPROVE' AND scored_at >= now() - interval '3 hours') x
WHERE r IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC
LIMIT 8;

-- Latency distribution (histogram, 0.25 ms buckets up to 10 ms)
CREATE OR REPLACE VIEW v_latency_histogram AS
SELECT LEAST(width_bucket(latency_ms, 0, 10, 40), 41) AS bucket,
       ((LEAST(width_bucket(latency_ms, 0, 10, 40), 41) - 1) * 0.25)::double precision AS bucket_start_ms,
       count(*) AS n
FROM scored_transactions
WHERE scored_at >= now() - interval '15 minutes'
GROUP BY 1;

-- Score distribution by outcome (for model-health page)
CREATE OR REPLACE VIEW v_score_distribution AS
SELECT width_bucket(fraud_score, 0, 1, 20) AS bucket,
       ((width_bucket(fraud_score, 0, 1, 20) - 1) * 0.05)::double precision AS score_from,
       COALESCE(CASE is_fraud WHEN 1 THEN 'Fraud' WHEN 0 THEN 'Legit' END, 'Pending') AS label,
       count(*) AS n
FROM scored_transactions
WHERE scored_at >= now() - interval '3 hours'
GROUP BY 1, 2, 3;

-- Latest PSI per feature + full history for trend lines
CREATE OR REPLACE VIEW v_drift_latest AS
SELECT DISTINCT ON (feature) feature, psi, level, window_n,
       (measured_at AT TIME ZONE 'Asia/Jakarta') AS measured_at_wib
FROM drift_metrics
ORDER BY feature, measured_at DESC;

-- Top 10 drifting signals for compact visuals
CREATE OR REPLACE VIEW v_drift_top AS
SELECT feature, psi, level, measured_at_wib FROM v_drift_latest ORDER BY psi DESC LIMIT 10;

CREATE OR REPLACE VIEW v_drift_history AS
SELECT (measured_at AT TIME ZONE 'Asia/Jakarta') AS measured_at_wib, feature, psi, level
FROM drift_metrics
WHERE measured_at >= now() - interval '3 hours';

CREATE OR REPLACE VIEW v_model_registry AS
SELECT model_version, (deployed_at AT TIME ZONE 'Asia/Jakarta') AS deployed_at_wib, variant, runtime, n_features,
       n_trees, t_review, t_block, valid_auprc, test_auprc, p99_model_ms, p99_e2e_ms
FROM model_registry;
