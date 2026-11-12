-- =============================================================
-- Fraud Detection — operational schema (PostgreSQL 16)
-- Written by the scoring API (async batch sink), read by Power BI / Tableau / Streamlit.
-- =============================================================

CREATE TABLE IF NOT EXISTS scored_transactions (
    tx_id          BIGINT PRIMARY KEY,
    event_ts       TIMESTAMP        NOT NULL,          -- transaction time (simulated clock)
    scored_at      TIMESTAMPTZ      NOT NULL,          -- wall-clock time the API scored it
    customer_id    INTEGER          NOT NULL,
    terminal_id    INTEGER          NOT NULL,
    device_id      BIGINT,
    channel        TEXT             NOT NULL,          -- POS | ONLINE | ATM
    category       TEXT             NOT NULL,
    city           TEXT,
    lat            DOUBLE PRECISION,
    lon            DOUBLE PRECISION,
    amount         NUMERIC(16, 0)   NOT NULL,          -- IDR
    fraud_score    DOUBLE PRECISION NOT NULL,          -- calibrated probability
    decision       TEXT             NOT NULL,          -- APPROVE | REVIEW | BLOCK
    model_version  TEXT             NOT NULL,
    reason_1       TEXT,
    reason_2       TEXT,
    reason_3       TEXT,
    latency_ms     REAL,
    feature_ms     REAL,
    model_ms       REAL,
    explain_ms     REAL,
    is_fraud       SMALLINT,                           -- NULL until the chargeback label arrives
    scenario       TEXT,
    labeled_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_scored_at        ON scored_transactions (scored_at DESC);
CREATE INDEX IF NOT EXISTS ix_decision_scored  ON scored_transactions (decision, scored_at DESC);

CREATE TABLE IF NOT EXISTS drift_metrics (
    id           BIGSERIAL PRIMARY KEY,
    measured_at  TIMESTAMPTZ NOT NULL,
    window_n     INTEGER     NOT NULL,
    feature      TEXT        NOT NULL,
    psi          DOUBLE PRECISION NOT NULL,
    level        TEXT        NOT NULL                  -- stable | moderate | significant
);
CREATE INDEX IF NOT EXISTS ix_drift_time ON drift_metrics (measured_at DESC);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version  TEXT PRIMARY KEY,
    deployed_at    TIMESTAMPTZ NOT NULL,
    variant        TEXT,
    runtime        TEXT,
    n_features     INTEGER,
    n_trees        INTEGER,
    t_review       DOUBLE PRECISION,
    t_block        DOUBLE PRECISION,
    valid_auprc    DOUBLE PRECISION,
    test_auprc     DOUBLE PRECISION,
    p99_model_ms   DOUBLE PRECISION,
    p99_e2e_ms     DOUBLE PRECISION
);
