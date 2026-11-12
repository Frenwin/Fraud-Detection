"""Live Fraud Operations dashboard (Streamlit) — reads the same SQL views as Power BI / Tableau.

    streamlit run dashboard/streamlit/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from frauddet.config import load_config  # noqa: E402
from frauddet.db import connect  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
LEVEL_COLOR = {"stable": GOOD, "moderate": WARNING, "significant": CRITICAL}

st.set_page_config(page_title="Fraud Ops · Live", page_icon="🛡️", layout="wide")
st.markdown(f"""<style>
.stApp {{background:{SURFACE};}} h1,h2,h3 {{color:{INK}; font-weight:600;}}
[data-testid="stMetricValue"] {{font-size:1.55rem;}} [data-testid="stMetricLabel"] {{color:{INK2};}}
.block-container {{padding-top:1.4rem;}}
</style>""", unsafe_allow_html=True)


def layout(fig, title, h=280, ytitle=None):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=14, color=INK)), height=h,
                      margin=dict(l=8, r=8, t=40, b=8), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(family="Segoe UI, system-ui, sans-serif", color=INK2, size=12),
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED), title=ytitle)
    return fig


@st.cache_resource
def conn():
    c = connect(load_config()["db_url"])
    c.autocommit = True
    return c


def q(sql: str) -> pd.DataFrame:
    try:
        return pd.read_sql(sql, conn())
    except Exception:
        st.cache_resource.clear()
        return pd.read_sql(sql, conn())


def rupiah(v) -> str:
    v = float(v or 0)
    if v >= 1e9:
        return f"Rp {v / 1e9:.2f} M"
    if v >= 1e6:
        return f"Rp {v / 1e6:.1f} jt"
    return f"Rp {v:,.0f}".replace(",", ".")


def pct(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.1%}"


st.title("🛡️ Real-time Fraud Operations")
tab_live, tab_health, tab_research = st.tabs(["Live operations", "Model health", "Research results"])


@st.fragment(run_every="2s")
def live():
    k = q("SELECT * FROM v_kpi_current").iloc[0]
    if not k.n_tx:
        st.info("Belum ada transaksi. Jalankan API dan `python -m frauddet.stream.replay`.")
        return
    st.caption(f"Model **{k.model_version}** · refresh tiap 2 detik · window rate = 5 menit terakhir")
    c = st.columns(7)
    c[0].metric("Throughput", f"{k.tps_5m:.1f} TPS")
    c[1].metric("Alert rate", pct(k.alert_rate_5m))
    c[2].metric("Block rate", pct(k.block_rate_5m))
    c[3].metric("p99 latency", f"{k.p99_latency_ms_5m:.2f} ms")
    c[4].metric("Fraud intercepted", rupiah(k.fraud_amount_intercepted_idr),
                f"{(k.fraud_amount_intercepted_idr or 0) / max(1, k.fraud_amount_total_idr or 1):.0%} of fraud value",
                delta_color="off")
    c[5].metric("Precision (labeled)", pct(k.precision_labeled))
    c[6].metric("Recall (labeled)", pct(k.recall_labeled))

    m = q("SELECT * FROM v_kpi_minute ORDER BY minute_wib")
    a, b = st.columns(2)
    f = go.Figure(go.Scatter(x=m.minute_wib, y=m.n_tx, mode="lines" if len(m) > 2 else "lines+markers",
                             line=dict(color=S1, width=2), marker=dict(size=8), name="tx/min",
                             hovertemplate="%{x|%H:%M}<br>%{y} tx<extra></extra>"))
    a.plotly_chart(layout(f, "Transactions per minute"), use_container_width=True)
    f = go.Figure()
    f.add_bar(x=m.minute_wib, y=m.n_alert - m.n_block, name="REVIEW", marker_color=S4,
              hovertemplate="%{x|%H:%M}<br>%{y} review<extra></extra>")
    f.add_bar(x=m.minute_wib, y=m.n_block, name="BLOCK", marker_color=S2,
              hovertemplate="%{x|%H:%M}<br>%{y} block<extra></extra>")
    f.update_layout(barmode="stack", bargap=0.15)
    b.plotly_chart(layout(f, "Alerts per minute"), use_container_width=True)

    a, b = st.columns([3, 2])
    aq = q("SELECT scored_at_wib, decision, fraud_score, amount_idr, channel, category, city, reason_1, reason_2, "
           "label_status FROM v_alert_queue LIMIT 40")
    a.markdown("**Analyst queue** — latest alerts with reason codes")
    a.dataframe(aq, hide_index=True, use_container_width=True, height=330,
                column_config={"scored_at_wib": st.column_config.DatetimeColumn("time", format="HH:mm:ss"),
                               "fraud_score": st.column_config.ProgressColumn("score", min_value=0, max_value=1,
                                                                              format="%.2f"),
                               "amount_idr": st.column_config.NumberColumn("amount (Rp)", format="localized")})
    sp = q("SELECT * FROM v_scenario_performance ORDER BY recall")
    f = go.Figure(go.Bar(x=sp.recall, y=sp.scenario, orientation="h", marker_color=S1,
                         text=[f"{r:.0%}  ({c}/{n})" for r, c, n in zip(sp.recall, sp.n_caught, sp.n_fraud)],
                         textposition="outside", textfont=dict(color=INK2),
                         hovertemplate="%{y}<br>recall %{x:.1%}<extra></extra>"))
    f.update_xaxes(range=[0, 1.3], tickvals=[0, .25, .5, .75, 1], tickformat=".0%")
    b.plotly_chart(layout(f, "Recall per fraud scenario (labeled)", h=360), use_container_width=True)

    a, b = st.columns([3, 2])
    cr = q("SELECT * FROM v_city_risk WHERE lat IS NOT NULL")
    f = go.Figure(go.Scattergeo(lat=cr.lat, lon=cr.lon, text=cr.city, mode="markers",
                                marker=dict(size=(cr.n_tx / cr.n_tx.max() * 38 + 6), color=cr.alert_rate,
                                            colorscale=[[0, "#cde2fb"], [1, "#104281"]], line=dict(color="white", width=1.5),
                                            colorbar=dict(title="alert rate", tickformat=".1%")),
                                customdata=cr[["n_tx", "n_alert", "n_fraud"]],
                                hovertemplate="<b>%{text}</b><br>%{customdata[0]} tx · %{customdata[1]} alerts · "
                                              "%{customdata[2]} fraud<extra></extra>"))
    f.update_geos(lataxis_range=[-11, 15], lonaxis_range=[94, 125], showcountries=True, countrycolor=AXIS,
                  landcolor="#f0efec", showocean=True, oceancolor=SURFACE, bgcolor=SURFACE)
    a.plotly_chart(layout(f, "Transaction & alert map", h=360), use_container_width=True)
    rf = q("SELECT * FROM v_reason_frequency ORDER BY n_alerts DESC LIMIT 10")
    f = go.Figure(go.Bar(x=rf.n_alerts, y=rf.reason, orientation="h", marker_color=S1,
                         hovertemplate="%{y}<br>%{x} alerts<extra></extra>"))
    f.update_yaxes(autorange="reversed")
    b.plotly_chart(layout(f, "Top reasons behind alerts", h=360), use_container_width=True)


@st.fragment(run_every="3s")
def health():
    m = q("SELECT * FROM v_kpi_minute ORDER BY minute_wib")
    if m.empty:
        st.info("Belum ada data.")
        return
    a, b = st.columns(2)
    f = go.Figure()
    f.add_scatter(x=m.minute_wib, y=m.p50_latency_ms, name="p50", line=dict(color=S1, width=2))
    f.add_scatter(x=m.minute_wib, y=m.p99_latency_ms, name="p99", line=dict(color=S2, width=2))
    slo = load_config()["slo"]["p99_e2e_ms"]
    f.add_hline(y=slo, line=dict(color=CRITICAL, width=1), annotation_text=f"SLO {slo} ms",
                annotation_font_color=CRITICAL)
    a.plotly_chart(layout(f, "Server-side scoring latency per minute", ytitle="ms"), use_container_width=True)
    h = q("SELECT * FROM v_latency_histogram ORDER BY bucket")
    f = go.Figure(go.Bar(x=h.bucket_start_ms, y=h.n, marker_color=S1, hovertemplate="%{x:.2f} ms<br>%{y}<extra></extra>"))
    f.update_layout(bargap=0.08)
    b.plotly_chart(layout(f, "Latency distribution (last 15 min)", ytitle="transactions"), use_container_width=True)

    a, b = st.columns(2)
    d = q("SELECT * FROM v_drift_latest ORDER BY psi DESC")
    if d.empty:
        a.info("PSI dihitung tiap 2.000 transaksi.")
    else:
        f = go.Figure(go.Bar(x=d.psi, y=d.feature, orientation="h", marker_color=[LEVEL_COLOR[x] for x in d.level],
                             text=[f"{p:.3f} · {lv}" for p, lv in zip(d.psi, d.level)], textposition="outside",
                             textfont=dict(color=INK2), hovertemplate="%{y}<br>PSI %{x:.3f}<extra></extra>"))
        f.update_yaxes(autorange="reversed")
        f.update_xaxes(range=[0, max(0.3, d.psi.max() * 1.35)])
        a.plotly_chart(layout(f, "Feature drift vs. training (PSI · 0.10 moderate · 0.25 significant)", h=520),
                       use_container_width=True)
    sd = q("SELECT * FROM v_score_distribution WHERE label <> 'Pending' ORDER BY bucket")
    f = go.Figure()
    for lab, col in (("Legit", S1), ("Fraud", S2)):
        s = sd[sd.label == lab]
        f.add_bar(x=s.score_from, y=s.n, name=lab, marker_color=col)
    f.update_layout(barmode="group", bargap=0.1)
    f.update_yaxes(type="log")
    b.plotly_chart(layout(f, "Score distribution by confirmed label (log scale)", h=520, ytitle="transactions"),
                   use_container_width=True)
    reg = q("SELECT * FROM v_model_registry ORDER BY deployed_at_wib DESC")
    st.markdown("**Model registry**")
    st.dataframe(reg, hide_index=True, use_container_width=True)


def research():
    res = ROOT / "reports" / "results"
    fig = ROOT / "reports" / "figures"
    summ = res / "optimize_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        c = st.columns(4)
        c[0].metric("Production model", s["selected"]["variant"].replace(" (ONNX)", ""))
        c[1].metric("Valid AUPRC", f"{s['selected']['valid_auprc']:.3f}")
        c[2].metric("Model p99", f"{s['selected']['p99_ms']:.3f} ms")
        c[3].metric("End-to-end p99", f"{s['e2e_production']['p99']:.2f} ms",
                    "SLO met" if s["slo_met"] else "SLO missed", delta_color="normal" if s["slo_met"] else "inverse")
    for title, name in [("Model comparison", "e1_model_comparison"), ("Accuracy vs. latency", "e9_pareto"),
                        ("Budgets", "e9_budgets"), ("End-to-end cost", "e10_end_to_end"),
                        ("Business value", "e6_cost_curve"), ("Scenario recall", "e8_scenario_recall")]:
        p = fig / f"{name}.png"
        if p.exists():
            st.markdown(f"**{title}**")
            st.image(str(p))
    for name in ["e1_model_comparison", "e9_variants"]:
        p = res / f"{name}.csv"
        if p.exists():
            with st.expander(f"Table · {name}"):
                st.dataframe(pd.read_csv(p), hide_index=True, use_container_width=True)


with tab_live:
    live()
with tab_health:
    health()
with tab_research:
    research()
