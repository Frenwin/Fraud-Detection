"""Generate the Power BI project  dashboard/powerbi/FraudOps.pbip  — "dashboard as code".

* Semantic model : TMDL, every table is DirectQuery on PostgreSQL (views in sql/views.sql + research r_* tables)
* Report         : PBIR (public JSON schemas), 4 pages, auto page refresh on the live pages
* Design         : navy header band, white cards on a cool-grey canvas, validated categorical palette,
                   status colours (good / warning / critical) only where a value has a meaning, always with a label

    python dashboard/powerbi/build_pbip.py          # needs the PostgreSQL container (reads column types)

Then open dashboard/powerbi/FraudOps.pbip in Power BI Desktop (credentials: fraud / fraud).
"""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from frauddet.config import load_config  # noqa: E402
from frauddet.db import connect  # noqa: E402

NAME = "FraudOps"
OUT = ROOT / "dashboard" / "powerbi"
SM = OUT / f"{NAME}.SemanticModel"
RP = OUT / f"{NAME}.Report"
SCHEMA = "https://developer.microsoft.com/json-schemas/fabric"
V_VISUAL, V_PAGE, V_REPORT = "2.4.0", "2.0.0", "3.0.0"
BASE_THEME = "Fluent2-CY26SU10"
PBI_THEMES = Path(r"C:\Program Files\Microsoft Power BI Desktop\bin\WebView2Resources\minerva\sharedresources\BaseThemes")
W, H, M, G = 1280, 800, 20, 12  # page size (16:10 = typical laptop), margin, gutter
HEADER_H = 64

# ---------------------------------------------------------------------------------------------- palette
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                      "#008300", "#6250d6")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
NAVY, NAVY_SUB = "#0d2a4f", "#b7d3f6"
CANVAS, CARD, BORDER, GRID = "#eef1f6", "#ffffff", "#e2e6ec", "#eef0f3"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
NEUTRAL = "#c3c2b7"

LIVE_TABLES = ["v_live_transactions", "v_kpi_current", "v_kpi_minute", "v_alert_queue", "v_scenario_performance",
               "v_city_risk", "v_reason_frequency", "v_latency_histogram", "v_score_distribution", "v_drift_top",
               "v_model_registry"]
RESEARCH_TABLES = ["r_e1_model_comparison", "r_e9_variants", "r_e10_end_to_end", "r_e11_loadtest",
                   "r_e6_threshold_sweep_valid", "v_e9_plot", "v_e11_plot"]
TYPE_MAP = {"bigint": "int64", "integer": "int64", "smallint": "int64", "double precision": "double", "real": "double",
            "numeric": "double", "text": "string", "character varying": "string", "boolean": "boolean",
            "timestamp without time zone": "dateTime", "timestamp with time zone": "dateTime"}
FORMATS = {"amount_idr": "#,0", "amount_blocked_idr": "#,0", "fraud_amount_idr": "#,0", "amount_caught_idr": "#,0",
           "fraud_score": "0.000", "alert_rate": "0.0%", "recall": "0%", "minute_wib": "HH:mm",
           "scored_at_wib": "HH:mm:ss", "psi": "0.000", "test_auprc": "0.000", "valid_auprc": "0.000",
           "p99_ms": "0.000", "throughput_rps": "#,0", "net_savings": "#,0"}

# ---------------------------------------------------------------------------------------------- measures
LT = "'v_live_transactions'"


def rp_label(measure: str) -> str:
    """Rupiah display string: Rp 464,0 jt / Rp 1,25 M (Indonesian separators)."""
    return (f"VAR v = [{measure}] RETURN IF(ISBLANK(v), \"–\", IF(ABS(v) >= 1e9, "
            f"\"Rp \" & FORMAT(v / 1e9, \"#,0.00\", \"id-ID\") & \" M\", "
            f"\"Rp \" & FORMAT(v / 1e6, \"#,0.0\", \"id-ID\") & \" jt\"))")


MEASURES = [
    ("v_live_transactions", "Total Transactions", f"COUNTROWS({LT})", "#,0"),
    ("v_live_transactions", "Alerts", f"SUM({LT}[is_alert])", "#,0"),
    ("v_live_transactions", "Blocks", f"SUM({LT}[is_block])", "#,0"),
    ("v_live_transactions", "Alert Rate", "DIVIDE([Alerts], [Total Transactions])", "0.00%"),
    ("v_live_transactions", "Block Rate", "DIVIDE([Blocks], [Total Transactions])", "0.00%"),
    ("v_live_transactions", "Labeled Transactions", f"CALCULATE([Total Transactions], NOT ISBLANK({LT}[is_fraud]))", "#,0"),
    ("v_live_transactions", "Fraud Confirmed", f"CALCULATE([Total Transactions], {LT}[is_fraud] = 1)", "#,0"),
    ("v_live_transactions", "Fraud Caught", f"CALCULATE([Total Transactions], {LT}[is_fraud] = 1, {LT}[is_alert] = 1)", "#,0"),
    ("v_live_transactions", "False Positives", f"CALCULATE([Total Transactions], {LT}[is_fraud] = 0, {LT}[is_alert] = 1)", "#,0"),
    ("v_live_transactions", "Precision", f"DIVIDE([Fraud Caught], CALCULATE([Alerts], NOT ISBLANK({LT}[is_fraud])))", "0.0%"),
    ("v_live_transactions", "Recall", "DIVIDE([Fraud Caught], [Fraud Confirmed])", "0.0%"),
    ("v_live_transactions", "Label Coverage", "DIVIDE([Labeled Transactions], [Total Transactions])", "0.0%"),
    ("v_live_transactions", "Total Amount (Rp)", f"SUM({LT}[amount_idr])", "#,0"),
    ("v_live_transactions", "Fraud Amount (Rp)", f"CALCULATE([Total Amount (Rp)], {LT}[is_fraud] = 1)", "#,0"),
    ("v_live_transactions", "Fraud Intercepted (Rp)", f"CALCULATE([Total Amount (Rp)], {LT}[is_fraud] = 1, {LT}[is_alert] = 1)", "#,0"),
    ("v_live_transactions", "Fraud Missed (Rp)", "[Fraud Amount (Rp)] - [Fraud Intercepted (Rp)]", "#,0"),
    ("v_live_transactions", "Loss Prevented %", "DIVIDE([Fraud Intercepted (Rp)], [Fraud Amount (Rp)])", "0.0%"),
    ("v_live_transactions", "Net Savings (Rp)", "[Fraud Intercepted (Rp)] - 15000 * [Alerts] - 25000 * [False Positives]", "#,0"),
    ("v_live_transactions", "Fraud Amount Label", rp_label("Fraud Amount (Rp)"), None),
    ("v_live_transactions", "Fraud Intercepted Label", rp_label("Fraud Intercepted (Rp)"), None),
    ("v_live_transactions", "Fraud Missed Label", rp_label("Fraud Missed (Rp)"), None),
    ("v_live_transactions", "Net Savings Label", rp_label("Net Savings (Rp)"), None),
    ("v_live_transactions", "Last Update",
     f"\"● LIVE · update \" & FORMAT(MAX({LT}[scored_at_wib]), \"HH:mm:ss\") & \" WIB\"", None),
    ("v_kpi_current", "TPS 5m", "MAX('v_kpi_current'[tps_5m])", "0.0"),
    ("v_kpi_current", "P50 Latency 5m (ms)", "MAX('v_kpi_current'[p50_latency_ms_5m])", "0.00"),
    ("v_kpi_current", "P99 Latency 5m (ms)", "MAX('v_kpi_current'[p99_latency_ms_5m])", "0.00"),
    ("v_kpi_current", "SLO p99 (ms)", "15", "0"),
    ("v_kpi_current", "Gauge Max (ms)", "20", "0"),
    ("v_kpi_current", "Latency Color", "IF([P99 Latency 5m (ms)] <= [SLO p99 (ms)], \"#0ca30c\", \"#d03b3b\")", None),
    ("v_kpi_current", "Latency Status",
     "IF([P99 Latency 5m (ms)] <= [SLO p99 (ms)], \"✓ Dalam SLO\", \"✕ Melewati SLO\")", None),
    ("v_kpi_current", "Model Version", "SELECTEDVALUE('v_kpi_current'[model_version], \"-\")", None),
    ("v_alert_queue", "Decision Color",
     "SWITCH(SELECTEDVALUE('v_alert_queue'[decision]), \"BLOCK\", \"#d03b3b\", \"REVIEW\", \"#a86400\", \"#52514e\")", None),
    ("v_scenario_performance", "Recall Color",
     "VAR r = MAX('v_scenario_performance'[recall]) RETURN SWITCH(TRUE(), r < 0.5, \"#d03b3b\", r < 0.8, \"#ec835a\", \"#2a78d6\")", None),
    ("v_drift_top", "Drift Color",
     "VAR p = MAX('v_drift_top'[psi]) RETURN SWITCH(TRUE(), p >= 0.25, \"#d03b3b\", p >= 0.10, \"#fab219\", \"#0ca30c\")", None),
    ("r_e9_variants", "Best Valid AUPRC", "MAX('r_e9_variants'[valid_auprc])", "0.000"),
    ("r_e9_variants", "Fastest Model p99 (ms)", "MIN('r_e9_variants'[p99_ms])", "0.000"),
    ("v_e9_plot", "Pareto Color", "IF(SELECTEDVALUE('v_e9_plot'[pareto], FALSE()), \"#2a78d6\", \"#c3c2b7\")", None),
    ("r_e1_model_comparison", "Rules Baseline AUPRC",
     "CALCULATE(MAX('r_e1_model_comparison'[test_auprc]), 'r_e1_model_comparison'[model] = \"Rules\")", "0.000"),
    ("r_e1_model_comparison", "Model Color",
     "IF(SELECTEDVALUE('r_e1_model_comparison'[model]) = \"Rules\", \"#c3c2b7\", \"#2a78d6\")", None),
    ("r_e11_loadtest", "Peak Throughput (req/s)", "MAX('r_e11_loadtest'[throughput_rps])", "#,0"),
]


def guid() -> str:
    return str(uuid.uuid4())


def q(name: str) -> str:
    """TMDL object name quoting."""
    return name if name.replace("_", "").isalnum() else "'" + name.replace("'", "''") + "'"


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ============================================================================================ semantic model
def table_columns(conn, table: str) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='public' "
                    "AND table_name=%s ORDER BY ordinal_position", (table,))
        rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"table/view {table} not found - run `python -m frauddet.db init` and `load-research` first")
    return rows


def build_semantic_model(conn) -> None:
    d = SM / "definition"
    dump(SM / "definition.pbism", {"$schema": f"{SCHEMA}/item/semanticModel/definitionProperties/1.0.0/schema.json",
                                   "version": "4.2", "settings": {}})
    write(d / "database.tmdl", "database\n\tcompatibilityLevel: 1600\n")
    tables = LIVE_TABLES + RESEARCH_TABLES
    order = json.dumps(["PgServer", "PgDatabase"] + tables)
    write(d / "model.tmdl", "model Model\n\tculture: en-US\n\tdefaultPowerBIDataSourceVersion: powerBI_V3\n"
                            "\tsourceQueryCulture: en-US\n\tdataAccessOptions\n\t\tlegacyRedirects\n"
                            "\t\treturnErrorValuesAsNull\n\n"
                            f"annotation PBI_QueryOrder = {order}\n\n"
                            "annotation __PBI_TimeIntelligenceEnabled = 0\n\n"
                            + "".join(f"ref table {q(t)}\n" for t in tables))
    write(d / "expressions.tmdl",
          'expression PgServer = "localhost:5434" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n'
          f"\tlineageTag: {guid()}\n\n\tannotation PBI_ResultType = Text\n\n"
          'expression PgDatabase = "frauddb" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n'
          f"\tlineageTag: {guid()}\n\n\tannotation PBI_ResultType = Text\n")
    for t in tables:
        lines = [f"table {q(t)}", f"\tlineageTag: {guid()}", ""]
        for name, mexpr, fmt in [(m[1], m[2], m[3]) for m in MEASURES if m[0] == t]:
            lines.append(f"\tmeasure {q(name)} = {mexpr}")
            if fmt:
                lines.append(f"\t\tformatString: {fmt}")
            lines += [f"\t\tlineageTag: {guid()}", ""]
        for col_, pgtype in table_columns(conn, t):
            dt = TYPE_MAP[pgtype]
            lines.append(f"\tcolumn {q(col_)}")
            lines.append(f"\t\tdataType: {dt}")
            if col_ in FORMATS:
                lines.append(f"\t\tformatString: {FORMATS[col_]}")
            elif dt == "dateTime":
                lines.append("\t\tformatString: General Date")
            lines += [f"\t\tlineageTag: {guid()}",
                      f"\t\tsummarizeBy: {'sum' if dt in ('int64', 'double') else 'none'}",
                      f"\t\tsourceColumn: {col_}", "", "\t\tannotation SummarizationSetBy = Automatic", ""]
        lines += [f"\tpartition {q(t)} = m", "\t\tmode: directQuery", "\t\tsource =",
                  "\t\t\t\tlet", "\t\t\t\t    Source = PostgreSQL.Database(PgServer, PgDatabase),",
                  f'\t\t\t\t    Data = Source{{[Schema="public",Item="{t}"]}}[Data]', "\t\t\t\tin", "\t\t\t\t    Data",
                  "", "\tannotation PBI_ResultType = Table", ""]
        write(d / "tables" / f"{t}.tmdl", "\n".join(lines))


# ============================================================================================ query helpers
def lit(v):
    if isinstance(v, bool):
        s = "true" if v else "false"
    elif isinstance(v, (int, float)):
        s = f"{v}D"
    else:
        s = "'" + str(v).replace("'", "''") + "'"
    return {"expr": {"Literal": {"Value": s}}}


def color(hex_: str):
    return {"solid": {"color": lit(hex_)}}


def col(table, column):
    return {"Column": {"Expression": {"SourceRef": {"Entity": table}}, "Property": column}}


def meas(table, name):
    return {"Measure": {"Expression": {"SourceRef": {"Entity": table}}, "Property": name}}


def color_by(table, measure):
    """Conditional colour = 'field value' rule driven by a DAX measure returning a hex string."""
    return {"solid": {"color": {"expr": meas(table, measure)}}}


AGG = {"sum": 0, "avg": 1, "min": 3, "max": 4}


def agg(table, column, fn="sum"):
    return {"Aggregation": {"Expression": col(table, column), "Function": AGG[fn]}}


def ref_of(field) -> str:
    if "Column" in field or "Measure" in field:
        c = field.get("Column") or field.get("Measure")
        return f"{c['Expression']['SourceRef']['Entity']}.{c['Property']}"
    c = field["Aggregation"]
    inner = c["Expression"]["Column"]
    fn = {0: "Sum", 1: "Avg", 3: "Min", 4: "Max"}[c["Function"]]
    return f"{fn}({inner['Expression']['SourceRef']['Entity']}.{inner['Property']})"


def proj(field, display=None):
    ref = ref_of(field)
    native = ref.split(".", 1)[1].rstrip(")") if "(" not in ref.split(".")[0] else ref
    p = {"field": field, "queryRef": ref, "nativeQueryRef": native}
    if display:
        p["displayName"] = display
    return p


WILDCARD = {"data": [{"dataViewWildcard": {"matchingOption": 1}}]}


# ============================================================================================ page builder
def card_container(title=None, bg=CARD, border=True, radius=10, pad=(10, 8, 14, 14)):
    o = {"background": [{"properties": {"show": lit(True), "color": color(bg), "transparency": lit(0)}}],
         "border": [{"properties": {"show": lit(border), "color": color(BORDER), "radius": lit(radius), "width": lit(1)}}],
         "padding": [{"properties": {"top": lit(pad[0]), "bottom": lit(pad[1]), "left": lit(pad[2]), "right": lit(pad[3])}}],
         "dropShadow": [{"properties": {"show": lit(False)}}],
         "subTitle": [{"properties": {"show": lit(False)}}],
         "visualHeader": [{"properties": {"show": lit(False)}}]}
    if title:
        o["title"] = [{"properties": {"show": lit(True), "text": lit(title), "fontColor": color(INK), "fontSize": lit(11),
                                      "bold": lit(True), "fontFamily": lit("Segoe UI Semibold")}}]
    else:
        o["title"] = [{"properties": {"show": lit(False)}}]
    return o


class Page:
    def __init__(self, name, display, subtitle, refresh_seconds=None):
        self.name, self.display, self.subtitle, self.refresh = name, display, subtitle, refresh_seconds
        self.visuals: list[dict] = []
        self._header()

    def add(self, name, x, y, w, h, visual, container=None):
        visual["visualContainerObjects"] = container if container is not None else card_container()
        visual.setdefault("drillFilterOtherVisuals", True)
        self.visuals.append({
            "$schema": f"{SCHEMA}/item/report/definition/visualContainer/{V_VISUAL}/schema.json", "name": name,
            "position": {"x": x, "y": y, "z": len(self.visuals) * 1000, "width": w, "height": h,
                         "tabOrder": len(self.visuals) * 1000},
            "visual": visual})

    # ---- building blocks -------------------------------------------------------------------------------
    def text(self, name, x, y, w, h, paragraphs, bg=None):
        runs = [{"textRuns": [{"value": t, "textStyle": {"fontWeight": wgt, "fontSize": f"{size}pt", "color": c,
                                                           "fontFamily": "Segoe UI"}}]} for t, size, wgt, c in paragraphs]
        cont = card_container(bg=bg or CARD, border=False, radius=0, pad=(6, 4, 8, 8))
        if bg is None:
            cont["background"] = [{"properties": {"show": lit(False)}}]
        self.add(name, x, y, w, h, {"visualType": "textbox", "objects": {"general": [{"properties": {"paragraphs": runs}}]}},
                 cont)

    def bar(self, name, x, y, w, h, hex_):
        """Thin coloured accent bar (empty textbox with a fill)."""
        cont = card_container(bg=hex_, border=False, radius=2, pad=(0, 0, 0, 0))
        self.add(name, x, y, w, h, {"visualType": "textbox",
                                    "objects": {"general": [{"properties": {"paragraphs": [{"textRuns": [{"value": " "}]}]}}]}},
                 cont)

    def _header(self):
        self.text("hdr_band", 0, 0, W, HEADER_H,
                  [(self.display, 17, "bold", "#ffffff"), (self.subtitle, 9, "normal", NAVY_SUB)], bg=NAVY)
        self.card("hdr_update", W - M - 290, 14, 290, 38, meas("v_live_transactions", "Last Update"), None, size=11,
                  value_color="#ffffff", bg=NAVY, border=False, show_label=False)

    def card(self, name, x, y, w, h, field, label, size=20, value_color=INK, value_color_measure=None, bg=CARD,
             border=True, show_label=True):
        vc = color_by(*value_color_measure) if value_color_measure else color(value_color)
        self.add(name, x, y, w, h, {
            "visualType": "card",
            "query": {"queryState": {"Values": {"projections": [proj(field, label)]}}},
            "objects": {"labels": [{"properties": {"fontSize": lit(size), "color": vc, "labelDisplayUnits": lit(0),
                                                   "fontFamily": lit("Segoe UI Semibold")}}],
                        "categoryLabels": [{"properties": {"show": lit(show_label), "fontSize": lit(9),
                                                           "color": color(INK2)}}]}},
            card_container(bg=bg, border=border, pad=(6, 6, 10, 10)))

    def kpis(self, y, items, h=84):
        n = len(items)
        w = (W - 2 * M - (n - 1) * G) / n
        for i, it in enumerate(items):
            name, field, label, accent = it[:4]
            extra = it[4] if len(it) > 4 else {}
            x = round(M + i * (w + G))
            self.card(name, x, y, round(w), h, field, label, **extra)
            self.bar(f"{name}_accent", x, y + 12, 4, h - 24, accent)

    def chart(self, name, x, y, w, h, vtype, roles: dict, title, objects=None, sort=None):
        qs = {role: {"projections": [proj(f, d) for f, d in fields]} for role, fields in roles.items()}
        query = {"queryState": qs}
        if sort:
            field, direction = sort
            query["sortDefinition"] = {"sort": [{"field": field, "direction": direction}], "isDefaultSort": False}
        self.add(name, x, y, w, h, {"visualType": vtype, "query": query, "objects": objects or {}},
                 card_container(title))

    def write(self, root: Path):
        pdir = root / "pages" / self.name
        objects = {"background": [{"properties": {"color": color(CANVAS), "transparency": lit(0)}}],
                   "outspace": [{"properties": {"color": color(CANVAS)}}]}
        if self.refresh:
            objects["pageRefresh"] = [{"properties": {"show": lit(True), "refreshType": lit(0),
                                                      "duration": lit(f"PT{self.refresh}S")}}]
        dump(pdir / "page.json", {"$schema": f"{SCHEMA}/item/report/definition/page/{V_PAGE}/schema.json",
                                  "name": self.name, "displayName": self.display, "displayOption": "FitToPage",
                                  "height": H, "width": W, "objects": objects})
        for v in self.visuals:
            dump(pdir / "visuals" / v["name"] / "visual.json", v)


# ---- formatting presets --------------------------------------------------------------------------------
def axis_fmt(hide_value_axis=False, log_x=False, log_y=False):
    cat = {"fontSize": lit(9), "labelColor": color(INK2), "showAxisTitle": lit(False), "gridlineShow": lit(False)}
    val = {"fontSize": lit(9), "labelColor": color(MUTED), "showAxisTitle": lit(False),
           "gridlineColor": color(GRID), "gridlineStyle": lit("solid")}
    if log_x:
        cat["axisScale"] = lit("log")
    if log_y:
        val["axisScale"] = lit("log")
    if hide_value_axis:
        val["show"] = lit(False)
    return {"categoryAxis": [{"properties": cat}], "valueAxis": [{"properties": val}]}


def legend(show=True):
    return {"legend": [{"properties": {"show": lit(show), "position": lit("TopRight"), "fontSize": lit(9),
                                       "labelColor": color(INK2), "showTitle": lit(False)}}]}


def bars(fill=BLUE, fill_measure=None, label_fmt=None):
    """Horizontal/vertical bars: data labels on, value axis hidden (labels carry the numbers)."""
    o = axis_fmt(hide_value_axis=True)
    o["labels"] = [{"properties": {"show": lit(True), "color": color(INK2), "fontSize": lit(9)}}]
    if fill_measure:
        o["dataPoint"] = [{"properties": {"fill": color_by(*fill_measure)}, "selector": WILDCARD}]
    else:
        o["dataPoint"] = [{"properties": {"fill": color(fill)}}]
    o.update(legend(False))
    return o


def series_colors(mapping: dict):
    return [{"selector": {"metadata": ref}, "properties": {"fill": color(c)}} for ref, c in mapping.items()]


def value_colors(table: str, column: str, mapping: dict):
    """Colour per category *value* of a Series column (colour follows the entity, not its rank)."""
    return [{"selector": {"data": [{"scopeId": {"Comparison": {"ComparisonKind": 0, "Left": col(table, column),
                                                               "Right": {"Literal": {"Value": f"'{v}'"}}}}}]},
             "properties": {"fill": color(c)}} for v, c in mapping.items()]


def lines(mapping: dict, area=False, **axis):
    o = axis_fmt(**axis)
    o["dataPoint"] = series_colors(mapping)
    o["lineStyles"] = [{"properties": {"strokeWidth": lit(2), "showMarker": lit(False), "lineChartType": lit("linear")}}]
    o.update(legend(len(mapping) > 1))
    return o


# ============================================================================================ pages
def grid_cols(n, total=W - 2 * M):
    w = (total - (n - 1) * G) / n
    return [(round(M + i * (w + G)), round(w)) for i in range(n)]


def page_live() -> Page:
    p = Page("live_operations", "Real-time Fraud Operations",
             "Transaksi dinilai FastAPI + LightGBM ONNX · DirectQuery PostgreSQL · refresh otomatis tiap 3 detik",
             refresh_seconds=3)
    y1 = HEADER_H + 14
    p.kpis(y1, [
        ("kpi_tps", meas("v_kpi_current", "TPS 5m"), "Throughput · TPS (5 mnt)", BLUE),
        ("kpi_alert", meas("v_live_transactions", "Alert Rate"), "Alert rate", YELLOW),
        ("kpi_block", meas("v_live_transactions", "Block Rate"), "Block rate", ORANGE),
        ("kpi_p99", meas("v_kpi_current", "P99 Latency 5m (ms)"), "p99 latency (ms) · SLO 15", VIOLET,
         {"value_color_measure": ("v_kpi_current", "Latency Color")}),
        ("kpi_intercept", meas("v_live_transactions", "Fraud Intercepted Label"), "Fraud dicegah", GREEN),
        ("kpi_precision", meas("v_live_transactions", "Precision"), "Precision (berlabel)", AQUA),
        ("kpi_recall", meas("v_live_transactions", "Recall"), "Recall (berlabel)", AQUA),
    ])
    y2, h2 = y1 + 84 + G, 240
    (x1, w1), (x2, w2) = grid_cols(2)
    p.chart("tx_per_minute", x1, y2, w1, h2, "areaChart",
            {"Category": [(col("v_kpi_minute", "minute_wib"), "Menit")],
             "Y": [(agg("v_kpi_minute", "n_tx"), "Transaksi")]},
            "Transaksi per menit", lines({"Sum(v_kpi_minute.n_tx)": BLUE}))
    stacked = axis_fmt()
    stacked["dataPoint"] = series_colors({"Sum(v_kpi_minute.n_review)": YELLOW, "Sum(v_kpi_minute.n_block)": ORANGE})
    stacked.update(legend(True))
    p.chart("alerts_per_minute", x2, y2, w2, h2, "columnChart",
            {"Category": [(col("v_kpi_minute", "minute_wib"), "Menit")],
             "Y": [(agg("v_kpi_minute", "n_review"), "REVIEW"), (agg("v_kpi_minute", "n_block"), "BLOCK")]},
            "Alert per menit · REVIEW vs BLOCK", stacked)
    y3 = y2 + h2 + G
    h3 = H - y3 - M
    tw = 790
    q_ = "v_alert_queue"
    widths = {"scored_at_wib": 58, "decision": 74, "fraud_score": 44, "amount_idr": 94, "city": 72, "reason_1": 176,
              "reason_2": 150, "label_status": 70}
    table_obj = {
        "columnHeaders": [{"properties": {"fontColor": color(INK2), "backColor": color("#f3f5f8"), "fontSize": lit(9),
                                          "wordWrap": lit(False),
                                          "bold": lit(True), "fontFamily": lit("Segoe UI Semibold")}}],
        "total": [{"properties": {"totals": lit(False)}}],
        "values": [{"properties": {"fontSize": lit(9), "fontColorPrimary": color(INK), "fontColorSecondary": color(INK),
                                   "wordWrap": lit(False),
                                   "backColorPrimary": color(CARD), "backColorSecondary": color("#f8f9fb")}},
                   {"properties": {"fontColor": color_by(q_, "Decision Color")},
                    "selector": {**WILDCARD, "metadata": f"{q_}.decision"}}],
        "grid": [{"properties": {"gridHorizontal": lit(True), "gridHorizontalColor": color(GRID),
                                 "gridVertical": lit(False), "rowPadding": lit(3)}}],
        "columnWidth": [{"properties": {"value": lit(v)}, "selector": {"metadata": f"{q_}.{k}"}} for k, v in widths.items()],
    }
    p.chart("alert_queue", M, y3, tw, h3, "tableEx",
            {"Values": [(col(q_, "scored_at_wib"), "Waktu"), (col(q_, "decision"), "Keputusan"),
                        (col(q_, "fraud_score"), "Skor"), (col(q_, "amount_idr"), "Nominal (Rp)"),
                        (col(q_, "city"), "Kota"), (col(q_, "reason_1"), "Alasan utama"),
                        (col(q_, "reason_2"), "Alasan ke-2"), (col(q_, "label_status"), "Label")]},
            "Antrean analis · alert terbaru beserta reason code", table_obj,
            sort=(col(q_, "scored_at_wib"), "Descending"))
    rx = M + tw + G
    p.chart("scenario_recall", rx, y3, W - M - rx, h3, "clusteredBarChart",
            {"Category": [(col("v_scenario_performance", "scenario"), "Skenario")],
             "Y": [(agg("v_scenario_performance", "recall", "max"), "Recall")]},
            "Recall per skenario fraud  (merah < 50%)",
            bars(fill_measure=("v_scenario_performance", "Recall Color")),
            sort=(agg("v_scenario_performance", "recall", "max"), "Descending"))
    return p


def page_business() -> Page:
    p = Page("business_impact", "Business Impact",
             "Nilai fraud yang dicegah vs biaya operasional (review Rp 15.000/alert · friksi nasabah Rp 25.000/false positive)",
             refresh_seconds=5)
    y1 = HEADER_H + 14
    p.kpis(y1, [
        ("kpi_fraud_amt", meas("v_live_transactions", "Fraud Amount Label"), "Nilai fraud terjadi", NEUTRAL),
        ("kpi_intercepted", meas("v_live_transactions", "Fraud Intercepted Label"), "Berhasil dicegah", GREEN,
         {"value_color": "#006300"}),
        ("kpi_missed", meas("v_live_transactions", "Fraud Missed Label"), "Lolos", CRITICAL),
        ("kpi_loss_prev", meas("v_live_transactions", "Loss Prevented %"), "Loss prevented", GREEN),
        ("kpi_net", meas("v_live_transactions", "Net Savings Label"), "Net savings (setelah biaya)", BLUE),
        ("kpi_fp", meas("v_live_transactions", "False Positives"), "False positives", YELLOW),
    ])
    y2, h2 = y1 + 84 + G, 290
    (x1, w1), (x2, w2) = grid_cols(2)
    clustered = bars()
    clustered["dataPoint"] = series_colors({"Sum(v_scenario_performance.amount_caught_idr)": BLUE,
                                            "Sum(v_scenario_performance.amount_idr)": NEUTRAL})
    clustered["labels"] = [{"properties": {"show": lit(False)}}]
    clustered["valueAxis"] = [{"properties": {"show": lit(True), "fontSize": lit(9), "labelColor": color(MUTED),
                                              "showAxisTitle": lit(False), "gridlineColor": color(GRID)}}]
    clustered.update(legend(True))
    p.chart("amount_by_scenario", x1, y2, w1, h2, "clusteredBarChart",
            {"Category": [(col("v_scenario_performance", "scenario"), "Skenario")],
             "Y": [(agg("v_scenario_performance", "amount_caught_idr"), "Dicegah (Rp)"),
                   (agg("v_scenario_performance", "amount_idr"), "Total fraud (Rp)")]},
            "Nilai fraud per skenario · dicegah vs total", clustered,
            sort=(agg("v_scenario_performance", "amount_idr"), "Descending"))
    p.chart("reason_freq", x2, y2, w2, h2, "clusteredBarChart",
            {"Category": [(col("v_reason_frequency", "reason"), "Alasan")],
             "Y": [(agg("v_reason_frequency", "n_alerts"), "Alert")]},
            "Top 8 alasan pemicu alert (reason code)", bars(VIOLET),
            sort=(agg("v_reason_frequency", "n_alerts"), "Descending"))
    y3 = y2 + h2 + G
    h3 = H - y3 - M
    (a, aw), (b, bw), (c, cw) = grid_cols(3)
    p.chart("alert_rate_category", a, y3, aw, h3, "clusteredBarChart",
            {"Category": [(col("v_live_transactions", "category"), "Kategori")],
             "Y": [(meas("v_live_transactions", "Alert Rate"), "Alert rate")]},
            "Alert rate per kategori merchant", bars(YELLOW),
            sort=(meas("v_live_transactions", "Alert Rate"), "Descending"))
    p.chart("city_alerts", b, y3, bw, h3, "clusteredBarChart",
            {"Category": [(col("v_city_risk", "city"), "Kota")],
             "Y": [(agg("v_city_risk", "n_alert"), "Alert")]},
            "Top 10 lokasi (kota / IP) berdasarkan alert", bars(BLUE), sort=(agg("v_city_risk", "n_alert"), "Descending"))
    p.chart("cost_curve", c, y3, cw, h3, "lineChart",
            {"Category": [(col("r_e6_threshold_sweep_valid", "alerts"), "Jumlah alert")],
             "Y": [(agg("r_e6_threshold_sweep_valid", "net_savings", "max"), "Net savings (Rp)")]},
            "Riset E6 · net savings vs jumlah alert (valid, 14 hari)",
            lines({"Max(r_e6_threshold_sweep_valid.net_savings)": GREEN}))
    return p


def page_health() -> Page:
    p = Page("model_health", "Model Health",
             "Latensi server-side, distribusi skor & drift fitur · PSI < 0.10 stabil · 0.10–0.25 moderat · > 0.25 signifikan",
             refresh_seconds=5)
    y1 = HEADER_H + 14
    gw = 300
    p.add("p99_gauge", M, y1, gw, 2 * 84 + G, {
        "visualType": "gauge",
        "query": {"queryState": {
            "Y": {"projections": [proj(meas("v_kpi_current", "P99 Latency 5m (ms)"), "p99 (ms)")]},
            "MaxValue": {"projections": [proj(meas("v_kpi_current", "Gauge Max (ms)"), "Max")]},
            "TargetValue": {"projections": [proj(meas("v_kpi_current", "SLO p99 (ms)"), "SLO")]}}},
        "objects": {"dataPoint": [{"properties": {"fill": color_by("v_kpi_current", "Latency Color"),
                                                  "target": color(INK)}}],
                    "calloutValue": [{"properties": {"fontSize": lit(22), "color": color(INK)}}],
                    "labels": [{"properties": {"show": lit(True), "fontSize": lit(8), "color": color(MUTED)}}]}},
        card_container("p99 latency vs SLO 15 ms"))
    rest = W - 2 * M - gw - G
    cw = (rest - 2 * G) / 3
    cards = [("kpi_p50", meas("v_kpi_current", "P50 Latency 5m (ms)"), "p50 latency (ms)", BLUE, {}),
             ("kpi_slo", meas("v_kpi_current", "Latency Status"), "Status SLO p99", GREEN,
              {"value_color_measure": ("v_kpi_current", "Latency Color"), "size": 16}),
             ("kpi_cov", meas("v_live_transactions", "Label Coverage"), "Label coverage (chargeback)", AQUA, {}),
             ("kpi_tps_h", meas("v_kpi_current", "TPS 5m"), "Throughput (TPS)", BLUE, {}),
             ("kpi_model", meas("v_kpi_current", "Model Version"), "Model produksi", VIOLET, {"size": 11}),
             ("kpi_alerts_h", meas("v_live_transactions", "Alerts"), "Total alert (3 jam)", ORANGE, {})]
    for i, (name, field, label, accent, extra) in enumerate(cards):
        x = round(M + gw + G + (i % 3) * (cw + G))
        y = y1 + (i // 3) * (84 + G)
        p.card(name, x, y, round(cw), 84, field, label, **extra)
        p.bar(f"{name}_accent", x, y + 12, 4, 60, accent)
    y2, h2 = y1 + 2 * 84 + 2 * G, 230
    (x1, w1), (x2, w2) = grid_cols(2)
    p.chart("latency_trend", x1, y2, w1, h2, "lineChart",
            {"Category": [(col("v_kpi_minute", "minute_wib"), "Menit")],
             "Y": [(agg("v_kpi_minute", "p50_latency_ms", "max"), "p50 (ms)"),
                   (agg("v_kpi_minute", "p99_latency_ms", "max"), "p99 (ms)")]},
            "Latensi scoring per menit (ms)",
            lines({"Max(v_kpi_minute.p50_latency_ms)": BLUE, "Max(v_kpi_minute.p99_latency_ms)": ORANGE}))
    hist = axis_fmt()
    hist["dataPoint"] = [{"properties": {"fill": color(BLUE)}}]
    p.chart("latency_hist", x2, y2, w2, h2, "columnChart",
            {"Category": [(col("v_latency_histogram", "bucket_start_ms"), "ms")],
             "Y": [(agg("v_latency_histogram", "n"), "Transaksi")]},
            "Distribusi latensi · 15 menit terakhir (bucket 0,25 ms)", hist)
    y3 = y2 + h2 + G
    h3 = H - y3 - M
    p.chart("drift_psi", x1, y3, w1, h3, "clusteredBarChart",
            {"Category": [(col("v_drift_top", "feature"), "Fitur")],
             "Y": [(agg("v_drift_top", "psi", "max"), "PSI")]},
            "Top 10 drift fitur vs training (hijau stabil · kuning moderat · merah signifikan)",
            bars(fill_measure=("v_drift_top", "Drift Color")), sort=(agg("v_drift_top", "psi", "max"), "Descending"))
    sd = axis_fmt()
    sd["dataPoint"] = value_colors("v_score_distribution", "label", {"Legit": BLUE, "Fraud": ORANGE, "Pending": NEUTRAL})
    sd.update(legend(True))
    p.chart("score_dist", x2, y3, w2, h3, "hundredPercentStackedColumnChart",
            {"Category": [(col("v_score_distribution", "score_from"), "Skor")],
             "Series": [(col("v_score_distribution", "label"), "Label")],
             "Y": [(agg("v_score_distribution", "n"), "Transaksi")]},
            "Komposisi label per bucket skor · kalibrasi live (oranye = fraud terkonfirmasi)", sd)
    return p


def page_research() -> Page:
    p = Page("research", "Research & Trade-offs",
             "Hasil eksperimen offline E1–E11 · seleksi model di valid, pelaporan di test out-of-time")
    y1 = HEADER_H + 14
    p.kpis(y1, [
        ("kpi_best_auprc", meas("r_e9_variants", "Best Valid AUPRC"), "Valid AUPRC terbaik", BLUE),
        ("kpi_rules", meas("r_e1_model_comparison", "Rules Baseline AUPRC"), "Baseline aturan manual (test)", NEUTRAL),
        ("kpi_fastest", meas("r_e9_variants", "Fastest Model p99 (ms)"), "Model tercepat · p99 (ms)", VIOLET),
        ("kpi_peak", meas("r_e11_loadtest", "Peak Throughput (req/s)"), "Throughput puncak / worker", ORANGE),
    ])
    y2, h2 = y1 + 84 + G, 290
    lw = 400
    p.chart("e1_models", M, y2, lw, h2, "clusteredBarChart",
            {"Category": [(col("r_e1_model_comparison", "model"), "Model")],
             "Y": [(agg("r_e1_model_comparison", "test_auprc", "max"), "Test AUPRC")]},
            "E1 · Test AUPRC per model", bars(fill_measure=("r_e1_model_comparison", "Model Color")),
            sort=(agg("r_e1_model_comparison", "test_auprc", "max"), "Descending"))
    sc = axis_fmt()
    sc["dataPoint"] = [{"properties": {"fill": color_by("v_e9_plot", "Pareto Color")}, "selector": WILDCARD}]
    sc["categoryLabels"] = [{"properties": {"show": lit(True), "fontSize": lit(8), "color": color(INK2)}}]
    sc["bubbles"] = [{"properties": {"bubbleSize": lit(-8)}}]
    sc.update(legend(False))
    p.chart("e9_pareto", M + lw + G, y2, W - 2 * M - lw - G, h2, "scatterChart",
            {"Category": [(col("v_e9_plot", "variant"), "Varian")],
             "X": [(agg("v_e9_plot", "log10_p99_ms", "max"), "log10 p99 (ms)")],
             "Y": [(agg("v_e9_plot", "valid_auprc", "max"), "Valid AUPRC")]},
            "E9 · Akurasi vs latensi · X = log10 p99 ms (−1 = 0,1 ms · 0 = 1 ms · 1 = 10 ms) · biru = Pareto", sc)
    y3 = y2 + h2 + G
    h3 = H - y3 - M
    (x1, w1), (x2, w2) = grid_cols(2)
    e10 = axis_fmt()
    e10["dataPoint"] = series_colors({"Sum(r_e10_end_to_end.features_mean_ms)": BLUE,
                                      "Sum(r_e10_end_to_end.model_mean_ms)": ORANGE,
                                      "Sum(r_e10_end_to_end.explain_mean_ms)": AQUA})
    e10.update(legend(True))
    p.chart("e10_e2e", x1, y3, w1, h3, "barChart",
            {"Category": [(col("r_e10_end_to_end", "mode"), "Strategi")],
             "Y": [(agg("r_e10_end_to_end", "features_mean_ms"), "features"),
                   (agg("r_e10_end_to_end", "model_mean_ms"), "model"),
                   (agg("r_e10_end_to_end", "explain_mean_ms"), "explain")]},
            "E10 · Biaya end-to-end per strategi explainability (ms)", e10)
    qw = (w2 - G) // 2
    endpoint_colors = {"/score": BLUE, "/score/async": ORANGE}  # colour follows the endpoint
    for i, (name, colname, title, logy) in enumerate([
            ("e11_throughput", "throughput_rps", "E11 · Throughput (req/s)", False),
            ("e11_p99", "p99_ms", "E11 · Client p99 latency (ms)", False)]):
        o = lines({}, log_y=logy)
        o["dataPoint"] = value_colors("v_e11_plot", "endpoint", endpoint_colors)
        o["lineStyles"] = [{"properties": {"strokeWidth": lit(2), "showMarker": lit(True), "lineChartType": lit("linear")}}]
        o.update(legend(True))
        p.chart(name, x2 + i * (qw + G), y3, qw, h3, "lineChart",
                {"Category": [(col("v_e11_plot", "clients"), "Klien konkuren")],
                 "Series": [(col("v_e11_plot", "endpoint"), "Endpoint")],
                 "Y": [(agg("v_e11_plot", colname, "max"), colname)]}, title, o)
    return p


# ============================================================================================ report
def build_report(pages: list[Page]) -> None:
    dump(RP / "definition.pbir", {"$schema": f"{SCHEMA}/item/report/definitionProperties/2.0.0/schema.json",
                                  "version": "4.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}}})
    d = RP / "definition"
    dump(d / "version.json", {"$schema": f"{SCHEMA}/item/report/definition/versionMetadata/1.0.0/schema.json",
                              "version": "2.0.0"})
    at_import = {"visual": V_VISUAL, "report": V_REPORT, "page": V_PAGE}
    dump(d / "report.json", {
        "$schema": f"{SCHEMA}/item/report/definition/report/{V_REPORT}/schema.json",
        # Styling is explicit on every visual. A RegisteredResources custom theme generated outside Desktop is
        # silently dropped by Power BI Desktop 2.159 (even though it validates against reportThemeSchema-2.159),
        # so fraud_ops_theme.json is shipped for manual import (View > Themes > Browse) instead.
        "themeCollection": {"baseTheme": {"name": BASE_THEME, "reportVersionAtImport": at_import, "type": "SharedResources"}},
        "resourcePackages": [
            {"name": "SharedResources", "type": "SharedResources",
             "items": [{"name": BASE_THEME, "path": f"BaseThemes/{BASE_THEME}.json", "type": "BaseTheme"}]}],
        "settings": {"useStylableVisualContainerHeader": True, "defaultDrillFilterOtherVisuals": True},
    })
    dump(d / "pages" / "pages.json", {"$schema": f"{SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json",
                                      "pageOrder": [p.name for p in pages], "activePageName": pages[0].name})
    for p in pages:
        p.write(d)
    sr = RP / "StaticResources"
    (sr / "SharedResources" / "BaseThemes").mkdir(parents=True, exist_ok=True)
    shutil.copy(PBI_THEMES / f"{BASE_THEME}.json", sr / "SharedResources" / "BaseThemes" / f"{BASE_THEME}.json")


def main(active_page: str | None = None, only: bool = False):
    for p in (SM, RP):
        if p.exists():
            shutil.rmtree(p)
    conn = connect(load_config()["db_url"])
    build_semantic_model(conn)
    pages = [page_live(), page_business(), page_health(), page_research()]
    if active_page:
        pages.sort(key=lambda p: p.name != active_page)
    if only:  # verification build: just one page
        pages = pages[:1]
    build_report(pages)
    if active_page and not only:  # keep the canonical tab order, only change the page that opens first
        order = ["live_operations", "business_impact", "model_health", "research"]
        dump(RP / "definition" / "pages" / "pages.json",
             {"$schema": f"{SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json",
              "pageOrder": order, "activePageName": active_page})
    dump(OUT / f"{NAME}.pbip", {"$schema": f"{SCHEMA}/pbip/pbipProperties/1.0.0/schema.json", "version": "1.0",
                                "artifacts": [{"report": {"path": f"{NAME}.Report"}}],
                                "settings": {"enableAutoRecovery": False}})
    write(OUT / ".gitignore", "**/.pbi/localSettings.json\n**/.pbi/cache.abf\n")
    n_vis = sum(len(p.visuals) for p in pages)
    print(f"[pbip] {OUT / (NAME + '.pbip')}  ({len(pages)} pages, {n_vis} visuals, "
          f"{len(LIVE_TABLES) + len(RESEARCH_TABLES)} DirectQuery tables, {len(MEASURES)} measures)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None, only="--only" in sys.argv)
