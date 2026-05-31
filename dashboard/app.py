from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from intelligence_engine import (
    ALERT_LEVEL_META,
    build_intelligence_snapshot,
    load_raw_csv,
)


st.set_page_config(
    page_title="Plataforma de Inteligencia de Conflicto",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_diario.csv"
STATUS_PATH = BASE_DIR / "data" / "processed" / "pipeline_status.json"
RSS_PATH = BASE_DIR / "data" / "raw" / "rss" / "rss_latest.csv"
WIKI_RAW_PATH = BASE_DIR / "data" / "raw" / "wikimedia" / "wikimedia_pageviews_raw.csv"
REVISION_RAW_PATH = BASE_DIR / "data" / "raw" / "wikipedia_revisions" / "wikipedia_revisions_raw.csv"
MODEL_PATH = BASE_DIR / "models" / "best_escalation_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"

BG = "#06131a"
SURFACE = "#0c1f29"
SURFACE_ALT = "#102833"
OUTLINE = "rgba(112, 211, 255, 0.16)"
CYAN = "#55d6ff"
CYAN_SOFT = "#2ea8c6"
RED = "#ff5c5c"
AMBER = "#ffb347"
GREEN = "#49dcb1"
INK = "#e8f5f7"
MUTED = "#87a9b5"


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');

        :root {
            --bg: #06131a;
            --surface: rgba(12, 31, 41, 0.88);
            --surface-strong: rgba(16, 40, 51, 0.94);
            --outline: rgba(112, 211, 255, 0.16);
            --cyan: #55d6ff;
            --cyan-soft: #2ea8c6;
            --red: #ff5c5c;
            --amber: #ffb347;
            --green: #49dcb1;
            --ink: #e8f5f7;
            --muted: #87a9b5;
        }

        .stApp {
            background:
                radial-gradient(circle at 14% 18%, rgba(85, 214, 255, 0.12), transparent 20%),
                radial-gradient(circle at 82% 12%, rgba(255, 92, 92, 0.10), transparent 22%),
                radial-gradient(circle at 50% 100%, rgba(255, 179, 71, 0.07), transparent 28%),
                linear-gradient(180deg, #030b10 0%, #07131a 38%, #061017 100%);
            color: var(--ink);
            font-family: "IBM Plex Sans", sans-serif;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
                repeating-linear-gradient(
                    0deg,
                    rgba(85, 214, 255, 0.00) 0px,
                    rgba(85, 214, 255, 0.00) 34px,
                    rgba(85, 214, 255, 0.03) 35px
                ),
                repeating-linear-gradient(
                    90deg,
                    rgba(85, 214, 255, 0.00) 0px,
                    rgba(85, 214, 255, 0.00) 34px,
                    rgba(85, 214, 255, 0.03) 35px
                );
            opacity: 0.32;
            z-index: 0;
        }

        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background: linear-gradient(
                180deg,
                rgba(85, 214, 255, 0.00) 0%,
                rgba(85, 214, 255, 0.035) 48%,
                rgba(85, 214, 255, 0.00) 100%
            );
            animation: scan-sweep 10s linear infinite;
            z-index: 0;
        }

        @keyframes scan-sweep {
            0% { transform: translateY(-100%); }
            100% { transform: translateY(100%); }
        }

        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stDeployButton,
        [data-testid="stDecoration"] {
            display: none !important;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            position: relative;
            z-index: 1;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(7, 22, 29, 0.98) 0%, rgba(6, 17, 23, 0.98) 100%);
            border-right: 1px solid rgba(112, 211, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        h1, h2, h3 {
            font-family: "Rajdhani", sans-serif;
            color: var(--ink);
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(12, 31, 41, 0.95), rgba(9, 23, 31, 0.92));
            border: 1px solid rgba(112, 211, 255, 0.14);
            border-radius: 18px;
            padding: 14px 14px 12px;
            box-shadow: inset 0 0 0 1px rgba(85, 214, 255, 0.03), 0 16px 32px rgba(0, 0, 0, 0.26);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-family: "Space Mono", monospace;
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-family: "Rajdhani", sans-serif;
            font-weight: 700;
            letter-spacing: 0.03em;
        }

        .war-shell {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .command-ribbon {
            display: grid;
            grid-template-columns: 1.6fr 1fr 1fr 1fr;
            gap: 12px;
        }

        .command-tile {
            position: relative;
            overflow: hidden;
            border-radius: 16px;
            padding: 14px 16px 12px;
            background: linear-gradient(180deg, rgba(13, 31, 39, 0.92), rgba(8, 21, 28, 0.92));
            border: 1px solid rgba(112, 211, 255, 0.12);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.24);
        }

        .command-tile::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, transparent 0%, rgba(85, 214, 255, 0.05) 40%, transparent 70%);
            animation: tile-sweep 7s ease-in-out infinite;
            pointer-events: none;
        }

        @keyframes tile-sweep {
            0%, 100% { transform: translateX(-100%); }
            55% { transform: translateX(120%); }
        }

        .command-kicker {
            color: var(--cyan);
            font-family: "Space Mono", monospace;
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .command-value {
            font-family: "Rajdhani", sans-serif;
            font-size: 1.8rem;
            line-height: 1;
            margin-top: 10px;
        }

        .command-copy {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 6px;
            line-height: 1.4;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.05fr 1.35fr 0.9fr;
            gap: 14px;
            align-items: stretch;
        }

        .hero-panel, .surface-panel {
            position: relative;
            overflow: hidden;
            border-radius: 22px;
            background:
                linear-gradient(180deg, rgba(12, 31, 41, 0.94), rgba(8, 21, 29, 0.95));
            border: 1px solid rgba(112, 211, 255, 0.12);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
        }

        .hero-panel::before, .surface-panel::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(135deg, rgba(85, 214, 255, 0.06), transparent 35%),
                linear-gradient(315deg, rgba(255, 92, 92, 0.05), transparent 30%);
            pointer-events: none;
        }

        .hero-panel-body {
            position: relative;
            z-index: 1;
            padding: 22px 22px 18px;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(85, 214, 255, 0.08);
            border: 1px solid rgba(85, 214, 255, 0.20);
            color: var(--cyan);
            font-family: "Space Mono", monospace;
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 14px;
        }

        .hero-title {
            font-family: "Rajdhani", sans-serif;
            font-size: 3rem;
            line-height: 0.92;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 10px;
        }

        .hero-copy {
            color: #b4d1d8;
            font-size: 1rem;
            line-height: 1.6;
            max-width: 95%;
            margin-bottom: 18px;
        }

        .hero-chip-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }

        .hero-chip {
            border-radius: 16px;
            padding: 12px 12px 10px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(112, 211, 255, 0.12);
        }

        .hero-chip .label {
            color: var(--muted);
            font-family: "Space Mono", monospace;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero-chip .value {
            color: var(--ink);
            font-family: "Rajdhani", sans-serif;
            font-size: 1.5rem;
            margin-top: 6px;
            line-height: 1.05;
        }

        .radar-shell {
            position: relative;
            height: 240px;
            border-radius: 20px;
            background:
                radial-gradient(circle at center, rgba(85, 214, 255, 0.15) 0%, rgba(85, 214, 255, 0.03) 42%, transparent 43%),
                radial-gradient(circle at center, rgba(85, 214, 255, 0.12) 0%, transparent 65%);
            border: 1px solid rgba(112, 211, 255, 0.12);
            overflow: hidden;
        }

        .radar-shell::before,
        .radar-shell::after {
            content: "";
            position: absolute;
            inset: 8%;
            border-radius: 999px;
            border: 1px solid rgba(85, 214, 255, 0.14);
        }

        .radar-shell::after {
            inset: 22%;
        }

        .radar-sweep {
            position: absolute;
            inset: -10%;
            background: conic-gradient(from 0deg, rgba(85,214,255,0.00) 0deg, rgba(85,214,255,0.20) 35deg, rgba(85,214,255,0.00) 90deg);
            animation: radar-spin 8s linear infinite;
        }

        @keyframes radar-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .radar-dot {
            position: absolute;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: var(--red);
            box-shadow: 0 0 0 0 rgba(255, 92, 92, 0.65);
            animation: pulse-alert 2.2s infinite;
        }

        .radar-dot.dot-a { top: 26%; left: 62%; }
        .radar-dot.dot-b { top: 56%; left: 38%; background: var(--amber); }
        .radar-dot.dot-c { top: 68%; left: 72%; background: var(--green); }

        @keyframes pulse-alert {
            0% { box-shadow: 0 0 0 0 rgba(255, 92, 92, 0.55); opacity: 0.85; }
            70% { box-shadow: 0 0 0 16px rgba(255, 92, 92, 0.0); opacity: 1; }
            100% { box-shadow: 0 0 0 0 rgba(255, 92, 92, 0.0); opacity: 0.8; }
        }

        .panel-title {
            font-family: "Rajdhani", sans-serif;
            font-size: 1.15rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--ink);
            margin-bottom: 0.25rem;
        }

        .panel-note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
            margin-bottom: 0.9rem;
        }

        .alert-stack {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .alert-card {
            border-radius: 16px;
            padding: 12px 14px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(112, 211, 255, 0.10);
        }

        .alert-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
        }

        .alert-tag {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-family: "Space Mono", monospace;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .alert-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            animation: blink 1.8s infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 0.45; }
            50% { opacity: 1; }
        }

        .alert-title {
            color: var(--ink);
            font-weight: 600;
            font-size: 0.98rem;
            margin-bottom: 4px;
        }

        .alert-copy {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .alert-metric {
            color: var(--ink);
            font-family: "Space Mono", monospace;
            font-size: 0.78rem;
            margin-top: 8px;
        }

        .section-grid {
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 14px;
            margin-top: 14px;
        }

        .surface-body {
            position: relative;
            z-index: 1;
            padding: 20px 20px 18px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            min-height: 46px;
            padding: 0 16px;
            border-radius: 999px;
            background: rgba(16, 40, 51, 0.88);
            border: 1px solid rgba(112, 211, 255, 0.16);
            color: var(--ink) !important;
            box-shadow: inset 0 0 0 1px rgba(85, 214, 255, 0.03);
        }

        .stTabs [data-baseweb="tab"] p {
            color: var(--ink) !important;
            font-family: "Space Mono", monospace;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0;
            white-space: nowrap;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(85, 214, 255, 0.18), rgba(255, 92, 92, 0.16));
            border-color: rgba(85, 214, 255, 0.38);
            box-shadow: 0 0 0 1px rgba(85, 214, 255, 0.18), 0 0 18px rgba(85, 214, 255, 0.08);
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background: transparent !important;
        }

        .stDataFrame, [data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            font-family: "Space Mono", monospace;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            background: rgba(85, 214, 255, 0.08);
            border: 1px solid rgba(85, 214, 255, 0.15);
            color: var(--ink);
            margin-top: 6px;
        }

        .source-panel {
            border-radius: 18px;
            padding: 14px;
            margin-bottom: 10px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(112, 211, 255, 0.10);
        }

        .source-panel strong {
            display: block;
            color: var(--ink);
            font-size: 1rem;
            margin-bottom: 4px;
        }

        .source-panel span {
            color: var(--muted);
            font-size: 0.86rem;
        }

        .method-note {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.5;
        }

        @media (max-width: 1200px) {
            .hero-grid, .section-grid, .command-ribbon {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "No existe data/processed/dataset_diario.csv. Ejecuta ./update_data.ps1."
        )
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


@st.cache_data
def load_pipeline_status():
    if not STATUS_PATH.exists():
        return None
    with open(STATUS_PATH, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


@st.cache_data
def build_snapshot(
    dataset_df: pd.DataFrame,
    rss_df: pd.DataFrame,
    wikimedia_raw_df: pd.DataFrame,
    revisions_raw_df: pd.DataFrame,
    metrics: dict | None,
):
    artifact = load_model_artifact()
    return build_intelligence_snapshot(
        dataset_df=dataset_df,
        rss_df=rss_df,
        wikimedia_raw_df=wikimedia_raw_df,
        revisions_raw_df=revisions_raw_df,
        model_artifact=artifact,
        metrics=metrics,
    )


def apply_date_window(frame: pd.DataFrame, date_window: tuple[pd.Timestamp, pd.Timestamp], date_column: str = "date") -> pd.DataFrame:
    if frame.empty or date_column not in frame.columns:
        return frame
    start_date, end_date = date_window
    filtered = frame.copy()
    filtered[date_column] = pd.to_datetime(filtered[date_column], errors="coerce")
    return filtered[(filtered[date_column] >= start_date) & (filtered[date_column] <= end_date)]


def render_dashboard_filters(dataset_df: pd.DataFrame, rss_df: pd.DataFrame, snapshot) -> dict:
    min_date = dataset_df["date"].min().date()
    max_date = dataset_df["date"].max().date()
    source_options = sorted(rss_df["source"].dropna().unique().tolist()) if not rss_df.empty else []
    region_options = snapshot.region_snapshot["region"].tolist() if not snapshot.region_snapshot.empty else []
    theme_options = ["Military", "Energy", "Diplomacy", "Risk"]

    with st.sidebar:
        st.markdown("## Filtros de misión")
        selected_range = st.date_input(
            "Rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date, end_date = min_date, max_date

        # UX helpers: botones para seleccionar / limpiar todo
        col_a, col_b = st.columns([1, 1])
        if col_a.button("Seleccionar todo"):
            selected_sources = source_options.copy()
            selected_regions = region_options.copy()
            selected_themes = theme_options.copy()
        elif col_b.button("Borrar selección"):
            selected_sources = []
            selected_regions = []
            selected_themes = []
        else:
            selected_sources = st.multiselect("Fuentes", source_options, default=source_options) if source_options else []
            selected_regions = st.multiselect("Regiones", region_options, default=region_options) if region_options else []
            selected_themes = st.multiselect("Categorías", theme_options, default=theme_options)

        # Mensajes cuando no hay opciones
        if not source_options:
            st.warning("No hay fuentes de noticias disponibles en data/raw/rss/rss_latest.csv")
        if not region_options:
            st.info("No hay regiones detectadas aún en el snapshot; revisa la ingesta de datos.")

        # Expander con metodología y fuentes resumidas
        with st.expander("Metodología y fuentes (resumen)", expanded=False):
            st.markdown(
                """
                - Unidad de análisis: día-región (ventana diaria agregada por país/región).
                - Fuentes activas detectadas en el proyecto: RSS (titulares), Wikimedia pageviews, Wikipedia revisions.
                - Modelado: clasificación/regresión sobre un `media_pressure_score` construido desde el corpus RSS y señales complementarias.
                - Estado del pipeline: revisa `data/processed/pipeline_status.json` para detalles de ingestión y cobertura.
                """,
                unsafe_allow_html=True,
            )

    return {
        "date_window": (pd.Timestamp(start_date), pd.Timestamp(end_date)),
        "sources": selected_sources,
        "regions": selected_regions,
        "themes": selected_themes,
    }


def render_sidebar(status_payload: dict | None, snapshot):
    with st.sidebar:
        st.markdown("## Panel de control")
        st.caption("Plataforma local de vigilancia geopolítica basada exclusivamente en las fuentes del proyecto.")

        threat_color = snapshot.executive["threat_color"]
        st.markdown(
            f"""
            <div class="status-chip">
                <span class="alert-dot" style="background:{threat_color};"></span>
                Estado de amenaza · {snapshot.executive['threat_level']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if status_payload:
            dataset_summary = status_payload.get("dataset_summary", {})
            st.markdown("### Estado del pipeline")
            st.write(f"Última ejecución: `{status_payload.get('finished_at', 'N/D')}`")
            st.write(
                f"Cobertura: `{dataset_summary.get('date_min', 'N/D')}` a `{dataset_summary.get('date_max', 'N/D')}`"
            )
            st.write(f"Filas: `{dataset_summary.get('rows', 'N/D')}`")

            st.markdown("### Disponibilidad de fuentes")
            for label, key in [
                ("Wikimedia Pageviews", "wikimedia"),
                ("Wikipedia Revisions", "wikipedia_revisions"),
                ("RSS / News Feeds", "rss"),
                ("GDELT", "gdelt"),
                ("ACLED", "acled"),
                ("NASA FIRMS", "firms"),
            ]:
                payload = status_payload.get("sources", {}).get(key, {"status": "skipped"})
                rows = payload.get("rows", "N/D")
                st.markdown(
                    f"""
                    <div class="source-panel">
                        <strong>{label}</strong>
                        <span>Status: {payload.get('status', 'unknown').upper()}</span><br/>
                        <span>Rows: {rows}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Alcance analítico")
        st.markdown(
            """
            <div class="method-note">
            Las señales militares, energéticas y regionales se derivan del corpus RSS disponible y
            de los temas monitorizados en Wikipedia. No se introducen eventos sintéticos, precios de
            petróleo inventados ni series regionales fabricadas.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Si quieres correlaciones reales con precios del petróleo o volatilidad de mercado, agrega una serie temporal válida (p. ej. Brent/WTI)."
        )


def plot_conflict_map(region_snapshot: pd.DataFrame) -> go.Figure:
    active = region_snapshot.copy()
    active["label"] = active["region"] + "<br>Mentions 7d: " + active["recent_mentions"].round(1).astype(str)

    fig = go.Figure()
    if not active.empty:
        hottest = active.sort_values("risk_intensity", ascending=False).head(4).reset_index(drop=True)
        if len(hottest) > 1:
            anchor = hottest.iloc[0]
            for _, row in hottest.iloc[1:].iterrows():
                fig.add_trace(
                    go.Scattergeo(
                        lon=[anchor["lon"], row["lon"]],
                        lat=[anchor["lat"], row["lat"]],
                        mode="lines",
                        line=dict(width=1.2 + 2.5 * row["risk_intensity"], color="rgba(85,214,255,0.25)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        fig.add_trace(
            go.Scattergeo(
                lon=active["lon"],
                lat=active["lat"],
                text=active["region"],
                customdata=active[["recent_mentions", "wiki_views_7d", "wiki_revisions_7d", "latest_headline"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Headline mentions (7d): %{customdata[0]:.0f}<br>"
                    "Wiki views (7d): %{customdata[1]:.0f}<br>"
                    "Wiki revisions (7d): %{customdata[2]:.0f}<br>"
                    "Latest headline: %{customdata[3]}<extra></extra>"
                ),
                mode="markers+text",
                textposition="top center",
                marker=dict(
                    size=active["marker_size"],
                    color=active["risk_intensity"],
                    colorscale=[
                        [0.0, "#1d6f8a"],
                        [0.5, "#36b5da"],
                        [0.75, "#ffb347"],
                        [1.0, "#ff5c5c"],
                    ],
                    line=dict(width=1.2, color="rgba(255,255,255,0.6)"),
                    opacity=0.92,
                    showscale=True,
                    colorbar=dict(
                        title="Risk",
                        thickness=10,
                        len=0.7,
                        x=0.98,
                        bgcolor="rgba(6,19,26,0.4)",
                        tickfont=dict(color=INK),
                        titlefont=dict(color=INK),
                    ),
                ),
                textfont=dict(color="#dff9ff", family="IBM Plex Sans", size=11),
                showlegend=False,
            )
        )

    fig.update_geos(
        scope="asia",
        projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)",
        landcolor="#0d2430",
        oceancolor="#07131a",
        showcountries=True,
        countrycolor="rgba(125, 196, 220, 0.28)",
        coastlinecolor="rgba(125, 196, 220, 0.12)",
        showocean=True,
        showlakes=False,
        lataxis=dict(showgrid=True, gridcolor="rgba(85,214,255,0.12)", dtick=5, range=[10, 42]),
        lonaxis=dict(showgrid=True, gridcolor="rgba(85,214,255,0.12)", dtick=10, range=[25, 65]),
        center=dict(lat=28, lon=45),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=510,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_theme_activity(theme_daily: pd.DataFrame, selected_themes: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    colors = {
        "theme_military": RED,
        "theme_energy": AMBER,
        "theme_diplomacy": GREEN,
        "theme_risk": CYAN,
    }
    labels = {
        "theme_military": "Military",
        "theme_energy": "Energy",
        "theme_diplomacy": "Diplomacy",
        "theme_risk": "Risk",
    }
    allowed = set(selected_themes or labels.values())
    for column in ["theme_military", "theme_energy", "theme_diplomacy", "theme_risk"]:
        if column not in theme_daily.columns or labels[column] not in allowed:
            continue
        fig.add_trace(
            go.Scatter(
                x=theme_daily["date"],
                y=theme_daily[column],
                mode="lines+markers",
                name=labels[column],
                line=dict(color=colors[column], width=2.4),
                marker=dict(size=5),
            )
        )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        xaxis_title="",
        yaxis_title="Keyword hits",
        font=dict(color=INK),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def plot_pressure_and_forecast(dataset_df: pd.DataFrame, snapshot) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataset_df["date"],
            y=dataset_df["media_pressure_score"],
            mode="lines",
            name="Observed pressure score",
            line=dict(color=CYAN, width=2.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset_df["date"],
            y=dataset_df["media_pressure_score"].rolling(14, min_periods=1).mean(),
            mode="lines",
            name="Rolling 14d mean",
            line=dict(color="rgba(255,255,255,0.45)", width=1.7, dash="dot"),
        )
    )
    threshold = float(dataset_df["target_threshold_q75"].iloc[0])
    fig.add_hline(
        y=threshold,
        line_color=AMBER,
        line_dash="dash",
        annotation_text="Escalation threshold",
        annotation_font_color=AMBER,
    )

    anomaly_frame = snapshot.anomaly_frame
    if not anomaly_frame.empty:
        flagged = anomaly_frame[anomaly_frame["anomaly_flag"]]
        if not flagged.empty:
            fig.add_trace(
                go.Scatter(
                    x=flagged["date"],
                    y=flagged["media_pressure_score"],
                    mode="markers",
                    name="Anomalies",
                    marker=dict(color=RED, size=9, line=dict(color="#fff", width=1)),
                )
            )

    forecast = snapshot.forecast_frame
    if not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["upper_bound"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["lower_bound"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(85,214,255,0.12)",
                name="Forecast confidence band",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["forecast_score"],
                mode="lines+markers",
                name="Forecast score",
                line=dict(color=RED, width=2.5, dash="dash"),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        xaxis_title="",
        yaxis_title="Pressure score",
        font=dict(color=INK),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def plot_region_risk_bars(region_snapshot: pd.DataFrame) -> go.Figure:
    top = region_snapshot.sort_values("risk_intensity", ascending=True).tail(8)
    fig = px.bar(
        top,
        x="risk_intensity",
        y="region",
        orientation="h",
        color="risk_intensity",
        color_continuous_scale=["#1d6f8a", "#36b5da", "#ffb347", "#ff5c5c"],
        text="recent_mentions",
    )
    fig.update_traces(texttemplate="%{text:.0f} mentions", textposition="outside")
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        xaxis_title="Risk intensity",
        yaxis_title="",
        coloraxis_showscale=False,
        font=dict(color=INK),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=False)
    return fig


def plot_regime_timeline(regime_frame: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        regime_frame,
        x="date",
        y="regime_label",
        color="regime_label",
        hover_data={"regime_id": True, "date": True},
        color_discrete_sequence=["#55d6ff", "#49dcb1", "#ffb347", "#ff5c5c"],
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        xaxis_title="",
        yaxis_title="Detected regime",
        font=dict(color=INK),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


def plot_model_comparison(model_metrics: pd.DataFrame) -> go.Figure:
    melted = model_metrics.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="value",
    ).dropna()
    fig = px.bar(
        melted,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        color_discrete_sequence=[CYAN, GREEN, AMBER, RED, "#a1f0ff"],
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        xaxis_title="",
        yaxis_title="Score",
        font=dict(color=INK),
        legend=dict(orientation="h", y=1.04, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def plot_feature_importance(feature_importance_frame: pd.DataFrame) -> go.Figure:
    top = feature_importance_frame.head(10).sort_values("importance", ascending=True)
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=[CYAN],
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        xaxis_title="Feature importance",
        yaxis_title="",
        font=dict(color=INK),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=False)
    return fig


def plot_shap_contributions(shap_frame: pd.DataFrame) -> go.Figure:
    top = shap_frame.head(10).sort_values("shap_value", ascending=True)
    colors = [GREEN if value < 0 else RED for value in top["shap_value"]]
    fig = go.Figure(
        go.Bar(
            x=top["shap_value"],
            y=top["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.3f}" for value in top["shap_value"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        xaxis_title="SHAP contribution to escalation probability",
        yaxis_title="",
        font=dict(color=INK),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.18)")
    fig.update_yaxes(showgrid=False)
    return fig


def render_command_ribbon(snapshot, status_payload):
    execu = snapshot.executive
    last_run = status_payload.get("finished_at", "N/D") if status_payload else "N/D"
    weekly_text = f"{execu['weekly_trend']:+.1f}"
    st.markdown(
        f"""
        <div class="command-ribbon">
            <div class="command-tile">
                <div class="command-kicker">Condición de amenaza</div>
                <div class="command-value" style="color:{execu['threat_color']};">{execu['threat_level'].upper()}</div>
                <div class="command-copy">Probabilidad de escalada {execu['escalation_probability']:.1%} · confianza del modelo {execu['model_confidence']:.1%}</div>
            </div>
            <div class="command-tile">
                <div class="command-kicker">Intensidad del conflicto</div>
                <div class="command-value">{execu['conflict_intensity']:.1f}</div>
                <div class="command-copy">Puntaje observado vs umbral de escalada {execu['score_threshold']:.1f}</div>
            </div>
            <div class="command-tile">
                <div class="command-kicker">Punto caliente regional</div>
                <div class="command-value">{execu['regional_hotspot']}</div>
                <div class="command-copy">Proxy de riesgo regional {execu['regional_risk_proxy']:.2f}</div>
            </div>
            <div class="command-tile">
                <div class="command-kicker">Latido del pipeline</div>
                <div class="command-value">{weekly_text}</div>
                <div class="command-copy">Cambio semanal · última actualización {last_run}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(snapshot, filters: dict):
    execu = snapshot.executive
    filtered_region_snapshot = snapshot.region_snapshot[
        snapshot.region_snapshot["region"].isin(filters["regions"])
    ] if filters["regions"] else snapshot.region_snapshot
    radar_html = """
        <div class="radar-shell">
            <div class="radar-sweep"></div>
            <div class="radar-dot dot-a"></div>
            <div class="radar-dot dot-b"></div>
            <div class="radar-dot dot-c"></div>
        </div>
    """
    alerts_html = "".join(
        f"""
        <div class="alert-card">
            <div class="alert-head">
                <div class="alert-tag">
                    <span class="alert-dot" style="background:{alert['color']};"></span>
                    {alert['label']}
                </div>
                <div style="color:{MUTED}; font-family:'Space Mono', monospace; font-size:0.72rem;">{alert['timestamp']}</div>
            </div>
            <div class="alert-title">{alert['title']}</div>
            <div class="alert-copy">{alert['description']}</div>
            <div class="alert-metric">{alert['metric']}</div>
        </div>
        """
        for alert in snapshot.alerts
    )
    st.markdown(
        f"""
        <div class="hero-grid">
            <div class="hero-panel">
                <div class="hero-panel-body">
                    <div class="hero-badge">Plataforma de inteligencia · Irán / Israel / EE. UU. / energía / escalada</div>
                    <div class="hero-title">Sala de monitoreo y escalada geopolítica</div>
                    <div class="hero-copy">
                        Vista operativa en tiempo real de atención pública, presión de titulares, churn editorial, detección de anomalías
                        y riesgo de escalada estimado por modelos. Centrado en el teatro de conflicto Irán‑Israel‑EE. UU.
                    </div>
                    <div class="hero-chip-grid">
                        <div class="hero-chip">
                            <div class="label">Probabilidad de escalada</div>
                            <div class="value">{execu['escalation_probability']:.1%}</div>
                        </div>
                        <div class="hero-chip">
                            <div class="label">Pico pronosticado</div>
                            <div class="value">{execu['forecast_peak']:.1f}</div>
                        </div>
                        <div class="hero-chip">
                            <div class="label">Señal militar</div>
                            <div class="value">{execu['military_signal']}</div>
                        </div>
                        <div class="hero-chip">
                            <div class="label">Señal energética</div>
                            <div class="value">{execu['energy_signal']}</div>
                        </div>
                        <div class="hero-chip">
                            <div class="label">Conteo temas de riesgo</div>
                            <div class="value">{execu['risk_signal']}</div>
                        </div>
                        <div class="hero-chip">
                            <div class="label">Fecha pico pronosticada</div>
                            <div class="value">{execu['forecast_peak_date']}</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="hero-panel">
                <div class="hero-panel-body">
                    <div class="panel-title">Mapa del teatro de conflicto</div>
                    <div class="panel-note">
                        Capa operativa regional basada en menciones regionales observadas en RSS y temas monitorizados en Wikipedia.
                    </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_conflict_map(filtered_region_snapshot), use_container_width=True)
    st.markdown('<div class="panel-note">Mapa geográfico de atención regional: tamaño y color indican intensidad de riesgo estimada; pase el cursor para ver métricas recientes y titulares.</div>', unsafe_allow_html=True)
    st.markdown(
        """
                </div>
            </div>
            <div class="hero-panel">
                <div class="hero-panel-body">
                    <div class="panel-title">Panel de Alertas Operativas</div>
                    <div class="panel-note">
                        Resumen operativo: perspectiva de escalada, banderas de anomalía, hotspots regionales y estrés en corredores energéticos.
                    </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(radar_html, unsafe_allow_html=True)
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alert-stack">{alerts_html}</div>', unsafe_allow_html=True)
    st.markdown(
        """
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_exec_metrics(snapshot):
    execu = snapshot.executive
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Nivel de amenaza", execu["threat_level"])
    col2.metric("Proxy de riesgo global", f"{execu['global_risk_proxy']:.1f}")
    col3.metric("Tendencia semanal", f"{execu['weekly_trend']:+.1f}")
    col4.metric("Tendencia mensual", f"{execu['monthly_trend']:+.1f}")
    col5.metric("Confianza del modelo", f"{execu['model_confidence']:.1%}")
    col6.metric("Score de anomalía", f"{execu['anomaly_score']:.3f}")


def render_overview_tab(dataset_df: pd.DataFrame, snapshot, filters: dict):
    filtered_dataset = apply_date_window(dataset_df, filters["date_window"])
    filtered_theme_daily = apply_date_window(snapshot.theme_daily, filters["date_window"])
    filtered_regime_frame = apply_date_window(snapshot.regime_frame, filters["date_window"])
    filtered_region_snapshot = snapshot.region_snapshot[
        snapshot.region_snapshot["region"].isin(filters["regions"])
    ] if filters["regions"] else snapshot.region_snapshot

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Presión de escalada observada y forecast a corto plazo</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Puntaje observado, marcas de anomalía y forecast autoregresivo a 10 días con banda de confianza construida a partir de residuos históricos.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_pressure_and_forecast(filtered_dataset, snapshot), use_container_width=True)
        st.markdown('<div class="panel-note">Score observado de presión mediática con marcadores de anomalía y forecast a 10 días (bandas de confianza). El forecast usa un modelo autoregresivo entrenado sobre el historial del `media_pressure_score`.</div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Distribución regional de riesgo</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Ranking de hotspots por menciones regionales en titulares, atención por tema y actividad editorial reciente.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_region_risk_bars(filtered_region_snapshot), use_container_width=True)
        st.markdown('<div class="panel-note">Ranking regional por menciones y señales editoriales; útil para identificar hotspots mediáticos en el periodo seleccionado.</div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Monitor de actividad por tema</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Capas de actividad derivadas de palabras clave para military, energy, diplomacy y risk en el corpus RSS.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_theme_activity(filtered_theme_daily, filters["themes"]), use_container_width=True)
        st.markdown('<div class="panel-note">Actividad por tema (military, energy, diplomacy, risk) derivada de conteos de keywords en el corpus RSS. No es una medida perfecta de intención militar.</div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Cronología de cambios de régimen</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Clustering no supervisado sobre estados de señales históricas para detectar cambios y patrones en la postura del conflicto.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_regime_timeline(filtered_regime_frame), use_container_width=True)
        st.markdown('<div class="panel-note">Línea temporal de regímenes detectados por clustering no supervisado. Cada régimen resume el perfil dominante de señales (views, revisions, rss, score).</div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)


def render_model_tab(snapshot):
    st.markdown('<div class="section-grid">', unsafe_allow_html=True)
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Desempeño de modelos</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Comparación de desempeño entre clasificadores candidatos usando el dataset integrado y holdout temporal.</div>',
            unsafe_allow_html=True,
        )
        if snapshot.model_metrics.empty:
            st.warning("No model metrics available.")
        else:
            st.plotly_chart(plot_model_comparison(snapshot.model_metrics), use_container_width=True)
            st.markdown('<div class="panel-note">Comparación de desempeño entre modelos candidatos usando la métrica seleccionada (f1 en este proyecto). Revisa `models/model_metrics.json` para detalles.</div>', unsafe_allow_html=True)
            display_df = snapshot.model_metrics.copy()
            for column in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                if column in display_df.columns:
                    display_df[column] = display_df[column].map(
                        lambda value: f"{value:.3f}" if pd.notna(value) else "N/D"
                    )
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Importancia de características</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Ranking de contribuciones (modelos tipo árbol) desde el clasificador de escalada en producción.</div>',
            unsafe_allow_html=True,
        )
        if snapshot.feature_importance_frame.empty:
            st.warning("No feature importance available.")
        else:
            st.plotly_chart(plot_feature_importance(snapshot.feature_importance_frame), use_container_width=True)
            st.markdown('<div class="panel-note">Importancia de características derivada del mejor modelo; útil para entender qué señales impulsan la predicción.</div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-grid">', unsafe_allow_html=True)
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
        if snapshot.shap_frame.empty:
            st.markdown('<div class="panel-title">Panel de impulsores del modelo</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-note">Última predicción interpretada usando la importancia de características del modelo en producción.</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(plot_feature_importance(snapshot.feature_importance_frame), use_container_width=True)
            st.markdown('<div class="panel-note">Importancia de características derivada del mejor modelo; útil para entender qué señales impulsan la predicción.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel-title">Panel SHAP</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-note">Última predicción explicada mediante contribuciones SHAP para evitar que la plataforma sea una caja negra.</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(plot_shap_contributions(snapshot.shap_frame), use_container_width=True)
            st.markdown('<div class="panel-note">Contribuciones SHAP por fecha/instancia. Interpreta con cuidado: SHAP aproxima la importancia local del modelo, no causa.</div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Impulsores de la última predicción</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Lectura operativa de las variables que más aumentaron o disminuyeron la estimación actual de escalada.</div>',
            unsafe_allow_html=True,
        )
        if snapshot.top_drivers:
            st.dataframe(pd.DataFrame(snapshot.top_drivers), use_container_width=True, hide_index=True)
        else:
            st.dataframe(pd.DataFrame(columns=["feature", "value", "effect"]), use_container_width=True, hide_index=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_event_tab(snapshot, filters: dict):
    filtered_events = apply_date_window(snapshot.recent_events, filters["date_window"])
    if filters["sources"] and "source" in filtered_events.columns:
        filtered_events = filtered_events[filtered_events["source"].isin(filters["sources"])]

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Flujo de eventos recientes</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Últimos titulares relevantes al conflicto extraídos del layer RSS disponible. Este es el flujo textual real hoy disponible.</div>',
            unsafe_allow_html=True,
        )
        if filtered_events.empty:
            st.warning("No event stream available.")
        else:
            display_df = filtered_events[["date", "source", "title_clean", "link"]].copy().head(20)
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df.columns = ["date", "source", "headline", "link"]
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=520)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Matriz de atención por tema</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Temas monitorizados en Wikipedia y su huella reciente en atención pública y actividad editorial.</div>',
            unsafe_allow_html=True,
        )
        if snapshot.topic_activity.empty:
            st.warning("No hay actividad por tema disponible.")
        else:
            topic_slice = snapshot.topic_activity.copy()
            topic_slice = (
                topic_slice.groupby("topic_label", as_index=False)[["views", "revision_count"]]
                .sum()
                .sort_values("views", ascending=False)
            )
            st.dataframe(topic_slice, use_container_width=True, hide_index=True, height=520)
        st.markdown("</div></div>", unsafe_allow_html=True)


def render_method_tab(status_payload: dict | None):
    st.markdown('<div class="surface-panel"><div class="surface-body">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Límites metodológicos y realidad de datos</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-note">Todo lo mostrado está acotado a lo que el proyecto contiene actualmente. No se introdujeron eventos sintéticos ni datos económicos inventados.</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Esta plataforma predice estados de alta presión mediática y atención pública, no bajas verificadas de víctimas ni órdenes operativas confirmadas."
    )
    st.write(
        "El mapa regional se alimenta de menciones regionales en el corpus RSS y de temas monitorizados en Wikipedia. Es una capa de atención geográfica observada, no una base de eventos geoespaciales sintética."
    )
    st.write(
        "Las vistas sobre energía y petróleo están limitadas a señales textuales en el RSS, ya que el proyecto no contiene series temporales de mercado (Brent, WTI, índices de envío o datos financieros)."
    )
    st.write(
        "El forecast se construye solo a partir del historial del `media_pressure_score` usando patrones autoregresivos. Es un forecast a corto plazo, no una simulación geopolítica completa."
    )
    if status_payload:
        st.write(f"Última ejecución del pipeline registrada en el proyecto: `{status_payload.get('finished_at', 'N/D')}`.")
    st.markdown("</div></div>", unsafe_allow_html=True)


inject_styles()

try:
    dataset_df = load_dataset()
except Exception as exc:
    st.error(str(exc))
    st.stop()

metrics_payload = load_metrics()
status_payload = load_pipeline_status()
rss_df = load_raw_csv(RSS_PATH, parse_dates=["date"])
wikimedia_raw_df = load_raw_csv(WIKI_RAW_PATH, parse_dates=["date"])
revisions_raw_df = load_raw_csv(REVISION_RAW_PATH, parse_dates=["date"])
snapshot = build_snapshot(dataset_df, rss_df, wikimedia_raw_df, revisions_raw_df, metrics_payload)
filters = render_dashboard_filters(dataset_df, rss_df, snapshot)

render_sidebar(status_payload, snapshot)

st.markdown('<div class="war-shell">', unsafe_allow_html=True)
render_command_ribbon(snapshot, status_payload)
render_hero(snapshot, filters)
render_exec_metrics(snapshot)

tab_overview, tab_models, tab_events, tab_method = st.tabs(
    [
        "Resumen Operativo",
        "Modelo y Explicabilidad",
        "Flujo de Eventos y Fuentes",
        "Metodología",
    ]
)

with tab_overview:
    render_overview_tab(dataset_df, snapshot, filters)

with tab_models:
    render_model_tab(snapshot)

with tab_events:
    render_event_tab(snapshot, filters)

with tab_method:
    render_method_tab(status_payload)

st.markdown("</div>", unsafe_allow_html=True)
