from __future__ import annotations

import json
import sys
from datetime import datetime
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
            border-right: 1px solid rgba(112, 211, 255, 0.12);
            backdrop-filter: blur(12px);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        .mission-title {
            color: var(--cyan);
            font-family: "Rajdhani", sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .mission-subtitle {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
            margin-bottom: 12px;
        }

        h1, h2, h3 {
            font-family: "Rajdhani", sans-serif;
            color: var(--ink);
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(16, 40, 51, 0.75), rgba(9, 23, 31, 0.85));
            border: 1px solid rgba(112, 211, 255, 0.18);
            border-radius: 20px;
            padding: 16px 16px 14px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(8px);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-4px);
            border-color: rgba(112, 211, 255, 0.4);
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
            border-radius: 24px;
            background:
                linear-gradient(180deg, rgba(16, 40, 51, 0.7), rgba(8, 21, 29, 0.8));
            border: 1px solid rgba(112, 211, 255, 0.15);
            box-shadow: 0 30px 70px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(14px);
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

        .method-stack {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .method-grid-4 {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            align-items: stretch;
        }

        .method-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            align-items: stretch;
        }

        .method-card {
            min-height: 170px;
            padding: 1.35rem;
        }

        .method-small-card {
            min-height: 145px;
            padding: 1.25rem;
        }

        .method-source {
            min-height: 132px;
            padding: 1.25rem;
        }

        .method-step {
            color: var(--cyan);
            font-family: "Space Mono", monospace;
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
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

            .method-grid-4, .method-grid-3 {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 760px) {
            .method-grid-4, .method-grid-3 {
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
    theme_options = ["Militar", "Energía", "Diplomacia", "Riesgo"]

    for key, options in [
        ("mission_sources", source_options),
        ("mission_regions", region_options),
        ("mission_themes", theme_options),
    ]:
        current = st.session_state.get(key)
        if current is None:
            st.session_state[key] = options.copy()
        else:
            st.session_state[key] = [value for value in current if value in options]

    with st.sidebar:
        st.markdown(
            """
            <div class="mission-title">Misión y filtros</div>
            <div class="mission-subtitle">Ajusta la ventana temporal y limita fuentes, regiones o categorías del análisis.</div>
            """,
            unsafe_allow_html=True,
        )

        selected_range = st.date_input(
            "Rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="mission_date_range",
        )

        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        elif isinstance(selected_range, list) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date = selected_range
            end_date = selected_range

        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
            start_date, end_date = end_date, start_date

        st.caption(f"Rango seleccionado: {pd.to_datetime(start_date).date()} → {pd.to_datetime(end_date).date()}")

        filter_scope = st.radio(
            "Alcance del análisis",
            ["Todo disponible", "Personalizado"],
            horizontal=False,
            key="mission_filter_scope",
        )

        if filter_scope == "Personalizado":
            col_a, col_b = st.columns([1, 1])
            if col_a.button("Seleccionar todo", key="mission_select_all"):
                st.session_state["mission_sources"] = source_options.copy()
                st.session_state["mission_regions"] = region_options.copy()
                st.session_state["mission_themes"] = theme_options.copy()
            if col_b.button("Limpiar", key="mission_clear"):
                st.session_state["mission_sources"] = []
                st.session_state["mission_regions"] = []
                st.session_state["mission_themes"] = []

            selected_sources = (
                st.multiselect("Fuentes", source_options, key="mission_sources") if source_options else []
            )
            selected_regions = (
                st.multiselect("Regiones", region_options, key="mission_regions") if region_options else []
            )
            selected_themes = st.multiselect("Categorías", theme_options, key="mission_themes")
        else:
            selected_sources = source_options.copy()
            selected_regions = region_options.copy()
            selected_themes = theme_options.copy()

        st.caption(
            f"Activos: {len(selected_sources)}/{len(source_options)} fuentes · "
            f"{len(selected_regions)}/{len(region_options)} regiones · "
            f"{len(selected_themes)}/{len(theme_options)} categorías"
        )

        if not source_options:
            st.warning("No hay fuentes de noticias disponibles en data/raw/rss/rss_latest.csv")
        if not region_options:
            st.info("No hay regiones detectadas aún en el snapshot; revisa la ingesta de datos.")

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
        "custom_filtering": filter_scope == "Personalizado",
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
                    "Menciones RSS (7d): %{customdata[0]:.0f}<br>"
                    "Vistas Wiki (7d): %{customdata[1]:.0f}<br>"
                    "Revisiones Wiki (7d): %{customdata[2]:.0f}<br>"
                    "Último titular: %{customdata[3]}<extra></extra>"
                ),
                mode="markers+text",
                textposition="top center",
                marker=dict(
                    size=active["marker_size"],
                    color=active["risk_intensity"],
                    colorscale=[
                        [0.0, "#00c3ff"],
                        [0.4, "#00ffcc"],
                        [0.7, "#ffb143"],
                        [1.0, "#ff4b2b"],
                    ],
                    line=dict(width=1.2, color="rgba(255,255,255,0.6)"),
                    opacity=0.92,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text="Riesgo",
                            font=dict(color=INK, family="Rajdhani")
                        ),
                        thickness=12,
                        len=0.62,
                        x=0.98,
                        bgcolor="rgba(6,19,26,0.45)",
                        tickfont=dict(color=INK, family="Space Mono"),
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
        lataxis=dict(showgrid=True, gridcolor="rgba(85,214,255,0.12)", dtick=4, range=[18, 38]),
        lonaxis=dict(showgrid=True, gridcolor="rgba(85,214,255,0.12)", dtick=5, range=[32, 58]),
        center=dict(lat=29, lon=45),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
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
        "theme_military": "Militar",
        "theme_energy": "Energía",
        "theme_diplomacy": "Diplomacia",
        "theme_risk": "Riesgo",
    }
    allowed = set(labels.values() if selected_themes is None else selected_themes)
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
        yaxis_title="Menciones (Palabras clave)",
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
            name="Puntaje de presión",
            line=dict(color=CYAN, width=2.8),
        )
    )

    threshold = float(dataset_df["target_threshold_q75"].iloc[0])
    fig.add_hline(
        y=threshold,
        line_color=AMBER,
        line_dash="dash",
        annotation_text="Umbral de escalada",
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
                    name="Anomalías térmicas",
                    marker=dict(color=RED, size=9, line=dict(color="#fff", width=1)),
                )
            )

    forecast = snapshot.forecast_frame
    if not forecast.empty:

        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["forecast_score"],
                mode="lines+markers",
                name="Puntaje pronosticado",
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
        yaxis_title="Puntaje de Presión",
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
    fig.update_traces(texttemplate="%{text:.0f} menciones", textposition="outside")
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        xaxis_title="Intensidad de Riesgo",
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
        yaxis_title="Fase del conflicto (Régimen)",
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
        yaxis_title="Métrica de precisión",
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
        xaxis_title="Importancia del indicador (Feature importance)",
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
        xaxis_title="Contribución del Shapley Value (Riesgo)",
        yaxis_title="",
        font=dict(color=INK),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.18)")
    fig.update_yaxes(showgrid=False)
    return fig


def plot_wikipedia_attention_timeline(topic_activity: pd.DataFrame) -> go.Figure:
    daily = (
        topic_activity.groupby(["date", "topic_label"], as_index=False)[["views", "revision_count"]]
        .sum()
        .sort_values("date")
    )
    fig = px.area(
        daily,
        x="date",
        y="views",
        color="topic_label",
        color_discrete_sequence=[CYAN, GREEN, AMBER, RED, "#a1f0ff"],
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        xaxis_title="",
        yaxis_title="Vistas diarias",
        font=dict(color=INK),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def plot_wikipedia_topic_mix(topic_activity: pd.DataFrame) -> go.Figure:
    topic_mix = (
        topic_activity.groupby("topic_label", as_index=False)[["views", "revision_count"]]
        .sum()
        .sort_values("views", ascending=True)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=topic_mix["views"],
            y=topic_mix["topic_label"],
            orientation="h",
            name="Vistas",
            marker_color=CYAN,
            text=topic_mix["views"],
            texttemplate="%{text:,.0f}",
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=topic_mix["revision_count"],
            y=topic_mix["topic_label"],
            mode="markers",
            name="Revisiones",
            marker=dict(color=AMBER, size=12, line=dict(color="#fff", width=1)),
            xaxis="x2",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        xaxis=dict(title="Vistas acumuladas", gridcolor="rgba(255,255,255,0.08)"),
        xaxis2=dict(title="Revisiones", overlaying="x", side="top", showgrid=False),
        yaxis_title="",
        font=dict(color=INK),
        legend=dict(orientation="h", y=1.14, x=0),
    )
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



def render_hero(snapshot, filters: dict, page: str = "Home"):
    execu = snapshot.executive
    hero_titles = {
        "Home": "Sala de monitoreo y escalada",
        "Map": "Teatro de Conflicto",
        "Models": "Inteligencia Predictiva y ML",
        "Events": "Monitor Táctico de Fuentes",
        "Method": "Metodología y Sesgos"
    }
    hero_copies = {
        "Home": "Vista operativa escalonada. Navega por las subsecciones abajo para ver cada métrica.",
        "Map": "Análisis geográfico y mapas de atención regional derivados de reportes RSS.",
        "Models": "Auditoría de inteligencia artificial. Navega las subsecciones para leer cada reporte.",
        "Events": "Flujo textual de fuentes OSINT. Usa las pestañas para cambiar de informe.",
        "Method": "Especificaciones del diseño, arquitectura y validaciones del proyecto."
    }
    htitle = hero_titles.get(page, "Monitor Geopolítico")
    hcopy = hero_copies.get(page, "Análisis de inteligencia OSINT.")

    with st.container():
        st.caption("Plataforma de inteligencia · Irán / Israel / EE. UU. / energía / escalada")
        st.title(htitle)
        st.write(hcopy)

        if page == "Home":
            col1, col2, col3 = st.columns(3)
            col1.metric("Probabilidad de escalada", f"{execu.get('escalation_probability', 0):.1%}")
            col2.metric("Pico pronosticado", f"{execu.get('forecast_peak', 0):.1f}")
            col3.metric("Fecha pico pronosticada", str(execu.get("forecast_peak_date", "N/D")))

            col4, col5, col6 = st.columns(3)
            col4.metric("Señal militar", str(execu.get("military_signal", "N/D")))
            col5.metric("Señal energética", str(execu.get("energy_signal", "N/D")))
            col6.metric("Conteo temas de riesgo", str(execu.get("risk_signal", "N/D")))


def render_map_page(snapshot, filters: dict):
    selected_regions = filters["regions"]
    if selected_regions:
        filtered_region_snapshot = snapshot.region_snapshot[
            snapshot.region_snapshot["region"].isin(selected_regions)
        ]
    else:
        filtered_region_snapshot = snapshot.region_snapshot.copy()
    
    st.markdown(
        """
        <div class="surface-panel" style="margin-top: 1.5rem; margin-bottom: 0.5rem; padding: 2rem;">
            <div class="panel-title" style="font-size: 1.6rem; color: #55d6ff;">Mapa Operacional del Conflicto</div>
            <div class="panel-note" style="margin-top: 1rem; font-size: 1.05rem;">
                <strong>Cómo leer e interpretar este mapa</strong><br><br>
                Este mapa está enfocado únicamente en el teatro Levante-Golfo Pérsico-Irán que analiza el proyecto: Israel/Gaza/Líbano/Siria/Iraq/Kuwait/Hormuz/Irán.<br><br>
                <strong>El color indica peligro potencial:</strong> Si una región se pinta de color ámbar o rojo intenso, significa que un alto porcentaje de las noticias actuales habla de incidentes severos o ataques armados ahí.<br>
                <strong>El tamaño de la burbuja:</strong> Indica literalmente qué tanta fama o menciones está atrayendo ese sitio geográfico (aunque sea por motivos no violentos). 
            </div>
        </div>
        """, unsafe_allow_html=True
    )
    st.plotly_chart(
        plot_conflict_map(filtered_region_snapshot),
        use_container_width=True,
        config={"scrollZoom": False, "displayModeBar": True},
    )


def render_exec_metrics(snapshot):
    execu = snapshot.executive
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Nivel de amenaza (Color)", execu["threat_level"])
    col2.metric("Nivel de riesgo global", f"{execu['global_risk_proxy']:.1f}")
    col3.metric("Tendencia Semanal", f"{execu['weekly_trend']:+.1f}")
    col4.metric("Tendencia Mensual", f"{execu['monthly_trend']:+.1f}")
    col5.metric("Confianza de Predicción", f"{execu['model_confidence']:.1%}")
    col6.metric("Puntaje de anomalías", f"{execu['anomaly_score']:.3f}")


def render_overview_tab(dataset_df: pd.DataFrame, snapshot, filters: dict):
    filtered_dataset = apply_date_window(dataset_df, filters["date_window"])
    filtered_theme_daily = apply_date_window(snapshot.theme_daily, filters["date_window"])
    filtered_regime_frame = apply_date_window(snapshot.regime_frame, filters["date_window"])
    filtered_region_snapshot = snapshot.region_snapshot
    
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
                <div style="color:#87a9b5; font-family:'Space Mono', monospace; font-size:0.72rem;">{alert['timestamp']}</div>
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
        <div class="surface-panel" style="margin-bottom: 2rem; padding: 1.5rem;">
            <div class="panel-title" style="font-size: 1.4rem;">Ticker de Alertas Inmediatas</div>
            <div class="panel-note" style="margin-bottom: 1rem; font-size: 1.05rem;">
                <strong>Resumen automático:</strong> El cerebro del sistema escribe notas automatizadas aquí si nota algo raro HOY. Si no hay avisos críticos, las alertas serán de rutina (en verde o azul).
            </div>
            {radar_html}
            <div style="height:14px"></div>
            <div class="alert-stack">{alerts_html}</div>
        </div>
        """, unsafe_allow_html=True
    )

    t1, t2, t3, t4 = st.tabs([
        "Pronóstico de Tensión Mediática", 
        "Top Riesgo Regional", 
        "Seguimiento por Temática", 
        "Fases Históricas"
    ])

    with t1:
        st.markdown(
            """
            <div class="surface-panel" style="padding: 1.5rem; margin-bottom: 1rem;">
                <div class="panel-title" style="font-size: 1.4rem;">Evolución de la Presión Mediática y Pronóstico a 10 días</div>
                <div class="panel-note" style="font-size: 1.05rem;">
                    <strong>Cómo interpretar este gráfico</strong><br>
                    Mide el "calor" de las noticias y la población. La línea celeste principal marca la cantidad de presión detectada este último tiempo.<br>
                    • Los <strong>Puntos Rojos</strong> son días donde ocurrieron "anomalías" (atentados, incidentes masivos que dispararon los medios repentinamente).<br>
                    • La <strong>Línea Roja final (a la derecha)</strong> es la predicción del software: te está indicando si en los próximos 10 días se espera que el mundo estalle con noticias de este tema (escalada) o si retornará a la calma. 
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.plotly_chart(plot_pressure_and_forecast(filtered_dataset, snapshot), use_container_width=True)

    with t2:
        st.markdown(
            """
            <div class="surface-panel" style="padding: 1.5rem; margin-bottom: 1rem;">
                <div class="panel-title" style="font-size: 1.4rem;">Ranking de Peligro Regional</div>
                <div class="panel-note" style="font-size: 1.05rem;">
                    <strong>Cómo interpretar este gráfico</strong><br>
                    Es una lista de mayor a menor. Mientras más larga es la barra hacia la derecha y más oscuro es su color (naranja/rojo), 
                    más vinculada está esa zona geográfica a menciones violentas o de riesgo en los principales noticiarios de hoy.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.plotly_chart(plot_region_risk_bars(filtered_region_snapshot), use_container_width=True)

    with t3:
        st.markdown(
            """
            <div class="surface-panel" style="padding: 1.5rem; margin-bottom: 1rem;">
                <div class="panel-title" style="font-size: 1.4rem;">Menciones Específicas por Temática</div>
                <div class="panel-note" style="font-size: 1.05rem;">
                    <strong>Cómo interpretar este gráfico</strong><br>
                    Analiza exclusivamente si el mundo entero está charlando actualmente sobre estrategias "Militares" (bombardeos, fuerzas), o de "Energía" (petróleo, gas, barriles). 
                    Cuando la línea amarilla (energía) se dispara de golpe, significa que las cadenas de texto mundiales advierten un fallo en esa cadena económica.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        theme_cols = [c for c in ["theme_military", "theme_energy", "theme_diplomacy", "theme_risk"] if c in filtered_theme_daily.columns]
        if theme_cols:
            st.line_chart(filtered_theme_daily.set_index("date")[theme_cols], use_container_width=True)
        else:
            st.info("No hay datos temáticos disponibles.")

    with t4:
        st.markdown(
            """
            <div class="surface-panel" style="padding: 1.5rem; margin-bottom: 1rem;">
                <div class="panel-title" style="font-size: 1.4rem;">Cronología Continua de Fases Operativas</div>
                <div class="panel-note" style="font-size: 1.05rem;">
                    <strong>Cómo interpretar este gráfico</strong><br>
                    Pinta la historia como un muro de colores. Te ayuda a entender si la última semana entera ha sido considerada por la máquina 
                    como un "régimen pacífico prolongado", o bien como un "régimen crítico volátil". Las caídas y subidas repentinas marcan un antes y un después en el conflicto mundial.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.plotly_chart(plot_regime_timeline(filtered_regime_frame), use_container_width=True)


def render_model_tab(snapshot):
    t1, t2, t3, t4 = st.tabs([
        "Comparativa del Algoritmo",
        "Indicadores de Mayor Peso",
        "Transparencia Clínica (SHAP)",
        "Registro Aislado"
    ])
    
    with t1:
        st.markdown(
            """
            <div class="surface-panel" style="padding: 1.5rem; margin-bottom: 1rem;">
                <div class="panel-title" style="font-size: 1.4rem;">Grilla de Métricas en Modelos Evaluados</div>
                <div class="panel-note" style="font-size: 1.05rem;">
                    <strong>Por qué es importante esto</strong><br>
                    Para que nuestra predicción no esté errada, el sistema compitió resolviendo exámenes del pasado. 
                    Aquí están las calificaciones que sacaron los distintos algoritmos: una barra altísima indica que casi no cometieron errores. La precisión se mide entre 0 y 1.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        if snapshot.model_metrics.empty:
            st.warning("No hay métricas de modelos disponibles.")
        else:
            st.plotly_chart(plot_model_comparison(snapshot.model_metrics), use_container_width=True)
            display_df = snapshot.model_metrics.copy()
            for column in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                if column in display_df.columns:
                    display_df[column] = display_df[column].map(
                        lambda value: f"{value:.3f}" if pd.notna(value) else "N/D"
                    )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with t2:
        st.markdown(
            """
            <div class="surface-panel" style="padding: 1.5rem; margin-bottom: 1rem;">
                <div class="panel-title" style="font-size: 1.4rem;">Variables Primordiales: ¿A qué le presta más atención el sistema?</div>
                <div class="panel-note" style="font-size: 1.05rem;">
                    <strong>Qué significa este ranking</strong><br>
                    Determina qué información ha sido históricamente la más vital. Si la variable más grande (la de arriba) recae por ejemplo en "wikipedia visits", 
                    significa que las visitas humanas tienen una correlación altísima de predecir o influir el resultado; y nos dice que el algoritmo depende de esos sensores.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        if snapshot.feature_importance_frame.empty:
            st.warning("No hay importancia de características disponible.")
        else:
            st.plotly_chart(plot_feature_importance(snapshot.feature_importance_frame), use_container_width=True)

    with t3:
        st.markdown(
            """
            <div class="surface-panel" style="padding: 1.5rem; margin-bottom: 1rem;">
                <div class="panel-title" style="font-size: 1.4rem;">Explicabilidad SHAP: Decisiones tomadas HOY</div>
                <div class="panel-note" style="font-size: 1.05rem;">
                    <strong>Transparencia total, para evitar "cajas negras":</strong><br>
                    Este diagrama mapea cómo se construyó el número de riesgo exacto de *hoy*. Las franjas <strong>Rojas</strong> son aquellos culpables que añadieron 
                    porcentajes de alarma a este día de trabajo táctico, empujando la amenaza global. Las barras <strong>Verdes</strong> son eventos o caídas estadísticas que calmaron el ambiente.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        if snapshot.shap_frame.empty:
            st.warning("Variables explicatorias no generadas en este registro.")
        else:
            st.plotly_chart(plot_shap_contributions(snapshot.shap_frame), use_container_width=True)

    with t4:
        st.markdown(
            """
            <div class="surface-panel" style="padding: 1.5rem; margin-bottom: 1rem;">
                <div class="panel-title" style="font-size: 1.4rem;">Reporte Analítico Cruzado (Data del SHAP)</div>
                <div class="panel-note" style="font-size: 1.05rem;">
                    <strong>Versión Textual Directa:</strong> Exclusivamente para analistas que desean descargar los datos o copiar y pegarlos. Son las sumas concretas de impacto.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        if snapshot.top_drivers:
            st.dataframe(pd.DataFrame(snapshot.top_drivers), use_container_width=True, hide_index=True)
        else:
            st.dataframe(pd.DataFrame(columns=["feature", "value", "effect"]), use_container_width=True, hide_index=True)


def render_event_tab(snapshot, filters: dict):
    filtered_events = apply_date_window(snapshot.recent_events, filters["date_window"])
    if "source" in filtered_events.columns:
        filtered_events = filtered_events[filtered_events["source"].isin(filters["sources"])]

    t1, t2 = st.tabs([
        "Cintas de Prensa (RSS Feeds Crudos)",
        "Monitor Mundial de Wikipedia (Búsquedas Sociales)"
    ])

    with t1:
        st.markdown(
            """
            <div class="surface-panel" style="padding: 1.5rem; margin-bottom: 1rem;">
                <div class="panel-title" style="font-size: 1.4rem;">Transmisión Informativa Global</div>
                <div class="panel-note" style="font-size: 1.05rem;">
                    <strong>De dónde ha ingerido los sucesos el bot</strong><br>
                    Esta pestaña documenta la matriz real de lectura del bot. Contiene artículos textuales sin modificar extraídos de canales libres como la BBC o Al-Jazeera.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        if filtered_events.empty:
            st.warning("No hay flujo de eventos disponible para los filtros seleccionados.")
        else:
            display_df = filtered_events[["date", "source", "title_clean", "link"]].copy().head(20)
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df.columns = ["date", "source", "headline", "link"]
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=520)

    with t2:
        st.markdown(
            """
            <div class="surface-panel" style="padding: 1.5rem; margin-bottom: 1rem;">
                <div class="panel-title" style="font-size: 1.4rem;">Radar de Temas Sociales Actuales</div>
                <div class="panel-note" style="font-size: 1.05rem;">
                    <strong>Por qué medimos el interés social</strong><br>
                    El número de usuarios leyendo artículos estratégicos de guerra (por ejemplo "Armamento Nuclear de Israel") sube de golpe cuando el ruido colectivo de fondo siente peligro inminente, lo cual suele ser la antesala de incidentes geográficos severos.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        if snapshot.topic_activity.empty:
            st.warning("No hay actividad por tema disponible.")
        else:
            topic_activity = apply_date_window(snapshot.topic_activity, filters["date_window"])
            if topic_activity.empty:
                st.warning("No hay actividad de Wikipedia para el rango de fechas seleccionado.")
                return

            total_views = float(topic_activity["views"].sum())
            total_revisions = float(topic_activity["revision_count"].sum())
            active_topics = int(topic_activity["topic_label"].nunique())
            latest_date = pd.to_datetime(topic_activity["date"]).max().strftime("%Y-%m-%d")
            top_topic = (
                topic_activity.groupby("topic_label")["views"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Vistas acumuladas", f"{total_views:,.0f}")
            col2.metric("Revisiones editoriales", f"{total_revisions:,.0f}")
            col3.metric("Temas activos", f"{active_topics}")
            col4.metric("Tema dominante", top_topic)

            left, right = st.columns([1.25, 0.95])
            with left:
                st.markdown(
                    """
                    <div class="surface-panel" style="padding: 1.2rem; margin: 1rem 0;">
                        <div class="panel-title" style="font-size: 1.15rem;">Pulso diario de atención pública</div>
                        <div class="panel-note">Evolución de lecturas por tema dentro de la ventana filtrada.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(plot_wikipedia_attention_timeline(topic_activity), use_container_width=True)

            with right:
                st.markdown(
                    """
                    <div class="surface-panel" style="padding: 1.2rem; margin: 1rem 0;">
                        <div class="panel-title" style="font-size: 1.15rem;">Composición tema-edición</div>
                        <div class="panel-note">Contrasta consumo público con actividad editorial.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(plot_wikipedia_topic_mix(topic_activity), use_container_width=True)

            topic_slice = (
                topic_activity.groupby(["topic_label", "region"], as_index=False)
                .agg(
                    views=("views", "sum"),
                    revision_count=("revision_count", "sum"),
                    active_days=("date", "nunique"),
                    latest_observation=("date", "max"),
                )
                .sort_values("views", ascending=False)
            )
            topic_slice["share_views"] = topic_slice["views"] / max(total_views, 1) * 100
            topic_slice["views_per_revision"] = topic_slice.apply(
                lambda row: row["views"] / row["revision_count"] if row["revision_count"] else row["views"],
                axis=1,
            )
            topic_slice["latest_observation"] = pd.to_datetime(topic_slice["latest_observation"]).dt.strftime("%Y-%m-%d")
            topic_slice = topic_slice[
                [
                    "topic_label",
                    "region",
                    "views",
                    "share_views",
                    "revision_count",
                    "views_per_revision",
                    "active_days",
                    "latest_observation",
                ]
            ].rename(
                columns={
                    "topic_label": "tema",
                    "region": "region",
                    "views": "vistas",
                    "share_views": "% atencion",
                    "revision_count": "revisiones",
                    "views_per_revision": "vistas por revision",
                    "active_days": "dias activos",
                    "latest_observation": "ultima observacion",
                }
            )
            st.markdown(
                f"""
                <div class="surface-panel" style="padding: 1.2rem; margin: 1rem 0;">
                    <div class="panel-title" style="font-size: 1.15rem;">Matriz enriquecida de temas Wikipedia</div>
                    <div class="panel-note">Última observación disponible: {latest_date}. El porcentaje indica qué parte de la atención del periodo concentra cada tema.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                topic_slice.style.format(
                    {
                        "vistas": "{:,.0f}",
                        "% atencion": "{:.1f}%",
                        "revisiones": "{:,.0f}",
                        "vistas por revision": "{:,.1f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=360,
            )


def render_method_tab(status_payload: dict | None):
    last_run = status_payload.get('finished_at', 'N/D') if status_payload else 'N/D'
    dataset_summary = status_payload.get("dataset_summary", {}) if status_payload else {}
    date_min = dataset_summary.get("date_min", "N/D")
    date_max = dataset_summary.get("date_max", "N/D")
    rows = dataset_summary.get("rows", "N/D")
    st.markdown(
        f"""
        <div class="surface-panel" style="padding: 2rem; margin-bottom: 1rem;">
            <div class="hero-badge">Diseño metodológico · OSINT · clasificación diaria</div>
            <div class="hero-title" style="font-size: 2.3rem; margin-top: 0.6rem;">Cómo funciona el sistema</div>
            <div class="hero-copy" style="max-width: 980px;">
                La plataforma integra señales abiertas para estimar si el próximo día puede entrar en un régimen de alta presión pública
                y mediática sobre el conflicto Irán-Israel-EE. UU. El resultado debe leerse como una alerta analítica, no como confirmación
                de eventos militares.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="surface-panel" style="padding: 1.6rem; margin-bottom: 1rem;">
            <div style="display:grid; grid-template-columns: 1.1fr 1.4fr; gap: 22px; align-items:start;">
                <div>
                    <div class="panel-title" style="font-size: 1.25rem;">Resumen del diseño</div>
                    <div class="panel-note" style="font-size: 1rem; line-height: 1.65;">
                        El proyecto trabaja con una serie diaria. Integra atención pública, actividad editorial y titulares RSS para estimar
                        días de alta presión mediática futura sobre el conflicto.
                    </div>
                </div>
                <div style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px;">
                    <div style="border-left: 2px solid #55d6ff; padding-left: 12px;">
                        <div class="method-step">Cobertura</div>
                        <div class="panel-title" style="font-size: 1.15rem;">{date_min} / {date_max}</div>
                    </div>
                    <div style="border-left: 2px solid #49dcb1; padding-left: 12px;">
                        <div class="method-step">Unidad</div>
                        <div class="panel-title" style="font-size: 1.15rem;">Día calendario</div>
                    </div>
                    <div style="border-left: 2px solid #ffb347; padding-left: 12px;">
                        <div class="method-step">Volumen</div>
                        <div class="panel-title" style="font-size: 1.15rem;">{rows} registros</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="surface-panel" style="padding: 1.6rem; margin-bottom: 1rem;">
            <div class="panel-title" style="font-size: 1.25rem; margin-bottom: 1rem;">Flujo metodológico</div>
            <div style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px;">
                <div>
                    <div class="method-step">01 · Fuentes</div>
                    <div class="panel-note" style="font-size: 1rem; line-height: 1.6;">
                        Wikimedia Pageviews, Wikipedia Revisions y RSS de Google News, BBC y Al Jazeera.
                    </div>
                </div>
                <div>
                    <div class="method-step">02 · Señales</div>
                    <div class="panel-note" style="font-size: 1rem; line-height: 1.6;">
                        Rezagos, promedios móviles, conteos de titulares, visitas, revisiones y calendario.
                    </div>
                </div>
                <div>
                    <div class="method-step">03 · Modelo</div>
                    <div class="panel-note" style="font-size: 1rem; line-height: 1.6;">
                        Clasificación de alta presión mediática del día siguiente con comparación de modelos supervisados.
                    </div>
                </div>
            </div>
        </div>

        <div class="surface-panel" style="padding: 1.6rem;">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 22px;">
                <div>
                    <div class="panel-title" style="font-size: 1.18rem;">Lectura correcta</div>
                    <div class="panel-note" style="font-size: 1rem; line-height: 1.65;">
                        Una subida del riesgo indica aumento de presión pública y mediática estimada. No confirma ataques, bajas ni decisiones
                        operativas. Debe contrastarse con fuentes primarias.
                    </div>
                </div>
                <div>
                    <div class="panel-title" style="font-size: 1.18rem;">Trazabilidad</div>
                    <div class="panel-note" style="font-size: 1rem; line-height: 1.65;">
                        Datos crudos, dataset diario procesado y métricas del modelo quedan almacenados en el repositorio. Última ejecución:
                        <strong>{last_run}</strong>.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

try:
    dataset_df = load_dataset()
except Exception as exc:
    st.error(f"Error cargando dataset: {exc}")
    st.stop()


# Componentes de datos
metrics_payload = load_metrics()
status_payload = load_pipeline_status()
rss_df = load_raw_csv(RSS_PATH, parse_dates=["date"])
wikimedia_raw_df = load_raw_csv(WIKI_RAW_PATH, parse_dates=["date"])
revisions_raw_df = load_raw_csv(REVISION_RAW_PATH, parse_dates=["date"])
snapshot = build_snapshot(dataset_df, rss_df, wikimedia_raw_df, revisions_raw_df, metrics_payload)

# SIDEBAR CONFIGURATION
with st.sidebar:
    menu_options = {
        "Centro de Mandos": "Home",
        "Teatro de Conflicto": "Map",
        "Inteligencia ML": "Models",
        "Monitor de Fuentes": "Events",
        "Metodología": "Method",
    }
    st.caption("Módulos del sistema")
    selection = st.radio("Módulos del Sistema", list(menu_options.keys()), label_visibility="collapsed")
    page = menu_options[selection]
    
    filters = render_dashboard_filters(dataset_df, rss_df, snapshot)
    
    st.caption(f"Actualizado en sesión: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# MAIN CONTENT AREA
st.markdown('<div class="war-shell">', unsafe_allow_html=True)
render_hero(snapshot, filters, page)

# Routing Logic
if page == "Home":
    render_exec_metrics(snapshot)
    render_overview_tab(dataset_df, snapshot, filters)
elif page == "Map":
    render_map_page(snapshot, filters)
elif page == "Models":
    render_model_tab(snapshot)
elif page == "Events":
    render_event_tab(snapshot, filters)
elif page == "Method":
    render_method_tab(status_payload)

st.markdown("</div>", unsafe_allow_html=True)
