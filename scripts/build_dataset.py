import os

import numpy as np
import pandas as pd

from project_paths import PROCESSED_DIR, RAW_DIR, ensure_dir


def _minmax(series):
    series = series.astype(float)
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_value) / (max_value - min_value)


def _load_csv(path, required=False):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"No existe {path}. Ejecuta primero el script de recoleccion correspondiente."
            )
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_gdelt(raw_dir):
    gdelt_path = os.path.join(raw_dir, "gdelt", "gdelt_daily.csv")
    gdelt_df = _load_csv(gdelt_path)
    if gdelt_df.empty:
        return pd.DataFrame(columns=["date", "volume", "avg_tone"])
    required_cols = {"date", "volume", "avg_tone"}
    missing = required_cols.difference(gdelt_df.columns)
    if missing:
        raise ValueError(f"GDELT no tiene las columnas requeridas: {sorted(missing)}")

    gdelt_df["date"] = pd.to_datetime(gdelt_df["date"])
    return gdelt_df.groupby("date", as_index=False).agg(
        volume=("volume", "sum"),
        avg_tone=("avg_tone", "mean"),
    )


def _load_wikimedia(raw_dir):
    wiki_path = os.path.join(raw_dir, "wikimedia", "wikimedia_daily.csv")
    wiki_df = _load_csv(wiki_path)
    if wiki_df.empty:
        return pd.DataFrame(columns=["date", "wiki_pageviews_total", "wiki_articles_tracked"])
    wiki_df["date"] = pd.to_datetime(wiki_df["date"])
    return wiki_df


def _load_wikipedia_revisions(raw_dir):
    revisions_path = os.path.join(raw_dir, "wikipedia_revisions", "wikipedia_revisions_daily.csv")
    revisions_df = _load_csv(revisions_path)
    if revisions_df.empty:
        return pd.DataFrame(
            columns=["date", "wiki_revision_count", "wiki_revision_users", "wiki_revision_articles"]
        )
    revisions_df["date"] = pd.to_datetime(revisions_df["date"])
    return revisions_df


def _load_rss(raw_dir):
    rss_path = os.path.join(raw_dir, "rss", "rss_daily_features.csv")
    rss_df = _load_csv(rss_path)
    if rss_df.empty:
        return pd.DataFrame(columns=["date", "rss_total"])
    rss_df["date"] = pd.to_datetime(rss_df["date"])
    if "rss_total" not in rss_df.columns:
        count_cols = [col for col in rss_df.columns if col != "date"]
        rss_df["rss_total"] = rss_df[count_cols].sum(axis=1)
    return rss_df


def _load_firms(raw_dir):
    firms_path = os.path.join(raw_dir, "firms", "firms_hotspots.csv")
    firms_df = _load_csv(firms_path)
    if firms_df.empty:
        return pd.DataFrame(columns=["date", "firms_hotspots_count", "firms_avg_brightness"])

    if "date" not in firms_df.columns and "acq_date" in firms_df.columns:
        firms_df = firms_df.rename(columns={"acq_date": "date"})
    if "date" not in firms_df.columns:
        raise ValueError("FIRMS no tiene columna date/acq_date.")

    brightness_col = "bright_ti4" if "bright_ti4" in firms_df.columns else "brightness"
    firms_df["date"] = pd.to_datetime(firms_df["date"])
    return firms_df.groupby("date", as_index=False).agg(
        firms_hotspots_count=("date", "count"),
        firms_avg_brightness=(brightness_col, "mean"),
    )


def build_dataset():
    """
    Integra fuentes reales en una tabla pais-region-dia.

    Target viable sin ACLED/GDELT: clasifica si el dia siguiente pertenece al cuartil alto
    de presion publica observada. Las features usan rezagos y ventanas historicas para
    evitar usar informacion del futuro.
    """
    print("Construyendo dataset integrado con fuentes reales...")
    raw_dir = str(RAW_DIR)

    gdelt_df = _load_gdelt(raw_dir)
    wiki_df = _load_wikimedia(raw_dir)
    revisions_df = _load_wikipedia_revisions(raw_dir)

    if not gdelt_df.empty:
        df = gdelt_df.copy()
        target_source = "GDELT_DOC_media_pressure_next_day_q75"
        source_frames = [wiki_df, revisions_df, _load_rss(raw_dir), _load_firms(raw_dir)]
    elif not wiki_df.empty:
        df = wiki_df.copy()
        target_source = "Wikimedia_public_attention_next_day_q75"
        source_frames = [gdelt_df, revisions_df, _load_rss(raw_dir), _load_firms(raw_dir)]
    else:
        raise FileNotFoundError(
            "No hay datos base reales. Ejecuta collect_gdelt.py o collect_wikimedia.py."
        )

    for source_df in source_frames:
        if not source_df.empty:
            df = pd.merge(df, source_df, on="date", how="left")
    df = df.loc[:, ~df.columns.duplicated()]

    df = df.sort_values("date").reset_index(drop=True)
    numeric_cols = [col for col in df.columns if col != "date"]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    for col in ["volume", "avg_tone", "wiki_pageviews_total", "wiki_revision_count"]:
        if col not in df.columns:
            df[col] = 0

    df["log_volume"] = np.log1p(df["volume"])
    df["negative_tone"] = (-df["avg_tone"]).clip(lower=0)
    df["log_wiki_pageviews"] = np.log1p(df["wiki_pageviews_total"])
    df["log_wiki_revisions"] = np.log1p(df["wiki_revision_count"])

    if target_source.startswith("GDELT"):
        df["pressure_score"] = 100 * (
            0.7 * _minmax(df["log_volume"]) + 0.3 * _minmax(df["negative_tone"])
        )
    else:
        df["pressure_score"] = 100 * (
            0.75 * _minmax(df["log_wiki_pageviews"])
            + 0.25 * _minmax(df["log_wiki_revisions"])
        )
    df["media_pressure_score"] = df["pressure_score"]
    df["escalation_score"] = df["pressure_score"]

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

    lag_sources = [
        "volume",
        "avg_tone",
        "negative_tone",
        "media_pressure_score",
        "wiki_pageviews_total",
        "wiki_revision_count",
        "rss_total",
        "firms_hotspots_count",
    ]
    for col in lag_sources:
        if col not in df.columns:
            df[col] = 0
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_7d_avg"] = df[col].rolling(window=7, min_periods=1).mean().shift(1)

    df["target_next_day_score"] = df["media_pressure_score"].shift(-1)
    threshold = df["target_next_day_score"].quantile(0.75)
    df["target_high_escalation_next_day"] = (
        df["target_next_day_score"] >= threshold
    ).astype(int)
    df["target_threshold_q75"] = threshold
    df["target_source"] = target_source

    df = df.dropna(subset=["target_next_day_score"]).fillna(0)

    output_dir = ensure_dir(PROCESSED_DIR)
    filepath = output_dir / "dataset_diario.csv"
    df.to_csv(filepath, index=False)

    print(f"Dataset guardado en {filepath}")
    print(f"Filas: {len(df)}, columnas: {len(df.columns)}")
    print(f"Target positivo: {df['target_high_escalation_next_day'].mean():.2%}")
    return df


if __name__ == "__main__":
    build_dataset()
