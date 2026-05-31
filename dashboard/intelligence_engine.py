from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

try:
    import shap
except ImportError:
    shap = None


REGION_META = {
    "Iran": {"lat": 32.0, "lon": 53.0, "keywords": ["iran", "tehran", "irgc", "bandar abbas", "persian"]},
    "Israel": {"lat": 31.2, "lon": 34.8, "keywords": ["israel", "tel aviv", "idf", "jerusalem"]},
    "Gaza": {"lat": 31.4, "lon": 34.3, "keywords": ["gaza", "hamas"]},
    "Lebanon": {"lat": 33.8, "lon": 35.8, "keywords": ["lebanon", "hezbollah", "beirut"]},
    "Syria": {"lat": 35.0, "lon": 38.5, "keywords": ["syria", "damascus"]},
    "Iraq": {"lat": 33.0, "lon": 44.0, "keywords": ["iraq", "baghdad"]},
    "Kuwait": {"lat": 29.3, "lon": 47.5, "keywords": ["kuwait"]},
    "Hormuz": {"lat": 26.6, "lon": 56.25, "keywords": ["hormuz", "strait of hormuz", "gulf"]},
    "Red Sea": {"lat": 20.5, "lon": 38.2, "keywords": ["red sea", "houthi", "shipping", "tanker"]},
}


THEME_KEYWORDS = {
    "military": [
        "strike",
        "strikes",
        "missile",
        "missiles",
        "drone",
        "airbase",
        "troops",
        "navy",
        "defense",
        "attack",
        "war",
        "arsenal",
        "air defenses",
        "irgc",
    ],
    "energy": [
        "oil",
        "crude",
        "energy",
        "hormuz",
        "shipping",
        "tanker",
        "gas",
        "refinery",
        "supply",
        "market",
    ],
    "diplomacy": [
        "ceasefire",
        "deal",
        "talks",
        "allies",
        "summit",
        "pause",
        "negotiation",
        "sanctions",
        "diplomatic",
    ],
    "risk": [
        "escalation",
        "threat",
        "war",
        "conflict",
        "crisis",
        "attack",
        "strike",
        "missile",
        "fragile",
    ],
}


WIKI_TOPIC_META = {
    "Iran-Israel_proxy_conflict": {"label": "Iran-Israel proxy conflict", "region": "Iran"},
    "Iran-Israel_relations": {"label": "Iran-Israel relations", "region": "Israel"},
    "Israel-Hamas_war": {"label": "Israel-Hamas war", "region": "Gaza"},
    "Red_Sea_crisis": {"label": "Red Sea crisis", "region": "Red Sea"},
    "Iran-Israel proxy conflict": {"label": "Iran-Israel proxy conflict", "region": "Iran"},
    "Iran-Israel relations": {"label": "Iran-Israel relations", "region": "Israel"},
    "Israel-Hamas war": {"label": "Israel-Hamas war", "region": "Gaza"},
    "Red Sea crisis": {"label": "Red Sea crisis", "region": "Red Sea"},
}


ALERT_LEVEL_META = {
    "critical": {"label": "Critical", "color": "#ff5c5c"},
    "moderate": {"label": "Moderate", "color": "#ffb347"},
    "info": {"label": "Informational", "color": "#49dcb1"},
}


AUTOREGRESSIVE_FEATURES = [
    "score_lag1",
    "score_lag3",
    "score_lag7",
    "score_lag14",
    "score_rolling_7",
    "score_rolling_14",
    "score_delta_1",
    "score_delta_7",
    "day_of_week",
    "month",
    "days_since_start",
]


def clean_html(raw_text: Any) -> str:
    text = unescape(str(raw_text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def latest_notna(series: pd.Series, default: float = 0.0) -> float:
    clean_series = series.dropna()
    if clean_series.empty:
        return default
    return float(clean_series.iloc[-1])


def minmax_series(series: pd.Series) -> pd.Series:
    numeric = series.astype(float)
    min_value = numeric.min()
    max_value = numeric.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - min_value) / (max_value - min_value)


def normalize_value(value: float, reference: pd.Series) -> float:
    if reference.empty:
        return 0.0
    min_value = float(reference.min())
    max_value = float(reference.max())
    if math.isclose(min_value, max_value):
        return 0.0
    return max(0.0, min(1.0, (float(value) - min_value) / (max_value - min_value)))


def keyword_hits(text: str, keywords: list[str]) -> int:
    lower_text = text.lower()
    return sum(1 for keyword in keywords if keyword in lower_text)


def classify_alert_level(probability: float, anomaly_flag: bool, military_intensity: float) -> str:
    if probability >= 0.72 or (anomaly_flag and military_intensity >= 0.6):
        return "critical"
    if probability >= 0.48 or anomaly_flag or military_intensity >= 0.35:
        return "moderate"
    return "info"


def build_alert_payload(level: str, title: str, description: str, metric: str, timestamp: str) -> dict[str, str]:
    meta = ALERT_LEVEL_META[level]
    return {
        "level": level,
        "label": meta["label"],
        "color": meta["color"],
        "title": title,
        "description": description,
        "metric": metric,
        "timestamp": timestamp,
    }


def prepare_rss_intelligence(rss_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rss_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    frame = rss_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["body_text"] = (frame["title"].fillna("") + " " + frame["description"].fillna("")).map(clean_html)
    frame["title_clean"] = frame["title"].fillna("").map(clean_html)

    for theme, keywords in THEME_KEYWORDS.items():
        frame[f"theme_{theme}"] = frame["body_text"].map(lambda text: keyword_hits(text, keywords))

    for region, meta in REGION_META.items():
        frame[f"region_{region}"] = frame["body_text"].map(lambda text: keyword_hits(text, meta["keywords"]))

    daily_theme = (
        frame.groupby("date", as_index=False)[[f"theme_{theme}" for theme in THEME_KEYWORDS]]
        .sum()
        .sort_values("date")
    )
    daily_region = (
        frame.groupby("date", as_index=False)[[f"region_{region}" for region in REGION_META]]
        .sum()
        .sort_values("date")
    )
    return daily_theme, daily_region, frame


def prepare_wiki_topic_activity(wikimedia_raw_df: pd.DataFrame, revisions_raw_df: pd.DataFrame) -> pd.DataFrame:
    if wikimedia_raw_df.empty and revisions_raw_df.empty:
        return pd.DataFrame()

    views_df = pd.DataFrame(columns=["date", "article", "views"])
    if not wikimedia_raw_df.empty:
        views_df = wikimedia_raw_df.copy()
        views_df["date"] = pd.to_datetime(views_df["date"], errors="coerce")

    revisions_daily = pd.DataFrame(columns=["date", "article", "revision_count"])
    if not revisions_raw_df.empty:
        revisions_frame = revisions_raw_df.copy()
        revisions_frame["date"] = pd.to_datetime(revisions_frame["date"], errors="coerce")
        revisions_daily = (
            revisions_frame.groupby(["date", "article"], as_index=False)
            .size()
            .rename(columns={"size": "revision_count"})
        )

    merged = views_df.merge(revisions_daily, on=["date", "article"], how="outer")
    merged["views"] = merged["views"].fillna(0)
    merged["revision_count"] = merged["revision_count"].fillna(0)
    merged["topic_label"] = merged["article"].map(lambda value: WIKI_TOPIC_META.get(value, {}).get("label", str(value)))
    merged["region"] = merged["article"].map(lambda value: WIKI_TOPIC_META.get(value, {}).get("region", "Regional"))
    return merged.sort_values(["date", "topic_label"]).reset_index(drop=True)


def build_autoregressive_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[["date", "media_pressure_score"]].copy().sort_values("date").reset_index(drop=True)
    frame["score_lag1"] = frame["media_pressure_score"].shift(1)
    frame["score_lag3"] = frame["media_pressure_score"].shift(3)
    frame["score_lag7"] = frame["media_pressure_score"].shift(7)
    frame["score_lag14"] = frame["media_pressure_score"].shift(14)
    frame["score_rolling_7"] = frame["media_pressure_score"].rolling(7, min_periods=1).mean().shift(1)
    frame["score_rolling_14"] = frame["media_pressure_score"].rolling(14, min_periods=1).mean().shift(1)
    frame["score_delta_1"] = frame["media_pressure_score"].diff(1).shift(1)
    frame["score_delta_7"] = frame["media_pressure_score"].diff(7).shift(1)
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["days_since_start"] = (frame["date"] - frame["date"].min()).dt.days
    return frame.dropna().reset_index(drop=True)


@dataclass
class IntelligenceSnapshot:
    executive: dict[str, Any]
    region_snapshot: pd.DataFrame
    recent_events: pd.DataFrame
    theme_daily: pd.DataFrame
    region_daily: pd.DataFrame
    topic_activity: pd.DataFrame
    alerts: list[dict[str, str]]
    anomaly_frame: pd.DataFrame
    forecast_frame: pd.DataFrame
    regime_frame: pd.DataFrame
    model_metrics: pd.DataFrame
    shap_frame: pd.DataFrame
    feature_importance_frame: pd.DataFrame
    top_drivers: list[dict[str, str]]


def compute_conflict_regimes(dataset_df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "wiki_pageviews_total",
        "wiki_revision_count",
        "rss_total",
        "media_pressure_score",
        "media_pressure_score_7d_avg",
    ]
    frame = dataset_df[["date"] + features].copy().fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(frame[features])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = kmeans.fit_predict(scaled)
    centroid_frame = pd.DataFrame(kmeans.cluster_centers_, columns=features)

    regime_names = {}
    for index, row in centroid_frame.iterrows():
        dominant = row.sort_values(ascending=False).index.tolist()[:2]
        readable = {
            "wiki_pageviews_total": "Public attention",
            "wiki_revision_count": "Editorial churn",
            "rss_total": "Headline pressure",
            "media_pressure_score": "Score shock",
            "media_pressure_score_7d_avg": "Sustained escalation",
        }
        regime_names[index] = " / ".join(readable[item] for item in dominant)

    frame["regime_id"] = labels
    frame["regime_label"] = frame["regime_id"].map(regime_names)
    frame["regime_change"] = frame["regime_id"].ne(frame["regime_id"].shift(1))
    return frame


def compute_anomalies(dataset_df: pd.DataFrame, theme_daily: pd.DataFrame) -> pd.DataFrame:
    base = dataset_df[[
        "date",
        "media_pressure_score",
        "wiki_pageviews_total",
        "wiki_revision_count",
        "rss_total",
    ]].copy()
    if not theme_daily.empty:
        base = base.merge(theme_daily, on="date", how="left")
    base = base.fillna(0).sort_values("date")
    feature_columns = [column for column in base.columns if column != "date"]
    detector = IsolationForest(random_state=42, contamination=0.05)
    detector.fit(base[feature_columns])
    scores = -detector.score_samples(base[feature_columns])
    labels = detector.predict(base[feature_columns])
    base["anomaly_score"] = scores
    base["anomaly_flag"] = labels == -1
    threshold = float(base["anomaly_score"].quantile(0.95))
    base["anomaly_severity"] = np.where(base["anomaly_score"] >= threshold, "high", "normal")
    return base


def compute_forecast(dataset_df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    frame = build_autoregressive_frame(dataset_df)
    train_cutoff = max(30, int(len(frame) * 0.8))
    train = frame.iloc[:train_cutoff].copy()
    test = frame.iloc[train_cutoff:].copy()

    regressor = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=1,
    )
    regressor.fit(train[AUTOREGRESSIVE_FEATURES], train["media_pressure_score"])

    residual_sigma = 0.0
    model_mae = 0.0
    if not test.empty:
        predicted_test = regressor.predict(test[AUTOREGRESSIVE_FEATURES])
        model_mae = float(mean_absolute_error(test["media_pressure_score"], predicted_test))
        residual_sigma = float(np.std(test["media_pressure_score"] - predicted_test))

    history = dataset_df[["date", "media_pressure_score"]].copy().sort_values("date")
    future_rows = []
    for step in range(1, horizon + 1):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        enriched = build_autoregressive_frame(history)
        last_features = enriched.iloc[[-1]].copy()
        last_features["date"] = next_date
        last_features["day_of_week"] = next_date.dayofweek
        last_features["month"] = next_date.month
        last_features["days_since_start"] = int((next_date - history["date"].min()).days)
        forecast_value = float(regressor.predict(last_features[AUTOREGRESSIVE_FEATURES])[0])
        future_rows.append(
            {
                "date": next_date,
                "forecast_score": forecast_value,
                "lower_bound": max(0.0, forecast_value - 1.96 * residual_sigma),
                "upper_bound": min(100.0, forecast_value + 1.96 * residual_sigma),
                "model_mae": model_mae,
            }
        )
        history = pd.concat(
            [
                history,
                pd.DataFrame({"date": [next_date], "media_pressure_score": [forecast_value]}),
            ],
            ignore_index=True,
        )

    forecast_df = pd.DataFrame(future_rows)
    forecast_df["exceeds_threshold_prob"] = np.clip(
        (forecast_df["forecast_score"] - forecast_df["lower_bound"])
        / (forecast_df["upper_bound"] - forecast_df["lower_bound"]).replace(0, np.nan),
        0,
        1,
    ).fillna(0.5)
    return forecast_df


def compute_shap_explainability(
    dataset_df: pd.DataFrame,
    model_artifact: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    if not model_artifact:
        return pd.DataFrame(), pd.DataFrame(), []

    features = model_artifact["features"]
    model = model_artifact["model"]
    available = dataset_df[features].fillna(0)
    latest_row = available.iloc[[-1]]
    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance": getattr(model, "feature_importances_", np.zeros(len(features))),
        }
    ).sort_values("importance", ascending=False)

    if shap is None:
        top_drivers = []
        latest_values = latest_row.iloc[0]
        for _, row in feature_importance.head(6).iterrows():
            feature = str(row["feature"])
            top_drivers.append(
                {
                    "feature": feature,
                    "value": f"{float(latest_values[feature]):.2f}",
                    "effect": f"model importance {float(row['importance']):.3f}",
                }
            )
        return pd.DataFrame(), feature_importance, top_drivers

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(latest_row)
    if isinstance(shap_values, list):
        positive_class = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        latest_shap = np.asarray(positive_class)[0]
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            latest_shap = shap_array[0, :, 1]
        else:
            latest_shap = shap_array[0]

    shap_frame = pd.DataFrame(
        {
            "feature": features,
            "value": latest_row.iloc[0].values,
            "shap_value": latest_shap,
        }
    )
    shap_frame["abs_shap"] = shap_frame["shap_value"].abs()
    shap_frame = shap_frame.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    top_drivers = []
    for _, row in shap_frame.head(6).iterrows():
        direction = "pushes risk higher" if row["shap_value"] > 0 else "pushes risk lower"
        top_drivers.append(
            {
                "feature": str(row["feature"]),
                "value": f"{row['value']:.2f}",
                "effect": f"{direction} ({row['shap_value']:+.3f})",
            }
        )

    return shap_frame, feature_importance, top_drivers


def build_region_snapshot(
    region_daily: pd.DataFrame,
    topic_activity: pd.DataFrame,
    recent_events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    latest_region_date = region_daily["date"].max() if not region_daily.empty else None
    latest_topic_date = topic_activity["date"].max() if not topic_activity.empty else None

    for region, meta in REGION_META.items():
        recent_mentions = 0.0
        region_momentum = 0.0
        if latest_region_date is not None and f"region_{region}" in region_daily.columns:
            recent_slice = region_daily[region_daily["date"] >= latest_region_date - pd.Timedelta(days=6)]
            prior_slice = region_daily[
                (region_daily["date"] < latest_region_date - pd.Timedelta(days=6))
                & (region_daily["date"] >= latest_region_date - pd.Timedelta(days=13))
            ]
            recent_mentions = float(recent_slice[f"region_{region}"].sum())
            region_momentum = recent_mentions - float(prior_slice[f"region_{region}"].sum()) if not prior_slice.empty else recent_mentions

        topic_views = 0.0
        topic_revisions = 0.0
        if latest_topic_date is not None:
            topic_slice = topic_activity[
                (topic_activity["region"] == region)
                & (topic_activity["date"] >= latest_topic_date - pd.Timedelta(days=6))
            ]
            topic_views = float(topic_slice["views"].sum()) if not topic_slice.empty else 0.0
            topic_revisions = float(topic_slice["revision_count"].sum()) if not topic_slice.empty else 0.0

        latest_headline = ""
        if not recent_events.empty:
            matched = recent_events[
                recent_events["body_text"].str.lower().str.contains("|".join(REGION_META[region]["keywords"]), na=False)
            ]
            if not matched.empty:
                latest_headline = str(matched.iloc[0]["title_clean"])

        rows.append(
            {
                "region": region,
                "lat": meta["lat"],
                "lon": meta["lon"],
                "recent_mentions": recent_mentions,
                "weekly_momentum": region_momentum,
                "wiki_views_7d": topic_views,
                "wiki_revisions_7d": topic_revisions,
                "latest_headline": latest_headline,
            }
        )

    snapshot = pd.DataFrame(rows)
    snapshot["risk_intensity"] = (
        minmax_series(snapshot["recent_mentions"])
        * 0.55
        + minmax_series(snapshot["wiki_views_7d"])
        * 0.30
        + minmax_series(snapshot["wiki_revisions_7d"])
        * 0.15
    )
    snapshot["risk_intensity"] = snapshot["risk_intensity"].round(4)
    snapshot["marker_size"] = 18 + 42 * snapshot["risk_intensity"]
    return snapshot.sort_values("risk_intensity", ascending=False).reset_index(drop=True)


def build_alerts(
    dataset_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    theme_daily: pd.DataFrame,
    region_snapshot: pd.DataFrame,
    prediction_probability: float,
) -> list[dict[str, str]]:
    latest_date = dataset_df["date"].max().strftime("%Y-%m-%d")
    latest_anomaly = anomaly_df.iloc[-1] if not anomaly_df.empty else None
    latest_theme = theme_daily.iloc[-1] if not theme_daily.empty else None
    military_intensity = 0.0
    energy_intensity = 0.0
    if latest_theme is not None:
        military_intensity = normalize_value(
            float(latest_theme.get("theme_military", 0.0)),
            theme_daily.get("theme_military", pd.Series(dtype=float)),
        )
        energy_intensity = normalize_value(
            float(latest_theme.get("theme_energy", 0.0)),
            theme_daily.get("theme_energy", pd.Series(dtype=float)),
        )

    level = classify_alert_level(
        prediction_probability,
        bool(latest_anomaly["anomaly_flag"]) if latest_anomaly is not None else False,
        military_intensity,
    )
    alerts = [
        build_alert_payload(
            level,
            "Escalation probability outlook",
            "Model-driven estimate for next-day escalation pressure using the current observed state.",
            f"{prediction_probability:.1%}",
            latest_date,
        )
    ]

    if latest_anomaly is not None and bool(latest_anomaly["anomaly_flag"]):
        alerts.append(
            build_alert_payload(
                "critical" if float(latest_anomaly["anomaly_score"]) >= float(anomaly_df["anomaly_score"].quantile(0.95)) else "moderate",
                "Anomalous signal cluster detected",
                "The multivariate anomaly detector flagged the latest observation as outside the historical norm.",
                f"{float(latest_anomaly['anomaly_score']):.3f}",
                latest_date,
            )
        )

    hottest_region = region_snapshot.iloc[0] if not region_snapshot.empty else None
    if hottest_region is not None and float(hottest_region["recent_mentions"]) > 0:
        alerts.append(
            build_alert_payload(
                "moderate" if float(hottest_region["risk_intensity"]) < 0.7 else "critical",
                f"Regional hotspot: {hottest_region['region']}",
                "This zone currently concentrates the strongest mixture of headline pressure and public attention.",
                f"{float(hottest_region['risk_intensity']):.2f}",
                latest_date,
            )
        )

    if latest_theme is not None and float(latest_theme.get("theme_energy", 0.0)) > 0:
        alerts.append(
            build_alert_payload(
                "info" if energy_intensity < 0.5 else "moderate",
                "Energy corridor stress signal",
                "Energy and shipping-related language is present in the current headline stream.",
                f"{int(latest_theme.get('theme_energy', 0))} mentions",
                latest_date,
            )
        )

    if not forecast_df.empty:
        peak_row = forecast_df.sort_values("forecast_score", ascending=False).iloc[0]
        alerts.append(
            build_alert_payload(
                "info",
                "Short-horizon forecast peak",
                "Highest forecasted media-pressure window over the next 10 days.",
                f"{peak_row['date'].strftime('%Y-%m-%d')} · {peak_row['forecast_score']:.1f}",
                latest_date,
            )
        )

    return alerts[:5]


def build_model_metrics_frame(metrics: dict[str, Any] | None) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    rows = []
    for model_name, values in metrics["results"].items():
        row = {"model": model_name}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)


def build_intelligence_snapshot(
    dataset_df: pd.DataFrame,
    rss_df: pd.DataFrame,
    wikimedia_raw_df: pd.DataFrame,
    revisions_raw_df: pd.DataFrame,
    model_artifact: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
) -> IntelligenceSnapshot:
    theme_daily, region_daily, enriched_rss = prepare_rss_intelligence(rss_df)
    topic_activity = prepare_wiki_topic_activity(wikimedia_raw_df, revisions_raw_df)
    anomaly_frame = compute_anomalies(dataset_df, theme_daily)
    forecast_frame = compute_forecast(dataset_df)
    regime_frame = compute_conflict_regimes(dataset_df)
    shap_frame, feature_importance_frame, top_drivers = compute_shap_explainability(
        dataset_df,
        model_artifact,
    )
    recent_events = enriched_rss.sort_values("date", ascending=False).reset_index(drop=True)
    region_snapshot = build_region_snapshot(region_daily, topic_activity, recent_events)
    model_metrics = build_model_metrics_frame(metrics)

    latest_row = dataset_df.iloc[-1]
    previous_week = dataset_df.tail(7)["media_pressure_score"].mean()
    prior_week = dataset_df.tail(14).head(7)["media_pressure_score"].mean() if len(dataset_df) >= 14 else previous_week
    previous_month = dataset_df.tail(30)["media_pressure_score"].mean()
    prior_month = dataset_df.tail(60).head(30)["media_pressure_score"].mean() if len(dataset_df) >= 60 else previous_month

    prediction_probability = 0.0
    model_confidence = 0.0
    if model_artifact:
        features = model_artifact["features"]
        model = model_artifact["model"]
        latest_features = dataset_df.iloc[[-1]][features].fillna(0)
        if hasattr(model, "predict_proba"):
            prediction_probability = float(model.predict_proba(latest_features)[0, 1])
            model_confidence = max(prediction_probability, 1 - prediction_probability)

    anomaly_latest = anomaly_frame.iloc[-1] if not anomaly_frame.empty else None
    threat_level = classify_alert_level(
        prediction_probability,
        bool(anomaly_latest["anomaly_flag"]) if anomaly_latest is not None else False,
        normalize_value(
            latest_notna(theme_daily.get("theme_military", pd.Series(dtype=float))),
            theme_daily.get("theme_military", pd.Series(dtype=float)),
        ) if not theme_daily.empty else 0.0,
    )

    alerts = build_alerts(
        dataset_df,
        forecast_frame,
        anomaly_frame,
        theme_daily,
        region_snapshot,
        prediction_probability,
    )

    executive = {
        "threat_level": ALERT_LEVEL_META[threat_level]["label"],
        "threat_color": ALERT_LEVEL_META[threat_level]["color"],
        "escalation_probability": prediction_probability,
        "model_confidence": model_confidence,
        "conflict_intensity": float(latest_row["media_pressure_score"]),
        "weekly_trend": float(previous_week - prior_week),
        "monthly_trend": float(previous_month - prior_month),
        "global_risk_proxy": float(dataset_df.tail(14)["media_pressure_score"].mean()),
        "regional_hotspot": str(region_snapshot.iloc[0]["region"]) if not region_snapshot.empty else "N/D",
        "regional_risk_proxy": float(region_snapshot.iloc[0]["risk_intensity"]) if not region_snapshot.empty else 0.0,
        "military_signal": int(latest_notna(theme_daily.get("theme_military", pd.Series(dtype=float)), 0.0)),
        "energy_signal": int(latest_notna(theme_daily.get("theme_energy", pd.Series(dtype=float)), 0.0)),
        "diplomacy_signal": int(latest_notna(theme_daily.get("theme_diplomacy", pd.Series(dtype=float)), 0.0)),
        "risk_signal": int(latest_notna(theme_daily.get("theme_risk", pd.Series(dtype=float)), 0.0)),
        "anomaly_flag": bool(anomaly_latest["anomaly_flag"]) if anomaly_latest is not None else False,
        "anomaly_score": float(anomaly_latest["anomaly_score"]) if anomaly_latest is not None else 0.0,
        "forecast_peak": float(forecast_frame["forecast_score"].max()) if not forecast_frame.empty else float(latest_row["media_pressure_score"]),
        "forecast_peak_date": forecast_frame.loc[forecast_frame["forecast_score"].idxmax(), "date"].strftime("%Y-%m-%d") if not forecast_frame.empty else latest_row["date"].strftime("%Y-%m-%d"),
        "score_threshold": float(latest_row["target_threshold_q75"]),
    }

    return IntelligenceSnapshot(
        executive=executive,
        region_snapshot=region_snapshot,
        recent_events=recent_events,
        theme_daily=theme_daily,
        region_daily=region_daily,
        topic_activity=topic_activity,
        alerts=alerts,
        anomaly_frame=anomaly_frame,
        forecast_frame=forecast_frame,
        regime_frame=regime_frame,
        model_metrics=model_metrics,
        shap_frame=shap_frame,
        feature_importance_frame=feature_importance_frame,
        top_drivers=top_drivers,
    )


def load_raw_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates or None)


def retrain_if_needed(project_root: Path) -> None:
    artifact_path = project_root / "models" / "best_escalation_model.pkl"
    if not artifact_path.exists():
        return
    artifact = joblib.load(artifact_path)
    version = getattr(artifact["model"], "__sklearn_version__", None)
    if version:
        return
