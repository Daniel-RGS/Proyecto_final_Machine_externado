import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from project_paths import MODELS_DIR, PROCESSED_DIR, ensure_dir


FEATURES = [
    "volume_lag1",
    "volume_7d_avg",
    "avg_tone_lag1",
    "avg_tone_7d_avg",
    "negative_tone_lag1",
    "negative_tone_7d_avg",
    "media_pressure_score_lag1",
    "media_pressure_score_7d_avg",
    "wiki_pageviews_total_lag1",
    "wiki_pageviews_total_7d_avg",
    "wiki_revision_count_lag1",
    "wiki_revision_count_7d_avg",
    "rss_total_lag1",
    "rss_total_7d_avg",
    "firms_hotspots_count_lag1",
    "firms_hotspots_count_7d_avg",
    "day_of_week",
    "month",
    "days_since_start",
]
TARGET = "target_high_escalation_next_day"


def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            return np.zeros_like(scores, dtype=float)
        return (scores - min_score) / (max_score - min_score)
    return model.predict(X)


def _metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    scores = _predict_scores(model, X_test)
    result = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }
    if y_test.nunique() > 1:
        result["roc_auc"] = roc_auc_score(y_test, scores)
    else:
        result["roc_auc"] = None
    return result


def train_models():
    print("Entrenando modelos con dataset real procesado...")
    data_path = PROCESSED_DIR / "dataset_diario.csv"

    if not data_path.exists():
        raise FileNotFoundError("No existe dataset_diario.csv. Ejecuta scripts/build_dataset.py primero.")

    df = pd.read_csv(data_path).sort_values("date").reset_index(drop=True)
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para entrenar: {missing}")

    X = df[FEATURES]
    y = df[TARGET].astype(int)
    if y.nunique() < 2:
        raise ValueError("El target tiene una sola clase. Revisa la construccion del dataset.")

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "Dummy majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]
        ),
        "Ridge Classifier": Pipeline(
            [("scaler", StandardScaler()), ("model", RidgeClassifier(class_weight="balanced"))]
        ),
        "KNN Classifier": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7))]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        ),
    }

    results = {}
    best_name = None
    best_model = None
    best_f1 = -1.0

    print("\nResultados holdout temporal")
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_metrics = _metrics(model, X_test, y_test)
        results[name] = model_metrics
        print(
            f"{name}: f1={model_metrics['f1']:.3f}, "
            f"precision={model_metrics['precision']:.3f}, recall={model_metrics['recall']:.3f}, "
            f"accuracy={model_metrics['accuracy']:.3f}"
        )
        if name != "Dummy majority baseline" and model_metrics["f1"] > best_f1:
            best_f1 = model_metrics["f1"]
            best_name = name
            best_model = model

    models_dir = ensure_dir(MODELS_DIR)
    model_path = models_dir / "best_escalation_model.pkl"
    metrics_path = models_dir / "model_metrics.json"

    metadata = {
        "best_model": best_name,
        "selection_metric": "f1",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "results": results,
    }
    joblib.dump({"model": best_model, "features": FEATURES, "metadata": metadata}, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMejor modelo: {best_name} (F1={best_f1:.3f})")
    print(f"Modelo guardado en {model_path}")
    print(f"Metricas guardadas en {metrics_path}")
    return metadata


if __name__ == "__main__":
    train_models()
