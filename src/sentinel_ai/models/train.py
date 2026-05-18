from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from sentinel_ai.data.synthetic import FEATURE_COLUMNS, TARGET_COLUMN
from sentinel_ai.features.preprocess import build_feature_pipeline


def train_model(
    data_path: Path,
    model_path: Path,
    metrics_path: Path,
    seed: int = 42,
) -> dict[str, float]:
    data = pd.read_csv(data_path)
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=220,
        l2_regularization=0.05,
        random_state=seed,
    )
    pipeline = build_feature_pipeline(model)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "rows": float(len(data)),
        "positive_rate": round(float(y.mean()), 4),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)
