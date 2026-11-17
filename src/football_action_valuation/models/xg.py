from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from football_action_valuation.evaluation.metrics import (
    probability_metrics,
    save_calibration_plot,
    save_json,
)
from football_action_valuation.features.xg import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    coerce_binary_columns,
)


def build_xg_pipeline(model_name: str = "logistic", *, random_state: int = 42) -> Pipeline:
    numeric_columns = NUMERIC_FEATURES + BINARY_FEATURES
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    if model_name == "logistic":
        estimator: Any = LogisticRegression(
            C=1.0,
            max_iter=2_000,
            random_state=random_state,
        )
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is optional. Install it with: pip install -e .[boosting]"
            ) from exc
        estimator = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            colsample_bytree=0.9,
            subsample=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=2,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(steps=[("preprocess", preprocessing), ("model", estimator)])


def _evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split in ("train", "validation", "test", "showcase"):
        subset = predictions.loc[predictions["split"] == split]
        if subset.empty:
            continue
        metrics[split] = {
            "model": probability_metrics(subset["goal"], subset["predicted_xg"]),
        }
        if subset["statsbomb_xg"].notna().all():
            metrics[split]["statsbomb_benchmark"] = probability_metrics(
                subset["goal"], subset["statsbomb_xg"]
            )
    return metrics


def train_xg_model(
    dataset_path: Path,
    artifact_dir: Path,
    *,
    model_name: str = "logistic",
    random_state: int = 42,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing prepared dataset: {dataset_path}")

    frame = coerce_binary_columns(pd.read_csv(dataset_path))
    train = frame.loc[frame["split"] == "train"]
    if train.empty:
        raise ValueError("The prepared dataset does not contain a training split.")
    if train["goal"].nunique() < 2:
        raise ValueError("Training data must contain both goals and non-goals.")

    pipeline = build_xg_pipeline(model_name, random_state=random_state)
    pipeline.fit(train[MODEL_FEATURES], train["goal"])

    predictions = frame.copy()
    predictions["predicted_xg"] = pipeline.predict_proba(frame[MODEL_FEATURES])[:, 1]
    metrics = _evaluate_predictions(predictions)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"xg_{model_name}.joblib"
    predictions_path = artifact_dir / f"xg_{model_name}_predictions.csv"
    metrics_path = artifact_dir / f"xg_{model_name}_metrics.json"
    metadata_path = artifact_dir / f"xg_{model_name}_metadata.json"
    calibration_path = artifact_dir / f"xg_{model_name}_calibration.png"

    joblib.dump(pipeline, model_path)
    predictions.to_csv(predictions_path, index=False)
    save_json(metrics, metrics_path)
    metadata = {
        "model": model_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "random_state": random_state,
        "features": MODEL_FEATURES,
        "training_rows": int(len(train)),
        "training_matches": int(train["match_id"].nunique()),
        "dataset": str(dataset_path),
    }
    save_json(metadata, metadata_path)
    save_calibration_plot(
        predictions,
        calibration_path,
        title=f"xG calibration — {model_name}",
    )

    return {
        "model": str(model_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "metadata": str(metadata_path),
        "calibration_plot": str(calibration_path),
        "scores": metrics,
    }
