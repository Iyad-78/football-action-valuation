from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "football-action-valuation-matplotlib")
)

import matplotlib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def probability_metrics(y_true: Any, y_probability: Any) -> dict[str, float | None]:
    truth = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(y_probability, dtype=float), 1e-7, 1 - 1e-7)
    result: dict[str, float | None] = {
        "samples": float(len(truth)),
        "positive_rate": float(truth.mean()) if len(truth) else None,
        "brier_score": float(brier_score_loss(truth, probabilities)),
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "roc_auc": None,
        "average_precision": None,
    }
    if len(np.unique(truth)) == 2:
        result["roc_auc"] = float(roc_auc_score(truth, probabilities))
        result["average_precision"] = float(average_precision_score(truth, probabilities))
    return result


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def save_calibration_plot(predictions: Any, path: Path, *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Perfect calibration")

    colors = {"validation": "#2563eb", "test": "#ea580c", "showcase": "#16a34a"}
    for split in ("validation", "test", "showcase"):
        subset = predictions.loc[predictions["split"] == split]
        if subset.empty or subset["goal"].nunique() < 2:
            continue
        observed, predicted = calibration_curve(
            subset["goal"], subset["predicted_xg"], n_bins=6, strategy="quantile"
        )
        axis.plot(
            predicted,
            observed,
            marker="o",
            linewidth=2,
            color=colors[split],
            label=f"{split} (n={len(subset)})",
        )

    axis.set(
        title=title,
        xlabel="Mean predicted probability",
        ylabel="Observed goal frequency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
