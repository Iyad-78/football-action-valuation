from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def compare_xg_models(
    model_dir: Path,
    output_dir: Path,
    *,
    model_names: list[str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    benchmark_added: set[str] = set()

    for model_name in model_names:
        metrics_path = model_dir / f"xg_{model_name}_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics for {model_name}: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as stream:
            metrics = json.load(stream)

        for split in ("validation", "test", "showcase"):
            if split not in metrics:
                continue
            records.append(
                {
                    "model": model_name,
                    "split": split,
                    **metrics[split]["model"],
                }
            )
            if split not in benchmark_added and "statsbomb_benchmark" in metrics[split]:
                records.append(
                    {
                        "model": "statsbomb_benchmark",
                        "split": split,
                        **metrics[split]["statsbomb_benchmark"],
                    }
                )
                benchmark_added.add(split)

    comparison = pd.DataFrame(records)
    numeric = ["brier_score", "log_loss", "roc_auc", "average_precision"]
    comparison[numeric] = comparison[numeric].round(4)
    comparison = comparison.sort_values(["split", "brier_score", "model"])

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "xg_model_comparison.csv"
    markdown_path = output_dir / "xg_model_comparison.md"
    comparison.to_csv(csv_path, index=False)
    markdown_path.write_text(
        "# xG model comparison\n\n"
        "Lower is better for Brier score and log-loss; higher is better for "
        "ROC-AUC and average precision. The StatsBomb values are an external "
        "benchmark, not a model trained in this repository.\n\n"
        + comparison.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    return {
        "models": model_names,
        "rows": int(len(comparison)),
        "csv": str(csv_path),
        "report": str(markdown_path),
    }

