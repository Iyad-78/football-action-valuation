from pathlib import Path

import numpy as np
import pandas as pd

from football_action_valuation.features.xg import MODEL_FEATURES
from football_action_valuation.models.xg import train_xg_model


def _synthetic_dataset(path: Path) -> None:
    generator = np.random.default_rng(42)
    rows = []
    for index in range(180):
        distance = float(generator.uniform(4, 40))
        probability = 1 / (1 + np.exp((distance - 14) / 4))
        split = "train" if index < 120 else "validation" if index < 150 else "test"
        rows.append(
            {
                "event_id": str(index),
                "match_id": index // 10,
                "match_date": "2022-01-01",
                "split": split,
                "statsbomb_xg": probability,
                "goal": int(generator.random() < probability),
                "x": 120 - distance,
                "y": float(generator.uniform(25, 55)),
                "distance_to_goal": distance,
                "shot_angle": float(generator.uniform(0.05, 0.8)),
                "minute": index % 90,
                "period": 1 if index % 90 < 45 else 2,
                "score_difference": 0,
                "body_part": "Right Foot",
                "technique": "Normal",
                "shot_type": "Open Play",
                "play_pattern": "Regular Play",
                "position": "Center Forward",
                "under_pressure": bool(index % 2),
                "first_time": False,
                "one_on_one": bool(index % 7 == 0),
                "open_goal": False,
            }
        )
    frame = pd.DataFrame(rows)
    assert set(MODEL_FEATURES) <= set(frame.columns)
    frame.to_csv(path, index=False)


def test_logistic_training_writes_reproducible_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "shots.csv"
    artifacts = tmp_path / "artifacts"
    _synthetic_dataset(dataset)

    result = train_xg_model(dataset, artifacts, model_name="logistic")

    assert Path(result["model"]).exists()
    assert Path(result["predictions"]).exists()
    assert Path(result["metrics"]).exists()
    assert result["scores"]["test"]["model"]["brier_score"] >= 0

