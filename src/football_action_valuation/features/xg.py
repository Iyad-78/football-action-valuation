from __future__ import annotations

import math
from typing import Any

from football_action_valuation.constants import (
    STATSBOMB_GOAL_Y_MAX,
    STATSBOMB_GOAL_Y_MIN,
    STATSBOMB_PITCH_LENGTH,
)

NUMERIC_FEATURES = [
    "x",
    "y",
    "distance_to_goal",
    "shot_angle",
    "minute",
    "period",
    "score_difference",
]

CATEGORICAL_FEATURES = [
    "body_part",
    "technique",
    "shot_type",
    "play_pattern",
    "position",
]

BINARY_FEATURES = [
    "under_pressure",
    "first_time",
    "one_on_one",
    "open_goal",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def nested_name(value: Any, *, default: str = "Unknown") -> str:
    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name is not None else default
    return default


def distance_to_goal(x: float, y: float) -> float:
    return math.hypot(STATSBOMB_PITCH_LENGTH - x, 40.0 - y)


def shot_angle(x: float, y: float) -> float:
    """Return the visible goal angle in radians for StatsBomb coordinates."""
    left = (STATSBOMB_PITCH_LENGTH - x, STATSBOMB_GOAL_Y_MIN - y)
    right = (STATSBOMB_PITCH_LENGTH - x, STATSBOMB_GOAL_Y_MAX - y)
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if left_norm == 0.0 or right_norm == 0.0:
        return math.pi / 2.0
    cosine = (left[0] * right[0] + left[1] * right[1]) / (left_norm * right_norm)
    return math.acos(max(-1.0, min(1.0, cosine)))


def coerce_binary_columns(frame: Any) -> Any:
    """Convert CSV-friendly boolean representations to zero/one values."""
    result = frame.copy()
    truthy = {True, "1", "true", "True", "TRUE"}
    for column in BINARY_FEATURES:
        result[column] = result[column].map(lambda value: int(value in truthy)).astype("int8")
    return result
