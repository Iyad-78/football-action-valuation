from __future__ import annotations

import numpy as np
import pandas as pd

from football_action_valuation.data.actions import ACTION_TYPES

NUMERIC = ["x", "y", "end_x", "end_y", "success", "pressure", "penalty", "score_diff", "minute"]
TYPE_MAP = {name: i + 1 for i, name in enumerate(sorted(ACTION_TYPES))}


def encode_sequences(frame: pd.DataFrame, window: int, codec: dict | None = None):
    """Identical history for all models; no match/period/goal-boundary crossing.

    Right padding, current actor's coordinate system, train-only scaling.
    Names, IDs, rewards and future targets are never model inputs.
    """
    if not 1 <= window <= 30:
        raise ValueError("window must be between 1 and 30")
    if frame.empty:
        raise ValueError("No actions supplied")
    if frame.duplicated(["match_id", "event_id"]).any():
        raise ValueError("Duplicate event IDs within a match")
    if codec is not None and codec.get("action_types") != TYPE_MAP:
        raise ValueError("Model vocabulary differs from this code version; retrain the model")
    values = frame[NUMERIC].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Action features must be finite")
    types = frame.action.map(TYPE_MAP).fillna(0).to_numpy(dtype=np.int64)
    n = len(frame)
    numeric = np.zeros((n, window, len(NUMERIC) + 2), dtype=np.float32)
    categorical = np.zeros((n, window), dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    teams = frame.team_id.to_numpy()
    times = frame.time.to_numpy(dtype=np.float32)
    keys = list(zip(frame.match_id, frame.period, frame.segment, strict=True))
    start = 0
    for i in range(n):
        if i == 0 or keys[i] != keys[i - 1]:
            start = i
        first = max(start, i - window + 1)
        count = i - first + 1
        lengths[i] = count
        block = values[first : i + 1].copy()
        same = teams[first : i + 1] == teams[i]
        block[~same, 0] = 120 - block[~same, 0]
        block[~same, 1] = 80 - block[~same, 1]
        block[~same, 2] = 120 - block[~same, 2]
        block[~same, 3] = 80 - block[~same, 3]
        block[~same, 7] *= -1
        numeric[i, :count, : len(NUMERIC)] = block
        numeric[i, :count, -2] = same
        numeric[i, :count, -1] = np.minimum(times[i] - times[first : i + 1], 120)
        categorical[i, :count] = types[first : i + 1]
    mask = np.arange(window)[None, :] < lengths[:, None]
    if codec is None:
        training = numeric[(frame.split.eq("train").to_numpy()[:, None] & mask)]
        codec = {
            "mean": training.mean(axis=0),
            "std": training.std(axis=0).clip(1e-5),
            "window": window,
            "features": NUMERIC + ["same_team", "age_seconds"],
            "action_types": TYPE_MAP,
        }
    numeric = (numeric - codec["mean"]) / codec["std"]
    numeric[~mask] = 0
    return numeric, categorical, lengths, codec


def tabular_view(numeric, categorical, lengths):
    onehot = np.eye(len(TYPE_MAP) + 1, dtype=np.float32)[categorical]
    mask = np.arange(categorical.shape[1])[None, :] < lengths[:, None]
    onehot[~mask] = 0
    return np.concatenate([numeric, onehot, mask[..., None]], axis=2).reshape(len(numeric), -1)
