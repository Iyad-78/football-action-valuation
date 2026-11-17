from __future__ import annotations

from pathlib import Path

WORLD_CUP_2022_COMPETITION_ID = 43
WORLD_CUP_2022_SEASON_ID = 106

STATSBOMB_PITCH_LENGTH = 120.0
STATSBOMB_PITCH_WIDTH = 80.0
STATSBOMB_GOAL_Y_MIN = 36.0
STATSBOMB_GOAL_Y_MAX = 44.0


def find_project_root(start: Path | None = None) -> Path:
    """Find the closest parent containing ``pyproject.toml``."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current

