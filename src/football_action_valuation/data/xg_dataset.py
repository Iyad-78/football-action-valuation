from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from football_action_valuation.constants import (
    WORLD_CUP_2022_COMPETITION_ID,
    WORLD_CUP_2022_SEASON_ID,
)
from football_action_valuation.data.statsbomb import events_path, matches_path, read_json
from football_action_valuation.features.xg import (
    distance_to_goal,
    nested_name,
    shot_angle,
)


def _team_name(match: dict[str, Any], side: str) -> str:
    team = match.get(f"{side}_team", {})
    return str(team.get(f"{side}_team_name", "Unknown"))


def _opponent(team: str, home_team: str, away_team: str) -> str:
    if team == home_team:
        return away_team
    if team == away_team:
        return home_team
    return "Unknown"


def extract_shots(
    events: Iterable[dict[str, Any]],
    match: dict[str, Any],
    *,
    include_penalties: bool = False,
) -> list[dict[str, Any]]:
    """Extract leakage-safe pre-shot features from one StatsBomb match."""
    match_id = int(match["match_id"])
    home_team = _team_name(match, "home")
    away_team = _team_name(match, "away")
    scores = {home_team: 0, away_team: 0}
    rows: list[dict[str, Any]] = []

    for event in events:
        if int(event.get("period", 1)) > 4:
            continue
        event_type = nested_name(event.get("type"))
        team = nested_name(event.get("team"))

        if event_type in {"Own Goal Against", "Own Goal For"}:
            if event_type == "Own Goal Against":
                opposing_team = _opponent(team, home_team, away_team)
                scores[opposing_team] = scores.get(opposing_team, 0) + 1
            continue

        if event_type != "Shot":
            continue

        shot = event.get("shot", {})
        shot_type = nested_name(shot.get("type"))
        outcome = nested_name(shot.get("outcome"))
        is_goal = outcome == "Goal"
        if shot_type == "Penalty" and not include_penalties:
            if is_goal:
                scores[team] = scores.get(team, 0) + 1
            continue

        location = event.get("location") or [None, None]
        if len(location) < 2 or location[0] is None or location[1] is None:
            continue

        x = float(location[0])
        y = float(location[1])
        opponent = _opponent(team, home_team, away_team)
        player = nested_name(event.get("player"))
        stage = nested_name(match.get("competition_stage"))

        rows.append(
            {
                "event_id": str(event.get("id", "")),
                "match_id": match_id,
                "match_date": str(match.get("match_date", "")),
                "stage": stage,
                "home_team": home_team,
                "away_team": away_team,
                "team": team,
                "opponent": opponent,
                "player": player,
                "position": nested_name(event.get("position")),
                "minute": int(event.get("minute", 0)),
                "second": int(event.get("second", 0)),
                "period": int(event.get("period", 1)),
                "x": x,
                "y": y,
                "distance_to_goal": distance_to_goal(x, y),
                "shot_angle": shot_angle(x, y),
                "body_part": nested_name(shot.get("body_part")),
                "technique": nested_name(shot.get("technique")),
                "shot_type": shot_type,
                "play_pattern": nested_name(event.get("play_pattern")),
                "under_pressure": bool(event.get("under_pressure", False)),
                "first_time": bool(shot.get("first_time", False)),
                "one_on_one": bool(shot.get("one_on_one", False)),
                "open_goal": bool(shot.get("open_goal", False)),
                "score_for": int(scores.get(team, 0)),
                "score_against": int(scores.get(opponent, 0)),
                "score_difference": int(scores.get(team, 0) - scores.get(opponent, 0)),
                "shot_outcome": outcome,
                "goal": int(is_goal),
                "statsbomb_xg": float(shot.get("statsbomb_xg", float("nan"))),
                "is_showcase": stage.casefold() == "final",
            }
        )

        if is_goal:
            scores[team] = scores.get(team, 0) + 1

    return rows


def assign_match_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign chronological train/validation/test splits at match level."""
    result = frame.copy()
    result["split"] = "unassigned"
    result.loc[result["is_showcase"], "split"] = "showcase"

    matches = (
        result.loc[~result["is_showcase"], ["match_id", "match_date"]]
        .drop_duplicates()
        .sort_values(["match_date", "match_id"])
        .reset_index(drop=True)
    )
    count = len(matches)
    if count == 0:
        return result
    if count == 1:
        split_names = ["train"]
    elif count == 2:
        split_names = ["train", "test"]
    else:
        train_count = max(1, int(count * 0.70))
        validation_count = max(1, int(count * 0.15))
        if train_count + validation_count >= count:
            train_count = max(1, count - 2)
            validation_count = 1
        split_names = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * (count - train_count - validation_count)
        )

    split_by_match = dict(zip(matches["match_id"], split_names, strict=True))
    mask = ~result["is_showcase"]
    result.loc[mask, "split"] = result.loc[mask, "match_id"].map(split_by_match)
    return result


def prepare_xg_dataset(
    raw_dir: Path,
    output_path: Path,
    *,
    competition_id: int = WORLD_CUP_2022_COMPETITION_ID,
    season_id: int = WORLD_CUP_2022_SEASON_ID,
    include_penalties: bool = False,
) -> dict[str, Any]:
    metadata_path = matches_path(raw_dir, competition_id, season_id)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run the data download command first."
        )

    matches = read_json(metadata_path)
    all_rows: list[dict[str, Any]] = []
    processed_matches = 0
    missing_matches = 0

    for match in matches:
        match_id = int(match["match_id"])
        path = events_path(raw_dir, match_id)
        if not path.exists():
            missing_matches += 1
            continue
        all_rows.extend(
            extract_shots(
                read_json(path),
                match,
                include_penalties=include_penalties,
            )
        )
        processed_matches += 1

    if not all_rows:
        raise ValueError("No shots were extracted from the downloaded event files.")

    frame = assign_match_splits(pd.DataFrame(all_rows))
    frame = frame.sort_values(["match_date", "match_id", "period", "minute", "second"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    return {
        "processed_matches": processed_matches,
        "missing_event_files": missing_matches,
        "shots": len(frame),
        "goals": int(frame["goal"].sum()),
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "output": str(output_path),
    }

