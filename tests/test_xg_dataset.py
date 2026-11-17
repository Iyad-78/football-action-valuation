import pandas as pd

from football_action_valuation.data.xg_dataset import assign_match_splits, extract_shots


def _match(match_id: int = 1, stage: str = "Group Stage") -> dict:
    return {
        "match_id": match_id,
        "match_date": "2022-11-20",
        "competition_stage": {"name": stage},
        "home_team": {"home_team_name": "Alpha"},
        "away_team": {"away_team_name": "Beta"},
    }


def _shot(team: str, outcome: str, x: float = 100, y: float = 40) -> dict:
    return {
        "id": f"{team}-{outcome}-{x}-{y}",
        "type": {"name": "Shot"},
        "team": {"name": team},
        "player": {"name": f"{team} player"},
        "position": {"name": "Center Forward"},
        "minute": 10,
        "second": 5,
        "period": 1,
        "location": [x, y],
        "play_pattern": {"name": "Regular Play"},
        "shot": {
            "type": {"name": "Open Play"},
            "outcome": {"name": outcome},
            "body_part": {"name": "Right Foot"},
            "technique": {"name": "Normal"},
            "statsbomb_xg": 0.25,
        },
    }


def test_extract_shots_uses_pre_shot_score() -> None:
    rows = extract_shots([_shot("Alpha", "Goal"), _shot("Beta", "Saved")], _match())
    assert rows[0]["score_difference"] == 0
    assert rows[0]["goal"] == 1
    assert rows[1]["score_difference"] == -1


def test_penalties_are_excluded_by_default_but_update_score() -> None:
    penalty = _shot("Alpha", "Goal")
    penalty["shot"]["type"] = {"name": "Penalty"}
    rows = extract_shots([penalty, _shot("Beta", "Saved")], _match())
    assert len(rows) == 1
    assert rows[0]["score_difference"] == -1


def test_splits_are_match_level_and_final_is_showcase() -> None:
    rows = []
    for match_id in range(1, 11):
        for shot_id in range(2):
            rows.append(
                {
                    "match_id": match_id,
                    "match_date": f"2022-11-{match_id:02d}",
                    "is_showcase": match_id == 10,
                    "shot_id": shot_id,
                }
            )
    result = assign_match_splits(pd.DataFrame(rows))
    assert set(result.loc[result["match_id"] == 10, "split"]) == {"showcase"}
    assert result.groupby("match_id")["split"].nunique().max() == 1
    assert {"train", "validation", "test", "showcase"} <= set(result["split"])

