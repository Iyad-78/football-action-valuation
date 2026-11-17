"""A deliberately small StatsBomb action vocabulary (not canonical SPADL)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_action_valuation.data.statsbomb import events_path, matches_path, read_json
from football_action_valuation.features.xg import nested_name

ACTION_TYPES = {
    "Foul Committed",
    "Pass",
    "Carry",
    "Dribble",
    "Shot",
    "Interception",
    "Ball Recovery",
    "Clearance",
    "Duel",
    "Miscontrol",
    "Dispossessed",
    "Own Goal Against",
}


def clock_seconds(event: dict) -> float:
    hours, minutes, seconds = event.get("timestamp", "00:00:00").split(":")
    return 3600 * float(hours) + 60 * float(minutes) + float(seconds)


def player_minutes(events: list[dict], match: dict) -> list[dict]:
    """Elapsed playing time, including stoppage time, excluding shootouts.

    Temporary exits are not subtracted. Roles use starting/first observed position.
    """
    lengths = {}
    for event in events:
        period = int(event.get("period", 1))
        if period <= 4:
            lengths[period] = max(lengths.get(period, 0), clock_seconds(event))
    offsets = {p: sum(v for k, v in lengths.items() if k < p) for p in lengths}
    end = sum(lengths.values())
    players = {}
    for event in events:
        period = int(event.get("period", 1))
        if period > 4:
            continue
        time = offsets[period] + clock_seconds(event)
        team = event.get("team", {})
        if nested_name(event.get("type")) == "Starting XI":
            for entry in event["tactics"]["lineup"]:
                player = entry["player"]
                players[player["id"]] = dict(
                    player_id=player["id"],
                    player=player["name"],
                    team_id=team["id"],
                    team=team["name"],
                    position=nested_name(entry.get("position")),
                    start=0.0,
                    end=end,
                )
        if nested_name(event.get("type")) == "Substitution":
            outgoing = event["player"]["id"]
            if outgoing in players:
                players[outgoing]["end"] = time
            incoming = event["substitution"]["replacement"]
            players[incoming["id"]] = dict(
                player_id=incoming["id"],
                player=incoming["name"],
                team_id=team["id"],
                team=team["name"],
                position="Unknown",
                start=time,
                end=end,
            )
        player_id = event.get("player", {}).get("id")
        if player_id in players:
            if players[player_id]["position"] == "Unknown" and "position" in event:
                players[player_id]["position"] = nested_name(event["position"])
            cards = [
                nested_name(event.get(k, {}).get("card"))
                for k in ("foul_committed", "bad_behaviour")
            ]
            if any(card in {"Red Card", "Second Yellow"} for card in cards):
                players[player_id]["end"] = time
    return [
        dict(match_id=match["match_id"], **p, minutes=max(0, p["end"] - p["start"]) / 60)
        for p in players.values()
    ]


def extract_actions(events: list[dict], match: dict, horizon: int = 10) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    rows = []
    scores = {}
    segment = 0
    previous_period = None
    teams = [match[f"{side}_team"][f"{side}_team_id"] for side in ("home", "away")]
    for event in sorted(events, key=lambda e: e["index"]):
        period = int(event.get("period", 1))
        if period > 4:
            continue
        if period != previous_period:
            segment += 1
            previous_period = period
        kind = nested_name(event.get("type"))
        if kind not in ACTION_TYPES:
            continue
        detail = event.get(kind.lower().replace(" ", "_"), {})
        if kind == "Duel" and nested_name(detail.get("type")) != "Tackle":
            continue
        player = event.get("player")
        location = event.get("location")
        if not player or not location:
            continue
        team = event["team"]
        opponent = next(t for t in teams if t != team["id"])
        outcome = nested_name(detail.get("outcome"), default="Complete")
        reward = int(kind == "Shot" and outcome == "Goal")
        reward -= int(kind == "Own Goal Against")
        end = detail.get("end_location", location)
        success = int(outcome in {"Complete", "Success", "Success In Play", "Won", "Goal"})
        if kind in {"Miscontrol", "Dispossessed", "Own Goal Against", "Foul Committed"}:
            success = 0
        if kind == "Ball Recovery":
            success = int(not detail.get("recovery_failure", False))
        rows.append(
            dict(
                event_id=event["id"],
                event_index=event["index"],
                match_id=match["match_id"],
                match_date=match["match_date"],
                kick_off=match.get("kick_off", "00:00:00"),
                is_showcase=nested_name(match.get("competition_stage")) == "Final",
                team_id=team["id"],
                team=team["name"],
                player_id=player["id"],
                player=player["name"],
                period=period,
                segment=segment,
                time=clock_seconds(event),
                minute=event.get("minute", 0),
                action=kind,
                x=float(location[0]),
                y=float(location[1]),
                end_x=float(end[0]),
                end_y=float(end[1]),
                success=success,
                pressure=int(event.get("under_pressure", False)),
                penalty=int(
                    (kind == "Shot" and nested_name(detail.get("type")) == "Penalty")
                    or (kind == "Foul Committed" and detail.get("penalty", False))
                ),
                score_diff=scores.get(team["id"], 0) - scores.get(opponent, 0),
                reward=reward,
            )
        )
        if reward:
            scorer = team["id"] if reward > 0 else opponent
            scores[scorer] = scores.get(scorer, 0) + 1
            segment += 1
    # Targets observe the future; features never do. Stop at the first goal.
    for i, row in enumerate(rows):
        future = rows[i + 1 : i + 1 + horizon]
        row["label_valid"] = int(
            row["reward"] == 0
            and len(future) == horizon
            and all(r["period"] == row["period"] for r in future)
        )
        row["scores"] = row["concedes"] = 0
        for nxt in future:
            if nxt["period"] != row["period"]:
                break
            if nxt["reward"]:
                same = nxt["team_id"] == row["team_id"]
                row["scores"] = int(same == (nxt["reward"] > 0))
                row["concedes"] = 1 - row["scores"]
                break
    return pd.DataFrame(rows)


def prepare_actions(raw_dir: Path, output_dir: Path, horizon: int = 10) -> dict:
    matches = read_json(matches_path(raw_dir, 43, 106))
    missing = [m["match_id"] for m in matches if not events_path(raw_dir, m["match_id"]).exists()]
    if missing:
        raise ValueError(f"Missing {len(missing)} matches. Run fav data download first.")
    ordinary = sorted(
        [m for m in matches if nested_name(m["competition_stage"]) != "Final"],
        key=lambda m: (m["match_date"], m["kick_off"], m["match_id"]),
    )
    split_map = {
        m["match_id"]: "train" if i < 44 else "validation" if i < 53 else "test"
        for i, m in enumerate(ordinary)
    }
    frames, players = [], []
    for match in matches:
        events = read_json(events_path(raw_dir, match["match_id"]))
        actions = extract_actions(events, match, horizon)
        actions["split"] = split_map.get(match["match_id"], "showcase")
        frames.append(actions)
        players.extend(player_minutes(events, match))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["match_date", "kick_off", "match_id", "event_index"], kind="stable")
        .reset_index(drop=True)
    )
    frame.to_csv(output_dir / "actions.csv", index=False)
    pd.DataFrame(players).to_csv(output_dir / "players.csv", index=False)
    return dict(
        actions=len(frame),
        matches=frame.match_id.nunique(),
        horizon=horizon,
        valid_labels=int(frame.label_valid.sum()),
        counts=frame.groupby("split").size().to_dict(),
    )
