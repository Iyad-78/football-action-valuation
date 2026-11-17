import numpy as np
import pandas as pd
import pytest

from football_action_valuation.analysis.actions import value_actions
from football_action_valuation.data.actions import extract_actions, player_minutes
from football_action_valuation.features.actions import encode_sequences


def match():
    return {
        "match_id": 1,
        "match_date": "2022-01-01",
        "kick_off": "12:00:00",
        "home_team": {"home_team_id": 1},
        "away_team": {"away_team_id": 2},
        "competition_stage": {"name": "Group Stage"},
    }


def event(index, team=1, kind="Pass", period=1, **extra):
    return dict(
        id=str(index),
        index=index,
        period=period,
        timestamp=f"00:{index:02d}:00",
        minute=index,
        type={"name": kind},
        team={"id": team, "name": str(team)},
        player={"id": team, "name": f"Player {team}"},
        location=[100, 40],
        **extra,
    )


def test_future_labels_perspective_period_and_shootout():
    events = [
        event(1),
        event(2, team=2),
        event(3, kind="Shot", shot={"outcome": {"name": "Goal"}}),
        event(4),
        event(5, period=2),
        event(6, period=5, kind="Shot"),
    ]
    frame = extract_actions(events, match(), horizon=2)
    assert frame.iloc[0].scores == 1
    assert frame.iloc[1].concedes == 1
    assert frame.iloc[2].label_valid == 0  # goal is a reward, not a learning example
    assert frame.iloc[3].label_valid == 0  # right-censored period
    assert frame.period.max() == 2
    assert frame.iloc[2].segment != frame.iloc[3].segment


def test_own_goal_is_counted_once():
    frame = extract_actions(
        [
            event(1),
            event(2, kind="Own Goal Against"),
            event(3, team=2, kind="Own Goal For"),
            event(4),
        ],
        match(),
        1,
    )
    assert frame.iloc[0].concedes == 1
    assert frame.reward.tolist() == [0, -1, 0]
    assert frame.iloc[-1].score_diff == -1


def test_minutes_substitutions_and_shootouts():
    lineup = event(
        0,
        kind="Starting XI",
        tactics={
            "lineup": [{"player": {"id": 1, "name": "A"}, "position": {"name": "Center Forward"}}]
        },
    )
    substitution = event(
        60, kind="Substitution", substitution={"replacement": {"id": 3, "name": "B"}}
    )
    substitution["timestamp"] = "01:00:00"
    end = event(90, kind="Half End")
    end["timestamp"] = "01:30:00"
    players = player_minutes([lineup, substitution, end, event(59, period=5)], match())
    assert players[0]["minutes"] == 60
    assert players[1]["minutes"] == 30


def test_value_switches_team_and_resets_after_goal():
    frame = pd.DataFrame(
        [
            dict(match_id=1, period=1, segment=1, team_id=1, p_score=0.2, p_concede=0.1, reward=0),
            dict(match_id=1, period=1, segment=1, team_id=2, p_score=0.3, p_concede=0.05, reward=0),
            dict(match_id=1, period=1, segment=1, team_id=2, p_score=0.1, p_concede=0.1, reward=1),
            dict(
                match_id=1, period=1, segment=2, team_id=1, p_score=0.02, p_concede=0.01, reward=0
            ),
        ]
    )
    valued = value_actions(frame, [0.02, 0.01])
    assert valued.iloc[1].action_value == pytest.approx(0.35)
    assert valued.iloc[2].action_value == pytest.approx(0.75)
    assert valued.iloc[3].action_value == pytest.approx(0)


def test_penalty_cost_is_charged_to_fouler_not_twice_to_shooter():
    frame = pd.DataFrame(
        [
            dict(
                match_id=1,
                period=1,
                segment=1,
                team_id=1,
                p_score=0.01,
                p_concede=0.02,
                reward=0,
                action="Foul Committed",
                penalty=1,
            ),
            dict(
                match_id=1,
                period=1,
                segment=1,
                team_id=2,
                p_score=0.1,
                p_concede=0.1,
                reward=1,
                action="Shot",
                penalty=1,
            ),
        ]
    )
    valued = value_actions(frame, [0.02, 0.01], 0.75)
    assert valued.iloc[0].action_value == pytest.approx(-0.76)
    assert valued.iloc[1].action_value == pytest.approx(0.25)


def test_saved_models_reproduce_predictions(tmp_path):
    from football_action_valuation.models.actions import predict_saved, train_actions

    # Independent of the football downloads: repeated fixtures with varied geometry.
    frames = []
    for match_id in range(12):
        metadata = {**match(), "match_id": match_id}
        events = [
            event(
                i,
                team=1 + i % 2,
                kind="Shot" if i % 7 == 0 else "Pass",
                shot={"outcome": {"name": "Goal" if i % 7 == 0 else "Saved"}},
            )
            for i in range(1, 40)
        ]
        frame = extract_actions(events, metadata, horizon=3)
        frame["split"] = ["train", "validation", "test", "showcase"][match_id // 3]
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    dataset = tmp_path / "actions.csv"
    frame.to_csv(dataset, index=False)
    selected = train_actions(dataset, tmp_path / "models", ["logistic"], window=3, epochs=1)
    assert selected["selected_model"] == "logistic"
    model_dir = tmp_path / "models" / "logistic"
    predictions = pd.read_csv(model_dir / "predictions.csv")
    reloaded = predict_saved(frame, model_dir)
    np.testing.assert_allclose(predictions[["p_score", "p_concede"]], reloaded, atol=1e-7)


def test_sequences_do_not_cross_boundaries_or_read_future():
    frame = extract_actions([event(1), event(2, team=2), event(3, period=2)], match(), 1)
    frame["split"] = "train"
    x, types, lengths, codec = encode_sequences(frame, 3)
    assert lengths.tolist() == [1, 2, 1]
    changed = frame.copy()
    changed.loc[2, "x"] = 0
    changed["scores"] = 1
    changed["player_id"] = 9999
    x2, types2, _, _ = encode_sequences(changed, 3, codec)
    np.testing.assert_array_equal(x[:2], x2[:2])
    np.testing.assert_array_equal(types, types2)
    assert np.all(x[0, 1:] == 0)


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
def test_network_padding_invariance(kind):
    torch = pytest.importorskip("torch")
    from football_action_valuation.models.action_networks import ActionNetwork

    torch.manual_seed(4)
    network = ActionNetwork(kind, 11, 12, 4).eval()
    numeric = torch.randn(2, 4, 11)
    types = torch.ones(2, 4, dtype=torch.long)
    lengths = torch.tensor([2, 4])
    with torch.no_grad():
        before = network(numeric, types, lengths)
        numeric[0, 2:] = 100
        after = network(numeric, types, lengths)
    assert before.shape == (2, 2)
    torch.testing.assert_close(before, after)
