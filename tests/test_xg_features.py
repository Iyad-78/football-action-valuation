import math

import pytest

from football_action_valuation.features.xg import (
    MODEL_FEATURES,
    distance_to_goal,
    shot_angle,
)


def test_distance_to_goal_is_shorter_near_goal() -> None:
    assert distance_to_goal(110, 40) < distance_to_goal(80, 40)


def test_central_shot_has_wider_angle_than_wide_shot() -> None:
    assert shot_angle(102, 40) > shot_angle(102, 5)
    assert 0 < shot_angle(102, 40) < math.pi


@pytest.mark.parametrize("leaky_feature", ["goal", "shot_outcome", "statsbomb_xg", "player"])
def test_model_features_do_not_contain_target_leakage(leaky_feature: str) -> None:
    assert leaky_feature not in MODEL_FEATURES

