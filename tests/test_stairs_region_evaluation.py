from __future__ import annotations

import pytest

from pbfm.evaluation.terrain import (
    _max_consecutive_stair_transitions,
    _separated_stairs_progress_metrics,
    _stairs_pre_ascent_offset,
    _stairs_progress_metrics,
)


def test_pre_ascent_reset_stays_inside_center_platform() -> None:
    assert _stairs_pre_ascent_offset(platform_width=1.0, edge_margin=0.35) == pytest.approx(0.15)


def test_stairs_progress_distinguishes_ascent_and_descent() -> None:
    ascent = _stairs_progress_metrics(
        ground_heights=[0.0, 0.1, 0.2, 0.4, 0.6],
        ground_clearances=[0.8] * 5,
        body_impacts=[0.1] * 4,
        step_height=0.1,
        num_steps=6,
        fall_clearance=0.45,
        min_descent_steps=3,
        max_allowed_body_impact=1.0,
    )
    assert ascent["ascent_initiated"]
    assert ascent["high_platform_reached"]
    assert ascent["ascent_success"]
    assert not ascent["descent_initiated"]

    descent = _stairs_progress_metrics(
        ground_heights=[0.6, 0.5, 0.4, 0.3],
        ground_clearances=[0.8, 0.75, 0.7, 0.7],
        body_impacts=[0.2, 0.3, 0.2],
        step_height=0.1,
        num_steps=6,
        fall_clearance=0.45,
        min_descent_steps=3,
        max_allowed_body_impact=1.0,
    )
    assert descent["descent_initiated"]
    assert descent["descending_steps_completed"] == 3
    assert descent["descent_success"]
    assert not descent["low_platform_reached"]


@pytest.mark.parametrize(
    ("impacts", "clearances"),
    [([2.0], [0.8, 0.8]), ([0.1], [0.8, 0.2])],
)
def test_descent_success_rejects_falls(
    impacts: list[float], clearances: list[float]
) -> None:
    result = _stairs_progress_metrics(
        ground_heights=[0.6, 0.2],
        ground_clearances=clearances,
        body_impacts=impacts,
        step_height=0.1,
        num_steps=6,
        fall_clearance=0.45,
        min_descent_steps=3,
        max_allowed_body_impact=1.0,
    )
    assert not result["descent_success"]


def test_separated_stairs_metrics_measure_center_to_outer_transition() -> None:
    result = _separated_stairs_progress_metrics(
        terrain="stairs_up",
        ground_heights=[-1.0, -1.0, -0.9, -0.8, -0.7],
        ground_clearances=[0.8] * 5,
        body_impacts=[0.1] * 4,
        planar_radii=[0.0, 0.3, 0.5, 0.8, 1.1],
        cumulative_planar_path=1.1,
        step_height=0.1,
        num_steps=3,
        center_width=0.8,
        fall_clearance=0.45,
        max_allowed_body_impact=1.0,
    )
    assert result["center_departed"]
    assert result["center_departure_step"] == 2
    assert result["first_transition"]
    assert result["first_transition_step"] == 2
    assert result["consecutive_steps_completed"] == 3
    assert result["outer_ground_reached"]
    assert not result["stalled_at_center"]


def test_separated_stairs_metrics_identify_center_loop() -> None:
    result = _separated_stairs_progress_metrics(
        terrain="stairs_down",
        ground_heights=[1.0] * 5,
        ground_clearances=[0.8] * 5,
        body_impacts=[0.0] * 4,
        planar_radii=[0.0, 0.2, 0.3, 0.2, 0.3],
        cumulative_planar_path=1.7,
        step_height=0.1,
        num_steps=10,
        center_width=0.8,
        fall_clearance=0.45,
        max_allowed_body_impact=1.0,
    )
    assert not result["center_departed"]
    assert result["stalled_at_center"]
    assert result["center_looped"]


def test_consecutive_stair_transitions_reject_skipped_levels() -> None:
    assert _max_consecutive_stair_transitions([0, 0, 1, 1, 2, 3]) == 3
    assert _max_consecutive_stair_transitions([0, 2, 3]) == 1


