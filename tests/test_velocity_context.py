import pytest
import torch

from pbfm.agents.velocity_context import (
    VelocityTaskContract, align_reference_velocity, world_to_yaw, yaw_to_world,
    velocity_observation, velocity_reward, velocity_transition_masks, merge_velocity_context,
    temporal_reference_velocity,
)


def test_future_mean_matches_behavior_context_indexing_and_clips_tail():
    velocity = torch.tensor([[0., 0.], [1., 10.], [2., 20.], [3., 30.], [4., 40.]])
    expected = torch.tensor([[1.5, 15.], [2.5, 25.], [3., 30.], [3.5, 35.], [4., 40.]])
    torch.testing.assert_close(
        temporal_reference_velocity(velocity, mode="future_mean", horizon=4), expected
    )
    original = temporal_reference_velocity(velocity, mode="instantaneous", horizon=99)
    torch.testing.assert_close(original, velocity)
    assert original.data_ptr() != velocity.data_ptr()


def test_temporal_reference_rejects_invalid_contracts():
    velocity = torch.zeros(3, 2)
    for horizon in (0, -1, True):
        with pytest.raises(ValueError):
            temporal_reference_velocity(velocity, mode="future_mean", horizon=horizon)
    with pytest.raises(ValueError):
        temporal_reference_velocity(velocity, mode="centered", horizon=8)


def test_start_yaw_alignment_preserves_speed_and_turning():
    v = torch.tensor([[0.8, 0.], [0., 0.8], [0., 0.]])
    h0 = torch.tensor([1., 0.])
    h1 = torch.tensor([0., 1.])
    aligned = align_reference_velocity(v, h0, h1)
    torch.testing.assert_close(aligned, torch.tensor([[0., .8], [-.8, 0.], [0., 0.]]))
    torch.testing.assert_close(world_to_yaw(aligned, h1), v)


def test_valid_stop_is_distinct_from_unknown():
    obs = velocity_observation(torch.zeros(2, 2), torch.tensor([[1., 0.], [1., 0.]]),
                               torch.tensor([[True], [False]]))
    torch.testing.assert_close(obs, torch.tensor([[0., 0., 1.], [0., 0., 0.]]))


def test_reward_prefers_motion_to_moonwalk_and_penalizes_sideways():
    target = torch.tensor([[.8, 0.]]).repeat(4, 1)
    actual = torch.tensor([[.8, 0.], [.4, 0.], [0., 0.], [0., .8]])
    r = velocity_reward(actual, target, torch.ones(4, 1, dtype=torch.bool)).flatten()
    assert r[0] == 0 and r[0] > r[1] > r[2] > r[3]


def test_reward_is_invariant_to_world_yaw():
    a = torch.tensor([[.3, -.2], [1., 2.]])
    t = torch.tensor([[.8, .1], [0., 0.]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    h = torch.tensor([.6, .8])
    torch.testing.assert_close(velocity_reward(a, t, valid),
                               velocity_reward(yaw_to_world(a, h), yaw_to_world(t, h), valid))


def test_invalid_reward_is_zero_and_does_not_leak_nan():
    r = velocity_reward(torch.tensor([[float('nan'), 2.]]), torch.zeros(1, 2),
                        torch.zeros(1, 1, dtype=torch.bool))
    assert r.item() == 0


def test_context_switch_terminal_and_auto_reset():
    kw = dict(current_valid=torch.ones(5, 1, dtype=torch.bool),
              next_valid=torch.tensor([[1], [1], [1], [0], [1]], dtype=torch.bool),
              context_id=torch.ones(5, 1, dtype=torch.long),
              next_context_id=torch.tensor([[1], [2], [1], [1], [1]]),
              terminated=torch.tensor([[0], [0], [1], [0], [0]], dtype=torch.bool),
              truncated=torch.tensor([[0], [0], [0], [0], [1]], dtype=torch.bool),
              outcome_observed=torch.tensor([[1], [1], [1], [1], [0]], dtype=torch.bool), gamma=.98)
    usable, discount = velocity_transition_masks(**kw)
    assert usable.flatten().tolist() == [True, True, True, True, False]
    torch.testing.assert_close(discount.flatten(), torch.tensor([.98, 0., 0., 0., 0.]))


def test_relabel_velocity_uses_same_mask_and_invalidates_random_goal():
    original = torch.tensor([[.8, 0., 1.], [.5, 0., 1.]])
    sampled = torch.zeros_like(original)  # random/goal has no velocity semantics
    mask = torch.tensor([[True], [False]])
    torch.testing.assert_close(merge_velocity_context(original, sampled, mask),
                               torch.tensor([[0., 0., 0.], [.5, 0., 1.]]))


def test_bad_contract_is_rejected():
    with pytest.raises(ValueError):
        VelocityTaskContract(error_scale_mps=0)
    with pytest.raises(ValueError):
        world_to_yaw(torch.ones(1, 2), torch.zeros(1, 2))
