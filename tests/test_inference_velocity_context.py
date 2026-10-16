import pytest
import torch

from pbfm.agents.inference_context import (
    aligned_tracking_velocity,
    build_tracking_commands,
    reward_velocity_command,
    tracking_context_arrays,
)
from pbfm.agents.velocity_context import align_reference_velocity


def example(name="walk1_subject1", mode="auto", repeats=1):
    heading = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    velocity = torch.tensor([[.8, 0.], [0., .4], [0., 0.]])
    return build_tracking_commands(heading, velocity, torch.tensor([[0., 1.]]),
                                   motion_name=name, frames=3, mode=mode, repeats=repeats)


def test_tracking_uses_full_reference_and_fixed_alignment():
    commands = example()
    torch.testing.assert_close(commands.heading_world, torch.tensor([[0., 1.], [-1., 0.], [0., -1.]]))
    torch.testing.assert_close(commands.velocity_world, torch.tensor([[0., .8], [-.4, 0.], [0., 0.]]))
    torch.testing.assert_close(commands.velocity_at(0, torch.tensor([[1., 0.]])), torch.tensor([[0., .8, 1.]]))
    torch.testing.assert_close(commands.velocity_at(0, torch.tensor([[0., 1.]])), torch.tensor([[.8, 0., 1.]]))
    # A valid stop is not an absent task.
    torch.testing.assert_close(commands.velocity_at(2, torch.tensor([[0., 1.]])), torch.tensor([[0., 0., 1.]]))


@pytest.mark.parametrize("name", ["dance1_subject2", "fallAndGetUp1_subject1", "01_walk_fast_A417"])
def test_unknown_family_is_explicitly_invalid_unless_overridden(name):
    commands = example(name)
    assert not commands.velocity_valid.any()
    assert commands.velocity_at(0, torch.tensor([[1., 0.]])).eq(0).all()
    assert example(name, mode="valid").velocity_valid.all()


def test_turning_repetition_preserves_body_relative_reference_velocity():
    commands = example(repeats=2)
    torch.testing.assert_close(commands.heading_world[3], commands.heading_world[2])
    torch.testing.assert_close(commands.velocity_world[3], torch.tensor([0., -.8]))
    assert len(commands.velocity_world) == 6
    with pytest.raises(IndexError):
        commands.velocity_at(6, torch.tensor([[1., 0.]]))


def test_training_and_inference_alignment_are_identical():
    reference = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    velocity = torch.tensor([[.8, 0.], [0., .4], [0., 0.]])
    commands = example()
    torch.testing.assert_close(commands.velocity_world,
        align_reference_velocity(velocity, reference[:1], torch.tensor([[0., 1.]])))
    torch.testing.assert_close(commands.velocity_world,
        aligned_tracking_velocity(velocity, reference, commands.heading_world))


@pytest.mark.parametrize("task,expected", [
    ("move-ego-0-0.8", [.8, 0., 1.]), ("move-ego-90-0.8", [0., .8, 1.]),
    ("move-ego--90-0.8", [0., -.8, 1.]), ("move-ego-0-0", [0., 0., 1.]),
    ("move-ego-low0.5-0-0.7", [.7, 0., 1.]), ("sitonground", [0., 0., 0.]),
])
def test_reward_semantics_are_egocentric(task, expected):
    for heading in ([[1., 0.]], [[0., 1.]]):
        torch.testing.assert_close(reward_velocity_command(task, torch.tensor(heading)),
                                   torch.tensor([expected]), atol=1e-6, rtol=1e-6)


def test_reward_invalid_and_unsupported_valid_modes():
    heading = torch.tensor([[1., 0.]])
    assert reward_velocity_command("move-ego-0-0.8", heading, mode="invalid").eq(0).all()
    with pytest.raises(ValueError):
        reward_velocity_command("sitonground", heading, mode="valid")


def test_context_records_transferable_heading0_coordinates():
    commands = example()
    arrays = tracking_context_arrays(torch.zeros(3, 8), commands, torch.tensor([[0., 1.]]), .02)
    torch.testing.assert_close(torch.from_numpy(arrays['heading_relative_xy']),
                              torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
    torch.testing.assert_close(torch.from_numpy(arrays['reference_velocity_heading0_xy']),
                              torch.tensor([[.8, 0.], [0., .4], [0., 0.]]))
    assert arrays['heading_valid'].all() and arrays['velocity_valid'].all()


def test_future_mean_uses_full_reference_and_exports_contract():
    heading = torch.tensor([[1., 0.]]).repeat(5, 1)
    velocity = torch.stack((torch.arange(5).float(), torch.zeros(5)), -1)
    commands = build_tracking_commands(
        heading, velocity, heading[:1], motion_name="walk1_subject1", frames=4,
        temporal_mode="future_mean", temporal_horizon=3,
    )
    # The final action frame can still use reference frame 4.  Smoothing an
    # already truncated four-frame tensor would incorrectly produce 3.0.
    torch.testing.assert_close(commands.velocity_world[:, 0], torch.tensor([1., 2., 3., 3.5]))
    arrays = tracking_context_arrays(torch.zeros(4, 8), commands, heading[:1], .02)
    assert arrays["velocity_temporal_mode"] == "future_mean"
    assert arrays["velocity_temporal_horizon"] == 3


def test_fixed_reward_command_is_not_temporally_smoothed():
    heading = torch.tensor([[1., 0.]])
    torch.testing.assert_close(
        reward_velocity_command("move-ego-0-0.8", heading), torch.tensor([[.8, 0., 1.]])
    )
