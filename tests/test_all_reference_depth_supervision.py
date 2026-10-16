import copy
import pytest
import torch

from pbfm.agents.inference_context import build_tracking_commands, reward_velocity_command
from pbfm.agents.velocity_context import reference_velocity_enabled, validate_reference_velocity
from test_qv_agent_integration import make_agent, make_batch, Provider


@pytest.mark.parametrize('name', ['walk1_subject1', 'run1_subject2', 'dance1_subject2',
                                 'fallAndGetUp1_subject1', 'fightAndSports1_subject4', '01_walk_A417'])
def test_all_reference_preserves_zero_velocity_and_alignment(name):
    heading = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    velocity = torch.tensor([[.4, .2], [0., 0.], [-.2, .3]])
    c = build_tracking_commands(heading, velocity, torch.tensor([[1., 0.]]),
                                motion_name=name, frames=3, policy='all_reference')
    assert c.velocity_valid.all()
    torch.testing.assert_close(c.velocity_world, velocity)
    assert c.velocity_at(1, heading[1:2])[0, 2] == 1
    assert reference_velocity_enabled(name, 'all_reference')


def test_missing_goal_is_not_invented_and_explicit_invalid_stays_invalid():
    h = torch.tensor([[1., 0.]])
    assert reward_velocity_command('arbitrary-goal', h).eq(0).all()
    c = build_tracking_commands(h, torch.zeros(1, 2), h, motion_name='dance', frames=1,
                               policy='all_reference', mode='invalid')
    assert not c.velocity_valid.any()
    with pytest.raises(ValueError):
        validate_reference_velocity(torch.full((1, 2), float('nan')), h)
    with pytest.raises(ValueError):
        validate_reference_velocity(torch.zeros(2, 2), h)


def test_depth_supervision_opt_in_preserves_actor_detach_and_forward():
    agent, dims = make_agent()
    actor = agent._model._actor
    obs = make_batch(agent, dims)['observation']
    obs['depth_image'] = torch.randint(1, 255, (8, 8, 36, 32), dtype=torch.uint8)
    z = agent._model.sample_z(8)
    before = actor(obs, z, .1)
    actor.cfg = actor.cfg.model_copy(update={'velocity_supervision_updates_depth': True})
    after = actor(obs, z, .1)
    torch.testing.assert_close(before.mean, after.mean, rtol=0, atol=0)
    torch.testing.assert_close(before.velocity_estimate, after.velocity_estimate, rtol=0, atol=0)
    after.mean.sum().backward()
    assert all(p.grad is None for p in actor.velocity_estimator.parameters())
    actor.zero_grad(set_to_none=True)
    prediction = actor(obs, z, .1).velocity_estimate
    (prediction - obs['velocity_target_yaw']).square().mean().backward()
    assert sum(p.grad.abs().sum() for p in actor.depth_encoder.parameters() if p.grad is not None) > 0
    assert sum(p.grad.abs().sum() for p in actor.velocity_estimator.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in actor.policy.parameters())


def test_depth_gradient_monitor_during_real_update():
    agent, dims = make_agent()
    actor = agent._model._actor
    actor.cfg = actor.cfg.model_copy(update={'velocity_supervision_updates_depth': True})
    batch = make_batch(agent, dims)
    metrics = agent.update({'train': Provider(batch), 'expert_slicer': Provider(batch, expert=True)}, 1024)
    assert metrics['depth_gradient_local/Estimator_norm'] > 0
    assert metrics['depth_gradient_local/FB_norm'] > 0
    assert all(torch.isfinite(v).all() for v in metrics.values())


def test_cache_dependency_includes_policy_and_contract():
    from pbfm.motion.expert import _cache_dependencies
    a = _cache_dependencies({'reference_velocity_policy': 'walk_run', 'reference_velocity_contract': 'v2'})
    b = _cache_dependencies({'reference_velocity_policy': 'all_reference', 'reference_velocity_contract': 'v2'})
    assert a != b
    assert a != _cache_dependencies({'reference_velocity_policy': 'walk_run', 'reference_velocity_contract': 'v3'})
    instantaneous = _cache_dependencies({
        'reference_velocity_policy': 'all_reference',
        'reference_velocity_contract': 'v3',
        'reference_velocity_temporal_mode': 'instantaneous',
        'reference_velocity_temporal_horizon': 1,
    })
    averaged = _cache_dependencies({
        'reference_velocity_policy': 'all_reference',
        'reference_velocity_contract': 'v3',
        'reference_velocity_temporal_mode': 'future_mean',
        'reference_velocity_temporal_horizon': 8,
    })
    assert instantaneous != averaged
