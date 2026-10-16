import copy

import pytest
import torch

from test_qv_agent_integration import make_agent, make_batch, Provider
from pbfm.agents.fb_cpr_aux.agent import velocity_estimation_loss_metrics


def test_estimator_cannot_read_goal_heading_or_privileged_fields():
    agent, dims = make_agent()
    obs = make_batch(agent, dims)['observation']
    base = agent._model.estimate_velocity(obs)
    changed = copy.deepcopy(obs)
    for key in ('heading', 'velocity_command', 'privileged_state', 'terrain_priv', 'velocity_target_yaw'):
        changed[key].fill_(12345.)
    torch.testing.assert_close(base, agent._model.estimate_velocity(changed), rtol=0, atol=0)


def test_actor_never_uses_teacher_velocity_even_when_provided():
    agent, dims = make_agent()
    obs = make_batch(agent, dims)['observation']
    z = agent._model.sample_z(8)
    base = agent._model.act(obs, z)
    obs['velocity_target_yaw'].fill_(float('nan'))
    torch.testing.assert_close(base, agent._model.act(obs, z), rtol=0, atol=0)
    for key in ('velocity_target_yaw', 'privileged_state', 'terrain_priv'):
        del obs[key]
    torch.testing.assert_close(base, agent._model.act(obs, z), rtol=0, atol=0)
    assert torch.isfinite(agent._model.estimate_velocity(obs)).all()


def test_estimator_only_receives_supervised_gradient():
    agent, dims = make_agent()
    actor = agent._model._actor
    obs = make_batch(agent, dims)['observation']
    z = agent._model.sample_z(8)
    dist = actor(obs, z, .1)
    dist.mean.sum().backward()
    assert all(p.grad is None for p in actor.velocity_estimator.parameters())
    actor.zero_grad(set_to_none=True)
    dist = actor(obs, z, .1)
    loss, _ = velocity_estimation_loss_metrics(dist.velocity_estimate, obs['velocity_target_yaw'])
    loss.backward()
    assert all(p.grad is not None for p in actor.velocity_estimator.parameters())
    estimator_ids = {id(p) for p in actor.velocity_estimator.parameters()}
    assert all(p.grad is None for p in actor.parameters() if id(p) not in estimator_ids)


def test_supervised_head_updates_with_invalid_velocity_command():
    agent, dims = make_agent()
    batch = make_batch(agent, dims)
    batch['velocity_valid'].zero_()
    batch['observation']['velocity_command'].zero_()
    batch['velocity_next'].zero_()
    before = [p.detach().clone() for p in agent._model._actor.velocity_estimator.parameters()]
    metrics = agent.update({'train': Provider(batch), 'expert_slicer': Provider(batch, expert=True)}, 1024)
    assert metrics['actor_gradient_local/V_norm'] == 0
    assert metrics['actor_gradient_local/Estimator_norm'] > 0
    assert any(not torch.equal(x,y) for x,y in zip(before, agent._model._actor.velocity_estimator.parameters()))


def test_estimator_checkpoint_and_optimizer_roundtrip(tmp_path):
    agent, dims = make_agent()
    batch = make_batch(agent, dims)
    agent.update({'train': Provider(batch), 'expert_slicer': Provider(batch, expert=True)}, 1024)
    agent.save(tmp_path)
    loaded = type(agent).load(tmp_path, device='cpu')
    obs = batch['observation']
    torch.testing.assert_close(agent._model.act(obs,batch['z']), loaded._model.act(obs,batch['z']), rtol=0,atol=0)
    torch.testing.assert_close(agent._model.estimate_velocity(obs), loaded._model.estimate_velocity(obs), rtol=0,atol=0)
    assert all(p in loaded.actor_optimizer.state for p in loaded._model._actor.velocity_estimator.parameters())


def test_false_motion_metric_and_nonfinite_target():
    prediction = torch.tensor([[.3,0.,0.], [0.,0.,0.], [1.,0.,0.]], requires_grad=True)
    target = torch.tensor([[0.,0.,0.], [0.,0.,0.], [1.,0.,0.]])
    _, metrics = velocity_estimation_loss_metrics(prediction,target)
    torch.testing.assert_close(metrics['velocity_estimator/false_motion_when_stopped'],torch.tensor(.5))
    target[0,0] = float('nan')
    with pytest.raises(ValueError):
        velocity_estimation_loss_metrics(prediction,target)


def test_full_actor_and_split_depth_are_identical():
    agent, dims = make_agent()
    batch = make_batch(agent, dims)
    obs = agent._model._normalize(batch['observation'])
    actor = agent._model._actor
    full = actor(obs,batch['z'],.1)
    split = actor.forward_from_depth_latent(obs,batch['z'],.1,actor.depth_encoder(obs['depth_image']))
    torch.testing.assert_close(full.mean,split.mean,rtol=0,atol=0)
    torch.testing.assert_close(full.velocity_estimate,split.velocity_estimate,rtol=0,atol=0)


def test_velocity_frame_matches_qv_command_frame():
    from pbfm.agents.behavior_context import root_heading_xy
    from pbfm.agents.velocity_context import world_to_yaw
    q = torch.tensor([[0.,0.,2**-.5,2**-.5]])
    local = world_to_yaw(torch.tensor([[0.,.8]]),root_heading_xy(q))
    torch.testing.assert_close(local,torch.tensor([[.8,0.]]),atol=1e-6,rtol=0)
