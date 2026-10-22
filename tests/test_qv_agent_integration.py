"""Small real neural-network updates, not a simulator or performance claim."""
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from pbfm.agents.fb_cpr_aux.config import FBcprAuxAgentConfig


def make_agent(num_dof=29, robot_train=None):
    torch.set_num_threads(2)
    repo = Path(__file__).resolve().parents[1]
    cfg = json.loads((repo / 'tests/fixtures/qv_estvel.json').read_text())['agent']
    cfg['model']['device'] = 'cpu'
    cfg['model']['archi']['z_dim'] = 8
    for key in ('actor', 'f', 'b', 'critic', 'discriminator', 'aux_critic', 'heading_critic', 'velocity_critic'):
        cfg['model']['archi'][key]['hidden_dim'] = 32
        cfg['model']['archi'][key]['hidden_layers'] = 2
    cfg['train']['batch_size'] = 8
    cfg['train']['actor_gradient_monitor_every'] = 1
    cfg['model']['obs_normalizer']['normalizers'] = {
        key: {'name': 'IdentityNormalizerConfig'} for key in cfg['model']['obs_normalizer']['normalizers']}
    dims = dict(state=(64,), privileged_state=(463,), last_action=(29,), history_actor=(372,),
                terrain_priv=(93,), heading=(2,), velocity_command=(3,), depth_image=(8,36,32))
    if num_dof == 24:
        dims.update(state=(54,), privileged_state=(373,), last_action=(24,), history_actor=(312,))
    if robot_train is not None:
        cfg['aux_rewards'] = list(robot_train.agent.aux_rewards)
        cfg['aux_rewards_scaling'] = dict(robot_train.agent.aux_rewards_scaling)
        actor = robot_train.agent.model.archi.actor
        cfg['model']['archi']['actor'].update(depth_height=actor.depth_height,
            depth_width=actor.depth_width, depth_processing_profile=actor.depth_processing_profile,
            depth_first_conv_stride=actor.depth_first_conv_stride,
            depth_frame_feature_dim=actor.depth_frame_feature_dim)
        dims['depth_image'] = (8,actor.depth_height,actor.depth_width)
    if cfg['model']['archi']['actor'].get('estimated_velocity_enabled'):
        dims['velocity_target_yaw'] = (3,)
    space = gym.spaces.Dict({key: gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
                             for key, shape in dims.items()})
    return FBcprAuxAgentConfig.model_validate(cfg).build(space, num_dof), dims


def make_batch(agent, dims):
    obs = {key: torch.randn(8, *shape) for key, shape in dims.items()}
    obs['heading'].zero_()
    obs['velocity_command'][:] = torch.tensor([.8, 0., 1.])
    heading = torch.tensor([[1., 0.]]).repeat(8, 1)
    valid = torch.ones(8, 1, dtype=torch.bool)
    batch = dict(observation=obs, action=torch.randn(8,dims['last_action'][0]), z=agent._model.sample_z(8),
        next={'observation': copy.deepcopy(obs), 'terminated': ~valid,
              'heading_forward_xy': heading, 'reference_velocity_world_xy': torch.ones(8,2)*.4,
              'reference_velocity_valid': valid},
        heading_next=obs['heading'].clone(), heading_z_next=agent._model.sample_z(8),
        heading_valid=valid, heading_source_type=torch.ones(8,1,dtype=torch.long),
        heading_reward=torch.zeros(8,1), heading_context_continues=valid,
        root_heading_xy=heading, next_root_heading_xy=heading,
        transition_terminated=~valid, transition_truncated=~valid,
        velocity_next=obs['velocity_command'].clone(), velocity_valid=valid,
        velocity_reward=-torch.ones(8,1), velocity_error_mps=torch.ones(8,1)*.5,
        heading_forward_xy=heading, reference_velocity_world_xy=torch.ones(8,2)*.4,
        reference_velocity_valid=valid,
        aux_rewards={key: torch.rand(8,1) for key in agent.cfg.aux_rewards})
    return batch


class Provider:
    def __init__(self, batch, expert=False):
        self.batch = copy.deepcopy(batch)
        if expert:
            for obs in (self.batch['observation'], self.batch['next']['observation']):
                for key in list(obs):
                    if key not in ('state', 'privileged_state', 'last_action'):
                        del obs[key]
    def sample(self, n):
        assert n == 8
        return copy.deepcopy(self.batch)


def test_real_update_qv_and_gradient_monitor():
    agent, dims = make_agent()
    batch = make_batch(agent, dims)
    before = [p.detach().clone() for p in agent._model._velocity_critic.parameters()]
    metrics = agent.update({'train':Provider(batch), 'expert_slicer':Provider(batch, expert=True)}, 1024)
    assert torch.isfinite(metrics['velocity/critic_loss'])
    assert any(not torch.equal(a,b) for a,b in zip(before,agent._model._velocity_critic.parameters()))
    assert 'actor_gradient_local/V_norm' in metrics
    assert metrics['actor_gradient_local/V_norm'] > 0
    assert 'velocity_critic_optimizer' in agent.optimizer_dict


def test_invalid_collection_context_has_no_actor_v_gradient():
    agent, dims = make_agent()
    batch = make_batch(agent, dims)
    batch['velocity_valid'].zero_()
    batch['observation']['velocity_command'].zero_()
    batch['velocity_next'].zero_()
    metrics = agent.update({'train':Provider(batch), 'expert_slicer':Provider(batch, expert=True)}, 1024)
    assert metrics['velocity/critic_loss'] == 0
    assert metrics['actor_gradient_local/V_norm'] == 0


def test_rollout_and_next_velocity_stay_with_sampled_trajectory():
    from pbfm.agents.fb_cpr_aux.agent import RolloutContextState
    agent, dims = make_agent()
    agent.cfg = agent.cfg.model_copy(update={'train': agent.cfg.train.model_copy(update={'rollout_expert_trajectories_length':4})})
    batch = make_batch(agent, dims)
    # Temporal reference velocity changes every frame; not just a constant.
    batch['reference_velocity_world_xy'] = torch.stack((torch.arange(8).float(), torch.zeros(8)), -1)
    class SequenceProvider(Provider):
        def sample(self, n, seq_length=None):
            assert n == 8 and seq_length == 4
            return copy.deepcopy(self.batch)
    provider = SequenceProvider(batch, expert=True)
    headings = torch.tensor([[0., 1.]]).repeat(4,1)
    state = agent.advance_rollout_context(RolloutContextState(), torch.zeros(4,1,dtype=torch.long),
                                         {'expert_slicer':provider}, current_heading_xy=headings)
    ids = state.expert_env_ids
    torch.testing.assert_close(state.velocity_world_xy[ids], torch.tensor([[0.,0.], [0.,4.]]))
    nxt, valid = agent.next_rollout_velocity_target(state, torch.zeros(4,1,dtype=torch.long))
    torch.testing.assert_close(nxt[ids], torch.tensor([[0.,1.], [0.,5.]]))
    assert valid.sum() == 2
    state = agent.advance_rollout_context(state, torch.ones(4,1,dtype=torch.long),
                                         {'expert_slicer':provider}, current_heading_xy=headings)
    torch.testing.assert_close(state.velocity_world_xy[ids], nxt[ids])
    assert state.velocity_valid.sum() == 2


def test_qv_checkpoint_roundtrip(tmp_path):
    agent, dims = make_agent()
    batch = make_batch(agent, dims)
    agent.update({'train':Provider(batch), 'expert_slicer':Provider(batch, expert=True)}, 1024)
    agent.save(tmp_path)
    loaded = type(agent).load(tmp_path, device='cpu')
    assert loaded.cfg.model.velocity_critic_enabled
    for a,b in zip(agent._model._velocity_critic.parameters(), loaded._model._velocity_critic.parameters()):
        torch.testing.assert_close(a,b)
    assert loaded.velocity_critic_optimizer.state_dict()['state']


def test_collector_does_not_reward_auto_reset_velocity():
    from types import SimpleNamespace
    from pbfm.training.workspace import Workspace
    from pbfm.agents.fb_cpr_aux.agent import RolloutContextState
    agent, _ = make_agent()
    workspace = Workspace.__new__(Workspace)
    workspace.cfg = SimpleNamespace(agent=agent.cfg)
    workspace.agent = agent
    heading = torch.tensor([[1.,0.], [1.,0.]])
    state = RolloutContextState(z=agent._model.sample_z(2), heading_target_xy=heading,
        heading_valid=torch.ones(2,1,dtype=torch.bool), context_id=torch.ones(2,1,dtype=torch.long),
        source_type=torch.ones(2,1,dtype=torch.long), velocity_world_xy=torch.tensor([[.8,0.],[.8,0.]]),
        velocity_valid=torch.ones(2,1,dtype=torch.bool))
    root = torch.zeros(2,13)
    root[:,7] = torch.tensor([.8, 99.])
    core = SimpleNamespace(base_quat=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1), robot_root_states=root)
    metadata = workspace._heading_transition_metadata(vector_env=SimpleNamespace(_env=core),
        state=state, step_count=torch.zeros(2,1,dtype=torch.long), current_heading_xy=heading,
        new_td={'time': np.array([[1],[0]])}, new_terminated=np.array([False,True]), new_truncated=np.array([False,False]))
    assert metadata['velocity_valid'].flatten().tolist() == [True, False]
    assert metadata['velocity_reward'].eq(0).all()
