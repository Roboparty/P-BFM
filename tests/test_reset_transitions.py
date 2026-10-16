"""Real successors, reset observations and compressed depth must never be interchanged."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from pbfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim, get_idxs
from pbfm.envs.mjlab.config import HumanoidVerseMjlabConfig
from pbfm.envs.mjlab.core import HumanoidVerseMjlabCore
from pbfm.envs.mjlab.history import HistoryHandler
from pbfm.envs.mjlab.vector_env import HumanoidVerseMjlabVectorEnv
from pbfm.training.workspace import transition_successor


def test_real_mjlab_timeout_preserves_final_state_depth_and_history(tmp_path):
    import os
    if os.environ.get("PBFM_TEST_RESET_GPU") != "1" or not torch.cuda.is_available():
        pytest.skip("requires a free GPU and PBFM_TEST_RESET_GPU=1")
    from pbfm.train import build_ufo_mjlab_config
    cfg = build_ufo_mjlab_config(
        device="cuda:0", work_dir=str(tmp_path), num_envs=4, num_env_steps=16,
        seed=4831, disable_obs_noise=True, terrain_mode="rp1_simple",
        data_path=os.environ["PBFM_TEST_RESET_DATA"],
    )
    env, _ = cfg.env.build(num_envs=4)
    try:
        env.reset(to_numpy=False)
        core = env.base_env
        history = {key: value.clone() for key, value in core.history_handler.history.items()}
        captured = {}
        reset = core.reset_finished

        def capture(ids):
            captured["qpos"], captured["qvel"] = env._get_qpos_qvel(to_numpy=False)
            captured["depth"] = core._direct_depth_runtime.observation().clone()
            reset(ids)

        core.reset_finished = capture
        core.mjlab_env.episode_length_buf.zero_()
        core.mjlab_env.episode_length_buf[0] = 100000
        obs, _, terminated, truncated, info = env.step(torch.zeros_like(core.actions), to_numpy=False)
        assert truncated.tolist() == [True, False, False, False]
        assert not terminated.any()
        assert info["final_env_ids"].tolist() == [0]
        for key in ("qpos", "qvel"):
            torch.testing.assert_close(info[f"final_{key}"], captured[key][[0]])
        torch.testing.assert_close(info["final_observation"]["depth_image"], captured["depth"][[0]])
        for key, value in core.history_handler.history.items():
            torch.testing.assert_close(value[1:, 1:], history[key][1:, :-1])
            assert not value[0, 1:].any()
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert info["final_observation"]["time"].item() > 0 and obs["time"][0].item() == 0
        successor = transition_successor(obs, info)
        torch.testing.assert_close(successor["qvel"][[0]], captured["qvel"][[0]])
        env.step(torch.zeros_like(core.actions), to_numpy=False)
        for key in ("qpos", "qvel"):
            torch.testing.assert_close(info[f"final_{key}"], captured[key][[0]])
    finally:
        env.close()


def make_env(true_termination=False):
    core = HumanoidVerseMjlabCore.__new__(HumanoidVerseMjlabCore)
    core.config = OmegaConf.create({"obs": {
        "obs_scales": {}, "noise_scales": {}, "obs_auxiliary": {"history_actor": {"dof_pos": 2}},
    }})
    core._creation_config = SimpleNamespace(
        evaluation_fast_path=True, velocity_context_enabled=True, include_velocity_estimator_target=True,
    )
    core.history_handler = HistoryHandler(2, core.config.obs.obs_auxiliary, {"dof_pos": 1}, "cpu")
    core.is_evaluating, core.add_noise_currculum = True, False
    core.num_envs, core.device = 2, "cpu"
    core.actions = torch.zeros(2, 1)
    core.dof_pos = torch.tensor([[10.0], [20.0]])
    core.default_dof_pos = core.default_dof_pos_offset = core.dof_vel = torch.zeros(2, 1)
    core.base_ang_vel = core.projected_gravity = torch.zeros(2, 3)
    core.robot_root_states = torch.zeros(2, 13)
    core.robot_root_states[:, 6] = 1
    core.base_quat = core.robot_root_states[:, 3:7]
    core.episode_length_buf = torch.zeros(2, dtype=torch.long)
    core.mjlab_env = SimpleNamespace(close=lambda: None)
    core.terrain_enabled = core.direct_depth_enabled = True
    frames = torch.zeros(2, 8, 2, 2, dtype=torch.uint8)
    core._direct_depth_runtime = SimpleNamespace(observation=lambda: frames)
    finished = torch.tensor([True, False])

    def advance(action):
        core.dof_pos.add_(1)
        core.actions.copy_(action)
        core.episode_length_buf.add_(1)
        core.robot_root_states[:, 7] = 0.8
        frames.add_(1)
        return None, torch.tensor([3.0, 4.0]), finished.clone(), {
            "time_outs": finished.clone() & (not true_termination),
            "aux_rewards": {"term": torch.tensor([5.0, 6.0])},
        }

    def reset(ids):
        core.dof_pos[ids] = 100
        core.actions[ids] = 0
        core.episode_length_buf[ids] = 0
        core.robot_root_states[ids, 7] = -3
        frames[ids] = 200
        core.history_handler.reset(ids)
        finished.zero_()

    core.step, core.reset_finished = advance, reset
    env = HumanoidVerseMjlabVectorEnv.__new__(HumanoidVerseMjlabVectorEnv)
    env._env = core
    env.include_last_action = env.include_history_actor = True
    return env


@pytest.mark.parametrize("numpy", [False, True])
@pytest.mark.parametrize("true_termination", [False, True])
def test_partial_reset_keeps_final_state_and_other_lane_history(numpy, true_termination):
    env = make_env(true_termination)
    env.base_env.get_observation(to_numpy=False)
    obs, _, terminated, truncated, info = env.step(torch.full((2, 1), 7.0), to_numpy=numpy)
    assert obs["state"][:, 0].tolist() == [100, 21]
    assert info["final_observation"]["state"][:, 0].tolist() == [11]
    assert info["final_observation"]["time"].tolist() == [[1]]
    assert obs["time"].tolist() == [[0], [1]]
    assert info["final_observation"]["last_action"].tolist() == [[7]]
    assert info["final_observation"]["history_actor"].tolist() == [[10, 0]]
    assert obs["history_actor"].tolist() == [[0, 0], [20, 0]]
    assert terminated.tolist() == [true_termination, False]
    assert truncated.tolist() == [not true_termination, False]
    assert np.all(np.asarray(info["final_observation"]["depth_image"]) == 1)
    assert np.all(np.asarray(obs["depth_image"])[0] == 200)
    successor = transition_successor(obs, info)
    assert successor["observation"]["state"][:, 0].tolist() == [11, 21]
    torch.testing.assert_close(successor["observation"]["velocity_target_yaw"][:, 0], torch.full((2,), 0.8))
    torch.testing.assert_close(successor["qvel"][:, 0], torch.full((2,), 0.8))
    following, _, _, _, _ = env.step(torch.ones(2, 1), to_numpy=False)
    assert following["history_actor"].tolist() == [[100, 0], [21, 20]]
    assert successor["observation"]["state"][:, 0].tolist() == [11, 21]
    assert np.all(np.asarray(info["final_observation"]["depth_image"]) == 1)


@pytest.mark.parametrize("true_termination", [False, True])
def test_real_vector_output_to_replay_keeps_final_and_first_reset_actions(true_termination):
    env = make_env(true_termination)
    obs = env.base_env.get_observation(to_numpy=False)
    replay = TrajectoryDictBufferMultiDim(
        capacity=5, n_dim=2, explicit_next=True, output_key_t=["observation", "action"],
        output_key_tp1=["observation", "terminated", "truncated"],
    )
    replay._get_idxs = get_idxs
    for _ in range(2):
        current = {key: value.clone()[None] for key, value in obs.items() if key != "time"}
        action = torch.full((2, 1), 7.0)
        obs, _, terminated, truncated, info = env.step(action, to_numpy=False)
        successor = transition_successor(obs, info)
        replay.extend({
            "observation": current, "action": action[None], "done": (terminated | truncated)[None, :, None],
            "next": {"observation": {key: value[None] for key, value in successor["observation"].items()},
                     "terminated": terminated[None, :, None], "truncated": truncated[None, :, None]},
        })
    for batch in (replay.get_full_buffer(), replay.sample(4096)):
        assert set(zip(batch["observation"]["state"][:, 0].tolist(),
                       batch["next"]["observation"]["state"][:, 0].tolist())) == {
            (10, 11), (100, 101), (20, 21), (21, 22),
        }
        final = batch["observation"]["state"][:, 0] == 10
        assert (batch["next"]["terminated"][final] == true_termination).all()
        assert (batch["next"]["truncated"][final] == (not true_termination)).all()


@pytest.mark.parametrize("n_dim,lanes", [(1, 1), (2, 1), (2, 3)])
def test_explicit_successors_wrap_sequences_and_save(tmp_path, n_dim, lanes):
    replay = TrajectoryDictBufferMultiDim(
        capacity=5, n_dim=n_dim, explicit_next=True, output_key_tp1=["observation", "terminated", "truncated"],
    )
    replay._get_idxs = get_idxs
    for step in range(8):
        value = torch.tensor([step * 10 + lane for lane in range(lanes)]).reshape(1, lanes, 1)
        flags = torch.full_like(value, step % 3 == 0, dtype=torch.bool)
        row = {"observation": value, "done": flags,
               "next": {"observation": value + 1, "terminated": flags, "truncated": torch.zeros_like(flags)}}
        if n_dim == 1:
            from torch.utils._pytree import tree_map
            row = tree_map(lambda value: value[:, 0], row)
        replay.extend(row)
    full = replay.get_full_buffer()
    assert set(full["observation"].flatten().tolist()) == {step * 10 + lane for step in range(3, 8) for lane in range(lanes)}
    assert torch.equal(full["next"]["observation"], full["observation"] + 1)
    sampled = replay.sample(4096)
    assert set(sampled["observation"].flatten().tolist()) == set(full["observation"].flatten().tolist())
    sequence = replay.sample(4096, seq_length=2)
    assert ((sequence["observation"].reshape(-1, 2)[:, 1] - sequence["observation"].reshape(-1, 2)[:, 0]) == 10).all()
    assert not sequence["next"]["terminated"].reshape(-1, 2)[:, 0].any()
    replay.save(tmp_path)
    restored = TrajectoryDictBufferMultiDim.load_for_training(tmp_path)
    assert torch.equal(restored.get_full_buffer()["next"]["observation"], full["next"]["observation"])


@pytest.mark.parametrize("cut_collection", [False, True])
def test_compact_successor_depth_stops_at_reset_and_ring_cursor(tmp_path, cut_collection):
    replay = TrajectoryDictBufferMultiDim(
        capacity=7, n_dim=2, explicit_next=True, compact_depth_history=True, depth_history_offsets=(3, 1, 0),
        output_key_tp1=["observation", "terminated", "truncated"],
    )
    replay._get_idxs = get_idxs
    starts = [0, 0]
    for step in range(11):
        # Deliberately make successor frames differ from reset frames.
        start_values = [100 * lane + step for lane in range(2)]
        depth = torch.tensor(start_values, dtype=torch.uint8).reshape(1, 2, 1, 1, 1).expand(1, 2, 3, 2, 2)
        done = torch.tensor([step % 3 == 2, step % 5 == 4]).reshape(1, 2, 1)
        replay.extend({"observation": {"state": torch.tensor(start_values).reshape(1, 2, 1), "depth_image": depth},
                       "done": done, "truncated": done, "transition_truncated": done,
                       "heading_context_continues": ~done, "next": {
                           "observation": {"state": torch.tensor(start_values).reshape(1, 2, 1) + 50, "depth_image": depth + 50},
                           "terminated": torch.zeros_like(done), "truncated": done}})
        if cut_collection and step == 7:
            replay.end_episodes()
            for key in ("done", "truncated", "transition_truncated"):
                assert replay.storage[key][0].all()
            assert replay.storage["next"]["truncated"][0].all()
            assert not replay.storage["heading_context_continues"][0].any()
        if step >= 4:
            indices = torch.tensor([[(step % 7), lane] for lane in range(2)])
            batch = replay._gather_transitions(indices)
            for lane in range(2):
                oldest = max(starts[lane], step - 6)
                expected = [100 * lane + max(step - offset, oldest) for offset in (3, 1, 0)]
                expected_next = [100 * lane + max(step + 1 - offset, oldest) for offset in (3, 1)] + [start_values[lane] + 50]
                assert batch["observation"]["depth_image"][lane, :, 0, 0].tolist() == expected
                assert batch["next"]["observation"]["depth_image"][lane, :, 0, 0].tolist() == expected_next
        for lane in range(2):
            if done[0, lane, 0] or (cut_collection and step == 7):
                starts[lane] = step + 1
    assert replay.storage["observation"]["depth_image"].shape == (7, 2, 2, 2)
    assert replay.storage["next"]["observation"]["depth_image"].shape == (7, 2, 2, 2)
    replay.save(tmp_path)
    restored = TrajectoryDictBufferMultiDim.load_for_training(tmp_path)
    torch.testing.assert_close(restored.get_full_buffer()["next"]["observation"]["depth_image"],
                               replay.get_full_buffer()["next"]["observation"]["depth_image"])


def test_sampler_last_start_and_legacy_training_guard(tmp_path):
    kwargs = dict(seq_length=1, num_slices=1, lengths=torch.tensor([4]), start_idx=torch.tensor([[0]]), storage_length=8)
    with patch("torch.rand", return_value=torch.tensor([0.999])):
        assert get_idxs(**kwargs, offset=1).item() == 2
        assert get_idxs(**kwargs, offset=0).item() == 3
    replay = TrajectoryDictBufferMultiDim(capacity=3, n_dim=2)
    replay.extend({"observation": torch.zeros(2, 1, 1), "done": torch.zeros(2, 1, 1, dtype=torch.bool)})
    replay.save(tmp_path)
    with pytest.raises(ValueError, match="Legacy replay"):
        TrajectoryDictBufferMultiDim.load_for_training(tmp_path)
    # Archives remain readable for historical inference, with no implicit migration/deletion.
    assert not TrajectoryDictBufferMultiDim.load(tmp_path).explicit_next
    with pytest.raises(ValueError, match="owns task reset"):
        HumanoidVerseMjlabConfig(lafan_tail_path="unused", device="cpu", auto_reset=True).build()


@pytest.mark.parametrize("terrain_boundary", [False, True])
def test_core_does_not_reset_before_vector_captures_successor(terrain_boundary):
    core = make_env().base_env
    core._creation_config.evaluation_fast_path = False
    core.terrain_aware_auxiliary = False
    core.terrain_enabled = terrain_boundary
    core._direct_depth_runtime = None
    core.last_actions = core.actions.clone()
    core.last_dof_vel = core.dof_vel.clone()
    core.applied_actions = core.actions.clone()
    core.extras = {}
    core.use_reward_penalty_curriculum = False
    core._normalized_action = core._apply_ctrl_delay = core._mjlab_action_input = lambda value: value
    core._refresh_state = lambda: None
    core.simulator = SimpleNamespace(refresh=lambda: None)
    core._update_terrain_tile_transitions = lambda: None
    core._check_terrain_boundary = lambda: torch.tensor([True, False])
    core.reset_idx = lambda *args, **kwargs: pytest.fail("reset ran before successor capture")
    def advance(action):
        core.dof_pos.add_(1)
        return None, None, torch.tensor([not terrain_boundary, False]), torch.zeros(2, dtype=torch.bool), {}
    core.mjlab_env.step = advance
    from pbfm.envs.mjlab import aux_rewards
    with patch.object(aux_rewards, "_compute_reward", return_value=(torch.tensor([-1., -2.]), {"impact": torch.tensor([2., 3.])})):
        _, reward, reset, info = HumanoidVerseMjlabCore.step(core, torch.ones(2, 1))
    assert core.dof_pos[:, 0].tolist() == [11, 21]
    assert reset.tolist() == [True, False]
    assert info["time_outs"].tolist() == [terrain_boundary, False]
    core.time_out_buf.zero_()
    assert info["time_outs"].tolist() == [terrain_boundary, False]
    assert reward.tolist() == [-1, -2]


def test_workspace_training_loop_collects_explicit_successors(tmp_path):
    import json
    from pathlib import Path
    from pbfm.training.config import TrainConfig
    from pbfm.training.workspace import Workspace
    from pbfm.agents.fb_cpr_aux.agent import RolloutContextState
    cfg = TrainConfig.model_validate(json.loads(
        (Path(__file__).resolve().parents[1] / "pbfm/configs/legacy_large.json").read_text(),
    ))
    cfg = cfg.model_copy(update=dict(
        online_parallel_envs=2, buffer_size=16, buffer_device="cpu", num_env_steps=4, num_seed_steps=100,
        checkpoint_every_steps=1000, distributed_world_size=1, distributed_sync=False,
        fail_on_nonfinite=False, disable_tqdm=True, resume_replay_warmup_steps=0,
    ))
    cfg = cfg.model_copy(update={"agent": cfg.agent.model_copy(update={
        "model": cfg.agent.model.model_copy(update={"heading_context_enabled": False, "velocity_critic_enabled": False}),
        "aux_rewards": ["term"],
    })})
    env = make_env()
    env.action_space = SimpleNamespace(sample=lambda: np.zeros((2, 1), dtype=np.float32))
    def initial_observation():
        qpos, qvel = env._get_qpos_qvel()
        return env.base_env.get_observation(), {"qpos": qpos, "qvel": qvel}
    env.reset = initial_observation
    workspace = Workspace.__new__(Workspace)
    workspace.cfg, workspace.work_dir = cfg, tmp_path
    workspace.train_env, workspace.train_env_info, workspace.prior_env = env, {}, None
    workspace.agent = SimpleNamespace(device="cpu")
    workspace._advance_rollout_state = lambda *args, **kwargs: RolloutContextState(z=torch.ones(2, 4))
    workspace._load_expert_buffer = lambda: None
    workspace.training_with_expert_data = False
    workspace._write_shared_artifacts = False
    workspace.distributed_rank, workspace.distributed_world_size = 0, 1
    workspace._checkpoint_local_time = workspace._checkpoint_global_time = workspace._optimizer_steps = 0
    workspace.evaluate, workspace.evaluations = False, {}
    captured = {}
    workspace.save = lambda **kwargs: captured.update(kwargs)
    workspace.train_online()
    replay = captured["replay_buffer"]["train"]
    replay._get_idxs = get_idxs
    assert replay.explicit_next and replay.end_key == "done"
    assert replay.storage["next"]["observation"]["depth_image"].shape == (8, 2, 2, 2)
    batch = replay.sample(4096)
    assert set(zip(batch["observation"]["state"][:, 0].tolist(),
                   batch["next"]["observation"]["state"][:, 0].tolist())) == {
        (10, 11), (100, 101), (20, 21), (21, 22),
    }


def test_heading_metadata_uses_final_yaw_without_changing_hv_masks():
    from test_qv_agent_integration import make_agent
    from pbfm.training.workspace import Workspace
    from pbfm.agents.fb_cpr_aux.agent import RolloutContextState
    agent, _ = make_agent()
    workspace = Workspace.__new__(Workspace)
    workspace.cfg, workspace.agent = SimpleNamespace(agent=agent.cfg), agent
    heading = torch.tensor([[1., 0.], [1., 0.]])
    state = RolloutContextState(
        z=agent._model.sample_z(2), heading_target_xy=heading, heading_valid=torch.ones(2, 1, dtype=torch.bool),
        context_id=torch.ones(2, 1, dtype=torch.long), source_type=torch.ones(2, 1, dtype=torch.long),
        velocity_world_xy=torch.tensor([[.8, 0.], [.8, 0.]]), velocity_valid=torch.ones(2, 1, dtype=torch.bool),
    )
    root = torch.zeros(2, 13)
    root[:, 7] = torch.tensor([99., .8])
    core = SimpleNamespace(base_quat=torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1), robot_root_states=root)
    # Before reset the first lane faced +Y, while its new spawn faces +X.
    final_qpos = np.array([[0., 0., 0., 2**-0.5, 0., 0., 2**-0.5]], dtype=np.float32)
    metadata = workspace._heading_transition_metadata(
        vector_env=SimpleNamespace(_env=core), state=state, step_count=torch.zeros(2, 1, dtype=torch.long),
        current_heading_xy=heading, new_td={"time": np.array([[0], [1]])},
        new_terminated=np.array([False, False]), new_truncated=np.array([True, False]),
        new_info={"final_env_ids": np.array([0]), "final_qpos": final_qpos},
    )
    torch.testing.assert_close(metadata["next_root_heading_xy"], torch.tensor([[0., 1.], [1., 0.]]), atol=1e-6, rtol=0)
    assert metadata["heading_valid"].flatten().tolist() == [False, True]
    assert metadata["velocity_valid"].flatten().tolist() == [False, True]
    assert not metadata["heading_context_continues"][0].item()
    assert metadata["velocity_reward"].eq(0).all()


def test_prior_collector_preserves_same_step_successor():
    from pbfm.training.workspace import Workspace, _PriorCollectorState
    from pbfm.agents.fb_cpr_aux.agent import RolloutContextState
    env = make_env()
    get_observation = env.base_env.get_observation
    def plane_observation(**kwargs):
        obs = get_observation(**kwargs)
        shape = (obs["state"].shape[0], 2)
        obs["terrain_priv"] = np.zeros(shape, dtype=np.float32) if kwargs.get("to_numpy", True) else torch.zeros(shape)
        return obs
    env.base_env.get_observation = plane_observation
    env.action_space = SimpleNamespace(sample=lambda: np.zeros((2, 1), dtype=np.float32))
    workspace = Workspace.__new__(Workspace)
    workspace.agent = SimpleNamespace(device="cpu")
    workspace.cfg = SimpleNamespace(agent=SimpleNamespace(model=SimpleNamespace(heading_context_enabled=False)),
                                    num_seed_steps=100, fail_on_nonfinite=False)
    workspace.prior_env = env
    workspace._advance_rollout_state = lambda *args, **kwargs: RolloutContextState(z=torch.ones(2, 4))
    state = _PriorCollectorState(td=plane_observation(), info={}, terminated=np.zeros(2, dtype=bool),
                                 truncated=np.zeros(2, dtype=bool), rollout=RolloutContextState())
    replay = TrajectoryDictBufferMultiDim(capacity=4, n_dim=2, end_key="episode_boundary", explicit_next=True,
                                          compact_depth_history=True, output_key_t=["observation", "transition_truncated"])
    for step in range(2):
        state, _ = workspace._step_prior_collector(state, replay_buffer={"prior": replay},
                                                   local_time=step, global_time=step, optimizer_steps=0)
    batch = replay.get_full_buffer()
    assert set(zip(batch["observation"]["state"][:, 0].tolist(),
                   batch["next"]["observation"]["state"][:, 0].tolist())) == {
        (10, 11), (100, 101), (20, 21), (21, 22),
    }
    final = batch["observation"]["state"][:, 0] == 10
    assert batch["transition_truncated"][final].all()
    assert batch["next"]["observation"]["depth_image"][final, -1].eq(1).all()
