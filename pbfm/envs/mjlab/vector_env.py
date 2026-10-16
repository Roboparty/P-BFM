"""Gymnasium vector interface for PBFM observations, actions, and transition metadata."""

from typing import Any, Dict, Union

import gymnasium
import numpy as np
import torch
from gymnasium import Env
from gymnasium.vector import VectorEnv

from pbfm.utils.torch_utils import xyzw_to_wxyz

from .core import HumanoidVerseMjlabCore


class HumanoidVerseMjlabVectorEnv(VectorEnv):
    """Gymnasium VectorEnv wrapper matching HumanoidVerseIsaacVectorEnv."""

    def __init__(
        self,
        env: HumanoidVerseMjlabCore,
        *,
        add_time_aware_observation: bool = True,
        include_last_action: bool = True,
        context_length: int | None = None,
        include_history_actor: bool = True,
        include_history_noaction: bool = False,
    ):
        super().__init__()
        self._env = env
        self.spec = None
        self.num_envs = env.num_envs
        self.add_time_aware_observation = add_time_aware_observation
        self.include_last_action = include_last_action
        self.context_length = context_length
        self.include_history_actor = include_history_actor
        self.include_history_noaction = include_history_noaction
        self.history_handler = None

        self.single_action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(env.num_dof,), dtype=np.float32)
        action_space_shape = (self.num_envs,) + self.single_action_space.shape
        self.action_space = gymnasium.spaces.Box(
            low=np.tile(self.single_action_space.low, (self.num_envs, 1)),
            high=np.tile(self.single_action_space.high, (self.num_envs, 1)),
            shape=action_space_shape,
            dtype=np.float32,
        )
        example_observation, _ = self.reset()
        observation_spaces = {}
        for key, value in example_observation.items():
            if np.issubdtype(value.dtype, np.unsignedinteger):
                bounds = np.iinfo(value.dtype)
                low, high = bounds.min, bounds.max
            else:
                low, high = -float("inf"), float("inf")
            observation_spaces[key] = gymnasium.spaces.Box(low=low, high=high, shape=value.shape, dtype=value.dtype)
        self.observation_space = gymnasium.spaces.Dict(observation_spaces)

    @property
    def single_observation_space(self):
        single_obs_spaces = {}
        for key, space in self.observation_space.spaces.items():
            single_obs_spaces[key] = gymnasium.spaces.Box(low=space.low[0], high=space.high[0], shape=space.shape[1:], dtype=space.dtype)
        return gymnasium.spaces.Dict(single_obs_spaces)

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> Env:
        return self._env

    @property
    def unwrapped(self):
        return self.base_env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        to_numpy: bool = True,
        reset_to_default_pose: bool = False,
        target_states: dict[str, torch.Tensor] | None = None,
    ):
        del seed, options, reset_to_default_pose
        self.base_env.reset_all(target_states=target_states)
        observation = self.base_env.get_observation(
            to_numpy=to_numpy,
            include_last_action=self.include_last_action,
            include_history_actor=self.include_history_actor,
        )
        qpos, qvel = self._get_qpos_qvel(to_numpy=to_numpy)
        return observation, {"qpos": qpos, "qvel": qvel}

    def _get_qpos_qvel(self, to_numpy: bool = True):
        base_pos_wxyz = torch.cat([self._env.robot_root_states[:, :3], xyzw_to_wxyz(self._env.robot_root_states[:, 3:7])], dim=-1)
        qpos = torch.cat([base_pos_wxyz, self._env.dof_pos], dim=-1)
        qvel = torch.cat([self._env.robot_root_states[:, 7:10], self._env.base_ang_vel, self._env.dof_vel], dim=-1)
        if to_numpy:
            return qpos.detach().cpu().numpy(), qvel.detach().cpu().numpy()
        return qpos, qvel

    def step(self, actions: Union[torch.Tensor, np.ndarray, Dict], to_numpy: bool = True):
        if isinstance(actions, dict):
            actions = actions["actions"]
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, device=self._env.device, dtype=torch.float32)
        _, reward, reset, new_info = self.base_env.step(actions)
        time_outs = new_info["time_outs"].bool()
        terminated = torch.logical_and(reset.bool(), ~time_outs)
        truncated = time_outs
        observation = self.base_env.get_observation(
            to_numpy=to_numpy,
            include_last_action=self.include_last_action,
            include_history_actor=self.include_history_actor,
        )
        qpos, qvel = self._get_qpos_qvel(to_numpy=to_numpy)
        new_info["qpos"] = qpos
        new_info["qvel"] = qvel
        if to_numpy:
            reward = reward.detach().cpu().numpy()
            terminated = terminated.detach().cpu().numpy()
            truncated = truncated.detach().cpu().numpy()
            new_info["aux_rewards"] = {k: v.detach().cpu().numpy() for k, v in new_info["aux_rewards"].items()}
        return observation, reward, terminated, truncated, new_info

    def close(self):
        return self.base_env.close()

    def render(self):
        return self.base_env.mjlab_env.render()
