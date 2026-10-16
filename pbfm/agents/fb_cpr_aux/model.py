# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from pbfm.agents.base_model import BaseModel
from pbfm.agents.nn_models import eval_mode
from pbfm.agents.pytree_utils import tree_get_batch_size

from .config import FBcprAuxModelConfig


class FBcprAuxModel(BaseModel):
    """Networks, target copies and inference for the complete PBFM algorithm."""

    config_class = FBcprAuxModelConfig

    def __init__(self, obs_space, action_dim: int, cfg: FBcprAuxModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg: FBcprAuxModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device
        self.amp_dtype = torch.bfloat16

        # create networks
        self._backward_map = arch.b.build(obs_space, arch.z_dim)
        self._forward_map = arch.f.build(obs_space, arch.z_dim, action_dim)
        self._actor = arch.actor.build(obs_space, arch.z_dim, action_dim)
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)

        # Retain the original construction boundaries: move each completed
        # network group before constructing the next group.
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

        self._discriminator = cfg.archi.discriminator.build(obs_space, cfg.archi.z_dim)
        self._critic = cfg.archi.critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

        self._aux_critic = cfg.archi.aux_critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        if cfg.heading_critic_enabled:
            self._heading_critic = cfg.archi.heading_critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        self._aux_reward_normalizer = cfg.norm_aux_reward.build()
        if cfg.velocity_critic_enabled:
            if not cfg.heading_critic_enabled or not cfg.heading_context_enabled:
                raise ValueError("QV v1 requires the collection-time heading branch")
            self._velocity_critic = cfg.archi.velocity_critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_critic = copy.deepcopy(self._critic)
        self._target_aux_critic = copy.deepcopy(self._aux_critic)
        if self.cfg.velocity_critic_enabled:
            self._target_velocity_critic = copy.deepcopy(self._velocity_critic)
        if self.cfg.heading_critic_enabled:
            self._target_heading_critic = copy.deepcopy(self._heading_critic)

    def _normalize(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._backward_map(self._normalize(obs))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._forward_map(self._normalize(obs), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, std: float):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._actor(self._normalize(obs), z, std)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean.float()
        return dist.sample().float()  # TODO we upcast to float32 to make sure the action can be converted to numpy later

    def reward_inference(
        self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            batch_size = tree_get_batch_size(next_obs)
            num_batches = int(np.ceil(batch_size / self.cfg.inference_batch_size))
            z = 0
            wr = reward if weight is None else reward * weight
            for i in range(num_batches):
                start_idx, end_idx = i * self.cfg.inference_batch_size, (i + 1) * self.cfg.inference_batch_size
                next_obs_slice = tree_map(lambda x: x[start_idx:end_idx].to(self.device), next_obs)
                B = self.backward_map(next_obs_slice)
                z += torch.matmul(wr[start_idx:end_idx].to(self.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def tracking_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)

    @torch.no_grad()
    def critic(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._critic(self._normalize(obs), z, action)

    @torch.no_grad()
    def discriminator(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._discriminator(self._normalize(obs), z)

    @torch.no_grad()
    def estimate_velocity(self, obs: dict[str, torch.Tensor]):
        """Deployable current velocity in yaw frame, metres/sec (no target)."""
        obs = self._normalize(obs)
        depth = self._actor.depth_encoder(obs[self._actor.cfg.depth_key])
        return self._actor.estimate_velocity_from_depth_latent(obs, depth)

    @torch.no_grad()
    def aux_critic(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._aux_critic(self._normalize(obs), z, action)

    @torch.no_grad()
    def heading_critic(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        if not self.cfg.heading_critic_enabled:
            raise RuntimeError("Heading critic is disabled in this checkpoint")
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._heading_critic(self._normalize(obs), z, action)
