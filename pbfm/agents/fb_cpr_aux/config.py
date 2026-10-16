# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Complete FB-CPR-Aux configuration, including inherited checkpoint defaults."""

import typing as tp
from typing import TYPE_CHECKING, Literal

import pydantic

from pbfm.agents.base import BaseConfig
from pbfm.agents.base_model import BaseModelConfig
from pbfm.agents.nn_filter_models import (
    ActorFilterArchiConfig,
    BackwardFilterArchiConfig,
    DiscriminatorFilterArchiConfig,
    ForwardFilterArchiConfig,
    ResidualActorFilterArchiConfig,
    SimpleActorFilterArchiConfig,
)
from pbfm.agents.nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    DirectDepthActorArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    ResidualActorArchiConfig,
    RewardNormalizerConfig,
    SimpleActorArchiConfig,
)
from pbfm.agents.normalizers import ObsNormalizerConfig

if TYPE_CHECKING:
    from .agent import FBcprAuxAgent
    from .model import FBcprAuxModel


class FBcprAuxModelArchiConfig(BaseConfig):
    z_dim: int = 100
    norm_z: bool = True
    f: ForwardArchiConfig | ForwardFilterArchiConfig = pydantic.Field(ForwardArchiConfig(), discriminator="name")
    b: BackwardArchiConfig | BackwardFilterArchiConfig = pydantic.Field(BackwardArchiConfig(), discriminator="name")
    actor: (
        ActorArchiConfig
        | ActorFilterArchiConfig
        | SimpleActorArchiConfig
        | ResidualActorArchiConfig
        | DirectDepthActorArchiConfig
        | SimpleActorFilterArchiConfig
        | ResidualActorFilterArchiConfig
    ) = pydantic.Field(SimpleActorArchiConfig(), discriminator="name")
    critic: ForwardArchiConfig | ForwardFilterArchiConfig = pydantic.Field(ForwardArchiConfig(), discriminator="name")
    discriminator: DiscriminatorArchiConfig | DiscriminatorFilterArchiConfig = pydantic.Field(
        DiscriminatorArchiConfig(), discriminator="name"
    )
    aux_critic: ForwardArchiConfig = ForwardArchiConfig()
    heading_critic: ForwardArchiConfig = ForwardArchiConfig()
    velocity_critic: ForwardArchiConfig = ForwardArchiConfig()


class FBcprAuxModelConfig(BaseModelConfig):
    name: tp.Literal["FBcprAuxModel"] = "FBcprAuxModel"
    archi: FBcprAuxModelArchiConfig = FBcprAuxModelArchiConfig()
    obs_normalizer: ObsNormalizerConfig = ObsNormalizerConfig()
    inference_batch_size: int = 500_000
    seq_length: int = 1
    actor_std: float = 0.2
    amp: bool = False
    norm_aux_reward: RewardNormalizerConfig = RewardNormalizerConfig()
    heading_context_enabled: bool = False
    heading_critic_enabled: bool = False
    velocity_critic_enabled: bool = False
    reference_velocity_policy: tp.Literal["walk_run", "all_reference"] = "walk_run"
    reference_velocity_temporal_mode: tp.Literal["instantaneous", "future_mean"] = "instantaneous"
    reference_velocity_temporal_horizon: int = pydantic.Field(default=1, ge=1)

    def build(self, obs_space, action_dim) -> "FBcprAuxModel":
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        from .model import FBcprAuxModel

        return FBcprAuxModel


class FBcprAuxAgentTrainConfig(BaseConfig):
    lr_f: float = 1e-4
    lr_b: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    fb_target_tau: float = 0.01
    ortho_coef: float = 1.0
    train_goal_ratio: float = 0.5
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    stddev_clip: float = 0.3
    q_loss_coef: float = 0.0
    batch_size: int = 1024
    discount: float = 0.99
    use_mix_rollout: bool = False
    update_z_every_step: int = 150
    z_buffer_size: int = 10000
    rollout_expert_trajectories: bool = False
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.25
    behavior_prior_enabled: bool = True
    lr_discriminator: float = 1e-4
    lr_critic: float = 1e-4
    critic_target_tau: float = 0.005
    critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 1
    scale_reg: bool = True
    # Contexts mix goal, expert-trajectory and remaining uniform samples.
    expert_asm_ratio: float = 0
    # Fraction of each minibatch relabeled with a sampled context.
    relabel_ratio: float | None = 1
    grad_penalty_discriminator: float = 10.0
    discriminator_loss: Literal["bce", "lsgan"] = "bce"
    discriminator_reward: Literal["log_odds", "amp"] = "log_odds"
    weight_decay_discriminator: float = 0.0
    lr_aux_critic: float = 1e-4
    reg_coeff_aux: float = 1.0
    aux_critic_pessimism_penalty: float = 0.5
    # Zero retains the continuously adapting reward normalizer.
    aux_reward_normalizer_freeze_global_steps: int = 0
    lr_heading_critic: float = 1e-4
    reg_coeff_heading: float = 0.0
    heading_critic_pessimism_penalty: float = 0.5
    # Disabled by default; audits the sampled/relabelled actor graph.
    actor_gradient_monitor_every: int = pydantic.Field(default=0, ge=0)
    lr_velocity_critic: float = 3e-4
    reg_coeff_velocity: float = 0.002
    velocity_error_scale_mps: float = pydantic.Field(default=0.5, gt=0)
    velocity_estimator_loss_coef: float = pydantic.Field(default=1.0, gt=0)


class FBcprAuxAgentConfig(BaseConfig):
    name: tp.Literal["FBcprAuxAgent"] = "FBcprAuxAgent"
    model: FBcprAuxModelConfig = FBcprAuxModelConfig()
    train: FBcprAuxAgentTrainConfig = FBcprAuxAgentTrainConfig()
    aux_rewards: list[str] = pydantic.Field(default_factory=list)
    aux_rewards_scaling: dict[str, float] = pydantic.Field(default_factory=dict)
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim: int) -> "FBcprAuxAgent":
        return self.object_class(
            obs_space=obs_space,
            action_dim=action_dim,
            cfg=self,
        )

    @property
    def object_class(self):
        from .agent import FBcprAuxAgent

        return FBcprAuxAgent
