# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import autograd
from torch.amp import autocast
from torch.utils._pytree import tree_map

from pbfm.agents.behavior_context import (
    HEADING_SOURCE_EXACT_TRACKING,
    HEADING_SOURCE_INVALID,
    align_heading_sequence,
    heading_observation,
    normalize_heading_xy,
    relative_heading_target,
)
from pbfm.agents.checkpoint import load_agent, save_agent
from pbfm.agents.execution import AgentExecutionMixin
from pbfm.agents.misc.zbuffer import ZBuffer
from pbfm.agents.nn_models import _soft_update_params, eval_mode, weight_init
from pbfm.agents.pytree_utils import tree_get_batch_size

from .config import FBcprAuxAgentConfig
from .model import FBcprAuxModel


@dataclass
class RolloutContextState:
    """Collector-local latent and expert-tracking rollout state.

    Training may drive more than one environment collector with the same
    agent. The temporal bookkeeping therefore belongs to the collector, not
    to the shared agent/model.
    """

    z: torch.Tensor | None = None
    heading_target_xy: torch.Tensor | None = None
    heading_valid: torch.Tensor | None = None
    context_id: torch.Tensor | None = None
    source_type: torch.Tensor | None = None
    expert_env_ids: torch.Tensor | None = None
    tracking_z: torch.Tensor | None = None
    tracking_heading_target_xy: torch.Tensor | None = None
    tracking_velocity_world_xy: torch.Tensor | None = None
    tracking_velocity_valid: torch.Tensor | None = None
    velocity_world_xy: torch.Tensor | None = None
    velocity_valid: torch.Tensor | None = None


def prior_transition_discount(
    prior_batch: dict[str, tp.Any],
    *,
    gamma: float,
    device: str | torch.device,
) -> torch.Tensor:
    """Return the behavior-prior discount aligned with the sampled action.

    Both main and canonical-plane collectors store the outcome of the action
    in the current replay slot. This keeps the historical conservative mask
    for artificial terrain-boundary resets even with explicit successors.
    The fallback is retained for archived replay that predates these fields.
    """

    if "transition_terminated" in prior_batch and "transition_truncated" in prior_batch:
        done = prior_batch["transition_terminated"].to(device) | prior_batch[
            "transition_truncated"
        ].to(device)
    else:
        done = prior_batch["next"]["terminated"].to(device)
        if "truncated" in prior_batch["next"]:
            done = done | prior_batch["next"]["truncated"].to(device)
    return float(gamma) * ~done


@dataclass
class RelabeledBehaviorContext:
    """A z and its matching deployable heading observation pair."""

    z: torch.Tensor
    heading: torch.Tensor
    next_heading: torch.Tensor
    heading_valid: torch.Tensor
    source_type: torch.Tensor
    velocity_command: torch.Tensor | None = None
    next_velocity_command: torch.Tensor | None = None


def _distribution_metrics(prefix: str, values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Summarize a diagnostic distribution without synchronizing to the CPU."""

    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        zero = torch.zeros((), device=values.device, dtype=torch.float32)
        return {
            f"{prefix}/mean": zero,
            f"{prefix}/p10": zero,
            f"{prefix}/p50": zero,
            f"{prefix}/p90": zero,
        }
    quantiles = torch.quantile(
        flat,
        torch.tensor((0.1, 0.5, 0.9), device=flat.device, dtype=flat.dtype),
    )
    return {
        f"{prefix}/mean": flat.mean(),
        f"{prefix}/p10": quantiles[0],
        f"{prefix}/p50": quantiles[1],
        f"{prefix}/p90": quantiles[2],
    }


class FBcprAuxAgent(AgentExecutionMixin):
    """Complete FB-CPR-Aux update pipeline, with no algorithm inheritance."""

    config_class = FBcprAuxAgentConfig

    def __init__(self, obs_space, action_dim, cfg: FBcprAuxAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        # make sure batch size is a multiple of seq_length
        seq_length = cfg.model.seq_length
        batch_size = cfg.train.batch_size
        assert (batch_size / seq_length) == (batch_size // seq_length), "Batch size should be divisable by seq_length"
        del seq_length, batch_size

        self._model: FBcprAuxModel = self.cfg.model.build(obs_space, action_dim)
        # This is just to be sure? I think it should not change since build
        self._model.to(self.device)
        self.setup_training()
        self.setup_compile()
        self.env_idx_with_expert_rollout = None

    @property
    def device(self):
        return self._model.device

    @classmethod
    def supported_evaluations(cls):
        return ["reward", "tracking"]

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.backward_optimizer = torch.optim.Adam(
            self._model._backward_map.parameters(),
            lr=self.cfg.train.lr_b,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            self._model._forward_map.parameters(),
            lr=self.cfg.train.lr_f,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

        # prepare parameter list
        self._forward_map_paramlist = tuple(x for x in self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

        self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self._model.device)

        # prepare parameter list
        self._critic_map_paramlist = tuple(x for x in self._model._critic.parameters())
        self._target_critic_map_paramlist = tuple(x for x in self._model._target_critic.parameters())

        self.critic_optimizer = torch.optim.Adam(
            self._model._critic.parameters(),
            lr=self.cfg.train.lr_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator.parameters(),
            lr=self.cfg.train.lr_discriminator,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay_discriminator,
        )

        if getattr(self.cfg.model.archi.actor, "estimated_velocity_enabled", False):
            if not self.cfg.model.velocity_critic_enabled:
                raise ValueError("Estimated velocity experiment requires QV")
            normalizer = self.cfg.model.obs_normalizer.normalizers.get("velocity_target_yaw")
            if normalizer is None or normalizer.name != "IdentityNormalizerConfig":
                raise ValueError("Velocity supervision must remain physical m/s (identity normalization)")

        # prepare parameter list
        self._aux_critic_map_paramlist = tuple(x for x in self._model._aux_critic.parameters())
        self._aux_target_critic_map_paramlist = tuple(x for x in self._model._target_aux_critic.parameters())

        self.aux_critic_optimizer = torch.optim.Adam(
            self._model._aux_critic.parameters(),
            lr=self.cfg.train.lr_aux_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        if self.cfg.model.velocity_critic_enabled:
            self.velocity_critic_optimizer = torch.optim.Adam(
                self._model._velocity_critic.parameters(), lr=self.cfg.train.lr_velocity_critic,
                capturable=self.cfg.cudagraphs and not self.cfg.compile,
                weight_decay=self.cfg.train.weight_decay)
        if self._heading_critic_enabled:
            self._heading_critic_map_paramlist = tuple(x for x in self._model._heading_critic.parameters())
            self._heading_target_critic_map_paramlist = tuple(
                x for x in self._model._target_heading_critic.parameters()
            )
            self.heading_critic_optimizer = torch.optim.Adam(
                self._model._heading_critic.parameters(),
                lr=self.cfg.train.lr_heading_critic,
                capturable=self.cfg.cudagraphs and not self.cfg.compile,
                weight_decay=self.cfg.train.weight_decay,
            )

    @property
    def optimizer_dict(self):
        optimizers = {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
        }
        optimizers.update(
            {
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            }
        )
        optimizers["aux_critic_optimizer"] = self.aux_critic_optimizer.state_dict()
        if self.cfg.model.velocity_critic_enabled:
            optimizers["velocity_critic_optimizer"] = self.velocity_critic_optimizer.state_dict()
        if self._heading_critic_enabled:
            optimizers["heading_critic_optimizer"] = self.heading_critic_optimizer.state_dict()
        return optimizers

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    @property
    def _behavior_prior_enabled(self) -> bool:
        return bool(getattr(self.cfg.train, "behavior_prior_enabled", True))

    @property
    def _heading_context_enabled(self) -> bool:
        return bool(
            getattr(getattr(self.cfg, "model", None), "heading_context_enabled", False)
        )

    @property
    def _heading_critic_enabled(self) -> bool:
        return bool(
            getattr(getattr(self.cfg, "model", None), "heading_critic_enabled", False)
        )

    @torch.no_grad()
    def sample_mixed_context_components(
        self,
        train_goal: torch.Tensor,
        expert_encodings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample z and retain the source/permutation needed by companion context."""

        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
            p_goal = self.cfg.train.train_goal_ratio
            p_expert_asm = self.cfg.train.expert_asm_ratio
            prob = torch.tensor(
                [p_goal, p_expert_asm, 1 - p_goal - p_expert_asm],
                dtype=torch.float32,
                device=self.device,
            )
            mix_idxs = torch.multinomial(prob, num_samples=self.cfg.train.batch_size, replacement=True).reshape(-1, 1)

            # zs obtained by encoding train goals
            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            train_goal = tree_map(lambda x: x[perm], train_goal)
            goals = self._model._backward_map(train_goal)
            goals = self._model.project_z(goals)
            z = torch.where(mix_idxs == 0, goals, z)

            # zs obtained by encoding expert trajectories
            expert_perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            z = torch.where(mix_idxs == 1, expert_encodings[expert_perm], z)

        return z, mix_idxs, expert_perm

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor, expert_encodings: torch.Tensor, *args, **kwargs):
        z, _source_type, _expert_perm = self.sample_mixed_context_components(train_goal, expert_encodings)
        return z

    @torch.no_grad()
    def encode_expert(
        self,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
    ):
        # encode expert trajectories through B
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            B_expert = self._model._backward_map(next_obs).detach()  # batch x d
            B_expert = B_expert.view(
                self.cfg.train.batch_size // self.cfg.model.seq_length,
                self.cfg.model.seq_length,
                B_expert.shape[-1],
            )  # N x L x d
            z_expert = B_expert.mean(dim=1)  # N x d
            z_expert = self._model.project_z(z_expert)
            z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)  # batch x d
        return z_expert

    @torch.no_grad()
    def _discriminator_conditioning_diagnostics(
        self,
        *,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        prior_obs: torch.Tensor | dict[str, torch.Tensor],
        prior_rollout_z: torch.Tensor,
        prior_relabel_z: torch.Tensor,
        max_samples: int = 256,
    ) -> dict[str, torch.Tensor]:
        """Compare D logits before and after prior-context relabeling.

        D is trained with ``prior_rollout_z`` while Q_D consumes
        ``prior_relabel_z``.  Keeping both evaluations on the same states and
        the same post-update discriminator separates saturation from a
        conditioning-distribution mismatch.  A small subset keeps this
        training-only diagnostic cheap.
        """

        batch_size = min(
            int(expert_z.shape[0]),
            int(prior_rollout_z.shape[0]),
            int(prior_relabel_z.shape[0]),
            int(max_samples),
        )
        expert_obs = tree_map(lambda x: x[:batch_size], expert_obs)
        prior_obs = tree_map(lambda x: x[:batch_size], prior_obs)
        expert_z = expert_z[:batch_size]
        prior_rollout_z = prior_rollout_z[:batch_size]
        prior_relabel_z = prior_relabel_z[:batch_size]

        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            expert_logits = self._model._discriminator.compute_logits(expert_obs, expert_z)
            rollout_logits = self._model._discriminator.compute_logits(prior_obs, prior_rollout_z)
            relabel_logits = self._model._discriminator.compute_logits(prior_obs, prior_relabel_z)
            rollout_rewards = self.discriminator_reward_from_logits(rollout_logits)
            relabel_rewards = self.discriminator_reward_from_logits(relabel_logits)

        metrics: dict[str, torch.Tensor] = {}
        metrics.update(_distribution_metrics("disc_diag/expert_logit", expert_logits))
        metrics.update(_distribution_metrics("disc_diag/policy_rollout_logit", rollout_logits))
        metrics.update(_distribution_metrics("disc_diag/policy_relabel_logit", relabel_logits))
        metrics.update(_distribution_metrics("disc_diag/reward_original_z", rollout_rewards))
        metrics.update(_distribution_metrics("disc_diag/reward_relabel_z", relabel_rewards))
        if getattr(
            getattr(self.cfg, "train", None), "discriminator_reward", "log_odds"
        ) == "amp":
            metrics["disc_diag/zero_reward_fraction_original"] = (
                rollout_rewards <= 0.0
            ).float().mean()
            metrics["disc_diag/zero_reward_fraction_relabel"] = (
                relabel_rewards <= 0.0
            ).float().mean()
        metrics["disc_diag/reward_original_relabel_abs_diff"] = (
            relabel_rewards - rollout_rewards
        ).abs().mean()

        relabeled = (prior_relabel_z.float() - prior_rollout_z.float()).abs().amax(dim=-1) > 1.0e-6
        logit_shift = (relabel_logits - rollout_logits).reshape(-1)
        metrics["disc_diag/relabel_fraction"] = relabeled.float().mean()
        metrics.update(
            _distribution_metrics(
                "disc_diag/relabel_logit_shift",
                logit_shift[relabeled],
            )
        )
        return metrics

    def _relabel_main_and_prior_z(
        self,
        *,
        main_next_obs: torch.Tensor | dict[str, torch.Tensor],
        prior_next_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        main_rollout_z: torch.Tensor,
        prior_rollout_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy z-only path for checkpoints without BehaviorContext inputs."""

        sampled_main_z = self.sample_mixed_z(train_goal=main_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(sampled_main_z)
        sampled_prior_z = self.sample_mixed_z(train_goal=prior_next_obs, expert_encodings=expert_z).clone()
        if self.cfg.train.relabel_ratio is None:
            return main_rollout_z, prior_rollout_z
        main_mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
        prior_mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
        return (
            torch.where(main_mask, sampled_main_z, main_rollout_z),
            torch.where(prior_mask, sampled_prior_z, prior_rollout_z),
        )

    def _sample_mixed_behavior_context(
        self,
        *,
        train_goal: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        root_heading_xy: torch.Tensor,
        next_root_heading_xy: torch.Tensor,
        expert_heading_xy: torch.Tensor,
        expert_next_heading_xy: torch.Tensor,
    ) -> RelabeledBehaviorContext:
        """Sample z together with a source-valid one-step heading context.

        Goal and random latents deliberately receive an invalid zero heading.
        Only an expert z sampled with exact expert transition metadata carries
        a relative-heading command.
        """

        sampled_z, mix_source, expert_perm = self.sample_mixed_context_components(
            train_goal,
            expert_z,
        )
        expert_selected = mix_source == 1
        target = root_heading_xy
        target_next = relative_heading_target(
            root_heading_xy,
            expert_heading_xy[expert_perm],
            expert_next_heading_xy[expert_perm],
        )
        heading = heading_observation(root_heading_xy, target, expert_selected)
        next_heading = heading_observation(next_root_heading_xy, target_next, expert_selected)
        source_type = torch.where(
            expert_selected,
            torch.full_like(mix_source, HEADING_SOURCE_EXACT_TRACKING),
            torch.full_like(mix_source, HEADING_SOURCE_INVALID),
        )
        velocity_command = next_velocity_command = None
        if bool(getattr(getattr(getattr(self, "cfg", None), "model", None), "velocity_critic_enabled", False)):
            from pbfm.agents.velocity_context import align_reference_velocity, velocity_observation
            batch = self._velocity_expert_batch
            ref_velocity = batch["reference_velocity_world_xy"].to(self.device)[expert_perm]
            ref_next_velocity = batch["next"]["reference_velocity_world_xy"].to(self.device)[expert_perm]
            vvalid = expert_selected & batch["reference_velocity_valid"].to(self.device)[expert_perm]
            next_vvalid = expert_selected & batch["next"]["reference_velocity_valid"].to(self.device)[expert_perm]
            aligned = align_reference_velocity(ref_velocity, expert_heading_xy[expert_perm], root_heading_xy)
            aligned_next = align_reference_velocity(ref_next_velocity, expert_heading_xy[expert_perm], root_heading_xy)
            velocity_command = velocity_observation(aligned, root_heading_xy, vvalid)
            next_velocity_command = velocity_observation(aligned_next, next_root_heading_xy, next_vvalid)
        return RelabeledBehaviorContext(
            z=sampled_z.clone(),
            heading=heading,
            next_heading=next_heading,
            heading_valid=expert_selected,
            source_type=source_type,
            velocity_command=velocity_command,
            next_velocity_command=next_velocity_command,
        )

    def _relabel_main_z(
        self,
        *,
        main_next_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        main_rollout_z: torch.Tensor,
    ) -> torch.Tensor:
        """Relabel only the main stream for the formal no-D configuration."""

        sampled_main_z = self.sample_mixed_z(
            train_goal=main_next_obs,
            expert_encodings=expert_z,
        ).clone()
        self.z_buffer.add(sampled_main_z)
        if self.cfg.train.relabel_ratio is None:
            return main_rollout_z
        main_mask = torch.rand(
            (main_rollout_z.shape[0], 1),
            device=self.device,
        ) <= self.cfg.train.relabel_ratio
        return torch.where(main_mask, sampled_main_z, main_rollout_z)

    def _relabel_main_context(
        self,
        *,
        main_obs: dict[str, torch.Tensor],
        main_next_obs: dict[str, torch.Tensor],
        main_batch: dict[str, tp.Any],
        expert_z: torch.Tensor,
        expert_heading_xy: torch.Tensor,
        expert_next_heading_xy: torch.Tensor,
        main_rollout_z: torch.Tensor,
    ) -> tuple[RelabeledBehaviorContext, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Atomically relabel main z/heading without sampling a dummy prior stream."""

        sampled = self._sample_mixed_behavior_context(
            train_goal=main_next_obs,
            expert_z=expert_z,
            root_heading_xy=main_batch["root_heading_xy"].to(self.device),
            next_root_heading_xy=main_batch["next_root_heading_xy"].to(self.device),
            expert_heading_xy=expert_heading_xy,
            expert_next_heading_xy=expert_next_heading_xy,
        )
        self.z_buffer.add(sampled.z)
        original = RelabeledBehaviorContext(
            z=main_rollout_z,
            heading=main_obs["heading"],
            next_heading=main_next_obs["heading"],
            heading_valid=main_batch["heading_valid"].to(self.device),
            source_type=main_batch["heading_source_type"].to(self.device),
        )
        if self.cfg.train.relabel_ratio is None:
            return original, main_obs, main_next_obs

        mask = torch.rand(
            (main_rollout_z.shape[0], 1),
            device=self.device,
        ) <= self.cfg.train.relabel_ratio
        merged = RelabeledBehaviorContext(
            z=torch.where(mask, sampled.z, original.z),
            heading=torch.where(mask, sampled.heading, original.heading),
            next_heading=torch.where(mask, sampled.next_heading, original.next_heading),
            heading_valid=torch.where(mask, sampled.heading_valid, original.heading_valid),
            source_type=torch.where(mask, sampled.source_type, original.source_type),
        )
        relabeled_obs = dict(main_obs)
        relabeled_next_obs = dict(main_next_obs)
        relabeled_obs["heading"] = merged.heading
        relabeled_next_obs["heading"] = merged.next_heading
        if self.cfg.model.velocity_critic_enabled:
            from pbfm.agents.velocity_context import merge_velocity_context
            relabeled_obs["velocity_command"] = merge_velocity_context(
                main_obs["velocity_command"], sampled.velocity_command, mask)
            relabeled_next_obs["velocity_command"] = merge_velocity_context(
                main_next_obs["velocity_command"], sampled.next_velocity_command, mask)
        return merged, relabeled_obs, relabeled_next_obs

    def _relabel_main_and_prior_context(
        self,
        *,
        main_obs: dict[str, torch.Tensor],
        main_next_obs: dict[str, torch.Tensor],
        prior_obs: dict[str, torch.Tensor],
        prior_next_obs: dict[str, torch.Tensor],
        main_batch: dict[str, tp.Any],
        prior_batch: dict[str, tp.Any],
        expert_z: torch.Tensor,
        expert_heading_xy: torch.Tensor,
        expert_next_heading_xy: torch.Tensor,
        main_rollout_z: torch.Tensor,
        prior_rollout_z: torch.Tensor,
    ) -> tuple[RelabeledBehaviorContext, RelabeledBehaviorContext, dict, dict, dict, dict]:
        """Relabel both streams atomically: z and heading always change together."""

        sampled_main = self._sample_mixed_behavior_context(
            train_goal=main_next_obs,
            expert_z=expert_z,
            root_heading_xy=main_batch["root_heading_xy"].to(self.device),
            next_root_heading_xy=main_batch["next_root_heading_xy"].to(self.device),
            expert_heading_xy=expert_heading_xy,
            expert_next_heading_xy=expert_next_heading_xy,
        )
        self.z_buffer.add(sampled_main.z)
        sampled_prior = self._sample_mixed_behavior_context(
            train_goal=prior_next_obs,
            expert_z=expert_z,
            root_heading_xy=prior_batch["root_heading_xy"].to(self.device),
            next_root_heading_xy=prior_batch["next_root_heading_xy"].to(self.device),
            expert_heading_xy=expert_heading_xy,
            expert_next_heading_xy=expert_next_heading_xy,
        )

        def original_context(batch, rollout_z, obs, next_obs):
            return RelabeledBehaviorContext(
                z=rollout_z,
                heading=obs["heading"],
                next_heading=next_obs["heading"],
                heading_valid=batch["heading_valid"].to(self.device),
                source_type=batch["heading_source_type"].to(self.device),
            )

        main_original = original_context(main_batch, main_rollout_z, main_obs, main_next_obs)
        prior_original = original_context(prior_batch, prior_rollout_z, prior_obs, prior_next_obs)

        if self.cfg.train.relabel_ratio is None:
            return main_original, prior_original, main_obs, main_next_obs, prior_obs, prior_next_obs
        main_mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
        prior_mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio

        def merge(original, sampled, mask, obs, next_obs):
            merged = RelabeledBehaviorContext(
                z=torch.where(mask, sampled.z, original.z),
                heading=torch.where(mask, sampled.heading, original.heading),
                next_heading=torch.where(mask, sampled.next_heading, original.next_heading),
                heading_valid=torch.where(mask, sampled.heading_valid, original.heading_valid),
                source_type=torch.where(mask, sampled.source_type, original.source_type),
            )
            relabeled_obs = dict(obs)
            relabeled_next_obs = dict(next_obs)
            relabeled_obs["heading"] = merged.heading
            relabeled_next_obs["heading"] = merged.next_heading
            return merged, relabeled_obs, relabeled_next_obs

        main_context, main_obs, main_next_obs = merge(
            main_original, sampled_main, main_mask, main_obs, main_next_obs
        )
        prior_context, prior_obs, prior_next_obs = merge(
            prior_original, sampled_prior, prior_mask, prior_obs, prior_next_obs
        )
        return (
            main_context,
            prior_context,
            main_obs,
            main_next_obs,
            prior_obs,
            prior_next_obs,
        )

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        behavior_prior_enabled = self._behavior_prior_enabled
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        main_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        dedicated_prior = behavior_prior_enabled and "prior" in replay_buffer
        if self.cfg.model.velocity_critic_enabled:
            if dedicated_prior:
                raise ValueError("QV v1 supports main-stream training only")
            self._velocity_expert_batch = expert_batch
        # No dedicated replay means the original UFO topology: D, Q_D and the
        # Actor's behavior-prior term use the same main-terrain batch.  Keeping
        # the optional prior provider preserves the canonical-plane ablation
        # without forking the optimizer implementation.
        prior_batch = None
        if behavior_prior_enabled:
            prior_batch = (
                replay_buffer["prior"].sample(self.cfg.train.batch_size)
                if dedicated_prior
                else main_batch
            )

        main_obs, main_action, main_next_obs = (
            tree_map(lambda x: x.to(self.device), main_batch["observation"]),
            main_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), main_batch["next"]["observation"]),
        )
        main_discount = self.cfg.train.discount * ~main_batch["next"]["terminated"].to(self.device)
        prior_obs = prior_action = prior_next_obs = prior_discount = None
        if behavior_prior_enabled:
            if prior_batch is None:
                raise RuntimeError("behavior_prior_enabled requires a behavior-prior batch")
            prior_obs, prior_action, prior_next_obs = (
                tree_map(lambda x: x.to(self.device), prior_batch["observation"]),
                prior_batch["action"].to(self.device),
                tree_map(lambda x: x.to(self.device), prior_batch["next"]["observation"]),
            )
            prior_discount = prior_transition_discount(
                prior_batch,
                gamma=self.cfg.train.discount,
                device=self.device,
            )
        expert_obs, expert_next_obs = (
            tree_map(lambda x: x.to(self.device), expert_batch["observation"]),
            tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"]),
        )

        if self._heading_context_enabled:
            required_main = {
                "heading_next",
                "heading_z_next",
                "heading_valid",
                "heading_source_type",
                "heading_reward",
                "heading_context_continues",
                "root_heading_xy",
                "next_root_heading_xy",
                "transition_terminated",
                "transition_truncated",
            }
            required_prior = {
                "heading_next",
                "heading_valid",
                "heading_source_type",
                "root_heading_xy",
                "next_root_heading_xy",
            }
            missing_main = sorted(required_main.difference(main_batch))
            missing_prior = (
                sorted(required_prior.difference(prior_batch))
                if behavior_prior_enabled and prior_batch is not None
                else []
            )
            if missing_main or missing_prior:
                raise RuntimeError(
                    "Heading-context replay metadata is incomplete; start a new replay buffer. "
                    f"main_missing={missing_main} prior_missing={missing_prior}"
                )
            # Environment successors contain command placeholders (archived
            # replay may contain already switched commands). Bind s_{t+1}
            # to the context that generated action_t.
            main_next_obs = dict(main_next_obs)
            main_next_obs["heading"] = main_batch["heading_next"].to(self.device)
            if self.cfg.model.velocity_critic_enabled:
                main_next_obs["velocity_command"] = main_batch["velocity_next"].to(self.device)
            if dedicated_prior:
                prior_next_obs = dict(prior_next_obs)
                prior_next_obs["heading"] = prior_batch["heading_next"].to(self.device)

        # Shared running statistics are updated by the main terrain stream
        # only. The canonical-plane prior is transformed in eval mode below.
        self._model._obs_normalizer(main_obs)
        self._model._obs_normalizer(main_next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            main_obs, main_next_obs = (
                self._model._obs_normalizer(main_obs),
                self._model._obs_normalizer(main_next_obs),
            )
            if dedicated_prior:
                prior_obs, prior_next_obs = (
                    self._model._obs_normalizer(prior_obs),
                    self._model._obs_normalizer(prior_next_obs),
                )
            elif behavior_prior_enabled:
                prior_obs, prior_next_obs = main_obs, main_next_obs
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        prior_terrain_var = None
        if dedicated_prior and isinstance(prior_obs, dict) and "terrain_priv" in prior_obs:
            raw_prior = prior_batch["observation"].get("terrain_priv")
            raw_prior_next = prior_batch["next"]["observation"].get("terrain_priv")
            if raw_prior is None or raw_prior_next is None:
                raise RuntimeError("Prior observations must contain terrain_priv at t and t+1")
            # The Agent intentionally accepts any prior_batch provider so A*/B/C
            # ablations do not fork the optimizer code. Workspace collection is
            # the source-of-truth assertion that C's plane stream is raw zero;
            # when this batch is canonical plane, additionally prove that shared
            # normalization did not create sample-wise geometry information.
            raw_plane = (
                torch.count_nonzero(raw_prior).item() == 0
                and torch.count_nonzero(raw_prior_next).item() == 0
            )
            if raw_plane:
                prior_terrain_var = torch.stack(
                    (
                        prior_obs["terrain_priv"].float().var(dim=0, unbiased=False).amax(),
                        prior_next_obs["terrain_priv"].float().var(dim=0, unbiased=False).amax(),
                    )
                ).amax()
                if prior_terrain_var.item() > 1.0e-7:
                    raise RuntimeError(
                        "Canonical-plane terrain_priv acquired sample-wise variation during normalization: "
                        f"max_variance={prior_terrain_var.item():.3e}"
                    )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        main_rollout_z = main_batch["z"].to(self.device)
        prior_rollout_z = (
            prior_batch["z"].to(self.device)
            if behavior_prior_enabled and prior_batch is not None
            else None
        )
        main_z = main_rollout_z
        prior_z = prior_rollout_z
        main_collection_obs = main_obs
        main_collection_next_obs = main_next_obs

        # D sees the selected policy stream's original rollout z, matching the
        # original UFO occupancy supervision. Relabeling happens only after D.
        metrics: dict[str, torch.Tensor] = {}
        if behavior_prior_enabled:
            grad_penalty = (
                self.cfg.train.grad_penalty_discriminator
                if self.cfg.train.grad_penalty_discriminator > 0
                else None
            )
            metrics.update(
                self.update_discriminator(
                    expert_obs=expert_obs,
                    expert_z=expert_z,
                    train_obs=prior_obs,
                    train_z=prior_z,
                    grad_penalty=grad_penalty,
                )
            )

        if self._heading_context_enabled:
            expert_heading_xy = expert_batch["heading_forward_xy"].to(self.device)
            expert_next_heading_xy = expert_batch["next"]["heading_forward_xy"].to(self.device)
            if dedicated_prior:
                (
                    main_context,
                    prior_context,
                    main_obs,
                    main_next_obs,
                    prior_obs,
                    prior_next_obs,
                ) = self._relabel_main_and_prior_context(
                    main_obs=main_obs,
                    main_next_obs=main_next_obs,
                    prior_obs=prior_obs,
                    prior_next_obs=prior_next_obs,
                    main_batch=main_batch,
                    prior_batch=prior_batch,
                    expert_z=expert_z,
                    expert_heading_xy=expert_heading_xy,
                    expert_next_heading_xy=expert_next_heading_xy,
                    main_rollout_z=main_rollout_z,
                    prior_rollout_z=prior_rollout_z,
                )
                main_z, prior_z = main_context.z, prior_context.z
            else:
                main_context, main_obs, main_next_obs = self._relabel_main_context(
                    main_obs=main_obs,
                    main_next_obs=main_next_obs,
                    main_batch=main_batch,
                    expert_z=expert_z,
                    expert_heading_xy=expert_heading_xy,
                    expert_next_heading_xy=expert_next_heading_xy,
                    main_rollout_z=main_rollout_z,
                )
                main_z = main_context.z
                if behavior_prior_enabled:
                    prior_context = main_context
                    prior_obs, prior_next_obs, prior_z = main_obs, main_next_obs, main_z
                else:
                    prior_context = None
        else:
            if dedicated_prior:
                main_z, prior_z = self._relabel_main_and_prior_z(
                    main_next_obs=main_next_obs,
                    prior_next_obs=prior_next_obs,
                    expert_z=expert_z,
                    main_rollout_z=main_rollout_z,
                    prior_rollout_z=prior_rollout_z,
                )
            else:
                main_z = self._relabel_main_z(
                    main_next_obs=main_next_obs,
                    expert_z=expert_z,
                    main_rollout_z=main_rollout_z,
                )
                if behavior_prior_enabled:
                    prior_obs, prior_next_obs, prior_z = main_obs, main_next_obs, main_z

        # Compute the heavier logit quantiles only once per collector step,
        # rather than once for every optimizer update at that step.  This is
        # diagnostics-only and does not alter gradients, losses, or RNG state.
        if behavior_prior_enabled and getattr(self, "_last_discriminator_diagnostic_step", None) != int(step):
            metrics.update(
                self._discriminator_conditioning_diagnostics(
                    expert_obs=expert_obs,
                    expert_z=expert_z,
                    prior_obs=prior_obs,
                    prior_rollout_z=prior_rollout_z,
                    prior_relabel_z=prior_z,
                )
            )
            self._last_discriminator_diagnostic_step = int(step)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=main_obs,
                action=main_action,
                discount=main_discount,
                next_obs=main_next_obs,
                goal=main_next_obs,
                z=main_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        if behavior_prior_enabled:
            metrics.update(
                self.update_critic(
                    obs=prior_obs,
                    action=prior_action,
                    discount=prior_discount,
                    next_obs=prior_next_obs,
                    z=prior_z,
                )
            )
        # compute scalar auxiliary reward as a weighted sum of the auxiliary rewards
        aux_reward = torch.zeros(
            (self.cfg.train.batch_size, 1),
            device=self.device,
            dtype=torch.float32,
        )
        for aux_reward_name in self.cfg.aux_rewards:
            # let's log even this information
            metrics[f"aux_rew/{aux_reward_name}"] = main_batch["aux_rewards"][aux_reward_name].mean()
            aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * main_batch["aux_rewards"][aux_reward_name].to(self.device)

        raw_aux_reward = aux_reward
        aux_normalizer = self._model._aux_reward_normalizer
        freeze_global_steps = int(
            getattr(self.cfg.train, "aux_reward_normalizer_freeze_global_steps", 0)
        )
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        global_step = int(step) * int(world_size)
        if (
            freeze_global_steps > 0
            and global_step >= freeze_global_steps
            and not aux_normalizer.is_frozen
        ):
            aux_normalizer.freeze(global_step=global_step)

        aux_reward = aux_normalizer(raw_aux_reward)
        metrics.update(
            {
                "aux_normalizer/raw_reward_mean": raw_aux_reward.detach().mean(),
                "aux_normalizer/raw_reward_std": raw_aux_reward.detach().std(unbiased=False),
                "aux_normalizer/normalized_reward_mean": aux_reward.detach().mean(),
                "aux_normalizer/normalized_reward_std": aux_reward.detach().std(unbiased=False),
                "aux_normalizer/used_std": aux_normalizer.used_std.detach().float().mean(),
                "aux_normalizer/frozen": torch.tensor(
                    float(aux_normalizer.is_frozen), device=self.device
                ),
                "aux_normalizer/frozen_at_global_step": (
                    aux_normalizer.frozen_at_global_step.detach().float().mean()
                ),
                "aux_normalizer/configured_freeze_global_step": torch.tensor(
                    float(freeze_global_steps), device=self.device
                ),
            }
        )

        metrics.update(
            self.update_aux_critic(
                obs=main_obs,
                action=main_action,
                discount=main_discount,
                aux_reward=aux_reward,
                next_obs=main_next_obs,
                z=main_z,
            )
        )
        heading_branch_active = self._heading_critic_enabled
        if self.cfg.model.velocity_critic_enabled:
            velocity_discount = (self.cfg.train.discount
                * main_batch["heading_context_continues"].to(self.device)
                * main_batch["velocity_valid"].to(self.device)
                * (main_batch["velocity_next"][:, 2:3].to(self.device) > 0)
                * ~main_batch["transition_terminated"].to(self.device)
                * ~main_batch["transition_truncated"].to(self.device))
            metrics.update(self.update_velocity_critic(
                main_collection_obs, main_action, velocity_discount,
                main_batch["velocity_reward"].to(self.device),
                main_batch["velocity_valid"].to(self.device), main_collection_next_obs,
                main_rollout_z, main_batch["heading_z_next"].to(self.device)))
            valid = main_batch["velocity_valid"].to(self.device).float()
            metrics["velocity/error_mps"] = (main_batch["velocity_error_mps"].to(self.device) * valid).sum() / valid.sum().clamp_min(1)
            metrics["velocity/valid_fraction"] = valid.mean()
        if heading_branch_active:
            heading_discount = (
                self.cfg.train.discount
                * main_batch["heading_context_continues"].to(self.device)
                * ~main_batch["transition_terminated"].to(self.device)
                * ~main_batch["transition_truncated"].to(self.device)
            )
            metrics.update(
                self.update_heading_critic(
                    obs=main_collection_obs,
                    action=main_action,
                    discount=heading_discount,
                    heading_reward=main_batch["heading_reward"].to(self.device),
                    heading_valid=main_batch["heading_valid"].to(self.device),
                    next_obs=main_collection_next_obs,
                    z=main_batch["z"].to(self.device),
                    next_z=main_batch["heading_z_next"].to(self.device),
                )
            )
        metrics.update(
            self._run_actor_update(
                main_obs=main_obs,
                main_z=main_z,
                prior_obs=prior_obs if behavior_prior_enabled else None,
                prior_z=prior_z if behavior_prior_enabled else None,
                heading_obs=main_collection_obs if heading_branch_active else None,
                heading_z=main_batch["z"].to(self.device) if heading_branch_active else None,
                heading_valid=(
                    main_batch["heading_valid"].to(self.device) if heading_branch_active else None
                ),
                velocity_valid=main_batch["velocity_valid"].to(self.device) if self.cfg.model.velocity_critic_enabled else None,
                prior_is_main=behavior_prior_enabled and not dedicated_prior,
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics["cfg/behavior_prior_enabled"] = torch.tensor(
            float(behavior_prior_enabled),
            device=self.device,
        )
        if behavior_prior_enabled:
            metrics["prior/discount_mean"] = prior_discount.float().mean().detach()
            metrics["prior/source_is_main"] = torch.tensor(
                float(not dedicated_prior), device=self.device
            )
        if prior_terrain_var is not None:
            metrics["prior/terrain_priv_normalized_var_max"] = prior_terrain_var.detach()
        if self._heading_context_enabled:
            metrics["heading/collection_valid_fraction"] = (
                main_batch["heading_valid"].float().mean().to(self.device)
            )
            metrics["heading/relabel_valid_fraction_main"] = main_context.heading_valid.float().mean().detach()
            if behavior_prior_enabled:
                metrics["heading/relabel_valid_fraction_prior"] = prior_context.heading_valid.float().mean().detach()
            metrics["cfg/reg_coeff_heading"] = torch.tensor(
                self.cfg.train.reg_coeff_heading, device=self.device
            )

        with torch.no_grad():
            if self.cfg.model.velocity_critic_enabled:
                _soft_update_params(tuple(self._model._velocity_critic.parameters()),
                    tuple(self._model._target_velocity_critic.parameters()), self.cfg.train.critic_target_tau)
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            if behavior_prior_enabled:
                _soft_update_params(
                    self._critic_map_paramlist,
                    self._target_critic_map_paramlist,
                    self.cfg.train.critic_target_tau,
                )
            _soft_update_params(
                self._aux_critic_map_paramlist,
                self._aux_target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )
            if heading_branch_active:
                _soft_update_params(
                    self._heading_critic_map_paramlist,
                    self._heading_target_critic_map_paramlist,
                    self.cfg.train.critic_target_tau,
                )

        return metrics

    def update_velocity_critic(self, obs, action, discount, reward, valid, next_obs, z, next_z):
        valid = valid.float()
        denom = valid.sum().clamp_min(1)
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                next_action = self._model._actor(next_obs, next_z, self._model.cfg.actor_std).sample(clip=self.cfg.train.stddev_clip)
                next_qs = self._model._target_velocity_critic(next_obs, next_z, next_action)
                _, unc, next_v = self.get_targets_uncertainty(next_qs, self.cfg.train.heading_critic_pessimism_penalty)
                target = reward + discount * next_v
            stage = self._training_stage("velocity_critic")
            qs = (self._model._velocity_critic if stage is None else stage)(obs, z, action)
            loss = .5 * ((qs - target.unsqueeze(0)).square() * valid.unsqueeze(0)).sum() / denom
        self.velocity_critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._sync_gradients_if_manual(self._model._velocity_critic.parameters())
        self.velocity_critic_optimizer.step()
        return {"velocity/critic_loss": loss.detach(),
                "velocity/Q": (qs.mean(0).detach() * valid).sum() / denom,
                "velocity/target_Q": (target * valid).sum() / denom,
                "velocity/reward": (reward * valid).sum() / denom,
                "velocity/uncertainty": (unc * valid).sum() / denom}

    def update_aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            # compute target critic
            with torch.no_grad():
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_aux_critic(next_obs, z, next_action)  # num_parallel x batch x 1
                # TODO AL: should we have aux_critic parameters here?
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.aux_critic_pessimism_penalty)
                target_Q = aux_reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            # compute critic loss
            aux_critic_stage = self._training_stage("aux_critic")
            aux_critic = self._model._aux_critic if aux_critic_stage is None else aux_critic_stage
            Qs = aux_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        self._sync_gradients_if_manual(self._model._aux_critic.parameters())
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_auxQ": target_Q.mean().detach(),
                "auxQ1": Qs.mean().detach(),
                "mean_next_auxQ": Q_mean.mean().detach(),
                "unc_auxQ": Q_unc.mean().detach(),
                "aux_critic_loss": aux_critic_loss.mean().detach(),
                "mean_aux_reward": aux_reward.mean().detach(),
            }
        return output_metrics

    def update_heading_critic(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        heading_reward: torch.Tensor,
        heading_valid: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        z: torch.Tensor,
        next_z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Train Q_H only on collection-time context-matched transitions."""

        valid = heading_valid.to(device=self.device, dtype=torch.float32)
        valid_count = valid.sum().clamp_min(1.0)
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.heading_critic.num_parallel
            with torch.no_grad():
                dist = self._model._actor(next_obs, next_z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_heading_critic(next_obs, next_z, next_action)
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(
                    next_Qs,
                    self.cfg.train.heading_critic_pessimism_penalty,
                )
                target_Q = heading_reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            heading_stage = self._training_stage("heading_critic")
            heading_critic = self._model._heading_critic if heading_stage is None else heading_stage
            Qs = heading_critic(obs, z, action)
            squared_error = torch.square(Qs - expanded_targets)
            heading_critic_loss = 0.5 * (squared_error * valid.unsqueeze(0)).sum() / valid_count

        self.heading_critic_optimizer.zero_grad(set_to_none=True)
        heading_critic_loss.backward()
        self._sync_gradients_if_manual(self._model._heading_critic.parameters())
        self.heading_critic_optimizer.step()

        with torch.no_grad():
            valid_denom = valid_count
            return {
                "heading/target_Q": (target_Q * valid).sum().detach() / valid_denom,
                "heading/Q": (Qs.mean(dim=0) * valid).sum().detach() / valid_denom,
                "heading/next_Q": (Q_mean * valid).sum().detach() / valid_denom,
                "heading/uncertainty": (Q_unc * valid).sum().detach() / valid_denom,
                "heading/critic_loss": heading_critic_loss.detach(),
                "heading/reward": (heading_reward * valid).sum().detach() / valid_denom,
            }

    def update_actor(
        self,
        main_obs: torch.Tensor | dict[str, torch.Tensor],
        main_z: torch.Tensor,
        prior_obs: torch.Tensor | dict[str, torch.Tensor] | None,
        prior_z: torch.Tensor | None,
        heading_obs: torch.Tensor | dict[str, torch.Tensor] | None = None,
        heading_z: torch.Tensor | None = None,
        heading_valid: torch.Tensor | None = None,
        velocity_valid: torch.Tensor | None = None,
        prior_is_main: bool = False,
        clip_grad_norm: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        actor_stage = self._training_stage("actor")
        actor = self._model._actor if actor_stage is None else actor_stage
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            # One Actor forward and optimizer step combines exactly the active
            # objective branches.  The formal no-D path omits the prior batch
            # entirely instead of feeding main terrain data through Q_D.
            obs_parts = [main_obs]
            z_parts = [main_z]
            if prior_obs is not None:
                if prior_z is None:
                    raise ValueError("prior_obs requires prior_z")
                if not prior_is_main:
                    obs_parts.append(prior_obs)
                    z_parts.append(prior_z)
            elif prior_z is not None:
                raise ValueError("prior_z requires prior_obs")
            if heading_obs is not None:
                if heading_z is None or heading_valid is None:
                    raise ValueError("Heading Actor branch requires heading_z and heading_valid")
                obs_parts.append(heading_obs)
                z_parts.append(heading_z)
            combined_obs = tree_map(lambda *parts: torch.cat(parts, dim=0), *obs_parts)
            combined_z = torch.cat(z_parts, dim=0)
            dist = actor(combined_obs, combined_z, self._model.cfg.actor_std)
            combined_action = dist.sample(clip=self.cfg.train.stddev_clip)
            main_batch_size = main_z.shape[0]
            estimator_loss = None
            estimator_metrics = {}
            if getattr(self.cfg.model.archi.actor, "estimated_velocity_enabled", False):
                if "velocity_target_yaw" not in main_obs:
                    raise RuntimeError("Current-velocity training needs aligned ground-truth targets")
                target = main_obs["velocity_target_yaw"].detach().float()
                prediction = dist.velocity_estimate[:main_batch_size].float()
                estimator_loss, estimator_metrics = velocity_estimation_loss_metrics(prediction, target)
            main_action = combined_action[:main_batch_size]
            offset = main_batch_size
            prior_action = None
            if prior_obs is not None:
                if prior_z is None:
                    raise ValueError("prior_obs requires prior_z")
                if prior_is_main:
                    prior_action = main_action
                else:
                    prior_action = combined_action[offset : offset + main_batch_size]
                    offset += main_batch_size
            heading_action = combined_action[offset:] if heading_obs is not None else None

            # Main RP1 terrain realization and physical auxiliary objectives.
            Qs_aux = self._model._aux_critic(main_obs, main_z, main_action)
            _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, self.cfg.train.actor_pessimism_penalty)  # batch

            Fs = self._model._forward_map(main_obs, main_z, main_action)
            Qs_fb = (Fs * main_z).sum(-1)  # num_parallel x batch
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

            weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            actor_loss = (
                - Q_aux.mean() * self.cfg.train.reg_coeff_aux * weight
                - Q_fb.mean()
            )
            Q_discriminator_mean = torch.zeros((), device=self.device, dtype=Q_fb.dtype)
            if prior_obs is not None:
                Qs_discriminator = self._model._critic(prior_obs, prior_z, prior_action)
                _, _, Q_discriminator = self.get_targets_uncertainty(
                    Qs_discriminator,
                    self.cfg.train.actor_pessimism_penalty,
                )
                Q_discriminator_mean = Q_discriminator.mean()
                actor_loss = (
                    actor_loss
                    - Q_discriminator_mean * self.cfg.train.reg_coeff * weight
                )
            Q_heading_mean = torch.zeros((), device=self.device, dtype=Q_fb.dtype)
            if heading_action is not None:
                Qs_heading = self._model._heading_critic(heading_obs, heading_z, heading_action)
                _, _, Q_heading = self.get_targets_uncertainty(
                    Qs_heading,
                    self.cfg.train.actor_pessimism_penalty,
                )
                valid = heading_valid.to(device=self.device, dtype=Q_heading.dtype).reshape(-1)
                Q_heading_mean = (Q_heading.reshape(-1) * valid).sum() / valid.sum().clamp_min(1.0)
                actor_loss = actor_loss - Q_heading_mean * self.cfg.train.reg_coeff_heading * weight
            Q_velocity_mean = torch.zeros((), device=self.device, dtype=Q_fb.dtype)
            if self.cfg.model.velocity_critic_enabled:
                if heading_action is None or velocity_valid is None:
                    raise ValueError("QV requires matching collection-time Actor branch")
                qs_velocity = self._model._velocity_critic(heading_obs, heading_z, heading_action)
                _, _, q_velocity = self.get_targets_uncertainty(qs_velocity, self.cfg.train.actor_pessimism_penalty)
                vvalid = velocity_valid.to(q_velocity).reshape(-1)
                Q_velocity_mean = (q_velocity.reshape(-1) * vvalid).sum() / vvalid.sum().clamp_min(1)
                actor_loss = actor_loss - self.cfg.train.reg_coeff_velocity * weight * Q_velocity_mean

            if estimator_loss is not None:
                actor_loss = actor_loss + self.cfg.train.velocity_estimator_loss_coef * estimator_loss

        gradient_metrics = {}
        interval = getattr(self.cfg.train, 'actor_gradient_monitor_every', 0)
        if interval:
            self._actor_gradient_audit_count = getattr(self, '_actor_gradient_audit_count', 0) + 1
            if self._actor_gradient_audit_count % interval == 0:
                from pbfm.agents.gradient_audit import actor_gradient_metrics
                terms = {'FB': -Q_fb.mean(), 'Aux': -Q_aux.mean()*self.cfg.train.reg_coeff_aux*weight}
                if prior_obs is not None:
                    terms['D'] = -Q_discriminator_mean*self.cfg.train.reg_coeff*weight
                if heading_action is not None:
                    terms['H'] = -Q_heading_mean*self.cfg.train.reg_coeff_heading*weight
                if self.cfg.model.velocity_critic_enabled:
                    terms['V'] = -Q_velocity_mean*self.cfg.train.reg_coeff_velocity*weight
                if estimator_loss is not None:
                    terms['Estimator'] = self.cfg.train.velocity_estimator_loss_coef * estimator_loss
                gradient_metrics = actor_gradient_metrics(terms, self._model._actor.parameters())
                if estimator_loss is not None and self._model._actor.cfg.velocity_supervision_updates_depth:
                    depth_metrics = actor_gradient_metrics(terms, self._model._actor.depth_encoder.parameters())
                    gradient_metrics.update({k.replace('actor_gradient_local/', 'depth_gradient_local/'): v
                                             for k, v in depth_metrics.items()})

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._sync_gradients_if_manual(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            regularizer_scale = (
                weight
                if isinstance(weight, torch.Tensor)
                else torch.tensor(float(weight), device=self.device, dtype=Q_fb.dtype)
            )
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator_mean.detach(),
                "Q_aux": Q_aux.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
                "Q_heading": Q_heading_mean.detach(),
                "Q_velocity": Q_velocity_mean.detach(),
                "actor_objective/velocity_weighted_value": (Q_velocity_mean * self.cfg.train.reg_coeff_velocity * regularizer_scale).detach(),
                # Signed value terms in the maximized Actor objective. These
                # are value-scale diagnostics, not gradient contributions.
                "actor_objective/fb_value": Q_fb.mean().detach(),
                "actor_objective/aux_weighted_value": (
                    Q_aux.mean() * self.cfg.train.reg_coeff_aux * regularizer_scale
                ).detach(),
                "actor_objective/discriminator_weighted_value": (
                    Q_discriminator_mean * self.cfg.train.reg_coeff * regularizer_scale
                ).detach(),
                "actor_objective/heading_weighted_value": (
                    Q_heading_mean * self.cfg.train.reg_coeff_heading * regularizer_scale
                ).detach(),
                "actor_objective/regularizer_scale": regularizer_scale.detach(),
            }
        output_metrics.update(gradient_metrics)
        output_metrics.update(estimator_metrics)
        return output_metrics

    def sample_action_from_norm_obs(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)
        return action

    def update_fb(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                # dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                # next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_action = self.sample_action_from_norm_obs(next_obs, z)
                target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # num_parallel x batch x z_dim
                target_B = self._model._target_backward_map(goal)  # batch x z_dim
                target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
                _, _, target_M = self.get_targets_uncertainty(target_Ms, self.cfg.train.fb_pessimism_penalty)  # batch x batch

            # compute FB loss
            fb_stage = self._training_stage("fb")
            if fb_stage is None:
                Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
                B = self._model._backward_map(goal)  # batch x z_dim
            else:
                Fs, B = fb_stage(obs, z, action, goal)
            Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

            diff = Ms - discount * target_M  # num_parallel x batch x batch
            fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
            fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            fb_loss = fb_offdiag + fb_diag

            # compute orthonormality loss for backward embedding
            Cov = torch.matmul(B, B.T)
            orth_loss_diag = -Cov.diag().mean()
            orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
            orth_loss = orth_loss_offdiag + orth_loss_diag
            fb_loss += self.cfg.train.ortho_coef * orth_loss

            q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
            if q_loss_coef is not None:
                with torch.no_grad():
                    next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                    _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.fb_pessimism_penalty)  # batch
                    # TODO: we disable autocast here to make sure B and cov have the same dtype (otherwise torch.linalg.solve fails)
                    with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=False):
                        cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                    # inv_cov = torch.inverse(cov)  # z_dim x z_dim
                    B_inv_conv = torch.linalg.solve(cov, B, left=False)
                    implicit_reward = (B_inv_conv * z).sum(dim=-1)  # batch
                    target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                    expanded_targets = target_Q.expand(Fs.shape[0], -1)
                Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
                q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
                fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        self._sync_gradients_if_manual(
            (*self._model._forward_map.parameters(), *self._model._backward_map.parameters())
        )
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
            }
        return output_metrics

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    @torch.compiler.disable
    def gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor | dict[str, torch.Tensor],
        real_z: torch.Tensor,
        fake_obs: torch.Tensor | dict[str, torch.Tensor],
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = tree_get_batch_size(real_obs)
        alpha = torch.rand(batch_size, 1, device=real_z.device)

        # TODO does not work for nested dicts
        interpolated_obs = {}
        interpolated_obs_list = []
        if isinstance(real_obs, torch.Tensor):
            if real_obs.shape != fake_obs.shape:
                raise ValueError(f"Shape mismatch: {real_obs.shape} vs {fake_obs.shape}")
            interpolated_obs = (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True)
            interpolated_obs_list.append(interpolated_obs)
        else:
            for key in real_obs.keys():
                real_obs_tensor = real_obs[key]
                fake_obs_tensor = fake_obs[key]
                if isinstance(real_obs_tensor, torch.Tensor):
                    interpolated_obs[key] = (alpha * real_obs_tensor + (1 - alpha) * fake_obs_tensor).requires_grad_(True)
                    interpolated_obs_list.append(interpolated_obs[key])
                else:
                    raise ValueError(f"Unsupported type for key {key}: {type(real_obs_tensor)}")

        interpolated_z = alpha * real_z + (1 - alpha) * fake_z
        interpolated_z = interpolated_z.requires_grad_(True)

        d_interpolates = self._model._discriminator.compute_logits(interpolated_obs, interpolated_z)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolated_obs_list + [interpolated_z],
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            # obs might contain entries that are not used by discriminator, in which case autograd.grad would complain about unused variables
            # We need to explicitely allow unused variables
            # TODO maybe remove these variables from the inputs completely instead? (we could read self._model._discriminator.filtered_space.keys, but this creates a dependency on the discriminator)
            allow_unused=True,
        )

        # Filter out None's from gradients: if any input is not used by the discriminator, autograd.grad will return None for its gradient
        gradients = [g for g in gradients if g is not None]
        cat_gradients = torch.cat(gradients, dim=1)
        gradient_penalty = ((cat_gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Example of code with grad_penalty_obs_weight
        # obs_dim = real_obs.shape[-1]
        # grad_obs = gradients[:, :obs_dim]
        # grad_z = gradients[:, obs_dim:]
        # w_obs = self.cfg.train.grad_penalty_obs_weight
        # w_z = 1 - w_obs
        # gradient_penalty_obs = ((grad_obs.norm(2, dim=1) - w_obs) ** 2).mean()
        # gradient_penalty_z = ((grad_z.norm(2, dim=1) - w_z) ** 2).mean()
        # gradient_penalty = 0.5 * (gradient_penalty_obs + gradient_penalty_z)
        return gradient_penalty

    def update_discriminator(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        train_obs: torch.Tensor | dict[str, torch.Tensor],
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            discriminator_stage = self._training_stage("discriminator")
            if discriminator_stage is None:
                expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
                unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)
            else:
                expert_logits = discriminator_stage(obs=expert_obs, z=expert_z)
                unlabeled_logits = discriminator_stage(obs=train_obs, z=train_z)
            if self.cfg.train.discriminator_loss == "bce":
                # Raw-logit binary cross entropy, retained for the original
                # UFO/MetaMotivo agents and controlled ablations.
                expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
                unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
                data_loss = torch.mean(expert_loss + unlabeled_loss)
            elif self.cfg.train.discriminator_loss == "lsgan":
                # AMP-style least-squares targets: expert=+1, policy=-1.
                expert_loss = 0.5 * (expert_logits - 1.0).square()
                unlabeled_loss = 0.5 * (unlabeled_logits + 1.0).square()
                data_loss = expert_loss.mean() + unlabeled_loss.mean()
            else:  # pragma: no cover - protected by the pydantic Literal
                raise ValueError(
                    f"Unsupported discriminator loss: {self.cfg.train.discriminator_loss!r}"
                )

            weighted_gp = torch.zeros((), device=data_loss.device, dtype=data_loss.dtype)
            loss = data_loss

            if grad_penalty is not None:
                wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
                weighted_gp = float(grad_penalty) * wgan_gp
                loss = loss + weighted_gp

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._sync_gradients_if_manual(self._model._discriminator.parameters())
        self.discriminator_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean().detach(),
                "disc_train_loss": unlabeled_loss.detach().mean().detach(),
                "disc/data_loss": data_loss.detach(),
                "disc/gp_weighted": weighted_gp.detach(),
                "disc/total_loss": loss.detach(),
                "disc/gp_fraction": (
                    weighted_gp / (data_loss.detach().abs() + weighted_gp.abs()).clamp_min(1.0e-8)
                ).detach(),
            }
            if self.cfg.train.discriminator_loss == "lsgan":
                output_metrics["disc/expert_lsgan"] = expert_loss.detach().mean()
                output_metrics["disc/policy_lsgan"] = unlabeled_loss.detach().mean()
            if grad_penalty is not None:
                output_metrics["disc_wgan_gp_loss"] = wgan_gp.detach()
                output_metrics["disc/gp_raw"] = wgan_gp.detach()
            else:
                output_metrics["disc/gp_raw"] = torch.zeros_like(loss.detach())
        return output_metrics

    def discriminator_reward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the configured behavior-prior reward.

        ``log_odds`` preserves the historical implementation.  ``amp`` uses
        the bounded least-squares reward from AMP, which is exactly in [0, 1]
        and therefore prevents discriminator-logit drift from changing the
        Q_D return scale.
        """

        logits = self._model._discriminator.compute_logits(obs=obs, z=z)
        return self.discriminator_reward_from_logits(logits)

    def discriminator_reward_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Map raw discriminator outputs to the configured scalar reward."""

        reward_mode = getattr(
            getattr(self.cfg, "train", None),
            "discriminator_reward",
            "log_odds",
        )
        if reward_mode == "log_odds":
            probability = torch.sigmoid(logits).clamp(1.0e-6, 1.0 - 1.0e-6)
            return torch.log(probability) - torch.log1p(-probability)
        if reward_mode == "amp":
            return (1.0 - 0.25 * (logits - 1.0).square()).clamp_min(0.0)
        raise ValueError(  # pragma: no cover - protected by the pydantic Literal
            f"Unsupported discriminator reward: {reward_mode!r}"
        )

    def update_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            # compute target critic
            with torch.no_grad():
                reward = self.discriminator_reward(obs=obs, z=z)
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_critic(next_obs, z, next_action)  # num_parallel x batch x 1
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.critic_pessimism_penalty)
                target_Q = reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            # compute critic loss
            critic_stage = self._training_stage("critic")
            critic = self._model._critic if critic_stage is None else critic_stage
            Qs = critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._sync_gradients_if_manual(self._model._critic.parameters())
        self.critic_optimizer.step()

        with torch.no_grad():
            online_Q_mean, online_Q_unc, _ = self.get_targets_uncertainty(Qs, 0.0)
            target_Q_abs_scale = Q_mean.abs().mean()
            output_metrics = {
                "target_Q": target_Q.mean().detach(),
                "Q1": Qs.mean().detach(),
                "mean_next_Q": Q_mean.mean().detach(),
                "unc_Q": Q_unc.mean().detach(),
                # ``unc_Q`` is target-critic disagreement, not a loss.  Keep
                # the historical key and additionally expose its scale-aware
                # form plus the individual ensemble heads.  The old ``Q1``
                # key is actually an all-head mean and is retained only for
                # dashboard compatibility.
                "qd/target_abs_scale": target_Q_abs_scale.detach(),
                "qd/relative_uncertainty": (
                    Q_unc.mean() / target_Q_abs_scale.clamp_min(1.0e-8)
                ).detach(),
                "qd/online_uncertainty": online_Q_unc.mean().detach(),
                "qd/online_abs_scale": online_Q_mean.abs().mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_disc_reward": reward.mean().detach(),
            }
            # The production configuration has two heads.  Logging them
            # separately is essential: the legacy ``Q1`` metric cannot reveal
            # head divergence because it averages the ensemble first.
            if num_parallel >= 2:
                output_metrics.update(
                    {
                        "qd/online_head0_mean": Qs[0].mean().detach(),
                        "qd/online_head1_mean": Qs[1].mean().detach(),
                        "qd/target_head0_mean": next_Qs[0].mean().detach(),
                        "qd/target_head1_mean": next_Qs[1].mean().detach(),
                    }
                )
        return output_metrics

    def _sample_tracking_context(self, replay_buffer, batch_dim, traj_length):
        batch = replay_buffer["expert_slicer"].sample(batch_dim * traj_length, seq_length=traj_length)  # N*T x obs_dim
        # Replay storage can be on CPU while B and its normalizer are on CUDA.
        # Move only the sampled model inputs, never the backing replay storage.
        next_obs = tree_map(lambda x: x.to(self.device), batch["next"]["observation"])
        z = self._model.backward_map(next_obs)  # NT x z_dim
        z = z.view(batch_dim, traj_length, z.shape[-1])  # N x T x z_dim
        for step in range(traj_length):
            end_idx = min(step + self.cfg.model.seq_length, traj_length)
            z[:, step] = z[:, step:end_idx].mean(dim=1)
        if "heading_forward_xy" not in batch:
            raise RuntimeError(
                "Exact tracking heading contexts require expert heading_forward_xy metadata; "
                "rebuild the expert buffer cache"
            )
        heading_reference_xy = normalize_heading_xy(
            batch["heading_forward_xy"].to(device=self.device, dtype=torch.float32)
        ).view(batch_dim, traj_length, 2)
        if getattr(self.cfg.model, "velocity_critic_enabled", False):
            from pbfm.agents.velocity_context import world_to_yaw
            # Companion fields from this exact sample, consumed immediately
            # by advance_rollout_context; no second independently drawn sample.
            self._sampled_velocity_local = world_to_yaw(
                batch["reference_velocity_world_xy"].to(self.device).view(batch_dim, traj_length, 2),
                heading_reference_xy,
            )
            self._sampled_velocity_valid = batch["reference_velocity_valid"].to(self.device).view(batch_dim, traj_length, 1)
        return self._model.project_z(z), heading_reference_xy  # N x T x (z_dim or 2)

    def advance_rollout_context(
        self,
        state: RolloutContextState,
        step_count: torch.Tensor,
        replay_buffer: dict | None = None,
        current_heading_xy: torch.Tensor | None = None,
    ) -> RolloutContextState:
        """Advance one collector without mutating agent-global rollout state.

        Expert-tracking environment identities remain collector-local and
        stable. Each such environment refreshes its own tracking trajectory at
        the trajectory boundary. This avoids one environment reset silently
        replacing every other environment's behavior context.
        """

        device = self._model.device
        counts = step_count.to(device=device, dtype=torch.long).reshape(-1)
        num_envs = counts.shape[0]
        heading_enabled = bool(
            getattr(getattr(self.cfg, "model", None), "heading_context_enabled", False)
        )
        if current_heading_xy is None:
            if heading_enabled:
                raise ValueError("Heading-enabled rollout context requires current_heading_xy")
            current_heading_xy = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
            current_heading_xy[:, 0] = 1.0
        else:
            current_heading_xy = normalize_heading_xy(
                current_heading_xy.to(device=device, dtype=torch.float32)
            )

        z = None if state.z is None else state.z.to(device)
        expert_env_ids = state.expert_env_ids
        tracking_z = state.tracking_z
        tracking_heading_target_xy = state.tracking_heading_target_xy
        velocity_enabled = bool(getattr(self.cfg.model, "velocity_critic_enabled", False))
        tracking_velocity = state.tracking_velocity_world_xy
        tracking_velocity_valid = state.tracking_velocity_valid

        if z is None:
            z = self._model.sample_z(num_envs, device=device)
            heading_target_xy = torch.zeros((num_envs, 2), device=device)
            heading_valid = torch.zeros((num_envs, 1), device=device, dtype=torch.bool)
            context_id = torch.ones((num_envs, 1), device=device, dtype=torch.long)
            source_type = torch.full(
                (num_envs, 1), HEADING_SOURCE_INVALID, device=device, dtype=torch.long
            )
            if self.cfg.train.rollout_expert_trajectories:
                if replay_buffer is None:
                    raise ValueError("Expert rollout contexts require an expert replay buffer")
                n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage * num_envs)
                expert_env_ids = torch.randperm(num_envs, device=device)[:n_elem]
                tracking_z, tracking_heading_reference_xy = self._sample_tracking_context(
                    replay_buffer,
                    n_elem,
                    self.cfg.train.rollout_expert_trajectories_length,
                )
                mod_time = counts[expert_env_ids] % self.cfg.train.rollout_expert_trajectories_length
                tracking_heading_target_xy = align_heading_sequence(
                    tracking_heading_reference_xy,
                    current_heading_xy[expert_env_ids],
                    mod_time,
                )
                if velocity_enabled:
                    from pbfm.agents.velocity_context import yaw_to_world
                    tracking_velocity = yaw_to_world(self._sampled_velocity_local, tracking_heading_target_xy)
                    tracking_velocity_valid = self._sampled_velocity_valid.clone()
        else:
            heading_target_xy = (
                state.heading_target_xy.to(device).clone()
                if state.heading_target_xy is not None
                else torch.zeros((num_envs, 2), device=device)
            )
            heading_valid = (
                state.heading_valid.to(device).clone()
                if state.heading_valid is not None
                else torch.zeros((num_envs, 1), device=device, dtype=torch.bool)
            )
            context_id = (
                state.context_id.to(device).clone()
                if state.context_id is not None
                else torch.zeros((num_envs, 1), device=device, dtype=torch.long)
            )
            source_type = (
                state.source_type.to(device).clone()
                if state.source_type is not None
                else torch.full((num_envs, 1), HEADING_SOURCE_INVALID, device=device, dtype=torch.long)
            )

        expert_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)
        if expert_env_ids is not None:
            expert_env_ids = expert_env_ids.to(device=device, dtype=torch.long)
            expert_mask[expert_env_ids] = True

        # Non-tracking contexts follow the existing random/z-buffer rollout
        # schedule and intentionally have no invented heading semantics.
        reset_random = (counts % self.cfg.train.update_z_every_step == 0) & ~expert_mask
        if bool(reset_random.any()):
            if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(num_envs, device=device)
            else:
                new_z = self._model.sample_z(num_envs, device=device)
            z[reset_random] = new_z[reset_random]
            heading_target_xy[reset_random] = 0.0
            heading_valid[reset_random] = False
            source_type[reset_random] = HEADING_SOURCE_INVALID
            context_id[reset_random] += 1

        if self.cfg.train.rollout_expert_trajectories:
            if replay_buffer is None:
                raise ValueError("Expert rollout contexts require an expert replay buffer")
            if expert_env_ids is None or tracking_z is None or tracking_heading_target_xy is None:
                raise RuntimeError("Expert rollout context was not initialized")
            trajectory_length = self.cfg.train.rollout_expert_trajectories_length
            mod_time = counts[expert_env_ids] % trajectory_length
            refresh_rows = torch.nonzero(mod_time == 0, as_tuple=False).reshape(-1)
            # Initialization already sampled every row above. On later calls,
            # refresh only expert environments whose own context expired.
            if state.z is not None and refresh_rows.numel() > 0:
                refreshed_z, refreshed_reference_xy = self._sample_tracking_context(
                    replay_buffer,
                    int(refresh_rows.numel()),
                    trajectory_length,
                )
                refreshed_env_ids = expert_env_ids[refresh_rows]
                refreshed_targets = align_heading_sequence(
                    refreshed_reference_xy,
                    current_heading_xy[refreshed_env_ids],
                    mod_time[refresh_rows],
                )
                tracking_z = tracking_z.clone()
                tracking_heading_target_xy = tracking_heading_target_xy.clone()
                tracking_z[refresh_rows] = refreshed_z
                tracking_heading_target_xy[refresh_rows] = refreshed_targets
                if velocity_enabled:
                    from pbfm.agents.velocity_context import yaw_to_world
                    tracking_velocity = tracking_velocity.clone()
                    tracking_velocity_valid = tracking_velocity_valid.clone()
                    tracking_velocity[refresh_rows] = yaw_to_world(self._sampled_velocity_local, refreshed_targets)
                    tracking_velocity_valid[refresh_rows] = self._sampled_velocity_valid
                context_id[refreshed_env_ids] += 1

            rows = torch.arange(expert_env_ids.numel(), device=device)
            z[expert_env_ids] = tracking_z[rows, mod_time]
            heading_target_xy[expert_env_ids] = tracking_heading_target_xy[rows, mod_time]
            heading_valid[expert_env_ids] = heading_enabled
            source_type[expert_env_ids] = HEADING_SOURCE_EXACT_TRACKING

        velocity_world = velocity_valid = None
        if velocity_enabled:
            velocity_world = torch.zeros((num_envs, 2), device=device)
            velocity_valid = torch.zeros((num_envs, 1), device=device, dtype=torch.bool)
            if expert_env_ids is not None and tracking_velocity is not None:
                rows = torch.arange(expert_env_ids.numel(), device=device)
                idx = counts[expert_env_ids] % self.cfg.train.rollout_expert_trajectories_length
                velocity_world[expert_env_ids] = tracking_velocity[rows, idx]
                velocity_valid[expert_env_ids] = tracking_velocity_valid[rows, idx]
        return RolloutContextState(
            z=z,
            heading_target_xy=heading_target_xy,
            heading_valid=heading_valid,
            context_id=context_id,
            source_type=source_type,
            expert_env_ids=expert_env_ids,
            tracking_z=tracking_z,
            tracking_heading_target_xy=tracking_heading_target_xy,
            tracking_velocity_world_xy=tracking_velocity,
            tracking_velocity_valid=tracking_velocity_valid,
            velocity_world_xy=velocity_world,
            velocity_valid=velocity_valid,
        )

    def next_rollout_velocity_target(self, state, step_count):
        target, valid = state.velocity_world_xy.clone(), state.velocity_valid.clone()
        if state.expert_env_ids is not None and state.tracking_velocity_world_xy is not None:
            ids = state.expert_env_ids
            length = self.cfg.train.rollout_expert_trajectories_length
            idx = (step_count.to(self.device).reshape(-1)[ids].long() % length + 1).clamp(max=length - 1)
            rows = torch.arange(ids.numel(), device=self.device)
            target[ids] = state.tracking_velocity_world_xy[rows, idx]
            valid[ids] = state.tracking_velocity_valid[rows, idx]
        return target, valid

    def next_rollout_heading_target(
        self,
        state: RolloutContextState,
        step_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current context's target evaluated at the resulting state."""

        if state.heading_target_xy is None or state.heading_valid is None:
            raise RuntimeError("Rollout heading context is not initialized")
        target = state.heading_target_xy.clone()
        valid = state.heading_valid.clone()
        if (
            state.expert_env_ids is not None
            and state.tracking_heading_target_xy is not None
            and state.expert_env_ids.numel() > 0
        ):
            ids = state.expert_env_ids.to(device=self.device, dtype=torch.long)
            counts = step_count.to(device=self.device, dtype=torch.long).reshape(-1)
            length = self.cfg.train.rollout_expert_trajectories_length
            current_index = counts[ids] % length
            next_index = torch.clamp(current_index + 1, max=length - 1)
            rows = torch.arange(ids.numel(), device=self.device)
            target[ids] = state.tracking_heading_target_xy[rows, next_index]
        return target, valid

    def next_rollout_z(
        self,
        state: RolloutContextState,
        step_count: torch.Tensor,
    ) -> torch.Tensor:
        """Return the current context's latent at the resulting state.

        Exact tracking contexts are time-varying even though their
        ``context_id`` remains constant, so Q_H must bootstrap with z_{t+1}
        rather than silently reusing z_t.
        """

        if state.z is None:
            raise RuntimeError("Rollout behavior context is not initialized")
        next_z = state.z.clone()
        if (
            state.expert_env_ids is not None
            and state.tracking_z is not None
            and state.expert_env_ids.numel() > 0
        ):
            ids = state.expert_env_ids.to(device=self.device, dtype=torch.long)
            counts = step_count.to(device=self.device, dtype=torch.long).reshape(-1)
            length = self.cfg.train.rollout_expert_trajectories_length
            current_index = counts[ids] % length
            next_index = torch.clamp(current_index + 1, max=length - 1)
            rows = torch.arange(ids.numel(), device=self.device)
            next_z[ids] = state.tracking_z[rows, next_index]
        return next_z

    def rollout_heading_context_continues(
        self,
        state: RolloutContextState,
        next_step_count: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """Return whether Q_H may bootstrap under the same valid context."""

        if state.heading_valid is None:
            raise RuntimeError("Rollout heading context is not initialized")
        counts = next_step_count.to(device=self.device, dtype=torch.long).reshape(-1)
        continues = state.heading_valid.reshape(-1).clone()
        expert_mask = torch.zeros_like(continues)
        if state.expert_env_ids is not None:
            ids = state.expert_env_ids.to(device=self.device, dtype=torch.long)
            expert_mask[ids] = True
            continues[ids] &= counts[ids] % self.cfg.train.rollout_expert_trajectories_length != 0
        continues[~expert_mask] &= counts[~expert_mask] % self.cfg.train.update_z_every_step != 0
        continues &= ~done.to(device=self.device, dtype=torch.bool).reshape(-1)
        return continues.unsqueeze(-1)

    def maybe_update_rollout_context(self, z: torch.Tensor | None, step_count: torch.Tensor, replay_buffer: None = None) -> torch.Tensor:
        # Backward-compatible single-collector entrypoint. New multi-stream
        # training owns RolloutContextState instances in the workspace.
        state = self.advance_rollout_context(
            RolloutContextState(
                z=z,
                heading_target_xy=getattr(self, "heading_target_xy", None),
                heading_valid=getattr(self, "heading_valid", None),
                context_id=getattr(self, "heading_context_id", None),
                source_type=getattr(self, "heading_source_type", None),
                expert_env_ids=self.env_idx_with_expert_rollout,
                tracking_z=getattr(self, "tracking_z", None),
                tracking_heading_target_xy=getattr(self, "tracking_heading_target_xy", None),
            ),
            step_count,
            replay_buffer,
        )
        self.env_idx_with_expert_rollout = state.expert_env_ids
        self.tracking_z = state.tracking_z
        self.heading_target_xy = state.heading_target_xy
        self.heading_valid = state.heading_valid
        self.heading_context_id = state.context_id
        self.heading_source_type = state.source_type
        self.tracking_heading_target_xy = state.tracking_heading_target_xy
        assert state.z is not None
        return state.z

    @classmethod
    def load(cls, path: str, device: str | None = None):
        return load_agent(cls.config_class, path, device=device)

    def save(self, output_folder: str) -> None:
        save_agent(self, output_folder)


def velocity_estimation_loss_metrics(prediction, target):
    """Unmasked state supervision, independent of target-command validity.

    Stopping is valid evidence. The false-motion diagnostic explicitly audits
    estimates that move while true horizontal speed is below 0.1 m/s.
    """
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[-1] != 3:
        raise ValueError("Velocity estimator requires matching [batch,3] targets")
    if not torch.isfinite(target).all():
        raise ValueError("Nonfinite velocity supervision")
    error = prediction - target.detach()
    loss = error.square().mean()
    with torch.no_grad():
        absolute = error.abs()
        metrics = {"velocity_estimator/loss": loss.detach()}
        for i, axis in enumerate("xyz"):
            metrics[f"velocity_estimator/mae_{axis}"] = absolute[:, i].mean()
            metrics[f"velocity_estimator/bias_{axis}"] = error[:, i].mean()
            metrics[f"velocity_estimator/p95_{axis}"] = torch.quantile(absolute[:, i], .95)
        stopped = torch.linalg.vector_norm(target[:, :2], dim=-1) < .1
        moving_estimate = torch.linalg.vector_norm(prediction[:, :2], dim=-1) > .2
        metrics["velocity_estimator/stopped_fraction"] = stopped.float().mean()
        metrics["velocity_estimator/false_motion_when_stopped"] = (
            (stopped & moving_estimate).float().sum() / stopped.sum().clamp_min(1)
        )
    return loss, metrics
