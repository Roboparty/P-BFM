# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os

from pbfm.motion.expert import (
    expert_buffer_cache_spec,
    find_compatible_expert_buffer_cache,
    load_expert_buffer_cache,
    load_expert_trajectories_from_motion_lib,
    save_expert_buffer_cache,
    source_mixed_expert_priorities,
)
from pbfm.envs.mjlab.config import HumanoidVerseMjlabConfig
from pbfm.training.evaluation.tracking import (
    HumanoidVerseMjlabTrackingEvaluation,
    HumanoidVerseMjlabTrackingEvaluationConfig,
)
from pbfm.training.evaluation.same_z_terrain import (
    SameZTerrainEvaluation,
)

os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_float32_matmul_precision("high")

import time
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import gymnasium
import numpy as np
import wandb
from packaging.version import Version
from torch.utils._pytree import tree_map
from tqdm import tqdm

from pbfm.agents.behavior_context import heading_observation, root_heading_xy
from pbfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from pbfm.agents.buffers.transition import DictBuffer, dtype_numpytotorch_lower_precision
from pbfm.agents.fb_cpr_aux.agent import RolloutContextState
from pbfm.agents.misc.loggers import CSVLogger
from pbfm.agents.utils import (
    AnchoredEveryNStepsChecker,
    EveryNStepsChecker,
    set_seed_everywhere,
)
from pbfm.training.distributed import (
    all_gather_objects,
    barrier,
    broadcast_agent_state,
    broadcast_object,
    reduce_metric_accumulators,
    sync_floating_buffers,
)
from pbfm.perception.instinct_direct_depth import DirectDepthConfig
from pbfm.training import checkpoint, config, diagnostics
from pbfm.training.evaluation import tracking

TRAIN_LOG_FILENAME = "train_log.txt"
REWARD_EVAL_LOG_FILENAME = "reward_eval_log.csv"
TRACKING_EVAL_LOG_FILENAME = "tracking_eval_log.csv"


def transition_successor(observation, info):
    """Keep the action result separate from the observation used for the next rollout."""
    successor = {"observation": {
        key: torch.as_tensor(value).clone() for key, value in observation.items() if key not in {"time", "history"}
    }}
    for key in ("qpos", "qvel"):
        if key in info:
            successor[key] = torch.as_tensor(info[key]).clone()
    if "final_env_ids" in info:
        for key, value in successor["observation"].items():
            ids = torch.as_tensor(info["final_env_ids"], device=value.device, dtype=torch.long)
            value[ids] = torch.as_tensor(info["final_observation"][key], device=value.device, dtype=value.dtype)
        for key in ("qpos", "qvel"):
            if key in successor:
                value = successor[key]
                ids = torch.as_tensor(info["final_env_ids"], device=value.device, dtype=torch.long)
                value[ids] = torch.as_tensor(info[f"final_{key}"], device=value.device, dtype=value.dtype)
    return successor


def clone_motion_lib_for_collector(motion_lib, *, num_envs: int):
    """Create an independent runtime view over immutable loaded-FK tensors.

    A second MotionLib construction would redo expensive FK preprocessing and
    keep another tensor copy on the same GPU.  The plane collector only
    reads motion/FK tensors, so it can share those immutable storages while
    owning every mutable sampling field and all object-level bookkeeping.

    Formal 1024-env G1 runs have all 862 LaFAN clips loaded. Small smoke runs
    intentionally clone the main collector's smaller loaded subset so they
    preserve the same reset distribution without forcing full preprocessing.
    """

    if int(getattr(motion_lib, "_num_motions", 0)) <= 0:
        raise RuntimeError("Canonical-plane MotionLib cloning requires loaded FK motions")

    cloned = copy.copy(motion_lib)
    cloned.num_envs = int(num_envs)
    mutable_tensor_fields = (
        "_curr_motion_ids",
        "_termination_history",
        "_success_rate",
        "_sampling_history",
        "_sampling_prob",
        "_sampling_batch_prob",
    )
    for field in mutable_tensor_fields:
        value = getattr(motion_lib, field, None)
        if isinstance(value, torch.Tensor):
            setattr(cloned, field, value.clone())
    cloned.curr_motion_keys = list(motion_lib.curr_motion_keys)
    cloned._refresh_sampling_batch_prob()
    return cloned


def init_wandb(cfg: config.TrainConfig):
    from pathlib import Path
    exp_name = cfg.wandb_run_name if cfg.wandb_run_name else Path(cfg.work_dir).name
    wandb_config = cfg.model_dump()
    wandb.init(entity=cfg.wandb_ename, project=cfg.wandb_pname, group=cfg.wandb_gname, name=exp_name, config=wandb_config, dir="./_wandb")


@dataclass
class _PriorCollectorState:
    td: dict[str, tp.Any]
    info: dict[str, tp.Any]
    terminated: np.ndarray
    truncated: np.ndarray
    rollout: RolloutContextState


class Workspace:
    def __init__(self, cfg: config.TrainConfig) -> None:
        self.cfg = cfg
        self.distributed_rank = int(self.cfg.distributed_rank)
        self.distributed_world_size = int(self.cfg.distributed_world_size)
        self._write_shared_artifacts = (not self.cfg.rank0_only_writes) or self.distributed_rank == 0

        # MJLab environments are created once and shared with evaluation.
        if isinstance(cfg.env, HumanoidVerseMjlabConfig):
            from omegaconf import OmegaConf

            self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
            self.obs_space = self.train_env.single_observation_space
            self.action_space = self.train_env.single_action_space
            self.prior_env = None
            self.prior_env_info = None
            if cfg.prior_plane_envs > 0:
                prior_cfg = config.make_canonical_plane_training_config(cfg.env)
                # Both collectors are live at the same time. Give the plane
                # collector an independent sampling/runtime view while sharing
                # only read-only full-FK tensor storage.
                prior_motion_lib = clone_motion_lib_for_collector(
                    self.train_env._env._motion_lib,
                    num_envs=cfg.prior_plane_envs,
                )
                self.prior_env, self.prior_env_info = prior_cfg.build(
                    num_envs=cfg.prior_plane_envs,
                    motion_lib=prior_motion_lib,
                )
                if self.prior_env.single_observation_space != self.obs_space:
                    raise RuntimeError("Canonical-plane and main collectors produced different observation spaces")
                if self.prior_env.single_action_space != self.action_space:
                    raise RuntimeError("Canonical-plane and main collectors produced different action spaces")
                print(
                    "[INFO] Built canonical-plane training collector: "
                    f"num_envs={cfg.prior_plane_envs}, terrain=plane, terrain_priv=flat_zero, "
                    "evaluation_fast_path=False, training_DR=preserved",
                    flush=True,
                )
        else:
            sample_env, _ = cfg.env.build(num_envs=1)
            self.obs_space = sample_env.observation_space
            self.action_space = sample_env.action_space
            self.prior_env = None
            self.prior_env_info = None

        assert "time" in self.obs_space.keys(), "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"
        # TODO for backwards consistency, we do not pass "time" to the agent, so we remove it from the obs_space we pass to the agent/model
        #      but would we need it at some point?
        del self.obs_space.spaces["time"]

        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        if self._write_shared_artifacts and isinstance(cfg.env, HumanoidVerseMjlabConfig):
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["unresolved_conf"], file)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME) if self._write_shared_artifacts else None

        set_seed_everywhere(self.cfg.seed)

        self.agent, self.cfg, self._checkpoint_status = checkpoint.create_agent_or_load_checkpoint(
            self.work_dir, self.cfg, agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        self._checkpoint_local_time = int(self._checkpoint_status["local_time"])
        self._checkpoint_global_time = int(self._checkpoint_status["global_time"])
        self._optimizer_steps = int(self._checkpoint_status["optimizer_steps"])
        if self.cfg.distributed_sync:
            broadcast_agent_state(self.agent, src=0)
            if self.cfg.distributed_gradient_sync == "ddp":
                enable_ddp = getattr(self.agent, "enable_distributed_gradient_sync", None)
                if not callable(enable_ddp):
                    raise RuntimeError(
                        f"Agent {type(self.agent).__name__} does not support DDP gradient overlap; "
                        "use distributed_gradient_sync='manual'."
                    )
                enable_ddp(bucket_cap_mb=self.cfg.ddp_bucket_cap_mb)
                if self._write_shared_artifacts:
                    print(
                        "[INFO] Gradient synchronization: "
                        f"mode=ddp bucket_cap_mb={self.cfg.ddp_bucket_cap_mb:g} "
                        "gradient_as_bucket_view=True",
                        flush=True,
                    )
        self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        else:
            self.evaluations = {eval_cfg: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0

        self.eval_loggers = {
            name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()
        } if self._write_shared_artifacts else {}

        if self._write_shared_artifacts and self.cfg.use_wandb:
            init_wandb(self.cfg)

        if self._write_shared_artifacts:
            with (self.work_dir / "config.json").open("w") as f:
                f.write(self.cfg.model_dump_json(indent=4))

        self.priorization_eval_name = None
        self._priority_eval_env = None
        if self.cfg.prioritization:
            for name, evaluation in self.evaluations.items():
                if isinstance(evaluation.cfg, HumanoidVerseMjlabTrackingEvaluationConfig):
                    self.priorization_eval_name = name
                    break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

        self.training_with_expert_data = True

        self.manager = None


    def train(self):
        self.start_time = time.time()
        try:
            self.train_online()
        finally:
            self._close_priority_eval_env()
            self._close_prior_env()

    def _close_prior_env(self) -> None:
        if self.prior_env is not None:
            self.prior_env.close()
            self.prior_env = None

    def _reset_prior_collector(self) -> _PriorCollectorState:
        if self.prior_env is None:
            raise RuntimeError("Canonical-plane collector is not enabled")
        td, info = self.prior_env.reset()
        diagnostics._assert_canonical_plane_terrain_priv(td, label="prior.reset.obs")
        zeros = np.zeros(self.cfg.prior_plane_envs, dtype=bool)
        # The reset observation must not be joined to the preceding replay
        # trajectory (startup, resume, and shared-motion-lib evaluation all
        # pass through this path).  Marking the current slot as an episode
        # boundary also makes compact depth reconstruction fill from the new
        # reset frame instead of reading pre-reset history.
        boundary = np.ones(self.cfg.prior_plane_envs, dtype=bool)
        return _PriorCollectorState(
            td=td,
            info=info,
            terminated=boundary,
            truncated=zeros.copy(),
            rollout=RolloutContextState(),
        )

    def _advance_rollout_state(
        self,
        state: RolloutContextState,
        step_count: torch.Tensor,
        replay_buffer: dict[str, tp.Any],
        current_heading_xy: torch.Tensor | None = None,
    ) -> RolloutContextState:
        advance = getattr(self.agent, "advance_rollout_context", None)
        if callable(advance):
            return advance(
                state,
                step_count,
                replay_buffer,
                current_heading_xy=current_heading_xy,
            )
        z = self.agent.maybe_update_rollout_context(
            z=state.z,
            step_count=step_count,
            replay_buffer=replay_buffer,
        )
        return RolloutContextState(z=z)

    @property
    def _heading_context_enabled(self) -> bool:
        return bool(getattr(self.cfg.agent.model, "heading_context_enabled", False))

    def _collector_root_heading_xy(self, vector_env) -> torch.Tensor:
        core = getattr(vector_env, "_env", vector_env)
        return root_heading_xy(core.base_quat.to(device=self.agent.device, dtype=torch.float32))

    def _attach_rollout_heading(
        self,
        obs: dict[str, torch.Tensor],
        state: RolloutContextState,
        current_heading_xy: torch.Tensor,
    ) -> None:
        if not self._heading_context_enabled:
            return
        if state.heading_target_xy is None or state.heading_valid is None:
            raise RuntimeError("Heading-enabled collector produced an incomplete BehaviorContext")
        obs["heading"] = heading_observation(
            current_heading_xy,
            state.heading_target_xy,
            state.heading_valid,
        )
        if getattr(self.cfg.agent.model, "velocity_critic_enabled", False):
            from pbfm.agents.velocity_context import velocity_observation
            obs["velocity_command"] = velocity_observation(
                state.velocity_world_xy, current_heading_xy, state.velocity_valid)

    def _heading_transition_metadata(
        self,
        *,
        vector_env,
        state: RolloutContextState,
        step_count: torch.Tensor,
        current_heading_xy: torch.Tensor | None,
        new_td: dict[str, tp.Any],
        new_terminated: np.ndarray,
        new_truncated: np.ndarray,
        new_info: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Capture action-outcome-aligned Q_H metadata before context switch."""

        if not self._heading_context_enabled:
            return {}
        if current_heading_xy is None:
            raise RuntimeError("Heading-enabled collector is missing the current root heading")
        if state.context_id is None or state.source_type is None or state.heading_valid is None:
            raise RuntimeError("Heading-enabled collector produced incomplete context metadata")
        next_root_heading = self._collector_root_heading_xy(vector_env)
        if new_info is not None and "final_env_ids" in new_info:
            from pbfm.utils.torch_utils import wxyz_to_xyzw
            ids = torch.as_tensor(new_info["final_env_ids"], device=self.agent.device, dtype=torch.long)
            final_qpos = torch.as_tensor(new_info["final_qpos"], device=self.agent.device)
            next_root_heading[ids] = root_heading_xy(wxyz_to_xyzw(final_qpos[:, 3:7]))
        next_target, next_valid = self.agent.next_rollout_heading_target(state, step_count)
        next_z = self.agent.next_rollout_z(state, step_count)
        next_heading = heading_observation(next_root_heading, next_target, next_valid)
        done = torch.as_tensor(
            np.logical_or(new_terminated, new_truncated),
            device=self.agent.device,
            dtype=torch.bool,
        )
        next_step_count = torch.as_tensor(
            new_td["time"],
            device=self.agent.device,
            dtype=torch.long,
        )
        continues = self.agent.rollout_heading_context_continues(
            state,
            next_step_count,
            done,
        )
        # Keep the historical H/V episode-boundary loss masks; this fix changes
        # successor storage, not their artificial reference-window objective.
        reward_valid = state.heading_valid.to(device=self.agent.device, dtype=torch.bool) & ~done.unsqueeze(-1)
        valid_float = reward_valid.to(dtype=torch.float32)
        # heading_observation is zero-centered, so its first component is
        # already valid * (1 - cos(error)).  Keep validity as replay metadata
        # for loss masking rather than exposing it as a network input.
        reward = -valid_float * next_heading[:, :1]
        metadata = {
            "heading_next": next_heading,
            "heading_z_next": next_z,
            "heading_valid": reward_valid,
            "heading_context_id": state.context_id,
            "heading_source_type": state.source_type,
            "heading_reward": reward,
            "heading_context_continues": continues,
            "root_heading_xy": current_heading_xy,
            "next_root_heading_xy": next_root_heading,
            "transition_terminated": torch.as_tensor(
                new_terminated, device=self.agent.device, dtype=torch.bool
            ).unsqueeze(-1),
            "transition_truncated": torch.as_tensor(
                new_truncated, device=self.agent.device, dtype=torch.bool
            ).unsqueeze(-1),
        }
        if getattr(self.cfg.agent.model, "velocity_critic_enabled", False):
            from pbfm.agents.velocity_context import velocity_observation, velocity_reward, VelocityTaskContract
            next_velocity_target, next_velocity_valid = self.agent.next_rollout_velocity_target(state, step_count)
            core = getattr(vector_env, "_env", vector_env)
            actual_velocity = core.robot_root_states[:, 7:9].to(self.agent.device)
            velocity_valid = state.velocity_valid & ~done.unsqueeze(-1)
            contract = VelocityTaskContract(error_scale_mps=self.cfg.agent.train.velocity_error_scale_mps)
            metadata.update(
                velocity_next=velocity_observation(next_velocity_target, next_root_heading, next_velocity_valid),
                velocity_valid=velocity_valid,
                velocity_reward=velocity_reward(actual_velocity, state.velocity_world_xy, velocity_valid, contract),
                velocity_error_mps=torch.where(velocity_valid,
                    torch.linalg.vector_norm(actual_velocity - state.velocity_world_xy, dim=-1, keepdim=True),
                    torch.zeros_like(actual_velocity[:, :1])),
            )
        return metadata

    def _step_prior_collector(
        self,
        state: _PriorCollectorState,
        *,
        replay_buffer: dict[str, tp.Any],
        local_time: int,
        global_time: int,
        optimizer_steps: int,
    ) -> tuple[_PriorCollectorState, float]:
        if self.prior_env is None:
            raise RuntimeError("Canonical-plane collector is not enabled")
        collection_start = time.perf_counter()
        with torch.no_grad():
            obs = tree_map(
                lambda x: torch.tensor(
                    x,
                    dtype=dtype_numpytotorch_lower_precision(x.dtype),
                    device=self.agent.device,
                ),
                state.td,
            )
            step_count = obs.pop("time")
            current_root_heading = self._collector_root_heading_xy(self.prior_env)
            state.rollout = self._advance_rollout_state(
                state.rollout,
                step_count,
                replay_buffer,
                current_heading_xy=current_root_heading,
            )
            if state.rollout.z is None:
                raise RuntimeError("Canonical-plane rollout did not produce z")
            self._attach_rollout_heading(obs, state.rollout, current_root_heading)
            if local_time < self.cfg.num_seed_steps:
                action = self.prior_env.action_space.sample().astype(np.float32)
            else:
                action = self.agent.act(obs=obs, z=state.rollout.z, mean=False)
                action = action.cpu().detach().numpy()

        new_td, _new_reward, new_terminated, new_truncated, new_info = self.prior_env.step(action)
        diagnostics._assert_canonical_plane_terrain_priv(new_td, label="prior.step.obs")
        if self.cfg.fail_on_nonfinite:
            diagnostics._assert_finite(
                new_td,
                label="prior.env.step.obs",
                rank=self.distributed_rank,
                local_time=local_time,
                global_time=global_time,
                optimizer_steps=optimizer_steps,
            )

        episode_boundary = np.logical_or(new_terminated, new_truncated)
        heading_metadata = self._heading_transition_metadata(
            vector_env=self.prior_env,
            state=state.rollout,
            step_count=step_count,
            current_heading_xy=current_root_heading,
            new_td=new_td,
            new_terminated=new_terminated,
            new_truncated=new_truncated,
            new_info=new_info,
        )
        successor = transition_successor(new_td, new_info)
        successor.update(terminated=new_terminated[..., None], truncated=new_truncated[..., None])
        data = {
            "observation": tree_map(lambda x: x[None, ...], obs),
            "action": action[None, ...],
            "z": state.rollout.z[None, ...],
            "episode_boundary": episode_boundary[None, ..., None],
            "transition_terminated": new_terminated[None, ..., None],
            "transition_truncated": new_truncated[None, ..., None],
            "step_count": step_count[None, ..., None],
            "next": tree_map(lambda value: value[None, ...], successor),
        }
        for key in (
            "heading_next",
            "heading_valid",
            "heading_context_id",
            "heading_source_type",
            "root_heading_xy",
            "next_root_heading_xy",
        ):
            if key in heading_metadata:
                data[key] = heading_metadata[key][None, ...]
        data["observation"].pop("history", None)
        replay_buffer["prior"].extend(data)
        return (
            _PriorCollectorState(
                td=new_td,
                info=new_info,
                terminated=new_terminated,
                truncated=new_truncated,
                rollout=state.rollout,
            ),
            time.perf_counter() - collection_start,
        )

    def _uses_fixed_flat_priority_eval(self, evaluation_name: str) -> bool:
        # Archived terrain checkpoints retain this tag; no preset is selected here.
        return (
            self.cfg.prioritization
            and self.cfg.tags.get("agent") in {"fb_terrain", "fb_depth"}
            and evaluation_name == self.priorization_eval_name
        )

    def _get_priority_eval_env(self):
        if self._priority_eval_env is None:
            core = self.train_env._env
            motion_lib = core._motion_lib
            eval_episode_length_s = config.priority_eval_episode_length_s(
                motion_lib._motion_num_frames,
                step_dt=float(core.dt),
                minimum_episode_length_s=float(core.mjlab_env.max_episode_length_s),
            )
            eval_cfg = config.make_flat_terrain_priority_eval_config(self.cfg.env).model_copy(
                update={"max_episode_length_s": eval_episode_length_s}
            )
            eval_num_envs = int(self.cfg.online_parallel_envs)
            local_motion_count = None
            if self.cfg.distributed_sync and self.distributed_world_size > 1:
                num_motions = int(motion_lib._num_unique_motions)
                local_motion_count = len(
                    tracking.distributed_motion_ids(num_motions, self.distributed_rank, self.distributed_world_size)
                )
                # Every rank must derive balanced_motion_chunks() with the same
                # capacity.  Allocating only the local shard size made the last
                # ranks use a smaller chunk size whenever the motion count was
                # not divisible by world_size, producing overlapping/missing
                # motion IDs at the final gather.
                eval_num_envs = tracking.distributed_eval_num_envs(num_motions, self.distributed_world_size)
            self._priority_eval_env, _ = eval_cfg.build(
                num_envs=eval_num_envs,
                motion_lib=motion_lib,
            )
            tags = getattr(self.cfg, "tags", {})
            print(
                "[INFO] Built persistent terrain-aware priority evaluation environment: "
                f"terrain=plane, terrain_height_observation=flat_zero, "
                f"direct_depth_sensor={'on' if tags.get('agent') == 'fb_depth' else 'off'}, "
                f"episode_length_s={eval_episode_length_s:g}, "
                f"num_envs={eval_num_envs}, rank={self.distributed_rank}, "
                f"motion_shard_size={local_motion_count}",
                flush=True,
            )
        return self._priority_eval_env

    def _close_priority_eval_env(self) -> None:
        if self._priority_eval_env is not None:
            self._priority_eval_env.close()
            self._priority_eval_env = None


    def _load_expert_buffer(self):
        if not self.training_with_expert_data:
            return None
        if not self.cfg.load_expert_data_from_motion_lib:
            raise RuntimeError(
                "This MJLab-focused build only supports expert data loaded from the motion library. "
                "Set load_expert_data_from_motion_lib=True."
            )

        started_at = time.time()
        if self.cfg.cache_expert_buffer:
            expert_buffer = self._load_or_build_cached_expert_buffer()
        else:
            if self._write_shared_artifacts:
                print(f"[INFO] Building expert motion buffer on {self.cfg.buffer_device}; cache disabled", flush=True)
            expert_buffer = load_expert_trajectories_from_motion_lib(
                self.train_env._env,
                self.cfg.agent,
                device=self.cfg.buffer_device,
            )
        expert_priorities = source_mixed_expert_priorities(
            self.train_env._env._motion_lib,
            expert_buffer.motion_ids,
            device=self.cfg.buffer_device,
        )
        expert_buffer.update_priorities(
            priorities=expert_priorities,
            idxs=torch.arange(len(expert_buffer.motion_ids), device=self.cfg.buffer_device),
        )
        if self._write_shared_artifacts:
            motion_lib = self.train_env._env._motion_lib
            motion_ids = torch.as_tensor(expert_buffer.motion_ids, device=motion_lib._motion_source_ids.device)
            source_ids = motion_lib._motion_source_ids[motion_ids]
            expert_source_ids = source_ids.to(expert_priorities.device)
            source_mass = {
                str(name): float(expert_priorities[expert_source_ids == source_idx].sum().item())
                for source_idx, name in enumerate(motion_lib._motion_source_names)
            }
            print(
                f"[INFO] Expert motion buffer ready: motions={len(expert_buffer.motion_ids)} "
                f"frames={len(expert_buffer)} source_mass={source_mass} elapsed={time.time() - started_at:.1f}s",
                flush=True,
            )
        return expert_buffer

    def _load_or_build_cached_expert_buffer(self):
        cache_dir, fingerprint, metadata = expert_buffer_cache_spec(
            self.train_env._env,
            self.cfg.agent,
            cache_root=self.cfg.expert_buffer_cache_root,
        )
        existing = None
        if not self.cfg.rebuild_expert_buffer_cache:
            existing = find_compatible_expert_buffer_cache(cache_dir, fingerprint, metadata)
        if existing is not None:
            cache_dir, fingerprint = existing

        expert_buffer = None
        rank0_builder = self.distributed_rank == 0 or not self.cfg.distributed_sync
        if rank0_builder:
            if existing is not None:
                if self._write_shared_artifacts:
                    print(f"[INFO] Loading cached expert buffer: {cache_dir}", flush=True)
                try:
                    expert_buffer = load_expert_buffer_cache(
                        self.train_env._env,
                        cache_dir,
                        fingerprint,
                        device=self.cfg.buffer_device,
                    )
                except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                    if self._write_shared_artifacts:
                        print(f"[WARN] Expert buffer cache could not be loaded; rebuilding: {exc}", flush=True)
                    existing = None
            if existing is None:
                if self._write_shared_artifacts:
                    print(
                        f"[INFO] Rank 0 building expert buffer and full-FK cache: {cache_dir}",
                        flush=True,
                    )
                expert_buffer = load_expert_trajectories_from_motion_lib(
                    self.train_env._env,
                    self.cfg.agent,
                    device=self.cfg.buffer_device,
                )
                save_expert_buffer_cache(
                    self.train_env._env,
                    expert_buffer,
                    cache_dir,
                    fingerprint,
                    metadata,
                )
                if self._write_shared_artifacts:
                    cache_bytes = sum(
                        path.stat().st_size
                        for path in (cache_dir / "expert_buffer.pt", cache_dir / "motion_lib_fk.pt")
                    )
                    print(
                        f"[INFO] Expert/full-FK cache atomically published: {cache_dir} "
                        f"size={cache_bytes / (1024**3):.2f} GiB",
                        flush=True,
                    )

        if self.cfg.distributed_sync:
            selection = broadcast_object(
                {"cache_dir": str(cache_dir), "fingerprint": fingerprint},
                src=0,
            )
            cache_dir = Path(selection["cache_dir"])
            fingerprint = str(selection["fingerprint"])
            barrier()
        if self.distributed_rank != 0 and self.cfg.distributed_sync:
            expert_buffer = load_expert_buffer_cache(
                self.train_env._env,
                cache_dir,
                fingerprint,
                device=self.cfg.buffer_device,
            )
        if expert_buffer is None:
            raise RuntimeError("Expert buffer cache initialization did not produce a buffer")
        return expert_buffer

    def train_online(self) -> None:
        expert_buffer = self._load_expert_buffer()
        print("Creating the training environment")

        if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
            train_env = self.train_env
            train_env_info = self.train_env_info
        else:
            train_env, train_env_info = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)

        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = self.work_dir / checkpoint.CHECKPOINT_DIR_NAME
        checkpoint_buffer_dir = checkpoint.checkpoint_buffer_path(
            checkpoint_dir, self.cfg, rank=self.distributed_rank,
            world_size=self.distributed_world_size, name="train",
        )
        loaded_checkpoint_buffer = checkpoint_buffer_dir.exists()
        loaded_prior_checkpoint_buffer = False
        if checkpoint_buffer_dir.exists():
            print("Loading checkpointed buffer")
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim.load_for_training(checkpoint_buffer_dir, device=self.cfg.buffer_device)
            else:
                raise ValueError(
                    "Unversioned transition replay cannot prove pre-reset successors. "
                    "Preserve the archive and resume model-only with fresh trajectory replay."
                )
            print(f"Loaded buffer of size {len(replay_buffer['train'])}")
        else:
            if self.cfg.use_trajectory_buffer:
                compact_depth_history = self.cfg.tags.get("agent") == "fb_depth"
                replay_buffer["train"] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,  # make sure to divide by num_envs
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="done",
                    output_key_t=config._trajectory_output_keys(self.cfg.agent),
                    output_key_tp1=["observation", "terminated", "truncated"],
                    explicit_next=True,
                    sparse_next=True,
                    compact_depth_history=compact_depth_history,
                    depth_history_offsets=DirectDepthConfig().sampled_ages,
                )
                if compact_depth_history:
                    print(
                        "[INFO] Sparse-successor depth replay enabled: storing one uint8 frame per row, "
                        "retaining pre-reset/newest successors and reconstructing 8-frame history when sampling",
                        flush=True,
                    )
            else:
                replay_buffer["train"] = DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device)
        if self.prior_env is not None:
            prior_checkpoint_buffer_dir = checkpoint.checkpoint_buffer_path(
                checkpoint_dir, self.cfg, rank=self.distributed_rank,
                world_size=self.distributed_world_size, name="prior",
            )
            if prior_checkpoint_buffer_dir.exists():
                loaded_prior_checkpoint_buffer = True
                replay_buffer["prior"] = TrajectoryDictBufferMultiDim.load_for_training(
                    prior_checkpoint_buffer_dir,
                    device=self.cfg.buffer_device,
                )
                prior_env_axis = int(replay_buffer["prior"].storage[replay_buffer["prior"].end_key].shape[1])
                if prior_env_axis != self.cfg.prior_plane_envs:
                    raise RuntimeError(
                        "Checkpointed canonical-plane replay env axis does not match the configured collector: "
                        f"buffer={prior_env_axis} configured={self.cfg.prior_plane_envs}"
                    )
                print(f"Loaded canonical-plane buffer of size {len(replay_buffer['prior'])}")
            else:
                replay_buffer["prior"] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.prior_buffer_size // self.cfg.prior_plane_envs,
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="episode_boundary",
                    output_key_t=config._prior_trajectory_output_keys(self.cfg.agent),
                    output_key_tp1=["observation"],
                    explicit_next=True,
                    sparse_next=True,
                    compact_depth_history=True,
                    depth_history_offsets=DirectDepthConfig().sampled_ages,
                )
                print(
                    "[INFO] Compact canonical-plane replay enabled: "
                    f"envs={self.cfg.prior_plane_envs}, capacity={self.cfg.prior_buffer_size}, "
                    f"time_slots={self.cfg.prior_buffer_size // self.cfg.prior_plane_envs}",
                    flush=True,
                )
            prior_sizes = all_gather_objects(int(replay_buffer["prior"].size()))
            if len(set(prior_sizes)) != 1:
                raise RuntimeError(
                    "Canonical-plane replay sizes differ across ranks; refusing to enter mismatched DDP update schedules: "
                    f"{prior_sizes}"
                )
        if self.training_with_expert_data:
            replay_buffer["expert_slicer"] = expert_buffer

        print("Starting training")
        replay_updates_start_local_time = self._checkpoint_local_time
        if self.cfg.resume_replay_warmup_steps > 0:
            if self._checkpoint_local_time <= 0:
                raise RuntimeError("resume_replay_warmup_steps requires a nonzero checkpoint step")
            if loaded_checkpoint_buffer:
                raise RuntimeError(
                    "resume_replay_warmup_steps was requested but a checkpoint replay buffer exists; "
                    "remove or quarantine the buffer explicitly before recovery"
                )
            replay_updates_start_local_time += int(self.cfg.resume_replay_warmup_steps)
            print(
                "[RECOVERY] Rebuilding replay with the loaded policy and no optimizer updates: "
                f"start_local_time={self._checkpoint_local_time}, "
                f"warmup_local_steps={self.cfg.resume_replay_warmup_steps}, "
                f"updates_start_local_time={replay_updates_start_local_time}, "
                f"updates_start_global_time={replay_updates_start_local_time * config._global_step_scale(self.cfg)}",
                flush=True,
            )
        replay_warmup_complete_logged = self.cfg.resume_replay_warmup_steps <= 0
        global_step_scale = config._global_step_scale(self.cfg)
        local_step_increment = self.cfg.online_parallel_envs
        global_step_increment = local_step_increment * global_step_scale
        max_local_time = (self.cfg.num_env_steps + global_step_scale - 1) // global_step_scale
        if self._write_shared_artifacts:
            print(
                "[INFO] Step accounting: "
                f"num_envs_per_rank={config._num_envs_per_rank(self.cfg)}, global_parallel_envs={config._global_parallel_envs(self.cfg)}, "
                f"local_step_increment={local_step_increment}, global_step_increment={global_step_increment}, "
                f"num_env_steps_global={self.cfg.num_env_steps}, num_seed_steps_local={self.cfg.num_seed_steps}, "
                f"update_agent_every_local={self.cfg.update_agent_every}, world_size={self.distributed_world_size}, "
                f"loss_mode={config._distributed_loss_mode(self.cfg)}, effective_batch_size={config._effective_batch_size(self.cfg)}, "
                f"replay_capacity_per_rank={config._replay_capacity_per_rank(self.cfg)}, "
                f"effective_replay_capacity={config._effective_replay_capacity(self.cfg)}, "
                f"trajectory_steps_per_rank={config._trajectory_steps_per_rank(self.cfg)}, "
                f"compile={self.cfg.agent.compile}"
            )
        progb = tqdm(
            total=self.cfg.num_env_steps,
            initial=min(self._checkpoint_global_time, self.cfg.num_env_steps),
            disable=self.cfg.disable_tqdm,
        )
        if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
            train_env._env._training_terrain_global_step = self._checkpoint_global_time
        for name in ("train", "prior"):
            if name in replay_buffer and isinstance(replay_buffer[name], TrajectoryDictBufferMultiDim):
                replay_buffer[name].end_episodes()
        td, info = train_env.reset()
        if self.cfg.fail_on_nonfinite:
            diagnostics._assert_finite(
                td,
                label="env.reset.obs",
                rank=self.distributed_rank,
                local_time=self._checkpoint_local_time,
                global_time=self._checkpoint_global_time,
                optimizer_steps=self._optimizer_steps,
            )
        # see https://farama.org/Vector-Autoreset-Mode
        terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        main_rollout_state = RolloutContextState()
        prior_collector_state = self._reset_prior_collector() if self.prior_env is not None else None
        completed_main_iterations = (
            self._checkpoint_local_time // self.cfg.online_parallel_envs
            if loaded_prior_checkpoint_buffer
            else 0
        )
        prior_env_steps = completed_main_iterations * self.cfg.prior_plane_envs
        prior_collection_seconds = 0.0
        prior_log_start_steps = prior_env_steps
        prior_log_start_seconds = 0.0
        total_metrics = None
        metric_update_counts: dict[str, int] = {}
        start_time = time.time()
        fps_start_time = time.time()
        checkpoint_time_checker = AnchoredEveryNStepsChecker(
            self._checkpoint_global_time,
            self.cfg.checkpoint_every_steps,
        )
        last_saved_global_time = self._checkpoint_global_time
        evaluation_time_checkers = {
            name: AnchoredEveryNStepsChecker(
                self._checkpoint_global_time,
                config._evaluation_interval_steps(
                    evaluation,
                    fallback=self.cfg.eval_every_steps,
                ),
            )
            for name, evaluation in self.evaluations.items()
        }
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_local_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.log_every_updates)

        final_local_time = self._checkpoint_local_time
        final_global_time = self._checkpoint_global_time
        for local_time in range(self._checkpoint_local_time, max_local_time + local_step_increment, local_step_increment):
            global_time = local_time * global_step_scale
            if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                train_env._env._training_terrain_global_step = global_time
            final_local_time = local_time
            final_global_time = global_time
            if global_time > self.cfg.num_env_steps:
                break
            if (local_time != self._checkpoint_local_time) and checkpoint_time_checker.check(global_time):
                checkpoint_time_checker.update_last_step(global_time)
                self.save(local_time=local_time, global_time=global_time, optimizer_steps=self._optimizer_steps, replay_buffer=replay_buffer)
                last_saved_global_time = global_time

            if global_time >= self.cfg.num_env_steps:
                break

            force_eval_at_resume = self.evaluate and global_time == self._checkpoint_global_time
            due_evaluation_names = [
                name
                for name, checker in evaluation_time_checkers.items()
                if force_eval_at_resume or checker.check(global_time)
            ]
            if due_evaluation_names:
                eval_metrics = {}
                due_evaluations = [self.evaluations[name] for name in due_evaluation_names]
                distributed_tracking_eval = (
                    self.cfg.distributed_sync
                    and self.distributed_world_size > 1
                    and any(
                        isinstance(evaluation, HumanoidVerseMjlabTrackingEvaluation)
                        for evaluation in due_evaluations
                    )
                )
                uses_humanoidverse_eval = any(
                    isinstance(
                        evaluation,
                        (HumanoidVerseMjlabTrackingEvaluation, SameZTerrainEvaluation),
                    )
                    for evaluation in due_evaluations
                )
                run_eval_on_this_rank = (
                    distributed_tracking_eval
                    or (not self.cfg.distributed_sync)
                    or self.distributed_rank == 0
                )
                if run_eval_on_this_rank:
                    eval_metrics = self.eval(
                        global_time,
                        replay_buffer=replay_buffer,
                        distributed_shard=distributed_tracking_eval,
                        write_outputs=not distributed_tracking_eval,
                        evaluation_names=due_evaluation_names,
                    )
                if distributed_tracking_eval:
                    if self.distributed_rank == 0:
                        self._record_evaluation_results(global_time, eval_metrics)
                    else:
                        eval_metrics = {}
                if self.cfg.distributed_sync:
                    barrier()
                for name in due_evaluation_names:
                    evaluation_time_checkers[name].update_last_step(global_time)
                if uses_humanoidverse_eval:
                    # reset if there is a pbfm evaluation
                    for name in ("train", "prior"):
                        if name in replay_buffer and isinstance(replay_buffer[name], TrajectoryDictBufferMultiDim):
                            replay_buffer[name].end_episodes()
                    td, info = train_env.reset()
                    if self.cfg.fail_on_nonfinite:
                        diagnostics._assert_finite(
                            td,
                            label="env.post_eval_reset.obs",
                            rank=self.distributed_rank,
                            local_time=local_time,
                            global_time=global_time,
                            optimizer_steps=self._optimizer_steps,
                        )
                    terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    main_rollout_state = RolloutContextState()
                    # The priority evaluator temporarily switches the shared
                    # motion library into evaluation mode.  Main and plane
                    # collectors must both restart after it is restored.
                    if self.prior_env is not None:
                        prior_collector_state = self._reset_prior_collector()

                if self.cfg.prioritization and self.priorization_eval_name in due_evaluation_names:
                    # priorities
                    priority_payload = None
                    # Distributed tracking evaluation runs on every rank, but the
                    # merged metrics only exist on rank 0. Compute priorities once
                    # there, then broadcast the unchanged payload to all ranks.
                    compute_priority_on_this_rank = (
                        not self.cfg.distributed_sync or self.distributed_rank == 0
                    )
                    if compute_priority_on_this_rank:
                        assert len(eval_metrics[self.priorization_eval_name]) == len(replay_buffer["expert_slicer"].motion_ids), (
                            "Mismatch in number of motions returned by the eval"
                        )
                        index_in_buffer, name_in_buffer = {}, {}
                        for i, motion_id in enumerate(replay_buffer["expert_slicer"].motion_ids):
                            index_in_buffer[motion_id] = i
                            if hasattr(replay_buffer["expert_slicer"], "file_names"):
                                name_in_buffer[motion_id] = replay_buffer["expert_slicer"].file_names[i]
                        motions_id, priorities, idxs = [], [], []
                        for _, metr in eval_metrics[self.priorization_eval_name].items():
                            motions_id.append(metr["motion_id"])
                            priorities.append(metr["emd"])
                            idxs.append(index_in_buffer[metr["motion_id"]])
                        non_finite_priorities = [
                            (motion_id, priority)
                            for motion_id, priority in zip(motions_id, priorities)
                            if not np.isfinite(priority)
                        ]
                        if non_finite_priorities:
                            raise RuntimeError(
                                "Priority evaluation produced non-finite EMD values; refusing to update "
                                f"motion sampling weights: {non_finite_priorities[:8]}"
                            )
                        priorities = (
                            torch.clamp(
                                torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                                min=self.cfg.prioritization_min_val,
                                max=self.cfg.prioritization_max_val,
                            )
                            * self.cfg.prioritization_scale
                        )

                        if self.cfg.prioritization_mode == "lin":
                            pass
                        elif self.cfg.prioritization_mode == "exp":
                            priorities = 2**priorities
                        elif self.cfg.prioritization_mode == "bin":
                            bins = torch.floor(priorities)
                            for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                                mask = bins == i
                                n = mask.sum().item()
                                if n > 0:
                                    priorities[mask] = 1 / n
                        else:
                            raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")
                        priority_payload = {
                            "priorities": priorities.detach().cpu(),
                            "motion_ids": motions_id,
                            "idxs": idxs,
                            "file_name": name_in_buffer,
                        }
                    if self.cfg.distributed_sync:
                        priority_payload = broadcast_object(priority_payload, src=0)
                    if priority_payload is None:
                        raise RuntimeError("Prioritization requires evaluation metrics, but no priority payload was produced.")
                    priorities = priority_payload["priorities"].to(self.agent.device)
                    motions_id = priority_payload["motion_ids"]
                    idxs = priority_payload["idxs"]
                    name_in_buffer = priority_payload["file_name"]

                    train_env._env._motion_lib.update_sampling_weight_by_id(
                        priorities=list(priorities), motions_id=motions_id, file_name=name_in_buffer
                    )
                    if self.prior_env is not None:
                        self.prior_env._env._motion_lib.update_sampling_weight_by_id(
                            priorities=list(priorities),
                            motions_id=motions_id,
                            file_name=name_in_buffer,
                        )

                    expert_priorities = source_mixed_expert_priorities(
                        train_env._env._motion_lib,
                        replay_buffer["expert_slicer"].motion_ids,
                        device=self.cfg.buffer_device,
                    )
                    replay_buffer["expert_slicer"].update_priorities(
                        priorities=expert_priorities,
                        idxs=torch.arange(
                            len(replay_buffer["expert_slicer"].motion_ids),
                            device=self.cfg.buffer_device,
                        ),
                    )

            if global_time + global_step_increment > self.cfg.num_env_steps:
                if self._write_shared_artifacts:
                    print(
                        "[INFO] Stopping before next rollout to avoid exceeding global sample budget: "
                        f"current_global_time={global_time}, next_global_time={global_time + global_step_increment}, "
                        f"num_env_steps_global={self.cfg.num_env_steps}"
                    )
                break

            with torch.no_grad():
                obs = tree_map(lambda x: torch.tensor(x, dtype=dtype_numpytotorch_lower_precision(x.dtype), device=self.agent.device), td)
                # TODO consistency with obs_space: remove time assigned by TimeAwareObservationWrapper
                step_count = obs.pop("time")
                current_root_heading = (
                    self._collector_root_heading_xy(train_env)
                    if self._heading_context_enabled
                    else None
                )

                history_context = None
                if "history" in obs:
                    # this works in inference mode
                    if len(obs["history"]["action"]) == 0:
                        history_context = self.agent._model._context_encoder.get_initial_context(self.cfg.online_parallel_envs)
                    else:
                        history_context = self.agent.history_inference(obs=obs["history"]["observation"], action=obs["history"]["action"])[
                            :, -1
                        ].clone()

                main_rollout_state = self._advance_rollout_state(
                    main_rollout_state,
                    step_count,
                    replay_buffer,
                    current_heading_xy=current_root_heading,
                )
                context = main_rollout_state.z
                if context is None:
                    raise RuntimeError("Main rollout did not produce z")
                self._attach_rollout_heading(obs, main_rollout_state, current_root_heading)
                if local_time < self.cfg.num_seed_steps:
                    action = train_env.action_space.sample().astype(np.float32)
                else:
                    # this works in inference mode
                    if history_context is not None:
                        action = self.agent.act(obs=obs, z=context, context=history_context, mean=False)
                    else:
                        action = self.agent.act(obs=obs, z=context, mean=False)
                    # TODO a bit hard-coded -- just to avoid moving stuff from cpu to cuda
                    if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                        action = action.cpu().detach().numpy()
                check_rollout_nonfinite = (
                    self.cfg.fail_on_nonfinite
                    and self.cfg.nonfinite_check_rollout_every_local_steps > 0
                    and local_time % self.cfg.nonfinite_check_rollout_every_local_steps == 0
                )
                if check_rollout_nonfinite:
                    diagnostics._assert_finite(
                        obs,
                        label="rollout.obs",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    diagnostics._assert_finite(
                        context,
                        label="rollout.context",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    diagnostics._assert_finite(
                        history_context,
                        label="rollout.history_context",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    diagnostics._assert_finite(
                        action,
                        label="rollout.action",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
            new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)
            heading_metadata = self._heading_transition_metadata(
                vector_env=train_env,
                state=main_rollout_state,
                step_count=step_count,
                current_heading_xy=current_root_heading,
                new_td=new_td,
                new_terminated=new_terminated,
                new_truncated=new_truncated,
                new_info=new_info,
            )
            # Q_D needs outcome flags even when BehaviorContext heading is
            # disabled. Heading-enabled collection already produced the same
            # values above; setdefault keeps a single source of truth.
            heading_metadata.setdefault(
                "transition_terminated",
                torch.as_tensor(
                    new_terminated, device=self.agent.device, dtype=torch.bool
                ).unsqueeze(-1),
            )
            heading_metadata.setdefault(
                "transition_truncated",
                torch.as_tensor(
                    new_truncated, device=self.agent.device, dtype=torch.bool
                ).unsqueeze(-1),
            )
            if check_rollout_nonfinite:
                diagnostics._assert_finite(
                    new_td,
                    label="env.step.obs",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )
                diagnostics._assert_finite(
                    new_reward,
                    label="env.step.reward",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )

            # we check if at the next iteration we will evaluate
            next_local_time = local_time + local_step_increment
            next_global_time = next_local_time * global_step_scale
            next_uses_humanoidverse_eval = self.evaluate and any(
                checker.check(next_global_time)
                and isinstance(
                    self.evaluations[name],
                    (HumanoidVerseMjlabTrackingEvaluation, SameZTerrainEvaluation),
                )
                for name, checker in evaluation_time_checkers.items()
            )
            if (
                isinstance(self.cfg.env, HumanoidVerseMjlabConfig)
                and next_uses_humanoidverse_eval
            ):
                # Make sure we set truncated since at the next iteration we are
                # forced to reset the shared training environment after eval.
                new_truncated = np.ones_like(new_truncated, dtype=bool)
                heading_metadata["transition_truncated"] = torch.ones(
                    (self.cfg.online_parallel_envs, 1), device=self.agent.device, dtype=torch.bool,
                )
                if "heading_context_continues" in heading_metadata:
                    heading_metadata["heading_context_continues"].zero_()

            successor = transition_successor(new_td, new_info)
            if Version(gymnasium.__version__) >= Version("1.0"):
                if self.cfg.use_trajectory_buffer:
                    data = {
                        "observation": tree_map(lambda x: x[None, ...], obs),
                        "action": action[None, ...],
                        "terminated": new_terminated[None, ..., None],
                        "truncated": new_truncated[None, ..., None],
                        "done": np.logical_or(new_terminated, new_truncated)[None, ..., None],
                        "next": {
                            **tree_map(lambda value: value[None, ...], successor),
                            "terminated": new_terminated[None, ..., None],
                            "truncated": new_truncated[None, ..., None],
                        },
                        "step_count": step_count[None, ..., None],
                        "reward": new_reward[None, ..., None],
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[None, ...]
                    if history_context is not None:
                        data["history_context"] = history_context[None, ...]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][None, ...]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][None, ...]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {k: v[None, ..., None] for k, v in new_info["aux_rewards"].items() if not k.startswith("_")}
                else:
                    # Same-step reset still executes the first action of the new episode.
                    indexes = np.ones_like(new_terminated, dtype=bool)

                    data = {
                        "observation": tree_map(lambda x: x[indexes], obs),
                        "action": action[indexes],
                        "step_count": step_count[indexes],
                        "reward": new_reward[indexes].reshape(-1, 1),
                        "next": {
                            "observation": successor["observation"],
                            "terminated": new_terminated[indexes].reshape(-1, 1),
                            "truncated": new_truncated[indexes].reshape(-1, 1),
                        },
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[indexes]
                    if history_context is not None:
                        data["history_context"] = history_context[indexes]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][indexes]
                        data["next"]["qpos"] = successor["qpos"]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][indexes]
                        data["next"]["qvel"] = successor["qvel"]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {
                            k: v[indexes].reshape(-1, 1) for k, v in new_info["aux_rewards"].items() if not k.startswith("_")
                        }
            else:
                raise NotImplementedError("still some work to do for gymnasium < 1.0")
            if heading_metadata:
                if not self.cfg.use_trajectory_buffer:
                    raise RuntimeError("BehaviorContext heading training requires trajectory replay")
                for key, value in heading_metadata.items():
                    data[key] = value[None, ...]
            if check_rollout_nonfinite:
                diagnostics._assert_finite(
                    data,
                    label="replay.extend.data",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )
            replay_buffer["train"].extend(data)

            if prior_collector_state is not None:
                prior_collector_state, prior_elapsed = self._step_prior_collector(
                    prior_collector_state,
                    replay_buffer=replay_buffer,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )
                prior_env_steps += self.cfg.prior_plane_envs
                prior_collection_seconds += prior_elapsed

            replay_warmup_complete = local_time >= replay_updates_start_local_time
            if replay_warmup_complete and not replay_warmup_complete_logged:
                print(
                    "[RECOVERY] Replay warmup complete; optimizer updates are enabled at "
                    f"local_time={local_time} global_time={global_time} "
                    f"replay_time_steps={len(replay_buffer['train'])}",
                    flush=True,
                )
                replay_warmup_complete_logged = True
            if (
                replay_warmup_complete
                and len(replay_buffer["train"]) > 0
                # Trajectory sampling draws slices with replacement, so the
                # plane replay does not need batch_size distinct transitions.
                # It only needs a real (s_t, s_{t+1}) segment after the
                # administrative startup boundary. Main num_seed_steps is far
                # longer than this in formal runs; the explicit guard keeps
                # small smoke runs useful as well.
                and ("prior" not in replay_buffer or len(replay_buffer["prior"]) >= 3)
                and local_time > self.cfg.num_seed_steps
                and update_agent_time_checker.check(local_time)
            ):
                update_agent_time_checker.update_last_step(local_time)
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(replay_buffer, local_time)
                    self._optimizer_steps += 1
                    if self.cfg.fail_on_nonfinite:
                        diagnostics._assert_finite(
                            metrics,
                            label="agent.update.metrics",
                            rank=self.distributed_rank,
                            local_time=local_time,
                            global_time=global_time,
                            optimizer_steps=self._optimizer_steps,
                        )
                        if (
                            self.cfg.nonfinite_check_model_every_updates > 0
                            and self._optimizer_steps % self.cfg.nonfinite_check_model_every_updates == 0
                        ):
                            diagnostics._assert_model_finite(
                                self.agent._model,
                                rank=self.distributed_rank,
                                local_time=local_time,
                                global_time=global_time,
                                optimizer_steps=self._optimizer_steps,
                            )
                    if self.cfg.distributed_sync:
                        sync_floating_buffers(self.agent._model)
                    total_metrics, metric_update_counts = diagnostics._accumulate_metrics(
                        total_metrics,
                        metric_update_counts,
                        metrics,
                    )

            if log_time_checker.check(global_time) and total_metrics is not None:
                log_time_checker.update_last_step(global_time)
                m_dict = {}
                reduced_metrics = (
                    reduce_metric_accumulators(total_metrics, metric_update_counts)
                    if self.cfg.distributed_average_metrics
                    else {
                        key: (total_metrics[key] / metric_update_counts[key]).mean()
                        for key in sorted(total_metrics)
                    }
                )
                for key, value in reduced_metrics.items():
                    m_dict[key] = np.round(value.item(), 6)
                m_dict.update(diagnostics.torso_contact_force_metrics(
                    train_env, enabled=self.cfg.log_torso_contact_forces,
                    threshold=self.cfg.torso_contact_force_threshold,
                ))
                m_dict.update(diagnostics.terrain_metrics(
                    train_env, distributed_sync=self.cfg.distributed_sync,
                    world_size=self.distributed_world_size,
                ))
                if isinstance(self.cfg.env, HumanoidVerseMjlabConfig) and self.cfg.env.terrain_training_curriculum:
                    from pbfm.terrains.training_curriculum import curriculum_metrics
                    terrain = train_env._env.mjlab_env.scene['terrain']
                    m_dict.update(curriculum_metrics(terrain, self.cfg.env, global_time))
                m_dict["duration [minutes]"] = (time.time() - start_time) / 60
                m_dict["FPS"] = (1 if global_time == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.distributed_sync and self.distributed_world_size > 1:
                    m_dict["distributed/world_size"] = self.distributed_world_size
                    m_dict["distributed/loss_mode"] = config._distributed_loss_mode(self.cfg)
                    m_dict["distributed/effective_batch_size"] = config._effective_batch_size(self.cfg)
                    m_dict["distributed/num_envs_per_rank"] = config._num_envs_per_rank(self.cfg)
                    m_dict["distributed/global_parallel_envs"] = config._global_parallel_envs(self.cfg)
                    m_dict["distributed/replay_capacity_per_rank"] = config._replay_capacity_per_rank(self.cfg)
                    m_dict["distributed/effective_replay_capacity"] = config._effective_replay_capacity(self.cfg)
                    m_dict["distributed/trajectory_steps_per_rank"] = config._trajectory_steps_per_rank(self.cfg)
                    m_dict["distributed/compile"] = int(bool(self.cfg.agent.compile))
                    m_dict["distributed/gradient_sync"] = self.cfg.distributed_gradient_sync
                    m_dict["distributed/ddp_bucket_cap_mb"] = float(self.cfg.ddp_bucket_cap_mb)
                m_dict["distributed/local_env_steps"] = int(local_time)
                m_dict["distributed/global_env_steps"] = int(global_time)
                m_dict["distributed/optimizer_steps"] = int(self._optimizer_steps)
                if prior_collector_state is not None:
                    prior_steps_since_log = prior_env_steps - prior_log_start_steps
                    prior_seconds_since_log = prior_collection_seconds - prior_log_start_seconds
                    prior_global_steps = prior_env_steps * global_step_scale
                    m_dict["prior/plane_env_steps"] = int(prior_global_steps)
                    m_dict["prior/total_sim_transitions"] = int(global_time + prior_global_steps)
                    m_dict["prior/collection_fps"] = float(
                        prior_steps_since_log * global_step_scale / max(prior_seconds_since_log, 1.0e-9)
                    )
                    m_dict["prior/replay_size_per_rank"] = int(replay_buffer["prior"].size())
                    prior_log_start_steps = prior_env_steps
                    prior_log_start_seconds = prior_collection_seconds
                if self._write_shared_artifacts and self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=global_time,
                    )
                if self._write_shared_artifacts:
                    print(m_dict)
                total_metrics = None
                metric_update_counts = {}
                fps_start_time = time.time()
                m_dict["timestep"] = global_time
                m_dict["local_timestep"] = local_time
                if self.train_logger is not None:
                    self.train_logger.log(m_dict)

            progb.update(global_step_increment)
            td = new_td
            terminated = new_terminated
            truncated = new_truncated
            done = np.logical_or(new_terminated.ravel(), new_truncated.ravel())
            info = new_info
        # Periodic checkpoint intervals need not divide the exact global step
        # budget. Persist the state that has consumed the final accepted main
        # rollout before closing the environments, otherwise a watchdog can
        # repeatedly resume from the preceding periodic checkpoint.
        final_global_time = min(final_global_time, self.cfg.num_env_steps)
        final_local_time = final_global_time // global_step_scale
        if final_global_time > last_saved_global_time:
            self.save(
                local_time=final_local_time,
                global_time=final_global_time,
                optimizer_steps=self._optimizer_steps,
                replay_buffer=replay_buffer,
            )
        train_env.close()

    def eval(
        self,
        t,
        replay_buffer,
        *,
        distributed_shard: bool = False,
        write_outputs: bool = True,
        evaluation_names: tp.Iterable[str] | None = None,
    ):
        selected_names = tuple(self.evaluations if evaluation_names is None else evaluation_names)
        unknown_names = sorted(set(selected_names) - set(self.evaluations))
        if unknown_names:
            raise KeyError(f"unknown evaluations requested: {unknown_names}")
        print(
            f"Starting evaluation at time {t} on rank {self.distributed_rank}"
            + (" (distributed motion shard)" if distributed_shard else ""),
            f"evaluations={selected_names}",
            flush=True,
        )
        evaluation_results = {}

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in selected_names:
            logger = self.eval_loggers.get(evaluation_name) if write_outputs else None
            evaluation = self.evaluations[evaluation_name]

            if (
                distributed_shard
                and self.distributed_rank != 0
                and not isinstance(evaluation, HumanoidVerseMjlabTrackingEvaluation)
            ):
                continue

            # NOTE we have this inside the loop so that the agent is not moved to cpu if we don't evaluate
            if not isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                self.agent._model.to("cpu")
            self.agent._model.train(False)

            if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                if isinstance(evaluation, SameZTerrainEvaluation):
                    evaluation_metrics, wandb_dict = evaluation.run(
                        timestep=t,
                        agent_or_model=self.agent,
                        replay_buffer=replay_buffer,
                        logger=logger,
                        base_env_config=self.cfg.env,
                        write_outputs=write_outputs,
                        motion_lib=self.train_env._env._motion_lib,
                        expert_buffer=replay_buffer.get("expert_slicer"),
                    )
                    if write_outputs and self._write_shared_artifacts and self.cfg.use_wandb and wandb_dict is not None:
                        wandb.log(
                            {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                            step=t,
                        )
                    evaluation_results[evaluation_name] = evaluation_metrics
                    continue
                eval_env = (
                    self._get_priority_eval_env()
                    if self._uses_fixed_flat_priority_eval(evaluation_name)
                    else self.train_env
                )
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                    env=eval_env,
                    write_outputs=write_outputs,
                    motion_lib=self.train_env._env._motion_lib,
                    expert_buffer=replay_buffer.get("expert_slicer"),
                    distributed=distributed_shard,
                )
            else:
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                )
            # For wandb dict, put it on wandb
            if write_outputs and self._write_shared_artifacts and self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        # this is important, move back the agent to cuda and
        # restart the training
        if not isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
            self.agent._model.to(self.cfg.agent.model.device)
        self.agent._model.train()

        return evaluation_results

    def _record_evaluation_results(self, timestep: int, evaluation_results: dict[str, dict[str, dict]]) -> None:
        if not self._write_shared_artifacts:
            return
        for evaluation_name, metrics in evaluation_results.items():
            evaluation = self.evaluations[evaluation_name]
            logger = self.eval_loggers.get(evaluation_name)
            evaluation.record_results(metrics, timestep=timestep, logger=logger)
            if self.cfg.use_wandb:
                wandb_dict = evaluation.summarize(metrics)
                wandb.log(
                    {f"eval/{evaluation_name}/{key}": value for key, value in wandb_dict.items()},
                    step=timestep,
                )

    def save(self, *, local_time: int, global_time: int, optimizer_steps: int, replay_buffer: Dict[str, tp.Any]) -> None:
        checkpoint.save_checkpoint(
            self.work_dir,
            self.cfg,
            self.agent,
            write_shared_artifacts=self._write_shared_artifacts,
            rank=self.distributed_rank,
            world_size=self.distributed_world_size,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
            replay_buffer=replay_buffer,
        )


def train_bfm_zero():
    raise RuntimeError(
        "Legacy train_bfm_zero entrypoint is disabled in this MJLab build. "
        "Use pbfm.train or ./run_train.sh."
    )

if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    train_bfm_zero()

# uv run --no-cache -m pbfm.meta_online_entry_point
