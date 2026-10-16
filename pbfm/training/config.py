"""Training configuration, replay schemas, and step-accounting contracts."""

import typing as tp
from typing import Dict, List

import numpy as np
import pydantic
import torch

from pbfm.agents.base import BaseConfig
from pbfm.agents.fb_cpr_aux.config import FBcprAuxAgentConfig
from pbfm.agents.utils import get_local_workdir
from pbfm.envs.mjlab.config import HumanoidVerseMjlabConfig
from pbfm.training.evaluation.tracking import HumanoidVerseMjlabTrackingEvaluationConfig
from pbfm.training.evaluation.same_z_terrain import SameZTerrainEvaluationConfig


Evaluation = tp.Annotated[
    tp.Union[
        HumanoidVerseMjlabTrackingEvaluationConfig,
        SameZTerrainEvaluationConfig,
    ],
    pydantic.Field(discriminator="name"),
]

Agent = FBcprAuxAgentConfig


def make_flat_terrain_priority_eval_config(env_cfg: HumanoidVerseMjlabConfig) -> HumanoidVerseMjlabConfig:
    """Copy a terrain-aware MJLab config and select analytic fixed-flat observations."""
    overrides = list(env_cfg.hydra_overrides)
    terrain_preset = any(value == "terrain=terrain_ufo_v0" for value in overrides)
    terrain_mode_index = next(
        (index for index, value in enumerate(overrides) if value.startswith("terrain.terrain_type=")),
        None,
    )
    if not terrain_preset or terrain_mode_index is None:
        raise ValueError("fixed-flat priority evaluation requires terrain=terrain_ufo_v0 and terrain.terrain_type")
    overrides[terrain_mode_index] = "terrain.terrain_type=plane"
    observation_mode_index = next(
        (index for index, value in enumerate(overrides) if value.startswith("terrain.terrain_priv.mode=")),
        None,
    )
    if observation_mode_index is None:
        overrides.append("terrain.terrain_priv.mode=flat_zero")
    else:
        overrides[observation_mode_index] = "terrain.terrain_priv.mode=flat_zero"
    return env_cfg.model_copy(update={"hydra_overrides": overrides, "evaluation_fast_path": True})


def priority_eval_episode_length_s(
    motion_num_frames,
    *,
    step_dt: float,
    minimum_episode_length_s: float,
) -> float:
    """Return a timeout-safe horizon for fast full-motion tracking evaluation.

    The fast evaluator executes one policy step per expert-buffer frame after
    the initial state.  Reserve one full extra step so the environment timeout
    cannot truncate the longest trajectory on its final action.
    """

    if step_dt <= 0.0 or not np.isfinite(step_dt):
        raise ValueError("priority evaluation step_dt must be finite and positive")
    if minimum_episode_length_s <= 0.0 or not np.isfinite(minimum_episode_length_s):
        raise ValueError("priority evaluation minimum horizon must be finite and positive")
    lengths = torch.as_tensor(motion_num_frames)
    if lengths.numel() == 0:
        raise ValueError("priority evaluation requires at least one motion")
    if not torch.isfinite(lengths).all() or bool((lengths < 2).any()):
        raise ValueError("priority evaluation motion lengths must be finite and at least two frames")
    max_frames = int(lengths.max().item())
    required_episode_length_s = max_frames * float(step_dt)
    return max(float(minimum_episode_length_s), required_episode_length_s)


def make_canonical_plane_training_config(env_cfg: HumanoidVerseMjlabConfig) -> HumanoidVerseMjlabConfig:
    """Copy the main training config while changing only terrain geometry.

    Observation noise, physical/domain randomization, direct depth, reset yaw,
    and action delays remain exactly as configured for the main collector.
    """

    plane_cfg = make_flat_terrain_priority_eval_config(env_cfg)
    return plane_cfg.model_copy(update={"evaluation_fast_path": False})


def _trajectory_output_keys(agent: Agent) -> list[str]:
    keys = [
        "observation",
        "action",
        "z",
        "terminated",
        "truncated",
        "step_count",
        "reward",
        # Action-outcome-aligned flags (also retained for archived models).
        "transition_terminated",
        "transition_truncated",
    ]
    if getattr(agent, "aux_rewards", None):
        keys.append("aux_rewards")
    if bool(getattr(agent.model, "velocity_critic_enabled", False)):
        keys.extend(["velocity_next", "velocity_valid", "velocity_reward", "velocity_error_mps"])
    if bool(getattr(agent.model, "heading_context_enabled", False)):
        keys.extend(
            [
                "heading_next",
                "heading_z_next",
                "heading_valid",
                "heading_context_id",
                "heading_source_type",
                "heading_reward",
                "heading_context_continues",
                "root_heading_xy",
                "next_root_heading_xy",
            ]
        )
    return keys


def _prior_trajectory_output_keys(agent: Agent) -> list[str]:
    keys = [
        "observation",
        "action",
        "z",
        "episode_boundary",
        "transition_terminated",
        "transition_truncated",
        "step_count",
    ]
    if bool(getattr(agent.model, "heading_context_enabled", False)):
        keys.extend(
            [
                "heading_next",
                "heading_valid",
                "heading_context_id",
                "heading_source_type",
                "root_heading_xy",
                "next_root_heading_xy",
            ]
        )
    return keys


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")
    motions: str | None = None
    motions_root: str | None = None

    env: HumanoidVerseMjlabConfig = pydantic.Field(discriminator="name")

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("g1mujoco_train"))

    seed: int = 0
    online_parallel_envs: int = 50
    # Dedicated canonical-plane policy collector used only by D/Q_D. Zero
    # preserves the legacy single-replay behavior for non-depth presets.
    prior_plane_envs: int = 0
    # Note: this is in env steps (multiples of online_parallel_envs)
    log_every_updates: int = 100_000
    num_env_steps: int = 30_000_000
    # Note: this is in env steps (multiples of online_parallel_envs)
    update_agent_every: int = 500
    # Note: this is in env steps (multiples of online_parallel_envs)
    num_seed_steps: int = 50_000
    # Recovery-only policy rollout steps per rank. When resuming a checkpoint
    # without replay buffers, collect with the loaded policy and defer all
    # optimizer updates until this many new local env steps are available.
    resume_replay_warmup_steps: int = 0
    num_agent_updates: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    checkpoint_every_steps: int = 5_000_000
    checkpoint_buffer: bool = True
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2
    prioritization_mode: str = "bin"  # ["bin", "exp", "lin"]
    padding_beginning: int = 0
    padding_end: int = 0

    # Buffer
    use_trajectory_buffer: bool = False
    buffer_size: int = 5_000_000
    prior_buffer_size: int = 0

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None
    wandb_run_name: str | None = None

    # misc
    load_expert_data_from_motion_lib: bool = True
    buffer_device: str = "cpu"
    cache_expert_buffer: bool = True
    rebuild_expert_buffer_cache: bool = False
    expert_buffer_cache_root: str | None = None
    # Default to True; otherwise you will spam the console with tqdm
    disable_tqdm: bool = True
    log_torso_contact_forces: bool = True
    torso_contact_force_threshold: float = 1.0
    distributed_rank: int = 0
    distributed_world_size: int = 1
    rank0_only_writes: bool = True
    checkpoint_rank_buffers: bool = True
    distributed_sync: bool = True
    distributed_global_steps: bool = True
    distributed_average_metrics: bool = True
    distributed_gradient_sync: tp.Literal["manual", "ddp"] = "manual"
    ddp_bucket_cap_mb: float = 25.0
    fail_on_nonfinite: bool = True
    nonfinite_check_model_every_updates: int = 0
    nonfinite_check_rollout_every_local_steps: int = 0

    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])
    # Note: this is in env steps (multiples of online_parallel_envs)
    eval_every_steps: int = 1_000_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    infra: dict = pydantic.Field(default_factory=dict)

    def model_post_init(self, context):
        # TODO prioritization needs tracking eval to work, but this is bit hacky to check for it
        if self.load_expert_data_from_motion_lib and not isinstance(self.env, HumanoidVerseMjlabConfig):
            raise ValueError("Loading expert data from motion library is only supported for HumanoidVerseMjlabConfig")
        if self.ddp_bucket_cap_mb <= 0:
            raise ValueError("ddp_bucket_cap_mb must be positive")
        if self.checkpoint_every_steps <= 0 or self.eval_every_steps <= 0:
            raise ValueError("checkpoint and fallback evaluation cadences must be positive")
        evaluation_configs = (
            self.evaluations.values()
            if isinstance(self.evaluations, dict)
            else self.evaluations
        )
        for evaluation in evaluation_configs:
            if evaluation.every_steps is not None and evaluation.every_steps <= 0:
                raise ValueError(
                    f"evaluation {evaluation.name_in_logs!r} cadence must be positive"
                )
        if self.prior_plane_envs < 0:
            raise ValueError("prior_plane_envs must be non-negative")
        if self.prior_buffer_size < 0:
            raise ValueError("prior_buffer_size must be non-negative")
        if self.prior_plane_envs > 0:
            if not isinstance(self.env, HumanoidVerseMjlabConfig):
                raise ValueError("canonical-plane prior collection requires HumanoidVerseMjlabConfig")
            if self.tags.get("agent") != "fb_depth":
                raise ValueError("canonical-plane prior collection is currently defined only for fb_depth")
            if not self.use_trajectory_buffer:
                raise ValueError("canonical-plane direct-depth collection requires trajectory replay")
            if self.prior_buffer_size < self.prior_plane_envs * 2:
                raise ValueError("prior_buffer_size must hold at least two steps per plane environment")
        if bool(getattr(self.agent.model, "heading_context_enabled", False)):
            if self.tags.get("agent") != "fb_depth":
                raise ValueError("BehaviorContext heading is currently defined only for fb_depth")
            if not self.use_trajectory_buffer:
                raise ValueError("BehaviorContext heading requires trajectory replay")
        if self.distributed_gradient_sync == "ddp" and not self.distributed_sync:
            raise ValueError("distributed_gradient_sync='ddp' requires distributed_sync=True")

        if self.prioritization:
            has_prioritization_eval = False
            for eval_type in self.evaluations:
                if isinstance(eval_type, HumanoidVerseMjlabTrackingEvaluationConfig):
                    has_prioritization_eval = True
                    break
            if not has_prioritization_eval:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")


        if self.motions is None or self.motions_root is None:
            if self.prioritization:
                raise ValueError("Prioritization requires expert data to be provided (motions and motions_root)")

        # Ensure all evaluations have unique log names
        if isinstance(self.evaluations, list):
            log_names = set()
            for eval_cfg in self.evaluations:
                if eval_cfg.name_in_logs in log_names:
                    raise ValueError(
                        f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                    )
                log_names.add(eval_cfg.name_in_logs)

    def build(self):
        from pbfm.training.workspace import Workspace

        return Workspace(self)


def _global_step_scale(cfg: TrainConfig) -> int:
    if cfg.distributed_sync and cfg.distributed_global_steps and int(cfg.distributed_world_size) > 1:
        return int(cfg.distributed_world_size)
    return 1


def _effective_batch_size(cfg: TrainConfig) -> int:
    return int(cfg.agent.train.batch_size) * _global_step_scale(cfg)


def _evaluation_interval_steps(evaluation, *, fallback: int) -> int:
    configured = getattr(evaluation.cfg, "every_steps", None)
    interval = int(fallback if configured is None else configured)
    if interval <= 0:
        raise ValueError(f"evaluation cadence must be positive, got {interval}")
    return interval


def _num_envs_per_rank(cfg: TrainConfig) -> int:
    return int(cfg.online_parallel_envs)


def _global_parallel_envs(cfg: TrainConfig) -> int:
    return _num_envs_per_rank(cfg) * _global_step_scale(cfg)


def _replay_capacity_per_rank(cfg: TrainConfig) -> int:
    return int(cfg.buffer_size)


def _effective_replay_capacity(cfg: TrainConfig) -> int:
    return _replay_capacity_per_rank(cfg) * _global_step_scale(cfg)


def _trajectory_steps_per_rank(cfg: TrainConfig) -> int:
    if cfg.use_trajectory_buffer:
        return int(cfg.buffer_size) // max(_num_envs_per_rank(cfg), 1)
    return int(cfg.buffer_size)


def _distributed_loss_mode(cfg: TrainConfig) -> str:
    if cfg.distributed_sync and int(cfg.distributed_world_size) > 1:
        return "local_loss_average"
    return "single_rank"
