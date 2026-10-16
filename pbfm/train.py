"""Train the final P3 fb_depth configuration with explicit runtime overrides."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _ensure_compile_cache(cache_root: str | Path | None = None) -> None:
    cache_dir = os.environ.get("UFO_CACHE_DIR") or os.environ.get("BFMZERO_MJLAB_CACHE_DIR")
    root = Path(cache_dir or cache_root or Path.cwd() / "cache").expanduser()
    for key, subdir in {
        "TMPDIR": "tmp",
        "TEMP": "tmp",
        "TMP": "tmp",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
        "WARP_CACHE_PATH": "warp",
    }.items():
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))


_ensure_compile_cache()

from pbfm.training.evaluation.same_z_terrain import SameZTerrainEvaluationConfig
from pbfm.agents.presets import build_fb_depth_agent
from pbfm.agents.presets.fb_depth import load_final_config
from pbfm.terrains.coverage import validate_motion_terrain_coverage
from pbfm.terrains.rp1_simple import RP1_PATCH_SIZE, RP1_TERRAIN_BORDER_WIDTH
from pbfm.training.config import TrainConfig
from pbfm.motion.data import prepare_motion_manifest
from pbfm.robots import assert_robot_configs_compatible, load_robot_training_spec, resolve_robot_config_path
from pbfm.utils.resource_paths import CONFIG_ROOT, remap_resource_path


_FINAL_DEFAULTS = load_final_config()
DEFAULT_AGENT = "fb_depth"
DEFAULT_NUM_ENVS = _FINAL_DEFAULTS["online_parallel_envs"]
DEFAULT_NUM_ENV_STEPS = _FINAL_DEFAULTS["num_env_steps"]
DEFAULT_CHECKPOINT_EVERY_STEPS = _FINAL_DEFAULTS["checkpoint_every_steps"]
DEFAULT_TRACKING_EVAL_EVERY_STEPS = _FINAL_DEFAULTS["evaluations"][0]["every_steps"]
DEFAULT_SAME_Z_EVAL_EVERY_STEPS = 0
DEFAULT_DATA_PATH = str(remap_resource_path(_FINAL_DEFAULTS["env"]["lafan_tail_path"]))
# Archived output paths are evidence, never defaults for a new run.
DEFAULT_WORK_DIR = "runs/pbfm_fb_depth"
DEFAULT_BUFFER_SIZE = _FINAL_DEFAULTS["buffer_size"]
DEFAULT_FB_UPDATE_Z_EVERY_STEP = _FINAL_DEFAULTS["agent"]["train"]["update_z_every_step"]
DEFAULT_UPDATE_Z_EVERY_STEP = DEFAULT_FB_UPDATE_Z_EVERY_STEP
DEFAULT_WANDB_PROJECT = _FINAL_DEFAULTS["wandb_pname"]
DEFAULT_ROBOT_CONFIG = str(remap_resource_path(_FINAL_DEFAULTS["env"]["robot_config_path"]))
DEFAULT_SEED = _FINAL_DEFAULTS["seed"]


def _resolve_training_robot_config(
    cli_robot_config: str | Path | None,
    manifest_robot_config: str | Path | None,
) -> Path:
    if cli_robot_config is not None and manifest_robot_config is not None:
        return assert_robot_configs_compatible(cli_robot_config, manifest_robot_config)
    if cli_robot_config is not None:
        return resolve_robot_config_path(cli_robot_config)
    if manifest_robot_config is not None:
        return resolve_robot_config_path(manifest_robot_config)
    return resolve_robot_config_path(DEFAULT_ROBOT_CONFIG)


def canonical_agent_name(agent: str) -> str:
    if agent != DEFAULT_AGENT:
        raise ValueError(f"Unsupported agent preset: {agent}. Supported presets: {DEFAULT_AGENT}")
    return agent


def _default_update_z_every_step(agent: str) -> int:
    canonical_agent_name(agent)
    return DEFAULT_FB_UPDATE_Z_EVERY_STEP


def build_ufo_mjlab_config(
    *,
    device: str,
    work_dir: str = DEFAULT_WORK_DIR,
    num_envs: int = DEFAULT_NUM_ENVS,
    num_env_steps: int = DEFAULT_NUM_ENV_STEPS,
    seed: int = DEFAULT_SEED,
    use_wandb: bool = False,
    wandb_run_name: str | None = None,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    checkpoint_every_steps: int = DEFAULT_CHECKPOINT_EVERY_STEPS,
    tracking_eval_every_steps: int = DEFAULT_TRACKING_EVAL_EVERY_STEPS,
    same_z_eval_every_steps: int = DEFAULT_SAME_Z_EVAL_EVERY_STEPS,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    disable_eval_prioritization: bool = False,
    smoke: bool = False,
    agent: str = DEFAULT_AGENT,
    data_path: str | list[str] | None = None,
    data_mix_weights: list[float] | None = None,
    update_z_every_step: int | None = None,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    prior_plane_envs: int | None = None,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    lr_scale: float = 1.0,
    clip_grad_norm: float | None = None,
    resume_replay_warmup_steps: int = _FINAL_DEFAULTS["resume_replay_warmup_steps"],
    cartwheel_aux_safe: bool = False,
    heading_context: bool = True,
    heading_reg_coeff: float | None = None,
    aux_reward_normalizer_freeze_global_steps: int | None = None,
    behavior_prior: bool = True,
    num_agent_updates: int | None = None,
    robot_config: str | Path | None = None,
    terrain_mode: str | None = None,
    cache_expert_buffer: bool = _FINAL_DEFAULTS["cache_expert_buffer"],
    rebuild_expert_buffer_cache: bool = False,
    gradient_sync: str = "auto",
    ddp_bucket_cap_mb: float = _FINAL_DEFAULTS["ddp_bucket_cap_mb"],
) -> TrainConfig:
    """Apply explicit overrides to a fresh copy of the archived final config."""
    canonical_agent_name(agent)
    raw = load_final_config()
    if aux_reward_normalizer_freeze_global_steps is not None:
        if aux_reward_normalizer_freeze_global_steps < 0:
            raise ValueError("aux_reward_normalizer_freeze_global_steps must be non-negative")
    if not behavior_prior and prior_plane_envs not in (None, 0):
        raise ValueError("behavior_prior=False requires prior_plane_envs=0")
    if prior_plane_envs is None:
        prior_plane_envs = raw["prior_plane_envs"] if behavior_prior else 0
    prior_plane_envs = int(prior_plane_envs)
    if prior_plane_envs < 0:
        raise ValueError("prior_plane_envs must be non-negative")
    for value_name, value in (
        ("num_envs", num_envs),
        ("num_env_steps", num_env_steps),
        ("buffer_size", buffer_size),
        ("checkpoint_every_steps", checkpoint_every_steps),
        ("tracking_eval_every_steps", tracking_eval_every_steps),
    ):
        if int(value) <= 0:
            raise ValueError(f"{value_name} must be positive")
    if resume_replay_warmup_steps < 0:
        raise ValueError("resume_replay_warmup_steps must be non-negative")
    if smoke and prior_plane_envs > 0:
        prior_plane_envs = min(prior_plane_envs, max(2, num_envs // 8))
    distributed_sync = distributed_world_size > 1
    if gradient_sync == "auto":
        gradient_sync = "ddp" if distributed_sync else "manual"
    if gradient_sync not in {"manual", "ddp"}:
        raise ValueError(f"Unsupported gradient synchronization mode: {gradient_sync!r}")
    if gradient_sync == "ddp" and not distributed_sync:
        raise ValueError("gradient_sync='ddp' requires more than one distributed rank")

    agent_device = "cuda" if device.startswith("cuda") else "cpu"
    agent_cfg = build_fb_depth_agent(
        device=agent_device,
        compile=raw["agent"]["compile"],
        update_z_every_step=update_z_every_step,
        lr_scale=lr_scale,
        clip_grad_norm=clip_grad_norm,
        cartwheel_aux_safe=cartwheel_aux_safe,
        heading_context=heading_context,
        heading_reg_coeff=heading_reg_coeff,
        behavior_prior=behavior_prior,
    )
    if aux_reward_normalizer_freeze_global_steps is not None:
        agent_cfg = agent_cfg.model_copy(
            update={"train": agent_cfg.train.model_copy(
                update={"aux_reward_normalizer_freeze_global_steps":
                        int(aux_reward_normalizer_freeze_global_steps)}
            )}
        )

    env = raw["env"]
    # Relocate resource paths without reconstructing the final robot metadata.
    for key in ("mjcf_path", "robot_config_path"):
        env[key] = str(remap_resource_path(env[key]))
    robot_training = env["robot_training"]
    robot_training["config_path"] = env["robot_config_path"]
    robot_training["robot"]["xml_path"] = env["mjcf_path"]
    env["lafan_tail_path"] = data_path or DEFAULT_DATA_PATH
    if data_mix_weights is not None or data_path is not None:
        env["data_mix_weights"] = data_mix_weights
    env.update(device=device, seed=seed, disable_obs_noise=disable_obs_noise,
               disable_domain_randomization=disable_dr)

    overrides = {"terrain.seed": str(seed)}
    if robot_config is not None:
        resolved_robot = resolve_robot_config_path(robot_config)
        if resolved_robot != resolve_robot_config_path(DEFAULT_ROBOT_CONFIG):
            spec = load_robot_training_spec(resolved_robot)
            env.update(mjcf_path=spec.robot.xml_path,
                       robot_config_path=str(spec.config_path),
                       robot_training=spec.to_env_dict())
            overrides.update({
                "robot": spec.hydra_robot,
                "robot.control.action_scale": str(spec.action_scale),
                "robot.control.action_clip_value": str(spec.action_clip_value),
                "robot.control.normalize_action_to": str(spec.normalize_action_to),
            })
            overrides.update(value.split("=", 1) for value in spec.hydra_overrides)
            metadata = OmegaConf.load(spec.config_path).get("metadata") or {}
            if metadata.get("review_status") == "draft":
                print("WARNING: Robot config is auto-generated draft; review training semantics before use.",
                      flush=True)
    if terrain_mode is not None:
        overrides["terrain.terrain_type"] = terrain_mode
    if cartwheel_aux_safe:
        overrides.update({
            "rewards.reward_scales.penalty_undesired_contact": "0.0",
            "rewards.reward_scales.penalty_feet_ori": "0.0",
            "rewards.reward_scales.feet_heading_alignment": "0.0",
            "rewards.reward_scales.penalty_slippage": "0.0",
            "rewards.reward_scales.penalty_ankle_roll": "0.0",
            "rewards.reward_scales.penalty_action_rate": "-0.1",
        })
    hydra_overrides = dict(value.split("=", 1) for value in env["hydra_overrides"])
    hydra_overrides.update(overrides)
    env["hydra_overrides"] = [f"{key}={value}" for key, value in hydra_overrides.items()]

    if smoke:
        raw.update(log_every_updates=1024, update_agent_every=1024,
                   num_seed_steps=1024, num_agent_updates=1, checkpoint_buffer=False)
    if num_agent_updates is not None:
        if num_agent_updates <= 0:
            raise ValueError("num_agent_updates must be positive")
        raw["num_agent_updates"] = int(num_agent_updates)
    run_eval_and_prioritization = not smoke and not disable_eval_prioritization
    if run_eval_and_prioritization:
        for evaluation in raw["evaluations"]:
            if evaluation["name"] == "HumanoidVerseMjlabTrackingEvaluationConfig":
                evaluation.update(every_steps=int(tracking_eval_every_steps), num_envs=num_envs)
        if int(same_z_eval_every_steps) > 0:
            raw["evaluations"].append(SameZTerrainEvaluationConfig(
                name="SameZTerrainEvaluationConfig",
                generate_videos=False,
                name_in_logs="same_z_terrain_eval",
                every_steps=int(same_z_eval_every_steps),
                seed=seed,
            ).model_dump())
    else:
        raw["evaluations"] = []
    main_buffer_size = min(int(buffer_size), 4096) if smoke else int(buffer_size)
    replay_time_slots = max(main_buffer_size // num_envs, 2)
    cache_root = os.environ.get("UFO_CACHE_DIR") or os.environ.get("BFMZERO_MJLAB_CACHE_DIR")
    raw.update(
        agent=agent_cfg,
        work_dir=work_dir,
        seed=seed,
        online_parallel_envs=num_envs,
        prior_plane_envs=prior_plane_envs,
        num_env_steps=num_env_steps,
        resume_replay_warmup_steps=int(resume_replay_warmup_steps),
        checkpoint_every_steps=checkpoint_every_steps,
        prioritization=raw["prioritization"] and run_eval_and_prioritization,
        buffer_size=main_buffer_size,
        prior_buffer_size=prior_plane_envs * replay_time_slots,
        use_wandb=use_wandb,
        wandb_ename=os.environ.get("WANDB_ENTITY"),
        wandb_pname=wandb_project or raw["wandb_pname"],
        wandb_run_name=wandb_run_name or "pbfm_fb_depth",
        buffer_device=agent_device,
        cache_expert_buffer=bool(cache_expert_buffer),
        rebuild_expert_buffer_cache=bool(rebuild_expert_buffer_cache),
        expert_buffer_cache_root=str(Path(cache_root or Path.cwd() / "cache").expanduser()),
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        distributed_sync=distributed_sync,
        distributed_gradient_sync=gradient_sync,
        ddp_bucket_cap_mb=float(ddp_bucket_cap_mb),
    )
    raw["tags"].update(distributed_rank=distributed_rank,
                       distributed_world_size=distributed_world_size)
    return TrainConfig.model_validate(raw)


def _select_device_and_rank(seed: int) -> tuple[str, int, int, int]:
    if "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        return f"cuda:{local_rank}", local_rank, rank, world_size

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        try:
            import torch

            if torch.cuda.is_available():
                os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
                return "cuda:0", 0, 0, 1
        except Exception:
            pass
        return "cpu", 0, 0, 1
    local_rank = 0
    rank = 0
    world_size = 1
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    return f"cuda:{local_rank}", local_rank, rank, world_size


def _init_distributed(local_rank: int, world_size: int) -> None:
    if world_size <= 1:
        return
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        init_kwargs = {
            "backend": "nccl",
            "init_method": "env://",
            "timeout": timedelta(hours=2),
        }
        try:
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(**init_kwargs)
        except TypeError:
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)


def run_train(args: argparse.Namespace, log_dir: Path) -> None:
    device, _local_rank, rank, world_size = _select_device_and_rank(args.seed)
    _init_distributed(_local_rank, world_size)
    seed = args.seed + rank
    cfg = build_ufo_mjlab_config(
        device=device,
        work_dir=str(log_dir),
        num_envs=args.num_envs,
        num_env_steps=args.num_env_steps,
        seed=seed,
        use_wandb=bool(args.use_wandb and rank == 0),
        wandb_run_name=args.wandb_run_name,
        wandb_project=args.wandb_project,
        checkpoint_every_steps=args.checkpoint_every_steps,
        tracking_eval_every_steps=args.tracking_eval_every_steps,
        same_z_eval_every_steps=args.same_z_eval_every_steps,
        distributed_rank=rank,
        distributed_world_size=world_size,
        disable_eval_prioritization=bool(args.disable_eval_prioritization),
        smoke=bool(args.smoke),
        agent=args.agent,
        data_path=args.data_path,
        data_mix_weights=args.data_mix_weights,
        update_z_every_step=args.update_z_every_step,
        buffer_size=args.buffer_size,
        prior_plane_envs=args.prior_plane_envs,
        disable_dr=bool(args.disable_dr),
        disable_obs_noise=bool(args.disable_obs_noise),
        lr_scale=args.lr_scale,
        clip_grad_norm=args.clip_grad_norm,
        resume_replay_warmup_steps=args.resume_replay_warmup_steps,
        cartwheel_aux_safe=bool(args.cartwheel_aux_safe),
        heading_context=bool(args.heading_context),
        heading_reg_coeff=float(args.heading_reg_coeff),
        aux_reward_normalizer_freeze_global_steps=int(
            args.aux_reward_normalizer_freeze_global_steps
        ),
        behavior_prior=bool(args.behavior_prior),
        num_agent_updates=args.num_agent_updates,
        robot_config=args.robot_config,
        terrain_mode=args.terrain_mode,
        cache_expert_buffer=bool(args.expert_buffer_cache),
        rebuild_expert_buffer_cache=bool(args.rebuild_expert_buffer_cache),
        gradient_sync=args.gradient_sync,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
    )
    behavior_prior_enabled = bool(
        getattr(cfg.agent.train, "behavior_prior_enabled", True)
    )
    behavior_prior_source = (
        "disabled"
        if not behavior_prior_enabled
        else ("plane" if cfg.prior_plane_envs > 0 else "main")
    )
    print(
        "[INFO] PBFM train: "
        f"agent={args.agent}, device={device}, rank={rank}/{world_size}, seed={seed}, work_dir={log_dir}, "
        f"robot_config={cfg.env.robot_config_path}, mjcf_path={cfg.env.mjcf_path}, "
        f"data_path={cfg.env.lafan_tail_path}, data_mix_weights={cfg.env.data_mix_weights}, "
        f"num_envs_per_rank={args.num_envs}, global_parallel_envs={args.num_envs * world_size}, "
        f"prior_plane_envs_per_rank={cfg.prior_plane_envs}, prior_buffer_size_per_rank={cfg.prior_buffer_size}, "
        f"num_env_steps_global={args.num_env_steps}, buffer_size_per_rank={cfg.buffer_size}, "
        f"num_agent_updates={cfg.num_agent_updates}, update_agent_every_local={cfg.update_agent_every}, "
        f"cartwheel_aux_safe={args.cartwheel_aux_safe}, lr_scale={args.lr_scale}, clip_grad_norm={args.clip_grad_norm}, "
        f"heading_context={cfg.agent.model.heading_context_enabled}, "
        f"heading_reg_coeff={getattr(cfg.agent.train, 'reg_coeff_heading', 0.0):g}, "
        f"aux_reward_normalizer_freeze_global_steps="
        f"{getattr(cfg.agent.train, 'aux_reward_normalizer_freeze_global_steps', 0)}, "
        f"behavior_prior_enabled={behavior_prior_enabled}, "
        f"behavior_prior_source={behavior_prior_source}, "
        f"discriminator_loss={getattr(cfg.agent.train, 'discriminator_loss', 'bce')}, "
        f"discriminator_reward={getattr(cfg.agent.train, 'discriminator_reward', 'log_odds')}, "
        f"resume_replay_warmup_steps={cfg.resume_replay_warmup_steps}, "
        f"disable_dr={cfg.env.disable_domain_randomization}, disable_obs_noise={cfg.env.disable_obs_noise}, "
        f"terrain_mode={args.terrain_mode}, compile={cfg.agent.compile}, "
        f"expert_buffer_cache={cfg.cache_expert_buffer}, gradient_sync={cfg.distributed_gradient_sync}, "
        f"ddp_bucket_cap_mb={cfg.ddp_bucket_cap_mb:g}",
        flush=True,
    )
    training_completed = False
    try:
        workspace = cfg.build()
        workspace.train()
        training_completed = True
    finally:
        if world_size > 1:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                if training_completed:
                    dist.destroy_process_group()
                elif rank == 0:
                    # A rank-local exception can leave peers inside a collective.
                    # Calling destroy_process_group() from only the failed rank
                    # then hides the original traceback behind a second deadlock.
                    print(
                        "[WARNING] Skipping coordinated process-group teardown after a training failure; "
                        "the distributed launcher will terminate peer ranks.",
                        file=sys.stderr,
                        flush=True,
                    )


def launch(args: argparse.Namespace) -> None:
    log_dir = Path(args.work_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    _ensure_compile_cache()
    if args.terrain_mode != "plane":
        terrain_cfg = OmegaConf.load(CONFIG_ROOT / "simulation/terrain/terrain_ufo_v0.yaml").terrain
        patch_size = (
            (RP1_PATCH_SIZE, RP1_PATCH_SIZE)
            if args.terrain_mode == "rp1_simple"
            else tuple(float(value) for value in terrain_cfg.patch_size)
        )
        border_width = (
            RP1_TERRAIN_BORDER_WIDTH
            if args.terrain_mode == "rp1_simple"
            else float(terrain_cfg.border_width)
        )
        sensor_radius = math.hypot(
            max(abs(float(terrain_cfg.terrain_priv.x_min)), abs(float(terrain_cfg.terrain_priv.x_max))),
            max(abs(float(terrain_cfg.terrain_priv.y_min)), abs(float(terrain_cfg.terrain_priv.y_max))),
        )
        report = validate_motion_terrain_coverage(
            args.data_path or DEFAULT_DATA_PATH,
            patch_size=patch_size,
            sensor_radius=sensor_radius,
            policy_margin=float(terrain_cfg.coverage.policy_margin),
            # Tiles are traversable. The worst assigned origin is at the
            # center of an outer tile, with the global border beyond it.
            safe_radius=(
                min(patch_size) / 2.0 + border_width
            ),
        )
        print(
            "[INFO] terrain coverage preflight passed: "
            f"max_excursion={report.max_excursion:.3f}m motion={report.motion_key!r}, "
            f"sensor_radius={report.sensor_radius:.3f}m, policy_margin={report.policy_margin:.3f}m, "
            f"required={report.required_radius:.3f}m < connected_safe_radius={report.patch_safe_radius:.3f}m",
            flush=True,
        )
    if args.gpu_ids in (None, "single"):
        run_train(args, log_dir)
        return

    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu_ids == "all":
        import torch

        num_gpus = torch.cuda.device_count()
        selected_gpus = None
    else:
        requested = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
        if existing_visible:
            visible = [x.strip() for x in existing_visible.split(",") if x.strip()]
            selected_gpus = [visible[i] for i in requested]
        else:
            selected_gpus = [str(i) for i in requested]
        num_gpus = len(selected_gpus)
    if num_gpus <= 1:
        if selected_gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
        run_train(args, log_dir)
        return

    import torchrunx

    logging.basicConfig(level=logging.INFO)
    if selected_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
    os.environ.setdefault("TORCHRUNX_LOG_DIR", str(log_dir / "torchrunx"))
    torchrunx.Launcher(
        hostnames=["localhost"],
        workers_per_host=num_gpus,
        backend=None,
        copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
        + (
            "MUJOCO*",
            "UFO_CACHE_DIR",
            "UFO_DATA_DIR",
            "BFMZERO_MJLAB_CACHE_DIR",
            "UV_CACHE_DIR",
            "PYTHONPYCACHEPREFIX",
            "TMPDIR",
            "TEMP",
            "TMP",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "WARP_CACHE_PATH",
        ),
    ).run(run_train, args, log_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final P3 fb_depth configuration.")
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        choices=[DEFAULT_AGENT],
        help="The historical large-network agent; defaults come from pbfm/configs/legacy_large.json.",
    )
    parser.add_argument(
        "--terrain-mode",
        choices=["plane", "flat", "slope", "stairs", "stairs_up", "stairs_down", "rough", "platforms", "mixed", "rp1_simple"],
        default=None,
        help=(
            "Physical terrain (default: rp1_simple). "
            "Use plane for an all-flat control with terrain observations preserved."
        ),
    )
    parser.add_argument("--gpu-ids", default="single", help="'single', 'all', or a comma-separated GPU id list relative to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help=(
            "G1 robot YAML used for training metadata. Defaults to configs/robots/g1_29dof.yaml. "
            "If omitted and --data-manifest declares robot_config, the manifest robot config is used."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num-env-steps", type=int, default=DEFAULT_NUM_ENV_STEPS)
    parser.add_argument("--checkpoint-every-steps", type=int, default=DEFAULT_CHECKPOINT_EVERY_STEPS)
    parser.add_argument(
        "--tracking-eval-every-steps",
        type=int,
        default=DEFAULT_TRACKING_EVAL_EVERY_STEPS,
        help="Flat tracking/EMD evaluation cadence in global environment steps.",
    )
    parser.add_argument(
        "--same-z-eval-every-steps",
        type=int,
        default=DEFAULT_SAME_Z_EVAL_EVERY_STEPS,
        help="Same-z terrain evaluation cadence in global environment steps. Set <=0 to disable.",
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        default=None,
        help="One or more motion data pickle files. Multiple files require --data-mix-weights to fix source ratios.",
    )
    parser.add_argument(
        "--data-mix-weights",
        type=float,
        nargs="+",
        default=None,
        help="Source-level sampling weights for multiple --data-path entries, e.g. 0.95 0.05.",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=None,
        help="YAML manifest describing weighted motion data sources. Cannot be combined with --data-path.",
    )
    parser.add_argument(
        "--rebuild-motion-cache",
        action="store_true",
        help="Rebuild manifest-generated motion pkl caches instead of reusing existing cache files.",
    )
    parser.add_argument(
        "--update-z-every-step",
        type=int,
        default=None,
        help=f"Override latent update interval. Defaults to {DEFAULT_UPDATE_Z_EVERY_STEP}.",
    )
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="Replay capacity per rank/GPU.")
    parser.add_argument(
        "--prior-plane-envs",
        type=int,
        default=None,
        help=(
            "Optional canonical-plane policy environments per rank for the fb_depth D/Q_D stream. "
            "The default 0 uses original-UFO main-terrain replay; positive values select the "
            "canonical-plane ablation."
        ),
    )
    parser.add_argument(
        "--behavior-prior",
        action=argparse.BooleanOptionalAction,
        default=_FINAL_DEFAULTS["agent"]["train"]["behavior_prior_enabled"],
        help=(
            "Enable the discriminator/Q_D behavior-prior branch. "
            "With --prior-plane-envs 0 it uses main-terrain replay; use "
            "--no-behavior-prior for the formal no-D ablation."
        ),
    )
    parser.add_argument(
        "--expert-buffer-cache",
        action=argparse.BooleanOptionalAction,
        default=_FINAL_DEFAULTS["cache_expert_buffer"],
        help="Reuse a content-addressed expert trajectory and full-FK cache.",
    )
    parser.add_argument(
        "--rebuild-expert-buffer-cache",
        action="store_true",
        help="Force rank 0 to atomically rebuild the expert/full-FK cache.",
    )
    parser.add_argument(
        "--gradient-sync",
        choices=["auto", "ddp", "manual"],
        default="auto",
        help="Distributed gradient synchronization; auto selects DDP for multiple ranks.",
    )
    parser.add_argument(
        "--ddp-bucket-cap-mb",
        type=float,
        default=_FINAL_DEFAULTS["ddp_bucket_cap_mb"],
        help="DDP gradient bucket capacity in MiB.",
    )
    parser.add_argument(
        "--num-agent-updates",
        type=int,
        default=None,
        help=(
            "Override optimizer updates per update trigger. For fair env-scaling ablations, use 32 with "
            "2048 envs/GPU and 64 with 4096 envs/GPU to match the 1024 envs/GPU update density."
        ),
    )
    parser.add_argument("--disable-dr", action="store_true", help="Disable domain randomization for training.")
    parser.add_argument("--disable-obs-noise", action="store_true", help="Disable observation noise for training.")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Scale FB learning rates.")
    parser.add_argument("--clip-grad-norm", type=float, default=_FINAL_DEFAULTS["agent"]["train"]["clip_grad_norm"], help="Enable FB actor/FB gradient clipping when > 0.")
    parser.add_argument(
        "--resume-replay-warmup-steps",
        type=int,
        default=_FINAL_DEFAULTS["resume_replay_warmup_steps"],
        help=(
            "Recovery-only local env steps to collect with a loaded policy before optimizer updates. "
            "Use only when a resumed checkpoint intentionally has no replay buffer."
        ),
    )
    parser.add_argument(
        "--cartwheel-aux-safe",
        action="store_true",
        help="Use a cartwheel-safe FB auxiliary reward set: remove locomotion contact/foot-shape penalties and reduce action-rate penalty.",
    )
    parser.add_argument(
        "--heading-context",
        action=argparse.BooleanOptionalAction,
        default=_FINAL_DEFAULTS["agent"]["model"]["heading_context_enabled"],
        help=(
            "Enable z-companion BehaviorContext heading observations. "
            "Required by the final P3 velocity critic."
        ),
    )
    parser.add_argument(
        "--heading-reg-coeff",
        type=float,
        default=None,
        help=(
            "Q_H Actor coefficient (default from pbfm/configs/legacy_large.json). "
            "Must be positive with the final P3 velocity critic."
        ),
    )
    parser.add_argument(
        "--aux-reward-normalizer-freeze-global-steps",
        type=int,
        default=_FINAL_DEFAULTS["agent"]["train"]["aux_reward_normalizer_freeze_global_steps"],
        help=(
            "Freeze the bias-corrected Aux reward mean/std at the first optimizer "
            "update at or after this global environment step. Zero keeps dynamic scaling."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--use-wandb", action="store_true", help="Opt in to W&B logging for this new run.")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--disable-eval-prioritization",
        action="store_true",
        help="Validation/debug only: skip tracking eval and expert prioritization without changing default training behavior.",
    )
    parser.add_argument("--smoke", action="store_true", help="Short local smoke settings: 16 envs, 2048 env steps, no W&B.")
    args = parser.parse_args()
    args.agent = canonical_agent_name(args.agent)
    if args.update_z_every_step is None:
        args.update_z_every_step = _default_update_z_every_step(args.agent)
    if args.heading_reg_coeff is None:
        args.heading_reg_coeff = _FINAL_DEFAULTS["agent"]["train"]["reg_coeff_heading"]
    if args.smoke:
        args.num_envs = min(args.num_envs, 16)
        args.num_env_steps = min(args.num_env_steps, 2048)
        args.use_wandb = False
    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        manifest_data = prepare_motion_manifest(args.data_manifest, rebuild_cache=bool(args.rebuild_motion_cache))
        args.data_path = manifest_data.train_data_paths
        args.data_mix_weights = manifest_data.train_data_weights
        manifest_robot_config = manifest_data.robot_config_path
    elif args.data_path is not None:
        data_path_count = len(args.data_path)
        if args.data_mix_weights is not None:
            if len(args.data_mix_weights) != data_path_count:
                raise ValueError("--data-mix-weights length must match --data-path length")
            if any(w < 0 for w in args.data_mix_weights) or sum(args.data_mix_weights) <= 0:
                raise ValueError("--data-mix-weights must be non-negative and sum to a positive value")
            weight_sum = float(sum(args.data_mix_weights))
            args.data_mix_weights = [float(w) / weight_sum for w in args.data_mix_weights]
        elif data_path_count > 1:
            args.data_mix_weights = [1.0 / data_path_count] * data_path_count
        if data_path_count == 1:
            args.data_path = args.data_path[0]
            args.data_mix_weights = None

    args.robot_config = _resolve_training_robot_config(args.robot_config, manifest_robot_config)

    if args.update_z_every_step <= 0:
        raise ValueError("--update-z-every-step must be positive")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive")
    if args.prior_plane_envs is not None and args.prior_plane_envs < 0:
        raise ValueError("--prior-plane-envs must be non-negative")
    if not args.behavior_prior and args.prior_plane_envs not in (None, 0):
        raise ValueError("--no-behavior-prior requires --prior-plane-envs 0")
    if args.num_agent_updates is not None and args.num_agent_updates <= 0:
        raise ValueError("--num-agent-updates must be positive")
    if args.lr_scale <= 0:
        raise ValueError("--lr-scale must be positive")
    if args.clip_grad_norm < 0:
        raise ValueError("--clip-grad-norm must be non-negative")
    if args.resume_replay_warmup_steps < 0:
        raise ValueError("--resume-replay-warmup-steps must be non-negative")
    if args.heading_reg_coeff < 0.0:
        raise ValueError("--heading-reg-coeff must be non-negative")
    if args.aux_reward_normalizer_freeze_global_steps < 0:
        raise ValueError("--aux-reward-normalizer-freeze-global-steps must be non-negative")
    if not args.heading_context or args.heading_reg_coeff <= 0.0:
        raise ValueError("The final P3 velocity critic requires --heading-context and a positive --heading-reg-coeff")
    if args.terrain_mode is None:
        args.terrain_mode = "rp1_simple"
    return args


def main() -> None:
    launch(parse_args())


if __name__ == "__main__":
    main()
