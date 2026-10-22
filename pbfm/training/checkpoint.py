"""Checkpoint loading, replay shard paths, and serialized training status."""

import json
import typing as tp
from pathlib import Path
from typing import Dict

from pbfm.perception.instinct_direct_depth import DEPTH_CONVENTION
from pbfm.training import config
from pbfm.training.distributed import barrier, module_sync_report

CHECKPOINT_DIR_NAME = "checkpoint"


def create_agent_or_load_checkpoint(work_dir: Path, cfg: config.TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    train_status_path = checkpoint_dir / "train_status.json"
    checkpoint_status = _initial_train_status(cfg)
    if train_status_path.exists():
        with train_status_path.open("r") as f:
            train_status = json.load(f)
        checkpoint_status = _normalize_train_status(train_status, cfg)

        print(
            f"Loading the agent at local_time={checkpoint_status['local_time']} "
            f"global_time={checkpoint_status['global_time']} optimizer_steps={checkpoint_status['optimizer_steps']}"
        )
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, cfg, checkpoint_status


def _depth_observation_contract(cfg: config.TrainConfig) -> dict[str, tp.Any] | None:
    """Return the serialized direct-depth image contract for safe resume."""
    actor = getattr(getattr(cfg.agent, "model", None), "archi", None)
    actor = getattr(actor, "actor", None)
    if getattr(actor, "name", None) != "direct_depth":
        return None
    from pbfm.perception.instinct_direct_depth import depth_config_for_profile, MAXCROP_PROFILE
    profile = getattr(getattr(cfg, "env", None), "direct_depth_profile", "legacy_rp1")
    direct = depth_config_for_profile(profile)
    contract = {
        "depth_convention": DEPTH_CONVENTION,
        "source_shape": [direct.height, direct.width],
        "crop": list(direct.crop),
        "output_shape": [direct.output_height, direct.output_width],
        "sampled_ages": list(direct.sampled_ages),
    }
    if profile == MAXCROP_PROFILE:
        contract.update(processing_profile=profile, depth_quantity="optical_z",
                        invalid_depth_m=2.5, clipping="max", gaussian_sigma=1.,
                        image_randomization=False, installation_resample="reset",
                        position_error_m=0.01, rpy_error_deg=[2., 10., 2.],
                        installation_frame="camera_link", delay_frames=[0, 1])
    elif profile == "rp1_pufo":
        contract.update(processing_profile=profile, depth_quantity="optical_z",
                        invalid_depth_m=2.5, clipping="max", normalization_range_m=[.1, 2.5],
                        gaussian_sigma=1., image_randomization=True,
                        image_noise="roboparty_train_ed0b933", installation_resample="startup",
                        position_error_m=.02, rpy_error_deg=[3., 10., 3.],
                        installation_frame="camera_link", position_frame="parent", delay_frames=[0, 1])
    return contract


def _validate_checkpoint_depth_observation_contract(
    train_status: dict[str, tp.Any], cfg: config.TrainConfig
) -> None:
    expected = _depth_observation_contract(cfg)
    if expected is None:
        return
    actual = train_status.get("depth_observation_contract")
    if actual != expected:
        raise RuntimeError(
            "Cannot resume a direct-depth checkpoint with a different or missing image convention: "
            f"checkpoint={actual!r}, requested={expected!r}. Start a fresh work_dir; legacy vertically "
            "flipped checkpoints and replay buffers are intentionally incompatible with optical_top_down_v1."
        )


def _initial_train_status(cfg: config.TrainConfig) -> dict[str, tp.Any]:
    return {
        "local_time": 0,
        "global_time": 0,
        "optimizer_steps": 0,
        "world_size": int(cfg.distributed_world_size),
        "loss_mode": config._distributed_loss_mode(cfg),
        "effective_batch_size": config._effective_batch_size(cfg),
        "depth_observation_contract": _depth_observation_contract(cfg),
    }


def _normalize_train_status(train_status: dict[str, tp.Any], cfg: config.TrainConfig) -> dict[str, tp.Any]:
    _validate_checkpoint_depth_observation_contract(train_status, cfg)
    status = _initial_train_status(cfg)
    scale = config._global_step_scale(cfg)
    current_world_size = int(cfg.distributed_world_size)
    if "world_size" not in train_status and current_world_size > 1:
        raise RuntimeError(
            "Cannot safely resume this checkpoint because checkpoint/train_status.json does not record world_size. "
            "Distributed checkpoints contain rank-local replay buffer shards; start a fresh work_dir or migrate buffers explicitly."
        )
    checkpoint_world_size = int(train_status.get("world_size", current_world_size))
    if checkpoint_world_size != current_world_size:
        raise RuntimeError(
            "Cannot resume checkpoint with a different distributed world_size: "
            f"checkpoint world_size={checkpoint_world_size}, current world_size={current_world_size}. "
            "Rank-local replay buffer shards cannot be automatically migrated; use a matching GPU count, "
            "start a fresh work_dir, or perform an explicit buffer migration."
        )
    if "local_time" in train_status:
        status["local_time"] = int(train_status["local_time"])
        status["global_time"] = int(train_status.get("global_time", status["local_time"] * scale))
    else:
        legacy_time = int(train_status.get("time", 0))
        status["global_time"] = legacy_time
        status["local_time"] = (legacy_time + scale - 1) // scale
    status["optimizer_steps"] = int(train_status.get("optimizer_steps", 0))
    status["world_size"] = checkpoint_world_size
    status["loss_mode"] = str(train_status.get("loss_mode", config._distributed_loss_mode(cfg)))
    status["effective_batch_size"] = int(train_status.get("effective_batch_size", config._effective_batch_size(cfg)))
    return status


def _make_train_status(cfg: config.TrainConfig, *, local_time: int, global_time: int, optimizer_steps: int) -> dict[str, tp.Any]:
    return {
        "time": int(global_time),
        "local_time": int(local_time),
        "global_time": int(global_time),
        "optimizer_steps": int(optimizer_steps),
        "world_size": int(cfg.distributed_world_size),
        "loss_mode": config._distributed_loss_mode(cfg),
        "effective_batch_size": config._effective_batch_size(cfg),
        "depth_observation_contract": _depth_observation_contract(cfg),
    }


def checkpoint_buffer_path(
    checkpoint_dir: Path, cfg: config.TrainConfig, *, rank: int, world_size: int, name: str = "train"
) -> Path:
    if name not in {"train", "prior"}:
        raise ValueError(f"Unsupported replay buffer name: {name!r}")
    if cfg.checkpoint_rank_buffers and world_size > 1:
        return checkpoint_dir / "buffers" / f"{name}_rank_{rank}"
    return checkpoint_dir / "buffers" / name


def save_checkpoint(
    work_dir: Path,
    cfg: config.TrainConfig,
    agent,
    *,
    write_shared_artifacts: bool,
    rank: int,
    world_size: int,
    local_time: int,
    global_time: int,
    optimizer_steps: int,
    replay_buffer: Dict[str, tp.Any],
) -> None:
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    sync_report = None
    if cfg.distributed_sync:
        barrier()
        sync_report = module_sync_report(agent._model, src=0)
        if sync_report["max_abs_diff_from_rank0"] > 1.0e-5:
            raise RuntimeError(f"Distributed model state diverged before checkpoint: {sync_report}")
    if write_shared_artifacts:
        print(f"Checkpointing at local_time={local_time} global_time={global_time} optimizer_steps={optimizer_steps}")
        agent.save(str(checkpoint_dir))
    if cfg.checkpoint_buffer:
        replay_buffer["train"].save(checkpoint_buffer_path(checkpoint_dir, cfg, rank=rank, world_size=world_size, name="train"))
        if "prior" in replay_buffer:
            replay_buffer["prior"].save(checkpoint_buffer_path(checkpoint_dir, cfg, rank=rank, world_size=world_size, name="prior"))
    if cfg.distributed_sync:
        barrier()
    if write_shared_artifacts:
        if sync_report is not None:
            with (checkpoint_dir / "distributed_sync.json").open("w+") as f:
                json.dump(sync_report, f, indent=4)
        with (checkpoint_dir / "train_status.json").open("w+") as f:
            json.dump(
                _make_train_status(
                    cfg,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=optimizer_steps,
                ),
                f,
                indent=4,
            )
    if cfg.distributed_sync:
        barrier()
