"""The training entrypoint must preserve every declared final P3 setting."""

import sys

import pytest

from pbfm.agents.presets.fb_depth import load_final_config
from pbfm.train import build_ufo_mjlab_config, parse_args
from pbfm.utils.resource_paths import remap_resource_path


def _assert_declared_fields(expected, actual, path=""):
    if isinstance(expected, dict):
        for key, value in expected.items():
            _assert_declared_fields(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, value in enumerate(expected):
            _assert_declared_fields(value, actual[index], f"{path}[{index}]")
    else:
        assert actual == expected, path


@pytest.mark.parametrize("device,world_size", [("cpu", 1), ("cuda:0", 8)])
def test_training_preserves_full_final_template(monkeypatch, tmp_path, device, world_size):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("UFO_CACHE_DIR", str(tmp_path / "cache"))
    expected = load_final_config()
    cfg = build_ufo_mjlab_config(device=device, distributed_world_size=world_size)
    agent_device = "cuda" if device.startswith("cuda") else "cpu"
    expected["agent"]["model"]["device"] = agent_device
    expected["env"]["device"] = device
    for key in ("lafan_tail_path", "mjcf_path", "robot_config_path"):
        expected["env"][key] = str(remap_resource_path(expected["env"][key]))
    expected["env"]["robot_training"]["config_path"] = expected["env"]["robot_config_path"]
    expected["env"]["robot_training"]["robot"]["xml_path"] = expected["env"]["mjcf_path"]
    expected.update(
        work_dir="runs/pbfm_fb_depth",
        use_wandb=False,
        wandb_run_name="pbfm_fb_depth",
        expert_buffer_cache_root=str(tmp_path / "cache"),
        buffer_device=agent_device,
        distributed_world_size=world_size,
        distributed_sync=world_size > 1,
        distributed_gradient_sync="ddp" if world_size > 1 else "manual",
    )
    expected["tags"]["distributed_world_size"] = world_size
    _assert_declared_fields(expected, cfg.model_dump(mode="json"))
    cfg.env.hydra_overrides.append("terrain.seed=999")
    assert "terrain.seed=999" not in build_ufo_mjlab_config(device=device).env.hydra_overrides
    assert load_final_config()["work_dir"] != cfg.work_dir


def test_cli_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_args()
    final = load_final_config()
    assert args.agent == "fb_depth"
    assert args.seed == final["seed"]
    assert args.num_envs == final["online_parallel_envs"]
    assert args.num_env_steps == final["num_env_steps"]
    assert args.checkpoint_every_steps == final["checkpoint_every_steps"]
    assert args.tracking_eval_every_steps == final["evaluations"][0]["every_steps"]
    assert args.heading_reg_coeff == final["agent"]["train"]["reg_coeff_heading"]
    assert args.update_z_every_step == final["agent"]["train"]["update_z_every_step"]
    assert args.buffer_size == final["buffer_size"]
    assert not args.use_wandb
    cfg = build_ufo_mjlab_config(
        device="cpu", robot_config=args.robot_config, seed=7,
        num_envs=16, num_env_steps=2048, terrain_mode="plane",
        update_z_every_step=37, heading_reg_coeff=0.003,
        disable_dr=True, disable_obs_noise=True,
        checkpoint_every_steps=100, tracking_eval_every_steps=200,
    )
    assert cfg.agent.train.update_z_every_step == 37
    assert cfg.agent.train.reg_coeff_heading == 0.003
    assert cfg.agent.model.velocity_critic_enabled
    assert cfg.agent.model.archi.actor.estimated_velocity_enabled
    assert cfg.env.direct_depth_profile == final["env"]["direct_depth_profile"]
    assert cfg.env.terrain_training_curriculum
    assert cfg.env.disable_domain_randomization and cfg.env.disable_obs_noise
    assert "terrain.terrain_type=plane" in cfg.env.hydra_overrides
    assert "terrain.seed=7" in cfg.env.hydra_overrides
    assert cfg.checkpoint_every_steps == 100
    assert cfg.evaluations[0].every_steps == 200


@pytest.mark.parametrize("flags", [["--no-heading-context"], ["--heading-reg-coeff", "0"]])
def test_cli_rejects_context_changes_incompatible_with_final_velocity(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["train.py", *flags])
    with pytest.raises(ValueError, match="velocity critic requires"):
        parse_args()
