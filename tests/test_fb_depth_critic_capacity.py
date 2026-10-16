"""Guard depth critic capacities, learning rates and contact reward defaults."""

import json
from pathlib import Path

import pytest

from pbfm.agents.presets.fb_depth import build_fb_depth_agent

FINAL_AGENT = json.loads((Path(__file__).parents[1] / "pbfm/configs/legacy_large.json").read_text())["agent"]


@pytest.mark.parametrize("behavior_prior", [False, True])
@pytest.mark.parametrize("heading_reg_coeff", [0.002, 0.01])
def test_compact_critics_in_supported_depth_modes(behavior_prior, heading_reg_coeff):
    cfg = build_fb_depth_agent(
        device="cpu",
        compile=False,
        behavior_prior=behavior_prior,
        heading_reg_coeff=heading_reg_coeff,
    )
    archi = cfg.model.archi
    for key, size in {"critic": (1024, 4), "aux_critic": (2048, 4), "heading_critic": (1024, 3), "velocity_critic": (1024, 3)}.items():
        critic = getattr(archi, key)
        assert (critic.hidden_dim, critic.hidden_layers) == size
        assert critic.model == "residual"
        assert critic.embedding_layers == 2
        assert critic.num_parallel == 2
        assert critic.ensemble_mode == "batch"
        assert critic.input_filter.key == FINAL_AGENT["model"]["archi"][key]["input_filter"]["key"]

    assert (archi.actor.hidden_dim, archi.actor.hidden_layers) == (2048, 6)
    assert (archi.f.hidden_dim, archi.f.hidden_layers, archi.f.num_parallel) == (2048, 6, 2)
    assert (archi.b.hidden_dim, archi.b.hidden_layers) == (256, 1)
    assert (archi.discriminator.hidden_dim, archi.discriminator.hidden_layers) == (1024, 3)
    assert archi.z_dim == 256
    assert archi.actor.depth_channels == 8
    assert archi.actor.depth_latent_dim == 256
    assert (archi.actor.depth_height, archi.actor.depth_width) == (18, 32)
    assert archi.actor.depth_first_conv_stride == 1
    assert archi.actor.depth_frame_feature_dim == 256
    assert archi.actor.estimated_velocity_enabled
    assert archi.actor.velocity_supervision_updates_depth
    assert cfg.model.seq_length == 8
    assert "heading" not in archi.b.input_filter.key
    assert "heading" not in archi.discriminator.input_filter.key
    assert cfg.model.heading_critic_enabled
    assert cfg.model.velocity_critic_enabled
    assert cfg.model.reference_velocity_policy == "all_reference"
    assert cfg.train.behavior_prior_enabled == behavior_prior
    assert cfg.train.reg_coeff == (0.01 if behavior_prior else 0.0)
    assert cfg.train.reg_coeff_aux == 0.02
    assert cfg.train.reg_coeff_heading == heading_reg_coeff
    assert cfg.train.reg_coeff_velocity == 0.002
    assert cfg.train.lr_discriminator == 5e-6
    assert cfg.train.discriminator_loss == "bce"
    assert cfg.train.discriminator_reward == "log_odds"
    assert cfg.aux_rewards_scaling == {
        "penalty_action_rate": -0.1,
        "limits_dof_pos": -10.0,
        "penalty_body_impact": -1.0,
        "penalty_slippage": -4.0,
        "penalty_ankle_roll": -1.0,
        "feet_stumble": -2.0,
    }
    assert set(cfg.aux_rewards) == set(cfg.aux_rewards_scaling)
    assert cfg.aux_rewards.count("feet_stumble") == 1
    assert "feet_at_plane" not in cfg.aux_rewards


def test_depth_preset_matches_final_config_and_overrides_do_not_mutate_defaults():
    assert build_fb_depth_agent().model_dump(mode="json") == FINAL_AGENT
    overridden = build_fb_depth_agent(
        device="cpu", compile=True, heading_reg_coeff=0.01,
        behavior_prior=False, cartwheel_aux_safe=True,
        lr_scale=2.0, update_z_every_step=37, clip_grad_norm=1.0,
    )
    overridden.model.archi.actor.input_filter.key.clear()
    assert build_fb_depth_agent().model_dump(mode="json") == FINAL_AGENT


@pytest.mark.parametrize(
    ("heading_context", "heading_reg_coeff"),
    [(False, None), (False, 0.002), (True, 0.0), (True, -0.001)],
)
def test_final_velocity_critic_rejects_disabled_heading(heading_context, heading_reg_coeff):
    with pytest.raises(ValueError, match="QV requires heading_context=True and positive heading_reg_coeff"):
        build_fb_depth_agent(
            device="cpu", heading_context=heading_context, heading_reg_coeff=heading_reg_coeff,
        )


def test_explicit_cartwheel_safe_reward_override_is_preserved():
    depth = build_fb_depth_agent(device="cpu", compile=False, cartwheel_aux_safe=True)
    assert depth.aux_rewards == ["penalty_torques", "penalty_action_rate", "limits_dof_pos", "limits_torque"]
    assert depth.aux_rewards_scaling == {
        "penalty_action_rate": -0.03,
        "limits_dof_pos": -10.0,
        "penalty_torques": 0.0,
        "limits_torque": 0.0,
    }


@pytest.mark.parametrize("lr_scale", [0.5, 1.0, 2.0])
def test_learning_rate_override_preserves_calibrated_discriminator_rate(lr_scale):
    depth = build_fb_depth_agent(device="cpu", compile=False, lr_scale=lr_scale)
    expected = dict(FINAL_AGENT["train"])
    for name in expected:
        if name.startswith("lr_") and name != "lr_discriminator":
            expected[name] *= lr_scale
    assert depth.train.model_dump(mode="json") == expected


def test_formal_training_config_and_serialization_preserve_compact_critics(tmp_path):
    from pbfm.train import build_ufo_mjlab_config

    cfg = build_ufo_mjlab_config(
        device="cpu",
        work_dir=str(tmp_path),
        num_envs=16,
        num_env_steps=2048,
        seed=1,
        use_wandb=False,
        wandb_run_name=None,
        smoke=False,
        agent="fb_depth",
        heading_context=True,
        heading_reg_coeff=0.002,
    )
    assert cfg.prior_plane_envs == 0
    assert all(evaluation.name_in_logs != "same_z_terrain_eval" for evaluation in cfg.evaluations)
    restored = type(cfg.agent).model_validate_json(cfg.agent.model_dump_json())
    assert restored.model_dump() == cfg.agent.model_dump()
    assert restored.train.lr_discriminator == 5e-6
    assert restored.aux_rewards_scaling["penalty_slippage"] == -4.0
    assert restored.aux_rewards_scaling["feet_stumble"] == -2.0
    assert "feet_stumble" in restored.aux_rewards
    assert "feet_at_plane" not in restored.aux_rewards
    archi = restored.model.archi
    assert (archi.critic.hidden_dim, archi.critic.hidden_layers) == (1024, 4)
    assert (archi.aux_critic.hidden_dim, archi.aux_critic.hidden_layers) == (2048, 4)
    assert (archi.heading_critic.hidden_dim, archi.heading_critic.hidden_layers) == (1024, 3)
    assert restored.model.heading_critic_enabled
