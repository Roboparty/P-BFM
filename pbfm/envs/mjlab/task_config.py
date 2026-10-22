"""Resolve the existing Hydra task and robot overrides without changing task defaults."""

import os
from typing import Any

import hydra
from omegaconf import OmegaConf

import pbfm
from pbfm.utils.helpers import pre_process_config
from pbfm.utils.resource_paths import CONFIG_ROOT, remap_resource_path, resolve_resource_path

if getattr(pbfm, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = os.path.dirname(pbfm.__file__)
else:
    HUMANOIDVERSE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HYDRA_CONFIG_DIR = str(CONFIG_ROOT / "simulation")
HYDRA_CONFIG_REL_PATH = os.path.join("exp", "bfm_zero", "bfm_zero")


def _resolve_humanoidverse_path(path_value: str | os.PathLike[str]) -> str:
    return str(remap_resource_path(path_value))


def _to_list(value) -> list:
    if value is None:
        return []
    return list(OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value)


def _to_float_dict(value) -> dict[str, float]:
    value = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return {str(k): float(v) for k, v in value.items()}


def _joint_limits_from_robot_training(robot_training: dict[str, Any], dof_names: tuple[str, ...]) -> tuple[list[float], list[float]]:
    joint_ranges = dict(robot_training.get("robot", {}).get("joint_ranges") or {})
    lower, upper = [], []
    for joint_name in dof_names:
        value = joint_ranges.get(joint_name)
        if value is None:
            lower.append(-3.14159)
            upper.append(3.14159)
        else:
            lower.append(float(value[0]))
            upper.append(float(value[1]))
    return lower, upper


def _patch_humanoidverse_robot_config(config, robot_training: dict[str, Any] | None) -> None:
    if not robot_training:
        return
    robot_info = dict(robot_training["robot"])
    dof_names = [str(name) for name in robot_info["control_joint_names"]]
    body_names = [str(name) for name in robot_info["body_names"]]
    feet = [str(name) for name in robot_info.get("feet") or []]
    lower, upper = _joint_limits_from_robot_training(robot_training, tuple(dof_names))

    config.robot.dof_names = dof_names
    config.robot.dof_obs_size = len(dof_names)
    config.robot.actions_dim = len(dof_names)
    config.robot.body_names = body_names
    config.robot.num_bodies = len(body_names)
    config.robot.key_bodies = list(robot_info.get("key_bodies") or [])
    config.robot.contact_bodies = list(robot_training.get("contact_bodies") or feet)
    config.robot.num_feet = len(config.robot.contact_bodies)
    config.robot.torso_name = str(robot_training.get("torso_name") or robot_info.get("base_body"))
    config.robot.penalize_contacts_on = list(robot_training.get("undesired_contact_bodies") or [])
    config.robot.terminate_after_contacts_on = list(robot_training.get("undesired_contact_bodies") or [])
    config.robot.left_ankle_dof_names = list(robot_training.get("left_ankle_dof_names") or [])
    config.robot.right_ankle_dof_names = list(robot_training.get("right_ankle_dof_names") or [])
    config.robot.dof_pos_lower_limit_list = lower
    config.robot.dof_pos_upper_limit_list = upper
    config.robot.dof_vel_limit_list = list(robot_training["velocity_limits"])
    config.robot.dof_effort_limit_list = list(robot_training["effort_limits"])
    config.robot.dof_effort_limit_scale = float(robot_training.get("effort_limit_scale", 1.0))

    if config.robot.get("init_state") is None:
        config.robot.init_state = OmegaConf.create({})
    if config.robot.get("control") is None:
        config.robot.control = OmegaConf.create({})

    config.robot.init_state.pos = list(robot_training["init_state"]["pos"])
    config.robot.init_state.rot = list(robot_training["init_state"]["rot"])
    config.robot.init_state.lin_vel = list(robot_training["init_state"]["lin_vel"])
    config.robot.init_state.ang_vel = list(robot_training["init_state"]["ang_vel"])
    config.robot.init_state.default_joint_angles = dict(robot_training["default_joint_angles"])
    config.robot.control.stiffness = dict(robot_training["stiffness"])
    config.robot.control.damping = dict(robot_training["damping"])
    config.robot.control.action_scale = float(robot_training["action_scale"])
    config.robot.control.action_clip_value = float(robot_training["action_clip_value"])
    config.robot.control.normalize_action_to = float(robot_training["normalize_action_to"])
    if "action_rescale" in robot_training:
        config.robot.control.action_rescale = bool(robot_training["action_rescale"])
    if "camera" in robot_training:
        for key, value in robot_training["camera"].items():
            config.terrain.direct_depth[key] = value
    if "foot_geometry" in robot_training:
        config.robot.foot_geometry = robot_training["foot_geometry"]
    if "aux_rewards_scaling" in robot_training:
        config.rewards.reward_scales = dict(robot_training["aux_rewards_scaling"])

    xml_path = resolve_resource_path(robot_info["xml_path"])
    if config.robot.get("asset") is None:
        config.robot.asset = OmegaConf.create({})
    config.robot.asset.asset_root = str(xml_path.parent)
    config.robot.asset.assetFileName = xml_path.name
    config.robot.asset.xml_file = str(xml_path)

    if config.robot.get("motion") is None:
        config.robot.motion = OmegaConf.create({})
    if config.robot.motion.get("asset") is None:
        config.robot.motion.asset = OmegaConf.create({})
    config.robot.motion.asset.assetRoot = str(xml_path.parent)
    config.robot.motion.asset.assetFileName = xml_path.name
    config.robot.motion.asset.urdfFileName = None

    extend_config = []
    for item in _to_list(config.robot.motion.get("extend_config", [])):
        parent_name = str(item.get("parent_name", ""))
        if parent_name in body_names:
            extend_config.append(dict(item))
    config.robot.motion.extend_config = extend_config
    config.robot.motion.nums_extend_bodies = len(extend_config)


def _compose_humanoidverse_config(
    *,
    num_envs: int,
    relative_config_path: str,
    hydra_overrides: list[str],
    lafan_tail_path: str | list[str],
    data_mix_weights: list[float] | None,
    disable_obs_noise: bool,
    disable_domain_randomization: bool,
    max_episode_length_s: float | None,
    root_height_obs: bool,
    robot_training: dict[str, Any] | None = None,
    direct_depth_profile: str = "legacy_rp1",
):
    with hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
        cfg = hydra.compose(config_name=relative_config_path, overrides=hydra_overrides or [])
    unresolved_conf = OmegaConf.to_container(cfg, resolve=False)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))

    cfg.num_envs = num_envs
    cfg.exp_base = "__no_exp_base__"
    cfg.env.config.headless = True
    OmegaConf.set_struct(cfg, False)
    _patch_humanoidverse_robot_config(cfg, robot_training)
    cfg.robot.asset.asset_root = _resolve_humanoidverse_path(cfg.robot.asset.asset_root)
    cfg.robot.motion.asset.assetRoot = _resolve_humanoidverse_path(cfg.robot.motion.asset.assetRoot)
    cfg.robot.motion.motion_file = (
        [str(remap_resource_path(path)) for path in lafan_tail_path]
        if isinstance(lafan_tail_path, list)
        else str(remap_resource_path(lafan_tail_path))
    )
    if data_mix_weights is not None:
        cfg.robot.motion.motion_file_weights = data_mix_weights

    pre_process_config(cfg)
    if cfg.terrain.get("direct_depth", None) is not None:
        cfg.terrain.direct_depth.processing_profile = direct_depth_profile

    if disable_obs_noise:
        for key in cfg.obs.noise_scales.keys():
            cfg.obs.noise_scales[key] = 0.0
    cfg.obs.root_height_obs = root_height_obs

    if disable_domain_randomization:
        cfg.domain_rand.randomize_ctrl_delay = False
        cfg.domain_rand.randomize_pd_gain = False
        cfg.domain_rand.randomize_base_com = False
        cfg.domain_rand.randomize_link_mass = False
        cfg.domain_rand.randomize_friction = False
        cfg.domain_rand.randomize_torque_rfi = False
        cfg.domain_rand.randomize_rfi_lim = False
        cfg.domain_rand.randomize_push_robots = False
        cfg.domain_rand.push_robots = False
        cfg.domain_rand.randomize_default_dof_pos = False
        if "motor_delay_steps" in cfg.domain_rand:
            cfg.domain_rand.motor_delay_steps = [0, 0]
        direct_depth = cfg.terrain.get("direct_depth", None)
        if direct_depth is not None:
            direct_depth.position_error = 0.0
            direct_depth.angle_error_deg = 0.0
            direct_depth.installation_dr_disabled = True

    assert cfg.env.config.termination.terminate_when_close_to_dof_pos_limit is False
    assert cfg.env.config.termination.terminate_when_close_to_dof_vel_limit is False
    assert cfg.env.config.termination.terminate_when_close_to_torque_limit is False
    assert cfg.env.config.termination.terminate_by_contact is False
    assert cfg.env.config.termination.terminate_by_gravity is False
    assert cfg.env.config.termination.terminate_by_low_height is False
    assert cfg.env.config.termination.terminate_when_motion_end is False
    assert cfg.env.config.termination.terminate_when_motion_far is False
    assert cfg.env.config.robot.control.normalize_action_to == cfg.env.config.robot.control.action_clip_value

    if max_episode_length_s is not None:
        cfg.env.config.max_episode_length_s = max_episode_length_s

    return cfg.env.config, unresolved_conf
