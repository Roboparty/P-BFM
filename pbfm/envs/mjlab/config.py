"""Construct the MJLab scene, sensors, actuators, and environment creation settings."""

import math
import typing as tp
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic
import torch

from pbfm.agents.base import BaseConfig
from pbfm.perception.depth_camera import convert_camera_quaternion_to_optical_xyzw, make_depth_camera_sensor_cfg
from pbfm.perception.instinct_direct_depth import (
    BASELINE_TYPE as DIRECT_DEPTH_BASELINE_TYPE,
)
from pbfm.perception.instinct_direct_depth import (
    DEPTH_CONVENTION as DIRECT_DEPTH_CONVENTION,
)
from pbfm.perception.instinct_direct_depth import (
    REFERENCE_COMMIT as DIRECT_DEPTH_REFERENCE_COMMIT,
)
from pbfm.perception.instinct_direct_depth import (
    REFERENCE_PROJECT as DIRECT_DEPTH_REFERENCE_PROJECT,
)
from pbfm.perception.instinct_direct_depth import (
    DirectDepthConfig,
)
from pbfm.terrains import make_terrain_entity_cfg, normalize_terrain_mode
from pbfm.terrains.presets import add_training_nonflat_guard_ring, add_training_outer_walls
from pbfm.terrains.terrain_height_sensor import PbfmTerrainHeightSensorCfg, WorldUpTerrainHeightSensorCfg
from pbfm.terrains.terrain_observation import RobotCentricGridPatternCfg
from pbfm.utils.resource_paths import remap_resource_path

from .task_config import HUMANOIDVERSE_DIR, HYDRA_CONFIG_REL_PATH, _compose_humanoidverse_config, _to_float_dict, _to_list

if TYPE_CHECKING:
    from .vector_env import HumanoidVerseMjlabVectorEnv


G1_MJLAB_MJCF_PATH = "pbfm/data/robots/g1_mjlab/g1_29dof.xml"
G1_MJLAB_ACTUATOR_SOURCE = "g1-mode_15"


def _direct_depth_config(config) -> DirectDepthConfig | None:
    """Parse and validate the direct-depth Hydra block."""
    if not bool(config.terrain.get("enabled", False)):
        return None
    raw = config.terrain.get("direct_depth", None)
    if raw is None or not bool(raw.get("enabled", False)):
        return None
    metadata = {
        "reference_project": DIRECT_DEPTH_REFERENCE_PROJECT,
        "reference_commit": DIRECT_DEPTH_REFERENCE_COMMIT,
        "baseline_type": DIRECT_DEPTH_BASELINE_TYPE,
        "replay_storage_dtype": "compact_uint8",
        "depth_convention": DIRECT_DEPTH_CONVENTION,
    }
    for key, expected in metadata.items():
        actual = str(raw.get(key, ""))
        if actual != expected:
            raise ValueError(f"terrain.direct_depth.{key}={actual!r}, expected frozen value {expected!r}")
    camera_frame_quat = torch.tensor(tuple(float(v) for v in raw.camera_frame_quat_wxyz), dtype=torch.float64)
    optical_quat = convert_camera_quaternion_to_optical_xyzw(
        camera_frame_quat,
        convention="world",
        input_order="wxyz",
    )
    cfg = DirectDepthConfig(
        depth_convention=str(raw.depth_convention),
        width=int(raw.width),
        height=int(raw.height),
        horizontal_fov_deg=float(raw.horizontal_fov_deg),
        vertical_fov_deg=float(raw.vertical_fov_deg),
        min_distance=float(raw.min_distance),
        max_distance=float(raw.max_distance),
        crop=tuple(int(v) for v in raw.crop),
        history_length=int(raw.history_length),
        history_skip_frames=int(raw.history_skip_frames),
        num_output_frames=int(raw.num_output_frames),
        delayed_frame_ranges=tuple(int(v) for v in raw.delayed_frame_ranges),
        position_error=float(raw.get("position_error", 0.05)),
        angle_error_rad=math.radians(float(raw.get("angle_error_deg", 5.0))),
        include_geom_groups=tuple(int(v) for v in raw.include_geom_groups),
        exclude_parent_body=bool(raw.exclude_parent_body),
        mount_body=str(raw.mount_body),
        mount_pos=tuple(float(v) for v in raw.mount_pos),
        optical_quat_torso_xyzw=tuple(float(v) for v in optical_quat),
    )
    from pbfm.perception.instinct_direct_depth import depth_config_for_profile
    cfg = depth_config_for_profile(str(raw.get("processing_profile", "legacy_rp1")), cfg)
    if bool(raw.get("installation_dr_disabled", False)):
        cfg = replace(cfg, position_error=0., angle_error_rad=0., angle_error_rpy_rad=(0., 0., 0.))
    cfg.validate()
    return cfg


def _reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    """Compute reflected motor inertia constants for Unitree G1 motors."""
    return rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2 + rotor_inertia[1] * gear_ratio[2] ** 2 + rotor_inertia[2]


_ARMATURE_5020 = _reflected_inertia_from_two_stage_planetary((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + (46 / 18), 1 + (56 / 16)))
_ARMATURE_7520_14 = _reflected_inertia_from_two_stage_planetary((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + (48 / 22)))
_ARMATURE_7520_22 = _reflected_inertia_from_two_stage_planetary((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
_ARMATURE_5010 = _reflected_inertia_from_two_stage_planetary((0.084e-4, 0.015e-4, 0.068e-4), (1, 4, 4))


def _g1_mjlab_mode15_actuator_params(dof_names: tp.Sequence[str]) -> dict[str, list[float]]:
    """Return per-DOF G1 mode-15 motor params in UFO order.

    The constants are vendored here so training does not depend on an external
    asset package or download path. Kp/Kd remain UFO values; this only
    supplies motor effort, velocity reference, armature and dry friction.
    """

    efforts: list[float] = []
    velocities: list[float] = []
    armatures: list[float] = []
    frictions: list[float] = []

    for joint_name in dof_names:
        if "_hip_pitch_joint" in joint_name:
            effort, velocity, armature = 139.0, 20.0, _ARMATURE_7520_22
        elif "_hip_yaw_joint" in joint_name or joint_name == "waist_yaw_joint":
            effort, velocity, armature = 88.0, 32.0, _ARMATURE_7520_14
        elif "_hip_roll_joint" in joint_name or "_knee_joint" in joint_name:
            effort, velocity, armature = 139.0, 20.0, _ARMATURE_7520_22
        elif "_ankle_pitch_joint" in joint_name or "_ankle_roll_joint" in joint_name:
            effort, velocity, armature = 50.0, 37.0, 2.0 * _ARMATURE_5020
        elif joint_name in ("waist_pitch_joint", "waist_roll_joint"):
            effort, velocity, armature = 50.0, 37.0, 2.0 * _ARMATURE_5020
        elif (
            "_shoulder_pitch_joint" in joint_name
            or "_shoulder_roll_joint" in joint_name
            or "_shoulder_yaw_joint" in joint_name
            or "_elbow_joint" in joint_name
            or "_wrist_roll_joint" in joint_name
        ):
            effort, velocity, armature = 25.0, 37.0, _ARMATURE_5020
        elif "_wrist_pitch_joint" in joint_name or "_wrist_yaw_joint" in joint_name:
            effort, velocity, armature = 13.4, 27.0, _ARMATURE_5010
        else:
            raise ValueError(f"No G1 mode-15 actuator parameters for joint: {joint_name}")

        efforts.append(effort)
        velocities.append(velocity)
        armatures.append(armature)
        frictions.append(0.01)

    return {
        "effort_limit": efforts,
        "velocity_limit": velocities,
        "armature": armatures,
        "friction": frictions,
    }


def _obs_joint_pos(env):
    return env.scene["robot"].data.joint_pos


def _zero_reward(env):
    return torch.zeros(env.num_envs, device=env.device)


def _mass_scale_range_to_pseudo_inertia_alpha(value) -> tuple[float, float]:
    """Convert an intuitive mass scale into MJLab's log-density parameter."""
    scale_range = [float(item) for item in _to_list(value)]
    if len(scale_range) != 2:
        raise ValueError(f"domain_rand.link_mass_range must have two entries, got {scale_range}")
    lower, upper = scale_range
    if lower <= 0.0 or upper < lower:
        raise ValueError(
            "domain_rand.link_mass_range must satisfy 0 < min <= max, "
            f"got {scale_range}"
        )
    return 0.5 * math.log(lower), 0.5 * math.log(upper)


def _match_joint_value(joint_name: str, value_by_substring: dict[str, float], default: float = 0.0) -> float:
    for key, value in value_by_substring.items():
        if key in joint_name:
            return float(value)
    return float(default)


def _actuator_params_from_training(dof_names: tp.Sequence[str], robot_training: dict[str, Any] | None) -> tuple[str, dict[str, list[float]]]:
    if not robot_training:
        return G1_MJLAB_ACTUATOR_SOURCE, _g1_mjlab_mode15_actuator_params(dof_names)
    actuator = dict(robot_training.get("actuator") or {})
    source = str(actuator.get("source", G1_MJLAB_ACTUATOR_SOURCE))
    if source in {"g1_mode15", "g1-mode_15"}:
        return source, _g1_mjlab_mode15_actuator_params(dof_names)
    if source != "yaml":
        raise ValueError(f"Unsupported training.actuator.source={source!r}")
    joints = actuator.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("training.actuator.source=yaml requires training.actuator.joints")
    params = {"effort_limit": [], "velocity_limit": [], "armature": [], "friction": []}
    for joint_name in dof_names:
        joint_params = joints.get(joint_name)
        if not isinstance(joint_params, dict):
            raise ValueError(f"training.actuator.joints is missing parameters for joint {joint_name!r}")
        for key in params:
            if key not in joint_params:
                raise ValueError(f"training.actuator.joints.{joint_name} is missing '{key}'")
            params[key].append(float(joint_params[key]))
    return source, params


def _default_joint_pos(config) -> torch.Tensor:
    values = [float(config.robot.init_state.default_joint_angles[name]) for name in config.robot.dof_names]
    return torch.tensor(values, dtype=torch.float32)


def _action_target_scale(config) -> torch.Tensor:
    dof_names = tuple(_to_list(config.robot.dof_names))
    stiffness = _to_float_dict(config.robot.control.stiffness)
    # UFO action_rescale uses the configured effort limits.  The Isaac
    # path does not apply dof_effort_limit_scale to those limits, so MJLab must
    # not do it either.
    effort_limits = [float(x) for x in _to_list(config.robot.dof_effort_limit_list)]
    scales = []
    for i, joint_name in enumerate(dof_names):
        kp = _match_joint_value(joint_name, stiffness)
        scale = float(config.robot.control.action_scale)
        if bool(config.robot.control.action_rescale):
            if kp <= 0.0:
                raise ValueError(f"Cannot action_rescale joint {joint_name}: stiffness={kp}")
            scale *= effort_limits[i] / kp
        scales.append(scale)
    return torch.tensor(scales, dtype=torch.float32)


def make_mjlab_ufo_env_cfg(
    config,
    *,
    num_envs: int,
    seed: int | None,
    mjcf_path: str | None,
    auto_reset: bool,
    robot_training: dict[str, Any] | None = None,
    evaluation_fast_path: bool = False,
):
    """Create an MJLab ManagerBasedRlEnvCfg with UFO robot metadata."""
    import mujoco
    from mjlab.actuator import DcMotorActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs import mdp as mjlab_mdp
    from mjlab.envs.mdp import dr as mjlab_dr
    from mjlab.envs.mdp import terminations as mjlab_terminations
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sensor import ObjRef
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
    from mjlab.sensor.raycast_sensor import GridPatternCfg
    from mjlab.sim import MujocoCfg, SimulationCfg

    dof_names = tuple(_to_list(config.robot.dof_names))
    body_names = tuple(_to_list(config.robot.body_names))
    xml_path = remap_resource_path(mjcf_path if mjcf_path is not None else G1_MJLAB_MJCF_PATH)
    xml_path = xml_path if xml_path.is_absolute() else Path(HUMANOIDVERSE_DIR).parent / xml_path
    if not xml_path.exists():
        raise FileNotFoundError(f"MJCF asset not found: {xml_path}")
    if "actuatorfrcrange" in xml_path.read_text():
        raise ValueError(f"MJLab robot XML must not contain actuatorfrcrange: {xml_path}")

    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(xml_path))
        # The UFO Isaac path uses implicit position PD targets. XML motor
        # actuators are removed so MJLab adds equivalent position actuators.
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        return spec

    stiffness = _to_float_dict(config.robot.control.stiffness)
    damping = _to_float_dict(config.robot.control.damping)
    effort_scale = float(getattr(config.robot, "dof_effort_limit_scale", 1.0))
    bfm_effort_limits = [float(x) for x in _to_list(config.robot.dof_effort_limit_list)]
    actuator_source, actuator_params = _actuator_params_from_training(dof_names, robot_training)
    effort_limits = actuator_params["effort_limit"]
    velocity_limits = actuator_params["velocity_limit"]
    armature = actuator_params["armature"]
    friction = actuator_params["friction"]

    actuators = []
    action_scale = {}
    for i, joint_name in enumerate(dof_names):
        kp = _match_joint_value(joint_name, stiffness)
        kd = _match_joint_value(joint_name, damping)
        effort_limit = effort_limits[i]
        actuators.append(
            DcMotorActuatorCfg(
                target_names_expr=(joint_name,),
                stiffness=kp,
                damping=kd,
                effort_limit=effort_limit,
                saturation_effort=effort_limit,
                velocity_limit=velocity_limits[i],
                armature=armature[i] if i < len(armature) else None,
                frictionloss=friction[i] if i < len(friction) else None,
            )
        )

        scale = float(config.robot.control.action_scale)
        if bool(config.robot.control.action_rescale):
            if kp <= 0.0:
                raise ValueError(f"Cannot action_rescale joint {joint_name}: stiffness={kp}")
            scale *= bfm_effort_limits[i] / kp
        action_scale[joint_name] = scale

    if len(actuators) != len(dof_names):
        raise ValueError(f"Expected one MJLab actuator per UFO dof, got {len(actuators)} for {len(dof_names)} dofs")
    scaled_effort_limits = [float(x) * effort_scale for x in bfm_effort_limits]
    if effort_scale != 1.0 and any(abs(a - b) < 1.0e-6 for a, b in zip(effort_limits, scaled_effort_limits)):
        raise ValueError("MJLab actuator effort limits unexpectedly include dof_effort_limit_scale")
    print(
        "[INFO] MJLab asset: "
        f"xml_path={xml_path}, actuator_source={actuator_source}, "
        f"actuator_count={len(actuators)}, joint_order={list(dof_names)}, "
        f"action_scale={[action_scale[name] for name in dof_names]}, "
        f"kp={[_match_joint_value(name, stiffness) for name in dof_names]}, "
        f"kd={[_match_joint_value(name, damping) for name in dof_names]}, "
        f"effort_limit={effort_limits}, velocity_limit={velocity_limits}, "
        f"armature={armature}, friction={friction}, "
        f"dof_effort_limit_scale={effort_scale} ignored_for_mjlab_actuator_limits",
        flush=True,
    )

    init_rot_xyzw = tuple(float(x) for x in config.robot.init_state.rot)
    init_rot_wxyz = (init_rot_xyzw[3], init_rot_xyzw[0], init_rot_xyzw[1], init_rot_xyzw[2])
    init_state = EntityCfg.InitialStateCfg(
        pos=tuple(float(x) for x in config.robot.init_state.pos),
        rot=init_rot_wxyz,
        lin_vel=tuple(float(x) for x in config.robot.init_state.lin_vel),
        ang_vel=tuple(float(x) for x in config.robot.init_state.ang_vel),
        joint_pos={name: float(config.robot.init_state.default_joint_angles[name]) for name in dof_names},
        joint_vel={".*": 0.0},
    )

    robot_cfg = EntityCfg(
        spec_fn=spec_fn,
        init_state=init_state,
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=1.0),
        sort_actuators=True,
    )
    sensors = [
        ContactSensorCfg(
            name="body_contact",
            primary=ContactMatch(mode="body", pattern=body_names, entity="robot"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=int(config.simulator.config.sim.control_decimation),
        )
    ]
    terrain_enabled = bool(config.terrain.get("enabled", False))
    terrain_mode = normalize_terrain_mode(str(config.terrain.get("terrain_type", "plane"))) if terrain_enabled else "plane"
    direct_depth_cfg = _direct_depth_config(config)
    terrain_observation_mode = str(config.terrain.terrain_priv.get("mode", "raycast")) if terrain_enabled else "raycast"
    if terrain_observation_mode not in {"raycast", "flat_zero"}:
        raise ValueError(f"Unsupported terrain observation mode: {terrain_observation_mode!r}")
    if terrain_observation_mode == "flat_zero" and terrain_mode not in {"flat", "plane"}:
        raise ValueError("terrain_priv.mode=flat_zero is only valid with terrain.terrain_type=flat or plane")
    if terrain_enabled:
        terrain_priv_cfg = config.terrain.terrain_priv
        grid_pattern = RobotCentricGridPatternCfg(
            x_min=float(terrain_priv_cfg.x_min),
            x_max=float(terrain_priv_cfg.x_max),
            y_min=float(terrain_priv_cfg.y_min),
            y_max=float(terrain_priv_cfg.y_max),
            resolution=float(terrain_priv_cfg.resolution),
        )
        if terrain_observation_mode == "raycast":
            sensors.append(
                PbfmTerrainHeightSensorCfg(
                    name="terrain_height",
                    frame=ObjRef(type="body", name=str(config.robot.body_names[0]), entity="robot"),
                    pattern=grid_pattern,
                    ray_alignment="yaw",
                    max_distance=float(terrain_priv_cfg.max_ray_distance),
                    include_geom_groups=(0,) if terrain_mode == "training_mix" else (5,),
                    padding_ray_group=0 if terrain_mode == "training_mix" else 5,
                    reduction="none",
                    debug_vis=bool(terrain_priv_cfg.get("debug_vis", False)),
                )
            )
        if bool(config.rewards.get("terrain_aware_auxiliary", False)) and not evaluation_fast_path:
            terrain_ray_groups = (0,) if terrain_mode in {"training_mix", "plane", "flat"} else (5,)
            sensors.extend(
                (
                    ContactSensorCfg(
                        name="foot_collision_contacts",
                        primary=ContactMatch(
                            mode="geom",
                            pattern="(left|right)_foot.*_collision",
                            entity="robot",
                        ),
                        fields=("force", "normal", "tangent"),
                        reduce="maxforce",
                        num_slots=2,
                        global_frame=True,
                        history_length=int(config.simulator.config.sim.control_decimation),
                    ),
                    WorldUpTerrainHeightSensorCfg(
                        name="foot_height_scanners",
                        frame=(
                            ObjRef(type="body", name="left_ankle_roll_link", entity="robot"),
                            ObjRef(type="body", name="right_ankle_roll_link", entity="robot"),
                        ),
                        pattern=GridPatternCfg(size=(0.23, 0.04), resolution=0.01),
                        ray_alignment="base",
                        ray_start_height=20.0,
                        local_center=(0.025, 0.0, -0.045),
                        max_distance=25.0,
                        include_geom_groups=terrain_ray_groups,
                        debug_vis=False,
                    ),
                )
            )
        if direct_depth_cfg is not None:
            sensors.append(
                make_depth_camera_sensor_cfg(
                    direct_depth_cfg.camera_config(),
                    torso_body_name=direct_depth_cfg.mount_body,
                    exclude_parent_body=direct_depth_cfg.exclude_parent_body,
                    ray_alignment="base",
                    ray_origin_shift=direct_depth_cfg.min_distance,
                    position_error=direct_depth_cfg.position_error,
                    angle_error_rad=direct_depth_cfg.angle_error_rad,
                    angle_error_rpy_rad=direct_depth_cfg.angle_error_rpy_rad,
                    installation_frame=direct_depth_cfg.installation_frame,
                    raycast_max_distance=direct_depth_cfg.raycast_max_distance,
                )
            )
            if bool(config.terrain.direct_depth.get("debug_terrain_reference", False)):
                terrain_reference_camera = replace(
                    direct_depth_cfg.camera_config(),
                    name="g1_direct_depth_terrain_reference",
                    include_geom_groups=(0, 5) if terrain_mode == "training_mix" else (5,),
                )
                sensors.append(
                    make_depth_camera_sensor_cfg(
                        terrain_reference_camera,
                        torso_body_name=direct_depth_cfg.mount_body,
                        exclude_parent_body=direct_depth_cfg.exclude_parent_body,
                        ray_alignment="base",
                        ray_origin_shift=direct_depth_cfg.min_distance,
                    )
                )

    terrain_entity_cfg = (
        make_terrain_entity_cfg(terrain_mode, env_spacing=float(config.env_spacing), config=config.terrain)
        if terrain_enabled
        else make_terrain_entity_cfg("plane", env_spacing=float(config.env_spacing))
    )
    scene_spec_fn = None
    if terrain_mode == "training_mix":
        terrain_generator = terrain_entity_cfg.terrain_generator
        if terrain_generator is None:
            raise ValueError("training_mix terrain requires a terrain generator")

        def scene_spec_fn(spec: mujoco.MjSpec) -> None:
            add_training_nonflat_guard_ring(spec, terrain_generator)
            add_training_outer_walls(spec, terrain_generator)
    observations = {
        "actor": ObservationGroupCfg(
            terms={"joint_pos": ObservationTermCfg(func=_obs_joint_pos)},
            concatenate_terms=True,
            enable_corruption=False,
        )
    }
    domain_rand = config.domain_rand
    action_class, extra_action = JointPositionActionCfg, {}
    if bool(domain_rand.get("randomize_torque_rfi", False)):
        from pbfm.envs.mjlab.torque_rfi import RfiJointPositionActionCfg
        action_class = RfiJointPositionActionCfg
        extra_action = dict(torque_limits=tuple(float(x) for x in config.robot.dof_effort_limit_list),
                            rfi_lim=float(domain_rand.rfi_lim))
    actions = {
        "actions": action_class(
            entity_name="robot",
            actuator_names=dof_names,
            preserve_order=True,
            scale=action_scale,
            use_default_offset=True,
            **extra_action,
        )
    }
    reward_keys = tuple(config.rewards.reward_scales.keys())
    rewards = {key: RewardTermCfg(func=_zero_reward, weight=0.0) for key in reward_keys}
    terminations = {
        "time_out": TerminationTermCfg(func=mjlab_terminations.time_out, time_out=True),
    }
    domain_rand = config.domain_rand
    events = {}
    if bool(domain_rand.get("randomize_pd_gain", False)):
        events["random_pd_gains"] = EventTermCfg(
            mode="reset", func=mjlab_dr.pd_gains,
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
                    "operation": "scale",
                    "kp_range": tuple(float(x) for x in domain_rand.kp_range),
                    "kd_range": tuple(float(x) for x in domain_rand.kd_range)},
        )
    if bool(domain_rand.get("push_robots", False)):
        max_push_vel_xy = float(domain_rand.max_push_vel_xy)
        max_push_ang_vel = float(domain_rand.get("max_push_ang_vel", 0.0))
        velocity_range = {
            "x": (-max_push_vel_xy, max_push_vel_xy),
            "y": (-max_push_vel_xy, max_push_vel_xy),
        }
        if max_push_ang_vel > 0.0:
            velocity_range.update(
                {
                    "roll": (-max_push_ang_vel, max_push_ang_vel),
                    "pitch": (-max_push_ang_vel, max_push_ang_vel),
                    "yaw": (-max_push_ang_vel, max_push_ang_vel),
                }
            )
        events["push_robots"] = EventTermCfg(
            func=mjlab_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=tuple(float(x) for x in _to_list(domain_rand.push_interval_s)),
            params={"velocity_range": velocity_range},
        )
    # pseudo_inertia reconstructs body_ipos from the model defaults, so it
    # must run before the independent torso COM-offset event below.
    if bool(domain_rand.get("randomize_link_mass", False)):
        events["random_link_pseudo_inertia"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "alpha_range": _mass_scale_range_to_pseudo_inertia_alpha(
                    domain_rand.link_mass_range
                ),
            },
        )
    if bool(domain_rand.get("randomize_base_com", False)):
        base_com_range = domain_rand.base_com_range
        events["random_base_com"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(str(config.robot.torso_name),)),
                "operation": "add",
                "ranges": {
                    0: tuple(float(x) for x in _to_list(base_com_range.x)),
                    1: tuple(float(x) for x in _to_list(base_com_range.y)),
                    2: tuple(float(x) for x in _to_list(base_com_range.z)),
                },
            },
        )
    if bool(domain_rand.get("randomize_friction", False)):
        events["random_geom_friction"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "operation": "abs",
                "axes": [0],
                "ranges": tuple(float(x) for x in _to_list(domain_rand.friction_range)),
            },
        )

    return ManagerBasedRlEnvCfg(
        decimation=int(config.simulator.config.sim.control_decimation),
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=float(config.env_spacing),
            terrain=terrain_entity_cfg,
            entities={"robot": robot_cfg},
            sensors=tuple(sensors),
            spec_fn=scene_spec_fn,
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        seed=seed,
        sim=SimulationCfg(
            nconmax=512,
            njmax=4096,
            contact_sensor_maxmatch=256,
            mujoco=MujocoCfg(timestep=1.0 / float(config.simulator.config.sim.fps)),
        ),
        episode_length_s=float(config.max_episode_length_s),
        auto_reset=auto_reset,
        scale_rewards_by_dt=False,
    )


class HumanoidVerseMjlabConfig(BaseConfig):
    name: tp.Literal["humanoidverse_mjlab"] = "humanoidverse_mjlab"
    terrain_training_curriculum: bool = False
    terrain_curriculum_start_global_step: int = pydantic.Field(default=0, ge=0)
    terrain_curriculum_interval: int = pydantic.Field(default=19_200_000, gt=0)
    terrain_curriculum_easy_fraction: float = pydantic.Field(default=0.25, ge=0, le=1)

    device: str = "cuda:0"
    lafan_tail_path: str | list[str]
    data_mix_weights: list[float] | None = None
    mjcf_path: str | None = None
    robot_config_path: str | None = None
    robot_training: dict[str, Any] | None = None
    max_episode_length_s: float | None = None
    disable_obs_noise: bool = False
    disable_domain_randomization: bool = False
    relative_config_path: str = HYDRA_CONFIG_REL_PATH
    include_last_action: bool = True
    hydra_overrides: tp.List[str] = pydantic.Field(default_factory=list)
    context_length: int | None = None
    include_history_actor: bool = False
    include_history_noaction: bool = False
    root_height_obs: bool = False
    auto_reset: bool = False
    seed: int | None = None
    evaluation_fast_path: bool = False
    velocity_context_enabled: bool = False
    include_velocity_estimator_target: bool = False
    virtual_edge_penalty_enabled: bool = False
    virtual_edge_radius: float = pydantic.Field(default=0.05, gt=0, lt=0.2)
    # Evaluation-only override. None preserves the frozen per-episode {0, 1}
    # training contract; deterministic same-z evaluation explicitly uses 0.
    fixed_direct_depth_delay_frames: int | None = None
    direct_depth_profile: tp.Literal["legacy_rp1", "optical_z_maxcrop18_reset_v2"] = "legacy_rp1"

    def model_post_init(self, context) -> None:
        if self.fixed_direct_depth_delay_frames is not None and int(self.fixed_direct_depth_delay_frames) not in {0, 1}:
            raise ValueError("fixed_direct_depth_delay_frames must be 0, 1, or None")

    def build(self, num_envs: int = 1, *, motion_lib=None) -> tp.Tuple["HumanoidVerseMjlabVectorEnv", tp.Any]:
        assert num_envs >= 1
        if self.auto_reset:
            raise ValueError("PBFM owns task reset; auto_reset must be False")
        from mjlab.envs import ManagerBasedRlEnv

        from .core import HumanoidVerseMjlabCore
        from .vector_env import HumanoidVerseMjlabVectorEnv

        hv_config, unresolved_conf = _compose_humanoidverse_config(
            num_envs=num_envs,
            relative_config_path=self.relative_config_path,
            hydra_overrides=list(self.hydra_overrides),
            lafan_tail_path=self.lafan_tail_path,
            data_mix_weights=self.data_mix_weights,
            disable_obs_noise=self.disable_obs_noise,
            disable_domain_randomization=self.disable_domain_randomization,
            max_episode_length_s=self.max_episode_length_s,
            root_height_obs=self.root_height_obs,
            robot_training=self.robot_training,
            direct_depth_profile=self.direct_depth_profile,
        )
        mjlab_cfg = make_mjlab_ufo_env_cfg(
            hv_config,
            num_envs=num_envs,
            seed=self.seed,
            mjcf_path=self.mjcf_path,
            auto_reset=self.auto_reset,
            robot_training=self.robot_training,
            evaluation_fast_path=self.evaluation_fast_path,
        )
        mjlab_env = ManagerBasedRlEnv(mjlab_cfg, device=self.device)
        core = HumanoidVerseMjlabCore(
            hv_config,
            mjlab_env,
            creation_config=self,
            motion_lib_override=motion_lib,
        )
        env = HumanoidVerseMjlabVectorEnv(
            core,
            include_last_action=self.include_last_action,
            context_length=self.context_length,
            include_history_actor=self.include_history_actor,
            include_history_noaction=self.include_history_noaction,
        )
        env._creation_config = self
        return env, {"unresolved_conf": unresolved_conf, "mjlab_env_cfg": mjlab_cfg}
