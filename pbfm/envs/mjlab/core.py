"""PBFM simulation state, motion lifecycle, stepping, and ordered episode resets."""

import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import OmegaConf

from pbfm.envs.mjlab.history import HistoryHandler as HVHistoryHandler
from pbfm.motion.library.motion_lib_robot import MotionLibRobot
from pbfm.perception.instinct_direct_depth import RP1DirectDepthRuntime
from pbfm.terrains import terrain_component_names
from pbfm.terrains.terrain_observation import RobotCentricGridPatternCfg
from pbfm.utils.torch_utils import quat_from_angle_axis, quat_mul, quat_rotate_inverse, wxyz_to_xyzw, xyzw_to_wxyz

from . import aux_rewards, observations, reset
from .aux_rewards import fixed_sole_weights
from .config import _action_target_scale, _default_joint_pos, _rp1_direct_depth_config
from .reset import (
    RESET_REGION_ID,
    RESET_REGION_NAMES,
    _random_yaw_quaternions,
    _small_random_quaternions,
    pre_descent_ground_probe_z,
    reset_region_labels_are_semantic,
    rotate_root_motion_by_yaw,
    sample_lie_down_reset_mask,
    separated_stairs_upright_reset_mask,
    terrain_family_reset_histogram,
    terrain_grid_boundary_margin,
    terrain_grid_coordinates,
)
from .task_config import _to_list

if TYPE_CHECKING:
    from .config import HumanoidVerseMjlabConfig


class _MjlabSimulatorView:
    """Compatibility view for code that expects ``env._env.simulator``."""

    def __init__(self, core: "HumanoidVerseMjlabCore") -> None:
        self._core = core
        self._body_list = list(core.body_names)
        self.__class__.__name__ = "MJLab"

    def refresh(self) -> None:
        core = self._core
        self.dof_pos = core.dof_pos
        self.dof_vel = core.dof_vel
        self.dof_state = torch.stack((core.dof_pos, core.dof_vel), dim=-1)
        self.robot_root_states = core.robot_root_states
        self.base_quat = core.base_quat
        self._rigid_body_pos = core.body_pos
        self._rigid_body_rot = core.body_rot
        self._rigid_body_vel = core.body_vel
        self._rigid_body_ang_vel = core.body_ang_vel
        self.contact_forces = core.contact_forces
        self.dof_pos_limits = core.dof_pos_limits
        self.hard_dof_pos_limits = core.hard_dof_pos_limits

    def render(self):
        return self._core.mjlab_env.render()


class HumanoidVerseMjlabCore:
    def __init__(
        self,
        hv_config,
        mjlab_env,
        *,
        creation_config: "HumanoidVerseMjlabConfig",
        motion_lib_override=None,
    ) -> None:
        if mjlab_env.cfg.auto_reset:
            raise ValueError("PBFM owns task reset; mjlab auto_reset must be False")
        self.config = hv_config
        self.mjlab_env = mjlab_env
        self.robot = mjlab_env.scene["robot"]
        self.device = str(mjlab_env.device)
        self.num_envs = int(mjlab_env.num_envs)
        self.dt = float(mjlab_env.step_dt)
        self.sim_dt = float(mjlab_env.physics_dt)
        self._creation_config = creation_config

        self.dof_names = tuple(_to_list(hv_config.robot.dof_names))
        self.body_names = tuple(_to_list(hv_config.robot.body_names))
        self.num_dof = len(self.dof_names)
        self.num_dofs = self.num_dof
        self.num_bodies = len(self.body_names)
        self.dim_actions = self.num_dof
        self.env_origins = mjlab_env.scene.env_origins
        self.terrain_enabled = bool(hv_config.terrain.get("enabled", False))
        self.terrain_mode = str(hv_config.terrain.get("terrain_type", "plane")) if self.terrain_enabled else "plane"
        self.direct_depth_config = _rp1_direct_depth_config(hv_config)
        self.direct_depth_enabled = self.direct_depth_config is not None
        self._direct_depth_runtime = (
            RP1DirectDepthRuntime(
                self.num_envs,
                self.device,
                self.direct_depth_config,
                enable_noise=not bool(creation_config.disable_obs_noise),
                fixed_delay_frames=creation_config.fixed_direct_depth_delay_frames,
            )
            if self.direct_depth_config is not None
            else None
        )
        self.terrain_observation_mode = (
            str(hv_config.terrain.terrain_priv.get("mode", "raycast")) if self.terrain_enabled else "raycast"
        )
        self.terrain_component_names = terrain_component_names(self.terrain_mode)
        self._terrain_stats_logged = False
        self._terrain_reference_index: int | None = None
        self._latest_terrain_observations: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._terrain_priv_clip = (
            float(hv_config.terrain.terrain_priv.clip) if self.terrain_enabled else 0.0
        )
        self._terrain_grid_cfg = (
            RobotCentricGridPatternCfg(
                x_min=float(hv_config.terrain.terrain_priv.x_min),
                x_max=float(hv_config.terrain.terrain_priv.x_max),
                y_min=float(hv_config.terrain.terrain_priv.y_min),
                y_max=float(hv_config.terrain.terrain_priv.y_max),
                resolution=float(hv_config.terrain.terrain_priv.resolution),
            )
            if self.terrain_enabled
            else None
        )
        self._terrain_dimension = (
            self._terrain_grid_cfg.dimension
            if self.terrain_enabled
            else 0
        )
        self._terrain_motion_offsets = torch.zeros(self.num_envs, 3, device=self.device)
        self._reset_region_ids = torch.full(
            (self.num_envs,), RESET_REGION_ID["flat_center"], device=self.device, dtype=torch.long
        )
        self._reset_region_counts = torch.zeros(len(RESET_REGION_NAMES), device=self.device, dtype=torch.long)
        self._reset_terrain_family_counts = torch.zeros(
            len(self.terrain_component_names), device=self.device, dtype=torch.long
        )
        self._reset_region_labels_are_semantic = reset_region_labels_are_semantic(
            self.terrain_component_names
        )
        self._lie_down_reset_count = torch.zeros((), device=self.device, dtype=torch.long)
        terrain_generator_cfg = None
        if self.terrain_enabled and self.terrain_mode != "plane":
            terrain_generator_cfg = self.mjlab_env.scene["terrain"].cfg.terrain_generator
        terrain_patch_values = (
            _to_list(terrain_generator_cfg.size) if terrain_generator_cfg is not None else None
        )
        if terrain_patch_values is not None and (
            len(terrain_patch_values) != 2 or min(float(value) for value in terrain_patch_values) <= 0.0
        ):
            raise ValueError("terrain.patch_size must contain two positive values")
        self._terrain_patch_size = (
            torch.tensor(terrain_patch_values, device=self.device, dtype=torch.float32)
            if terrain_patch_values is not None
            else None
        )
        if self._terrain_patch_size is not None:
            terrain_origins = self.mjlab_env.scene["terrain"].terrain_origins
            self._terrain_grid_rows, self._terrain_grid_cols = terrain_origins.shape[:2]
        else:
            self._terrain_grid_rows = self._terrain_grid_cols = 0
        self._terrain_global_border_width = (
            float(terrain_generator_cfg.border_width) if terrain_generator_cfg is not None else 0.0
        )
        self._last_terrain_tile = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self._terrain_tile_crossing_count = torch.zeros((), device=self.device, dtype=torch.long)
        self._terrain_transition_counts = torch.zeros(
            (len(self.terrain_component_names), len(self.terrain_component_names)),
            device=self.device,
            dtype=torch.long,
        )
        self._terrain_reset_footprint_indices: torch.Tensor | None = None
        self._terrain_boundary_min = torch.tensor(float("inf"), device=self.device)
        self._terrain_boundary_violation_count = torch.zeros((), device=self.device, dtype=torch.long)
        if self._terrain_grid_cfg is not None:
            corners_x = max(abs(self._terrain_grid_cfg.x_min), abs(self._terrain_grid_cfg.x_max))
            corners_y = max(abs(self._terrain_grid_cfg.y_min), abs(self._terrain_grid_cfg.y_max))
            self._terrain_sensor_radius = float((corners_x**2 + corners_y**2) ** 0.5)
        else:
            self._terrain_sensor_radius = 0.0
        self._terrain_boundary_required = self._terrain_sensor_radius + (
            float(hv_config.terrain.coverage.get("boundary_safety_margin", 0.5))
            if self.terrain_enabled
            else 0.0
        )
        self._terrain_fail_on_boundary = bool(
            self.terrain_enabled and hv_config.terrain.coverage.get("fail_on_boundary_violation", True)
        )

        mjlab_joint_names = tuple(self.robot.joint_names)
        mjlab_body_names = tuple(self.robot.body_names)
        missing_joints = [name for name in self.dof_names if name not in mjlab_joint_names]
        missing_bodies = [name for name in self.body_names if name not in mjlab_body_names]
        if missing_joints:
            raise ValueError(f"MJLab robot asset is missing joints from HumanoidVerse config: {missing_joints}")
        if missing_bodies:
            raise ValueError(f"MJLab robot asset is missing bodies from HumanoidVerse config: {missing_bodies}")
        self._joint_ids = torch.tensor([mjlab_joint_names.index(name) for name in self.dof_names], device=self.device, dtype=torch.long)
        self._body_ids = torch.tensor([mjlab_body_names.index(name) for name in self.body_names], device=self.device, dtype=torch.long)

        action_term = self.mjlab_env.action_manager.get_term("actions")
        action_target_names = tuple(action_term.target_names)
        if len(action_target_names) != self.num_dof or set(action_target_names) != set(self.dof_names):
            raise ValueError(
                "MJLab action target joints do not match HumanoidVerse dof_names: "
                f"target_names={list(action_target_names)}, dof_names={list(self.dof_names)}"
            )
        self._action_term_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in action_target_names], device=self.device, dtype=torch.long
        )
        if action_target_names != self.dof_names:
            print(
                "[INFO] MJLab action target order differs from HumanoidVerse dof order: "
                f"action_target_names={list(action_target_names)}",
                flush=True,
            )

        self.default_dof_pos = _default_joint_pos(hv_config).to(self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.default_dof_pos_offset = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.action_target_scale = _action_target_scale(hv_config).to(self.device).unsqueeze(0)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        lower = torch.tensor(_to_list(hv_config.robot.dof_pos_lower_limit_list), dtype=torch.float32, device=self.device)
        upper = torch.tensor(_to_list(hv_config.robot.dof_pos_upper_limit_list), dtype=torch.float32, device=self.device)
        self.hard_dof_pos_limits = torch.stack((lower, upper), dim=-1)
        limit_scale = float(hv_config.rewards.reward_limit.soft_dof_pos_limit)
        center = (lower + upper) * 0.5
        radius = (upper - lower) * 0.5 * limit_scale
        self.dof_pos_limits = torch.stack((center - radius, center + radius), dim=-1)
        self.torque_limits = torch.tensor(_to_list(hv_config.robot.dof_effort_limit_list), device=self.device, dtype=torch.float32)
        self.dof_vel_limits = torch.tensor(_to_list(hv_config.robot.dof_vel_limit_list), device=self.device, dtype=torch.float32)

        self.actions = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.applied_actions = torch.zeros_like(self.actions)
        self._ctrl_delay_enabled = bool(self.config.domain_rand.get("randomize_ctrl_delay", False))
        delay_range = _to_list(self.config.domain_rand.get("ctrl_delay_step_range", [0, 0]))
        if len(delay_range) != 2:
            raise ValueError(f"domain_rand.ctrl_delay_step_range must have two entries, got {delay_range}")
        self._ctrl_delay_min_step = int(delay_range[0]) if self._ctrl_delay_enabled else 0
        self._ctrl_delay_max_step = int(delay_range[1]) if self._ctrl_delay_enabled else 0
        if self._ctrl_delay_min_step < 0 or self._ctrl_delay_max_step < self._ctrl_delay_min_step:
            raise ValueError(f"domain_rand.ctrl_delay_step_range must satisfy 0 <= min <= max, got {delay_range}")
        self._ctrl_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ctrl_delay_env_indices = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._ctrl_delay_buffer = torch.zeros(
            self._ctrl_delay_max_step + 1,
            self.num_envs,
            self.num_dof,
            device=self.device,
        )
        if self._ctrl_delay_enabled:
            print(
                "[INFO] MJLab motor action delay randomization enabled: "
                f"ctrl_delay_step_range=[{self._ctrl_delay_min_step}, {self._ctrl_delay_max_step}]",
                flush=True,
            )
        self.torques = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.extras: dict[str, Any] = {"aux_rewards": {}}

        self._init_reward_scales()
        self.terrain_aware_auxiliary = bool(hv_config.rewards.get("terrain_aware_auxiliary", False))
        self._validate_aux_reward_semantics(hv_config)
        self.feet_indices = torch.tensor([self.body_names.index(name) for name in hv_config.robot.contact_bodies], device=self.device, dtype=torch.long)
        self.virtual_terrain_edges = None
        if self._creation_config.virtual_edge_penalty_enabled:
            from pbfm.terrains.edge_penalty import VirtualTerrainEdges, exposed_box_edges, foot_volume_points
            terrain_groups = (0, 5) if self.terrain_mode in {'rp1_simple', 'plane', 'flat'} else (5,)
            edges = exposed_box_edges(self.mjlab_env.sim.mj_model, group=terrain_groups)
            self.virtual_terrain_edges = VirtualTerrainEdges(edges, device=self.device,
                radius=self._creation_config.virtual_edge_radius)
            self._edge_foot_points = foot_volume_points(self.device)
            print(f'[virtual edges] {len(edges)} exposed box-top segments, radius={self._creation_config.virtual_edge_radius}; no physical colliders', flush=True)
        self.torso_index = self.body_names.index(hv_config.robot.torso_name)
        penalized = []
        for pattern in _to_list(hv_config.robot.penalize_contacts_on):
            penalized.extend([i for i, name in enumerate(self.body_names) if pattern in name])
        self.penalised_contact_indices = torch.tensor(sorted(set(penalized)), device=self.device, dtype=torch.long)
        self.robot_weight = (
            self._read_robot_weight()
            if self.terrain_aware_auxiliary
            else torch.ones(self.num_envs, device=self.device)
        )
        self.left_ankle_dof_indices = torch.tensor([self.dof_names.index(n) for n in hv_config.robot.left_ankle_dof_names], device=self.device)
        self.right_ankle_dof_indices = torch.tensor([self.dof_names.index(n) for n in hv_config.robot.right_ankle_dof_names], device=self.device)

        self._init_motion_extend()
        self.is_evaluating = False
        self.average_episode_length = 0.0
        self.last_episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.num_compute_average_epl = float(self.config.rewards.num_compute_average_epl)
        self.add_noise_currculum = bool(self.config.obs.get("add_noise_currculum", False))
        self.current_noise_curriculum_value = float(self.config.obs.get("noise_initial_value", 1.0))
        self._init_motion_lib(motion_lib_override)
        self.history_handler = HVHistoryHandler(self.num_envs, hv_config.obs.obs_auxiliary, hv_config.obs.obs_dims, self.device)
        self.feet_height_sensor = self.mjlab_env.scene.sensors.get("foot_height_scanners")
        self.feet_height_dim = 0
        if self.feet_height_sensor is not None:
            if self.feet_height_sensor.num_frames != 2:
                raise ValueError(
                    f"Expected two foot height scan frames, got {self.feet_height_sensor.num_frames}"
                )
            self.feet_height_dim = int(self.feet_height_sensor.num_rays_per_frame)
            self._feet_at_plane_weights = fixed_sole_weights(self.feet_height_sensor._local_offsets[:, :2])
        self.use_contact_in_obs_max = bool(hv_config.get("use_contact_in_obs_max", False))
        self.simulator = _MjlabSimulatorView(self)

        self._refresh_state()
        self.simulator.refresh()

    def _init_reward_scales(self) -> None:
        self.reward_scales = dict(OmegaConf.to_container(self.config.rewards.reward_scales, resolve=True))
        self.reward_scales = {k: float(v) * self.dt for k, v in self.reward_scales.items() if float(v) != 0.0}
        self.reward_names = list(self.reward_scales.keys())
        self.reward_penalty_reward_names = set(_to_list(self.config.rewards.reward_penalty_reward_names))
        self.use_reward_penalty_curriculum = bool(self.config.rewards.reward_penalty_curriculum)
        self.reward_penalty_scale = float(self.config.rewards.reward_initial_penalty_scale)

    def _validate_aux_reward_semantics(self, hv_config) -> None:
        contact_bodies = _to_list(hv_config.robot.get("contact_bodies", None))
        if len(contact_bodies) < 2:
            raise ValueError(
                "robot.contact_bodies must contain at least 2 bodies because the current MJLab reward "
                "implementation computes biped foot auxiliary terms unconditionally"
            )

        if "penalty_ankle_roll" in self.reward_scales:
            missing_fields = []
            if len(_to_list(hv_config.robot.get("left_ankle_dof_names", None))) < 2:
                missing_fields.append("robot.left_ankle_dof_names")
            if len(_to_list(hv_config.robot.get("right_ankle_dof_names", None))) < 2:
                missing_fields.append("robot.right_ankle_dof_names")
            if missing_fields:
                raise ValueError(
                    f"{', '.join(missing_fields)} must contain at least 2 joints because reward 'penalty_ankle_roll' is enabled"
                )

    def _update_average_episode_length(self, env_ids: torch.Tensor) -> None:
        if self.is_evaluating or len(env_ids) == 0:
            return
        current = torch.mean(self.last_episode_length_buf[env_ids].float()).item()
        ratio = min(float(len(env_ids)) / max(self.num_compute_average_epl, 1.0), 1.0)
        self.average_episode_length = self.average_episode_length * (1.0 - ratio) + current * ratio

    def _update_reward_penalty_curriculum(self) -> None:
        if not self.use_reward_penalty_curriculum:
            return
        if self.average_episode_length < float(self.config.rewards.reward_penalty_level_down_threshold):
            self.reward_penalty_scale *= 1.0 - float(self.config.rewards.reward_penalty_degree)
        elif self.average_episode_length > float(self.config.rewards.reward_penalty_level_up_threshold):
            self.reward_penalty_scale *= 1.0 + float(self.config.rewards.reward_penalty_degree)
        self.reward_penalty_scale = float(
            np.clip(
                self.reward_penalty_scale,
                float(self.config.rewards.reward_min_penalty_scale),
                float(self.config.rewards.reward_max_penalty_scale),
            )
        )

    def _init_motion_lib(self, motion_lib_override=None) -> None:
        self.config.robot.motion.step_dt = self.dt
        if motion_lib_override is None:
            self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
            self._motion_lib.load_motions_for_training(max_num_seqs=self.num_envs)
        else:
            self._motion_lib = motion_lib_override
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions
        self._resample_motion_time_and_ids(torch.arange(self.num_envs, device=self.device))

    def _init_motion_extend(self) -> None:
        extend_parent_ids, extend_pos, extend_rot = [], [], []
        for item in _to_list(self.config.robot.motion.extend_config):
            extend_parent_ids.append(self.body_names.index(item["parent_name"]))
            extend_pos.append(item["pos"])
            extend_rot.append(item["rot"])
        self.num_extend_bodies = len(extend_parent_ids)
        if self.num_extend_bodies:
            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1, 1)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1, 1)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.body_names = tuple(list(self.body_names) + [item["joint_name"] for item in _to_list(self.config.robot.motion.extend_config)])
        else:
            self.extend_body_parent_ids = torch.empty(0, device=self.device, dtype=torch.long)
        self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, device=self.device)

    def _resample_motion_time_and_ids(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_start_times[env_ids] = 0.0
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])

    def _randomize_default_dof_pos_offset(self, env_ids: torch.Tensor) -> None:
        if bool(self.config.domain_rand.get("randomize_default_dof_pos", False)):
            offset_range = self.config.domain_rand.default_dof_pos_noise_range
            self.default_dof_pos_offset[env_ids] = torch.empty(
                len(env_ids), self.num_dof, device=self.device, dtype=torch.float32
            ).uniform_(float(offset_range[0]), float(offset_range[1]))
        else:
            self.default_dof_pos_offset[env_ids] = 0.0

    def _refresh_state(self) -> None:
        data = self.robot.data
        self.dof_pos = data.joint_pos[:, self._joint_ids].clone()
        self.dof_vel = data.joint_vel[:, self._joint_ids].clone()
        root_pose_w = data.root_link_pose_w.clone()
        root_vel_w = data.root_link_vel_w.clone()
        self.base_quat = wxyz_to_xyzw(root_pose_w[:, 3:7])
        self.robot_root_states = torch.cat([root_pose_w[:, :3], self.base_quat, root_vel_w], dim=-1)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, root_vel_w[:, :3], w_last=True)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, root_vel_w[:, 3:6], w_last=True)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec, w_last=True)
        body_pose = data.body_link_pose_w[:, self._body_ids].clone()
        body_vel = data.body_link_vel_w[:, self._body_ids].clone()
        self.body_pos = body_pose[..., :3]
        self.body_rot = wxyz_to_xyzw(body_pose[..., 3:7])
        self.body_vel = body_vel[..., :3]
        self.body_ang_vel = body_vel[..., 3:6]
        self.torques = data.qfrc_actuator[:, self._joint_ids].clone()
        self.contact_forces = self._read_contact_forces()
        if self.terrain_aware_auxiliary:
            self.contact_force_history = self._read_contact_force_history()
        self.episode_length_buf = self.mjlab_env.episode_length_buf.clone()

    def _read_contact_forces(self) -> torch.Tensor:
        forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        sensor = self.mjlab_env.scene.sensors.get("body_contact")
        if sensor is None:
            return forces
        contact_data = sensor.data
        if contact_data.force is None:
            return forces
        names = [name.split("/")[-1] for name in sensor.primary_names]
        for i, name in enumerate(names):
            if name in self.body_names[: self.num_bodies]:
                forces[:, self.body_names.index(name), :] = contact_data.force[:, i, :]
        return forces

    def _read_contact_force_history(self) -> torch.Tensor:
        sensor = self.mjlab_env.scene.sensors.get("body_contact")
        history_length = int(self.config.simulator.config.sim.control_decimation)
        history = torch.zeros(self.num_envs, self.num_bodies, history_length, 3, device=self.device)
        if sensor is None or sensor.data.force_history is None:
            return history
        names = [name.split("/")[-1] for name in sensor.primary_names]
        for i, name in enumerate(names):
            if name in self.body_names[: self.num_bodies]:
                history[:, self.body_names.index(name), :, :] = sensor.data.force_history[:, i, :, :]
        return history

    def _read_foot_contact_force_history(self) -> torch.Tensor:
        history = self._read_contact_force_history()
        return history[:, self.feet_indices]

    def _read_foot_collision_force_history(self) -> torch.Tensor:
        """Return separate world-frame contact histories for each sole geometry/slot."""

        sensor = self.mjlab_env.scene.sensors.get("foot_collision_contacts")
        history_length = int(self.config.simulator.config.sim.control_decimation)
        if sensor is None or sensor.data.force_history is None:
            return torch.zeros(self.num_envs, 0, history_length, 3, device=self.device)
        return sensor.data.force_history

    def _read_robot_weight(self) -> torch.Tensor:
        scene_body_ids = self.robot.data.indexing.body_ids[self._body_ids]
        body_mass = self.mjlab_env.sim.model.body_mass
        if body_mass.ndim == 1:
            robot_mass = body_mass[scene_body_ids].sum().expand(self.num_envs)
        else:
            robot_mass = body_mass[:, scene_body_ids].sum(dim=1)
        gravity = torch.as_tensor(
            self.mjlab_env.cfg.sim.mujoco.gravity,
            device=self.device,
            dtype=robot_mass.dtype,
        ).norm()
        return robot_mass * gravity

    def _current_terrain_tiles(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._terrain_patch_size is None:
            zeros = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            return zeros, zeros, torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        return terrain_grid_coordinates(
            self.robot_root_states[:, :2],
            self._terrain_patch_size,
            num_rows=self._terrain_grid_rows,
            num_cols=self._terrain_grid_cols,
        )

    def _current_terrain_type_ids(self) -> torch.Tensor:
        _rows, cols, inside = self._current_terrain_tiles()
        return torch.where(inside, cols, torch.full_like(cols, -1))

    def _update_terrain_tile_transitions(self) -> None:
        if self._terrain_patch_size is None:
            return
        rows, cols, inside = self._current_terrain_tiles()
        current = rows * self._terrain_grid_cols + cols
        changed = inside & (self._last_terrain_tile >= 0) & (current != self._last_terrain_tile)
        previous_types = torch.remainder(self._last_terrain_tile.clamp_min(0), self._terrain_grid_cols)
        current_types = cols
        flat_indices = previous_types * len(self.terrain_component_names) + current_types
        counts = torch.zeros_like(self._terrain_transition_counts).flatten()
        counts.scatter_add_(0, flat_indices, changed.to(dtype=counts.dtype))
        self._terrain_transition_counts += counts.reshape_as(self._terrain_transition_counts)
        self._terrain_tile_crossing_count += torch.count_nonzero(changed)
        self._last_terrain_tile[inside] = current[inside]

    def _terrain_boundary_margin(self) -> torch.Tensor | None:
        if self._terrain_patch_size is None:
            return None
        return terrain_grid_boundary_margin(
            self.robot_root_states[:, :2],
            self._terrain_patch_size,
            num_rows=self._terrain_grid_rows,
            num_cols=self._terrain_grid_cols,
            border_width=self._terrain_global_border_width,
        )

    def _check_terrain_boundary(self) -> torch.Tensor:
        margin = self._terrain_boundary_margin()
        if margin is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._terrain_boundary_min = torch.minimum(self._terrain_boundary_min, margin.min())
        violations = margin < self._terrain_boundary_required
        self._terrain_boundary_violation_count += torch.count_nonzero(violations)
        if self._terrain_fail_on_boundary and torch.any(violations).item():
            bad_margin = margin[violations].min().item()
            raise RuntimeError(
                "terrain boundary invariant failed: a robot or its sensing footprint left the global terrain grid; "
                f"minimum margin={bad_margin:.6f}m, required={self._terrain_boundary_required:.6f}m"
            )
        return violations

    def get_observation(self, *, to_numpy: bool = True, include_last_action: bool = True, include_history_actor: bool = True, env_ids=None):
        return observations.get_observation(
            self,
            to_numpy=to_numpy,
            include_last_action=include_last_action,
            include_history_actor=include_history_actor,
            env_ids=env_ids,
        )

    def _normalized_action(self, actions: torch.Tensor) -> torch.Tensor:
        if bool(self.config.robot.control.normalize_action):
            actions = actions * float(self.config.robot.control.normalize_action_to) / float(self.config.robot.control.normalize_action_from)
        return torch.clamp(actions, -float(self.config.robot.control.action_clip_value), float(self.config.robot.control.action_clip_value))

    def _mjlab_action_input(self, actions: torch.Tensor | None = None) -> torch.Tensor:
        action_indices = self._action_term_dof_indices
        action_source = self.actions if actions is None else actions
        return action_source[:, action_indices] + self.default_dof_pos_offset[:, action_indices] / torch.clamp(
            self.action_target_scale[:, action_indices], min=1.0e-6
        )

    def _sample_ctrl_delay(self, env_ids: torch.Tensor) -> None:
        """Sample one integer motor-command delay per episode and clear stale commands."""
        if self._ctrl_delay_enabled:
            self._ctrl_delay_steps[env_ids] = torch.randint(
                self._ctrl_delay_min_step,
                self._ctrl_delay_max_step + 1,
                (len(env_ids),),
                device=self.device,
            )
        else:
            self._ctrl_delay_steps[env_ids] = 0
        self._ctrl_delay_buffer[:, env_ids] = 0.0

    def _apply_ctrl_delay(self, actions: torch.Tensor) -> torch.Tensor:
        """Return each environment's command from its sampled FIFO delay."""
        if not self._ctrl_delay_enabled or self._ctrl_delay_max_step == 0:
            return actions
        self._ctrl_delay_buffer = torch.roll(self._ctrl_delay_buffer, shifts=1, dims=0)
        self._ctrl_delay_buffer[0] = actions
        return self._ctrl_delay_buffer[self._ctrl_delay_steps, self._ctrl_delay_env_indices]

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device, dtype=torch.float32)
        pre_body_vel = (
            self.body_vel.clone()
            if self.terrain_aware_auxiliary and not self._creation_config.evaluation_fast_path
            else None
        )
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.actions[:] = self._normalized_action(actions)
        self.applied_actions[:] = self._apply_ctrl_delay(self.actions)
        mjlab_actions = self._mjlab_action_input(self.applied_actions)
        _, _, terminated, time_outs, _ = self.mjlab_env.step(mjlab_actions)
        if self._direct_depth_runtime is not None:
            self._direct_depth_runtime.append_from_sensor(self.mjlab_env.scene.sensors["g1_direct_depth"])
        self._refresh_state()
        boundary_resets = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.terrain_enabled:
            self._update_terrain_tile_transitions()
            boundary_resets = self._check_terrain_boundary()
        if self._creation_config.evaluation_fast_path:
            reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            aux = {}
        else:
            reward, aux = aux_rewards._compute_reward(self, pre_body_vel=pre_body_vel)
        # The global map boundary is an artificial domain limit, not a behavior failure.
        # Truncate and reset before an invalid off-terrain transition can continue.
        time_outs = torch.logical_or(time_outs.bool(), boundary_resets)
        reset = torch.logical_or(terminated.bool(), time_outs.bool())
        self.reset_buf = reset
        self.time_out_buf = time_outs.bool()
        self.rew_buf = reward
        self.extras["aux_rewards"] = {k: v.clone().detach() for k, v in aux.items()}
        if self.use_reward_penalty_curriculum:
            self.extras["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float32, device=self.device)
            self.extras["average_episode_length"] = torch.tensor(self.average_episode_length, dtype=torch.float32, device=self.device)
        if self.add_noise_currculum:
            self.extras["current_noise_curriculum_value"] = torch.tensor(
                self.current_noise_curriculum_value, dtype=torch.float32, device=self.device
            )
        # The vector adapter saves the real successor before resetting finished lanes.
        self.simulator.refresh()
        return None, reward, reset, {
            "time_outs": time_outs.bool().clone(),
            "boundary_resets": boundary_resets,
            "aux_rewards": self.extras["aux_rewards"],
        }

    def reset_finished(self, env_ids: torch.Tensor) -> None:
        self.last_episode_length_buf[env_ids] = self.episode_length_buf[env_ids]
        self._update_average_episode_length(env_ids)
        self._update_reward_penalty_curriculum()
        observations._update_obs_noise_curriculum(self)
        self.reset_idx(env_ids)

    def reset_all(self, target_states: dict[str, torch.Tensor] | None = None):
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(env_ids, target_states=target_states)
        return None, {}

    def reset_idx(self, env_ids: torch.Tensor, target_states: dict[str, torch.Tensor] | None = None) -> None:
        if len(env_ids) == 0:
            return
        from pbfm.terrains.training_curriculum import apply_reset_curriculum
        apply_reset_curriculum(self, env_ids, target_states)
        self.mjlab_env.reset(env_ids=env_ids)
        self._randomize_default_dof_pos_offset(env_ids)
        ground_probe_z = None
        adaptive_lift_mask = None
        if self.terrain_enabled:
            generic_probe_clearance = float(self.config.terrain.reset.ground_probe_clearance)
            max_ray_distance = float(self.config.terrain.terrain_priv.max_ray_distance)
            if not 0.0 < generic_probe_clearance < max_ray_distance:
                raise ValueError(
                    "terrain.reset.ground_probe_clearance must be positive and below max_ray_distance"
                )
            ground_probe_z = self.env_origins[env_ids, 2] + generic_probe_clearance
        if target_states is not None:
            self._terrain_motion_offsets[env_ids] = 0.0
            root_xyzw = target_states["root_states"][env_ids].to(self.device, dtype=torch.float32)
            desired_clearance = root_xyzw[:, 2] - self.env_origins[env_ids, 2]
            dof_state = target_states["dof_states"][env_ids].to(self.device, dtype=torch.float32)
            joint_pos = dof_state[..., 0]
            joint_vel = dof_state[..., 1]
        else:
            self._resample_motion_time_and_ids(env_ids)
            motion_times = self.motion_start_times[env_ids]
            motion_res = self._motion_lib.get_motion_state(self.motion_ids[env_ids], motion_times, offset=self.env_origins[env_ids])
            root_pos = motion_res["root_pos"]
            root_rot = motion_res["root_rot"]
            root_vel = motion_res["root_vel"]
            root_ang_vel = motion_res["root_ang_vel"]
            yaw_rotation = _random_yaw_quaternions(len(env_ids), self.device)
            root_rot, root_vel, root_ang_vel = rotate_root_motion_by_yaw(
                root_rot,
                root_vel,
                root_ang_vel,
                yaw_rotation,
            )
            desired_clearance = root_pos[:, 2] - self.env_origins[env_ids, 2]
            reset_region_ids = torch.full(
                (len(env_ids),), RESET_REGION_ID["flat_center"], device=self.device, dtype=torch.long
            )
            elevated_reset_mask = torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
            if self.terrain_enabled:
                # LaFAN world XY is arbitrary. Move the yaw-randomized sampled
                # frame to a terrain-aware reset region.
                terrain_shift_xy = self.env_origins[env_ids, :2] - root_pos[:, :2]
                self._terrain_motion_offsets[env_ids] = 0.0
                self._terrain_motion_offsets[env_ids, :2] = terrain_shift_xy
                root_pos = root_pos.clone()
                root_pos[:, :2] += terrain_shift_xy
                reset_region_ids, reset_xy, elevated_reset_mask = reset._sample_terrain_reset_positions(
                    self,
                    env_ids,
                    root_rot,
                    root_vel,
                )
                root_pos[:, :2] = reset_xy
                self._reset_region_ids[env_ids] = reset_region_ids
                if not self.is_evaluating:
                    reset_family_ids = (
                        torch.zeros_like(env_ids)
                        if self._terrain_patch_size is None
                        else self.mjlab_env.scene["terrain"].terrain_types[env_ids]
                    )
                    self._reset_terrain_family_counts += terrain_family_reset_histogram(
                        reset_family_ids,
                        num_families=len(self.terrain_component_names),
                    )
                    self._reset_region_counts += torch.bincount(
                        reset_region_ids, minlength=len(RESET_REGION_NAMES)
                    )
            lie_down_mask = torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
            if self.config.get("lie_down_init", False):
                upright_stairs = separated_stairs_upright_reset_mask(reset_region_ids)
                lie_down_mask = sample_lie_down_reset_mask(
                    torch.rand(len(env_ids), device=self.device),
                    probability=float(getattr(self.config, "lie_down_init_prob", 0.0)),
                    excluded=upright_stairs,
                )
                if not self.is_evaluating:
                    self._lie_down_reset_count += torch.count_nonzero(lie_down_mask)
                if torch.any(lie_down_mask):
                    root_pos = root_pos.clone()
                    root_rot = root_rot.clone()
                    desired_clearance = desired_clearance.clone()
                    desired_clearance[lie_down_mask] = 0.5
                    sign = 1 if random.random() < 0.5 else -1
                    rot_quat = quat_from_angle_axis(
                        torch.tensor(sign * (-torch.pi / 2), device=self.device),
                        torch.tensor([1.0, 0.0, 0.0], device=self.device),
                        w_last=True,
                    )
                    root_rot[lie_down_mask] = quat_mul(
                        rot_quat.expand_as(root_rot[lie_down_mask]), root_rot[lie_down_mask], w_last=True
                    )
            root_pos = root_pos + torch.randn_like(root_pos) * float(self.config.init_noise_scale.root_pos) * float(self.config.noise_to_initial_level)
            desired_clearance = desired_clearance + (
                root_pos[:, 2] - motion_res["root_pos"][:, 2]
            )
            if self.terrain_enabled:
                self._terrain_motion_offsets[env_ids, :2] = root_pos[:, :2] - motion_res["root_pos"][:, :2]
                # Every pose can contain low limbs or a get-up configuration.
                # Lift only when the sampled footprint contains terrain above
                # the pelvis-center ground; perfectly flat resets stay exact.
                adaptive_lift_mask = torch.ones(len(env_ids), device=self.device, dtype=torch.bool)
            if torch.any(elevated_reset_mask):
                selected_env_ids = env_ids[elevated_reset_mask]
                stairs_cfg = self.config.terrain.stairs
                terrain_priv_cfg = self.config.terrain.terrain_priv
                ground_probe_z[elevated_reset_mask] = pre_descent_ground_probe_z(
                    self.env_origins[selected_env_ids, 2],
                    num_steps=int(stairs_cfg.num_steps),
                    max_step_height=max(float(value) for value in stairs_cfg.step_height_range),
                    probe_clearance=float(stairs_cfg.get("pre_descent_ground_probe_clearance", 1.5)),
                    max_ray_distance=float(terrain_priv_cfg.max_ray_distance),
                )
            root_rot = quat_mul(
                _small_random_quaternions(
                    len(env_ids),
                    float(self.config.init_noise_scale.root_rot) * 3.14 / 180.0 * float(self.config.noise_to_initial_level),
                    self.device,
                ),
                root_rot,
                w_last=True,
            )
            root_vel = root_vel + torch.randn_like(root_vel) * float(self.config.init_noise_scale.root_vel) * float(self.config.noise_to_initial_level)
            root_ang_vel = root_ang_vel + torch.randn_like(root_ang_vel) * float(self.config.init_noise_scale.root_ang_vel) * float(
                self.config.noise_to_initial_level
            )
            root_xyzw = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
            joint_pos = motion_res["dof_pos"] + torch.randn_like(motion_res["dof_pos"]) * float(self.config.init_noise_scale.dof_pos) * float(
                self.config.noise_to_initial_level
            )
            joint_vel = motion_res["dof_vel"] + torch.randn_like(motion_res["dof_vel"]) * float(self.config.init_noise_scale.dof_vel) * float(
                self.config.noise_to_initial_level
            )

        if self.terrain_enabled:
            root_xyzw, ground_z = reset._place_reset_on_local_ground(
                self,
                env_ids,
                root_xyzw,
                joint_pos,
                joint_vel,
                desired_clearance,
                probe_world_z=ground_probe_z,
                adaptive_lift_mask=adaptive_lift_mask,
            )
            if target_states is None:
                self._terrain_motion_offsets[env_ids, 2] = ground_z - self.env_origins[env_ids, 2]

        root_wxyz = torch.cat([root_xyzw[:, :3], xyzw_to_wxyz(root_xyzw[:, 3:7]), root_xyzw[:, 7:13]], dim=-1)
        self.robot.write_root_state_to_sim(root_wxyz, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self._joint_ids, env_ids=env_ids)
        self.mjlab_env.scene.write_data_to_sim()
        self.mjlab_env.sim.forward()
        sensors_invalidated = False
        if self.terrain_enabled and self.terrain_observation_mode == "raycast":
            # The ground probe above populated Sensor.data's per-step cache.
            # Invalidate it before sensing the final reset pose.
            self.mjlab_env.scene.sensors["terrain_height"].update(0.0)
            sensors_invalidated = True
        if self._direct_depth_runtime is not None:
            depth_sensor = self.mjlab_env.scene.sensors["g1_direct_depth"]
            randomize_installation = getattr(depth_sensor, "randomize_installation", None)
            if randomize_installation is not None:
                randomize_installation(env_ids)
            depth_sensor.update(0.0)
            sensors_invalidated = True
        if sensors_invalidated:
            self.mjlab_env.sim.sense()
        if self._direct_depth_runtime is not None:
            self._direct_depth_runtime.reset_from_sensor(
                self.mjlab_env.scene.sensors["g1_direct_depth"], env_ids
            )
        self.mjlab_env._manual_reset_pending[env_ids] = False
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.applied_actions[env_ids] = 0.0
        self._sample_ctrl_delay(env_ids)
        self.history_handler.reset(env_ids)
        self._refresh_state()
        if self.terrain_enabled and self._terrain_patch_size is not None:
            rows, cols, inside = self._current_terrain_tiles()
            current = rows * self._terrain_grid_cols + cols
            selected_inside = inside[env_ids]
            self._last_terrain_tile[env_ids[selected_inside]] = current[env_ids[selected_inside]]
            self._last_terrain_tile[env_ids[~selected_inside]] = -1
        self.simulator.refresh()

    def set_is_evaluating(self, global_rank: int = 0):
        self.is_evaluating = True
        self.begin_seq_motion_samples(global_rank)

    def begin_seq_motion_samples(self, global_rank: int = 0):
        self._motion_lib.load_motions_for_evaluation(start_idx=global_rank * self.num_envs)
        self.reset_all()

    def set_is_training(self):
        self.is_evaluating = False
        self.resample_motion()

    def resample_motion(self):
        self._motion_lib.load_motions_for_training(max_num_seqs=self.num_envs)
        self.reset_all()

    def close(self):
        return self.mjlab_env.close()
