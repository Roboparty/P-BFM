"""Assemble actor, history, privileged terrain, and depth observations in draw order."""

import numpy as np
import torch
from torch.utils._pytree import tree_map

from pbfm.motion.observations import compute_humanoid_observations_max
from pbfm.terrains.terrain_observation import flat_zero_observations, observations_from_clearances, reference_ray_index
from pbfm.utils.torch_utils import my_quat_rotate, quat_mul


def _update_obs_noise_curriculum(self) -> None:
    if not self.add_noise_currculum:
        return
    if self.average_episode_length < float(self.config.obs.soft_dof_pos_curriculum_level_down_threshold):
        self.current_noise_curriculum_value *= 1.0 - float(self.config.obs.soft_dof_pos_curriculum_degree)
    elif self.average_episode_length > float(self.config.obs.soft_dof_pos_curriculum_level_up_threshold):
        self.current_noise_curriculum_value *= 1.0 + float(self.config.obs.soft_dof_pos_curriculum_degree)
    self.current_noise_curriculum_value = float(
        np.clip(
            self.current_noise_curriculum_value,
            float(self.config.obs.noise_value_min),
            float(self.config.obs.noise_value_max),
        )
    )


def _apply_obs_scale_noise(self, key: str, value: torch.Tensor) -> torch.Tensor:
    obs_scales = self.config.obs.obs_scales
    noise_scales = self.config.obs.noise_scales
    scale = float(obs_scales.get(key, 1.0))
    noise_scale = 0.0 if self.is_evaluating else float(noise_scales.get(key, 0.0))
    if self.add_noise_currculum:
        noise_scale *= self.current_noise_curriculum_value
    if noise_scale != 0.0:
        value = value + (torch.rand_like(value) * 2.0 - 1.0) * noise_scale
    return value * scale


def _extend_body_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if self.num_extend_bodies == 0:
        return self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel
    rotated_pos = my_quat_rotate(
        self.body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
        self.extend_body_pos_in_parent.reshape(-1, 3),
    ).view(self.num_envs, -1, 3)
    extend_pos = rotated_pos + self.body_pos[:, self.extend_body_parent_ids]
    extend_rot = quat_mul(
        self.body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
        self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
        w_last=True,
    ).view(self.num_envs, -1, 4)
    extend_ang_vel = self.body_ang_vel[:, self.extend_body_parent_ids]
    extend_vel = self.body_vel[:, self.extend_body_parent_ids] + torch.cross(
        extend_ang_vel, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2
    )
    return (
        torch.cat([self.body_pos, extend_pos], dim=1),
        torch.cat([self.body_rot, extend_rot], dim=1),
        torch.cat([self.body_vel, extend_vel], dim=1),
        torch.cat([self.body_ang_vel, extend_ang_vel], dim=1),
    )


def _compute_reference_and_privileged_obs(self) -> None:
    body_pos, body_rot, body_vel, body_ang_vel = _extend_body_state(self)
    self._rigid_body_pos_extend = body_pos
    self._rigid_body_rot_extend = body_rot
    self._rigid_body_vel_extend = body_vel
    self._rigid_body_ang_vel_extend = body_ang_vel

    motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
    motion_res = self._motion_lib.get_motion_state(
        self.motion_ids,
        motion_times,
        offset=self.env_origins + self._terrain_motion_offsets,
    )
    self.ref_body_pos_extend = motion_res["rg_pos_t"]
    self.ref_body_rot_extend = motion_res["rg_rot_t"]
    self.ref_body_vel_extend = motion_res["body_vel_t"]
    self.ref_body_ang_vel_extend = motion_res["body_ang_vel_t"]
    self.dif_global_body_pos = self.ref_body_pos_extend - body_pos
    self.dif_joint_angles = motion_res["dof_pos"] - self.dof_pos
    self.dif_joint_velocities = motion_res["dof_vel"] - self.dof_vel
    obs_dict = compute_humanoid_observations_max(
        body_pos,
        body_rot,
        body_vel,
        body_ang_vel,
        local_root_obs=True,
        root_height_obs=bool(self.config.obs.get("root_height_obs", True)),
    )
    if self.terrain_enabled and "root_height" in obs_dict:
        self._latest_terrain_observations = _terrain_observations(self)
        root_clearance, _terrain_actor, _terrain_priv = self._latest_terrain_observations
        obs_dict["root_height"] = root_clearance
    self._max_local_self = torch.cat([v for v in obs_dict.values()], dim=-1)


def _raw_actor_obs(self) -> dict[str, torch.Tensor]:
    if not self._creation_config.evaluation_fast_path:
        _compute_reference_and_privileged_obs(self)
    dof_pos_rel = self.dof_pos - (self.default_dof_pos + self.default_dof_pos_offset)
    obs_data = {
        "actions": _apply_obs_scale_noise(self, "actions", self.actions),
        "base_ang_vel": _apply_obs_scale_noise(self, "base_ang_vel", self.base_ang_vel),
        "dof_pos": _apply_obs_scale_noise(self, "dof_pos", dof_pos_rel),
        "dof_vel": _apply_obs_scale_noise(self, "dof_vel", self.dof_vel),
        "projected_gravity": _apply_obs_scale_noise(self, "projected_gravity", self.projected_gravity),
    }
    if not self._creation_config.evaluation_fast_path:
        obs_data["max_local_self"] = _apply_obs_scale_noise(self, "max_local_self", self._max_local_self)
    history_config = self.config.obs.obs_auxiliary["history_actor"]
    history_tensors = []
    for key in sorted(history_config.keys()):
        history_length = history_config[key]
        history_tensor = self.history_handler.query(key)[:, :history_length]
        history_tensors.append(history_tensor.reshape(history_tensor.shape[0], -1))
    history_actor = torch.cat(history_tensors, dim=1)
    history_actor = _apply_obs_scale_noise(self, "history_actor", history_actor)
    raw = {
        **obs_data,
        "history_actor": history_actor,
    }
    self.obs_buf_dict_raw = {"actor_obs": raw}
    for key in history_config.keys():
        value = obs_data[key]
        self.history_handler.add(key, value)
    return raw


def _terrain_sensor_clearances(self) -> torch.Tensor:
    """Return pelvis-to-terrain clearances and initialize the local origin ray."""
    sensor = self.mjlab_env.scene.sensors.get("terrain_height")
    if sensor is None:
        raise RuntimeError("terrain-aware environment is missing terrain_height sensor")
    clearances = sensor.data.heights
    if clearances.ndim == 3:
        clearances = clearances[:, 0]
    if self._terrain_reference_index is None:
        offsets = sensor._local_offsets  # MJLab exposes no public pattern-offset accessor.
        if offsets is None:
            raise RuntimeError("terrain_height sensor has not been initialized")
        self._terrain_reference_index = reference_ray_index(offsets)
    return clearances


def _terrain_observations(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return root clearance, actor ranges, and privileged terrain geometry."""
    if self.terrain_observation_mode == "flat_zero":
        # The priority evaluator uses a true MuJoCo plane at world z=0.
        pelvis_clearance = self.body_pos[:, 0, 2:3]
        return flat_zero_observations(pelvis_clearance, self._terrain_dimension)
    clearances = _terrain_sensor_clearances(self)
    return observations_from_clearances(clearances, self._terrain_reference_index, clip=self._terrain_priv_clip)


def _terrain_actor_obs(self) -> torch.Tensor:
    """Return pelvis-to-terrain ranges for the terrain-conditioned actor."""
    _root_clearance, terrain_actor, _terrain_priv = _terrain_observations(self)
    return terrain_actor


def _log_terrain_priv_stats(self, terrain_priv: torch.Tensor) -> None:
    terrain_ids = self._current_terrain_type_ids()
    for terrain_id, name in enumerate(self.terrain_component_names):
        values = terrain_priv[terrain_ids == terrain_id]
        if values.numel() == 0:
            continue
        print(
            "[INFO] terrain_priv: "
            f"terrain={name}, shape={tuple(terrain_priv.shape)}, count={values.shape[0]}, "
            f"min={values.min().item():.4f}, max={values.max().item():.4f}, "
            f"mean={values.mean().item():.4f}, std={values.std(unbiased=False).item():.4f}",
            flush=True,
        )


def get_observation(self, *, to_numpy: bool = True, include_last_action: bool = True, include_history_actor: bool = True):
    raw_obs = _raw_actor_obs(self)
    obs = {
        "state": torch.cat([raw_obs["dof_pos"], raw_obs["dof_vel"], raw_obs["projected_gravity"], raw_obs["base_ang_vel"]], dim=-1),
    }
    if not self._creation_config.evaluation_fast_path:
        obs["privileged_state"] = raw_obs["max_local_self"]
    if include_last_action:
        obs["last_action"] = raw_obs["actions"]
    obs["time"] = self.episode_length_buf.unsqueeze(-1)
    if include_history_actor:
        obs["history_actor"] = raw_obs["history_actor"]
    # Behavior heading is a workspace/inference command, not privileged
    # simulator state. The environment contributes only a fixed-shape
    # placeholder so the observation contract is stable; collectors
    # overwrite it after atomically advancing their BehaviorContext.
    obs["heading"] = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
    if self._creation_config.velocity_context_enabled:
        obs["velocity_command"] = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
    if self._creation_config.include_velocity_estimator_target:
        from pbfm.agents.behavior_context import root_heading_xy
        from pbfm.agents.velocity_context import world_to_yaw
        # Same instant as this observation (including reset observations).
        # Training-only critic/supervision field: Actor must never read it.
        world_velocity = self.robot_root_states[:, 7:10]
        xy = world_to_yaw(world_velocity[:, :2], root_heading_xy(self.base_quat))
        obs["velocity_target_yaw"] = torch.cat((xy, world_velocity[:, 2:3]), dim=-1).detach().clone()
    if self.terrain_enabled:
        if self.direct_depth_enabled:
            if self._direct_depth_runtime is None:
                raise RuntimeError("direct-depth runtime was not initialized")
            obs["depth_image"] = self._direct_depth_runtime.observation()
        else:
            terrain_observations = self._latest_terrain_observations or _terrain_observations(self)
            _root_clearance, terrain_actor, terrain_priv = terrain_observations
            obs["terrain_actor"] = terrain_actor
        if not self._creation_config.evaluation_fast_path:
            terrain_observations = self._latest_terrain_observations or _terrain_observations(self)
            _root_clearance, _terrain_actor, terrain_priv = terrain_observations
            obs["terrain_priv"] = terrain_priv
            if not self._terrain_stats_logged:
                _log_terrain_priv_stats(self, terrain_priv)
                self._terrain_stats_logged = True
    if to_numpy:
        obs = tree_map(lambda x: x.detach().cpu().numpy(), obs)
    return obs
