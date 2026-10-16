"""Training auxiliary rewards from the current simulation state and contact history."""

import torch
import torch.nn.functional as F

from pbfm.terrains.terrain_height_sensor import clipped_height_clearance
from pbfm.utils.torch_utils import my_quat_rotate, quat_rotate_inverse, wrap_to_pi


def peak_contact_force(force_history: torch.Tensor) -> torch.Tensor:
    """Select each body's strongest net contact force over a policy step."""
    if force_history.ndim != 4 or force_history.shape[-1] != 3:
        raise ValueError(f"force_history must have shape [B, N, H, 3], got {tuple(force_history.shape)}")
    peak_index = torch.linalg.vector_norm(force_history, dim=-1).argmax(dim=-1, keepdim=True)
    gather_index = peak_index.unsqueeze(-1).expand(*peak_index.shape, 3)
    return torch.gather(force_history, dim=2, index=gather_index).squeeze(2)


def body_contact_severity(
    contact_force: torch.Tensor,
    pre_body_vel: torch.Tensor,
    robot_weight: torch.Tensor,
    *,
    force_threshold_ratio: float = 0.05,
    smooth_l1_beta: float = 0.25,
    impact_velocity_scale: float = 1.0,
    impact_velocity_cap: float = 2.0,
    severity_cap: float = 4.0,
) -> torch.Tensor:
    """Continuous loaded-contact severity with an incoming-speed gain.

    MJLab contact force points from the primary body toward the secondary
    contact object. Positive velocity along that direction is therefore an
    incoming velocity.
    """
    if contact_force.shape != pre_body_vel.shape or contact_force.shape[-1] != 3:
        raise ValueError("contact_force and pre_body_vel must have matching [B, N, 3] shapes")
    weight = robot_weight.reshape(-1, 1).clamp_min(1.0e-6)
    force_norm = torch.linalg.vector_norm(contact_force, dim=-1)
    force_direction = contact_force / (force_norm.unsqueeze(-1) + 1.0e-6)
    impact_velocity = torch.relu((pre_body_vel * force_direction).sum(dim=-1))
    force_ratio = torch.relu(force_norm / weight - force_threshold_ratio)
    force_severity = F.smooth_l1_loss(
        force_ratio,
        torch.zeros_like(force_ratio),
        beta=smooth_l1_beta,
        reduction="none",
    )
    impact_gain = 1.0 + torch.clamp(
        impact_velocity / impact_velocity_scale,
        min=0.0,
        max=impact_velocity_cap,
    ).square()
    return torch.clamp((force_severity * impact_gain).sum(dim=1), max=severity_cap)


def tangential_contact_speed(
    contact_force: torch.Tensor,
    body_vel: torch.Tensor,
    robot_weight: torch.Tensor,
    *,
    force_threshold_ratio: float = 0.05,
) -> torch.Tensor:
    """Sum loaded bodies' velocity tangent to their contact-force direction."""
    if contact_force.shape != body_vel.shape or contact_force.shape[-1] != 3:
        raise ValueError("contact_force and body_vel must have matching [B, N, 3] shapes")
    weight = robot_weight.reshape(-1, 1).clamp_min(1.0e-6)
    force_norm = torch.linalg.vector_norm(contact_force, dim=-1)
    force_direction = contact_force / (force_norm.unsqueeze(-1) + 1.0e-6)
    normal_velocity = (body_vel * force_direction).sum(dim=-1, keepdim=True)
    tangent_velocity = body_vel - normal_velocity * force_direction
    loaded = force_norm / weight > force_threshold_ratio
    return (torch.linalg.vector_norm(tangent_velocity, dim=-1) * loaded).sum(dim=1)


def fixed_sole_weights(
    local_xy: torch.Tensor,
    edge_weight: float = 0.1,
    center_weight: float = 1.0,
) -> torch.Tensor:
    """Return a terrain-independent 2-D sole-edge-to-center weight map."""

    if local_xy.ndim != 2 or local_xy.shape[-1] != 2 or local_xy.shape[0] == 0:
        raise ValueError(f"local_xy must be a non-empty (points, 2) tensor, got {tuple(local_xy.shape)}")
    if not 0.0 <= edge_weight <= center_weight:
        raise ValueError(
            f"Expected 0 <= edge_weight <= center_weight, got {edge_weight} and {center_weight}"
        )

    def axis_center_profile(values: torch.Tensor) -> torch.Tensor:
        span = values.max() - values.min()
        if float(span) <= 0.0:
            return torch.ones_like(values)
        fraction = ((values - values.min()) / span).clamp(0.0, 1.0)
        profile = 1.0 - 2.0 * torch.abs(fraction - 0.5)
        return profile / profile.max().clamp_min(torch.finfo(values.dtype).eps)

    x_profile = axis_center_profile(local_xy[:, 0])
    y_profile = axis_center_profile(local_xy[:, 1])
    center_profile = torch.minimum(x_profile, y_profile)
    return edge_weight + (center_weight - edge_weight) * center_profile


def feet_stumble_penalty(
    force_history: torch.Tensor,
    horizontal_ratio: float = 1.0,
    force_threshold: float = 20.0,
    force_scale: float = 100.0,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Penalize the largest horizontal-dominant foot contact in a policy step."""

    if force_history.ndim != 4 or force_history.shape[-1] != 3:
        raise ValueError(f"Expected force history shaped (N, contacts, history, 3), got {tuple(force_history.shape)}")
    if force_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("force_scale and max_penalty must be positive")
    if force_history.shape[1] == 0 or force_history.shape[2] == 0:
        return torch.zeros(force_history.shape[0], dtype=force_history.dtype, device=force_history.device)
    force_xy = torch.linalg.norm(force_history[..., :2], dim=-1)
    force_z = torch.abs(force_history[..., 2])
    excess = torch.relu(force_xy - horizontal_ratio * force_z - force_threshold)
    return torch.clamp(excess / force_scale, max=max_penalty).amax(dim=(1, 2))


def _support_deficiency(
    clearance: torch.Tensor,
    point_weights: torch.Tensor | None,
    *,
    height_tolerance: float,
    transition_width: float,
) -> torch.Tensor:
    width = max(float(transition_width), 1.0e-6)
    support = torch.sigmoid((height_tolerance - clearance) / width)
    unsupported = 1.0 - support
    boundary_eps = 8.0 * torch.finfo(clearance.dtype).eps
    unsupported = torch.where(
        clearance > height_tolerance + boundary_eps,
        unsupported,
        torch.zeros_like(unsupported),
    )
    if point_weights is None:
        return unsupported.mean(dim=-1)
    weights = point_weights.to(device=unsupported.device, dtype=unsupported.dtype)
    return (unsupported * weights).sum(dim=-1) / weights.sum().clamp_min(torch.finfo(unsupported.dtype).eps)


def feet_at_plane_penalty(
    clearance: torch.Tensor,
    is_contact: torch.Tensor,
    point_weights: torch.Tensor,
    *,
    height_tolerance: float = 0.030,
    transition_width: float = 0.005,
) -> torch.Tensor:
    """Penalize unsupported sole area for both feet on every terrain type."""

    if clearance.ndim != 3 or is_contact.shape != clearance.shape[:2]:
        raise ValueError(
            f"Expected clearance=(N, 2, points), is_contact=(N, 2), got "
            f"{tuple(clearance.shape)}, {tuple(is_contact.shape)}"
        )
    if point_weights.ndim != 1 or point_weights.numel() != clearance.shape[-1]:
        raise ValueError(f"Expected one weight per terrain point, got {tuple(point_weights.shape)}")
    weighted = _support_deficiency(
        clearance,
        point_weights,
        height_tolerance=height_tolerance,
        transition_width=transition_width,
    )
    return torch.sum(weighted * is_contact.to(weighted.dtype), dim=1)


def _compute_reward(self, *, pre_body_vel: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    aux: dict[str, torch.Tensor] = {}
    contact = self.contact_forces
    foot_contact = contact[:, self.feet_indices, 2] > 1.0
    aux["penalty_torques"] = torch.sum(torch.square(self.torques), dim=1)
    aux["penalty_action_rate"] = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    lower, upper = self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1]
    aux["limits_dof_pos"] = torch.sum((-(self.dof_pos - lower).clip(max=0.0)) + ((self.dof_pos - upper).clip(min=0.0)), dim=1)
    vel_limit = self.dof_vel_limits * float(self.config.rewards.reward_limit.soft_dof_vel_limit)
    aux["limits_dof_vel"] = torch.sum((torch.abs(self.dof_vel) - vel_limit).clip(min=0.0, max=1.0), dim=1)
    torque_limit = self.torque_limits * float(self.config.rewards.reward_limit.soft_torque_limit)
    aux["limits_torque"] = torch.sum((torch.abs(self.torques) - torque_limit).clip(min=0.0), dim=1)
    if len(self.penalised_contact_indices) > 0:
        undesired = torch.any(torch.abs(contact[:, self.penalised_contact_indices, :]) > 1.0, dim=(1, 2))
    else:
        undesired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    aux["penalty_undesired_contact"] = undesired.float()
    if self.terrain_aware_auxiliary:
        if pre_body_vel is None:
            raise ValueError("terrain-aware auxiliary rewards require pre-step body velocity")
        peak_force = peak_contact_force(self.contact_force_history)
        aux["penalty_body_impact"] = body_contact_severity(
            peak_force[:, self.penalised_contact_indices],
            pre_body_vel[:, self.penalised_contact_indices],
            self.robot_weight,
        )
    left_ankle_roll = self.dof_pos[:, self.left_ankle_dof_indices[1:2]]
    right_ankle_roll = self.dof_pos[:, self.right_ankle_dof_indices[1:2]]
    aux["penalty_ankle_roll"] = torch.sum(torch.square(left_ankle_roll) + torch.square(right_ankle_roll), dim=1)
    left_quat = self.body_rot[:, self.feet_indices[0]]
    right_quat = self.body_rot[:, self.feet_indices[1]]
    left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec, w_last=True)
    right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec, w_last=True)
    aux["penalty_feet_ori"] = (
        torch.sum(torch.square(left_gravity[:, :2]), dim=1).sqrt() * foot_contact[:, 0]
        + torch.sum(torch.square(right_gravity[:, :2]), dim=1).sqrt() * foot_contact[:, 1]
    )
    foot_vel = self.body_vel[:, self.feet_indices]
    if self.terrain_aware_auxiliary:
        aux["penalty_slippage"] = tangential_contact_speed(
            contact[:, self.feet_indices],
            foot_vel,
            self.robot_weight,
        )
    else:
        aux["penalty_slippage"] = torch.sum(
            torch.norm(foot_vel, dim=-1) * (torch.norm(contact[:, self.feet_indices, :], dim=-1) > 1.0),
            dim=1,
        )
    forward_left = my_quat_rotate(left_quat, self.forward_vec)
    forward_right = my_quat_rotate(right_quat, self.forward_vec)
    root_forward = my_quat_rotate(self.base_quat, self.forward_vec)
    heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])
    aux["feet_heading_alignment"] = torch.abs(wrap_to_pi(torch.atan2(forward_left[:, 1], forward_left[:, 0]) - heading_root)) + torch.abs(
        wrap_to_pi(torch.atan2(forward_right[:, 1], forward_right[:, 0]) - heading_root)
    )
    foot_collision_history = self._read_foot_collision_force_history()
    aux["feet_stumble"] = feet_stumble_penalty(foot_collision_history)
    if self.virtual_terrain_edges is not None:
        from pbfm.terrains.edge_penalty import foot_points_world, penetration_cost
        points, velocities = foot_points_world(self.body_pos[:,self.feet_indices],
            self.body_rot[:,self.feet_indices], self.body_vel[:,self.feet_indices],
            self.body_ang_vel[:,self.feet_indices], self._edge_foot_points)
        depth = self.virtual_terrain_edges.depth(points)
        aux['penalty_virtual_edge'] = penetration_cost(depth, velocities)
    if self.feet_height_sensor is not None:
        foot_sensor = self.feet_height_sensor
        foot_contact_history = self._read_foot_contact_force_history()
        vertical_foot_contact = torch.abs(foot_contact_history[..., 2]).amax(dim=2) > 1.0
        foot_hits = foot_sensor.data.hit_pos_w.reshape(self.num_envs, 2, self.feet_height_dim, 3)
        foot_distances = foot_sensor.data.distances.reshape(self.num_envs, 2, self.feet_height_dim)
        sole_z = foot_sensor.sample_points_w.reshape(self.num_envs, 2, self.feet_height_dim, 3)[..., 2]
        clearance = clipped_height_clearance(sole_z, foot_hits[..., 2], foot_distances)
        aux["feet_at_plane"] = feet_at_plane_penalty(
            clearance,
            vertical_foot_contact,
            self._feet_at_plane_weights,
        )
    else:
        aux["feet_at_plane"] = torch.zeros(self.num_envs, device=self.device)
    reward = torch.zeros(self.num_envs, device=self.device)
    for name, scale in self.reward_scales.items():
        if name not in aux:
            continue
        rew = aux[name] * scale
        if name in self.reward_penalty_reward_names and self.use_reward_penalty_curriculum:
            rew *= self.reward_penalty_scale
        reward += rew
    return reward, aux
