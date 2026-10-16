"""Terrain reset sampling and ground placement; core owns reset sequencing."""

import numpy as np
import torch

from pbfm.utils.torch_utils import my_quat_rotate, quat_mul, xyzw_to_wxyz

from . import observations

RESET_REGION_NAMES = (
    "flat_center",
    "slope_center",
    "slope_random",
    "stairs_center",
    "stairs_up_center",
    "stairs_down_center",
    "stairs_pre_descent",
    "stairs_pre_ascent",
    "stairs_intercycle",
    "stairs_tread",
    "rough_center",
    "rough_patch",
    "platforms_center",
    "platforms_patch",
    "tile_seam",
)
RESET_REGION_ID = {name: index for index, name in enumerate(RESET_REGION_NAMES)}
RESET_REGION_COMPONENT_NAMES = frozenset(
    {"flat", "slope", "stairs", "stairs_up", "stairs_down", "rough", "platforms"}
)


def reset_region_labels_are_semantic(component_names: tuple[str, ...]) -> bool:
    """Whether the legacy reset-region IDs describe these terrain assets."""

    return all(name in RESET_REGION_COMPONENT_NAMES for name in component_names)


def terrain_family_reset_histogram(terrain_ids: torch.Tensor, *, num_families: int) -> torch.Tensor:
    """Count assigned reset families without using the robot's later tile location."""

    if terrain_ids.ndim != 1:
        raise ValueError("terrain reset family IDs must be one-dimensional")
    counts = torch.bincount(terrain_ids, minlength=num_families)
    if counts.shape[0] != num_families:
        raise ValueError("terrain reset family ID is outside the configured family range")
    return counts


def select_pre_descent_directions(
    root_velocity: torch.Tensor,
    root_rotation: torch.Tensor,
    *,
    speed_threshold: float = 0.10,
    fallback_side_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the cardinal descent side nearest the unmodified motion state."""
    if root_velocity.ndim != 2 or root_velocity.shape[1] < 2:
        raise ValueError("root_velocity must have shape [B, >=2]")
    if root_rotation.shape != (root_velocity.shape[0], 4):
        raise ValueError("root_rotation must have shape [B, 4]")
    if speed_threshold < 0.0:
        raise ValueError("speed_threshold must be non-negative")

    batch_size = root_velocity.shape[0]
    device = root_velocity.device
    dtype = root_velocity.dtype
    velocity_xy = root_velocity[:, :2]
    speed = torch.linalg.vector_norm(velocity_xy, dim=-1)
    local_forward = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    local_forward[:, 0] = 1.0
    heading_xy = my_quat_rotate(root_rotation, local_forward)[:, :2]
    heading_norm = torch.linalg.vector_norm(heading_xy, dim=-1)
    direction_hint = torch.where((speed >= speed_threshold).unsqueeze(-1), velocity_xy, heading_xy)
    reliable = (speed >= speed_threshold) | (heading_norm > 1.0e-6)

    cardinal = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        device=device,
        dtype=dtype,
    )
    side_indices = torch.matmul(direction_hint, cardinal.T).argmax(dim=-1)
    if not torch.all(reliable):
        if fallback_side_indices is None:
            fallback_side_indices = torch.randint(0, 4, (batch_size,), device=device)
        else:
            fallback_side_indices = fallback_side_indices.to(device=device, dtype=torch.long)
            if fallback_side_indices.shape != (batch_size,):
                raise ValueError("fallback_side_indices must have shape [B]")
            if torch.any((fallback_side_indices < 0) | (fallback_side_indices > 3)):
                raise ValueError("fallback_side_indices must be in [0, 3]")
        side_indices = torch.where(reliable, side_indices, fallback_side_indices)
    return cardinal[side_indices]


def pre_descent_reset_positions(
    patch_centers: torch.Tensor,
    directions: torch.Tensor,
    *,
    platform_width: float,
    step_depth: float,
    num_steps: int,
    plateau_width: float,
    edge_margin: float,
) -> torch.Tensor:
    """Place pelvis XY inside the first high plateau, facing its descent edge."""
    if patch_centers.shape != directions.shape or patch_centers.ndim != 2 or patch_centers.shape[1] != 2:
        raise ValueError("patch_centers and directions must have matching [B, 2] shapes")
    if min(platform_width, step_depth, plateau_width) <= 0.0 or num_steps <= 0:
        raise ValueError("stairs dimensions and num_steps must be positive")
    if not 0.0 < edge_margin < plateau_width:
        raise ValueError("edge_margin must lie inside the high plateau")
    descent_edge = platform_width / 2.0 + num_steps * step_depth + plateau_width
    return patch_centers + directions * (descent_edge - edge_margin)


def pre_descent_ground_probe_z(
    env_origin_z: torch.Tensor,
    *,
    num_steps: int,
    max_step_height: float,
    probe_clearance: float,
    max_ray_distance: float,
) -> torch.Tensor:
    """Return a collision-free ray origin above the tallest stair plateau."""
    if num_steps <= 0 or min(max_step_height, probe_clearance, max_ray_distance) <= 0.0:
        raise ValueError("stairs probe dimensions must be positive")
    probe_offset = num_steps * max_step_height + probe_clearance
    if probe_offset >= max_ray_distance:
        raise ValueError("stairs ground probe must remain within the terrain ray range")
    return env_origin_z + probe_offset


def sample_pre_descent_reset_mask(
    terrain_ids: torch.Tensor,
    *,
    stairs_id: int,
    probability: float,
    samples: torch.Tensor,
) -> torch.Tensor:
    """Sample a fixed fraction of stairs resets without affecting other terrains."""
    if terrain_ids.shape != samples.shape:
        raise ValueError("terrain_ids and samples must have matching shapes")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return (terrain_ids == int(stairs_id)) & (samples < probability)


def sample_lie_down_reset_mask(
    samples: torch.Tensor,
    *,
    probability: float,
    excluded: torch.Tensor,
) -> torch.Tensor:
    """Sample lie-down resets while preserving explicit safe-reset exclusions."""
    if samples.shape != excluded.shape:
        raise ValueError("samples and excluded must have matching shapes")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return (samples < probability) & ~excluded


def separated_stairs_upright_reset_mask(region_ids: torch.Tensor) -> torch.Tensor:
    """Identify stair-family center resets that must retain the motion pose."""
    return (region_ids == RESET_REGION_ID["stairs_up_center"]) | (
        region_ids == RESET_REGION_ID["stairs_down_center"]
    )


def sample_terrain_reset_regions(
    terrain_ids: torch.Tensor,
    samples: torch.Tensor,
    *,
    component_names: tuple[str, ...],
    slope_random_prob: float,
    stairs_probabilities: tuple[float, float, float, float, float],
    rough_random_prob: float,
    platforms_random_prob: float,
) -> torch.Tensor:
    """Assign one terrain-aware reset region to every environment."""
    if terrain_ids.shape != samples.shape:
        raise ValueError("terrain_ids and samples must have matching shapes")
    for probability in (slope_random_prob, rough_random_prob, platforms_random_prob):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("reset probabilities must be in [0, 1]")
    if len(stairs_probabilities) != 5 or any(value < 0.0 for value in stairs_probabilities):
        raise ValueError("stairs_probabilities must contain five non-negative values")
    if not np.isclose(sum(stairs_probabilities), 1.0):
        raise ValueError("stairs reset probabilities must sum to 1")

    names = tuple(component_names)
    unknown = sorted(
        set(names) - {"flat", "slope", "stairs", "stairs_up", "stairs_down", "rough", "platforms"}
    )
    if unknown:
        raise ValueError(f"unsupported terrain components for reset sampling: {unknown}")
    region_ids = torch.full_like(terrain_ids, RESET_REGION_ID["flat_center"], dtype=torch.long)

    def terrain_mask(name: str) -> torch.Tensor:
        if name not in names:
            return torch.zeros_like(terrain_ids, dtype=torch.bool)
        return terrain_ids == names.index(name)

    slope = terrain_mask("slope")
    region_ids[slope] = RESET_REGION_ID["slope_center"]
    region_ids[slope & (samples < slope_random_prob)] = RESET_REGION_ID["slope_random"]

    stairs = terrain_mask("stairs")
    region_ids[stairs] = RESET_REGION_ID["stairs_center"]
    center_prob, pre_descent_prob, pre_ascent_prob, intercycle_prob, _tread_prob = stairs_probabilities
    threshold_pre_descent = center_prob + pre_descent_prob
    threshold_pre_ascent = threshold_pre_descent + pre_ascent_prob
    threshold_intercycle = threshold_pre_ascent + intercycle_prob
    region_ids[stairs & (samples >= center_prob) & (samples < threshold_pre_descent)] = RESET_REGION_ID[
        "stairs_pre_descent"
    ]
    region_ids[stairs & (samples >= threshold_pre_descent) & (samples < threshold_pre_ascent)] = RESET_REGION_ID[
        "stairs_pre_ascent"
    ]
    region_ids[stairs & (samples >= threshold_pre_ascent) & (samples < threshold_intercycle)] = RESET_REGION_ID[
        "stairs_intercycle"
    ]
    region_ids[stairs & (samples >= threshold_intercycle)] = RESET_REGION_ID["stairs_tread"]

    region_ids[terrain_mask("stairs_up")] = RESET_REGION_ID["stairs_up_center"]
    region_ids[terrain_mask("stairs_down")] = RESET_REGION_ID["stairs_down_center"]

    rough = terrain_mask("rough")
    region_ids[rough] = RESET_REGION_ID["rough_center"]
    region_ids[rough & (samples < rough_random_prob)] = RESET_REGION_ID["rough_patch"]

    platforms = terrain_mask("platforms")
    region_ids[platforms] = RESET_REGION_ID["platforms_center"]
    region_ids[platforms & (samples < platforms_random_prob)] = RESET_REGION_ID["platforms_patch"]
    return region_ids


def sample_terrain_seam_reset_mask(
    terrain_ids: torch.Tensor,
    samples: torch.Tensor,
    *,
    component_names: tuple[str, ...],
    probability: float,
    excluded_components: tuple[str, ...] = (),
) -> torch.Tensor:
    """Sample seam resets without diluting excluded terrain distributions."""
    if terrain_ids.shape != samples.shape:
        raise ValueError("terrain_ids and samples must have matching shapes")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("seam reset probability must be in [0, 1]")
    names = tuple(component_names)
    mask = samples < probability
    for name in excluded_components:
        if name in names:
            mask &= terrain_ids != names.index(name)
    return mask


def stairs_transition_reset_positions(
    patch_centers: torch.Tensor,
    directions: torch.Tensor,
    region_ids: torch.Tensor,
    *,
    platform_width: float,
    step_depth: float,
    num_steps: int,
    plateau_width: float,
    edge_margin: float,
) -> torch.Tensor:
    """Place stairs resets on flat transition regions, never on a tread."""
    if patch_centers.shape != directions.shape or patch_centers.ndim != 2 or patch_centers.shape[1] != 2:
        raise ValueError("patch_centers and directions must have matching [B, 2] shapes")
    if region_ids.shape != (patch_centers.shape[0],):
        raise ValueError("region_ids must have shape [B]")
    if min(platform_width, step_depth, plateau_width) <= 0.0 or num_steps <= 0:
        raise ValueError("stairs dimensions and num_steps must be positive")
    if not 0.0 < edge_margin < min(platform_width / 2.0, plateau_width):
        raise ValueError("edge_margin must fit inside both center and plateau regions")

    offsets = torch.zeros(len(region_ids), device=patch_centers.device, dtype=patch_centers.dtype)
    pre_descent = region_ids == RESET_REGION_ID["stairs_pre_descent"]
    pre_ascent = region_ids == RESET_REGION_ID["stairs_pre_ascent"]
    intercycle = region_ids == RESET_REGION_ID["stairs_intercycle"]
    offsets[pre_descent] = platform_width / 2.0 + num_steps * step_depth + plateau_width - edge_margin
    offsets[pre_ascent] = platform_width / 2.0 - edge_margin
    offsets[intercycle] = (
        platform_width / 2.0 + 2 * num_steps * step_depth + 2 * plateau_width - edge_margin
    )
    return patch_centers + directions * offsets.unsqueeze(-1)


def terrain_grid_coordinates(
    root_xy: torch.Tensor,
    patch_size: torch.Tensor,
    *,
    num_rows: int,
    num_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map world XY to connected terrain-grid row/column indices."""
    if root_xy.ndim != 2 or root_xy.shape[1] != 2 or patch_size.shape != (2,):
        raise ValueError("root_xy must be [B, 2] and patch_size must be [2]")
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("terrain grid dimensions must be positive")
    lower = -0.5 * patch_size * root_xy.new_tensor([num_rows, num_cols])
    indices = torch.floor((root_xy - lower) / patch_size).to(torch.long)
    rows, cols = indices[:, 0], indices[:, 1]
    inside = (rows >= 0) & (rows < num_rows) & (cols >= 0) & (cols < num_cols)
    return rows.clamp(0, num_rows - 1), cols.clamp(0, num_cols - 1), inside


def terrain_grid_boundary_margin(
    root_xy: torch.Tensor,
    patch_size: torch.Tensor,
    *,
    num_rows: int,
    num_cols: int,
    border_width: float = 0.0,
) -> torch.Tensor:
    """Distance to the outer boundary of the complete connected tile grid."""
    if border_width < 0.0:
        raise ValueError("border_width must be non-negative")
    half_extent = 0.5 * patch_size * root_xy.new_tensor([num_rows, num_cols]) + border_width
    return torch.min(half_extent - root_xy.abs(), dim=-1).values


def _small_random_quaternions(n: int, max_angle: float, device: str) -> torch.Tensor:
    axis = torch.randn((n, 3), device=device)
    axis = axis / torch.clamp(torch.norm(axis, dim=1, keepdim=True), min=1.0e-6)
    angles = max_angle * torch.rand((n, 1), device=device)
    sin_half_angle = torch.sin(angles / 2)
    cos_half_angle = torch.cos(angles / 2)
    return torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)


def _random_yaw_quaternions(n: int, device: str) -> torch.Tensor:
    """Sample independent world-yaw rotations uniformly from [-pi, pi]."""
    angles = (torch.rand(n, device=device) * 2.0 - 1.0) * torch.pi
    half = 0.5 * angles
    return torch.stack(
        (torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)),
        dim=1,
    )


def rotate_root_motion_by_yaw(
    root_rotation: torch.Tensor,
    root_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    yaw_rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate a root pose and its world-frame velocities by the same yaw."""
    batch_size = root_rotation.shape[0]
    if root_rotation.shape != (batch_size, 4) or yaw_rotation.shape != (batch_size, 4):
        raise ValueError("root_rotation and yaw_rotation must have matching [B, 4] shapes")
    if root_velocity.shape != (batch_size, 3) or root_angular_velocity.shape != (batch_size, 3):
        raise ValueError("root velocities must have matching [B, 3] shapes")
    return (
        quat_mul(yaw_rotation, root_rotation, w_last=True),
        my_quat_rotate(yaw_rotation, root_velocity),
        my_quat_rotate(yaw_rotation, root_angular_velocity),
    )


def _terrain_patch_centers(self) -> torch.Tensor:
    """Return geometric patch centers, which may differ from rough spawn origins."""
    if self._terrain_patch_size is None:
        raise RuntimeError("patch centers are undefined for infinite plane terrain")
    terrain = self.mjlab_env.scene["terrain"]
    levels = terrain.terrain_levels.to(dtype=torch.float32)
    types = terrain.terrain_types.to(dtype=torch.float32)
    terrain_origins = terrain.terrain_origins
    num_rows, num_cols = terrain_origins.shape[:2]
    centers = torch.empty((self.num_envs, 2), device=self.device, dtype=torch.float32)
    centers[:, 0] = (levels + 0.5 - num_rows / 2.0) * self._terrain_patch_size[0]
    centers[:, 1] = (types + 0.5 - num_cols / 2.0) * self._terrain_patch_size[1]
    return centers


def _sample_terrain_reset_positions(
    self,
    env_ids: torch.Tensor,
    root_rotation: torch.Tensor,
    root_velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample training-only terrain regions and their world XY positions."""
    if self._terrain_patch_size is None:
        region_ids = torch.full(
            (len(env_ids),), RESET_REGION_ID["flat_center"], device=self.device, dtype=torch.long
        )
        return region_ids, self.env_origins[env_ids, :2].clone(), torch.zeros_like(region_ids, dtype=torch.bool)
    terrain_entity = self.mjlab_env.scene["terrain"]
    terrain_ids = terrain_entity.terrain_types[env_ids]
    positions = self.env_origins[env_ids, :2].clone()
    if not self._reset_region_labels_are_semantic:
        region_ids = torch.full(
            (len(env_ids),), RESET_REGION_ID["flat_center"], device=self.device, dtype=torch.long
        )
        return region_ids, positions, torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
    region_ids = torch.empty_like(terrain_ids, dtype=torch.long)
    for terrain_id, terrain_name in enumerate(self.terrain_component_names):
        region_ids[terrain_ids == terrain_id] = RESET_REGION_ID[f"{terrain_name}_center"]
    if self.is_evaluating:
        return region_ids, positions, torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)

    reset_cfg = self.config.terrain.reset
    stairs_prob_cfg = reset_cfg.stairs_probabilities
    region_ids = sample_terrain_reset_regions(
        terrain_ids,
        torch.rand(len(env_ids), device=self.device),
        component_names=self.terrain_component_names,
        slope_random_prob=float(reset_cfg.slope_random_prob),
        stairs_probabilities=(
            float(stairs_prob_cfg.center),
            float(stairs_prob_cfg.pre_descent),
            float(stairs_prob_cfg.pre_ascent),
            float(stairs_prob_cfg.intercycle),
            float(stairs_prob_cfg.tread),
        ),
        rough_random_prob=float(reset_cfg.rough_random_prob),
        platforms_random_prob=float(reset_cfg.platforms_random_prob),
    )

    patch_centers = _terrain_patch_centers(self)[env_ids]
    slope_random = region_ids == RESET_REGION_ID["slope_random"]
    if torch.any(slope_random):
        radius_min, radius_max = (float(value) for value in reset_cfg.slope_radius_range)
        if not 0.0 <= radius_min < radius_max:
            raise ValueError("terrain.reset.slope_radius_range must be increasing and non-negative")
        count = int(slope_random.sum().item())
        angles = torch.rand(count, device=self.device) * (2.0 * torch.pi)
        radii = radius_min + torch.rand(count, device=self.device) * (radius_max - radius_min)
        direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        positions[slope_random] = patch_centers[slope_random] + direction * radii.unsqueeze(-1)

    stairs_mask = torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
    for name in ("stairs_pre_descent", "stairs_pre_ascent", "stairs_intercycle"):
        stairs_mask |= region_ids == RESET_REGION_ID[name]
    if torch.any(stairs_mask):
        stairs_cfg = self.config.terrain.stairs
        directions = select_pre_descent_directions(
            root_velocity[stairs_mask],
            root_rotation[stairs_mask],
            speed_threshold=float(stairs_cfg.get("pre_descent_speed_threshold", 0.10)),
        )
        positions[stairs_mask] = stairs_transition_reset_positions(
            patch_centers[stairs_mask],
            directions,
            region_ids[stairs_mask],
            platform_width=float(stairs_cfg.platform_width),
            step_depth=float(stairs_cfg.step_depth),
            num_steps=int(stairs_cfg.num_steps),
            plateau_width=float(stairs_cfg.plateau_width),
            edge_margin=float(reset_cfg.stairs_edge_margin),
        )

    stairs_tread = region_ids == RESET_REGION_ID["stairs_tread"]
    if torch.any(stairs_tread):
        stairs_cfg = self.config.terrain.stairs
        count = int(stairs_tread.sum().item())
        directions = select_pre_descent_directions(
            root_velocity[stairs_tread],
            root_rotation[stairs_tread],
            speed_threshold=float(stairs_cfg.get("pre_descent_speed_threshold", 0.10)),
        )
        step_ids = torch.randint(int(stairs_cfg.num_steps), (count,), device=self.device)
        ascending = torch.rand(count, device=self.device) < 0.5
        ascent_radius = float(stairs_cfg.platform_width) / 2.0 + (
            step_ids.to(torch.float32) + 0.5
        ) * float(stairs_cfg.step_depth)
        descent_radius = (
            float(stairs_cfg.platform_width) / 2.0
            + int(stairs_cfg.num_steps) * float(stairs_cfg.step_depth)
            + float(stairs_cfg.plateau_width)
            + (step_ids.to(torch.float32) + 0.5) * float(stairs_cfg.step_depth)
        )
        radii = torch.where(ascending, ascent_radius, descent_radius)
        positions[stairs_tread] = patch_centers[stairs_tread] + directions * radii.unsqueeze(-1)

    rough_patch = region_ids == RESET_REGION_ID["rough_patch"]
    if torch.any(rough_patch):
        patch_options = terrain_entity.flat_patches.get("spawn")
        if patch_options is None:
            raise RuntimeError("rough reset sampling requires terrain flat_patches['spawn']")
        selected_env_ids = env_ids[rough_patch]
        options = patch_options[
            terrain_entity.terrain_levels[selected_env_ids],
            terrain_entity.terrain_types[selected_env_ids],
        ]
        option_ids = torch.randint(options.shape[1], (len(selected_env_ids),), device=self.device)
        positions[rough_patch] = options[torch.arange(len(selected_env_ids), device=self.device), option_ids, :2]

    platforms_patch = region_ids == RESET_REGION_ID["platforms_patch"]
    if torch.any(platforms_patch):
        patch_options = terrain_entity.flat_patches.get("platform_spawn")
        if patch_options is None:
            raise RuntimeError("platform reset sampling requires terrain flat_patches['platform_spawn']")
        selected_env_ids = env_ids[platforms_patch]
        options = patch_options[
            terrain_entity.terrain_levels[selected_env_ids],
            terrain_entity.terrain_types[selected_env_ids],
        ]
        option_ids = torch.randint(options.shape[1], (len(selected_env_ids),), device=self.device)
        positions[platforms_patch] = options[
            torch.arange(len(selected_env_ids), device=self.device), option_ids, :2
        ]

    seam_mask = sample_terrain_seam_reset_mask(
        terrain_ids,
        torch.rand(len(env_ids), device=self.device),
        component_names=self.terrain_component_names,
        probability=float(reset_cfg.seam_reset_prob),
        excluded_components=tuple(str(name) for name in reset_cfg.get("seam_excluded_terrains", ())),
    )
    if torch.any(seam_mask):
        cols = terrain_entity.terrain_types[env_ids[seam_mask]]
        velocity_y = root_velocity[seam_mask, 1]
        signs = torch.where(velocity_y >= 0.0, 1.0, -1.0)
        signs = torch.where(cols == 0, 1.0, signs)
        signs = torch.where(cols == self._terrain_grid_cols - 1, -1.0, signs)
        seam_positions = patch_centers[seam_mask].clone()
        seam_positions[:, 0] += (
            torch.rand(len(seam_positions), device=self.device) - 0.5
        ) * min(4.0, float(self._terrain_patch_size[0]) * 0.4)
        seam_positions[:, 1] += signs * (
            float(self._terrain_patch_size[1]) / 2.0 - float(reset_cfg.seam_inset)
        )
        positions[seam_mask] = seam_positions
        region_ids[seam_mask] = RESET_REGION_ID["tile_seam"]

    elevated_mask = (region_ids == RESET_REGION_ID["stairs_pre_descent"]) | (
        region_ids == RESET_REGION_ID["stairs_tread"]
    )
    return region_ids, positions, elevated_mask


def _place_reset_on_local_ground(
    self,
    env_ids: torch.Tensor,
    root_xyzw: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    desired_clearance: torch.Tensor,
    probe_world_z: torch.Tensor | None = None,
    adaptive_lift_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Query ground at final XY and return roots placed at desired clearance."""
    root_xyzw = root_xyzw.clone()
    desired_clearance = desired_clearance.reshape(-1)
    if self.terrain_observation_mode == "flat_zero":
        ground_z = torch.zeros_like(desired_clearance)
        root_xyzw[:, 2] = ground_z + desired_clearance
        return root_xyzw, ground_z

    provisional = root_xyzw.clone()
    provisional[:, 2] = self.env_origins[env_ids, 2] + desired_clearance
    if probe_world_z is not None:
        probe_world_z = probe_world_z.reshape(-1).to(device=provisional.device, dtype=provisional.dtype)
        if probe_world_z.shape != desired_clearance.shape:
            raise ValueError("probe_world_z must match desired_clearance")
        use_probe = torch.isfinite(probe_world_z)
        provisional[:, 2] = torch.where(use_probe, probe_world_z, provisional[:, 2])
    provisional_wxyz = torch.cat(
        [provisional[:, :3], xyzw_to_wxyz(provisional[:, 3:7]), provisional[:, 7:13]], dim=-1
    )
    self.robot.write_root_state_to_sim(provisional_wxyz, env_ids=env_ids)
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self._joint_ids, env_ids=env_ids)
    self.mjlab_env.scene.write_data_to_sim()
    self.mjlab_env.sim.forward()
    self.mjlab_env.sim.sense()

    clearances = observations._terrain_sensor_clearances(self)[env_ids]
    center_clearance = clearances[:, self._terrain_reference_index]
    max_distance = float(self.config.terrain.terrain_priv.max_ray_distance)
    valid = torch.isfinite(center_clearance) & (center_clearance >= 0.0) & (center_clearance < max_distance * 0.999)
    if not torch.all(valid).item():
        bad_ids = env_ids[~valid].detach().cpu().tolist()
        bad_values = center_clearance[~valid].detach().cpu().tolist()
        raise RuntimeError(
            "terrain reset ground query failed at final XY: "
            f"env_ids={bad_ids[:16]}, center_clearances={bad_values[:16]}"
        )
    ground_z = provisional[:, 2] - center_clearance
    lift = torch.zeros_like(desired_clearance)
    if adaptive_lift_mask is not None:
        adaptive_lift_mask = adaptive_lift_mask.reshape(-1).to(device=self.device, dtype=torch.bool)
        if adaptive_lift_mask.shape != desired_clearance.shape:
            raise ValueError("adaptive_lift_mask must match desired_clearance")
        if torch.any(adaptive_lift_mask):
            if self._terrain_reset_footprint_indices is None:
                sensor = self.mjlab_env.scene.sensors["terrain_height"]
                offsets = sensor._local_offsets
                if offsets is None:
                    raise RuntimeError("terrain_height sensor offsets are unavailable during reset")
                if offsets.ndim == 3:
                    offsets = offsets[0]
                radius = float(self.config.terrain.reset.adaptive_lift_footprint_radius)
                footprint = torch.linalg.vector_norm(offsets[:, :2], dim=-1) <= radius
                self._terrain_reset_footprint_indices = torch.nonzero(footprint, as_tuple=False).flatten()
                if len(self._terrain_reset_footprint_indices) == 0:
                    raise RuntimeError("adaptive reset footprint contains no terrain rays")
            footprint_clearances = clearances[:, self._terrain_reset_footprint_indices]
            footprint_heights = provisional[:, 2:3] - footprint_clearances
            footprint_heights = torch.where(
                torch.isfinite(footprint_heights),
                footprint_heights,
                torch.full_like(footprint_heights, -torch.inf),
            )
            highest_ground = footprint_heights.amax(dim=-1)
            extra = float(self.config.terrain.reset.adaptive_lift_extra_clearance)
            height_delta = torch.relu(highest_ground - ground_z)
            adaptive_lift = torch.where(
                height_delta > 1.0e-4,
                height_delta + extra,
                torch.zeros_like(height_delta),
            )
            lift = torch.where(adaptive_lift_mask, adaptive_lift, lift)
    root_xyzw[:, 2] = ground_z + desired_clearance + lift
    return root_xyzw, ground_z
