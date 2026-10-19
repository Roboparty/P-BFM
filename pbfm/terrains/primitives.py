"""Terrain primitives that are not provided by MJLab."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    FlatPatchSamplingCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPerlinNoiseTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    SubTerrainCfg,
)
from mjlab.terrains.heightfield_terrains import _compute_flat_patches, color_by_height
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput


def _add_box(
    body: mujoco.MjsBody,
    *,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    bottom: float,
    top: float,
    color: tuple[float, float, float, float],
) -> TerrainGeometry:
    height = max(top - bottom, 1e-6)
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(max(size_xy[0] / 2, 1e-6), max(size_xy[1] / 2, 1e-6), height / 2),
        pos=(center_xy[0], center_xy[1], bottom + height / 2),
    )
    return TerrainGeometry(geom=geom, color=color)


def _remove_height_material(output: TerrainOutput) -> TerrainOutput:
    for geometry in output.geometries:
        if geometry.geom is not None:
            geometry.geom.material = ""
    return output


def enforce_flat_hfield_border(
    output: TerrainOutput, spec: mujoco.MjSpec, width: float, *, height_offset: float = 0.0
) -> TerrainOutput:
    """Put the border at local z=0, with grid vertices exactly at its inner edge.

    MJLab truncates border_width / resolution; with 0.2 m cells this cannot
    represent a 0.5 m border. Resample to a nearby aligned grid, then flatten
    the inclusive border vertices so interpolation cannot leak into the band.
    Optional lifting applies only to the interior (used by perimeter tiles).
    """
    if width <= 0:
        return output
    for geometry in output.geometries:
        geom = geometry.geom
        if geom is None or geom.type != mujoco.mjtGeom.mjGEOM_HFIELD:
            continue
        field = spec.hfield(geom.hfieldname)
        if field is None:
            # In an attached Scene spec, MuJoCo's name lookup can lag behind
            # add_hfield(). The asset already exists in the live asset list.
            matches = [asset for asset in spec.hfields if asset.name == geom.hfieldname]
            if len(matches) != 1:
                raise ValueError(f"Cannot resolve heightfield {geom.hfieldname!r}")
            field = matches[0]
        sx, sy, sz, base = field.size
        rows, cols = int(field.nrow), int(field.ncol)
        heights = np.asarray(field.userdata).reshape(rows, cols) * sz + geom.pos[2] + height_offset
        new_axes = []
        for extent, count in ((2 * sx, cols), (2 * sy, rows)):
            bands = round(extent / width)
            if bands <= 2 or not math.isclose(bands * width, extent):
                raise ValueError("flat border must divide the heightfield extent")
            intervals = math.ceil((count - 1) / bands) * bands
            new_axes.append(np.linspace(0.0, extent, intervals + 1))
        x, y = new_axes
        along_x = np.stack([np.interp(x, np.linspace(0, 2 * sx, cols), row) for row in heights])
        heights = np.stack([
            np.interp(y, np.linspace(0, 2 * sy, rows), column) for column in along_x.T
        ], axis=1)
        border = ((x <= width + 1e-9) | (x >= 2 * sx - width - 1e-9))[None, :] | (
            (y <= width + 1e-9) | (y >= 2 * sy - width - 1e-9)
        )[:, None]
        heights[border] = 0.0
        lower, upper = float(heights.min()), float(heights.max())
        scale = max(upper - lower, 1e-6)
        field.nrow, field.ncol = len(y), len(x)
        field.size = (sx, sy, scale, max(base, 1e-3))
        field.userdata = ((heights - lower) / scale).astype(np.float32).ravel().tolist()
        pos = np.asarray(geom.pos).copy()
        pos[2] = lower
        geom.pos = pos
    output.origin[2] += height_offset
    return output


def _mark_as_terrain(output: TerrainOutput, *, group: int = 5) -> TerrainOutput:
    """Put collision terrain in the requested ray-query geom group."""
    for geometry in output.geometries:
        if geometry.geom is not None:
            geometry.geom.group = group
    return output


def _add_square_ring(
    body: mujoco.MjsBody,
    *,
    center_xy: tuple[float, float],
    inner_width: float,
    outer_width: float,
    bottom: float,
    top: float,
    color: tuple[float, float, float, float],
) -> list[TerrainGeometry]:
    """Build a square annulus from four non-overlapping boxes."""
    band_width = (outer_width - inner_width) / 2
    if band_width <= 1e-6:
        return []
    center_x, center_y = center_xy
    offset = inner_width / 2 + band_width / 2
    geometries = []
    for y in (center_y - offset, center_y + offset):
        geometries.append(
            _add_box(
                body,
                center_xy=(center_x, y),
                size_xy=(outer_width, band_width),
                bottom=bottom,
                top=top,
                color=color,
            )
        )
    for x in (center_x - offset, center_x + offset):
        geometries.append(
            _add_box(
                body,
                center_xy=(x, center_y),
                size_xy=(band_width, inner_width),
                bottom=bottom,
                top=top,
                color=color,
            )
        )
    return geometries


@dataclass(kw_only=True)
class TerrainBoxFlatCfg(BoxFlatTerrainCfg):
    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        return _mark_as_terrain(super().function(difficulty, spec, rng))


@dataclass(kw_only=True)
class TerrainInvertedPyramidStairsCfg(BoxInvertedPyramidStairsTerrainCfg):
    """Stairs with a safe center platform and ascent in every heading."""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        return _mark_as_terrain(super().function(difficulty, spec, rng))


@dataclass(kw_only=True)
class TerrainPyramidStairsCfg(BoxPyramidStairsTerrainCfg):
    """Ascending stairs marked for the dedicated terrain ray group."""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        return _mark_as_terrain(super().function(difficulty, spec, rng))


@dataclass(kw_only=True)
class TerrainRandomGridCfg(BoxRandomGridTerrainCfg):
    """Low platforms marked for the dedicated terrain ray group."""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        return _mark_as_terrain(super().function(difficulty, spec, rng))


@dataclass(kw_only=True)
class TerrainBoundedStairsCfg(SubTerrainCfg):
    """Repeat bounded up-platform-down stair cycles across the patch."""

    step_height_range: tuple[float, float] = (0.10, 0.18)
    step_width: float = 0.30
    platform_width: float = 1.5
    num_steps: int = 6
    plateau_width: float = 0.8
    border_width: float = 0.5
    base_thickness: float = 0.25

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del rng
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if min(self.step_width, self.platform_width, self.plateau_width) <= 0.0:
            raise ValueError("stair widths must be positive")
        available_width = min(self.size) - 2 * self.border_width
        cycle_radius = 2 * self.num_steps * self.step_width + 2 * self.plateau_width
        available_radius = (available_width - self.platform_width) / 2
        num_cycles = int(available_radius // cycle_radius)
        if num_cycles <= 0:
            raise ValueError("terrain patch is too small for one complete up-platform-down cycle")

        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )
        center_x = self.size[0] / 2
        center_y = self.size[1] / 2
        body = spec.body("terrain")
        geometries = [
            _add_box(
                body,
                center_xy=(center_x, center_y),
                size_xy=self.size,
                bottom=-self.base_thickness,
                top=0.0,
                color=(0.32, 0.32, 0.32, 1.0),
            )
        ]

        center_xy = (center_x, center_y)
        peak_height = self.num_steps * step_height
        current_width = self.platform_width
        for cycle in range(num_cycles):
            for step in range(1, self.num_steps + 1):
                outer_width = current_width + 2 * self.step_width
                top = step * step_height
                color = (0.30 + 0.04 * step, 0.34 + 0.03 * step, 0.38 + 0.02 * step, 1.0)
                geometries.extend(
                    _add_square_ring(
                        body,
                        center_xy=center_xy,
                        inner_width=current_width,
                        outer_width=outer_width,
                        bottom=0.0,
                        top=top,
                        color=color,
                    )
                )
                current_width = outer_width

            plateau_outer_width = current_width + 2 * self.plateau_width
            geometries.extend(
                _add_square_ring(
                    body,
                    center_xy=center_xy,
                    inner_width=current_width,
                    outer_width=plateau_outer_width,
                    bottom=0.0,
                    top=peak_height,
                    color=(0.52, 0.52, 0.50, 1.0),
                )
            )
            current_width = plateau_outer_width

            for step in range(1, self.num_steps + 1):
                outer_width = current_width + 2 * self.step_width
                top = (self.num_steps - step) * step_height
                if top > 0.0:
                    geometries.extend(
                        _add_square_ring(
                            body,
                            center_xy=center_xy,
                            inner_width=current_width,
                            outer_width=outer_width,
                            bottom=0.0,
                            top=top,
                            color=(0.42, 0.43, 0.43, 1.0),
                        )
                    )
                current_width = outer_width

            # The base box supplies the low platform before the next cycle.
            current_width += 2 * self.plateau_width

        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_x, center_y, 0.0]),
                geometries=geometries,
            )
        )


@dataclass(kw_only=True)
class TerrainSeparatedPyramidStairsCfg(SubTerrainCfg):
    """A fixed-step square staircase with a center-only transition start."""

    direction: str
    step_height_range: tuple[float, float] = (0.10, 0.18)
    step_width: float = 0.30
    platform_width: float = 0.80
    num_steps: int = 10
    border_width: float = 0.50
    base_thickness: float = 0.25

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del rng
        if self.direction not in {"up", "down"}:
            raise ValueError("stair direction must be 'up' or 'down'")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if min(self.step_width, self.platform_width, self.base_thickness) <= 0.0:
            raise ValueError("stair dimensions must be positive")

        stair_outer_width = self.platform_width + 2 * self.num_steps * self.step_width
        available_width = min(self.size) - 2 * self.border_width
        if stair_outer_width >= available_width:
            raise ValueError("separated stairs need a flat outer seam inside the terrain patch")

        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )
        peak_height = self.num_steps * step_height
        center_xy = (self.size[0] / 2.0, self.size[1] / 2.0)
        body = spec.body("terrain")
        base_top = -peak_height if self.direction == "up" else 0.0
        base_bottom = base_top - self.base_thickness
        base_size = (
            (self.platform_width, self.platform_width)
            if self.direction == "up"
            else self.size
        )
        geometries = [
            _add_box(
                body,
                center_xy=center_xy,
                size_xy=base_size,
                bottom=base_bottom,
                top=base_top,
                color=(0.30, 0.32, 0.34, 1.0),
            )
        ]

        if self.direction == "down":
            geometries.append(
                _add_box(
                    body,
                    center_xy=center_xy,
                    size_xy=(self.platform_width, self.platform_width),
                    bottom=0.0,
                    top=peak_height,
                    color=(0.54, 0.54, 0.50, 1.0),
                )
            )

        current_width = self.platform_width
        for step in range(1, self.num_steps + 1):
            outer_width = current_width + 2 * self.step_width
            if self.direction == "up":
                top = -(self.num_steps - step) * step_height
                bottom = base_top
            else:
                top = (self.num_steps - step) * step_height
                bottom = 0.0
            if top > bottom + 1.0e-6:
                geometries.extend(
                    _add_square_ring(
                        body,
                        center_xy=center_xy,
                        inner_width=current_width,
                        outer_width=outer_width,
                        bottom=bottom,
                        top=top,
                        color=(0.34 + 0.025 * step, 0.37 + 0.02 * step, 0.40, 1.0),
                    )
                )
            current_width = outer_width

        if self.direction == "up":
            # Fill the remainder at z=0 so every tile boundary shares the
            # connected map's common collision and ray-cast height.
            geometries.extend(
                _add_square_ring(
                    body,
                    center_xy=center_xy,
                    inner_width=current_width,
                    # A tiny overlap removes exact-edge ray gaps between
                    # independently compiled neighboring tiles.
                    outer_width=min(self.size) + 2.0e-3,
                    bottom=base_top,
                    top=0.0,
                    color=(0.42, 0.43, 0.43, 1.0),
                )
            )

        center_height = -peak_height if self.direction == "up" else peak_height
        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_xy[0], center_xy[1], center_height]),
                geometries=geometries,
            )
        )


@dataclass(kw_only=True)
class TerrainTraversalCourseCfg(SubTerrainCfg):
    """A square-symmetric center-out route: stairs up, platform, stairs down."""

    flat_run: float = 1.80
    step_height: float = 0.12
    step_depth: float = 0.30
    num_steps: int = 5
    top_platform_length: float = 0.80
    connector_length: float = 1.00
    include_ramp: bool = True
    ramp_length: float = 2.50
    ramp_angle_deg: float = 8.0
    border_width: float = 0.50
    base_thickness: float = 0.25
    horizontal_scale: float = 0.10

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, rng
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if min(self.step_height, self.step_depth, self.ramp_length, self.horizontal_scale) <= 0.0:
            raise ValueError("course step and terrain dimensions must be positive")
        if self.include_ramp:
            raise ValueError("box-based traversal course does not support include_ramp=True")

        stairs_start = self.flat_run
        stairs_up_end = stairs_start + self.num_steps * self.step_depth
        stairs_down_start = stairs_up_end + self.top_platform_length
        stairs_down_end = stairs_down_start + self.num_steps * self.step_depth
        ramp_start = stairs_down_end + self.connector_length
        ramp_end = ramp_start + self.ramp_length
        safe_forward_extent = min(self.size) / 2.0 - self.border_width
        course_end = ramp_end if self.include_ramp else stairs_down_end
        if course_end >= safe_forward_extent:
            raise ValueError("course does not fit inside the terrain patch")

        center_x = self.size[0] / 2.0
        center_y = self.size[1] / 2.0
        terrain_body = spec.body("terrain")
        color = (0.38, 0.44, 0.40, 1.0)
        geometries = [
            _add_box(
                terrain_body,
                center_xy=(center_x, center_y),
                size_xy=self.size,
                bottom=-self.base_thickness,
                top=0.0,
                color=color,
            )
        ]

        inner_width = 2.0 * self.flat_run
        for level in range(1, self.num_steps + 1):
            outer_width = inner_width + 2.0 * self.step_depth
            geometries.extend(
                _add_square_ring(
                    terrain_body,
                    center_xy=(center_x, center_y),
                    inner_width=inner_width,
                    outer_width=outer_width,
                    bottom=0.0,
                    top=level * self.step_height,
                    color=color,
                )
            )
            inner_width = outer_width

        stair_peak = self.num_steps * self.step_height
        outer_width = inner_width + 2.0 * self.top_platform_length
        geometries.extend(
            _add_square_ring(
                terrain_body,
                center_xy=(center_x, center_y),
                inner_width=inner_width,
                outer_width=outer_width,
                bottom=0.0,
                top=stair_peak,
                color=color,
            )
        )
        inner_width = outer_width

        for level in range(1, self.num_steps + 1):
            outer_width = inner_width + 2.0 * self.step_depth
            height = (self.num_steps - level) * self.step_height
            if height > 0.0:
                geometries.extend(
                    _add_square_ring(
                        terrain_body,
                        center_xy=(center_x, center_y),
                        inner_width=inner_width,
                        outer_width=outer_width,
                        bottom=0.0,
                        top=height,
                        color=color,
                    )
                )
            inner_width = outer_width

        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_x, center_y, 0.0]),
                geometries=geometries,
            )
        )


@dataclass(kw_only=True)
class NeutralHfPerlinNoiseTerrainCfg(HfPerlinNoiseTerrainCfg):
    """Perlin terrain without its height-colored texture material."""

    geom_group: int = 5
    exact_flat_border: bool = False
    interior_height_offset: float = 0.0

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        output = _mark_as_terrain(
            _remove_height_material(super().function(difficulty, spec, rng)),
            group=self.geom_group,
        )
        if output.flat_patches is not None and len(output.flat_patches.get("spawn", ())) > 0:
            output.origin = output.flat_patches["spawn"][0].copy()
        if self.exact_flat_border:
            output = enforce_flat_hfield_border(output, spec, self.border_width, height_offset=self.interior_height_offset)
        return output


@dataclass(kw_only=True)
class NeutralHfPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    """Pyramid slope terrain without its height-colored texture material."""

    geom_group: int = 5
    exact_flat_border: bool = False

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        output = _mark_as_terrain(
            _remove_height_material(super().function(difficulty, spec, rng)),
            group=self.geom_group,
        )
        return enforce_flat_hfield_border(output, spec, self.border_width) if self.exact_flat_border else output


@dataclass(kw_only=True)
class NeutralHfDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg):
    """Discrete obstacles without height material, visible to terrain rays."""

    geom_group: int = 5
    exact_flat_border: bool = False
    interior_height_offset: float = 0.0
    obstacle_width_sampling_stride_cells: int = 4

    def obstacle_width_candidates_cells(self) -> np.ndarray:
        """Return inclusive width bins without silently undershooting the configured minimum.

        MJLab's upstream primitive floors the minimum width and advances by four
        height-field cells.  At the training preset's 0.2 m resolution, a configured 0.3--1.0 m
        interval therefore degenerates to only 0.2 and 1.0 m.  This explicit
        contract uses ceil/floor at the boundaries and a configurable stride.
        """
        stride = int(self.obstacle_width_sampling_stride_cells)
        if stride <= 0:
            raise ValueError("obstacle_width_sampling_stride_cells must be positive")
        lower = int(math.ceil(self.obstacle_width_range[0] / self.horizontal_scale - 1e-9))
        upper = int(math.floor(self.obstacle_width_range[1] / self.horizontal_scale + 1e-9))
        if lower <= 0 or upper < lower:
            raise ValueError(
                f"No representable obstacle width in {self.obstacle_width_range} "
                f"at horizontal_scale={self.horizontal_scale}"
            )
        return np.arange(lower, upper + 1, stride, dtype=np.int64)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        body = spec.body("terrain")
        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"Border width ({self.border_width}) must be >= horizontal scale "
                f"({self.horizontal_scale})"
            )

        obs_height = self.obstacle_height_range[0] + difficulty * (
            self.obstacle_height_range[1] - self.obstacle_height_range[0]
        )
        border_pixels = int(self.border_width / self.horizontal_scale)
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)
        obs_h = int(obs_height / self.vertical_scale)
        platform_pixels = int(self.platform_width / self.horizontal_scale)

        inner_width_pixels = width_pixels - 2 * border_pixels if border_pixels > 0 else width_pixels
        inner_length_pixels = length_pixels - 2 * border_pixels if border_pixels > 0 else length_pixels
        noise = np.zeros((inner_width_pixels, inner_length_pixels), dtype=np.int16)
        width_candidates = self.obstacle_width_candidates_cells()

        # Keep upstream placement alignment while sampling every representable
        # width requested by this project-specific primitive.
        x_range = np.arange(0, inner_width_pixels, 4)
        y_range = np.arange(0, inner_length_pixels, 4)
        for _ in range(self.num_obstacles):
            if len(x_range) == 0 or len(y_range) == 0:
                break
            if self.obstacle_height_mode == "choice":
                height = rng.choice(np.array([-obs_h, -obs_h // 2, obs_h // 2, obs_h]))
            else:
                height = obs_h
            width = int(rng.choice(width_candidates))
            length = width if self.square_obstacles else int(rng.choice(width_candidates))
            x = int(rng.choice(x_range))
            y = int(rng.choice(y_range))
            noise[x : min(x + width, inner_width_pixels), y : min(y + length, inner_length_pixels)] = height

        # Preserve the obstacle-free center reset platform.
        center_x = inner_width_pixels // 2
        center_y = inner_length_pixels // 2
        half_platform = platform_pixels // 2
        noise[
            max(center_x - half_platform, 0) : min(center_x + half_platform, inner_width_pixels),
            max(center_y - half_platform, 0) : min(center_y + half_platform, inner_length_pixels),
        ] = 0

        if border_pixels > 0:
            outer = np.zeros((width_pixels, length_pixels), dtype=np.int16)
            outer[
                border_pixels : border_pixels + inner_width_pixels,
                border_pixels : border_pixels + inner_length_pixels,
            ] = noise
            noise = outer

        elevation_min = int(np.min(noise))
        elevation_max = int(np.max(noise))
        elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1
        max_physical_height = elevation_range * self.vertical_scale
        base_thickness = max_physical_height * self.base_thickness_ratio
        normalized = (noise - elevation_min) / elevation_range

        unique_id = uuid.uuid4().hex
        field = spec.add_hfield(
            name=f"hfield_{unique_id}",
            size=[self.size[0] / 2, self.size[1] / 2, max_physical_height, base_thickness],
            nrow=noise.shape[0],
            ncol=noise.shape[1],
            userdata=normalized.flatten().astype(np.float32).tolist(),
        )
        z_offset = elevation_min * self.vertical_scale if self.obstacle_height_mode == "choice" else 0.0
        material_name = color_by_height(spec, noise, unique_id, normalized)
        hfield_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=[self.size[0] / 2, self.size[1] / 2, z_offset],
            material=material_name,
        )
        flat_patches = _compute_flat_patches(
            noise,
            self.vertical_scale,
            self.horizontal_scale,
            z_offset,
            self.flat_patch_sampling,
            rng,
        )
        output = TerrainOutput(
            origin=np.array([self.size[0] / 2, self.size[1] / 2, self.origin_z_offset]),
            geometries=[TerrainGeometry(geom=hfield_geom, hfield=field)],
            flat_patches=flat_patches,
        )
        output = _mark_as_terrain(_remove_height_material(output), group=self.geom_group)
        return enforce_flat_hfield_border(
            output, spec, self.border_width, height_offset=self.interior_height_offset
        ) if self.exact_flat_border else output


def spawn_patch_sampling(
    *, patch_radius: float, patch_center_range: float, patch_size: tuple[float, float]
) -> dict[str, FlatPatchSamplingCfg]:
    """Constrain rough-terrain reset patches to the central safe region."""
    return {
        "spawn": FlatPatchSamplingCfg(
            num_patches=8,
            patch_radius=patch_radius,
            max_height_diff=0.025,
            x_range=(patch_size[0] / 2 - patch_center_range, patch_size[0] / 2 + patch_center_range),
            y_range=(patch_size[1] / 2 - patch_center_range, patch_size[1] / 2 + patch_center_range),
        )
    }


@dataclass(kw_only=True)
class ScatteredBoxPlatformsTerrainCfg(SubTerrainCfg):
    """Irregular non-overlapping rectangular platforms on a connected base."""

    easy_height_range: tuple[float, float] = (0.05, 0.10)
    hard_height_range: tuple[float, float] = (0.10, 0.18)
    center_width: float = 1.2
    border_width: float = 0.5
    gap_width: float = 0.25
    min_platform_width: float = 0.60
    raised_fraction: float = 1.0
    row_count_range: tuple[int, int] = (5, 7)
    column_count_range: tuple[int, int] = (4, 7)
    spawn_patch_margin: float = 0.25
    num_spawn_patches: int = 12
    base_thickness: float = 0.2

    @staticmethod
    def _partition(length: float, count: int, rng: np.random.Generator) -> np.ndarray:
        weights = rng.uniform(0.85, 1.15, size=count)
        return length * weights / weights.sum()

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        if min(self.size) <= 2 * self.border_width:
            raise ValueError("Terrain patch is too small for the requested flat border.")
        if self.center_width <= 0.0 or self.center_width >= min(self.size) - 2 * self.border_width:
            raise ValueError("center_width must fit inside the platform field.")
        if self.gap_width < 0.0 or self.min_platform_width <= 0.0 or not 0.0 <= self.raised_fraction <= 1.0:
            raise ValueError("gap_width and raised_fraction are invalid.")
        if self.num_spawn_patches <= 0 or self.spawn_patch_margin < 0.0:
            raise ValueError("Platform spawn-patch parameters must be positive.")

        easy_low, easy_high = self.easy_height_range
        hard_low, hard_high = self.hard_height_range
        if not 0.0 <= easy_low <= easy_high or not 0.0 <= hard_low <= hard_high:
            raise ValueError("Platform height ranges must be ordered and non-negative.")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        height_low = easy_low + difficulty * (hard_low - easy_low)
        height_high = easy_high + difficulty * (hard_high - easy_high)

        center_x = self.size[0] / 2
        center_y = self.size[1] / 2
        interior_x = self.size[0] - 2 * self.border_width
        interior_y = self.size[1] - 2 * self.border_width
        body = spec.body("terrain")
        geometries = [
            _add_box(
                body,
                center_xy=(center_x, center_y),
                size_xy=self.size,
                bottom=-self.base_thickness,
                top=0.0,
                color=(0.18, 0.38, 0.24, 1.0),
            )
        ]

        min_rows, max_rows = self.row_count_range
        min_cols, max_cols = self.column_count_range
        if min_rows <= 0 or min_cols <= 0 or min_rows > max_rows or min_cols > max_cols:
            raise ValueError("Platform row/column count ranges are invalid.")
        row_count = int(rng.integers(min_rows, max_rows + 1))
        rows_below = (row_count - 1) // 2
        rows_above = row_count - 1 - rows_below
        side_y = (interior_y - self.center_width) / 2
        row_heights = np.concatenate(
            (
                self._partition(side_y, rows_below, rng),
                np.array([self.center_width]),
                self._partition(side_y, rows_above, rng),
            )
        )
        spawn_candidates: list[np.ndarray] = []
        y_cursor = self.border_width

        for row_index, row_height in enumerate(row_heights):
            column_count = int(rng.integers(min_cols, max_cols + 1))
            columns_left = (column_count - 1) // 2
            columns_right = column_count - 1 - columns_left
            side_x = (interior_x - self.center_width) / 2
            column_widths = np.concatenate(
                (
                    self._partition(side_x, columns_left, rng),
                    np.array([self.center_width]),
                    self._partition(side_x, columns_right, rng),
                )
            )
            x_cursor = self.border_width
            for column_index, column_width in enumerate(column_widths):
                rect_x0 = x_cursor + self.gap_width / 2
                rect_x1 = x_cursor + column_width - self.gap_width / 2
                rect_y0 = y_cursor + self.gap_width / 2
                rect_y1 = y_cursor + row_height - self.gap_width / 2
                x_cursor += column_width
                is_center = row_index == rows_below and column_index == columns_left
                if rect_x1 <= rect_x0 or rect_y1 <= rect_y0:
                    continue
                if is_center or rng.random() >= self.raised_fraction:
                    continue

                height = float(rng.uniform(height_low, height_high))
                height_alpha = 0.0 if height_high <= height_low else (height - height_low) / (height_high - height_low)
                color = (0.22 + 0.10 * height_alpha, 0.56 + 0.18 * height_alpha, 0.34, 1.0)
                rectangles = [(rect_x0, rect_x1, rect_y0, rect_y1)]
                for part_x0, part_x1, part_y0, part_y1 in rectangles:
                    rect_size = (part_x1 - part_x0, part_y1 - part_y0)
                    if min(rect_size) < self.min_platform_width:
                        continue
                    rect_center = ((part_x0 + part_x1) / 2, (part_y0 + part_y1) / 2)
                    geometries.append(
                        _add_box(
                            body,
                            center_xy=rect_center,
                            size_xy=rect_size,
                            bottom=0.0,
                            top=height,
                            color=color,
                        )
                    )
                    if min(rect_size) >= 2 * self.spawn_patch_margin:
                        spawn_candidates.append(np.array([rect_center[0], rect_center[1], height], dtype=np.float64))
            y_cursor += row_height

        if not spawn_candidates:
            raise RuntimeError("Scattered platform generation produced no safe raised reset patch.")
        selected = [spawn_candidates[index % len(spawn_candidates)] for index in range(self.num_spawn_patches)]
        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_x, center_y, 0.0]),
                geometries=geometries,
                flat_patches={"platform_spawn": np.stack(selected)},
            )
        )
