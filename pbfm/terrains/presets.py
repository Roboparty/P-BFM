"""Robot-independent physical terrain presets for training and evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import mujoco
import numpy as np
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    FlatPatchSamplingCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from pbfm.terrains.primitives import (
    NeutralHfDiscreteObstaclesTerrainCfg,
    NeutralHfPerlinNoiseTerrainCfg,
    NeutralHfPyramidSlopedTerrainCfg,
    ScatteredBoxPlatformsTerrainCfg,
    TerrainBoundedStairsCfg,
    TerrainBoxFlatCfg,
    TerrainInvertedPyramidStairsCfg,
    TerrainSeparatedPyramidStairsCfg,
    TerrainTraversalCourseCfg,
    spawn_patch_sampling,
)

TerrainMode = Literal[
    "plane", "flat", "slope", "stairs", "stairs_up", "stairs_down", "rough", "platforms",
    "mixed", "training_mix", "course",
]
SUPPORTED_TERRAINS: tuple[TerrainMode, ...] = (
    "plane",
    "flat",
    "slope",
    "stairs",
    "stairs_up",
    "stairs_down",
    "rough",
    "platforms",
    "mixed",
    "training_mix",
    "course",
)
TERRAIN_COMPONENT_NAMES = ("flat", "slope", "stairs_up", "stairs_down", "rough", "platforms")
TRAINING_TERRAIN_COMPONENT_NAMES = (
    "flat",
    "perlin_rough",
    "low_stairs_up",
    "low_stairs_down",
    "low_platforms",
    "hf_pyramid_slope_inv",
    "boxes",
)
TRAINING_TERRAIN_PROPORTIONS = {
    # The asset names describe the height profile towards the tile center.
    # From a center reset, ``low_stairs_down`` is ascent and
    # ``low_stairs_up`` is descent (see TRAINING_CENTER_RESET_PROFILES below).
    "flat": 0.20,
    "perlin_rough": 0.10,
    "low_stairs_up": 0.20,
    "low_stairs_down": 0.20,
    "low_platforms": 0.10,
    "hf_pyramid_slope_inv": 0.10,
    "boxes": 0.10,
}
# MJLab names the stair assets by their height profile when approaching the
# tile center. Training resets at that center, so the initial policy traversal is
# the opposite direction: the pyramid starts high and descends outwards,
# while the inverted pyramid starts low and ascends outwards. Keep the raw
# asset names for geometry/checkpoint compatibility and use these profiles in
# training diagnostics and same-z evaluation output.
TRAINING_CENTER_RESET_PROFILES = {
    "flat": "flat",
    "perlin_rough": "perlin_rough",
    "low_stairs_up": "low_stairs_descent_from_center",
    "low_stairs_down": "low_stairs_ascent_from_center",
    "low_platforms": "low_platforms",
    "hf_pyramid_slope_inv": "hf_pyramid_slope_inv",
    "boxes": "boxes",
}
TRAINING_CENTER_RESET_VERTICAL_DIRECTIONS = {
    "flat": "non_stair",
    "perlin_rough": "non_stair",
    "low_stairs_up": "descent",
    "low_stairs_down": "ascent",
    "low_platforms": "non_stair",
    "hf_pyramid_slope_inv": "non_stair",
    "boxes": "non_stair",
}
TRAINING_TERRAIN_REFERENCE_PROJECT = "UFO-rp1"
TRAINING_TERRAIN_REFERENCE_COMMIT = "8c364e1001734097aac58e5033a1b5076925d3c5"
TRAINING_PATCH_SIZE = 5.0
# Inside each tile: adjacent non-flat interiors have a 1 m flat corridor.
TRAINING_TILE_FLAT_BORDER_WIDTH = 0.5
TRAINING_CENTER_PLATFORM_WIDTH = 1.2
TRAINING_NONFLAT_GUARD_WIDTH = 10.0
TRAINING_FLAT_SAFETY_WIDTH = 2.0
TRAINING_TERRAIN_BORDER_WIDTH = TRAINING_NONFLAT_GUARD_WIDTH + TRAINING_FLAT_SAFETY_WIDTH
TRAINING_GUARD_TILE_RINGS = int(TRAINING_NONFLAT_GUARD_WIDTH / TRAINING_PATCH_SIZE)
TRAINING_GUARD_HEIGHT_LIFT = 0.05
TRAINING_GUARD_FAMILIES = ("perlin_rough", "boxes")
TRAINING_STAIR_LEVELS = 5
TRAINING_STAIR_STEP_HEIGHT_RANGE = (0.10, 0.20)
TRAINING_ROUGH_HEIGHT_RANGE = (0.02, 0.06)
TRAINING_BOX_HEIGHT_RANGE = (0.05, 0.15)
TRAINING_BOX_WIDTH_RANGE = (0.3, 1.0)
TRAINING_BOX_WIDTH_SAMPLING_STRIDE_CELLS = 1
TRAINING_STAIR_BORDER_WIDTH = 0.5
TRAINING_STAIR_STEP_WIDTH = 0.3
TRAINING_STAIR_PLATFORM_WIDTH = TRAINING_CENTER_PLATFORM_WIDTH
TRAINING_OUTER_WALL_HEIGHT = 5.0
TRAINING_OUTER_WALL_THICKNESS = 0.05
TRAINING_OUTER_WALL_COLOR = (0.5, 0.5, 0.5, 1.0)


def normalize_terrain_mode(mode: str) -> str:
    """Read the terrain name used by archived environment configurations."""
    return "training_mix" if mode == "rp1_simple" else mode


def center_reset_profile(asset_family: str) -> tuple[str, str]:
    """Return the direction-correct center-reset label for a terrain asset."""

    try:
        return (
            TRAINING_CENTER_RESET_PROFILES[asset_family],
            TRAINING_CENTER_RESET_VERTICAL_DIRECTIONS[asset_family],
        )
    except KeyError as error:
        raise ValueError(f"unknown terrain asset family: {asset_family!r}") from error


def _get(config: Any, path: str, default: Any) -> Any:
    value = config
    for part in path.split("."):
        if value is None:
            return default
        value = value.get(part, default) if isinstance(value, Mapping) else getattr(value, part, default)
        if value is default:
            return default
    return value


def terrain_component_names(mode: TerrainMode) -> tuple[str, ...]:
    mode = normalize_terrain_mode(mode)
    if mode == "training_mix":
        return TRAINING_TERRAIN_COMPONENT_NAMES
    if mode == "mixed":
        return TERRAIN_COMPONENT_NAMES
    if mode == "plane":
        return ("flat",)
    if mode == "course":
        return ("course",)
    return (mode,)


def _terrain_mix(config: Any) -> dict[str, float]:
    raw = _get(config, "terrain_mix", None)
    default_weight = 1.0 / len(TERRAIN_COMPONENT_NAMES)
    weights = {name: float(_get(raw, name, default_weight)) for name in TERRAIN_COMPONENT_NAMES}
    if any(value < 0.0 for value in weights.values()) or sum(weights.values()) <= 0.0:
        raise ValueError(f"terrain_mix must be non-negative and have positive total weight: {weights}")
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def make_ufo_v0_generator_cfg(mode: TerrainMode, config: Any) -> TerrainGeneratorCfg:
    """Create collision terrain and spawn origins from one MJLab generator."""
    mode = normalize_terrain_mode(mode)
    if mode == "training_mix":
        return make_training_mix_generator_cfg()
    selected = terrain_component_names(mode)
    weights = _terrain_mix(config)
    if mode != "mixed":
        weights = {name: float(name == mode) for name in TERRAIN_COMPONENT_NAMES}

    if mode == "course":
        size_path = "course.patch_size"
    elif mode == "stairs":
        size_path = "stairs.legacy_patch_size"
    else:
        size_path = "patch_size"
    size = tuple(float(x) for x in _get(config, size_path, (8.0, 8.0)))
    slope_min_deg = float(_get(config, "slope.min_angle_deg", 5.0))
    slope_max_deg = float(_get(config, "slope.max_angle_deg", 12.0))
    step_height = tuple(float(x) for x in _get(config, "stairs.step_height_range", (0.10, 0.18)))
    horizontal_scale = float(_get(config, "heightfield.horizontal_scale", 0.1))
    spawn_center_range = float(_get(config, "spawn.center_range", 1.25))

    all_sub_terrains = {
        "flat": TerrainBoxFlatCfg(proportion=weights["flat"]),
        # A safe center platform becomes higher away from the origin. This
        # keeps arbitrary LaFAN headings valid while providing uphill terrain.
        "slope": NeutralHfPyramidSlopedTerrainCfg(
            proportion=weights["slope"],
            slope_range=(math.tan(math.radians(slope_min_deg)), math.tan(math.radians(slope_max_deg))),
            platform_width=float(_get(config, "slope.platform_width", 1.5)),
            inverted=True,
            border_width=float(_get(config, "slope.border_width", 0.5)),
            horizontal_scale=horizontal_scale,
            vertical_scale=float(_get(config, "heightfield.vertical_scale", 0.005)),
        ),
        # Retain the old combined staircase as an explicit legacy inference
        # mode; mixed training uses separate ascent and descent families.
        "stairs": TerrainBoundedStairsCfg(
            proportion=1.0,
            step_height_range=step_height,
            step_width=float(_get(config, "stairs.step_depth", 0.30)),
            platform_width=float(_get(config, "stairs.legacy_platform_width", 1.0)),
            num_steps=int(_get(config, "stairs.legacy_num_steps", 6)),
            plateau_width=float(_get(config, "stairs.plateau_width", 0.8)),
            border_width=float(_get(config, "stairs.border_width", 0.5)),
        ),
        "stairs_up": TerrainSeparatedPyramidStairsCfg(
            proportion=weights["stairs_up"],
            direction="up",
            step_height_range=step_height,
            step_width=float(_get(config, "stairs.step_depth", 0.30)),
            platform_width=float(_get(config, "stairs.platform_width", 0.80)),
            num_steps=int(_get(config, "stairs.num_steps", 10)),
            border_width=float(_get(config, "stairs.border_width", 0.5)),
        ),
        "stairs_down": TerrainSeparatedPyramidStairsCfg(
            proportion=weights["stairs_down"],
            direction="down",
            step_height_range=step_height,
            step_width=float(_get(config, "stairs.step_depth", 0.30)),
            platform_width=float(_get(config, "stairs.platform_width", 0.80)),
            num_steps=int(_get(config, "stairs.num_steps", 10)),
            border_width=float(_get(config, "stairs.border_width", 0.5)),
        ),
        "rough": NeutralHfPerlinNoiseTerrainCfg(
            proportion=weights["rough"],
            height_range=tuple(float(x) for x in _get(config, "rough.amplitude_range", (0.03, 0.08))),
            octaves=int(_get(config, "rough.octaves", 2)),
            persistence=float(_get(config, "rough.persistence", 0.25)),
            lacunarity=float(_get(config, "rough.lacunarity", 2.0)),
            scale=float(_get(config, "rough.spatial_scale", 20.0)),
            horizontal_scale=horizontal_scale,
            resolution=horizontal_scale,
            border_width=float(_get(config, "rough.border_width", 0.5)),
            flat_patch_sampling=spawn_patch_sampling(
                patch_radius=float(_get(config, "spawn.patch_radius", 0.45)),
                patch_center_range=spawn_center_range,
                patch_size=size,
            ),
        ),
        "platforms": ScatteredBoxPlatformsTerrainCfg(
            proportion=weights["platforms"],
            easy_height_range=tuple(float(x) for x in _get(config, "platforms.easy_height_range", (0.05, 0.10))),
            hard_height_range=tuple(float(x) for x in _get(config, "platforms.hard_height_range", (0.10, 0.18))),
            center_width=float(_get(config, "platforms.center_width", 1.2)),
            border_width=float(_get(config, "platforms.border_width", 0.5)),
            gap_width=float(_get(config, "platforms.gap_width", 0.25)),
            min_platform_width=float(_get(config, "platforms.min_platform_width", 0.60)),
            raised_fraction=float(_get(config, "platforms.raised_fraction", 1.0)),
            row_count_range=tuple(int(x) for x in _get(config, "platforms.row_count_range", (5, 7))),
            column_count_range=tuple(int(x) for x in _get(config, "platforms.column_count_range", (4, 7))),
            spawn_patch_margin=float(_get(config, "reset.platforms_edge_margin", 0.25)),
            num_spawn_patches=int(_get(config, "platforms.num_spawn_patches", 12)),
            flat_patch_sampling={
                "platform_spawn": FlatPatchSamplingCfg(
                    num_patches=int(_get(config, "platforms.num_spawn_patches", 12)),
                    patch_radius=float(_get(config, "reset.platforms_edge_margin", 0.25)),
                )
            },
        ),
        "course": TerrainTraversalCourseCfg(
            proportion=1.0,
            flat_run=float(_get(config, "course.flat_run", 1.80)),
            step_height=float(_get(config, "course.step_height", 0.12)),
            step_depth=float(_get(config, "course.step_depth", 0.30)),
            num_steps=int(_get(config, "course.num_steps", 5)),
            top_platform_length=float(_get(config, "course.top_platform_length", 0.80)),
            connector_length=float(_get(config, "course.connector_length", 1.00)),
            include_ramp=bool(_get(config, "course.include_ramp", True)),
            ramp_length=float(_get(config, "course.ramp_length", 2.50)),
            ramp_angle_deg=float(_get(config, "course.ramp_angle_deg", 8.0)),
            border_width=float(_get(config, "course.border_width", 0.50)),
            horizontal_scale=float(_get(config, "heightfield.horizontal_scale", 0.10)),
        ),
    }
    sub_terrains = {name: all_sub_terrains[name] for name in selected}
    return TerrainGeneratorCfg(
        seed=int(_get(config, "seed", 0)),
        size=size,
        border_width=float(_get(config, "border_width", 1.0)),
        num_rows=int(_get(config, "num_rows", 10)),
        num_cols=len(sub_terrains),
        curriculum=True,
        difficulty_range=tuple(float(x) for x in _get(config, "difficulty_range", (0.0, 1.0))),
        color_scheme="none",
        sub_terrains=sub_terrains,
        add_lights=False,
    )


def make_training_mix_generator_cfg() -> TerrainGeneratorCfg:
    """Build 5 m tiles with a 0.5 m z=0 flat border on every side."""
    return TerrainGeneratorCfg(
        seed=0,
        size=(TRAINING_PATCH_SIZE, TRAINING_PATCH_SIZE),
        border_width=TRAINING_TERRAIN_BORDER_WIDTH,
        num_rows=10,
        num_cols=7,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        color_scheme="none",
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=TRAINING_TERRAIN_PROPORTIONS["flat"]),
            "perlin_rough": NeutralHfPerlinNoiseTerrainCfg(
                proportion=TRAINING_TERRAIN_PROPORTIONS["perlin_rough"],
                height_range=TRAINING_ROUGH_HEIGHT_RANGE,
                octaves=2,
                persistence=0.25,
                lacunarity=2.0,
                scale=20.0,
                horizontal_scale=0.2,
                resolution=0.2,
                border_width=TRAINING_TILE_FLAT_BORDER_WIDTH,
                exact_flat_border=True,
                geom_group=0,
            ),
            "low_stairs_up": BoxPyramidStairsTerrainCfg(
                proportion=TRAINING_TERRAIN_PROPORTIONS["low_stairs_up"],
                step_height_range=TRAINING_STAIR_STEP_HEIGHT_RANGE,
                step_width=TRAINING_STAIR_STEP_WIDTH,
                platform_width=TRAINING_STAIR_PLATFORM_WIDTH,
                border_width=TRAINING_TILE_FLAT_BORDER_WIDTH,
            ),
            "low_stairs_down": BoxInvertedPyramidStairsTerrainCfg(
                proportion=TRAINING_TERRAIN_PROPORTIONS["low_stairs_down"],
                step_height_range=TRAINING_STAIR_STEP_HEIGHT_RANGE,
                step_width=TRAINING_STAIR_STEP_WIDTH,
                platform_width=TRAINING_STAIR_PLATFORM_WIDTH,
                border_width=TRAINING_TILE_FLAT_BORDER_WIDTH,
            ),
            "low_platforms": BoxRandomGridTerrainCfg(
                proportion=TRAINING_TERRAIN_PROPORTIONS["low_platforms"],
                grid_width=0.8,
                grid_height_range=(0.03, 0.10),
                platform_width=TRAINING_CENTER_PLATFORM_WIDTH,
                merge_similar_heights=True,
                height_merge_threshold=0.025,
                max_merge_distance=3,
                border_width=TRAINING_TILE_FLAT_BORDER_WIDTH,
            ),
            "hf_pyramid_slope_inv": NeutralHfPyramidSlopedTerrainCfg(
                proportion=TRAINING_TERRAIN_PROPORTIONS["hf_pyramid_slope_inv"],
                slope_range=(0.0, 0.2),
                platform_width=TRAINING_CENTER_PLATFORM_WIDTH,
                inverted=True,
                border_width=TRAINING_TILE_FLAT_BORDER_WIDTH,
                exact_flat_border=True,
                horizontal_scale=0.2,
                geom_group=0,
            ),
            "boxes": NeutralHfDiscreteObstaclesTerrainCfg(
                proportion=TRAINING_TERRAIN_PROPORTIONS["boxes"],
                num_obstacles=50,
                obstacle_height_mode="choice",
                obstacle_width_range=TRAINING_BOX_WIDTH_RANGE,
                obstacle_height_range=TRAINING_BOX_HEIGHT_RANGE,
                obstacle_width_sampling_stride_cells=TRAINING_BOX_WIDTH_SAMPLING_STRIDE_CELLS,
                platform_width=TRAINING_CENTER_PLATFORM_WIDTH,
                border_width=TRAINING_TILE_FLAT_BORDER_WIDTH,
                exact_flat_border=True,
                horizontal_scale=0.2,
                vertical_scale=0.005,
                geom_group=0,
            ),
        },
        add_lights=False,
    )


def _make_training_guard_sub_terrains() -> dict[str, Any]:
    """Create efficient non-flat heightfields for the non-spawn guard ring."""
    return {
        "perlin_rough": NeutralHfPerlinNoiseTerrainCfg(
            proportion=0.5,
            height_range=TRAINING_ROUGH_HEIGHT_RANGE,
            octaves=2,
            persistence=0.25,
            lacunarity=2.0,
            scale=20.0,
            horizontal_scale=0.2,
            resolution=0.2,
            border_width=TRAINING_TILE_FLAT_BORDER_WIDTH,
            exact_flat_border=True,
            interior_height_offset=TRAINING_GUARD_HEIGHT_LIFT,
            geom_group=0,
        ),
        "boxes": NeutralHfDiscreteObstaclesTerrainCfg(
            proportion=0.5,
            num_obstacles=50,
            obstacle_height_mode="choice",
            obstacle_width_range=TRAINING_BOX_WIDTH_RANGE,
            obstacle_height_range=TRAINING_BOX_HEIGHT_RANGE,
            obstacle_width_sampling_stride_cells=TRAINING_BOX_WIDTH_SAMPLING_STRIDE_CELLS,
            platform_width=TRAINING_CENTER_PLATFORM_WIDTH,
            border_width=TRAINING_TILE_FLAT_BORDER_WIDTH,
            exact_flat_border=True,
            interior_height_offset=TRAINING_GUARD_HEIGHT_LIFT,
            horizontal_scale=0.2,
            vertical_scale=0.005,
            geom_group=0,
        ),
    }


def add_training_nonflat_guard_ring(
    spec: mujoco.MjSpec,
    generator: TerrainGeneratorCfg,
) -> tuple[int, int]:
    """Add two non-spawn 5 m tile rings over the inner 10 m of the border.

    The generator-owned border remains as a continuous collision underlay.
    Guard interiors are lifted slightly above it, but their 0.5 m borders
    remain at z=0 to join the core without a ledge. The outermost 2 m
    remains exposed flat terrain before the collision walls.
    """
    terrain_body = spec.body("terrain")
    if terrain_body is None:
        raise ValueError("Terrain guard ring requires a compiled terrain body")
    tile_x, tile_y = (float(value) for value in generator.size)
    rings_x = int(round(TRAINING_NONFLAT_GUARD_WIDTH / tile_x))
    rings_y = int(round(TRAINING_NONFLAT_GUARD_WIDTH / tile_y))
    if not math.isclose(rings_x * tile_x, TRAINING_NONFLAT_GUARD_WIDTH) or not math.isclose(
        rings_y * tile_y, TRAINING_NONFLAT_GUARD_WIDTH
    ):
        raise ValueError("Terrain non-flat guard width must be an integer number of terrain tiles")
    if not math.isclose(float(generator.border_width), TRAINING_TERRAIN_BORDER_WIDTH):
        raise ValueError(
            "Terrain generator border must equal the non-flat guard plus flat safety width"
        )

    num_rows = int(generator.num_rows)
    num_cols = len(generator.sub_terrains) if generator.curriculum else int(generator.num_cols)
    core_x_min = -num_rows * tile_x / 2.0
    core_y_min = -num_cols * tile_y / 2.0
    guard_cfgs = _make_training_guard_sub_terrains()
    for guard_cfg in guard_cfgs.values():
        guard_cfg.size = (tile_x, tile_y)
    rng = np.random.default_rng(int(generator.seed or 0) + 104_729)
    lower, upper = (float(value) for value in generator.difficulty_range)
    tile_count = 0
    geom_count = 0
    for row in range(-rings_x, num_rows + rings_x):
        for col in range(-rings_y, num_cols + rings_y):
            if 0 <= row < num_rows and 0 <= col < num_cols:
                continue
            family = TRAINING_GUARD_FAMILIES[tile_count % len(TRAINING_GUARD_FAMILIES)]
            guard_cfg = guard_cfgs[family]
            phase = ((tile_count % 10) + 0.5) / 10.0
            difficulty = lower + phase * (upper - lower)
            output = guard_cfg.function(difficulty, spec, rng)
            world_offset = np.array(
                (
                    core_x_min + row * tile_x,
                    core_y_min + col * tile_y,
                    0.0,
                )
            )
            for local_geom_index, geometry in enumerate(output.geometries):
                geom = geometry.geom
                if geom is None:
                    continue
                geom.pos = np.asarray(geom.pos) + world_offset
                geom.name = f"g1depth_guard_tile_{tile_count:03d}_geom_{local_geom_index:03d}"
                geom.group = 0
                geom.mass = 0.0
                geom.rgba = (0.5, 0.5, 0.5, 1.0)
                geom_count += 1
            tile_count += 1
    return tile_count, geom_count


def add_training_outer_walls(spec: mujoco.MjSpec, generator: TerrainGeneratorCfg) -> None:
    """Add four collision walls outside the 10 m guard and 2 m flat safety ring."""
    terrain_body = spec.body("terrain")
    if terrain_body is None:
        raise ValueError("Terrain outer walls require a compiled terrain body")
    num_cols = len(generator.sub_terrains) if generator.curriculum else generator.num_cols
    outer_x = generator.num_rows * generator.size[0] + 2.0 * generator.border_width
    outer_y = num_cols * generator.size[1] + 2.0 * generator.border_width
    half_height = TRAINING_OUTER_WALL_HEIGHT / 2.0
    half_thickness = TRAINING_OUTER_WALL_THICKNESS / 2.0
    walls = (
        (
            "rp1_outer_wall_x_negative",
            (-outer_x / 2.0 - half_thickness, 0.0, half_height),
            (half_thickness, outer_y / 2.0 + TRAINING_OUTER_WALL_THICKNESS, half_height),
        ),
        (
            "rp1_outer_wall_x_positive",
            (outer_x / 2.0 + half_thickness, 0.0, half_height),
            (half_thickness, outer_y / 2.0 + TRAINING_OUTER_WALL_THICKNESS, half_height),
        ),
        (
            "rp1_outer_wall_y_negative",
            (0.0, -outer_y / 2.0 - half_thickness, half_height),
            (outer_x / 2.0, half_thickness, half_height),
        ),
        (
            "rp1_outer_wall_y_positive",
            (0.0, outer_y / 2.0 + half_thickness, half_height),
            (outer_x / 2.0, half_thickness, half_height),
        ),
    )
    for name, pos, size in walls:
        wall = terrain_body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
        wall.rgba = TRAINING_OUTER_WALL_COLOR
        wall.group = 0
        wall.contype = 1
        wall.conaffinity = 1


def make_terrain_entity_cfg(
    mode: TerrainMode,
    *,
    env_spacing: float,
    config: Any = None,
    inference_all_stairs: bool = False,
    inference_all_flat: bool = False,
) -> TerrainEntityCfg:
    mode = normalize_terrain_mode(mode)
    if mode == "plane":
        return TerrainEntityCfg(terrain_type="plane", env_spacing=env_spacing)
    if mode not in SUPPORTED_TERRAINS:
        raise ValueError(f"Unsupported terrain mode: {mode!r}. Expected one of {SUPPORTED_TERRAINS}")
    generator = make_ufo_v0_generator_cfg(mode, config)
    if mode == "training_mix":
        if inference_all_stairs and inference_all_flat:
            raise ValueError("inference_all_stairs and inference_all_flat are mutually exclusive")
        if inference_all_stairs:
            generator.sub_terrains = {
                name: BoxPyramidStairsTerrainCfg(
                    proportion=TRAINING_TERRAIN_PROPORTIONS[name],
                    step_height_range=(0.15, 0.15),
                    step_width=TRAINING_STAIR_STEP_WIDTH,
                    platform_width=TRAINING_STAIR_PLATFORM_WIDTH,
                    border_width=TRAINING_TILE_FLAT_BORDER_WIDTH,
                )
                for name in generator.sub_terrains
            }
        elif inference_all_flat:
            generator.sub_terrains = {
                name: BoxFlatTerrainCfg(proportion=TRAINING_TERRAIN_PROPORTIONS[name])
                for name in generator.sub_terrains
            }
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=generator,
        env_spacing=None,
        max_init_terrain_level=generator.num_rows - 1,
    )
