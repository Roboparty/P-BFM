"""Check real MuJoCo collision surfaces, not just border configuration values."""
import mujoco
import numpy as np
import pytest

from pbfm.terrains.presets import (
    make_training_mix_generator_cfg, _make_training_guard_sub_terrains,
    make_terrain_entity_cfg,
)


def ray_height(model, data, x, y):
    geomid = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        model, data, np.array([x, y, 3.0]), np.array([0., 0., -1.]),
        None, 1, -1, geomid,
    )
    assert distance >= 0, (x, y, "missing collision floor")
    return 3.0 - distance


@pytest.mark.parametrize("difficulty", [0., .5, 1.])
@pytest.mark.parametrize("guard", [False, True])
def test_all_tiles_have_half_meter_zero_height_border(difficulty, guard):
    terrains = _make_training_guard_sub_terrains() if guard else make_training_mix_generator_cfg().sub_terrains
    for name, cfg in terrains.items():
        cfg.size = (5., 5.)
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        cfg.function(difficulty, spec, np.random.default_rng(31))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Four sides, near the outer edge, middle and inner edge of the band.
        for distance in (.001, .25, .4999):
            for along in np.linspace(.001, 4.999, 25):
                for x, y in ((distance, along), (5-distance, along),
                             (along, distance), (along, 5-distance)):
                    assert abs(ray_height(model, data, x, y)) < 1e-6, (name, x, y)


def test_stairs_keep_height_and_have_five_rises_with_zero_border():
    for name in ("low_stairs_up", "low_stairs_down"):
        cfg = make_training_mix_generator_cfg().sub_terrains[name]
        cfg.size = (5., 5.)
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        cfg.function(1., spec, np.random.default_rng(1))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        heights = np.array([ray_height(model, data, x, 2.5) for x in np.linspace(.01, 2.5, 251)])
        changes = np.diff(heights)
        rises = changes[np.abs(changes) > 1e-6]
        assert len(rises) == 5
        np.testing.assert_allclose(np.abs(rises), .20, atol=1e-6)


def test_all_stairs_inference_uses_same_border():
    generator = make_terrain_entity_cfg("training_mix", env_spacing=2., inference_all_stairs=True).terrain_generator
    assert all(cfg.border_width == .5 for cfg in generator.sub_terrains.values())


def test_heightfield_asset_lookup_before_scene_name_index_refresh():
    from types import SimpleNamespace
    from pbfm.terrains.primitives import enforce_flat_hfield_border
    cfg = make_training_mix_generator_cfg().sub_terrains['perlin_rough']
    cfg.size = (5.,5.)
    cfg.exact_flat_border = False
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name='terrain')
    output = cfg.function(.8,spec,np.random.default_rng(1))
    delayed_lookup = SimpleNamespace(hfield=lambda name:None, hfields=spec.hfields)
    enforce_flat_hfield_border(output,delayed_lookup,.5)
    model = spec.compile(); data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    assert abs(ray_height(model,data,.4999,2.5)) < 1e-6
