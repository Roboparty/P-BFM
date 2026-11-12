import numpy as np

from utils.depth_preprocess import (
    DepthCropConfig,
    apply_conservative_output_invalid_mask,
    gate_metric_depth,
    gaussian_blur_metric_depth,
    resize_full_fov_depth_area,
    scale_intrinsics_full_fov,
    crop_and_resize_depth_area,
    crop_and_resize_depth_with_conservative_invalid_mask,
    crop_and_scale_intrinsics,
)


def test_conservative_output_mask_never_leaks_finite_self_depth():
    depth = np.ones((3, 4), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[1, 2] = True
    filtered = apply_conservative_output_invalid_mask(depth, mask)
    assert np.isnan(filtered[1, 2])
    assert np.count_nonzero(np.isfinite(filtered)) == depth.size - 1


def test_metric_depth_gate_invalidates_without_clamping():
    source = np.array(
        [[np.nan, -1.0, 0.0, 0.0999, 0.1, 0.2, 2.0, 2.0001, np.inf]],
        dtype=np.float32,
    )
    gated = gate_metric_depth(source, min_depth_m=0.1, max_depth_m=2.0)
    assert gated.dtype == np.float32
    np.testing.assert_array_equal(
        np.isfinite(gated),
        [[False, False, False, False, True, True, True, False, False]],
    )
    np.testing.assert_allclose(gated[0, 4:7], [0.1, 0.2, 2.0])


def test_metric_depth_gate_rejects_invalid_contract():
    with np.testing.assert_raises_regex(ValueError, "finite and positive"):
        gate_metric_depth(np.ones((2, 2), dtype=np.float32), max_depth_m=np.inf)
    with np.testing.assert_raises_regex(ValueError, "finite and non-negative"):
        gate_metric_depth(
            np.ones((2, 2), dtype=np.float32), min_depth_m=-0.1, max_depth_m=2.0
        )
    with np.testing.assert_raises_regex(ValueError, "less than"):
        gate_metric_depth(
            np.ones((2, 2), dtype=np.float32), min_depth_m=2.0, max_depth_m=2.0
        )
    with np.testing.assert_raises_regex(ValueError, "shape"):
        gate_metric_depth(np.ones(2, dtype=np.float32), max_depth_m=2.0)


def test_gaussian_blur_preserves_invalid_mask_and_constant_depth():
    source = np.full((21, 25), 1.25, dtype=np.float32)
    source[8:13, 10:15] = np.nan
    blurred = gaussian_blur_metric_depth(source, sigma_px=1.5, radius_px=9)

    np.testing.assert_array_equal(np.isfinite(blurred), np.isfinite(source))
    np.testing.assert_allclose(blurred[np.isfinite(blurred)], 1.25, atol=2.0e-6)


def test_gaussian_blur_smooths_depth_edge_without_inpainting():
    source = np.ones((31, 31), dtype=np.float32)
    source[:, 16:] = 2.0
    source[15, 15] = np.nan
    blurred = gaussian_blur_metric_depth(source, sigma_px=1.5, radius_px=9)

    assert np.isnan(blurred[15, 15])
    assert 1.0 < blurred[15, 14] < blurred[15, 16] < 2.0
    np.testing.assert_allclose(blurred[:, 0], 1.0, atol=2.0e-6)
    np.testing.assert_allclose(blurred[:, -1], 2.0, atol=2.0e-6)


def test_gaussian_blur_rejects_invalid_contract():
    source = np.ones((2, 2), dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "sigma_px"):
        gaussian_blur_metric_depth(source, sigma_px=np.inf, radius_px=9)
    with np.testing.assert_raises_regex(ValueError, "radius_px"):
        gaussian_blur_metric_depth(source, sigma_px=1.0, radius_px=-1)


def test_scale_intrinsics_uses_pixel_centers():
    source = np.array(
        ((240.0, 0.0, 239.5), (0.0, 240.0, 134.5), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    target = scale_intrinsics_full_fov(
        source,
        source_width=480,
        source_height=270,
        target_width=64,
        target_height=36,
    )
    np.testing.assert_allclose(target[0, 0], 32.0)
    np.testing.assert_allclose(target[1, 1], 32.0)
    np.testing.assert_allclose(target[0, 2], 31.5)
    np.testing.assert_allclose(target[1, 2], 17.5)


def test_reference_crop_is_applied_at_native_resolution_and_updates_k():
    crop = DepthCropConfig(top=6, left=4, right=4)
    assert crop.bounds(source_width=480, source_height=270) == (45, 270, 30, 450)
    source = np.array(
        ((240.0, 0.0, 239.5), (0.0, 240.0, 134.5), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    target = crop_and_scale_intrinsics(
        source,
        source_width=480,
        source_height=270,
        target_width=64,
        target_height=36,
        crop=crop,
    )
    np.testing.assert_allclose(target[0, 2], 31.5)
    np.testing.assert_allclose(target[1, 2], (134.5 - 45.0 + 0.5) * 36.0 / 225.0 - 0.5)


def test_crop_resize_preserves_conservative_invalid_mask():
    crop = DepthCropConfig(top=6, left=4, right=4)
    depth = np.ones((36, 64), dtype=np.float32)
    depth[:6] = 9.0
    mask = np.zeros_like(depth, dtype=bool)
    mask[6:8, 4:8] = True
    resized = crop_and_resize_depth_area(
        depth,
        target_width=28,
        target_height=15,
        crop=crop,
    )
    np.testing.assert_allclose(resized, 1.0)
    conservative, output_mask = crop_and_resize_depth_with_conservative_invalid_mask(
        depth,
        mask,
        target_width=28,
        target_height=15,
        crop=crop,
    )
    assert np.any(output_mask)
    assert np.all(np.isnan(conservative[output_mask]))


def test_area_resize_preserves_constant_depth_and_nan_holes():
    source = np.full((270, 480), 1.25, dtype=np.float32)
    source[0:8, 0:8] = np.nan
    target = resize_full_fov_depth_area(
        source, target_width=64, target_height=36
    )
    assert target.shape == (36, 64)
    np.testing.assert_allclose(target[np.isfinite(target)], 1.25, atol=1.0e-6)


def test_area_resize_does_not_turn_fully_invalid_regions_into_zero():
    source = np.full((270, 480), 2.0, dtype=np.float32)
    source[:15, :15] = np.nan
    target = resize_full_fov_depth_area(
        source, target_width=64, target_height=36
    )
    assert np.isnan(target[0, 0])
    assert np.isnan(target[0, 1])
    assert np.isnan(target[1, 0])
    assert np.isnan(target[1, 1])
    assert np.all(target[2:, 2:] == 2.0)


def test_compact_area_resize_matches_dense_reference():
    rng = np.random.default_rng(17)
    source = rng.uniform(0.2, 2.5, size=(19, 31)).astype(np.float32)
    source[rng.random(source.shape) < 0.2] = np.nan

    target = resize_full_fov_depth_area(
        source, target_width=7, target_height=5
    )

    def dense_weights(source_size, target_size):
        scale = source_size / target_size
        weights = np.zeros((target_size, source_size), dtype=np.float32)
        for dst in range(target_size):
            start, end = dst * scale, (dst + 1) * scale
            for src in range(int(np.floor(start)), int(np.ceil(end))):
                overlap = max(0.0, min(end, src + 1.0) - max(start, src))
                weights[dst, src] = overlap / scale
        return weights

    valid = np.isfinite(source) & (source > 0.0)
    values = np.where(valid, source, 0.0)
    weights_y = dense_weights(19, 5)
    weights_x = dense_weights(31, 7)
    numerator = weights_y @ values @ weights_x.T
    denominator = weights_y @ valid.astype(np.float32) @ weights_x.T
    expected = np.full((5, 7), np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=expected, where=denominator > 0.0)
    # The sparse and dense forms accumulate float32 products in a different
    # order, so allow a few ULPs while still detecting any stencil mismatch.
    np.testing.assert_allclose(target, expected, rtol=0.0, atol=5.0e-7)
