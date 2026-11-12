"""Full-FOV depth resizing shared by simulated and live sensor producers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2 as _cv2
except ImportError:  # SciPy remains the portable/reference fallback.
    _cv2 = None

from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class DepthCropConfig:
    """ROI in pixels of a reference 64x36 image, applied before resize."""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    reference_width: int = 64
    reference_height: int = 36

    def validate(self) -> None:
        margins = (self.top, self.bottom, self.left, self.right)
        if any(not isinstance(value, int) or value < 0 for value in margins):
            raise ValueError("depth crop margins must be non-negative integers")
        if min(self.reference_width, self.reference_height) <= 0:
            raise ValueError("depth crop reference dimensions must be positive")
        if self.top + self.bottom >= self.reference_height:
            raise ValueError("vertical depth crop removes the full image")
        if self.left + self.right >= self.reference_width:
            raise ValueError("horizontal depth crop removes the full image")

    def bounds(self, *, source_width: int, source_height: int) -> tuple[int, int, int, int]:
        self.validate()
        if min(source_width, source_height) <= 0:
            raise ValueError("source dimensions must be positive")
        top = round(self.top * source_height / self.reference_height)
        bottom = source_height - round(self.bottom * source_height / self.reference_height)
        left = round(self.left * source_width / self.reference_width)
        right = source_width - round(self.right * source_width / self.reference_width)
        if top >= bottom or left >= right:
            raise ValueError("scaled depth crop removes the full source image")
        return top, bottom, left, right


def crop_depth_roi(depth_m: np.ndarray, crop: DepthCropConfig) -> np.ndarray:
    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError("depth_m must have shape [height, width]")
    top, bottom, left, right = crop.bounds(
        source_width=depth.shape[1],
        source_height=depth.shape[0],
    )
    return depth[top:bottom, left:right]


def gate_metric_depth(
    depth_m: np.ndarray,
    *,
    max_depth_m: float,
    min_depth_m: float = 0.0,
) -> np.ndarray:
    """Invalidate metric depth outside the configured range without clipping.

    Values beyond the sensor range must not be clamped onto either boundary,
    because that would create a synthetic surface at the configured limit.
    """
    if not np.isfinite(max_depth_m) or max_depth_m <= 0.0:
        raise ValueError("max_depth_m must be finite and positive")
    if not np.isfinite(min_depth_m) or min_depth_m < 0.0:
        raise ValueError("min_depth_m must be finite and non-negative")
    if min_depth_m >= max_depth_m:
        raise ValueError("min_depth_m must be less than max_depth_m")
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must have shape [height, width]")
    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    return np.where(valid, depth, np.nan).astype(np.float32, copy=False)


def gaussian_blur_metric_depth(
    depth_m: np.ndarray,
    *,
    sigma_px: float,
    radius_px: int,
) -> np.ndarray:
    """Blur metric depth without expanding its original valid mask.

    This matches the estimator training augmentation: numerator and valid-mask
    weights are blurred separately with zero padding, then normalized. Invalid
    input pixels remain ``NaN`` and therefore never become synthetic geometry.
    A fixed deployment sigma is used instead of sampling a new random sigma on
    every frame.
    """
    sigma_px = float(sigma_px)
    radius_px = int(radius_px)
    if not np.isfinite(sigma_px) or sigma_px < 0.0:
        raise ValueError("sigma_px must be finite and non-negative")
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must have shape [height, width]")
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > 0.0)
    if sigma_px <= 1.0e-6 or radius_px == 0:
        return np.where(valid, depth, np.nan).astype(np.float32, copy=False)

    values = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
    weights = valid.astype(np.float32)
    if _cv2 is not None:
        coordinates = np.arange(-radius_px, radius_px + 1, dtype=np.float32)
        kernel = np.exp(
            -(coordinates * coordinates) / np.float32(2.0 * sigma_px * sigma_px)
        ).astype(np.float32, copy=False)
        kernel /= np.sum(kernel, dtype=np.float32)
        blurred_values = _cv2.sepFilter2D(
            values,
            ddepth=-1,
            kernelX=kernel,
            kernelY=kernel,
            borderType=_cv2.BORDER_CONSTANT,
        )
        blurred_weights = _cv2.sepFilter2D(
            weights,
            ddepth=-1,
            kernelX=kernel,
            kernelY=kernel,
            borderType=_cv2.BORDER_CONSTANT,
        )
    else:
        truncate = float(radius_px) / sigma_px
        blurred_values = gaussian_filter(
            values,
            sigma=sigma_px,
            mode="constant",
            cval=0.0,
            truncate=truncate,
            output=np.float32,
        )
        blurred_weights = gaussian_filter(
            weights,
            sigma=sigma_px,
            mode="constant",
            cval=0.0,
            truncate=truncate,
            output=np.float32,
        )
    result = np.full(depth.shape, np.nan, dtype=np.float32)
    np.divide(
        blurred_values,
        blurred_weights,
        out=result,
        where=valid & (blurred_weights > 1.0e-8),
    )
    return result


def scale_intrinsics_full_fov(
    intrinsic_matrix: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Scale calibrated intrinsics using pixel-center image coordinates."""
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("depth image dimensions must be positive")
    matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("intrinsic_matrix must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) <= 1.0e-12:
        raise ValueError("intrinsic_matrix must be invertible")

    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    target = matrix.copy()
    target[0, 0] *= scale_x
    target[0, 1] *= scale_x
    target[0, 2] = (matrix[0, 2] + 0.5) * scale_x - 0.5
    target[1, 0] *= scale_y
    target[1, 1] *= scale_y
    target[1, 2] = (matrix[1, 2] + 0.5) * scale_y - 0.5
    return target.astype(np.float32)


def crop_and_scale_intrinsics(
    intrinsic_matrix: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    crop: DepthCropConfig,
) -> np.ndarray:
    """Shift K for the native ROI, then resize using pixel-center geometry."""
    top, bottom, left, right = crop.bounds(
        source_width=source_width,
        source_height=source_height,
    )
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("intrinsic_matrix must be a finite 3x3 matrix")
    cropped = intrinsic.copy()
    cropped[0, 2] -= left
    cropped[1, 2] -= top
    return scale_intrinsics_full_fov(
        cropped,
        source_width=right - left,
        source_height=bottom - top,
        target_width=target_width,
        target_height=target_height,
    )


def _area_downsample_axis(
    image: np.ndarray,
    *,
    target_size: int,
    axis: int,
) -> np.ndarray:
    """Integrate piecewise-constant pixels over exact target intervals.

    Prefix integrals avoid both a sparse advanced-indexing tensor and a BLAS
    thread team. This is mathematically the same full-area filter, but keeps
    the onboard 60 Hz producer's post-sync work below one camera period.
    """
    source_size = int(image.shape[axis])
    if source_size <= 0 or not 0 < int(target_size) <= source_size:
        raise ValueError("area-resize dimensions must be positive and downsampling")
    scale = float(source_size) / float(target_size)
    boundaries = np.arange(target_size + 1, dtype=np.float64) * scale
    lower = np.floor(boundaries).astype(np.int64)
    fractions = boundaries - lower

    work = np.asarray(image, dtype=np.float64)
    zero_shape = list(work.shape)
    zero_shape[axis] = 1
    prefix = np.concatenate(
        (
            np.zeros(zero_shape, dtype=np.float64),
            np.cumsum(work, axis=axis, dtype=np.float64),
        ),
        axis=axis,
    )
    integrated = np.take(prefix, lower, axis=axis)
    boundary_pixels = np.take(
        work,
        np.minimum(lower, source_size - 1),
        axis=axis,
    )
    fraction_shape = [1] * work.ndim
    fraction_shape[axis] = fractions.size
    integrated += boundary_pixels * fractions.reshape(fraction_shape)
    return np.diff(integrated, axis=axis) / scale


def resize_full_fov_depth_area(
    depth_m: np.ndarray,
    *,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Area-downsample metric depth, preserving invalid samples as NaN.

    Each output value is the valid-area-weighted mean over the complete source
    image.  No crop, inpainting, spatial filter, or hole filling is applied.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must have shape [height, width]")
    source_height, source_width = depth.shape
    if (source_height, source_width) == (target_height, target_width):
        return depth.copy()
    if target_height > source_height or target_width > source_width:
        raise ValueError("resize_full_fov_depth_area only supports downsampling")

    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > 0.0)
    values = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
    if _cv2 is not None:
        weighted_values = _cv2.resize(
            values,
            (int(target_width), int(target_height)),
            interpolation=_cv2.INTER_AREA,
        )
        valid_weights = _cv2.resize(
            valid.astype(np.float32),
            (int(target_width), int(target_height)),
            interpolation=_cv2.INTER_AREA,
        )
        result = np.full((target_height, target_width), np.nan, dtype=np.float32)
        np.divide(
            weighted_values,
            valid_weights,
            out=result,
            where=valid_weights > 0.0,
        )
        return result

    weighted_values = _area_downsample_axis(
        _area_downsample_axis(values, target_size=target_width, axis=1),
        target_size=target_height,
        axis=0,
    )
    valid_weights = _area_downsample_axis(
        _area_downsample_axis(
            valid.astype(np.float32), target_size=target_width, axis=1
        ),
        target_size=target_height,
        axis=0,
    )
    result = np.full((target_height, target_width), np.nan, dtype=np.float32)
    np.divide(weighted_values, valid_weights, out=result, where=valid_weights > 0.0)
    return result


def crop_and_resize_depth_area(
    depth_m: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    crop: DepthCropConfig,
) -> np.ndarray:
    """Crop the native frame before validity-aware area downsampling."""
    return resize_full_fov_depth_area(
        crop_depth_roi(depth_m, crop),
        target_width=target_width,
        target_height=target_height,
    )


def crop_and_resize_depth_with_conservative_invalid_mask(
    depth_m: np.ndarray,
    invalid_mask: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    crop: DepthCropConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop depth/mask together and invalidate every output cell touched."""
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(invalid_mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth_m and invalid_mask must have matching 2-D shapes")
    cropped_depth = crop_depth_roi(depth, crop)
    cropped_mask = crop_depth_roi(mask, crop)
    resized = resize_full_fov_depth_area(
        cropped_depth,
        target_width=target_width,
        target_height=target_height,
    )
    mask_fraction = resize_full_fov_depth_area(
        np.where(cropped_mask, 1.0, np.nan).astype(np.float32),
        target_width=target_width,
        target_height=target_height,
    )
    resized_mask = np.isfinite(mask_fraction)
    return apply_conservative_output_invalid_mask(resized, resized_mask), resized_mask


def apply_conservative_output_invalid_mask(
    depth_m: np.ndarray,
    invalid_mask: np.ndarray,
) -> np.ndarray:
    """Keep every output cell touched by the semantic self mask invalid."""
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(invalid_mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth_m and invalid_mask must have the same 2-D shape")
    return np.where(mask, np.nan, depth).astype(np.float32, copy=False)
