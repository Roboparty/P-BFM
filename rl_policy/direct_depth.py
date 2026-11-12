"""Exact runtime contract for the RP1-style direct-depth policies.

Producer packets always retain conventional optical row order.  Image-row
orientation is an explicit per-policy contract: the legacy step182403072
policy expects one vertical flip, while optical-topdown policies exported from
the corrected training pipeline do not.  The configured transform is applied
exactly once here, before crop/blur and history insertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DirectDepthConfig:
    processing_profile: str = "legacy_rp1"
    crop_top: int = 0
    crop_bottom: int = 0
    source_height: int = 36
    source_width: int = 64
    crop_left: int = 16
    crop_right: int = 16
    min_depth_m: float = 0.0
    max_depth_m: float = 2.5
    history_length: int = 37
    history_skip_frames: int = 5
    output_frames: int = 8
    fixed_delay_frames: int = 0
    vertical_flip: bool = True
    gaussian_enabled: bool = True

    @classmethod
    def from_policy_config(cls, policy_config: Mapping[str, Any]) -> "DirectDepthConfig":
        """Construct and validate the preprocessing contract from policy YAML."""

        raw = policy_config.get("direct_depth_preprocess")
        if raw is None:
            raise ValueError(
                "direct_depth_preprocess is required; vertical orientation must be explicit"
            )
        if not isinstance(raw, Mapping):
            raise TypeError("direct_depth_preprocess must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown direct_depth_preprocess keys: {unknown}")
        config = cls(**dict(raw))
        config.validate()
        return config

    @property
    def output_height(self) -> int:
        return self.source_height - self.crop_top - self.crop_bottom

    @property
    def output_width(self) -> int:
        return self.source_width - self.crop_left - self.crop_right

    @property
    def sampled_ages(self) -> tuple[int, ...]:
        return tuple(
            reversed(
                range(
                    0,
                    self.output_frames * self.history_skip_frames,
                    self.history_skip_frames,
                )
            )
        )

    def validate(self) -> None:
        if not isinstance(self.gaussian_enabled, bool):
            raise TypeError("gaussian_enabled must be a boolean")
        if not isinstance(self.vertical_flip, bool):
            raise TypeError("direct-depth vertical_flip must be a boolean")
        if (self.source_height, self.source_width) != (36, 64):
            raise ValueError("direct depth requires a 36x64 source frame")
        if self.processing_profile not in {"legacy_rp1", "optical_z_maxcrop18_reset_v2"}:
            raise ValueError("Unknown depth processing profile")
        expected_crop = (18, 0, 16, 16) if self.processing_profile == "optical_z_maxcrop18_reset_v2" else (0, 0, 16, 16)
        if (self.crop_top, self.crop_bottom, self.crop_left, self.crop_right) != expected_crop:
            raise ValueError(f"Depth profile requires crop {expected_crop}")
        if self.processing_profile != "legacy_rp1" and self.vertical_flip:
            raise ValueError("optical_z_maxcrop18_reset_v2 requires no vertical flip")
        if self.history_length != 37:
            raise ValueError("direct depth requires a 37-tick history")
        if self.sampled_ages != (35, 30, 25, 20, 15, 10, 5, 0):
            raise ValueError("direct depth requires the frozen skip-5 sampling ages")
        if self.fixed_delay_frames not in (0, 1):
            raise ValueError("fixed direct-depth delay must be 0 or 1 tick")
        if not np.isfinite(self.min_depth_m) or self.min_depth_m != 0.0:
            raise ValueError("direct-depth minimum range is frozen at 0.0 m")
        if not np.isfinite(self.max_depth_m) or self.max_depth_m != 2.5:
            raise ValueError("direct-depth max range is frozen at 2.5 m")


def preprocess_direct_depth_frame(
    depth_m: np.ndarray,
    config: DirectDepthConfig = DirectDepthConfig(),
) -> np.ndarray:
    """Return one quantized normalized ``[36, 32]`` policy frame.

    Match the selected policy's configured input orientation: sanitize invalid
    values and samples outside [0.0, 2.5] m, optionally flip image rows,
    horizontally crop, apply the fixed 3x3 Gaussian blur, clamp, normalize and
    quantize.  The source packet is never modified in place.
    """

    config.validate()
    depth = np.asarray(depth_m, dtype=np.float32)
    expected = (config.source_height, config.source_width)
    if depth.shape != expected:
        raise ValueError(f"direct depth frame must have shape {expected}, got {depth.shape}")
    if config.processing_profile == "optical_z_maxcrop18_reset_v2":
        depth = np.where(np.isfinite(depth) & (depth >= config.min_depth_m), depth, config.max_depth_m)
        depth = np.clip(depth, config.min_depth_m, config.max_depth_m).astype(np.float32, copy=False)
        depth = depth[config.crop_top:config.source_height-config.crop_bottom,
                      config.crop_left:config.source_width-config.crop_right]
        if not config.gaussian_enabled:
            return np.rint(depth / config.max_depth_m * 255.0).astype(np.uint8)
        gaussian = np.exp(-np.arange(-1, 2, dtype=np.float32) ** 2 / np.float32(2.0))
        gaussian /= gaussian.sum()
        kernel = gaussian[:, None] * gaussian[None, :]
        padded = np.pad(depth, ((1, 1), (1, 1)), mode="reflect")
        blurred = sum(
            float(kernel[row, col]) * padded[row:row + depth.shape[0], col:col + depth.shape[1]]
            for row in range(3) for col in range(3)
        )
        normalized = np.clip(blurred, config.min_depth_m, config.max_depth_m) / config.max_depth_m
        return np.rint(normalized * 255.0).astype(np.uint8)
    valid = (
        np.isfinite(depth)
        & (depth >= config.min_depth_m)
        & (depth <= config.max_depth_m)
    )
    depth = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
    if config.vertical_flip:
        depth = depth[::-1]
    depth = depth[:, config.crop_left : config.source_width - config.crop_right]

    if not config.gaussian_enabled:
        return np.rint(depth / config.max_depth_m * 255.0).astype(np.uint8)

    # Fixed separable [1, 2, 1] x [1, 2, 1] / 16 kernel with reflect padding.
    padded = np.pad(depth, ((1, 1), (1, 1)), mode="reflect")
    blurred = (
        padded[:-2, :-2]
        + 2.0 * padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + 2.0 * padded[1:-1, :-2]
        + 4.0 * padded[1:-1, 1:-1]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    ) * (1.0 / 16.0)
    normalized = np.clip(blurred, 0.0, config.max_depth_m) / config.max_depth_m
    return np.rint(normalized * 255.0).astype(np.uint8)


class DirectDepthHistory:
    """Maintain the trained 37-tick ring and expose 8 oldest-to-newest frames."""

    def __init__(self, config: DirectDepthConfig = DirectDepthConfig()) -> None:
        config.validate()
        self.config = config
        self._history = np.zeros(
            (config.history_length, config.output_height, config.output_width),
            dtype=np.uint8,
        )
        self._write_index = 0
        self._initialized = False

    def reset(self) -> None:
        self._history.fill(0)
        self._write_index = 0
        self._initialized = False

    def append(self, depth_m: np.ndarray, *, reset: bool = False) -> np.ndarray:
        frame = preprocess_direct_depth_frame(depth_m, self.config)
        if reset or not self._initialized:
            self._history[:] = frame[None, ...]
            self._write_index = 0
            self._initialized = True
        current_index = self._write_index
        self._history[current_index] = frame
        indices = [
            (current_index - age - self.config.fixed_delay_frames)
            % self.config.history_length
            for age in self.config.sampled_ages
        ]
        output = self._history[np.asarray(indices, dtype=np.int64)].copy()
        self._write_index = (current_index + 1) % self.config.history_length
        return output


__all__ = [
    "DirectDepthConfig",
    "DirectDepthHistory",
    "preprocess_direct_depth_frame",
]
