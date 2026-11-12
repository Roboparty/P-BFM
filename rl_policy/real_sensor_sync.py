"""Sensor clock mapping and pinhole-camera validation."""

from __future__ import annotations

from collections import deque

import numpy as np


class AffineClockMapper:
    """Map a source clock into host monotonic time using recent arrival pairs."""

    def __init__(
        self,
        max_samples: int = 256,
        max_rate_error: float = 0.02,
        arrival_offset_quantile: float = 0.05,
    ) -> None:
        if (
            max_samples < 2
            or not 0.0 < max_rate_error < 1.0
            or not 0.0 <= arrival_offset_quantile <= 0.5
        ):
            raise ValueError("invalid clock-mapper configuration")
        self.samples: deque[tuple[float, float]] = deque(maxlen=int(max_samples))
        self.max_rate_error = float(max_rate_error)
        self.arrival_offset_quantile = float(arrival_offset_quantile)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def add(self, source_s: float, host_monotonic_s: float) -> float:
        source_s = float(source_s)
        host_monotonic_s = float(host_monotonic_s)
        if not np.isfinite(source_s) or not np.isfinite(host_monotonic_s):
            raise ValueError("clock samples must be finite")
        if self.samples and source_s <= self.samples[-1][0]:
            self.samples.clear()
        self.samples.append((source_s, host_monotonic_s))
        return self.map(source_s)

    def map(self, source_s: float) -> float:
        if not self.samples:
            raise RuntimeError("clock mapper has no samples")
        source_s = float(source_s)
        source = np.asarray([item[0] for item in self.samples], dtype=np.float64)
        host = np.asarray([item[1] for item in self.samples], dtype=np.float64)
        if source.size < 3 or float(np.ptp(source)) <= 1.0e-9:
            offset = float(np.quantile(host - source, self.arrival_offset_quantile))
            return source_s + offset
        source_center = float(np.mean(source))
        host_center = float(np.mean(host))
        centered = source - source_center
        slope = float(np.dot(centered, host - host_center) / np.dot(centered, centered))
        if abs(slope - 1.0) > self.max_rate_error:
            slope = 1.0
        intercept_samples = host - slope * source
        intercept = float(np.quantile(intercept_samples, self.arrival_offset_quantile))
        return slope * source_s + intercept


def require_pinhole_depth(distortion_name: str, coefficients: np.ndarray) -> None:
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("RealSense distortion coefficients are non-finite")
    if np.max(np.abs(coefficients), initial=0.0) > 1.0e-8:
        raise RuntimeError(
            f"active RealSense depth profile has nonzero {distortion_name} distortion; "
            "rectify it before publishing pinhole K"
        )
