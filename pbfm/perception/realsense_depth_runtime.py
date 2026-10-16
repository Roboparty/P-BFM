"""RealSense camera calibration, metric depth conversion, and live capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pbfm.perception.depth_preprocessing import resize_full_fov_depth, scale_full_fov_intrinsics


@dataclass(frozen=True)
class RealSenseCalibration:
    """D435i intrinsics and the calibrated optical-camera-to-torso transform.

    ``intrinsic_matrix`` is the native depth-stream matrix in pixel units. The
    target matrix is derived by scaling the complete native image, preserving
    the full field of view. ``optical_quat_torso_xyzw`` maps optical-camera
    vectors (+x right, +y down, +z forward) into the torso frame.
    """

    native_width: int = 1280
    native_height: int = 720
    target_width: int = 64
    target_height: int = 36
    intrinsic_matrix: tuple[float, ...] = (
        640.0,
        0.0,
        640.0,
        0.0,
        640.0,
        360.0,
        0.0,
        0.0,
        1.0,
    )
    depth_scale_m: float = 0.001
    mount_pos_torso: tuple[float, float, float] = (0.0487988662332928, 0.01, 0.4378029937970051)
    optical_quat_torso_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def validate(self) -> None:
        if min(self.native_width, self.native_height, self.target_width, self.target_height) <= 0:
            raise ValueError("RealSense image dimensions must be positive")
        matrix = self.native_intrinsics()
        if not torch.isfinite(matrix).all() or abs(float(torch.linalg.det(matrix))) <= 1.0e-12:
            raise ValueError("intrinsic_matrix must be finite and invertible")
        if self.depth_scale_m <= 0.0:
            raise ValueError("depth_scale_m must be positive")
        quaternion = torch.tensor(self.optical_quat_torso_xyzw, dtype=torch.float64)
        if not torch.isfinite(quaternion).all() or torch.linalg.vector_norm(quaternion) <= 1.0e-8:
            raise ValueError("optical_quat_torso_xyzw must be finite and non-zero")
        if len(self.mount_pos_torso) != 3:
            raise ValueError("mount_pos_torso must have three components")

    def native_intrinsics(self) -> torch.Tensor:
        values = torch.as_tensor(self.intrinsic_matrix, dtype=torch.float64)
        if values.numel() != 9:
            raise ValueError("intrinsic_matrix must contain nine values")
        return values.reshape(3, 3)

    def target_intrinsics(self) -> torch.Tensor:
        self.validate()
        return scale_full_fov_intrinsics(
            self.native_intrinsics(),
            native_height=self.native_height,
            native_width=self.native_width,
            target_height=self.target_height,
            target_width=self.target_width,
        )

    @classmethod
    def from_json(cls, path: Path) -> "RealSenseCalibration":
        import json

        payload = json.loads(path.expanduser().read_text())
        if not isinstance(payload, dict):
            raise ValueError("calibration JSON must contain an object")
        return cls(**payload)


def depth_to_meters(depth: torch.Tensor, *, depth_scale_m: float) -> torch.Tensor:
    """Convert native D435i depth values to meters and preserve invalid as NaN."""
    if depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be positive")
    depth = torch.as_tensor(depth)
    if not depth.is_floating_point():
        depth = depth.to(torch.float32)
    meters = depth * float(depth_scale_m)
    valid = torch.isfinite(meters) & (meters > 0.0)
    return torch.where(valid, meters, torch.full_like(meters, float("nan")))


class RealSenseDepthSource:
    """Optional pyrealsense2 depth source; pose synchronization remains external."""

    def __init__(self, *, width: int = 1280, height: int = 720, fps: int = 30, serial: str | None = None) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("Install pyrealsense2 on the hardware host to use the live D435i source") from error
        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        self.depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())

    def read_depth(self) -> tuple[torch.Tensor, float]:
        frames = self.pipeline.wait_for_frames()
        frame = frames.get_depth_frame()
        if frame is None:
            raise RuntimeError("D435i returned no depth frame")
        import numpy as np

        return torch.from_numpy(np.asanyarray(frame.get_data()).copy()), float(frame.get_timestamp()) / 1000.0

    def close(self) -> None:
        self.pipeline.stop()
