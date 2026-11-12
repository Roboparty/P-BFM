#!/usr/bin/env python3
"""Timestamped sensor and context messages shared by the deployment processes."""

from __future__ import annotations

import numpy as np
import struct

# Same-host sensor channels used by the release runtime.
PORTS = {"low_state": 5590, "terrain_depth": 5592, "direct_depth_latent": 5594}


class TerrainDepthMessage:
    """Synchronized local optical-depth packet for terrain perception.

    Depth is metric optical-Z (+x right, +y down, +z forward).  Projection is
    intentionally odometry-free: the packet contains only exposure-time
    projected gravity, waist joints, and the fixed torso-to-camera mounting.
    No world pose, global yaw, or linear odometry enters the estimator path.
    """

    MAGIC = b"UFOD"
    VERSION = 4
    RESET_HISTORY = 1
    _HEADER = struct.Struct("<4sHHHHQddddd")
    _FIXED_FLOATS = 9 + 3 + 4 + 3 + 3 + 29 + 29 + 3

    def __init__(
        self,
        *,
        depth_z: np.ndarray,
        intrinsic_matrix: np.ndarray,
        camera_pos_torso: np.ndarray,
        camera_optical_quat_torso_xyzw: np.ndarray,
        projected_gravity: np.ndarray,
        waist_joint_pos: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        timestamp_s: float,
        capture_monotonic_s: float,
        camera_receive_monotonic_s: float,
        sync_ready_monotonic_s: float,
        producer_publish_monotonic_s: float,
        frame_number: int,
        reset_history: bool = False,
    ) -> None:
        depth = np.asarray(depth_z, dtype=np.float32)
        if depth.ndim != 2 or min(depth.shape) <= 0:
            raise ValueError("depth_z must be a non-empty [H, W] array")
        if depth.shape[0] > 65535 or depth.shape[1] > 65535:
            raise ValueError("depth_z dimensions exceed the wire format")
        matrix = np.asarray(intrinsic_matrix, dtype=np.float32)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("intrinsic_matrix must be finite with shape [3, 3]")
        if abs(float(np.linalg.det(matrix))) <= 1.0e-12:
            raise ValueError("intrinsic_matrix must be invertible")

        def vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
            result = np.asarray(value, dtype=np.float32).reshape(-1)
            if result.size != size or not np.all(np.isfinite(result)):
                raise ValueError(f"{name} must contain {size} finite values")
            return result.copy()

        if not np.isfinite(timestamp_s) or float(timestamp_s) < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        pipeline_times = {
            "capture_monotonic_s": capture_monotonic_s,
            "camera_receive_monotonic_s": camera_receive_monotonic_s,
            "sync_ready_monotonic_s": sync_ready_monotonic_s,
            "producer_publish_monotonic_s": producer_publish_monotonic_s,
        }
        for name, value in pipeline_times.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(sync_ready_monotonic_s) < float(camera_receive_monotonic_s):
            raise ValueError(
                "sync_ready_monotonic_s cannot precede camera_receive_monotonic_s"
            )
        if float(producer_publish_monotonic_s) < float(sync_ready_monotonic_s):
            raise ValueError(
                "producer_publish_monotonic_s cannot precede sync_ready_monotonic_s"
            )
        if int(frame_number) < 0:
            raise ValueError("frame_number must be non-negative")
        self.depth_z = depth.copy()
        self.intrinsic_matrix = matrix.copy()
        self.camera_pos_torso = vector("camera_pos_torso", camera_pos_torso, 3)
        self.camera_optical_quat_torso_xyzw = vector(
            "camera_optical_quat_torso_xyzw",
            camera_optical_quat_torso_xyzw,
            4,
        )
        quaternion_norm = float(np.linalg.norm(self.camera_optical_quat_torso_xyzw))
        if quaternion_norm <= 1.0e-8:
            raise ValueError("camera_optical_quat_torso_xyzw must have non-zero norm")
        self.camera_optical_quat_torso_xyzw /= quaternion_norm
        self.projected_gravity = vector("projected_gravity", projected_gravity, 3)
        gravity_norm = float(np.linalg.norm(self.projected_gravity))
        if gravity_norm <= 1.0e-8:
            raise ValueError("projected_gravity must have non-zero norm")
        self.projected_gravity /= gravity_norm
        self.waist_joint_pos = vector("waist_joint_pos", waist_joint_pos, 3)
        self.joint_pos = vector("joint_pos", joint_pos, 29)
        self.joint_vel = vector("joint_vel", joint_vel, 29)
        self.base_ang_vel = vector("base_ang_vel", base_ang_vel, 3)
        self.timestamp_s = float(timestamp_s)
        self.capture_monotonic_s = float(capture_monotonic_s)
        self.camera_receive_monotonic_s = float(camera_receive_monotonic_s)
        self.sync_ready_monotonic_s = float(sync_ready_monotonic_s)
        self.producer_publish_monotonic_s = float(producer_publish_monotonic_s)
        self.frame_number = int(frame_number)
        self.reset_history = bool(reset_history)

    def to_bytes(self) -> bytes:
        height, width = self.depth_z.shape
        flags = self.RESET_HISTORY if self.reset_history else 0
        header = self._HEADER.pack(
            self.MAGIC,
            self.VERSION,
            width,
            height,
            flags,
            self.frame_number,
            self.timestamp_s,
            self.capture_monotonic_s,
            self.camera_receive_monotonic_s,
            self.sync_ready_monotonic_s,
            self.producer_publish_monotonic_s,
        )
        fixed = np.concatenate(
            (
                self.intrinsic_matrix.reshape(-1),
                self.camera_pos_torso,
                self.camera_optical_quat_torso_xyzw,
                self.projected_gravity,
                self.waist_joint_pos,
                self.joint_pos,
                self.joint_vel,
                self.base_ang_vel,
            )
        ).astype(np.float32, copy=False)
        return header + fixed.tobytes() + self.depth_z.astype(np.float32, copy=False).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "TerrainDepthMessage":
        if len(data) < cls._HEADER.size:
            raise ValueError("TerrainDepthMessage data is too short")
        (
            magic,
            version,
            width,
            height,
            flags,
            frame_number,
            timestamp_s,
            capture_monotonic_s,
            camera_receive_monotonic_s,
            sync_ready_monotonic_s,
            producer_publish_monotonic_s,
        ) = cls._HEADER.unpack(data[: cls._HEADER.size])
        if magic != cls.MAGIC:
            raise ValueError("TerrainDepthMessage magic mismatch")
        if version != cls.VERSION:
            raise ValueError(f"Unsupported TerrainDepthMessage version: {version}")
        if flags & ~cls.RESET_HISTORY:
            raise ValueError(f"Unsupported TerrainDepthMessage flags: {flags}")
        float_count = cls._FIXED_FLOATS + int(width) * int(height)
        expected = cls._HEADER.size + float_count * np.dtype(np.float32).itemsize
        if len(data) != expected:
            raise ValueError(
                f"TerrainDepthMessage size mismatch: expected {expected} bytes, got {len(data)}"
            )
        values = np.frombuffer(data, dtype=np.float32, offset=cls._HEADER.size)
        cursor = 0
        matrix = values[cursor : cursor + 9].reshape(3, 3)
        cursor += 9
        camera_pos_torso = values[cursor : cursor + 3]
        cursor += 3
        camera_quat_torso = values[cursor : cursor + 4]
        cursor += 4
        projected_gravity = values[cursor : cursor + 3]
        cursor += 3
        waist_joint_pos = values[cursor : cursor + 3]
        cursor += 3
        joint_pos = values[cursor : cursor + 29]
        cursor += 29
        joint_vel = values[cursor : cursor + 29]
        cursor += 29
        base_ang_vel = values[cursor : cursor + 3]
        cursor += 3
        depth = values[cursor:].reshape(height, width)
        return cls(
            depth_z=depth,
            intrinsic_matrix=matrix,
            camera_pos_torso=camera_pos_torso,
            camera_optical_quat_torso_xyzw=camera_quat_torso,
            projected_gravity=projected_gravity,
            waist_joint_pos=waist_joint_pos,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            base_ang_vel=base_ang_vel,
            timestamp_s=timestamp_s,
            capture_monotonic_s=capture_monotonic_s,
            camera_receive_monotonic_s=camera_receive_monotonic_s,
            sync_ready_monotonic_s=sync_ready_monotonic_s,
            producer_publish_monotonic_s=producer_publish_monotonic_s,
            frame_number=frame_number,
            reset_history=bool(flags & cls.RESET_HISTORY),
        )


class DirectDepthLatentMessage:
    """Latest-only 256D depth latent produced by the direct-depth encoder worker."""

    MAGIC = b"UFOL"
    VERSION = 1
    DIMENSION = 256
    _HEADER = struct.Struct("<4sHHQQQdddddddfff")

    def __init__(
        self,
        *,
        depth_latent: np.ndarray,
        source_frame_number: int,
        worker_sequence: int,
        reset_generation: int,
        source_timestamp_s: float,
        capture_monotonic_s: float,
        camera_receive_monotonic_s: float,
        sync_ready_monotonic_s: float,
        producer_publish_monotonic_s: float,
        worker_start_monotonic_s: float,
        completion_monotonic_s: float,
        preprocess_s: float,
        encoder_s: float,
        worker_total_s: float,
    ) -> None:
        latent = np.asarray(depth_latent, dtype=np.float32).reshape(-1)
        if latent.shape != (self.DIMENSION,) or not np.all(np.isfinite(latent)):
            raise ValueError(
                f"depth_latent must contain {self.DIMENSION} finite float32 values"
            )
        integers = {
            "source_frame_number": source_frame_number,
            "worker_sequence": worker_sequence,
            "reset_generation": reset_generation,
        }
        for name, value in integers.items():
            if int(value) < 0 or int(value) > np.iinfo(np.uint64).max:
                raise ValueError(f"{name} must fit in uint64")
        times = {
            "source_timestamp_s": source_timestamp_s,
            "capture_monotonic_s": capture_monotonic_s,
            "camera_receive_monotonic_s": camera_receive_monotonic_s,
            "sync_ready_monotonic_s": sync_ready_monotonic_s,
            "producer_publish_monotonic_s": producer_publish_monotonic_s,
            "worker_start_monotonic_s": worker_start_monotonic_s,
            "completion_monotonic_s": completion_monotonic_s,
            "preprocess_s": preprocess_s,
            "encoder_s": encoder_s,
            "worker_total_s": worker_total_s,
        }
        for name, value in times.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        ordered_times = (
            ("camera_receive_monotonic_s", camera_receive_monotonic_s),
            ("sync_ready_monotonic_s", sync_ready_monotonic_s),
            ("producer_publish_monotonic_s", producer_publish_monotonic_s),
            ("worker_start_monotonic_s", worker_start_monotonic_s),
            ("completion_monotonic_s", completion_monotonic_s),
        )
        for (previous_name, previous), (current_name, current) in zip(
            ordered_times, ordered_times[1:]
        ):
            if float(current) < float(previous):
                raise ValueError(f"{current_name} cannot precede {previous_name}")

        self.depth_latent = latent.copy()
        self.source_frame_number = int(source_frame_number)
        self.worker_sequence = int(worker_sequence)
        self.reset_generation = int(reset_generation)
        self.source_timestamp_s = float(source_timestamp_s)
        self.capture_monotonic_s = float(capture_monotonic_s)
        self.camera_receive_monotonic_s = float(camera_receive_monotonic_s)
        self.sync_ready_monotonic_s = float(sync_ready_monotonic_s)
        self.producer_publish_monotonic_s = float(producer_publish_monotonic_s)
        self.worker_start_monotonic_s = float(worker_start_monotonic_s)
        self.completion_monotonic_s = float(completion_monotonic_s)
        self.preprocess_s = float(preprocess_s)
        self.encoder_s = float(encoder_s)
        self.worker_total_s = float(worker_total_s)

    def to_bytes(self) -> bytes:
        header = self._HEADER.pack(
            self.MAGIC,
            self.VERSION,
            self.DIMENSION,
            self.source_frame_number,
            self.worker_sequence,
            self.reset_generation,
            self.source_timestamp_s,
            self.capture_monotonic_s,
            self.camera_receive_monotonic_s,
            self.sync_ready_monotonic_s,
            self.producer_publish_monotonic_s,
            self.worker_start_monotonic_s,
            self.completion_monotonic_s,
            self.preprocess_s,
            self.encoder_s,
            self.worker_total_s,
        )
        return header + self.depth_latent.astype(np.float32, copy=False).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "DirectDepthLatentMessage":
        expected = cls._HEADER.size + cls.DIMENSION * np.dtype(np.float32).itemsize
        if len(data) != expected:
            raise ValueError(
                f"DirectDepthLatentMessage size mismatch: expected {expected} bytes, got {len(data)}"
            )
        (
            magic,
            version,
            dimension,
            source_frame_number,
            worker_sequence,
            reset_generation,
            source_timestamp_s,
            capture_monotonic_s,
            camera_receive_monotonic_s,
            sync_ready_monotonic_s,
            producer_publish_monotonic_s,
            worker_start_monotonic_s,
            completion_monotonic_s,
            preprocess_s,
            encoder_s,
            worker_total_s,
        ) = cls._HEADER.unpack(data[: cls._HEADER.size])
        if magic != cls.MAGIC:
            raise ValueError("DirectDepthLatentMessage magic mismatch")
        if version != cls.VERSION:
            raise ValueError(f"Unsupported DirectDepthLatentMessage version: {version}")
        if dimension != cls.DIMENSION:
            raise ValueError(
                f"DirectDepthLatentMessage dimension mismatch: {dimension} != {cls.DIMENSION}"
            )
        latent = np.frombuffer(
            data,
            dtype=np.float32,
            count=cls.DIMENSION,
            offset=cls._HEADER.size,
        )
        return cls(
            depth_latent=latent,
            source_frame_number=source_frame_number,
            worker_sequence=worker_sequence,
            reset_generation=reset_generation,
            source_timestamp_s=source_timestamp_s,
            capture_monotonic_s=capture_monotonic_s,
            camera_receive_monotonic_s=camera_receive_monotonic_s,
            sync_ready_monotonic_s=sync_ready_monotonic_s,
            producer_publish_monotonic_s=producer_publish_monotonic_s,
            worker_start_monotonic_s=worker_start_monotonic_s,
            completion_monotonic_s=completion_monotonic_s,
            preprocess_s=preprocess_s,
            encoder_s=encoder_s,
            worker_total_s=worker_total_s,
        )


class RealtimeContextMessage:
    """Latest-only realtime behavior context for teleoperation.

    Monotonic timestamps are intentionally part of the wire contract.  The
    supported onboard flow runs the realtime context encoder and policy on the
    same host, so the policy can reject a packet whose source pose or encoded
    result is already stale instead of refreshing freshness merely because an
    old packet has just arrived.

    ``heading_relative_xy`` is the live reference heading relative to the
    teleop heading anchor.  The policy applies it to the robot heading latched
    when tracking starts, matching packaged tracking-context semantics.
    """

    MAGIC = b"UFOZ"
    VERSION = 1
    Z_DIMENSION = 256
    HEADING_DIMENSION = 2
    HEADING_VALID = 1 << 0
    REFERENCE_LIVE = 1 << 1
    _KNOWN_FLAGS = HEADING_VALID | REFERENCE_LIVE
    _HEADER = struct.Struct("<4sHHQQdddI")

    def __init__(
        self,
        *,
        z: np.ndarray,
        heading_relative_xy: np.ndarray,
        stream_id: int,
        sequence: int,
        source_receive_monotonic_s: float,
        encoder_completion_monotonic_s: float,
        publish_monotonic_s: float,
        heading_valid: bool,
        reference_live: bool,
    ) -> None:
        latent = np.asarray(z, dtype=np.float32).reshape(-1)
        if latent.shape != (self.Z_DIMENSION,) or not np.all(np.isfinite(latent)):
            raise ValueError(
                f"z must contain {self.Z_DIMENSION} finite float32 values"
            )
        heading = np.asarray(heading_relative_xy, dtype=np.float32).reshape(-1)
        if heading.shape != (self.HEADING_DIMENSION,) or not np.all(np.isfinite(heading)):
            raise ValueError(
                f"heading_relative_xy must contain {self.HEADING_DIMENSION} finite values"
            )
        if bool(heading_valid):
            heading_norm = float(np.linalg.norm(heading))
            if not np.isfinite(heading_norm) or heading_norm <= 1.0e-6:
                raise ValueError("valid heading_relative_xy must have non-zero norm")
            heading = (heading / heading_norm).astype(np.float32)
        else:
            heading = np.zeros(self.HEADING_DIMENSION, dtype=np.float32)

        for name, value in {"stream_id": stream_id, "sequence": sequence}.items():
            if int(value) < 0 or int(value) > np.iinfo(np.uint64).max:
                raise ValueError(f"{name} must fit in uint64")

        times = {
            "source_receive_monotonic_s": source_receive_monotonic_s,
            "encoder_completion_monotonic_s": encoder_completion_monotonic_s,
            "publish_monotonic_s": publish_monotonic_s,
        }
        for name, value in times.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(encoder_completion_monotonic_s) < float(source_receive_monotonic_s):
            raise ValueError(
                "encoder_completion_monotonic_s cannot precede source_receive_monotonic_s"
            )
        if float(publish_monotonic_s) < float(encoder_completion_monotonic_s):
            raise ValueError(
                "publish_monotonic_s cannot precede encoder_completion_monotonic_s"
            )

        self.z = latent.copy()
        self.heading_relative_xy = heading.copy()
        self.stream_id = int(stream_id)
        self.sequence = int(sequence)
        self.source_receive_monotonic_s = float(source_receive_monotonic_s)
        self.encoder_completion_monotonic_s = float(encoder_completion_monotonic_s)
        self.publish_monotonic_s = float(publish_monotonic_s)
        self.heading_valid = bool(heading_valid)
        self.reference_live = bool(reference_live)

    def to_bytes(self) -> bytes:
        flags = 0
        if self.heading_valid:
            flags |= self.HEADING_VALID
        if self.reference_live:
            flags |= self.REFERENCE_LIVE
        header = self._HEADER.pack(
            self.MAGIC,
            self.VERSION,
            self.Z_DIMENSION,
            self.stream_id,
            self.sequence,
            self.source_receive_monotonic_s,
            self.encoder_completion_monotonic_s,
            self.publish_monotonic_s,
            flags,
        )
        payload = np.concatenate((self.z, self.heading_relative_xy)).astype(
            np.float32, copy=False
        )
        return header + payload.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "RealtimeContextMessage":
        expected = cls._HEADER.size + (
            cls.Z_DIMENSION + cls.HEADING_DIMENSION
        ) * np.dtype(np.float32).itemsize
        if len(data) != expected:
            raise ValueError(
                f"RealtimeContextMessage size mismatch: expected {expected} bytes, got {len(data)}"
            )
        (
            magic,
            version,
            z_dimension,
            stream_id,
            sequence,
            source_receive_monotonic_s,
            encoder_completion_monotonic_s,
            publish_monotonic_s,
            flags,
        ) = cls._HEADER.unpack(data[: cls._HEADER.size])
        if magic != cls.MAGIC:
            raise ValueError("RealtimeContextMessage magic mismatch")
        if version != cls.VERSION:
            raise ValueError(f"Unsupported RealtimeContextMessage version: {version}")
        if z_dimension != cls.Z_DIMENSION:
            raise ValueError(
                f"RealtimeContextMessage z dimension mismatch: {z_dimension} != {cls.Z_DIMENSION}"
            )
        if flags & ~cls._KNOWN_FLAGS:
            raise ValueError(f"Unsupported RealtimeContextMessage flags: {flags}")
        values = np.frombuffer(data, dtype=np.float32, offset=cls._HEADER.size)
        return cls(
            z=values[: cls.Z_DIMENSION],
            heading_relative_xy=values[cls.Z_DIMENSION :],
            stream_id=stream_id,
            sequence=sequence,
            source_receive_monotonic_s=source_receive_monotonic_s,
            encoder_completion_monotonic_s=encoder_completion_monotonic_s,
            publish_monotonic_s=publish_monotonic_s,
            heading_valid=bool(flags & cls.HEADING_VALID),
            reference_live=bool(flags & cls.REFERENCE_LIVE),
        )

class LowStateMessage:
    """Binary message containing base and joint state information."""

    MAGIC = b"UFOS"
    VERSION = 1
    HAS_TORQUE = 1
    _HEADER = struct.Struct("<4sHHHQd")

    def __init__(
        self,
        quaternion: np.ndarray,
        gyroscope: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray | None = None,
        tick: int = 0,
        capture_monotonic_s: float | None = None,
    ):
        self.quaternion = np.asarray(quaternion, dtype=np.float32)
        if self.quaternion.size != 4:
            raise ValueError("Quaternion must have exactly 4 elements")

        self.gyroscope = np.asarray(gyroscope, dtype=np.float32)
        if self.gyroscope.size != 3:
            raise ValueError("Gyroscope must have exactly 3 elements")

        self.joint_positions = np.asarray(joint_positions, dtype=np.float32)
        self.joint_velocities = np.asarray(joint_velocities, dtype=np.float32)
        if self.joint_positions.size != self.joint_velocities.size:
            raise ValueError("Joint position and velocity arrays must match in length")

        if joint_torques is not None:
            joint_torques = np.asarray(joint_torques, dtype=np.float32)
            if joint_torques.size != self.joint_positions.size:
                raise ValueError("Joint torque array must match joint positions length")
        self.joint_torques = joint_torques
        self.tick = int(tick)
        if self.tick < 0:
            raise ValueError("Low-state tick must be non-negative")
        if capture_monotonic_s is None:
            raise ValueError("capture_monotonic_s is required")
        if not np.isfinite(capture_monotonic_s) or float(capture_monotonic_s) < 0.0:
            raise ValueError("capture_monotonic_s must be finite and non-negative")
        self.capture_monotonic_s = float(capture_monotonic_s)

    def to_bytes(self) -> bytes:
        count = self.joint_positions.size
        if count > 65535:
            raise ValueError("Low-state joint count exceeds the wire format")
        flags = self.HAS_TORQUE if self.joint_torques is not None else 0
        header = self._HEADER.pack(
            self.MAGIC,
            self.VERSION,
            count,
            flags,
            self.tick,
            self.capture_monotonic_s,
        )

        payload_parts = [
            self.quaternion.astype(np.float32, copy=False).tobytes(),
            self.gyroscope.astype(np.float32, copy=False).tobytes(),
            self.joint_positions.astype(np.float32, copy=False).tobytes(),
            self.joint_velocities.astype(np.float32, copy=False).tobytes(),
        ]

        if self.joint_torques is not None:
            payload_parts.append(self.joint_torques.astype(np.float32, copy=False).tobytes())

        return header + b''.join(payload_parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'LowStateMessage':
        header_size = cls._HEADER.size
        if len(data) < header_size + 28:  # header + quaternion(16) + gyro(12)
            raise ValueError("LowStateMessage data is too short")

        magic, version, count, flags, tick, capture_monotonic_s = cls._HEADER.unpack(
            data[:header_size]
        )
        if magic != cls.MAGIC:
            raise ValueError("LowStateMessage magic mismatch")
        if version != cls.VERSION:
            raise ValueError(f"Unsupported LowStateMessage version: {version}")
        if flags & ~cls.HAS_TORQUE:
            raise ValueError(f"Unsupported LowStateMessage flags: {flags}")
        offset = header_size

        quat_end = offset + 16
        gyro_end = quat_end + 12
        quaternion = np.frombuffer(data[offset:quat_end], dtype=np.float32).copy()
        gyroscope = np.frombuffer(data[quat_end:gyro_end], dtype=np.float32).copy()

        segment_size = count * 4
        pos_end = gyro_end + segment_size
        vel_end = pos_end + segment_size
        expected = vel_end + (segment_size if flags & cls.HAS_TORQUE else 0)
        if len(data) != expected:
            raise ValueError(
                f"LowStateMessage size mismatch: expected {expected} bytes, got {len(data)}"
            )

        joint_positions = np.frombuffer(data[gyro_end:pos_end], dtype=np.float32).copy()
        joint_velocities = np.frombuffer(data[pos_end:vel_end], dtype=np.float32).copy()

        joint_torques = None
        if flags & cls.HAS_TORQUE:
            torque_end = vel_end + segment_size
            if torque_end > len(data):
                raise ValueError("LowStateMessage torque data is incomplete")
            joint_torques = np.frombuffer(data[vel_end:torque_end], dtype=np.float32).copy()

        return cls(
            quaternion=quaternion,
            gyroscope=gyroscope,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            tick=tick,
            capture_monotonic_s=capture_monotonic_s,
        )
