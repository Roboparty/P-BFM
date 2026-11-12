from __future__ import annotations

import numpy as np
import zmq
import time


from utils.strings import unitree_joint_names
from loguru import logger
from utils.common import (
    PORTS,
    LowStateMessage,
    TerrainDepthMessage,
    DirectDepthLatentMessage,
)

def validate_state_source(robot_config):
    if (robot_config.get("ROBOT_TYPE") != "g1_real"
            or robot_config.get("LOW_STATE_SOURCE", "zmq") != "zmq"):
        raise ValueError("Release requires ROBOT_TYPE=g1_real and LOW_STATE_SOURCE=zmq")


class StateProcessor:
    """Validate timestamped ZMQ sensor packets and map joints to policy order."""
    def __init__(
        self,
        robot_config,
        dest_joint_names,
        *,
        subscribe_terrain_depth: bool = True,
        subscribe_direct_depth_latent: bool = False,
    ):
        validate_state_source(robot_config)
        self.low_state_receive_timeout_s = self._positive_timeout_s(
            robot_config.get("LOW_STATE_RECEIVE_TIMEOUT_MS", 100.0),
            "LOW_STATE_RECEIVE_TIMEOUT_MS",
        )
        self.low_state_capture_timeout_s = self._positive_timeout_s(
            robot_config.get("LOW_STATE_CAPTURE_TIMEOUT_MS", 100.0),
            "LOW_STATE_CAPTURE_TIMEOUT_MS",
        )
        self.capture_future_tolerance_s = self._positive_timeout_s(
            robot_config.get("CAPTURE_FUTURE_TOLERANCE_MS", 10.0),
            "CAPTURE_FUTURE_TOLERANCE_MS",
        )
        self.latest_low_state: LowStateMessage | None = None
        self.latest_low_state_receive_monotonic: float | None = None
        self.last_low_state_error = "no low-state packet received"

        self.low_state_port = robot_config.get(
            "LOW_STATE_PORT", PORTS["low_state"]
        )
        state_host = robot_config.get("LOW_STATE_HOST", "127.0.0.1")
        state_endpoint = f"tcp://{state_host}:{self.low_state_port}"

        self.zmq_context = zmq.Context.instance()
        self.low_state_socket: zmq.Socket = self.zmq_context.socket(zmq.SUB)
        self.low_state_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.low_state_socket.setsockopt(zmq.CONFLATE, 1)
        self.low_state_socket.setsockopt(zmq.RCVTIMEO, 10)
        self.low_state_socket.setsockopt(zmq.LINGER, 0)
        self.low_state_socket.connect(state_endpoint)
        logger.info(f"Low-state subscriber connected to {state_endpoint}")

        terrain_depth_port = robot_config.get("TERRAIN_DEPTH_PORT")
        if subscribe_terrain_depth and terrain_depth_port is not None:
            terrain_depth_host = robot_config.get("TERRAIN_DEPTH_HOST", "127.0.0.1")
            terrain_depth_endpoint = f"tcp://{terrain_depth_host}:{int(terrain_depth_port)}"
            self.terrain_depth_socket: zmq.Socket = self.zmq_context.socket(zmq.SUB)
            self.terrain_depth_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.terrain_depth_socket.setsockopt(zmq.CONFLATE, 1)
            self.terrain_depth_socket.setsockopt(zmq.RCVTIMEO, 0)
            self.terrain_depth_socket.setsockopt(zmq.LINGER, 0)
            self.terrain_depth_socket.connect(terrain_depth_endpoint)
            self.latest_terrain_depth: TerrainDepthMessage | None = None
            self.latest_terrain_depth_monotonic: float | None = None
            logger.info(f"Terrain depth subscriber connected to {terrain_depth_endpoint}")

        latent_port = robot_config.get("DIRECT_DEPTH_LATENT_PORT")
        if subscribe_direct_depth_latent and latent_port is not None:
            latent_host = robot_config.get("DIRECT_DEPTH_LATENT_HOST", "127.0.0.1")
            latent_endpoint = f"tcp://{latent_host}:{int(latent_port)}"
            self.direct_depth_latent_socket: zmq.Socket = self.zmq_context.socket(zmq.SUB)
            self.direct_depth_latent_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.direct_depth_latent_socket.setsockopt(zmq.CONFLATE, 1)
            self.direct_depth_latent_socket.setsockopt(zmq.RCVTIMEO, 0)
            self.direct_depth_latent_socket.setsockopt(zmq.LINGER, 0)
            self.direct_depth_latent_socket.connect(latent_endpoint)
            self.latest_direct_depth_latent: DirectDepthLatentMessage | None = None
            self.latest_direct_depth_latent_monotonic: float | None = None
            self.latest_direct_depth_latent_order: tuple[int, int] | None = None
            logger.info(f"Direct-depth latent subscriber connected to {latent_endpoint}")

        # Initialize joint mapping
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [unitree_joint_names.index(name) for name in dest_joint_names]
        self.joint_names = dest_joint_names

        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)

        # create views of qpos and qvel
        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]

        self.root_quat_b = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]

        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]

    @staticmethod
    def _positive_timeout_s(value_ms, name: str) -> float:
        value_s = float(value_ms) / 1000.0
        if not np.isfinite(value_s) or value_s <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value_s

    def _freshness_error(
        self,
        *,
        stream_name: str,
        capture_monotonic_s: float,
        receive_monotonic_s: float,
        max_capture_age_s: float,
        max_receive_age_s: float,
        now: float,
    ) -> str | None:
        receive_age_s = now - float(receive_monotonic_s)
        capture_age_s = now - float(capture_monotonic_s)
        if receive_age_s < -self.capture_future_tolerance_s:
            return f"{stream_name} receive timestamp is in the future"
        if capture_age_s < -self.capture_future_tolerance_s:
            return f"{stream_name} capture timestamp is in the future"
        if receive_age_s > max_receive_age_s:
            return f"{stream_name} receive stream is stale ({receive_age_s * 1000.0:.1f} ms)"
        if capture_age_s > max_capture_age_s:
            return f"{stream_name} capture is stale ({capture_age_s * 1000.0:.1f} ms)"
        return None

    def _prepare_low_state(self):
        self._receive_low_state()
        if not self.latest_low_state:
            self.last_low_state_error = "no low-state packet received"
            return False

        low_state = self.latest_low_state
        received = self.latest_low_state_receive_monotonic
        if received is None:
            self.last_low_state_error = "low-state receive timestamp is unavailable"
            return False
        freshness_error = self._freshness_error(
            stream_name="low-state",
            capture_monotonic_s=low_state.capture_monotonic_s,
            receive_monotonic_s=received,
            max_capture_age_s=self.low_state_capture_timeout_s,
            max_receive_age_s=self.low_state_receive_timeout_s,
            now=time.monotonic(),
        )
        if freshness_error is not None:
            self.last_low_state_error = freshness_error
            return False
        self.root_quat_b[:] = low_state.quaternion
        self.root_ang_vel_b[:] = low_state.gyroscope

        source_joint_pos = low_state.joint_positions
        source_joint_vel = low_state.joint_velocities
        if (
            source_joint_pos.size <= max(self.joint_indices_in_source)
            or source_joint_vel.size <= max(self.joint_indices_in_source)
            or not np.all(np.isfinite(source_joint_pos))
            or not np.all(np.isfinite(source_joint_vel))
            or not np.all(np.isfinite(low_state.quaternion))
            or not np.all(np.isfinite(low_state.gyroscope))
        ):
            self.last_low_state_error = "low-state contains invalid joint or IMU values"
            return False
        for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
            self.joint_pos[dst_idx] = source_joint_pos[src_idx]
            self.joint_vel[dst_idx] = source_joint_vel[src_idx]

        self.last_low_state_error = ""
        return True

    def _receive_low_state(self):
        """Fetch the most recent low state message from the ZMQ socket."""
        if not hasattr(self, "low_state_socket"):
            return

        while True:
            try:
                data = self.low_state_socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            try:
                self.latest_low_state = LowStateMessage.from_bytes(data)
                self.latest_low_state_receive_monotonic = time.monotonic()
            except Exception as exc:
                logger.warning(f"Failed to decode low state message: {exc}")

    def _receive_terrain_depth(self) -> None:
        """Fetch the newest synchronized depth packet without blocking."""
        if not hasattr(self, "terrain_depth_socket"):
            return

        while True:
            try:
                data = self.terrain_depth_socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            try:
                self.latest_terrain_depth = TerrainDepthMessage.from_bytes(data)
                self.latest_terrain_depth_monotonic = time.monotonic()
            except Exception as exc:
                logger.warning(f"Failed to decode terrain depth message: {exc}")

    def get_latest_terrain_depth(
        self,
        *,
        max_receive_age_s: float,
        max_capture_age_s: float,
    ) -> TerrainDepthMessage:
        """Return a fresh terrain-depth packet or fail closed."""
        if max_receive_age_s <= 0.0 or max_capture_age_s <= 0.0:
            raise ValueError("terrain freshness limits must be positive")
        if not hasattr(self, "terrain_depth_socket"):
            raise RuntimeError("terrain depth subscriber is not configured")
        self._receive_terrain_depth()
        frame = self.latest_terrain_depth
        received = self.latest_terrain_depth_monotonic
        if frame is None or received is None:
            raise RuntimeError("no terrain depth frame received")
        freshness_error = self._freshness_error(
            stream_name="terrain depth",
            capture_monotonic_s=frame.capture_monotonic_s,
            receive_monotonic_s=received,
            max_capture_age_s=max_capture_age_s,
            max_receive_age_s=max_receive_age_s,
            now=time.monotonic(),
        )
        if freshness_error is not None:
            raise RuntimeError(freshness_error)
        return frame



    def _receive_direct_depth_latent(self) -> None:
        """Fetch the newest direct-depth encoder result without blocking."""
        if not hasattr(self, "direct_depth_latent_socket"):
            return

        while True:
            try:
                data = self.direct_depth_latent_socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            try:
                result = DirectDepthLatentMessage.from_bytes(data)
                order = (result.reset_generation, result.worker_sequence)
                previous = self.latest_direct_depth_latent_order
                if previous is not None and order < previous:
                    logger.warning(
                        "Ignoring out-of-order direct-depth latent result "
                        f"generation/sequence={order} < {previous}"
                    )
                    continue
                self.latest_direct_depth_latent = result
                self.latest_direct_depth_latent_monotonic = time.monotonic()
                self.latest_direct_depth_latent_order = order
            except Exception as exc:
                logger.warning(f"Failed to decode direct-depth latent message: {exc}")

    def get_latest_direct_depth_latent(
        self,
        *,
        max_receive_age_s: float,
        max_capture_age_s: float,
        max_result_age_s: float,
    ) -> DirectDepthLatentMessage:
        """Return a fresh completed direct-depth latent or fail closed."""
        if (
            max_receive_age_s <= 0.0
            or max_capture_age_s <= 0.0
            or max_result_age_s <= 0.0
        ):
            raise ValueError("direct-depth latent freshness limits must be positive")
        if not hasattr(self, "direct_depth_latent_socket"):
            raise RuntimeError("direct-depth latent subscriber is not configured")
        self._receive_direct_depth_latent()
        result = self.latest_direct_depth_latent
        received = self.latest_direct_depth_latent_monotonic
        if result is None or received is None:
            raise RuntimeError("no direct-depth latent received")
        now = time.monotonic()
        freshness_error = self._freshness_error(
            stream_name="direct-depth latent",
            capture_monotonic_s=result.capture_monotonic_s,
            receive_monotonic_s=received,
            max_capture_age_s=max_capture_age_s,
            max_receive_age_s=max_receive_age_s,
            now=now,
        )
        if freshness_error is not None:
            raise RuntimeError(freshness_error)
        result_age_s = now - float(result.completion_monotonic_s)
        if result_age_s < -self.capture_future_tolerance_s:
            raise RuntimeError("direct-depth latent completion timestamp is in the future")
        if result_age_s > max_result_age_s:
            raise RuntimeError(
                f"direct-depth latent completion is stale ({result_age_s * 1000.0:.1f} ms)"
            )
        return result
