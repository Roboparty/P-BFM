#!/usr/bin/env python3
"""No-actuation D435i + G1 state producer for the direct-depth policy.

By default this process owns only RealSense; g1_state_publisher.py publishes
low-state independently. The supervisor restarts only this camera process on
USB failures. Explicit sdk2/ros2 backends retain the legacy combined mode.
The policy consumes the optical-Z image itself: there is no terrain
projection, self-occlusion removal, odometry or TemporalCompletion estimator.
It never imports g1_interface and never creates a motor-command publisher.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.training_depth_rays import TrainingDepthSampler

from utils.depth_preprocess import (  # noqa: E402
    DepthCropConfig,
    crop_and_resize_depth_area,
    crop_and_scale_intrinsics,
    gate_metric_depth,
)

import struct
import zmq  # noqa: E402

from rl_policy.real_sensor_sync import (  # noqa: E402
    AffineClockMapper,
    require_pinhole_depth,
)
from utils.common import LowStateMessage, TerrainDepthMessage  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402
REALSENSE_STALL_EXIT_CODE = 75


@dataclass(frozen=True)
class CapturedDepthFrame:
    """One RealSense depth frame with co-sampled host receive clocks."""

    depth_frame: Any
    receive_monotonic_s: float
    receive_realtime_s: float
    acquisition_sequence: int
    capture_monotonic_s: float
    frame_number: int
    reset_epoch: int


class LatestRealSenseCapture:
    """Continuously acquire RealSense frames into a bounded latest window.

    RealSense exposure must not share a serial critical path with state
    interpolation, FK, resizing, and serialization. The capture thread owns
    ``wait_for_frames`` while the producer thread atomically selects only the
    newest frame whose exposure is covered by both state buffers. The small
    window exists only for future-sample bracketing; obsolete frames are
    released instead of forming a processing FIFO.
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        wait_timeout_ms: int = 1000,
        buffer_capacity: int = 8,
        clock_warmup_frames: int = 8,
        monotonic_clock=time.monotonic,
        realtime_clock=time.time,
    ) -> None:
        if min(int(wait_timeout_ms), int(buffer_capacity), int(clock_warmup_frames)) <= 0:
            raise ValueError(
                "RealSense wait timeout, buffer capacity, and clock warmup must be positive"
            )
        self.pipeline = pipeline
        self.wait_timeout_ms = int(wait_timeout_ms)
        self.buffer_capacity = int(buffer_capacity)
        self.clock_warmup_frames = int(clock_warmup_frames)
        self.monotonic_clock = monotonic_clock
        self.realtime_clock = realtime_clock
        self._condition = threading.Condition()
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._frames: deque[CapturedDepthFrame] = deque()
        self._sequence = 0
        self._clock_mapper = AffineClockMapper()
        self._timestamp_domain: str | None = None
        self._previous_device_timestamp_s: float | None = None
        self._previous_frame_number: int | None = None
        self._reset_epoch = 1
        self._clock_sample_count = 0
        self.acquired_frames = 0
        self.overwritten_frames = 0
        self.selection_skips = 0
        self.capture_timeouts = 0
        self.last_error = ""

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("RealSense capture thread is already started")
            self._stop_requested = False
            self._thread = threading.Thread(
                target=self._run,
                name="pbfm-realsense-latest",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stop_requested:
                    return
            try:
                frames = self.pipeline.wait_for_frames(
                    timeout_ms=self.wait_timeout_ms
                )
                receive_monotonic_s = float(self.monotonic_clock())
                receive_realtime_s = float(self.realtime_clock())
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                timestamp_domain = str(depth_frame.get_frame_timestamp_domain())
                device_timestamp_s = float(depth_frame.get_timestamp()) * 1.0e-3
                frame_number = int(depth_frame.get_frame_number())
                if (
                    self._timestamp_domain is not None
                    and timestamp_domain != self._timestamp_domain
                ):
                    self._clock_mapper = AffineClockMapper()
                    self._clock_sample_count = 0
                    self._reset_epoch += 1
                if (
                    self._previous_device_timestamp_s is not None
                    and device_timestamp_s <= self._previous_device_timestamp_s
                ):
                    self._clock_mapper = AffineClockMapper()
                    self._clock_sample_count = 0
                    self._reset_epoch += 1
                if (
                    self._previous_frame_number is not None
                    and frame_number <= self._previous_frame_number
                ):
                    self._reset_epoch += 1
                self._timestamp_domain = timestamp_domain
                self._previous_device_timestamp_s = device_timestamp_s
                self._previous_frame_number = frame_number
                if "global_time" in timestamp_domain.lower():
                    capture_monotonic_s = _realtime_source_to_monotonic(
                        device_timestamp_s,
                        receive_monotonic_s,
                        receive_realtime_s,
                    )
                else:
                    capture_monotonic_s = self._clock_mapper.add(
                        device_timestamp_s, receive_monotonic_s
                    )
                # A RealSense exposure cannot occur after the host has received
                # the frame.  On some devices the fitted hardware-clock mapper
                # can briefly predict a timestamp a few milliseconds in the
                # future, which makes downstream freshness checks reject an
                # otherwise valid latest frame.  Clamp to the co-sampled receive
                # time so freshness remains conservative and monotonic-domain
                # safe.
                capture_monotonic_s = min(
                    float(capture_monotonic_s),
                    receive_monotonic_s,
                )
                self._clock_sample_count += 1
                if self._clock_sample_count < self.clock_warmup_frames:
                    continue
            except RuntimeError as exc:
                with self._condition:
                    if self._stop_requested:
                        return
                    self.capture_timeouts += 1
                    self.last_error = str(exc)
                    self._condition.notify_all()
                continue

            with self._condition:
                if self._stop_requested:
                    return
                self._sequence += 1
                if len(self._frames) >= self.buffer_capacity:
                    self._frames.popleft()
                    self.overwritten_frames += 1
                self._frames.append(CapturedDepthFrame(
                    depth_frame=depth_frame,
                    receive_monotonic_s=receive_monotonic_s,
                    receive_realtime_s=receive_realtime_s,
                    acquisition_sequence=self._sequence,
                    capture_monotonic_s=capture_monotonic_s,
                    frame_number=frame_number,
                    reset_epoch=self._reset_epoch,
                ))
                self.acquired_frames += 1
                self.last_error = ""
                self._condition.notify_all()

    def take_latest(
        self,
        *,
        timeout_s: float,
        ready: Callable[[CapturedDepthFrame], bool] | None = None,
    ) -> CapturedDepthFrame:
        timeout_s = float(timeout_s)
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("RealSense take timeout must be finite and positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                selected_index = None
                for index in range(len(self._frames) - 1, -1, -1):
                    if ready is None or ready(self._frames[index]):
                        selected_index = index
                        break
                if selected_index is not None:
                    self.selection_skips += selected_index
                    for _ in range(selected_index):
                        self._frames.popleft()
                    sample = self._frames.popleft()
                    return sample
                if self._stop_requested:
                    raise RuntimeError("RealSense capture is stopping")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    detail = f": {self.last_error}" if self.last_error else ""
                    raise RuntimeError(
                        f"no RealSense depth frame available within "
                        f"{timeout_s * 1000.0:.0f} ms{detail}"
                    )
                # State-buffer coverage changes independently of camera frame
                # arrival, so poll readiness briefly even without a capture
                # condition notification.
                self._condition.wait(timeout=min(0.002, remaining))

    def stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(2.0, self.wait_timeout_ms / 1000.0 + 0.5))
            if thread.is_alive():
                raise RuntimeError("RealSense capture thread did not stop")


@dataclass
class RealSenseRuntime:
    """Started RealSense pipeline and immutable active-profile calibration."""

    pipeline: Any
    profile: Any
    capture: LatestRealSenseCapture
    depth_scale: float
    raw_k: np.ndarray
    output_k: np.ndarray
    distortion_name: str
    distortion_coeffs: tuple[float, ...]


def _safe_stop_capture(capture: LatestRealSenseCapture | None) -> None:
    if capture is None:
        return
    try:
        capture.stop()
    except RuntimeError as exc:
        print(f"[WARN] RealSense capture shutdown failed: {exc}")


def _safe_stop_pipeline(pipeline: Any | None) -> None:
    if pipeline is None:
        return
    try:
        pipeline.stop()
    except RuntimeError as exc:
        print(f"[WARN] RealSense pipeline shutdown failed: {exc}")


def _close_realsense_runtime(runtime: RealSenseRuntime | None) -> None:
    if runtime is None:
        return
    _safe_stop_capture(runtime.capture)
    _safe_stop_pipeline(runtime.pipeline)


def _start_realsense_runtime(
    rs: Any,
    args: argparse.Namespace,
    *,
    depth_crop: DepthCropConfig,
) -> RealSenseRuntime:
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(args.camera_serial)
    rs_config.enable_stream(
        rs.stream.depth,
        args.depth_width,
        args.depth_height,
        rs.format.z16,
        args.depth_fps,
    )
    profile = None
    capture = None
    try:
        profile = pipeline.start(rs_config)
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intrinsics = depth_profile.get_intrinsics()
        if (intrinsics.width, intrinsics.height) != (
            args.depth_width,
            args.depth_height,
        ):
            raise RuntimeError(
                "active RealSense profile dimensions do not match the requested profile: "
                f"{intrinsics.width}x{intrinsics.height}"
            )
        distortion_name = str(intrinsics.model)
        require_pinhole_depth(distortion_name, intrinsics.coeffs)
        raw_k = np.array(
            (
                (intrinsics.fx, 0.0, intrinsics.ppx),
                (0.0, intrinsics.fy, intrinsics.ppy),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        output_k = crop_and_scale_intrinsics(
            raw_k,
            source_width=args.depth_width,
            source_height=args.depth_height,
            target_width=args.output_width,
            target_height=args.output_height,
            crop=depth_crop,
        )
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        if not np.isfinite(depth_scale) or depth_scale <= 0.0:
            raise RuntimeError(f"invalid RealSense depth scale: {depth_scale}")
        capture = LatestRealSenseCapture(
            pipeline,
            wait_timeout_ms=args.camera_wait_timeout_ms,
            buffer_capacity=args.camera_buffer_capacity,
            clock_warmup_frames=args.clock_warmup_frames,
        )
        capture.start()
        return RealSenseRuntime(
            pipeline=pipeline,
            profile=profile,
            capture=capture,
            depth_scale=depth_scale,
            raw_k=raw_k,
            output_k=output_k,
            distortion_name=distortion_name,
            distortion_coeffs=tuple(float(value) for value in intrinsics.coeffs),
        )
    except Exception:
        _safe_stop_capture(capture)
        if profile is not None:
            _safe_stop_pipeline(pipeline)
        raise


def _as_finite_array(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(list(value), dtype=np.float64).reshape(-1)
    if result.size != size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _realtime_source_to_monotonic(
    source_realtime_s: float,
    receive_monotonic_s: float,
    receive_realtime_s: float,
    *,
    max_abs_age_s: float = 5.0,
) -> float:
    """Map a realtime-domain source stamp with a co-sampled host offset."""
    values = np.asarray(
        (source_realtime_s, receive_monotonic_s, receive_realtime_s),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("realtime clock samples must be finite")
    source_age_s = float(receive_realtime_s - source_realtime_s)
    if abs(source_age_s) > float(max_abs_age_s):
        raise ValueError(
            f"realtime source stamp differs from host receive time by {source_age_s:.3f} s"
        )
    return float(source_realtime_s + receive_monotonic_s - receive_realtime_s)


class LowStateBridge:
    """Validate SDK/ROS samples and map the controller tick into host time."""

    def __init__(
        self,
        low_state_publisher: zmq.Socket,
    ) -> None:
        self.low_state_publisher = low_state_publisher
        self.low_state_clock = AffineClockMapper()
        self.low_state_count = 0
        self.last_error = ""
        self.last_low_state_tick: int | None = None
        self.low_state_tick_epoch = 0

    def on_low_state(self, message: Any) -> None:
        receive_monotonic_s = time.monotonic()
        try:
            motor_state = list(message.motor_state)
            if len(motor_state) < 29:
                raise ValueError(f"motor_state has {len(motor_state)} entries")
            joint_positions = np.asarray(
                [motor_state[index].q for index in range(29)], dtype=np.float32
            )
            joint_velocities = np.asarray(
                [motor_state[index].dq for index in range(29)], dtype=np.float32
            )
            joint_torques = np.asarray(
                [motor_state[index].tau_est for index in range(29)], dtype=np.float32
            )
            imu = message.imu_state
            quaternion = _as_finite_array(imu.quaternion, 4, "low-state quaternion")
            norm = float(np.linalg.norm(quaternion))
            if not np.isfinite(norm) or norm <= 1.0e-10:
                raise ValueError("quaternion must be finite and non-zero")
            gyroscope = _as_finite_array(imu.gyroscope, 3, "low-state gyroscope")
            if not (
                np.all(np.isfinite(joint_positions))
                and np.all(np.isfinite(joint_velocities))
                and np.all(np.isfinite(joint_torques))
            ):
                raise ValueError("low-state joint arrays contain non-finite values")
            raw_tick_value = getattr(message, "tick", None)
            if raw_tick_value is None:
                tick = self.low_state_count
                capture_monotonic_s = receive_monotonic_s
            else:
                tick = int(raw_tick_value)
                if not 0 <= tick <= 0xFFFFFFFF:
                    raise ValueError(f"low-state tick is outside uint32: {tick}")
                if tick == self.last_low_state_tick:
                    self.last_error = f"duplicate low-state tick rejected: {tick}"
                    return
                if self.last_low_state_tick is not None:
                    forward_delta = (tick - self.last_low_state_tick) & 0xFFFFFFFF
                    if forward_delta > 0x7FFFFFFF:
                        # Controller/DDS restart, rather than uint32 rollover.
                        self.low_state_clock = AffineClockMapper()
                        self.low_state_tick_epoch = 0
                    elif tick < self.last_low_state_tick:
                        self.low_state_tick_epoch += 1 << 32
                self.last_low_state_tick = tick
                controller_timestamp_s = (
                    self.low_state_tick_epoch + tick
                ) * 1.0e-3
                capture_monotonic_s = self.low_state_clock.add(
                    controller_timestamp_s, receive_monotonic_s
                )
            packet = LowStateMessage(
                quaternion=quaternion,
                gyroscope=gyroscope,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_torques=joint_torques,
                tick=tick,
                capture_monotonic_s=capture_monotonic_s,
            )
            try:
                self.low_state_publisher.send(packet.to_bytes(), flags=zmq.DONTWAIT)
            except zmq.Again:
                pass
            self.low_state_count += 1
        except Exception as exc:
            self.last_error = f"low-state rejected: {exc}"


def _dds_topic(topic: str) -> str:
    topic = str(topic).strip()
    if not topic:
        raise ValueError("DDS topic must be non-empty")
    return topic if topic.startswith("rt/") else "rt/" + topic.lstrip("/")


class _LatestPacketQueue:
    """Move SDK callback data onto the main thread before touching ZMQ."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: bytes | None = None

    def send(self, packet: bytes, flags: int = 0) -> None:
        del flags
        with self._lock:
            self._packet = bytes(packet)

    def flush_to(self, publisher: zmq.Socket) -> None:
        with self._lock:
            packet = self._packet
            self._packet = None
        if packet is not None:
            try:
                publisher.send(packet, flags=zmq.DONTWAIT)
            except zmq.Again:
                pass


class _SdkPollingExecutor:
    """Executor-shaped adapter that flushes SDK2 state on the main thread."""

    def __init__(self, packet_queue: _LatestPacketQueue, publisher: zmq.Socket) -> None:
        self.packet_queue = packet_queue
        self.publisher = publisher

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        self.packet_queue.flush_to(self.publisher)
        if timeout_sec > 0.0:
            time.sleep(timeout_sec)


class Sdk2StateRuntime:
    """Read-only Unitree SDK2 source; this class has no LowCmd imports."""

    def __init__(
        self,
        low_state_publisher: zmq.Socket,
        network_interface: str,
        low_state_topic: str,
        sample_rate_hz: float,
    ) -> None:
        sample_rate_hz = float(sample_rate_hz)
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError("SDK2 low-state sample rate must be finite and positive")
        self.sample_period_s = 1.0 / sample_rate_hz
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        except Exception as exc:
            raise RuntimeError(
                "Unitree SDK2 read-only state dependencies are unavailable"
            ) from exc

        packet_queue = _LatestPacketQueue()
        self.bridge = LowStateBridge(packet_queue)
        self.executor = _SdkPollingExecutor(packet_queue, low_state_publisher)
        ChannelFactoryInitialize(0, str(network_interface))
        self.low_state_subscriber = ChannelSubscriber(
            _dds_topic(low_state_topic), LowState_
        )
        # Poll KEEP_LAST to avoid a backlog of obsolete controller samples.
        self.low_state_subscriber.Init()
        self._stop_event = threading.Event()
        self._threads = (
            threading.Thread(
                target=self._poll,
                args=(self.low_state_subscriber, self.bridge.on_low_state),
                name="ufo_lowstate_reader",
                daemon=True,
            ),
        )
        for thread in self._threads:
            thread.start()

    def _poll(self, subscriber: Any, callback: Any) -> None:
        next_read_s = time.monotonic()
        while not self._stop_event.is_set():
            delay_s = next_read_s - time.monotonic()
            if delay_s > 0.0 and self._stop_event.wait(delay_s):
                return
            message = subscriber.Read(timeout=0.05)
            if message is not None:
                callback(message)
            next_read_s += self.sample_period_s
            if next_read_s < time.monotonic():
                next_read_s = time.monotonic()

    @staticmethod
    def ok() -> bool:
        return True

    def close(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=0.2)
        self.low_state_subscriber.Close()


class Ros2StateRuntime:
    """Optional ROS2 state source retained for bridge-based deployments."""

    def __init__(
        self,
        low_state_publisher: zmq.Socket,
        low_state_topic: str,
        ros_args: list[str],
    ) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from unitree_hg.msg import LowState
        except Exception as exc:
            raise RuntimeError("ROS2 state dependencies are unavailable") from exc

        self.rclpy = rclpy
        self.bridge = LowStateBridge(low_state_publisher)
        rclpy.init(args=ros_args)
        self.node = rclpy.create_node("ufo_realsense_g1_terrain_producer")
        self.node.create_subscription(
            LowState,
            low_state_topic,
            self.bridge.on_low_state,
            qos_profile_sensor_data,
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def ok(self) -> bool:
        return bool(self.rclpy.ok())

    def close(self) -> None:
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self.rclpy.shutdown()


def _publisher(context: zmq.Context, endpoint: str) -> zmq.Socket:
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(endpoint)
    return socket


def _spin_ready(executor: Any, count: int = 8) -> None:
    if executor is None:
        return
    for _ in range(count):
        executor.spin_once(timeout_sec=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="No-actuation synchronized D435i/G1 terrain producer"
    )
    parser.add_argument("--robot-config", default="config/robot/g1_real.yaml")
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--depth-width", type=int, default=480)
    parser.add_argument("--depth-height", type=int, default=270)
    parser.add_argument("--depth-fps", type=int, default=60)
    parser.add_argument("--output-width", type=int, default=64)
    parser.add_argument("--output-height", type=int, default=36)
    parser.add_argument("--low-state-topic", default="/lowstate")
    parser.add_argument(
        "--odom-topic",
        default=None,
        help="Deprecated and ignored: Phase-2G local projection does not use odometry",
    )
    parser.add_argument(
        "--state-backend",
        choices=("external", "sdk2", "ros2"),
        default="external",
        help="external: camera only with independent state publisher (default); sdk2/ros2: legacy combined mode",
    )
    parser.add_argument(
        "--network-interface",
        default=None,
        help="Unitree SDK2 DDS interface; defaults to G1_INTERFACE or robot config INTERFACE",
    )
    parser.add_argument("--bind-address", default="127.0.0.1")
    parser.add_argument("--stats-interval-s", type=float, default=5.0)
    parser.add_argument("--clock-warmup-frames", type=int, default=8)
    parser.add_argument(
        "--camera-wait-timeout-ms",
        type=int,
        default=1000,
        help="RealSense wait_for_frames timeout inside the capture thread",
    )
    parser.add_argument(
        "--camera-take-timeout-s",
        type=float,
        default=1.0,
        help="If no latest depth frame is available for this long, restart RealSense",
    )
    parser.add_argument(
        "--camera-restart-backoff-s",
        type=float,
        default=0.5,
        help="Delay before retrying after a RealSense stream stall",
    )
    parser.add_argument(
        "--camera-stall-action",
        choices=("exit", "restart"),
        default="exit",
        help=(
            "What to do after camera-take-timeout-s without a depth frame. "
            "'exit' lets an external supervisor restart the whole process and "
            "avoids repeated in-process librealsense stop/start crashes; "
            "'restart' keeps the old in-process pipeline restart behavior."
        ),
    )
    parser.add_argument(
        "--camera-buffer-capacity",
        type=int,
        default=8,
        help="Latest-frame buffer size used only to bridge capture/producer scheduling",
    )
    parser.add_argument(
        "--max-camera-restarts",
        type=int,
        default=0,
        help="Maximum RealSense restarts before failing; 0 means unlimited",
    )
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument(
        "--allow-nominal-robot-center-proxy",
        action="store_true",
        help=(
            "Deprecated compatibility flag; Phase-2G does not use robot_center or odometry"
        ),
    )
    args, ros_args = parser.parse_known_args()

    if args.bind_address not in {"127.0.0.1", "localhost", "::1"}:
        parser.error(
            "capture_monotonic_s is host-local; producer endpoints must remain loopback-only"
        )
    if min(
        args.depth_width,
        args.depth_height,
        args.depth_fps,
        args.output_width,
        args.output_height,
        args.clock_warmup_frames,
        args.camera_wait_timeout_ms,
        args.camera_buffer_capacity,
    ) <= 0:
        parser.error(
            "depth dimensions, FPS, clock warmup, camera wait timeout, "
            "and camera buffer capacity must be positive"
        )
    if (
        not np.isfinite(float(args.camera_take_timeout_s))
        or args.camera_take_timeout_s <= 0.0
    ):
        parser.error("--camera-take-timeout-s must be finite and positive")
    if (
        not np.isfinite(float(args.camera_restart_backoff_s))
        or args.camera_restart_backoff_s <= 0.0
    ):
        parser.error("--camera-restart-backoff-s must be finite and positive")
    if args.max_camera_restarts < 0:
        parser.error("--max-camera-restarts must be non-negative")
    config_path = Path(args.robot_config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    robot_config = load_yaml_config(config_path)
    low_state_port = int(robot_config["LOW_STATE_PORT"])
    terrain_port = int(robot_config["TERRAIN_DEPTH_PORT"])
    sensor_config = robot_config["REAL_TERRAIN_SENSOR"]
    try:
        min_depth_m = float(sensor_config["MIN_DEPTH_M"])
        max_depth_m = float(sensor_config["MAX_DEPTH_M"])
    except (KeyError, TypeError, ValueError):
        parser.error("REAL_TERRAIN_SENSOR depth limits must be finite numbers")
    if not np.isfinite(min_depth_m) or min_depth_m != 0.0:
        parser.error("REAL_TERRAIN_SENSOR.MIN_DEPTH_M must be exactly 0.0")
    if not np.isfinite(max_depth_m) or max_depth_m != 2.5:
        parser.error("REAL_TERRAIN_SENSOR.MAX_DEPTH_M must be exactly 2.5")
    try:
        depth_crop = DepthCropConfig(**dict(sensor_config.get("DEPTH_CROP", {})))
        depth_crop.validate()
        depth_crop.bounds(
            source_width=args.depth_width,
            source_height=args.depth_height,
        )
    except (TypeError, ValueError) as exc:
        parser.error(f"REAL_TERRAIN_SENSOR.DEPTH_CROP is invalid: {exc}")
    torso_to_optical_position = _as_finite_array(
        sensor_config["TORSO_TO_CAMERA_OPTICAL_POSITION"],
        3,
        "TORSO_TO_CAMERA_OPTICAL_POSITION",
    )
    torso_to_optical_quaternion = _as_finite_array(
        sensor_config["TORSO_TO_CAMERA_OPTICAL_QUATERNION_XYZW"],
        4,
        "TORSO_TO_CAMERA_OPTICAL_QUATERNION_XYZW",
    )
    for name, quaternion in (
        ("TORSO_TO_CAMERA_OPTICAL_QUATERNION_XYZW", torso_to_optical_quaternion),
    ):
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-8:
            parser.error(f"REAL_TERRAIN_SENSOR.{name} must be non-zero")
        quaternion /= norm

    try:
        import pyrealsense2 as rs
    except Exception as exc:
        print(f"[FAIL] hardware dependency import failed: {exc.__class__.__name__}: {exc}")
        return 1

    context = zmq.Context.instance()
    low_state_pub = None
    terrain_pub = _publisher(context, f"tcp://{args.bind_address}:{terrain_port}")
    native_pub = None
    native_last = 0.0
    try:
        native_pub = _publisher(context, "tcp://127.0.0.1:5596")
        print("[INFO] Native unmodified Z16 preview on loopback:5596, camera rate (60 Hz)")
    except Exception as exc:
        print(f"[WARN] Native preview disabled: {exc}")
    state_runtime = None
    try:
        if args.state_backend != "ros2" and ros_args:
            parser.error(f"unrecognized arguments: {' '.join(ros_args)}")
        if args.state_backend != "external":
            print("[WARN] legacy combined camera/state mode; use the supervised launcher for process isolation")
            low_state_pub = _publisher(
                context, f"tcp://{args.bind_address}:{low_state_port}"
            )
        if args.state_backend == "sdk2":
            if ros_args:
                parser.error(f"unrecognized arguments for sdk2 backend: {' '.join(ros_args)}")
            network_interface = (
                args.network_interface
                or os.environ.get("G1_INTERFACE")
                or robot_config.get("INTERFACE")
            )
            if not network_interface:
                parser.error(
                    "--network-interface, G1_INTERFACE, or robot config INTERFACE is required"
                )
            state_runtime = Sdk2StateRuntime(
                low_state_pub,
                str(network_interface),
                args.low_state_topic,
                float(robot_config.get("SDK2_LOW_STATE_RATE_HZ", 200.0)),
            )
        elif args.state_backend == "ros2":
            state_runtime = Ros2StateRuntime(
                low_state_pub,
                args.low_state_topic,
                ros_args,
            )
    except Exception as exc:
        print(f"[FAIL] state backend initialization failed: {exc.__class__.__name__}: {exc}")
        if low_state_pub is not None:
            low_state_pub.close(linger=0)
        if native_pub is not None:
            native_pub.close(linger=0)
        terrain_pub.close(linger=0)
        return 1
    state_bridge = None if state_runtime is None else state_runtime.bridge
    executor = None if state_runtime is None else state_runtime.executor

    def state_ok() -> bool:
        return state_runtime is None or state_runtime.ok()

    print("[INFO] NO-ACTUATION: no g1_interface and no motor command publisher")
    print(f"[INFO] state_backend={args.state_backend}")
    if args.state_backend == "external":
        print("[INFO] CAMERA-ONLY: no DDS subscription or low-state socket; camera restart leaves state publisher running")
    print(
        "[INFO] direct-depth path: optical-Z image -> policy CNN+GRU; "
        "no projection, odometry, self mask or estimator"
    )

    rs_runtime = None
    published = 0
    previous_frame_number = None
    previous_reset_epoch = None
    reset_history = True
    started_monotonic_s = time.monotonic()
    last_stats_s = started_monotonic_s
    camera_restarts = 0
    camera_start_failures = 0
    retired_capture_stats = {
        "acquired_frames": 0,
        "overwritten_frames": 0,
        "selection_skips": 0,
        "capture_timeouts": 0,
    }

    def _record_capture_stats(runtime: RealSenseRuntime | None) -> None:
        if runtime is None:
            return
        capture_stats = runtime.capture
        retired_capture_stats["acquired_frames"] += capture_stats.acquired_frames
        retired_capture_stats["overwritten_frames"] += capture_stats.overwritten_frames
        retired_capture_stats["selection_skips"] += capture_stats.selection_skips
        retired_capture_stats["capture_timeouts"] += capture_stats.capture_timeouts

    def _close_current_realsense() -> None:
        nonlocal rs_runtime
        if rs_runtime is None:
            return
        _record_capture_stats(rs_runtime)
        _close_realsense_runtime(rs_runtime)
        rs_runtime = None

    def _spin_for_backoff() -> None:
        deadline = time.monotonic() + float(args.camera_restart_backoff_s)
        while state_ok():
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                return
            _spin_ready(executor)
            time.sleep(min(0.05, remaining_s))

    def _print_realsense_profile(runtime: RealSenseRuntime, *, restarted: bool) -> None:
        label = "RealSense restarted" if restarted else "RealSense"
        print(
            f"[INFO] {label} {args.depth_width}x{args.depth_height}@{args.depth_fps}, "
            f"depth_scale={runtime.depth_scale:.9f}, "
            f"direct_depth_range_m=[{min_depth_m:.3f}, {max_depth_m:.3f}], "
            f"resize={args.depth_width}x{args.depth_height}->{args.output_width}x{args.output_height}, "
            f"policy_preprocess=crop16+fixed3x3_gaussian+uint8, "
            f"robot_pixels=kept, "
            f"distortion={runtime.distortion_name}, coeffs={list(runtime.distortion_coeffs)}"
        )
        print(f"[INFO] active-profile K=\n{runtime.raw_k}")

    def _open_realsense_with_retry(*, restarted: bool) -> RealSenseRuntime | None:
        nonlocal camera_start_failures
        while state_ok():
            if (
                args.max_camera_restarts
                and camera_restarts + camera_start_failures
                >= args.max_camera_restarts
            ):
                raise RuntimeError(
                    f"RealSense restart limit reached: "
                    f"restarts={camera_restarts}, start_failures={camera_start_failures}, "
                    f"limit={args.max_camera_restarts}"
                )
            try:
                runtime = _start_realsense_runtime(
                    rs,
                    args,
                    depth_crop=depth_crop,
                )
                _print_realsense_profile(runtime, restarted=restarted)
                return runtime
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                camera_start_failures += 1
                print(
                    f"[WARN] RealSense start failed; retrying in "
                    f"{args.camera_restart_backoff_s:.2f}s "
                    f"(start_failures={camera_start_failures}): "
                    f"{exc.__class__.__name__}: {exc}"
                )
                _spin_for_backoff()
        return None

    try:
        rs_runtime = _open_realsense_with_retry(restarted=False)

        while state_ok():
            if (
                args.duration_s > 0.0
                and time.monotonic() - started_monotonic_s >= args.duration_s
            ):
                break
            _spin_ready(executor)
            if rs_runtime is None:
                rs_runtime = _open_realsense_with_retry(restarted=True)
                if rs_runtime is None:
                    break
                reset_history = True
                previous_frame_number = None
                previous_reset_epoch = None
            # Direct visual control does not wait for pose bracketing. The
            # independently published LowStateMessage supplies proprioception.
            try:
                captured = rs_runtime.capture.take_latest(
                    timeout_s=args.camera_take_timeout_s
                )
            except RuntimeError as exc:
                camera_restarts += 1
                if args.camera_stall_action == "exit":
                    print(
                        f"[ERROR] RealSense depth stream stalled; exiting producer "
                        f"for external process restart "
                        f"(restart_request={camera_restarts}, "
                        f"exit_code={REALSENSE_STALL_EXIT_CODE}): {exc}"
                    )
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(REALSENSE_STALL_EXIT_CODE)
                print(
                    f"[WARN] RealSense depth stream stalled; restarting pipeline "
                    f"(restart={camera_restarts}): {exc}"
                )
                _close_current_realsense()
                reset_history = True
                previous_frame_number = None
                previous_reset_epoch = None
                _spin_for_backoff()
                continue
            depth_frame = captured.depth_frame
            receive_monotonic_s = captured.receive_monotonic_s
            capture_monotonic_s = captured.capture_monotonic_s
            _spin_ready(executor)
            sync_ready_monotonic_s = time.monotonic()

            frame_number = captured.frame_number
            if (
                previous_reset_epoch is not None
                and captured.reset_epoch != previous_reset_epoch
            ):
                reset_history = True
            previous_reset_epoch = captured.reset_epoch
            if previous_frame_number is not None and frame_number <= previous_frame_number:
                reset_history = True
            previous_frame_number = frame_number

            raw_z16 = np.asanyarray(depth_frame.get_data())
            if raw_z16.shape != (args.depth_height, args.depth_width):
                raise RuntimeError(f"unexpected depth shape: {raw_z16.shape}")
            if native_pub is not None:
                native_last = time.monotonic()
                try:
                    header = struct.pack("<4sIIQd", b"RZ16", raw_z16.shape[1], raw_z16.shape[0],
                                         frame_number, rs_runtime.depth_scale)
                    native_pub.send(header + raw_z16.astype("<u2", copy=False).tobytes(), flags=zmq.DONTWAIT)
                except zmq.Again:
                    pass
                except Exception as exc:
                    print(f"[WARN] Native preview disabled after error: {exc}")
                    native_pub.close(linger=0)
                    native_pub = None
            policy_z16 = raw_z16
            spatial_cfg = sensor_config.get("SPATIAL_FILTER", {})
            spatial_env = os.environ.get("DEPTH_SPATIAL_FILTER")
            spatial_enabled = bool(spatial_cfg.get("enabled", False)) if spatial_env is None else spatial_env == "1"
            if spatial_enabled:
                temporal_cfg = sensor_config.get("TEMPORAL_FILTER", {})
                temporal_env = os.environ.get("DEPTH_TEMPORAL_FILTER")
                temporal_enabled = bool(temporal_cfg.get("enabled", False)) if temporal_env is None else temporal_env == "1"
                hole_fill_env = os.environ.get("DEPTH_HOLE_FILLING_MODE")
                hole_fill_mode = None
                if hole_fill_env is not None:
                    hole_fill_mode = int(hole_fill_env)
                    if hole_fill_mode not in (0, 1, 2):
                        raise ValueError("DEPTH_HOLE_FILLING_MODE must be 0, 1, or 2")
                preserve_sensor_holes = os.environ.get("DEPTH_PRESERVE_SENSOR_HOLES", "1") == "1"
                spatial = getattr(rs_runtime, "spatial_filter", None)
                if spatial is None:
                    spatial = rs.spatial_filter()
                    magnitude = max(1, min(5, int(spatial_cfg.get("magnitude", 1))))
                    alpha = max(0.0, min(1.0, float(spatial_cfg.get("alpha", 0.8))))
                    delta = max(1, min(100, int(spatial_cfg.get("delta", 10))))
                    holes_fill = max(0, min(5, int(spatial_cfg.get("holes_fill", 0))))
                    spatial.set_option(rs.option.filter_magnitude, magnitude)
                    spatial.set_option(rs.option.filter_smooth_alpha, alpha)
                    spatial.set_option(rs.option.filter_smooth_delta, delta)
                    spatial.set_option(rs.option.holes_fill, holes_fill)
                    rs_runtime.spatial_filter = spatial
                    if hole_fill_mode is not None:
                        rs_runtime.hole_filling_filter = rs.hole_filling_filter(hole_fill_mode)
                    if temporal_enabled:
                        rs_runtime.disparity_to_depth = rs.disparity_transform(False)
                        rs_runtime.depth_to_disparity = rs.disparity_transform(True)
                        temporal = rs.temporal_filter()
                        talpha = max(0.0, min(1.0, float(temporal_cfg.get("alpha", 0.4))))
                        tdelta = max(1, min(100, int(temporal_cfg.get("delta", 20))))
                        persistence = max(0, min(8, int(temporal_cfg.get("persistence_control", 2))))
                        temporal.set_option(rs.option.filter_smooth_alpha, talpha)
                        temporal.set_option(rs.option.filter_smooth_delta, tdelta)
                        rs_runtime.temporal_filter = temporal
                    temporal_state = "on" if temporal_enabled else "off"
                    print(
                        f"[INFO] Spatial enabled: magnitude={magnitude} alpha={alpha} "
                        f"delta={delta} holes_fill={holes_fill}; temporal={temporal_state}; "
                        f"independent_hole_fill={hole_fill_mode}; "
                        f"preserve_sensor_holes={preserve_sensor_holes}",
                        flush=True,
                    )
                filtered_frame = depth_frame
                if temporal_enabled:
                    filtered_frame = rs_runtime.depth_to_disparity.process(filtered_frame)
                hole_filling_filter = getattr(rs_runtime, "hole_filling_filter", None)
                if hole_filling_filter is not None:
                    filtered_frame = hole_filling_filter.process(filtered_frame)
                filtered_frame = spatial.process(filtered_frame).as_depth_frame()
                if temporal_enabled:
                    filtered_frame = rs_runtime.temporal_filter.process(filtered_frame).as_depth_frame()
                    filtered_frame = rs_runtime.disparity_to_depth.process(filtered_frame).as_depth_frame()
                policy_z16 = np.asanyarray(filtered_frame.get_data()).copy()
                if preserve_sensor_holes:
                    # Preserve sensor holes even if SDK behavior changes.
                    policy_z16[raw_z16 == 0] = 0
            depth_m = policy_z16.astype(np.float32) * rs_runtime.depth_scale
            if os.environ.get("DEPTH_TRAINING_RAYS", "0") == "1":
                if (args.output_width, args.output_height) != (64, 36):
                    raise ValueError("Training rays require 64x36 transport")
                sampler = getattr(rs_runtime, "training_sampler", None)
                if sampler is None:
                    sampler = TrainingDepthSampler(rs_runtime.raw_k, depth_m.shape, os.environ.get("DEPTH_PROCESSING_PROFILE", "legacy_rp1"))
                    rs_runtime.training_sampler = sampler
                    print(f"[INFO] training ray nearest sampling: cropped coverage={sampler.covered.mean():.4f}; profile={sampler.processing_profile}; range limit=2.5m; no hole fill", flush=True)
                depth_small = sampler.sample(depth_m)
                rs_runtime.output_k = sampler.output_k
            else:
                depth_m = gate_metric_depth(
                    depth_m,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
                depth_small = crop_and_resize_depth_area(
                    depth_m,
                    target_width=args.output_width,
                    target_height=args.output_height,
                    crop=depth_crop,
                )
                # RealSense invalid pixels and rays outside the max depth use the
                # same zero sentinel as the direct-depth training contract.
                depth_small = np.nan_to_num(
                    depth_small, nan=0.0, posinf=max_depth_m, neginf=0.0
                ).astype(np.float32)
            # Marks the beginning of packet construction/serialization/send
            # enqueue. All A--E timestamps use this host monotonic clock.
            producer_publish_monotonic_s = time.monotonic()
            packet = TerrainDepthMessage(
                depth_z=depth_small,
                intrinsic_matrix=rs_runtime.output_k,
                camera_pos_torso=torso_to_optical_position,
                camera_optical_quat_torso_xyzw=torso_to_optical_quaternion,
                # These fields remain in wire version 4 for transport tooling,
                # but the direct policy reads proprioception from LOW_STATE.
                projected_gravity=np.array((0.0, 0.0, -1.0)),
                waist_joint_pos=np.zeros(3),
                joint_pos=np.zeros(29),
                joint_vel=np.zeros(29),
                base_ang_vel=np.zeros(3),
                timestamp_s=capture_monotonic_s,
                capture_monotonic_s=capture_monotonic_s,
                camera_receive_monotonic_s=receive_monotonic_s,
                sync_ready_monotonic_s=sync_ready_monotonic_s,
                producer_publish_monotonic_s=producer_publish_monotonic_s,
                frame_number=frame_number,
                reset_history=reset_history,
            )
            try:
                terrain_pub.send(packet.to_bytes(), flags=zmq.DONTWAIT)
            except zmq.Again:
                pass
            reset_history = False
            published += 1

            now = time.monotonic()
            if now - last_stats_s >= args.stats_interval_s:
                elapsed = max(now - started_monotonic_s, 1.0e-9)
                active_capture = rs_runtime.capture if rs_runtime is not None else None
                acquired_frames = retired_capture_stats["acquired_frames"] + (
                    active_capture.acquired_frames if active_capture is not None else 0
                )
                overwritten_frames = retired_capture_stats["overwritten_frames"] + (
                    active_capture.overwritten_frames if active_capture is not None else 0
                )
                selection_skips = retired_capture_stats["selection_skips"] + (
                    active_capture.selection_skips if active_capture is not None else 0
                )
                capture_timeouts = retired_capture_stats["capture_timeouts"] + (
                    active_capture.capture_timeouts if active_capture is not None else 0
                )
                state_stats = (
                    "lowstate=external"
                    if state_bridge is None
                    else f"lowstate={state_bridge.low_state_count} last_state_error={state_bridge.last_error!r}"
                )
                print(
                    f"[INFO] depth_publish_rate={published / elapsed:.2f} Hz "
                    f"camera_acquire_rate={acquired_frames / elapsed:.2f} Hz "
                    f"camera_overwrites={overwritten_frames} "
                    f"camera_selection_skips={selection_skips} "
                    f"camera_timeouts={capture_timeouts} "
                    f"camera_restarts={camera_restarts} "
                    f"camera_start_failures={camera_start_failures} "
                    f"sync_wait=disabled self_mask=disabled "
                    f"{state_stats}"
                )
                last_stats_s = now
    except KeyboardInterrupt:
        print("[INFO] interrupted")
    except Exception as exc:
        print(f"[FAIL] producer stopped: {exc.__class__.__name__}: {exc}")
        return 1
    finally:
        _close_current_realsense()
        if state_runtime is not None:
            state_runtime.close()
        if low_state_pub is not None:
            low_state_pub.close(linger=0)
        if native_pub is not None:
            native_pub.close(linger=0)
        terrain_pub.close(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
