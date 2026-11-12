#!/usr/bin/env python3
"""Configurable-rate direct-depth encoder worker.

This process subscribes to the no-actuation RealSense producer's 64x36 optical-Z
packets, applies the frozen direct-depth preprocessing/history contract, runs
``depth_encoder.onnx`` and publishes the latest 256D ``depth_latent`` for
the 50 Hz policy loop.
"""

from __future__ import annotations

import argparse
import csv
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq
from loguru import logger


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_policy.direct_depth import DirectDepthConfig  # noqa: E402
from rl_policy.utils.state_processor import StateProcessor  # noqa: E402
from utils.common import DirectDepthLatentMessage  # noqa: E402
from utils.latency_stats import LatencyStats
from utils.phase_schedule import next_phase_tick
from utils.timestamp_depth_history import TimestampDepthHistory, resolve_encoder_rate
from utils.config import load_yaml_config  # noqa: E402
from utils.onnx_module import create_inference_session  # noqa: E402


def _backend(policy_config: dict[str, Any]) -> str:
    backend = str(
        policy_config.get(
            "depth_inference_backend",
            policy_config.get("inference_backend", "onnx"),
        )
    ).strip().lower()
    if backend not in {"onnx", "tensorrt"}:
        raise ValueError(f"Unsupported depth inference backend: {backend!r}")
    return backend


def _model_root_artifact(model_path: Path, configured_path: str) -> Path:
    path = Path(str(configured_path)).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    model_root = model_path.expanduser().resolve(strict=False).parent.parent
    return (model_root / path).resolve(strict=False)


def _encoder_path(model_path: Path, policy_config: dict[str, Any]) -> Path:
    configured = policy_config.get("direct_depth_encoder_model_path")
    if not configured:
        raise ValueError("direct_depth_encoder_model_path is required")
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = model_path.expanduser().resolve(strict=False).parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Direct-depth encoder ONNX not found: {path}")
    return path


def _engine_path(model_path: Path, policy_config: dict[str, Any]) -> Path | None:
    if _backend(policy_config) != "tensorrt":
        return None
    configured = policy_config.get("tensorrt_depth_encoder_engine_path")
    if not configured:
        raise ValueError(
            "tensorrt_depth_encoder_engine_path is required when depth backend is TensorRT"
        )
    path = _model_root_artifact(model_path, str(configured))
    if not path.is_file():
        raise FileNotFoundError(f"Direct-depth encoder TensorRT engine not found: {path}")
    return path


def _providers(policy_config: dict[str, Any]) -> list[str]:
    providers = policy_config.get("onnx_providers", ["CPUExecutionProvider"])
    if isinstance(providers, str):
        providers = [providers]
    if not isinstance(providers, list) or not all(isinstance(item, str) for item in providers):
        raise ValueError("policy_config['onnx_providers'] must be a string or list of strings")
    return providers or ["CPUExecutionProvider"]


def _numpy_dtype(node_arg: Any) -> np.dtype[Any]:
    mapping = {
        "tensor(float16)": np.dtype(np.float16),
        "tensor(float)": np.dtype(np.float32),
        "tensor(float32)": np.dtype(np.float32),
    }
    try:
        return mapping[node_arg.type]
    except KeyError as exc:
        raise TypeError(
            f"unsupported encoder input dtype {node_arg.type!r} for {node_arg.name}"
        ) from exc


class DirectDepthEncoderWorker:
    def __init__(
        self,
        *,
        robot_config: dict[str, Any],
        policy_config: dict[str, Any],
        model_path: Path,
        rate_hz: float,
        source_receive_timeout_ms: float,
        source_capture_timeout_ms: float,
        startup_timeout_s: float,
        stats_interval_s: float,
        duration_s: float | None,
        trace_path: Path | None,
        ready_file: Path | None,
    ) -> None:
        for name, value in {
            "rate_hz": rate_hz,
            "source_receive_timeout_ms": source_receive_timeout_ms,
            "source_capture_timeout_ms": source_capture_timeout_ms,
            "startup_timeout_s": startup_timeout_s,
            "stats_interval_s": stats_interval_s,
        }.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if duration_s is not None and (
            not np.isfinite(duration_s) or float(duration_s) <= 0.0
        ):
            raise ValueError("duration_s must be finite and positive")
        for host_key in ("TERRAIN_DEPTH_HOST", "DIRECT_DEPTH_LATENT_HOST"):
            host = str(robot_config.get(host_key, "127.0.0.1")).strip()
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError(
                    f"{host_key} must be loopback because monotonic timestamps are host-local"
                )

        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz
        self.source_receive_timeout_s = float(source_receive_timeout_ms) / 1000.0
        self.source_capture_timeout_s = float(source_capture_timeout_ms) / 1000.0
        self.startup_timeout_s = float(startup_timeout_s)
        self.stats_interval_s = float(stats_interval_s)
        self.duration_s = None if duration_s is None else float(duration_s)
        self.trace_path = trace_path
        self.ready_file = ready_file
        self.exit_requested = False
        self.start_monotonic = time.monotonic()
        self.last_warning_monotonic = 0.0
        self.worker_sequence = 0
        self.generation_epoch = time.monotonic_ns()
        self.last_source_frame_number: int | None = None
        self.had_valid_input = False
        self.input_outage_active = False
        self.input_outages = 0
        self.input_recoveries = 0
        self.input_unavailable_ticks = 0
        self.processed_results = 0
        self.latency_stats = LatencyStats("depth")
        self.duplicate_ticks = 0
        self.source_frames_skipped = 0
        self.send_drops = 0
        self.trace_records: list[tuple[Any, ...]] = []

        self.phase_lock = bool(policy_config.get("depth_policy_phase_lock", False))
        depth_config = DirectDepthConfig.from_policy_config(policy_config)
        self.history = TimestampDepthHistory(depth_config)
        logger.info(f"Encoder rate={self.rate_hz} Hz; capture-time history ages=700..0 ms in 100 ms steps")
        logger.info(
            f"Depth preprocessing: vertical_flip={depth_config.vertical_flip} (height only), "
            f"profile={depth_config.processing_profile} crop_top={depth_config.crop_top} "
            f"output={depth_config.output_height}x{depth_config.output_width}; history order=oldest-to-newest"
        )
        self.state_processor = StateProcessor(
            robot_config,
            policy_config["isaac_joint_names"],
            subscribe_terrain_depth=True,
            subscribe_direct_depth_latent=False,
        )
        backend = _backend(policy_config)
        encoder_path = _encoder_path(model_path, policy_config)
        engine_path = _engine_path(model_path, policy_config)
        logger.info(
            f"Loading direct-depth encoder {backend.upper()} at {self.rate_hz:.1f} Hz"
        )
        self.session = create_inference_session(
            str(encoder_path),
            backend=backend,
            engine_path=str(engine_path) if engine_path is not None else None,
            providers=_providers(policy_config),
            intra_op_num_threads=policy_config.get(
                "onnx_depth_intra_op_num_threads",
                policy_config.get("onnx_intra_op_num_threads"),
            ),
            inter_op_num_threads=int(policy_config.get("onnx_inter_op_num_threads", 1)),
            allow_spinning=bool(policy_config.get("onnx_allow_spinning", False)),
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "depth_image":
            raise RuntimeError("direct-depth encoder must have one input named depth_image")
        if len(outputs) != 1 or outputs[0].name != "depth_latent":
            raise RuntimeError("direct-depth encoder must have one output named depth_latent")
        expected_shape = [1, depth_config.output_frames, depth_config.output_height, depth_config.output_width]
        if list(inputs[0].shape) != expected_shape:
            raise RuntimeError(f"Encoder shape {inputs[0].shape} differs from preprocessing {expected_shape}")
        self.input_dtype = _numpy_dtype(inputs[0])
        self.output_name = outputs[0].name

        latent_port = int(robot_config["DIRECT_DEPTH_LATENT_PORT"])
        if not 1 <= latent_port <= 65535:
            raise ValueError("DIRECT_DEPTH_LATENT_PORT must be in [1, 65535]")
        self.zmq_context = zmq.Context.instance()
        self.latent_socket: zmq.Socket = self.zmq_context.socket(zmq.PUB)
        self.latent_socket.setsockopt(zmq.SNDHWM, 1)
        self.latent_socket.setsockopt(zmq.LINGER, 0)
        self.latent_socket.bind(f"tcp://*:{latent_port}")
        self.history_socket: zmq.Socket = self.zmq_context.socket(zmq.PUB)
        self.history_socket.setsockopt(zmq.SNDHWM, 1)
        self.history_socket.setsockopt(zmq.LINGER, 0)
        self.history_socket.bind("tcp://*:5597")
        logger.info(f"Direct-depth latent publisher bound to tcp://*:{latent_port}; history debug bound to tcp://*:5597")
        logger.warning(
            "NO-ACTUATION: encoder worker does not import g1_interface or publish motor commands"
        )

    def _request_exit(self, _signum, _frame) -> None:
        self.exit_requested = True

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._request_exit)

    def _warn_throttled(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_warning_monotonic >= 1.0:
            logger.warning(message)
            self.last_warning_monotonic = now

    def _input_failure(self, reason: str) -> bool:
        if self.had_valid_input:
            self.input_unavailable_ticks += 1
            if not self.input_outage_active:
                self.input_outage_active = True
                self.input_outages += 1
                self.history.reset()
                self.last_source_frame_number = None
                self.generation_epoch = time.monotonic_ns()
            self._warn_throttled(
                "Direct-depth encoder input temporarily unavailable; no latent "
                f"published and history reset: {reason}"
            )
            return True
        startup_age = time.monotonic() - self.start_monotonic
        if startup_age > self.startup_timeout_s:
            logger.error(
                f"Direct-depth encoder startup timeout after {startup_age * 1000.0:.0f} ms: {reason}"
            )
            return False
        self._warn_throttled(f"Waiting for direct-depth input: {reason}")
        return True

    def _input_recovered(self) -> None:
        if not self.input_outage_active:
            return
        self.input_outage_active = False
        self.input_recoveries += 1
        logger.info("Direct-depth encoder input recovered; history restarted")

    def _write_trace(self) -> None:
        if self.trace_path is None or not self.trace_records:
            return
        if not self.trace_path.parent.is_dir():
            logger.error(f"Encoder timing trace parent does not exist: {self.trace_path.parent}")
            return
        header = (
            "worker_tick",
            "host_monotonic_s",
            "source_frame_number",
            "worker_sequence",
            "reset_generation",
            "capture_age_ms",
            "camera_delivery_ms",
            "producer_to_worker_ms",
            "preprocess_ms",
            "encoder_ms",
            "worker_total_ms",
            "deadline_miss",
            "duplicate_frame",
        )
        with self.trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(self.trace_records)
        logger.info(f"Wrote {len(self.trace_records)} encoder timing ticks to {self.trace_path}")

    def run(self) -> int:
        self._install_signal_handlers()
        if self.ready_file is not None:
            self.ready_file.parent.mkdir(parents=True, exist_ok=True)
            self.ready_file.write_text("ready\n", encoding="utf-8")
        next_tick = next_phase_tick(time.monotonic(), self.dt) if self.phase_lock else time.monotonic()
        last_stats = next_tick
        ticks = 0
        try:
            while not self.exit_requested:
                now = time.monotonic()
                if self.duration_s is not None and now - self.start_monotonic >= self.duration_s:
                    break
                if now < next_tick:
                    time.sleep(min(next_tick - now, 0.002))
                    continue
                if self.phase_lock and now >= next_tick + self.dt:
                    missed = int((now - next_tick) / self.dt)
                    next_tick += missed * self.dt
                    self.history.reset()
                    logger.warning(f"Depth schedule skipped {missed} ticks; history reset")
                tick_start_monotonic = time.monotonic()
                tick_start_perf = time.perf_counter()
                deadline = tick_start_monotonic + self.dt
                next_tick += self.dt
                ticks += 1

                try:
                    frame = self.state_processor.get_latest_terrain_depth(
                        max_receive_age_s=self.source_receive_timeout_s,
                        max_capture_age_s=self.source_capture_timeout_s,
                    )
                except Exception as exc:
                    if not self._input_failure(str(exc)):
                        return 2
                    continue
                self.had_valid_input = True
                self._input_recovered()

                reset_history = bool(frame.reset_history)
                duplicate_frame = False
                if self.last_source_frame_number is not None:
                    if int(frame.frame_number) == self.last_source_frame_number:
                        duplicate_frame = True
                        self.duplicate_ticks += 1
                    elif int(frame.frame_number) < self.last_source_frame_number:
                        reset_history = True
                        self.generation_epoch = time.monotonic_ns()
                    elif int(frame.frame_number) > self.last_source_frame_number + 1:
                        self.source_frames_skipped += (
                            int(frame.frame_number) - self.last_source_frame_number - 1
                        )
                self.last_source_frame_number = int(frame.frame_number)

                worker_start_monotonic_s = time.monotonic()
                preprocess_started = time.perf_counter()
                depth_image = self.history.append(
                    frame.depth_z,
                    timestamp_s=frame.capture_monotonic_s,
                    reset=reset_history,
                ).astype(np.float32) * (1.0 / 255.0)
                preprocess_s = time.perf_counter() - preprocess_started
                try:
                    history_packet = struct.pack("<4sIIQd", b"DHST", int(depth_image.shape[-1]), int(depth_image.shape[-2]), int(frame.frame_number), float(frame.timestamp_s)) + (np.clip(depth_image * 255.0, 0, 255).astype(np.uint8, copy=False)).tobytes()
                    self.history_socket.send(history_packet, flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
                encoder_started = time.perf_counter()
                latent = self.session.run(
                    [self.output_name],
                    {"depth_image": depth_image[None, ...].astype(self.input_dtype, copy=False)},
                )[0]
                encoder_s = time.perf_counter() - encoder_started
                completion_monotonic_s = time.monotonic()
                worker_total_s = time.perf_counter() - tick_start_perf

                message = DirectDepthLatentMessage(
                    depth_latent=np.asarray(latent, dtype=np.float32).reshape(-1),
                    source_frame_number=int(frame.frame_number),
                    worker_sequence=self.worker_sequence,
                    reset_generation=self.generation_epoch,
                    source_timestamp_s=float(frame.timestamp_s),
                    capture_monotonic_s=float(frame.capture_monotonic_s),
                    camera_receive_monotonic_s=float(frame.camera_receive_monotonic_s),
                    sync_ready_monotonic_s=float(frame.sync_ready_monotonic_s),
                    producer_publish_monotonic_s=float(frame.producer_publish_monotonic_s),
                    worker_start_monotonic_s=worker_start_monotonic_s,
                    completion_monotonic_s=completion_monotonic_s,
                    preprocess_s=preprocess_s,
                    encoder_s=encoder_s,
                    worker_total_s=worker_total_s,
                )
                self.latency_stats.add(
                    camera_processing=float(frame.producer_publish_monotonic_s)-float(frame.camera_receive_monotonic_s),
                    producer_to_worker=worker_start_monotonic_s-float(frame.producer_publish_monotonic_s),
                    preprocess=preprocess_s, encoder=encoder_s, worker=worker_total_s,
                    schedule_lag=max(0., tick_start_monotonic-(next_tick-self.dt)),
                    capture_age=completion_monotonic_s-float(frame.capture_monotonic_s))
                try:
                    self.latent_socket.send(message.to_bytes(), flags=zmq.NOBLOCK)
                    self.processed_results += 1
                    self.worker_sequence += 1
                except zmq.Again:
                    self.send_drops += 1

                if self.trace_path is not None:
                    self.trace_records.append(
                        (
                            ticks,
                            completion_monotonic_s,
                            int(frame.frame_number),
                            int(self.worker_sequence),
                            int(self.generation_epoch),
                            (completion_monotonic_s - float(frame.capture_monotonic_s)) * 1000.0,
                            (
                                float(frame.camera_receive_monotonic_s)
                                - float(frame.capture_monotonic_s)
                            )
                            * 1000.0,
                            (
                                worker_start_monotonic_s
                                - float(frame.producer_publish_monotonic_s)
                            )
                            * 1000.0,
                            preprocess_s * 1000.0,
                            encoder_s * 1000.0,
                            worker_total_s * 1000.0,
                            int(completion_monotonic_s > deadline),
                            int(duplicate_frame),
                        )
                    )

                if completion_monotonic_s - last_stats >= self.stats_interval_s:
                    elapsed = completion_monotonic_s - self.start_monotonic
                    logger.info(
                        f"encoder_tick_rate={ticks / elapsed:.2f} Hz "
                        f"latent_rate={self.processed_results / elapsed:.2f} Hz "
                        f"duplicates={self.duplicate_ticks} "
                        f"source_frames_skipped={self.source_frames_skipped} "
                        f"send_drops={self.send_drops} "
                        f"input_outages={self.input_outages} "
                        f"input_recoveries={self.input_recoveries} "
                        f"input_unavailable_ticks={self.input_unavailable_ticks}"
                    )
                    last_stats = completion_monotonic_s
        finally:
            self._write_trace()
            if self.ready_file is not None:
                try:
                    self.ready_file.unlink()
                except FileNotFoundError:
                    pass
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Configurable-rate direct-depth encoder worker")
    parser.add_argument("--robot-config", default="config/robot/g1_real.yaml")
    parser.add_argument("--policy-config", default="model/g1_policy/policy.yaml")
    parser.add_argument("--model-path", default="model/g1_policy/onnx/actor.onnx")
    parser.add_argument("--rate-hz", type=float, default=None, help="Override depth_encoder_rate_hz from policy YAML")
    parser.add_argument("--source-receive-timeout-ms", type=float, default=100.0)
    parser.add_argument("--source-capture-timeout-ms", type=float, default=120.0)
    parser.add_argument("--startup-timeout-s", type=float, default=5.0)
    parser.add_argument("--stats-interval-s", type=float, default=5.0)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--timing-trace", default="")
    parser.add_argument("--ready-file", default="")
    args = parser.parse_args()

    robot_config = load_yaml_config(Path(args.robot_config))
    policy_config = load_yaml_config(Path(args.policy_config))
    worker = DirectDepthEncoderWorker(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=Path(args.model_path).expanduser(),
        rate_hz=resolve_encoder_rate(policy_config, args.rate_hz),
        source_receive_timeout_ms=float(args.source_receive_timeout_ms),
        source_capture_timeout_ms=float(args.source_capture_timeout_ms),
        startup_timeout_s=float(args.startup_timeout_s),
        stats_interval_s=float(args.stats_interval_s),
        duration_s=None if float(args.duration_s) <= 0.0 else float(args.duration_s),
        trace_path=Path(args.timing_trace).expanduser() if args.timing_trace else None,
        ready_file=Path(args.ready_file).expanduser() if args.ready_file else None,
    )
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main())
