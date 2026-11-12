#!/usr/bin/env python3
"""Independent no-actuation G1 low-state publisher, unaffected by camera restarts.

Reuse the existing SDK2/ROS2 state conversion and controller-clock mapping.
Only this process owns the low-state ZMQ PUB socket. No cached packet is
republished or assigned a new capture timestamp when DDS input disappears.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These helpers have no hardware imports at module scope. In particular,
# importing them does not import pyrealsense2 or create a camera context.
from scripts.onboard.realsense_g1_terrain_producer import (  # noqa: E402
    Ros2StateRuntime, Sdk2StateRuntime, _publisher,
)
from utils.config import load_yaml_config  # noqa: E402


def serve(runtime, stop, *, rate_hz, ready_file=None, duration_s=0.0,
          startup_timeout_s=5.0, stats_interval_s=5.0):
    """Flush only newly received state, independent of any camera event loop."""
    started = time.monotonic()
    next_tick = last_stats = started
    last_count = 0
    ready = False
    period = 1.0 / rate_hz
    while not stop.is_set() and runtime.ok():
        if stop.wait(max(0.0, next_tick - time.monotonic())):
            break
        runtime.executor.spin_once(timeout_sec=0.0)
        now = time.monotonic()
        count = runtime.bridge.low_state_count
        if not ready and count:
            if ready_file is not None:
                Path(ready_file).write_text("ready\n", encoding="utf-8")
            ready = True
            print("[INFO] independent low-state publisher ready; camera lifecycle is separate", flush=True)
        if not ready and now - started >= startup_timeout_s:
            raise RuntimeError("no valid low-state samples before startup timeout")
        if now - last_stats >= stats_interval_s:
            print(
                f"[INFO] independent_lowstate_rate={(count - last_count) / (now - last_stats):.2f}Hz "
                f"samples={count} last_state_error={runtime.bridge.last_error!r}", flush=True,
            )
            last_stats, last_count = now, count
        if duration_s > 0 and now - started >= duration_s:
            break
        next_tick = max(next_tick + period, now)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", default="config/robot/g1_real.yaml")
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--state-backend", choices=("sdk2", "ros2"), default="sdk2")
    parser.add_argument("--low-state-topic", default="/lowstate")
    parser.add_argument("--bind-address", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="Override low-state port for isolated no-actuation diagnostics")
    parser.add_argument("--rate-hz", type=float, default=None)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--startup-timeout-s", type=float, default=5.0)
    parser.add_argument("--stats-interval-s", type=float, default=5.0)
    args, ros_args = parser.parse_known_args()
    if ros_args and args.state_backend != "ros2":
        parser.error(f"unrecognized arguments: {' '.join(ros_args)}")
    if args.bind_address not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("low-state timestamps are host-local; bind address must be loopback")
    path = args.robot_config if Path(args.robot_config).is_absolute() else ROOT / args.robot_config
    config = load_yaml_config(path)
    rate_hz = float(config.get("SDK2_LOW_STATE_RATE_HZ", 200.0) if args.rate_hz is None else args.rate_hz)
    for name, value in (("rate-hz", rate_hz), ("startup-timeout-s", args.startup_timeout_s),
                        ("stats-interval-s", args.stats_interval_s)):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    if not np.isfinite(args.duration_s) or args.duration_s < 0:
        parser.error("--duration-s must be finite and nonnegative")
    port = int(config["LOW_STATE_PORT"] if args.port is None else args.port)
    if not 1 <= port <= 65535:
        parser.error("low-state port must be in [1,65535]")
    interface = args.network_interface or config.get("INTERFACE")
    if args.state_backend == "sdk2" and not interface:
        parser.error("network interface is required for SDK2")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: stop.set())
    context = zmq.Context()
    publisher = runtime = None
    try:
        publisher = _publisher(context, f"tcp://{args.bind_address}:{port}")
        print(f"[INFO] NO-ACTUATION: independent G1 state publisher on {args.bind_address}:{port}, rate={rate_hz:g}Hz", flush=True)
        if args.state_backend == "sdk2":
            runtime = Sdk2StateRuntime(publisher, str(interface), args.low_state_topic, rate_hz)
        else:
            runtime = Ros2StateRuntime(publisher, args.low_state_topic, ros_args)
        return serve(runtime, stop, rate_hz=rate_hz, ready_file=args.ready_file,
                     duration_s=args.duration_s, startup_timeout_s=args.startup_timeout_s,
                     stats_interval_s=args.stats_interval_s)
    except Exception as exc:
        print(f"[FAIL] independent state publisher: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            if publisher is not None:
                publisher.close(linger=0)
            context.term()


if __name__ == "__main__":
    raise SystemExit(main())
