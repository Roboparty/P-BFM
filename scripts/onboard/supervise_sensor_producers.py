#!/usr/bin/env python3
"""Keep G1 state alive while independently restarting a camera-only producer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


class SensorSupervisor:
    def __init__(self, state_command, camera_command, *, ready_file,
                 restart_backoff_s=0.75, max_restarts=0, startup_timeout_s=15.0):
        self.state_command = state_command
        self.camera_command = camera_command
        self.ready_file = Path(ready_file)
        self.restart_backoff_s = restart_backoff_s
        self.max_restarts = max_restarts
        self.startup_timeout_s = startup_timeout_s
        self.stop = threading.Event()
        self.state = None
        self.camera = None

    def close(self):
        # Signal both children before waiting. Never kill unrelated processes.
        children = [child for child in (self.camera, self.state) if child is not None]
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3.0)

    def run(self):
        handlers = {
            sig: signal.signal(sig, lambda _signum, _frame: self.stop.set())
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            self.state = subprocess.Popen(self.state_command, cwd=str(ROOT))
            print(f"[INFO] independent state process pid={self.state.pid}", flush=True)
            deadline = time.monotonic() + self.startup_timeout_s
            while not self.ready_file.is_file():
                if self.stop.wait(0.05):
                    return 0
                if self.state.poll() is not None:
                    raise RuntimeError(f"state publisher failed at startup: exit={self.state.returncode}")
                if time.monotonic() >= deadline:
                    raise RuntimeError("state publisher readiness timeout")
            failures = 0
            next_camera_start = time.monotonic()
            while not self.stop.is_set():
                if self.state.poll() is not None:
                    # Do not keep a camera-only deployment alive without state.
                    raise RuntimeError(f"state publisher exited ({self.state.returncode}); stopping sensor group")
                if self.camera is None and time.monotonic() >= next_camera_start:
                    self.camera = subprocess.Popen(self.camera_command, cwd=str(ROOT))
                    print(f"[INFO] camera-only process pid={self.camera.pid} attempt={failures + 1}; state pid={self.state.pid} unchanged", flush=True)
                if self.camera is not None and self.camera.poll() is not None:
                    status = self.camera.returncode
                    if status == 0:
                        return 0
                    failures += 1
                    if self.max_restarts and failures >= self.max_restarts:
                        raise RuntimeError(f"camera restart limit reached ({failures}), last exit={status}")
                    print(f"[WARN] camera exited ({status}); restart in {self.restart_backoff_s:.2f}s; independent state continues", flush=True)
                    self.camera = None
                    next_camera_start = time.monotonic() + self.restart_backoff_s
                self.stop.wait(0.05)
            return 0
        except (OSError, RuntimeError) as exc:
            print(f"[FAIL] sensor supervisor: {exc}", flush=True)
            return 1
        finally:
            self.close()
            for sig, previous in handlers.items():
                signal.signal(sig, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", default="config/robot/g1_real.yaml")
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--depth-fps", type=int, default=60)
    parser.add_argument("--restart-backoff-s", type=float, default=0.75)
    parser.add_argument("--max-restarts", type=int, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.restart_backoff_s) or args.restart_backoff_s <= 0:
        parser.error("restart backoff must be finite and positive")
    if args.max_restarts < 0 or args.depth_fps <= 0:
        parser.error("max-restarts must be nonnegative and depth-fps positive")
    with tempfile.TemporaryDirectory(prefix="pbfm-state-ready-") as temporary:
        ready_file = Path(temporary) / "ready"
        state_command = [
            sys.executable, "-u", str(ROOT / "scripts/onboard/g1_state_publisher.py"),
            "--robot-config", args.robot_config, "--network-interface", args.network_interface,
            "--state-backend", "sdk2", "--ready-file", str(ready_file),
        ]
        camera_command = [
            sys.executable, "-u", str(ROOT / "scripts/onboard/realsense_g1_terrain_producer.py"),
            "--robot-config", args.robot_config, "--camera-serial", args.camera_serial,
            "--depth-fps", str(args.depth_fps), "--state-backend", "external",
            "--camera-stall-action", "exit",
        ]
        return SensorSupervisor(state_command, camera_command, ready_file=ready_file,
                                restart_backoff_s=args.restart_backoff_s,
                                max_restarts=args.max_restarts).run()


if __name__ == "__main__":
    raise SystemExit(main())
