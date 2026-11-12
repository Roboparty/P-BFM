#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROBOT_CONFIG="${ROBOT_CONFIG:-config/robot/g1_real.yaml}"
CAMERA_SERIAL="${CAMERA_SERIAL:-auto}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-eth0}"
DEPTH_FPS="${DEPTH_FPS:-60}"
RESTART_BACKOFF_S="${RESTART_BACKOFF_S:-0.75}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"

TASK_PYTHONPATH_DEFAULT="${PWD}:${PYTHONPATH:-}"
export PYTHONPATH="${TASK_PYTHONPATH:-$TASK_PYTHONPATH_DEFAULT}"

if [[ "${CAMERA_SERIAL}" == "auto" ]]; then
  mapfile -t detected_camera_serials < <(
    "${PYTHON_BIN}" - <<'PY'
import pyrealsense2 as rs
for device in rs.context().query_devices():
    print(device.get_info(rs.camera_info.serial_number))
PY
  )
  if [[ "${#detected_camera_serials[@]}" -ne 1 ]]; then
    echo "[FAIL] expected exactly one RealSense for CAMERA_SERIAL=auto; detected ${#detected_camera_serials[@]}. Set CAMERA_SERIAL explicitly." >&2
    exit 1
  fi
  CAMERA_SERIAL="${detected_camera_serials[0]}"
  echo "[INFO] auto-detected RealSense serial=${CAMERA_SERIAL}"
fi

# One terminal, two independent children. Camera exits must not restart DDS
# state or interrupt low-state publication on port 5590.
exec "${PYTHON_BIN}" -u scripts/onboard/supervise_sensor_producers.py \
  --robot-config "${ROBOT_CONFIG}" \
  --camera-serial "${CAMERA_SERIAL}" \
  --network-interface "${NETWORK_INTERFACE}" \
  --depth-fps "${DEPTH_FPS}" \
  --restart-backoff-s "${RESTART_BACKOFF_S}" \
  --max-restarts "${MAX_RESTARTS}"
