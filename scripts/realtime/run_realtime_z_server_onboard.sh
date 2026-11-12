#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[run_realtime_z_server_onboard] $*"
}

fail() {
  echo "[run_realtime_z_server_onboard] ERROR: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_UFO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

UFO_ROOT="${UFO_ROOT:-${DEFAULT_UFO_ROOT}}"
Z_PY="${Z_PY:-python3}"
TELEOP_SITE_PACKAGES="${TELEOP_SITE_PACKAGES:-${UFO_ROOT}/.venv/lib/python3.10/site-packages}"
TELEOP_LIBRARY_PATH="${TELEOP_LIBRARY_PATH:-${UFO_ROOT}/.local/lib}"
MODEL_DIR="${MODEL_DIR:-${UFO_ROOT}/model/g1_policy}"
BACKWARD_ONNX="${BACKWARD_ONNX:-${MODEL_DIR}/onnx/backward_encoder.onnx}"
BACKWARD_ONNX_SHA256="${BACKWARD_ONNX_SHA256:-135127a36ec87e2046530a569a65fe5a245a9e1c7872db310002b316d56bce43}"
MUJOCO_XML="${MUJOCO_XML:-${UFO_ROOT}/data/robots/g1/g1_29dof_pbfm_training.xml}"
DEVICE="${DEVICE:-cpu}"
ENABLE_PICO_Z_CONTROL="${ENABLE_PICO_Z_CONTROL:-0}"
PUBLISH_QV_VELOCITY="${PUBLISH_QV_VELOCITY:-0}"
INITIAL_MODE="${INITIAL_MODE:-follow}"
POSE_TRANSITION_MS="${POSE_TRANSITION_MS:-300}"
if [[ -n "${RESUME_RAMP_MS+x}" || -n "${PAUSE_RAMP_MS+x}" ]]; then
  fail 'RESUME_RAMP_MS/PAUSE_RAMP_MS were removed; use POSE_TRANSITION_MS (default 300).'
fi

[[ -d "${UFO_ROOT}" ]] || fail "UFO_ROOT does not exist: ${UFO_ROOT}"
[[ -f "${UFO_ROOT}/scripts/realtime/realtime_z_server.py" ]] || \
  fail "missing realtime z server under UFO_ROOT: ${UFO_ROOT}"

if [[ -d "${TELEOP_SITE_PACKAGES}" ]]; then
  export PYTHONPATH="${TELEOP_SITE_PACKAGES}:${UFO_ROOT}:${PYTHONPATH:-}"
elif [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${UFO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${UFO_ROOT}"
fi
if [[ -d "${TELEOP_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${TELEOP_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

python_has_realtime_deps() {
  local python_bin="$1"

  if [[ ! -x "${python_bin}" ]] && ! command -v "${python_bin}" >/dev/null 2>&1; then
    return 1
  fi

  "${python_bin}" - <<'PY' >/dev/null 2>&1
import mujoco
import numpy
import onnxruntime
import zmq
PY
}

if ! command -v "${Z_PY}" >/dev/null 2>&1 && [[ ! -x "${Z_PY}" ]]; then
  fail "python not found or not executable: ${Z_PY}"
fi
if ! python_has_realtime_deps "${Z_PY}"; then
  "${Z_PY}" - <<'PY' || true
import mujoco
import numpy
import onnxruntime
import zmq
print("realtime z deps ok")
PY
  fail "selected Python failed realtime z dependency probe: ${Z_PY}"
fi
for arg in "$@"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    cd "${UFO_ROOT}"
    exec "${Z_PY}" scripts/realtime/realtime_z_server.py "$@"
  fi
done
if [[ ! -f "${BACKWARD_ONNX}" ]]; then
  fail "missing backward ONNX: ${BACKWARD_ONNX}"
fi
if [[ -n "${BACKWARD_ONNX_SHA256}" ]]; then
  actual_backward_sha256="$(sha256sum "${BACKWARD_ONNX}" | awk '{print $1}')"
  [[ "${actual_backward_sha256}" == "${BACKWARD_ONNX_SHA256}" ]] || \
    fail "backward ONNX hash mismatch: expected ${BACKWARD_ONNX_SHA256}, got ${actual_backward_sha256}"
fi
if [[ ! -f "${MUJOCO_XML}" ]]; then
  fail "missing MuJoCo XML: ${MUJOCO_XML}"
fi

cd "${UFO_ROOT}"

log "repo: ${UFO_ROOT}"
log "python: ${Z_PY}"
log "backward ONNX: ${BACKWARD_ONNX} (${BACKWARD_ONNX_SHA256})"
log "training-shape MuJoCo XML: ${MUJOCO_XML}"

cmd=(
  "${Z_PY}" scripts/realtime/realtime_z_server.py
  --teleop_req tcp://127.0.0.1:28701
  --teleop_rep tcp://127.0.0.1:28702
  --teleop_ctrl tcp://127.0.0.1:28703
  --z_bind tcp://*:28711
  --hz 50
  --backward_onnx "${BACKWARD_ONNX}"
  --mujoco_xml "${MUJOCO_XML}"
  --device "${DEVICE}"
  --root_height_obs
  --wall-clock-dt
  --no-freeze-static-pose
  --fix-quat-continuity
  --angvel-delta-frame world
  --max-retarget-age-ms "${MAX_RETARGET_AGE_MS:-200}"
  --max-z-delta "${MAX_Z_DELTA:-0}"
  --initial-mode "${INITIAL_MODE}"
  --pose-transition-ms "${POSE_TRANSITION_MS}"
)

if [[ "${ENABLE_PICO_Z_CONTROL}" == "1" ]]; then
  cmd+=(--enable-pico-control)
fi
if [[ "${PUBLISH_QV_VELOCITY}" == "1" ]]; then
  log "QV velocity context: enabled"
  cmd+=(--publish-qv-velocity)
fi

if [[ "${TELEOP_ZERO_LATERAL_ROOT_MOTION:-0}" == "1" ]]; then
  cmd+=(--zero-lateral-root-motion)
fi
exec "${cmd[@]}" "$@"
