#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
policy_override="${POLICY_CONFIG:-}"
source "${ROOT}/scripts/deploy_env.sh"
cd "${ROOT}"
command="${1:---help}"
if (( $# )); then shift; fi
case "${command}" in
  --help|-h)
    cat <<'HELP'
P-BFM G1 Deployment
  bash run.sh verify                  Offline model/config/context verification
  bash run.sh list                    Available tasks
  bash run.sh camera                  Camera + read-only low-state services
  bash run.sh encoder [--rate-hz 50]   CPU depth encoder, separate foreground process
  bash run.sh depth                   Depth stages webpage, default port 8090
  bash run.sh retarget                Pico retarget bridge (SDK service required)
  bash run.sh z                       Real-time backward encoder/context publisher
  bash run.sh build                   Build + check the TensorRT Actor on this Jetson
  bash run.sh real TASK               Motor policy; requires UFO_REAL_ROBOT_OK=1
Use separate terminals for camera, encoder, depth, and (for teleop) retarget/z.
Set G1_INTERFACE to the robot's low-level Ethernet interface before hardware use.
HELP
    ;;
  list)
    "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
for task in json.loads(Path('model/g1_policy/tasks.json').read_text()):
    print(task['task'])
PY
    ;;
  verify) exec "${PYTHON_BIN}" scripts/verify_release.py "$@" ;;
  camera)
    : "${G1_INTERFACE:?Set G1_INTERFACE to the low-level robot interface}"
    exec bash scripts/onboard/run_realsense_g1_terrain_producer_supervised.sh "$@"
    ;;
  encoder)
    exec "${PYTHON_BIN}" -u scripts/onboard/direct_depth_encoder_worker.py \
      --robot-config "${ROBOT_CONFIG}" --policy-config "${POLICY_CONFIG}" --model-path "${MODEL_PATH}" "$@"
    ;;
  depth) exec "${PYTHON_BIN}" scripts/vis/depth_stages_web.py --policy-config "${POLICY_CONFIG}" "$@" ;;
  retarget) exec bash scripts/teleop/teleop_pose_50hz_onboard.sh "$@" ;;
  z) exec bash scripts/realtime/run_realtime_z_server_onboard.sh "$@" ;;
  build)
    export MODEL_ROOT="${MODEL_DIR}" ACTOR_ONNX="${MODEL_PATH}"
    export ACTOR_ENGINE
    ACTOR_ENGINE="$("${PYTHON_BIN}" scripts/verify_release.py --print-engine-path)"
    export OUTPUT_DIR="$(dirname -- "${ACTOR_ENGINE}")"
    bash scripts/onboard/build_tensorrt_engines.sh "$@"
    exec "${PYTHON_BIN}" scripts/verify_release.py --engine "${ACTOR_ENGINE}"
    ;;
  real)
    : "${G1_INTERFACE:?Set G1_INTERFACE to the low-level robot interface}"
    task="${1:?Choose a task shown by bash run.sh list}"
    shift
    (( $# == 0 )) || { echo 'real accepts only TASK; set POLICY_CONFIG / ROBOT_CONFIG before launch.' >&2; exit 2; }
    if [[ ! "${task}" =~ ^[a-zA-Z0-9_.-]+$ || ! -f "config/tasks/${task}.yaml" ]]; then
      echo "Unknown task: ${task}" >&2; exit 2
    fi
    [[ "${UFO_REAL_ROBOT_OK:-}" == 1 ]] || { echo 'Set UFO_REAL_ROBOT_OK=1 to explicitly enable motor control.' >&2; exit 2; }
    export TASK_CONFIG="${ROOT}/config/tasks/${task}.yaml"
    if [[ "${task}" == teleop ]]; then
      export POLICY_CONFIG="${policy_override:-${MODEL_DIR}/teleop_policy.yaml}" ENABLE_PICO_POLICY_CONTROL=1 PICO_ONLY_CONTROL=1
    else
      export ENABLE_PICO_POLICY_CONTROL=0 PICO_ONLY_CONTROL=0
    fi
    exec flock -n /tmp/pbfm-g1-real-policy.lock bash run_g1_qv_policy_onboard.sh
    ;;
  *) echo "Unknown command: ${command}; use bash run.sh --help" >&2; exit 2 ;;
esac
