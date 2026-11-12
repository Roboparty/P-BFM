#!/usr/bin/env bash
# Source from any working directory; native SDK libraries stay in the local environment.
UFO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UFO_ROOT
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export Z_PY="${Z_PY:-${PYTHON_BIN}}" TELEOP_PY="${TELEOP_PY:-${PYTHON_BIN}}"
export PYTHONPATH="${UFO_ROOT}:${PYTHONPATH:-}"
export MODEL_DIR="${UFO_ROOT}/model/g1_policy"
export MODEL_PATH="${MODEL_DIR}/onnx/actor.onnx"
export POLICY_CONFIG="${POLICY_CONFIG:-${MODEL_DIR}/policy.yaml}"
export ROBOT_CONFIG="${ROBOT_CONFIG:-${UFO_ROOT}/config/robot/g1_real.yaml}"
export BACKWARD_ONNX="${MODEL_DIR}/onnx/backward_encoder.onnx"
export BACKWARD_ONNX_SHA256=135127a36ec87e2046530a569a65fe5a245a9e1c7872db310002b316d56bce43
export NETWORK_INTERFACE="${G1_INTERFACE:-${NETWORK_INTERFACE:-eth0}}"
export CAMERA_SERIAL="${CAMERA_SERIAL:-auto}" DEPTH_FPS="${DEPTH_FPS:-60}"
export DEPTH_TRAINING_RAYS=1 DEPTH_PROCESSING_PROFILE=optical_z_maxcrop18_reset_v2
export MAX_Z_DELTA="${MAX_Z_DELTA:-0.75}" POSE_TRANSITION_MS="${POSE_TRANSITION_MS:-300}"
export PUBLISH_QV_VELOCITY=1 TELEOP_ZERO_LATERAL_ROOT_MOTION=0
export TELEOP_POLICY_CONFIG="${UFO_ROOT}/config/policy/g1_policy.yaml"
export PBFM_LATENCY_STATS="${PBFM_LATENCY_STATS:-1}"
