#!/usr/bin/env bash
# Build target-specific strict-FP32 direct-depth Actor engine on the G1 Orin.
# This script has no robot SDK imports and cannot publish motor commands.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT}/model/g1_policy}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_ROOT}/tensorrt}"

ACTOR_ONNX="${ACTOR_ONNX:-${MODEL_ROOT}/onnx/actor.onnx}"
ACTOR_LOWERED="${ACTOR_LOWERED:-${OUTPUT_DIR}/actor_trt85.onnx}"
ACTOR_ENGINE="${ACTOR_ENGINE:-${OUTPUT_DIR}/actor_fp32.engine}"

for path in "${ACTOR_ONNX}" "${TRTEXEC}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required file: ${path}" >&2
    exit 1
  fi
done
mkdir -p "${OUTPUT_DIR}"

if [[ "${SKIP_LOWERING:-0}" == "1" ]]; then
  for path in "${ACTOR_LOWERED}"; do
    if [[ ! -f "${path}" ]]; then
      echo "missing lowered ONNX with SKIP_LOWERING=1: ${path}" >&2
      exit 1
    fi
  done
else
  "${PYTHON_BIN}" "${ROOT}/scripts/onboard/lower_onnx_for_tensorrt85.py" \
    "${ACTOR_ONNX}" "${ACTOR_LOWERED}"
fi

ACTOR_TRT_ARGS=(
  --onnx="${ACTOR_LOWERED}"
  --saveEngine="${ACTOR_ENGINE}"
  --noTF32
  --memPoolSize=workspace:2048
  --buildOnly
)

if [[ -n "${ACTOR_SHAPES:-}" ]]; then
  ACTOR_TRT_ARGS+=(
    --minShapes="${ACTOR_SHAPES}"
    --optShapes="${ACTOR_SHAPES}"
    --maxShapes="${ACTOR_SHAPES}"
  )
fi

"${TRTEXEC}" "${ACTOR_TRT_ARGS[@]}"

if [[ "${PRINT_ACTOR_SHAPES:-0}" == "1" ]]; then
  "${PYTHON_BIN}" - "${ACTOR_ONNX}" <<'PY'
import sys
import onnx

model = onnx.load(sys.argv[1])
shapes = []
for value in model.graph.input:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        else:
            raise SystemExit(f"dynamic actor input is unsupported: {value.name}")
    shapes.append(f"{value.name}:{'x'.join(dims)}")
print(",".join(shapes))
PY
fi

sha256sum "${ACTOR_ENGINE}"
echo "TensorRT Actor engine built. Depth Encoder uses ONNX Runtime on CPU."
