from __future__ import annotations

import onnxruntime as ort
import time
from pathlib import Path
from typing import Any, Dict


def create_onnx_session(
    path: str,
    *,
    providers: list[str],
    intra_op_num_threads: int | None = None,
    inter_op_num_threads: int = 1,
    allow_spinning: bool = False,
) -> ort.InferenceSession:
    """Create a bounded ONNX Runtime session suitable for a realtime loop."""
    options = ort.SessionOptions()
    if intra_op_num_threads is not None:
        if int(intra_op_num_threads) <= 0:
            raise ValueError("intra_op_num_threads must be positive when configured")
        options.intra_op_num_threads = int(intra_op_num_threads)
    if int(inter_op_num_threads) <= 0:
        raise ValueError("inter_op_num_threads must be positive")
    options.inter_op_num_threads = int(inter_op_num_threads)
    if not allow_spinning:
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(path, sess_options=options, providers=providers)


def create_inference_session(
    model_path: str,
    *,
    backend: str = "onnx",
    engine_path: str | None = None,
    providers: list[str] | None = None,
    intra_op_num_threads: int | None = None,
    inter_op_num_threads: int = 1,
    allow_spinning: bool = False,
) -> Any:
    """Create either the portable ONNX session or the onboard TensorRT adapter."""
    normalized = str(backend).strip().lower()
    if normalized == "onnx":
        return create_onnx_session(
            str(Path(model_path).expanduser()),
            providers=providers or ["CPUExecutionProvider"],
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
            allow_spinning=allow_spinning,
        )
    if normalized == "tensorrt":
        if not engine_path:
            raise ValueError("engine_path is required for the TensorRT inference backend")
        from utils.tensorrt_module import TensorRTSession

        return TensorRTSession(engine_path)
    raise ValueError(f"Unsupported inference backend: {backend!r}")


class Timer:
    def __init__(self, perf_dict: Dict[str, float], name: str):
        self.perf_dict = perf_dict
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.perf_counter() - self.start_time
        if self.name not in self.perf_dict:
            self.perf_dict[self.name] = 0
        self.perf_dict[self.name] += elapsed_time
