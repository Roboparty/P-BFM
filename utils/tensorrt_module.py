"""Small TensorRT 8.x/10.x session adapter with an ONNX Runtime-like interface.

The onboard Jetson image ships TensorRT and ``libcudart`` but not PyCUDA.
This module therefore owns only the small amount of CUDA runtime plumbing
needed for fixed-shape inference.  TensorRT is imported lazily so workstation
Sim2Sim keeps working without any NVIDIA Python packages installed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TensorMetadata:
    """Subset of ONNX Runtime's NodeArg API used by the deploy runtime."""

    name: str
    shape: tuple[int, ...]
    type: str


class _CudaRuntime:
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        candidates = [
            ctypes.util.find_library("cudart"),
            "libcudart.so",
            "libcudart.so.11.0",
        ]
        self.lib = None
        errors: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.lib = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
        if self.lib is None:
            raise RuntimeError(
                "Could not load the CUDA runtime library (tried: "
                + "; ".join(errors)
                + ")"
            )

        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int

    def _check(self, code: int, operation: str) -> None:
        if int(code) == 0:
            return
        raw = self.lib.cudaGetErrorString(int(code))
        detail = raw.decode("utf-8", errors="replace") if raw else "unknown CUDA error"
        raise RuntimeError(f"{operation} failed: CUDA error {code}: {detail}")

    def malloc(self, size_bytes: int) -> int:
        if int(size_bytes) <= 0:
            raise ValueError("CUDA allocation size must be positive")
        pointer = ctypes.c_void_p()
        self._check(
            self.lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(int(size_bytes))),
            "cudaMalloc",
        )
        if pointer.value is None:
            raise RuntimeError("cudaMalloc returned a null pointer")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        if pointer:
            self._check(self.lib.cudaFree(ctypes.c_void_p(int(pointer))), "cudaFree")

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        if stream.value is None:
            raise RuntimeError("cudaStreamCreate returned a null stream")
        return int(stream.value)

    def destroy_stream(self, stream: int) -> None:
        if stream:
            self._check(
                self.lib.cudaStreamDestroy(ctypes.c_void_p(int(stream))),
                "cudaStreamDestroy",
            )

    def copy_async(
        self,
        destination: int,
        source: int,
        size_bytes: int,
        direction: int,
        stream: int,
    ) -> None:
        self._check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(int(destination)),
                ctypes.c_void_p(int(source)),
                ctypes.c_size_t(int(size_bytes)),
                ctypes.c_int(int(direction)),
                ctypes.c_void_p(int(stream)),
            ),
            "cudaMemcpyAsync",
        )

    def synchronize(self, stream: int) -> None:
        self._check(
            self.lib.cudaStreamSynchronize(ctypes.c_void_p(int(stream))),
            "cudaStreamSynchronize",
        )


def _ort_type(dtype: np.dtype[Any]) -> str:
    mapping = {
        np.dtype(np.float16): "tensor(float16)",
        np.dtype(np.float32): "tensor(float)",
        np.dtype(np.float64): "tensor(double)",
        np.dtype(np.int8): "tensor(int8)",
        np.dtype(np.int32): "tensor(int32)",
        np.dtype(np.int64): "tensor(int64)",
        np.dtype(np.bool_): "tensor(bool)",
    }
    return mapping.get(np.dtype(dtype), f"tensor({np.dtype(dtype).name})")


def _numpy_dtype_from_tensorrt(trt: Any, dtype: Any) -> np.dtype[Any]:
    """Map TRT 8.x dtypes without its deprecated ``np.bool`` helper."""
    mapping = {
        trt.float32: np.dtype(np.float32),
        trt.float16: np.dtype(np.float16),
        trt.int8: np.dtype(np.int8),
        trt.int32: np.dtype(np.int32),
        trt.bool: np.dtype(np.bool_),
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT binding dtype: {dtype}")
    return mapping[dtype]


class TensorRTSession:
    """Deserialize one engine and expose ``get_inputs/get_outputs/run``.

    The deployment engines use one fixed batch-1 optimization profile.  Shape
    changes are supported for diagnostics, but cause safe buffer reallocation.
    """

    def __init__(self, engine_path: str | Path) -> None:
        path = Path(engine_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {path}")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT Python bindings are unavailable; use the onboard Python 3.8 "
                "environment with /usr/lib/python3.8/dist-packages on PYTHONPATH"
            ) from exc

        self._trt = trt
        self._cuda = _CudaRuntime()
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine: {path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"TensorRT could not create execution context: {path}")
        if int(self._engine.num_optimization_profiles) != 1:
            raise RuntimeError(
                "Only one TensorRT optimization profile is supported by the deploy adapter"
            )

        self._named_io = hasattr(self._engine, "num_io_tensors") and not hasattr(self._engine, "num_bindings")
        self._binding_count = int(self._engine.num_io_tensors if self._named_io else self._engine.num_bindings)
        self._binding_names = [str(self._engine.get_tensor_name(i) if self._named_io else self._engine.get_binding_name(i)) for i in range(self._binding_count)]
        self.engine_path = path
        self._stream = self._cuda.create_stream()
        self._device_bindings = [0] * self._binding_count
        self._device_allocations: list[int] = []
        self._host_outputs: dict[str, np.ndarray] = {}
        self._allocated_shapes: tuple[tuple[int, ...], ...] | None = None
        self._active_input_shapes: dict[str, tuple[int, ...]] = {}
        self._input_indices: dict[str, int] = {}
        self._output_indices: dict[str, int] = {}
        self._dtypes: dict[str, np.dtype[Any]] = {}

        for index in range(self._binding_count):
            name = self._binding_names[index]
            if name in self._dtypes:
                raise RuntimeError(f"TensorRT engine contains duplicate binding name: {name}")
            dtype = _numpy_dtype_from_tensorrt(
                trt, self._engine.get_tensor_dtype(name) if self._named_io else self._engine.get_binding_dtype(index)
            )
            self._dtypes[name] = dtype
            is_input = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT if self._named_io else self._engine.binding_is_input(index)
            target = self._input_indices if is_input else self._output_indices
            target[name] = index
        if not self._input_indices or not self._output_indices:
            raise RuntimeError("TensorRT engine must have at least one input and one output")

    def _metadata(self, indices: Mapping[str, int]) -> list[TensorMetadata]:
        result: list[TensorMetadata] = []
        for name, index in sorted(indices.items(), key=lambda item: item[1]):
            shape = tuple(int(value) for value in (self._engine.get_tensor_shape(name) if self._named_io else self._engine.get_binding_shape(index)))
            result.append(TensorMetadata(name=name, shape=shape, type=_ort_type(self._dtypes[name])))
        return result

    def get_inputs(self) -> list[TensorMetadata]:
        return self._metadata(self._input_indices)

    def get_outputs(self) -> list[TensorMetadata]:
        return self._metadata(self._output_indices)

    def _release_allocations(self) -> None:
        allocations, self._device_allocations = self._device_allocations, []
        self._device_bindings = [0] * self._binding_count
        self._host_outputs = {}
        self._allocated_shapes = None
        for pointer in allocations:
            self._cuda.free(pointer)

    def _allocate_for_current_shapes(self) -> None:
        shapes = tuple(
            tuple(int(value) for value in (self._context.get_tensor_shape(self._binding_names[index]) if self._named_io else self._context.get_binding_shape(index)))
            for index in range(self._binding_count)
        )
        if any(not shape or any(value <= 0 for value in shape) for shape in shapes):
            raise RuntimeError(f"TensorRT produced unresolved binding shapes: {shapes}")
        if shapes == self._allocated_shapes:
            return
        self._release_allocations()
        try:
            for index, shape in enumerate(shapes):
                name = self._binding_names[index]
                count = int(np.prod(shape, dtype=np.int64))
                size_bytes = count * int(self._dtypes[name].itemsize)
                pointer = self._cuda.malloc(size_bytes)
                self._device_allocations.append(pointer)
                self._device_bindings[index] = pointer
                if name in self._output_indices:
                    self._host_outputs[name] = np.empty(shape, dtype=self._dtypes[name])
        except Exception:
            self._release_allocations()
            raise
        self._allocated_shapes = shapes

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, np.ndarray],
        _run_options: Any | None = None,
    ) -> list[np.ndarray]:
        del _run_options
        expected_inputs = set(self._input_indices)
        actual_inputs = set(input_feed)
        if actual_inputs != expected_inputs:
            raise ValueError(
                f"TensorRT input mismatch: expected {sorted(expected_inputs)}, got {sorted(actual_inputs)}"
            )
        requested_outputs = list(self._output_indices) if output_names is None else list(output_names)
        unknown_outputs = set(requested_outputs) - set(self._output_indices)
        if unknown_outputs:
            raise ValueError(f"Unknown TensorRT outputs requested: {sorted(unknown_outputs)}")

        contiguous_inputs: dict[str, np.ndarray] = {}
        for name, index in self._input_indices.items():
            array = np.asarray(input_feed[name])
            expected_dtype = self._dtypes[name]
            if array.dtype != expected_dtype:
                raise TypeError(
                    f"TensorRT input {name!r} requires dtype {expected_dtype}, got {array.dtype}"
                )
            array = np.ascontiguousarray(array)
            shape = tuple(int(value) for value in array.shape)
            if self._active_input_shapes.get(name) != shape:
                if not (self._context.set_input_shape(name, shape) if self._named_io else self._context.set_binding_shape(index, shape)):
                    raise ValueError(
                        f"TensorRT rejected shape {array.shape} for input binding {name!r}"
                    )
                self._active_input_shapes[name] = shape
            contiguous_inputs[name] = array
        if not self._named_io and not bool(self._context.all_binding_shapes_specified):
            raise RuntimeError("TensorRT input binding shapes are incomplete")

        self._allocate_for_current_shapes()
        for name, array in contiguous_inputs.items():
            pointer = self._device_bindings[self._input_indices[name]]
            self._cuda.copy_async(
                pointer,
                int(array.ctypes.data),
                int(array.nbytes),
                _CudaRuntime.HOST_TO_DEVICE,
                self._stream,
            )

        if self._named_io:
            for index, name in enumerate(self._binding_names):
                if not self._context.set_tensor_address(name, self._device_bindings[index]):
                    raise RuntimeError(f"TensorRT rejected address for {name}")
            ok = self._context.execute_async_v3(stream_handle=self._stream)
        else:
            ok = self._context.execute_async_v2(
                bindings=self._device_bindings, stream_handle=self._stream
            )
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 returned false")
        for name, array in self._host_outputs.items():
            pointer = self._device_bindings[self._output_indices[name]]
            self._cuda.copy_async(
                int(array.ctypes.data),
                pointer,
                int(array.nbytes),
                _CudaRuntime.DEVICE_TO_HOST,
                self._stream,
            )
        self._cuda.synchronize(self._stream)
        return [self._host_outputs[name].copy() for name in requested_outputs]

    def close(self) -> None:
        if getattr(self, "_cuda", None) is None:
            return
        self._release_allocations()
        stream, self._stream = getattr(self, "_stream", 0), 0
        if stream:
            self._cuda.destroy_stream(stream)
        self._context = None
        self._engine = None
        self._runtime = None
        self._cuda = None

    def __enter__(self) -> "TensorRTSession":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors run during interpreter teardown; explicit close()
            # remains strict and reports CUDA cleanup errors.
            pass
