#!/usr/bin/env python3
"""Lower ONNX LayerNormalization for the TensorRT 8.5 parser.

The original artifacts remain immutable. This creates a deployment-only ONNX
whose LayerNormalization nodes are expanded into elementary reductions and
pointwise operations supported by TensorRT 8.5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _attribute(node, name: str, default):
    import onnx

    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def _finite_input_root(tensor_name: str, producers: dict[str, object]) -> str:
    current = tensor_name
    visited = set()
    while current not in visited:
        visited.add(current)
        node = producers.get(current)
        if node is None or node.op_type not in {"Cast", "Identity"} or len(node.input) != 1:
            break
        current = node.input[0]
    return current


def _prune_unused_nodes(model) -> None:
    nodes = list(model.graph.node)
    producer = {output: index for index, node in enumerate(nodes) for output in node.output}
    required_tensors = [output.name for output in model.graph.output]
    required_nodes = set()
    while required_tensors:
        tensor = required_tensors.pop()
        node_index = producer.get(tensor)
        if node_index is None or node_index in required_nodes:
            continue
        required_nodes.add(node_index)
        required_tensors.extend(nodes[node_index].input)
    del model.graph.node[:]
    model.graph.node.extend(
        node for index, node in enumerate(nodes) if index in required_nodes
    )


def _tensor_ranks(model) -> dict[str, int]:
    import onnx

    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    ranks: dict[str, int] = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        shape = value.type.tensor_type.shape
        if not shape.dim:
            continue
        ranks[value.name] = len(shape.dim)
    return ranks


def _constant_tensor_nodes(model) -> dict[str, object]:
    return {
        node.output[0]: node
        for node in model.graph.node
        if node.op_type == "Constant" and len(node.output) == 1
    }


def _constant_node_array(node):
    import onnx

    for attribute in node.attribute:
        if attribute.name == "value":
            return onnx.numpy_helper.to_array(attribute.t)
    return None


def _set_constant_node_array(node, array: np.ndarray) -> None:
    import onnx

    replacement = onnx.helper.make_attribute(
        "value", onnx.numpy_helper.from_array(array)
    )
    for index, attribute in enumerate(node.attribute):
        if attribute.name == "value":
            node.attribute[index].CopyFrom(replacement)
            return
    node.attribute.append(replacement)


def _replace_negative_axis_constants(model) -> int:
    """TensorRT 8.5 rejects negative axes in opset-18 axis tensors."""

    import onnx

    ranks = _tensor_ranks(model)
    initializer_by_name = {item.name: item for item in model.graph.initializer}
    constant_by_output = _constant_tensor_nodes(model)
    changed = 0

    for node in model.graph.node:
        if node.op_type not in {"ReduceMean", "Squeeze", "Unsqueeze"}:
            continue
        if len(node.input) < 2:
            continue
        axes_name = node.input[1]
        initializer = initializer_by_name.get(axes_name)
        constant_node = constant_by_output.get(axes_name)
        if initializer is not None:
            axes = onnx.numpy_helper.to_array(initializer)
        elif constant_node is not None:
            axes = _constant_node_array(constant_node)
        else:
            continue
        if axes is None:
            continue
        axes_arr = np.asarray(axes, dtype=np.int64)
        if axes_arr.size == 0 or not np.any(axes_arr < 0):
            continue
        input_rank = ranks.get(node.input[0])
        if input_rank is None:
            continue
        if node.op_type == "Unsqueeze":
            rank_for_axes = input_rank + int(axes_arr.size)
        else:
            rank_for_axes = input_rank
        positive_axes = axes_arr.copy()
        positive_axes[positive_axes < 0] += rank_for_axes
        if np.any(positive_axes < 0):
            continue
        positive_axes = positive_axes.astype(np.int64)
        if initializer is not None:
            initializer.CopyFrom(onnx.numpy_helper.from_array(positive_axes, axes_name))
        else:
            _set_constant_node_array(constant_node, positive_axes)
        changed += 1
    return changed


def _axis_tensor_array(model, tensor_name: str):
    import onnx

    for initializer in model.graph.initializer:
        if initializer.name == tensor_name:
            return onnx.numpy_helper.to_array(initializer)
    constant_node = _constant_tensor_nodes(model).get(tensor_name)
    if constant_node is not None:
        return _constant_node_array(constant_node)
    return None


def _set_int_attribute(node, name: str, value: int) -> None:
    import onnx

    replacement = onnx.helper.make_attribute(name, int(value))
    for index, attribute in enumerate(node.attribute):
        if attribute.name == name:
            node.attribute[index].CopyFrom(replacement)
            return
    node.attribute.append(replacement)


def _fuse_reduce_mean_unsqueeze_keepdims(model) -> int:
    """Replace ReduceMean(keepdims=0)+Unsqueeze(axis) with keepdims=1.

    PyTorch exports InstanceNorm-like blocks this way. TensorRT 8.5 can be
    brittle around opset-18 Unsqueeze axes, while the fused ReduceMean is
    exactly equivalent when both ops use the same single axis.
    """

    nodes = list(model.graph.node)
    producer = {output: node for node in nodes for output in node.output}
    consumers: dict[str, list[object]] = {}
    for node in nodes:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    replacements: dict[str, str] = {}
    remove_ids: set[int] = set()
    fused = 0
    for node in nodes:
        if node.op_type != "Unsqueeze" or len(node.input) < 2 or len(node.output) != 1:
            continue
        reduce_node = producer.get(node.input[0])
        if reduce_node is None or reduce_node.op_type != "ReduceMean":
            continue
        if len(consumers.get(reduce_node.output[0], [])) != 1:
            continue
        keepdims = int(_attribute(reduce_node, "keepdims", 1))
        if keepdims != 0:
            continue
        reduce_axes = _axis_tensor_array(model, reduce_node.input[1]) if len(reduce_node.input) >= 2 else None
        unsqueeze_axes = _axis_tensor_array(model, node.input[1])
        if reduce_axes is None or unsqueeze_axes is None:
            continue
        reduce_axes = np.asarray(reduce_axes, dtype=np.int64).reshape(-1)
        unsqueeze_axes = np.asarray(unsqueeze_axes, dtype=np.int64).reshape(-1)
        if reduce_axes.size != 1 or unsqueeze_axes.size != 1:
            continue
        if int(reduce_axes[0]) != int(unsqueeze_axes[0]):
            continue
        _set_int_attribute(reduce_node, "keepdims", 1)
        replacements[node.output[0]] = reduce_node.output[0]
        remove_ids.add(id(node))
        fused += 1

    if fused == 0:
        return 0
    for node in nodes:
        if id(node) in remove_ids:
            continue
        for index, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[index] = replacements[input_name]
    del model.graph.node[:]
    model.graph.node.extend(node for node in nodes if id(node) not in remove_ids)
    _prune_unused_nodes(model)
    return fused


def lower_for_tensorrt85(
    source: Path,
    output: Path,
    *,
    finite_inputs: set[str] | None = None,
) -> tuple[int, int]:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(source))
    default_domain_opsets = [
        int(opset.version)
        for opset in model.opset_import
        if opset.domain in {"", "ai.onnx"}
    ]
    opset_version = max(default_domain_opsets) if default_domain_opsets else 13
    finite_inputs = set(finite_inputs or ())
    graph_input_names = {item.name for item in model.graph.input}
    unknown_finite_inputs = finite_inputs - graph_input_names
    if unknown_finite_inputs:
        raise ValueError(
            f"--finite-input names are not graph inputs: {sorted(unknown_finite_inputs)}"
        )
    producers = {output: node for node in model.graph.node for output in node.output}
    lowered_nodes = []
    lowered_count = 0
    finite_check_count = 0
    existing_names = {initializer.name for initializer in model.graph.initializer}

    for node_index, node in enumerate(model.graph.node):
        if node.op_type in {"IsInf", "IsNaN"}:
            if len(node.input) != 1 or len(node.output) != 1:
                raise ValueError(f"{node.op_type} {node.name!r} has unsupported arity")
            root = _finite_input_root(node.input[0], producers)
            if root not in finite_inputs:
                raise ValueError(
                    f"{node.op_type} {node.name!r} traces to {root!r}, which is not "
                    "declared with --finite-input"
                )
            prefix = f"ufo_trt85_finite_{node_index}"
            shape = f"{prefix}_shape"
            false_value = helper.make_tensor(
                f"{prefix}_false",
                TensorProto.BOOL,
                [1],
                [False],
            )
            lowered_nodes.extend(
                (
                    helper.make_node(
                        "Shape", [root], [shape], name=f"{prefix}/Shape"
                    ),
                    helper.make_node(
                        "ConstantOfShape",
                        [shape],
                        [node.output[0]],
                        value=false_value,
                        name=f"{prefix}/AllFalse",
                    ),
                )
            )
            finite_check_count += 1
            continue
        if node.op_type != "LayerNormalization":
            lowered_nodes.append(node)
            continue
        if len(node.input) != 3 or len(node.output) != 1:
            raise ValueError(
                f"LayerNormalization {node.name!r} must have three inputs and one output"
            )
        axis = int(_attribute(node, "axis", -1))
        if axis != -1:
            raise ValueError(
                f"LayerNormalization {node.name!r} axis={axis}; only final-axis lowering is supported"
            )
        epsilon = float(_attribute(node, "epsilon", 1.0e-5))
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"LayerNormalization {node.name!r} has invalid epsilon")

        prefix = f"ufo_trt85_layernorm_{node_index}"
        epsilon_name = f"{prefix}_epsilon"
        if epsilon_name in existing_names:
            raise ValueError(f"generated initializer name already exists: {epsilon_name}")
        existing_names.add(epsilon_name)
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(epsilon, dtype=np.float32), epsilon_name)
        )

        def make_reduce_mean(input_name: str, output_name: str, name: str):
            if opset_version >= 18:
                axes_name = f"{name.replace('/', '_')}_axes"
                if axes_name in existing_names:
                    raise ValueError(
                        f"generated initializer name already exists: {axes_name}"
                    )
                existing_names.add(axes_name)
                model.graph.initializer.append(
                    numpy_helper.from_array(
                        np.asarray([-1], dtype=np.int64), axes_name
                    )
                )
                return helper.make_node(
                    "ReduceMean",
                    [input_name, axes_name],
                    [output_name],
                    keepdims=1,
                    name=name,
                )
            return helper.make_node(
                "ReduceMean",
                [input_name],
                [output_name],
                axes=[-1],
                keepdims=1,
                name=name,
            )

        mean = f"{prefix}_mean"
        centered = f"{prefix}_centered"
        squared = f"{prefix}_squared"
        variance = f"{prefix}_variance"
        variance_epsilon = f"{prefix}_variance_epsilon"
        standard_deviation = f"{prefix}_standard_deviation"
        normalized = f"{prefix}_normalized"
        scaled = f"{prefix}_scaled"
        lowered_nodes.extend(
            (
                make_reduce_mean(node.input[0], mean, f"{prefix}/ReduceMean"),
                helper.make_node(
                    "Sub",
                    [node.input[0], mean],
                    [centered],
                    name=f"{prefix}/Sub",
                ),
                helper.make_node(
                    "Mul",
                    [centered, centered],
                    [squared],
                    name=f"{prefix}/Square",
                ),
                make_reduce_mean(squared, variance, f"{prefix}/Variance"),
                helper.make_node(
                    "Add",
                    [variance, epsilon_name],
                    [variance_epsilon],
                    name=f"{prefix}/AddEpsilon",
                ),
                helper.make_node(
                    "Sqrt",
                    [variance_epsilon],
                    [standard_deviation],
                    name=f"{prefix}/Sqrt",
                ),
                helper.make_node(
                    "Div",
                    [centered, standard_deviation],
                    [normalized],
                    name=f"{prefix}/Normalize",
                ),
                helper.make_node(
                    "Mul",
                    [normalized, node.input[1]],
                    [scaled],
                    name=f"{prefix}/Scale",
                ),
                helper.make_node(
                    "Add",
                    [scaled, node.input[2]],
                    [node.output[0]],
                    name=f"{prefix}/Bias",
                ),
            )
        )
        lowered_count += 1

    del model.graph.node[:]
    model.graph.node.extend(lowered_nodes)
    _prune_unused_nodes(model)
    negative_axis_count = _replace_negative_axis_constants(model)
    reduce_unsqueeze_fused_count = _fuse_reduce_mean_unsqueeze_keepdims(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata["ufo.tensorrt85.layernorm_lowered"] = str(lowered_count)
    metadata["ufo.tensorrt85.finite_checks_lowered"] = str(finite_check_count)
    metadata["ufo.tensorrt85.negative_axes_rewritten"] = str(negative_axis_count)
    metadata["ufo.tensorrt85.reduce_unsqueeze_fused"] = str(
        reduce_unsqueeze_fused_count
    )
    metadata["ufo.tensorrt85.assumed_finite_inputs"] = ",".join(sorted(finite_inputs))
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return lowered_count, finite_check_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--finite-input",
        action="append",
        default=[],
        help=(
            "Graph input guaranteed finite by the runtime. IsInf/IsNaN checks "
            "that trace only through Cast/Identity to this input become all-false."
        ),
    )
    args = parser.parse_args()
    layer_norm_count, finite_check_count = lower_for_tensorrt85(
        args.source.expanduser(),
        args.output.expanduser(),
        finite_inputs=set(args.finite_input),
    )
    print(
        f"[OK] lowered {layer_norm_count} LayerNormalization and "
        f"{finite_check_count} finite-check node(s): {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
