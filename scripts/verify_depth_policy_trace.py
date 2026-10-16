"""Verify exported actions/estimates on recorded real rollout observations.

Also report pooled (not mean-of-batch) estimator statistics. This is interface
verification and a descriptive small-sample audit, not an efficacy comparison.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnxruntime as ort
import torch

from pbfm.agents.load_utils import load_model_from_checkpoint_dir
from pbfm.inference.export.depth_policy import depth_to_onnx_float


def prepare_trace_batch(data, inputs, depth_key, start, stop):
    """Build two independent input paths, retaining PyTorch depth byte dtype."""
    raw = {k: np.asarray(data[k][start:stop]) for k in inputs}
    obs = {k: torch.from_numpy(v) if k == depth_key else torch.from_numpy(v.astype(np.float32))
           for k, v in raw.items() if k != "z"}
    z = torch.from_numpy(raw["z"].astype(np.float32))
    feed = {k: depth_to_onnx_float(v) if k == depth_key else v.astype(np.float32)
            for k, v in raw.items()}
    return obs, z, feed


def verify_trace(model, session, trace):
    inputs = [x.name for x in session.get_inputs()]
    depth_key = model.cfg.archi.actor.depth_key
    if depth_key not in inputs:
        raise ValueError("Export does not expose the checkpoint's depth input")
    if [x.name for x in session.get_outputs()] != ["action", "estimated_velocity"]:
        raise ValueError("Trace verifier requires action and estimated_velocity outputs")
    with np.load(trace, allow_pickle=False) as data:
        count = len(data["z"])
        if count == 0:
            raise ValueError("Empty rollout cannot verify an interface")
        required = inputs + ["velocity_estimate", "velocity_true_yaw"]
        if any(len(data[k]) != count for k in required):
            raise ValueError("Misaligned trace lengths")
        maxima = np.zeros(2)
        saved_estimate_error = 0.0
        onnx_saved_estimate_error = 0.0
        onnx_depth_min, onnx_depth_max = float("inf"), -float("inf")
        for start in range(0, count, 16):
            stop = min(start + 16, count)
            obs, z, feed = prepare_trace_batch(data, inputs, depth_key, start, stop)
            onnx_depth_min = min(onnx_depth_min, float(feed[depth_key].min()))
            onnx_depth_max = max(onnx_depth_max, float(feed[depth_key].max()))
            with torch.no_grad():
                expected = (model.act(obs, z).numpy(), model.estimate_velocity(obs).numpy())
            actual = session.run(None, feed)
            for i, (a, b) in enumerate(zip(expected, actual)):
                np.testing.assert_allclose(a, b, atol=2e-4, rtol=2e-4)
                maxima[i] = max(maxima[i], float(np.abs(a - b).max()))
            # Independent anchor: the estimate was saved during the original
            # GPU rollout, before this verifier's preprocessing existed.
            recorded = data["velocity_estimate"][start:stop]
            np.testing.assert_allclose(expected[1], recorded, atol=2e-4, rtol=2e-4,
                                       err_msg="Raw PyTorch recomputation disagrees with saved rollout velocity_estimate")
            np.testing.assert_allclose(actual[1], recorded, atol=2e-4, rtol=2e-4,
                                       err_msg="ONNX disagrees with saved rollout velocity_estimate")
            saved_estimate_error = max(saved_estimate_error, float(np.abs(expected[1] - recorded).max()))
            onnx_saved_estimate_error = max(onnx_saved_estimate_error, float(np.abs(actual[1] - recorded).max()))
        # Accuracy metrics use the ORIGINAL saved predictions, not recomputed
        # values. Correcting verification must not silently redefine this MAE.
        error = data["velocity_estimate"] - data["velocity_true_yaw"]
        if not np.isfinite(error).all():
            raise ValueError("Nonfinite velocity estimate or truth")
        stopped = np.linalg.norm(data["velocity_true_yaw"][:, :2], axis=1) < .1
        false_motion = stopped & (np.linalg.norm(data["velocity_estimate"][:, :2], axis=1) > .2)
        return {
            "path": str(trace), "trace_sha256": hashlib.sha256(Path(trace).read_bytes()).hexdigest(),
            "frames": count, "max_abs_action_error": maxima[0], "max_abs_estimate_error": maxima[1],
            "max_abs_pytorch_vs_saved_estimate_error": saved_estimate_error,
            "max_abs_onnx_vs_saved_estimate_error": onnx_saved_estimate_error,
            "source_depth_dtype": str(data[depth_key].dtype),
            "source_depth_min": float(data[depth_key].min()), "source_depth_max": float(data[depth_key].max()),
            "pytorch_depth_dtype_preserved": True, "onnx_depth_dtype": "float32",
            "onnx_depth_min": onnx_depth_min, "onnx_depth_max": onnx_depth_max,
            "onnx_depth_scale": (1.0 / 255.0 if data[depth_key].dtype == np.uint8 else 1.0),
            "velocity_axis_order": "xyz", "accuracy_source": "original_saved_rollout_predictions",
            "mae_mps": np.abs(error).mean(axis=0).tolist(), "bias_mps": error.mean(axis=0).tolist(),
            "pooled_p95_abs_error_mps": np.quantile(np.abs(error), .95, axis=0).tolist(),
            "stopped_count": int(stopped.sum()), "false_motion_count": int(false_motion.sum()),
            "false_motion_when_stopped": float(false_motion.sum()/stopped.sum()) if stopped.any() else None,
            "velocity_valid_fraction": float(data["velocity_command"][:, 2].mean()),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--traces", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    model = load_model_from_checkpoint_dir(args.checkpoint, device="cpu").eval()
    model.cfg = model.cfg.model_copy(update={"amp": False})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    report = {"checkpoint": str(args.checkpoint), "onnx": str(args.onnx),
              "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(), "traces": [],
              "verification_contract": "raw_pytorch_vs_float01_onnx_and_saved_estimate_v2"}
    for trace in args.traces:
        report["traces"].append(verify_trace(model, session, trace))
    report["status"] = "passed"
    report["evidence_limit"] = "Interface parity on a short rollout; not proof of improved stairs or deployment safety."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
