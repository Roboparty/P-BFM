"""Read-only offline verification of the single-checkpoint deployment bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_yaml_config
from utils.manual_goal import ManualGoalSequence
from rl_policy.direct_depth import DirectDepthConfig, preprocess_direct_depth_frame
from utils.training_depth_rays import TrainingDepthSampler
from utils.command_timeout_contract import validate_command_timeout_contract
from rl_policy.ufo_qv_policy import _model_root_artifact


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def verify_config(policy_path, robot_path):
    """Validate the resolved launch configuration against the packaged contract."""
    policy = load_yaml_config(policy_path)
    robot = load_yaml_config(robot_path)
    contract = json.loads((ROOT / 'model/g1_policy/deployment_contract.json').read_text())
    for kind, actual in [('policy', policy), ('robot', robot)]:
        for key, expected in contract[kind].items():
            # Mapping order defines observation packing and regex gain matching.
            checked(json.dumps(actual.get(key)) == json.dumps(expected),
                    f'{kind} config violates deployment contract: {key}')
    validate_command_timeout_contract(robot)
    checked(policy.get('depth_inference_backend') == 'onnx', 'Release depth encoder must use ONNX')
    checked(policy.get('actor_inference_backend') in ('onnx', 'tensorrt'), 'Unsupported Actor backend')
    return policy, robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, help="Also compare a locally built TensorRT Actor with ONNX")
    parser.add_argument('--policy-config', type=Path, default=os.environ.get('POLICY_CONFIG', ROOT / 'model/g1_policy/policy.yaml'))
    parser.add_argument('--robot-config', type=Path, default=os.environ.get('ROBOT_CONFIG', ROOT / 'config/robot/g1_real.yaml'))
    parser.add_argument('--task-config', type=Path, help='Also validate the task selected for launch')
    parser.add_argument('--runtime', action='store_true', help='Verify the exact configured Actor engine before launch')
    parser.add_argument('--print-engine-path', action='store_true', help='Print the configured engine destination for the build script')
    args = parser.parse_args()
    model = ROOT / "model/g1_policy"
    manifest = json.loads((model / "manifest.json").read_text())
    checked(manifest["checkpoint_step"] == 364822528, "Wrong checkpoint")
    checked(set(manifest["onnx_sha256"]) == {"actor.onnx", "depth_encoder.onnx", "backward_encoder.onnx"}, "Unexpected model inventory")
    for name, expected in manifest["onnx_sha256"].items():
        checked(digest(model / "onnx" / name) == expected, f"Model hash mismatch: {name}")
    for name, expected in manifest["asset_sha256"].items():
        checked(digest(ROOT / name) == expected, f"Context/reference hash mismatch: {name}")

    policy, robot = verify_config(args.policy_config, args.robot_config)
    actor_path = model / 'onnx/actor.onnx'
    encoder_path = Path(policy['direct_depth_encoder_model_path']).expanduser()
    if not encoder_path.is_absolute():
        encoder_path = actor_path.parent / encoder_path
    checked(digest(encoder_path) == manifest['onnx_sha256']['depth_encoder.onnx'], 'Configured depth encoder hash mismatch')
    engine_path = None
    if policy['actor_inference_backend'] == 'tensorrt':
        engine_path = _model_root_artifact(actor_path, policy['tensorrt_actor_engine_path'])
    if args.print_engine_path:
        checked(engine_path is not None, 'Actor backend must be tensorrt for build')
        print(engine_path)
        return
    if args.runtime:
        checked(args.task_config is not None, '--runtime requires --task-config')
        if args.engine is not None:
            checked(engine_path == args.engine.expanduser().resolve(), 'Engine argument differs from the configured runtime engine')
        args.engine = engine_path
    teleop_policy = load_yaml_config(model / "teleop_policy.yaml")
    checked(teleop_policy["teleop_velocity_enabled"], "Teleop QV must be enabled")
    checked((ROOT / teleop_policy["standing_fallback_z_path"]).is_file(), "Missing standing latent")
    depth_config = DirectDepthConfig.from_policy_config(policy)
    checked(depth_config.processing_profile == "optical_z_maxcrop18_reset_v2", "Wrong depth profile")
    checked(not depth_config.vertical_flip and depth_config.gaussian_enabled, "Wrong depth orientation/Gaussian")
    intrinsics = np.array([[240., 0, 239.5], [0, 240., 134.5], [0, 0, 1.]])
    sampler = TrainingDepthSampler(intrinsics, (270, 480), depth_config.processing_profile)
    processed = preprocess_direct_depth_frame(sampler.sample(np.zeros((270, 480), np.float32)), depth_config)
    checked(processed.shape == (18, 32) and bool(np.all(processed == 255)), "Invalid depth must become maximum distance")

    task_count = 0
    task_paths = [ROOT / 'config/tasks' / (entry['task'] + '.yaml')
                  for entry in json.loads((model / 'tasks.json').read_text())]
    if args.task_config is not None:
        task_paths = list(dict.fromkeys([p.resolve() for p in task_paths + [args.task_config]]))
    for task_path in task_paths:
        task = load_yaml_config(task_path)
        task_count += 1
        if task["ctx_source"] == "zmq":
            checked(task["window_size"] == 1, "Unexpected realtime context averaging")
            if args.task_config is not None and task_path.resolve() == args.task_config.resolve():
                checked(policy.get('teleop_velocity_enabled'), 'Selected teleop policy requires QV velocity')
                standing = ROOT / policy['standing_fallback_z_path']
                checked(digest(standing) == manifest['asset_sha256']['model/g1_policy/standing_z.npy'], 'Wrong teleop standing z')
            continue
        path = ROOT / task["ctx_path"]
        known_contexts = {(ROOT / relative).resolve(): sha for relative, sha in manifest['asset_sha256'].items()}
        checked(path.resolve() in known_contexts, f'Unregistered context in selected task: {path}')
        checked(digest(path) == known_contexts[path.resolve()], f'Context hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as context:
            z = context["z"]
            checked(z.ndim == 2 and z.shape[1] == 256 and bool(np.isfinite(z).all()), f"Invalid z: {path.name}")
            checked(0 <= task["start"] < task["end"] <= len(z), "Invalid playback bounds")
            checked(abs(float(context["dt"]) - 0.02) < 1e-9, "Context must be 50 Hz")
            checked(context["heading_relative_xy"].shape == (len(z), 2), "Invalid heading shape")
            for key in ["heading_relative_xy", "heading_valid"]:
                checked(bool(np.isfinite(context[key]).all()), f"Nonfinite {key}")
            if task.get("context_contract") == "reward_goal_yaw_command_v1":
                checked(str(context["context_contract"]) == task["context_contract"], "Reward contract mismatch")
                checked(context["velocity_command_yaw"].shape == (len(z), 3), "Invalid reward velocity")
            else:
                checked(context["reference_velocity_heading0_xy"].shape == (len(z), 2), "Invalid tracking velocity")
        if task.get("manual_goal_selection"):
            ManualGoalSequence(ROOT / task["goal_schedule_path"], path)
        if "goal_standing_z_path" in task:
            checked((ROOT / task["goal_standing_z_path"]).is_file(), "Missing goal standing latent")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    encoder = ort.InferenceSession(str(encoder_path), options, providers=["CPUExecutionProvider"])
    actor = ort.InferenceSession(str(actor_path), options, providers=["CPUExecutionProvider"])
    trt = None
    if args.engine:
        from utils.tensorrt_module import TensorRTSession
        trt = TensorRTSession(str(args.engine.resolve(strict=True)))
    errors, engine_errors = [], []
    try:
        with np.load(model / "onnx/reference_cases.npz", allow_pickle=False) as reference:
            for index in range(len(reference["action"])):
                latent = encoder.run(["depth_latent"], {"depth_image": reference["depth_image"][index]})[0]
                feed = {node.name: latent if node.name == "depth_latent" else reference[node.name][index] for node in actor.get_inputs()}
                actual = actor.run(["action"], feed)[0]
                np.testing.assert_allclose(actual, reference["action"][index], rtol=0, atol=1e-4)
                errors.append(float(np.max(abs(actual - reference["action"][index]))))
                if trt is not None:
                    on_target = trt.run(["action"], feed)[0]
                    np.testing.assert_allclose(on_target, actual, rtol=0, atol=1e-4)
                    engine_errors.append(float(np.max(abs(on_target - actual))))
    finally:
        if trt is not None:
            trt.close()
    print(json.dumps({"checkpoint_step": 364822528, "tasks_checked": task_count,
                      "reference_cases": len(errors), "max_action_error": max(errors),
                      "tensorrt_max_error": max(engine_errors) if engine_errors else None,
                      "policy_config": str(args.policy_config.resolve()),
                      "robot_config": str(args.robot_config.resolve()),
                      "actor_engine": str(args.engine.resolve()) if args.engine else None,
                      "hardware_control": "not started"}, indent=2))


if __name__ == "__main__":
    main()
