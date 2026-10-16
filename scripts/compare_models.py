#!/usr/bin/env python3
"""Exact CPU regression between two PBFM source trees, using the small test fixture.

Run with the project virtualenv's Python. Only temporary checkpoints and an
optional JSON report are written; neither source tree nor its configs are edited.
The optional crop18-update mode runs one synthetic CPU update, not a training
experiment, full-size P3 model, or simulator test.
"""

import argparse
import copy
import dataclasses
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile


def equal_tree(left, right, path="root"):
    """Require exact values as well as matching keys, tensor shapes and dtypes."""
    import torch

    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), f"{path}: tensor type changed"
        assert left.shape == right.shape, f"{path}: shape changed"
        assert left.dtype == right.dtype, f"{path}: dtype changed"
        assert torch.isfinite(left).all() and torch.isfinite(right).all(), f"{path}: nonfinite"
        if not torch.equal(left, right):
            delta = (left.to(torch.float64) - right.to(torch.float64)).abs().max().item()
            raise AssertionError(f"{path}: tensor values differ (max_abs={delta:.9g})")
    elif isinstance(left, dict):
        assert isinstance(right, dict), f"{path}: mapping type changed"
        assert left.keys() == right.keys(), (
            f"{path}: keys differ; missing={left.keys() - right.keys()}, "
            f"extra={right.keys() - left.keys()}"
        )
        for key in left:
            equal_tree(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), f"{path}: sequence changed"
        for index, (a, b) in enumerate(zip(left, right)):
            equal_tree(a, b, f"{path}[{index}]")
    else:
        assert type(left) is type(right) and left == right, f"{path}: {left!r} != {right!r}"


def snapshot(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if dataclasses.is_dataclass(value):
        return {field.name: snapshot(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(item) for item in value)
    return value


def forward_outputs(model, batch):
    import torch

    # Match deployment flags across agent-created and model-only loaded models.
    # CPU GRU can differ by ~1e-8 when parameter requires_grad flags differ,
    # even under no_grad, so all compared forwards use the same frozen mode.
    model.eval().requires_grad_(False)
    obs, z, action = batch["observation"], batch["z"], batch["action"]
    with torch.no_grad():
        normalized = model._normalize(obs)
        return snapshot({
            "actor": model.act(obs, z),
            "depth_encoder": model._actor.depth_encoder(normalized[model._actor.cfg.depth_key]),
            "estimated_velocity": model.estimate_velocity(obs),
            "B": model.backward_map(obs),
            "F": model.forward_map(obs, z, action),
            "QD": model.critic(obs, z, action),
            "D": model.discriminator(obs, z),
            "QAux": model.aux_critic(obs, z, action),
            "QH": model.heading_critic(obs, z, action),
            "QV": model._velocity_critic(normalized, z, action),
            "goal_z": model.goal_inference(obs),
            "tracking_z": model.tracking_inference(obs),
            "reward_z": model.reward_inference(obs, torch.linspace(-1, 1, 8).reshape(8, 1)),
        })


def context_outputs(agent, fixture, batch, package, seed):
    import torch

    # The existing context fixture uses two expert sequences of four frames.
    original_cfg = agent.cfg
    agent.cfg = agent.cfg.model_copy(update={"train": agent.cfg.train.model_copy(
        update={"rollout_expert_trajectories_length": 4})})
    context_module = importlib.import_module(f"{package}.agents.fb_cpr_aux.agent")
    if not hasattr(context_module, 'RolloutContextState'):
        context_module = importlib.import_module(f"{package}.agents.fb.agent")
    context_class = context_module.RolloutContextState
    sequence_batch = copy.deepcopy(batch)
    sequence_batch["reference_velocity_world_xy"] = torch.stack((torch.arange(8).float(), torch.zeros(8)), -1)

    class SequenceProvider(fixture.Provider):
        def sample(self, n, seq_length=None):
            assert n == 8 and seq_length == 4
            return copy.deepcopy(self.batch)

    provider = {"expert_slicer": SequenceProvider(sequence_batch, expert=True)}
    counts = torch.zeros(4, 1, dtype=torch.long)
    headings = torch.tensor([[0., 1.]]).repeat(4, 1)
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(seed)
            state = agent.advance_rollout_context(context_class(), counts, provider, current_heading_xy=headings)
            result = {
                "initial": snapshot(state),
                "next_velocity": snapshot(agent.next_rollout_velocity_target(state, counts)),
                "next_heading": snapshot(agent.next_rollout_heading_target(state, counts)),
                "next_z": snapshot(agent.next_rollout_z(state, counts)),
                "continues": snapshot(agent.rollout_heading_context_continues(
                    state, counts + 1, torch.tensor([[False], [True], [False], [False]]))),
            }
            result["advanced"] = snapshot(agent.advance_rollout_context(
                state, counts + 1, provider, current_heading_xy=headings))
            return result
    finally:
        agent.cfg = original_cfg


def check_checkpoint(path, loader, expected_state, expected_optimizers, batches, outputs,
                     context, fixture, package, seed):
    import safetensors.torch

    stored = safetensors.torch.load_file(str(path / "model/model.safetensors"), device="cpu")
    equal_tree(expected_state, stored, "checkpoint.stored_state")
    loaded_model = loader.load_model_from_checkpoint_dir(str(path), device="cpu")
    loaded_agent = loader.load_agent_from_checkpoint_dir(str(path), device="cpu")
    for kind, model in (("model", loaded_model), ("agent", loaded_agent._model)):
        # Check before strict reload, so the assertion cannot hide an incomplete loader.
        equal_tree(expected_state, snapshot(model.state_dict()), f"checkpoint.{kind}.state")
        model.load_state_dict(stored, strict=True)
        for depth_type, batch in batches.items():
            equal_tree(outputs[depth_type], forward_outputs(model, batch), f"checkpoint.{kind}.{depth_type}")
    equal_tree(expected_optimizers, snapshot(loaded_agent.optimizer_dict), "checkpoint.optimizers")
    if context is not None:
        equal_tree(context, context_outputs(loaded_agent, fixture, batches["uint8"], package, seed), "checkpoint.context")


def worker(args):
    import gymnasium as gym
    import numpy as np
    import torch

    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / "tests")]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location("regression_fixture", root / "tests/test_qv_agent_integration.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    agent, dims = fixture.make_agent()
    if args.mode == "crop18-update":
        # Keep the source fixture/config files untouched. Rebuild its small
        # heads with the formal depth profile in this process only.
        config = agent.cfg.model_dump()
        config["model"]["archi"]["actor"].update(
            depth_channels=8, depth_height=18, depth_width=32,
            depth_first_conv_stride=1, depth_frame_feature_dim=256,
            depth_processing_profile="optical_z_maxcrop18_reset_v2",
            velocity_supervision_updates_depth=True,
        )
        config["model"]["reference_velocity_policy"] = "all_reference"
        dims["depth_image"] = (8, 18, 32)
        spaces = dict(agent.obs_space.spaces)
        spaces["depth_image"] = gym.spaces.Box(-np.inf, np.inf, shape=dims["depth_image"], dtype=np.float32)
        agent = type(agent.cfg).model_validate(config).build(gym.spaces.Dict(spaces), agent.action_dim)
    state = snapshot(agent._model.state_dict())
    rng_after_init = torch.get_rng_state()
    batch = fixture.make_batch(agent, dims)
    depth_key = agent._model._actor.cfg.depth_key
    shape = batch["observation"][depth_key].shape
    batch["observation"][depth_key] = (torch.arange(np.prod(shape)).reshape(shape) % 256).to(torch.uint8)
    training = None
    if args.mode == "crop18-update":
        batch["next"]["observation"][depth_key] = batch["observation"][depth_key].roll(1, dims=-1)
        training = {"initial_state": state, "inputs": snapshot(batch)}
        metrics = agent.update({"train": fixture.Provider(batch),
                               "expert_slicer": fixture.Provider(batch, expert=True)}, 1024)
        training.update(metrics=snapshot(metrics), rng_after_update=torch.get_rng_state())
        assert metrics["depth_gradient_local/Estimator_norm"] > 0, "Estimator did not supervise depth"
        updated_state = snapshot(agent._model.state_dict())
        for prefix in ("_actor.depth_encoder.", "_actor.velocity_estimator.", "_velocity_critic."):
            assert any(not torch.equal(value, updated_state[key]) for key, value in state.items()
                       if key.startswith(prefix)), f"No parameter update in {prefix}"
        for name, optimizer in agent.optimizer_dict.items():
            assert optimizer["state"], f"{name}: missing Adam history after update"
            for history in optimizer["state"].values():
                assert history["step"].item() == 1, f"{name}: expected one Adam step"
                assert "exp_avg" in history and "exp_avg_sq" in history, f"{name}: missing Adam moments"
        state = updated_state
    # Exercise invalid, valid-stop, and valid-moving command rows.
    batch["observation"]["velocity_command"][:3] = torch.tensor([[0., 0., 0.], [0., 0., 1.], [.8, .2, 1.]])
    batches = {"uint8": batch, "float32": copy.deepcopy(batch)}
    batches["float32"]["observation"][depth_key] = batch["observation"][depth_key].float() / 255.
    outputs = {name: forward_outputs(agent._model, value) for name, value in batches.items()}
    context = context_outputs(agent, fixture, batch, args.package, args.seed + 1)
    equal_tree(state, snapshot(agent._model.state_dict()), "forward_does_not_mutate_state")
    optimizers = snapshot(agent.optimizer_dict)
    checkpoint = args.snapshot.parent / f"{args.snapshot.stem}-checkpoint"
    agent.save(str(checkpoint))
    loader = importlib.import_module(f"{args.package}.agents.load_utils")
    # Existing save/load does not serialize the z_buffer filled by update.
    # Compare warm contexts across packages, but do not claim a full-context
    # checkpoint roundtrip after training. Model/Adam checks remain strict.
    checkpoint_context = context if training is None else None
    check_checkpoint(checkpoint, loader, state, optimizers, batches, outputs, checkpoint_context,
                     fixture, args.package, args.seed + 1)
    if args.reference_checkpoint:
        check_checkpoint(args.reference_checkpoint, loader, state, optimizers, batches, outputs,
                         checkpoint_context, fixture, args.package, args.seed + 1)
    data = {
        "config": json.loads(agent.cfg.model_dump_json()),
        "rng_after_init": rng_after_init,
        "state": state,
        "inputs": snapshot(batches),
        "outputs": outputs,
        "context": context,
        "optimizers": optimizers,
    }
    if training is not None:
        data["training"] = training
    torch.save(data, args.snapshot)
    actor_config = agent._model._actor.cfg
    print(json.dumps({"package": args.package, "state_tensors": len(state),
                      "forward_outputs": list(outputs["uint8"]), "checkpoint": str(checkpoint),
                      "strict_roundtrip": True, "reference_checkpoint_loaded": bool(args.reference_checkpoint),
                      "depth_config": {"shape": list(dims[depth_key]),
                                       "first_conv_stride": actor_config.depth_first_conv_stride,
                                       "frame_feature_dim": actor_config.depth_frame_feature_dim,
                                       "profile": actor_config.depth_processing_profile,
                                       "velocity_supervision_updates_depth": actor_config.velocity_supervision_updates_depth,
                                       "reference_velocity_policy": agent.cfg.model.reference_velocity_policy},
                      "optimizer_history_entries": {name: len(value["state"]) for name, value in optimizers.items()},
                      "update_metric_count": len(training["metrics"]) if training else 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-package", default="pbfm")
    parser.add_argument("--candidate-package", default="pbfm")
    parser.add_argument("--mode", choices=("forward", "crop18-update"), default="forward")
    parser.add_argument("--seed", type=int, default=4831)
    parser.add_argument("--output", type=Path, help="Optional retained JSON report")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--package", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reference-checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    report = {"status": "failed", "seed": args.seed, "device": "cpu", "mode": args.mode,
              "comparison": "torch.equal (zero tolerance)",
              "baseline_root": str(args.baseline_root.resolve()), "candidate_root": str(args.candidate_root.resolve()),
              "scope": "Existing small fixture; frozen eval forward, context, serialization; no simulator",
              "synthetic_cpu_updates": int(args.mode == "crop18-update"),
              "limits": ["Synthetic reduced-width model, not the full P3 checkpoint",
                         "No GPU/DDP or simulator test; observation normalizers use fixture Identity configs"]}
    if args.mode == "forward":
        report["limits"].append("Optimizer structure is checked, but Adam moments are unpopulated without an update")
    else:
        report["limits"].append("Existing checkpoint API omits z_buffer/collector state; warm context is compared across packages, not across reload")
    try:
        import torch

        with tempfile.TemporaryDirectory(prefix="pbfm-model-regression-") as temporary:
            temporary = Path(temporary)
            snapshots, workers = [], []
            # Baseline first: candidate must also load the actual old-package checkpoint.
            for label, root, package in (("baseline", args.baseline_root, args.baseline_package),
                                         ("candidate", args.candidate_root, args.candidate_package)):
                destination = temporary / f"{label}.pt"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--root", str(root.resolve()),
                           "--baseline-root", str(args.baseline_root.resolve()),
                           "--candidate-root", str(args.candidate_root.resolve()),
                           "--package", package, "--mode", args.mode, "--seed", str(args.seed), "--snapshot", str(destination)]
                if label == "candidate":
                    command += ["--reference-checkpoint", str(temporary / "baseline-checkpoint")]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           PYTHONHASHSEED=str(args.seed), PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root.resolve()))
                completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=180)
                if completed.returncode:
                    raise RuntimeError(f"{label} worker failed:\n{completed.stdout}\n{completed.stderr}")
                workers.append(json.loads(completed.stdout.strip().splitlines()[-1]))
                snapshots.append(torch.load(destination, weights_only=True, map_location="cpu"))
            equal_tree(snapshots[0], snapshots[1])
            report.update(status="passed", state_tensors=len(snapshots[0]["state"]), workers=workers,
                          checks=list(snapshots[0]), candidate_loads_baseline_checkpoint=True,
                          strict_state_keys_shapes_dtypes_values=True, exact_equality=True,
                          populated_optimizer_history_checked=args.mode == "crop18-update",
                          context_checkpoint_roundtrip_checked=args.mode == "forward")
    except Exception as error:
        report["error"] = str(error)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
