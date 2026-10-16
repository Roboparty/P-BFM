"""Production FP32 export of a direct-depth Actor, including its estimator.

Non-depth inputs are environment observations before model normalization.
The ONNX depth input is float32 in [0,1]: environment uint8 depth MUST be
converted with /255 outside this graph. Floating depth is already normalized.
Depth preprocessing/history and action scaling/PD remain external.
No critic, teacher, target-velocity label or privileged state is exported.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def depth_to_onnx_float(depth):
    """Explicit environment -> float ONNX boundary; never rescale float twice.

    Keep the original uint8 array/tensor for the PyTorch reference path. Only
    the independent ONNX feed should call this function.
    """
    array = np.asarray(depth)
    if array.dtype == np.uint8:
        return array.astype(np.float32) / np.float32(255.0)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"Depth must be uint8 or normalized floating point, got {array.dtype}")
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError("Floating ONNX depth must be finite and normalized to [0,1]; uint8 needs /255")
    return array.astype(np.float32)


class DirectDepthPolicyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        from pbfm.agents.nn_models import DirectDepthActor
        if not isinstance(model._actor, DirectDepthActor):
            raise TypeError("This exporter supports DirectDepthActor only")
        keys = list(model.cfg.archi.actor.input_filter.key)
        allowed = {"state", "last_action", "history_actor", "heading", "velocity_command"}
        if not set(keys).issubset(allowed):
            raise ValueError(f"Unreviewed direct-depth deployment inputs: {set(keys) - allowed}")
        self.keys = keys + [model.cfg.archi.actor.depth_key]
        self.actor = copy.deepcopy(model._actor).cpu().float().eval()
        # Copy ONLY the deployed normalizers; never require training-only keys.
        normalizers = model._obs_normalizer._normalizers
        if not isinstance(normalizers, nn.ModuleDict):
            raise TypeError("Direct-depth export requires per-key observation normalizers")
        # Moving uint8 conversion to the ONNX input is equivalent only if
        # observation normalization does not change depth before DepthEncoder.
        if not isinstance(normalizers[model.cfg.archi.actor.depth_key], nn.Identity):
            raise ValueError("Float depth export requires an identity depth observation normalizer")
        self.normalizers = nn.ModuleDict({k: copy.deepcopy(normalizers[k]).cpu().float().eval() for k in self.keys})
        self.std = float(model.cfg.actor_std)
        self.has_estimate = bool(model.cfg.archi.actor.estimated_velocity_enabled)
        self.eval().requires_grad_(False)

    def forward(self, *inputs):
        obs = {k: self.normalizers[k](x) for k, x in zip(self.keys, inputs[:-1])}
        dist = self.actor(obs, inputs[-1], self.std)
        if self.has_estimate:
            return dist.mean.float(), dist.velocity_estimate.float()
        return dist.mean.float()


def export_direct_depth_policy(model, output_path, *, batch_size=1, opset_version=17, verify=True):
    """The generic production export route, not a synthetic test-only wrapper."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    wrapper = DirectDepthPolicyWrapper(model)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = wrapper.keys + ["z"]
    outputs = ["action"] + (["estimated_velocity"] if wrapper.has_estimate else [])
    shapes = {k: list(model.obs_space[k].shape) for k in wrapper.keys}
    shapes["z"] = [int(model.cfg.archi.z_dim)]
    from pbfm.perception.instinct_direct_depth import depth_config_for_profile
    direct = depth_config_for_profile(model.cfg.archi.actor.depth_processing_profile)
    if shapes[model.cfg.archi.actor.depth_key] != [direct.num_output_frames, direct.output_height, direct.output_width]:
        raise ValueError("Cannot export inconsistent model shape / depth preprocessing contract")
    generator = torch.Generator().manual_seed(4831)

    def examples(batch):
        values = [torch.randn((batch, *shapes[k]), generator=generator) * .1 for k in names]
        depth_index = names.index(model.cfg.archi.actor.depth_key)
        values[depth_index] = torch.rand(values[depth_index].shape, generator=generator)
        if "velocity_command" in names:
            command = values[names.index("velocity_command")]
            # Invalid, valid stop, valid move -- third component is NOT vz.
            rows = torch.tensor([[0., 0., 0.], [0., 0., 1.], [.8, -.2, 1.]])
            command.copy_(rows[torch.arange(batch) % 3])
        values[-1] = model.project_z(values[-1]).detach().cpu()
        return tuple(values)

    kwargs = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with torch.no_grad():
        torch.onnx.export(wrapper, examples(batch_size), str(path), input_names=names,
            output_names=outputs, dynamic_axes={k: {0: "batch"} for k in names + outputs},
            opset_version=max(17, opset_version), **kwargs)
    import onnx
    onnx.checker.check_model(str(path))
    checks = []
    if verify:
        # Absence of ORT is a failed verification, never a silent "pass".
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if [i.name for i in session.get_inputs()] != names:
            raise RuntimeError("ONNX input signature differs from deployment contract")
        for depth_dtype in ("float32", "uint8"):
            for batch in (1, 2, 3):
                inputs = list(examples(batch))
                if depth_dtype == "uint8":
                    # Exercise all byte values, including nonzero 128 and 255.
                    shape = inputs[depth_index := names.index(model.cfg.archi.actor.depth_key)].shape
                    inputs[depth_index] = (torch.arange(int(np.prod(shape))) % 256).to(torch.uint8).reshape(shape)
                with torch.no_grad():
                    # Intentionally preserve uint8 here: the encoder must do /255.
                    expected = wrapper(*inputs)
                expected = expected if isinstance(expected, tuple) else (expected,)
                feed = {k: (depth_to_onnx_float(x.numpy()) if k == model.cfg.archi.actor.depth_key else x.numpy())
                        for k, x in zip(names, inputs)}
                actual = session.run(None, feed)
                for name, a, b in zip(outputs, expected, actual):
                    np.testing.assert_allclose(a.numpy(), b, atol=2e-4, rtol=2e-4)
                    checks.append(dict(batch=batch, source_depth_dtype=depth_dtype, output=name,
                                       max_abs_error=float(np.abs(a.numpy() - b).max())))
    metadata = {
        "format": "direct_depth_full_v1", "input_names": names, "input_shapes": shapes,
        "actor_input_keys": wrapper.keys, "actor_input_dims": shapes,
        "z_dim": shapes["z"][0], "output_action_dim": int(model.action_dim),
        "output_names": outputs, "precision": "FP32", "verification": checks,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "includes": "frozen observation normalization, depth CNN/GRU, current-velocity estimator when enabled, deterministic Actor",
        "velocity_command": "[target_vx,target_vy,valid] in current yaw frame; valid stop=[0,0,1], invalid=[0,0,0]",
        "estimated_velocity": "[current_vx,current_vy,current_vz]; yaw-horizontal xy, world-up z; m/s",
        "external_runtime": "Supply training-compatible state/history, heading error and z; ONNX depth MUST be float32 [0,1] (uint8 history /255, NOT float32 byte values or camera metres); apply original action scaling and PD outside ONNX.",
        "depth_input_contract": {
            "processing_profile": direct.processing_profile,
            "depth_quantity": "optical_z",
            "crop_top_bottom_left_right": list(direct.crop),
            "source_shape": [direct.height, direct.width],
            "max_distance_m": direct.max_distance,
            "invalid_fill_m": 0.0 if direct.processing_profile == "legacy_rp1" else direct.max_distance,
            "gaussian": "binomial_3x3" if direct.processing_profile == "legacy_rp1" else "3x3_sigma1",
            "sampled_ages": list(direct.sampled_ages),
            "onnx_dtype": "float32", "onnx_range": [0.0, 1.0],
            "environment_uint8_conversion": "astype(float32) / 255.0",
            "environment_float_conversion": "already normalized [0,1]; no extra scaling",
            "pytorch_reference": "preserve original uint8; DepthEncoder performs /255 internally",
        },
        "depth_shape": shapes[model.cfg.archi.actor.depth_key],
        "depth_encoder_architecture": {
            "first_conv_stride": model.cfg.archi.actor.depth_first_conv_stride,
            "frame_feature_dim": model.cfg.archi.actor.depth_frame_feature_dim,
            "gru_hidden_dim": model.cfg.archi.actor.depth_latent_dim,
            "pool_shape": [2, 4],
        },
        "evidence_limit": "Synthetic FP32 parity is not closed-loop or hardware safety validation.",
    }
    path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
