# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed optimizer stages, actor gradient isolation and compilation."""

from contextlib import contextmanager

import torch

from pbfm.training.distributed import average_gradients, wrap_distributed_stage


class _FBTrainingStage(torch.nn.Module):
    def __init__(self, forward_map: torch.nn.Module, backward_map: torch.nn.Module) -> None:
        super().__init__()
        self.forward_map = forward_map
        self.backward_map = backward_map

    def forward(self, obs, z, action, goal):
        return self.forward_map(obs, z, action), self.backward_map(goal)


class _MethodTrainingStage(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, method_name: str = "forward") -> None:
        super().__init__()
        self.wrapped_module = module
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        return getattr(self.wrapped_module, self.method_name)(*args, **kwargs)


@contextmanager
def _without_parameter_gradients(*modules: torch.nn.Module):
    parameters = tuple(parameter for module in modules for parameter in module.parameters())
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, requires_grad):
            parameter.requires_grad_(enabled)


class AgentExecutionMixin:
    """Execution policy only; algorithm losses and update order live in the agent."""

    def enable_distributed_gradient_sync(self, *, bucket_cap_mb: float = 25.0) -> None:
        """Enable DDP reducer hooks separately for every optimizer stage."""
        if getattr(self, "_distributed_training_stages", None):
            return
        stages: dict[str, torch.nn.Module] = {
            "fb": _FBTrainingStage(self._model._forward_map, self._model._backward_map),
            "actor": _MethodTrainingStage(self._model._actor),
        }
        if hasattr(self._model, "_discriminator"):
            stages["discriminator"] = _MethodTrainingStage(self._model._discriminator, "compute_logits")
        if hasattr(self._model, "_critic"):
            stages["critic"] = _MethodTrainingStage(self._model._critic)
        if hasattr(self._model, "_aux_critic"):
            stages["aux_critic"] = _MethodTrainingStage(self._model._aux_critic)
        if hasattr(self._model, "_heading_critic"):
            stages["heading_critic"] = _MethodTrainingStage(self._model._heading_critic)
        if hasattr(self._model, "_velocity_critic"):
            stages["velocity_critic"] = _MethodTrainingStage(self._model._velocity_critic)
        self._distributed_training_stages = {
            name: wrap_distributed_stage(stage, bucket_cap_mb=bucket_cap_mb)
            for name, stage in stages.items()
        }

    def _training_stage(self, name: str) -> torch.nn.Module | None:
        return getattr(self, "_distributed_training_stages", {}).get(name)

    def _sync_gradients_if_manual(self, parameters) -> None:
        if not getattr(self, "_distributed_training_stages", None):
            average_gradients(parameters)

    def _run_actor_update(self, *args, **kwargs):
        if not getattr(self, "_distributed_training_stages", None):
            return self.update_actor(*args, **kwargs)
        dependencies = [self._model._forward_map]
        if hasattr(self._model, "_critic"):
            dependencies.append(self._model._critic)
        if hasattr(self._model, "_aux_critic"):
            dependencies.append(self._model._aux_critic)
        if hasattr(self._model, "_heading_critic"):
            dependencies.append(self._model._heading_critic)
        if hasattr(self._model, "_velocity_critic"):
            dependencies.append(self._model._velocity_critic)
        with _without_parameter_gradients(*dependencies):
            return self.update_actor(*args, **kwargs)

    def setup_compile(self):
        # Preserve the interleaved compile/CUDA-graph wrapper order of the
        # complete algorithm, including compilation of the final actor update.
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_fb = torch.compile(self.update_fb, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.update_actor = torch.compile(self.update_actor, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)

        print(f"cudagraphs {self.cfg.cudagraphs}")
        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_fb = CudaGraphModule(self.update_fb, warmup=5)
            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_critic = torch.compile(self.update_critic, mode=mode)
            self.update_discriminator = torch.compile(self.update_discriminator, mode=mode)
            self.encode_expert = torch.compile(self.encode_expert, mode=mode, fullgraph=True)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_critic = CudaGraphModule(self.update_critic, warmup=5)
            self.update_discriminator = CudaGraphModule(self.update_discriminator, warmup=5)
            self.encode_expert = CudaGraphModule(self.encode_expert, warmup=5)

        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_aux_critic = torch.compile(self.update_aux_critic, mode=mode)
            if self._heading_critic_enabled:
                self.update_heading_critic = torch.compile(self.update_heading_critic, mode=mode)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_aux_critic = CudaGraphModule(self.update_aux_critic, warmup=5)
            if self._heading_critic_enabled:
                self.update_heading_critic = CudaGraphModule(self.update_heading_critic, warmup=5)
