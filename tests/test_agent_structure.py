"""Algorithm ownership and execution ordering after the inheritance removal."""

from types import SimpleNamespace

import pytest
import tensordict.nn
import torch

from pbfm.agents.base_model import BaseModel
from pbfm.agents.execution import AgentExecutionMixin
from pbfm.agents.fb_cpr_aux.agent import FBcprAuxAgent
from pbfm.agents.fb_cpr_aux.config import FBcprAuxAgentConfig
from pbfm.agents.fb_cpr_aux.model import FBcprAuxModel


def test_complete_algorithm_owns_its_networks_and_losses():
    assert FBcprAuxAgent.__bases__ == (AgentExecutionMixin,)
    assert FBcprAuxModel.__bases__ == (BaseModel,)
    for name in (
        "__init__", "setup_training", "update", "update_fb", "update_discriminator",
        "update_critic", "update_aux_critic", "update_heading_critic",
        "update_velocity_critic", "update_actor", "advance_rollout_context",
    ):
        assert name in FBcprAuxAgent.__dict__
    cfg = FBcprAuxAgentConfig()
    assert cfg.name == "FBcprAuxAgent"
    assert cfg.model.name == "FBcprAuxModel"
    assert cfg.model.archi.name == "FBcprAuxModelArchiConfig"
    assert cfg.train.name == "FBcprAuxAgentTrainConfig"


@pytest.mark.parametrize("compile_enabled,cudagraphs", [(True, False), (False, True), (True, True)])
def test_compile_and_cuda_graphs_keep_the_original_stage_order(monkeypatch, compile_enabled, cudagraphs):
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent.cfg = SimpleNamespace(
        compile=compile_enabled, cudagraphs=cudagraphs,
        model=SimpleNamespace(heading_critic_enabled=True),
    )
    events = []

    def record(kind):
        def wrap(method, **kwargs):
            events.append((kind, method.__name__, kwargs))
            return SimpleNamespace(__name__=method.__name__, wrappers=method.wrappers + (kind,))
        return wrap

    groups = (
        (("update_fb", "update_actor", "sample_mixed_z"), ("update_fb", "update_actor")),
        (("update_critic", "update_discriminator", "encode_expert"),
         ("update_critic", "update_discriminator", "encode_expert")),
        (("update_aux_critic", "update_heading_critic"), ("update_aux_critic", "update_heading_critic")),
    )
    for compile_names, _ in groups:
        for name in compile_names:
            setattr(agent, name, SimpleNamespace(__name__=name, wrappers=()))
    monkeypatch.setattr(torch, "compile", record("compile"))
    monkeypatch.setattr(tensordict.nn, "CudaGraphModule", record("cuda"))

    agent.setup_compile()

    expected = []
    for compile_names, cuda_names in groups:
        if compile_enabled:
            for name in compile_names:
                kwargs = {"mode": None if cudagraphs else "reduce-overhead"}
                if name in {"sample_mixed_z", "encode_expert"}:
                    kwargs["fullgraph"] = True
                expected.append(("compile", name, kwargs))
        if cudagraphs:
            expected.extend(("cuda", name, {"warmup": 5}) for name in cuda_names)
    assert events == expected
    assert agent.update_actor.wrappers == tuple(
        kind for kind, enabled in (("compile", compile_enabled), ("cuda", cudagraphs)) if enabled
    )
