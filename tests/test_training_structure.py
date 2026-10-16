"""CPU checks for the extracted checkpoint and diagnostic responsibilities."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pbfm.training import checkpoint, config, diagnostics


def test_checkpoint_preserves_shards_status_and_write_order(tmp_path):
    events = []
    cfg = SimpleNamespace(
        distributed_sync=True,
        distributed_global_steps=True,
        distributed_world_size=2,
        checkpoint_rank_buffers=True,
        checkpoint_buffer=True,
        agent=SimpleNamespace(train=SimpleNamespace(batch_size=16), model=None),
    )
    checkpoint_dir = tmp_path / "checkpoint"

    def save_agent(path):
        events.append("agent")
        assert Path(path) == checkpoint_dir
        Path(path).mkdir()

    def save_buffer(path):
        events.append(path.name)
        assert path.parent == checkpoint_dir / "buffers"
        assert not (checkpoint_dir / "train_status.json").exists()

    def barrier():
        events.append("barrier")
        if events.count("barrier") == 3:
            assert (checkpoint_dir / "train_status.json").exists()

    def sync_report(model, *, src):
        events.append("sync")
        assert src == 0
        return {"max_abs_diff_from_rank0": 0.0}

    agent = SimpleNamespace(_model=torch.nn.Linear(1, 1), save=save_agent)
    buffer = SimpleNamespace(save=save_buffer)
    with patch.object(checkpoint, "barrier", barrier), patch.object(checkpoint, "module_sync_report", sync_report):
        checkpoint.save_checkpoint(
            tmp_path, cfg, agent,
            write_shared_artifacts=True, rank=0, world_size=2,
            local_time=12, global_time=24, optimizer_steps=3,
            replay_buffer={"train": buffer, "prior": buffer},
        )
    assert events == ["barrier", "sync", "agent", "train_rank_0", "prior_rank_0", "barrier", "barrier"]
    status = json.loads((checkpoint_dir / "train_status.json").read_text())
    assert status == {
        "time": 24, "local_time": 12, "global_time": 24, "optimizer_steps": 3,
        "world_size": 2, "loss_mode": "local_loss_average", "effective_batch_size": 32,
        "depth_observation_contract": None,
    }
    restored = checkpoint._normalize_train_status(status, cfg)
    assert restored["local_time"] == 12 and restored["global_time"] == 24
    assert checkpoint._normalize_train_status({"time": 25, "world_size": 2}, cfg)["local_time"] == 13
    with pytest.raises(RuntimeError, match="different distributed world_size"):
        checkpoint._normalize_train_status({**status, "world_size": 1}, cfg)
    with pytest.raises(ValueError, match="Unsupported replay buffer name"):
        checkpoint.checkpoint_buffer_path(checkpoint_dir, cfg, rank=0, world_size=2, name="expert")
    cfg.distributed_global_steps = False
    assert config._global_step_scale(cfg) == 1


def test_diagnostics_preserve_nested_failures_and_contact_threshold():
    with pytest.raises(FloatingPointError, match=r"rank=2, local_time=3, global_time=6, optimizer_steps=4") as error:
        diagnostics._assert_finite(
            {"values": [torch.tensor([1.0, float("nan")]), np.array([np.inf])]},
            label="rollout", rank=2, local_time=3, global_time=6, optimizer_steps=4,
        )
    assert "rollout.values[0]" in str(error.value)
    assert "rollout.values[1]" in str(error.value)
    env = SimpleNamespace(
        torso_index=0,
        simulator=SimpleNamespace(
            contact_forces=torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 3.0, 4.0]]]),
            _rigid_body_pos=torch.tensor([[[0.0, 0.0, 0.2]], [[0.0, 0.0, 0.8]]]),
        ),
    )
    metrics = diagnostics.torso_contact_force_metrics(env, enabled=True, threshold=1.0)
    assert metrics["torso_contact_force_contact_count"] == 1
    assert metrics["torso_contact_force_contact_frac"] == 0.5
    assert metrics["torso_z_contact_mean"] == 0.8
    assert diagnostics.torso_contact_force_metrics(env, enabled=False, threshold=1.0) == {}
