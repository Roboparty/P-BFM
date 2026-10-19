"""Sparse storage must change bytes, not the transitions used for learning."""
import copy

import h5py
import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_map

from pbfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim, get_idxs


def assert_same(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    else:
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def pair(capacity=11, compact=True):
    kwargs = dict(capacity=capacity, n_dim=2, explicit_next=True,
                  compact_depth_history=compact, depth_history_offsets=(3, 1, 0),
                  output_key_t=["observation", "action", "done"],
                  output_key_tp1=["observation", "terminated", "truncated", "qpos"])
    buffers = [TrajectoryDictBufferMultiDim(**kwargs, sparse_next=sparse) for sparse in (False, True)]
    for buffer in buffers:
        buffer._get_idxs = get_idxs
    return buffers


def observation(value, commands=True):
    obs = {"state": value.float(), "privileged_state": value.float().repeat(1, 7),
           "depth_image": value.to(torch.uint8)[:, :, None, None].expand(-1, 3, 2, 2).clone()}
    if commands:
        obs.update(heading=torch.zeros(len(value), 2), velocity_command=torch.zeros(len(value), 3))
    return obs


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("commands", [False, True])
@pytest.mark.parametrize("capacity", [1, 11])
def test_sparse_dense_parity_over_resets_wrap_save_and_append(tmp_path, compact, commands, capacity):
    dense, sparse = pair(capacity, compact)
    current = torch.tensor([[10], [70], [130]])
    for step in range(35):
        obs, nxt = observation(current, commands), observation(current + 1, commands)
        if commands:
            obs["heading"].fill_(step)
            obs["velocity_command"][:] = torch.tensor([.8, 0., step % 2])
            nxt["heading"].fill_(-99)
            nxt["velocity_command"].fill_(-77)
        ended = torch.tensor([[step % 4 == 2], [step % 7 == 3], [step % 3 == 1]])
        row = dict(observation=obs, action=current.float(), done=ended,
                   next=dict(observation=nxt, terminated=ended, truncated=torch.zeros_like(ended), qpos=current.float() + 1))
        for buffer in (dense, sparse):
            buffer.extend(tree_map(lambda v: v[None], row))
        current += 1
        current[ended] += 20
        if step in (5, 17):
            for buffer in (dense, sparse):
                buffer.end_episodes()
            current += 10
        for buffer in (dense, sparse):
            torch.manual_seed(step)
            batch = buffer.sample(256)
            if buffer is dense:
                expected = batch
            else:
                assert_same(batch, expected)
        assert_same(dense.get_full_buffer(), sparse.get_full_buffer())
        sparse._validate_sparse_storage()
        if step in (0, 5, 18, 34):
            sparse.save(tmp_path)
            # Overrides a saved CUDA device without requiring CUDA on the reader.
            import json
            cfg = json.loads((tmp_path / "config.json").read_text())
            cfg["device"] = "cuda:99"
            (tmp_path / "config.json").write_text(json.dumps(cfg))
            sparse = TrajectoryDictBufferMultiDim.load_for_training(tmp_path, device="cpu")
            sparse._get_idxs = get_idxs
            assert_same(dense.get_full_buffer(), sparse.get_full_buffer())
    assert "state" not in sparse.storage["next"]["observation"]
    assert len(sparse._terminal_observations) <= capacity


def test_sparse_batched_extend_and_sequence_sampling():
    dense, sparse = pair(compact=False)
    current = torch.arange(8).reshape(8, 1, 1).expand(-1, 2, -1).float()
    row = dict(observation={"state": current}, action=current, done=torch.zeros(8, 2, 1, dtype=torch.bool),
               next=dict(observation={"state": current + 1}, terminated=torch.zeros(8, 2, 1, dtype=torch.bool),
                         truncated=torch.zeros(8, 2, 1, dtype=torch.bool), qpos=current + 1))
    for buffer in (dense, sparse):
        buffer.extend(row)
        torch.manual_seed(1)
        batch = buffer.sample(4096, seq_length=4)
        if buffer is dense:
            expected = batch
        else:
            assert_same(batch, expected)
    assert sparse._terminal_observations == {}


@pytest.mark.parametrize("n_dim,explicit", [(1, True), (2, False)])
def test_sparse_requires_action_aligned_vector_replay(n_dim, explicit):
    with pytest.raises(ValueError, match="Sparse successors require"):
        TrajectoryDictBufferMultiDim(capacity=4, n_dim=n_dim, explicit_next=explicit, sparse_next=True)


def test_sparse_rejects_mismatched_batches_before_writing():
    _, sparse = pair(compact=False)
    value = torch.ones(2, 3, 1)
    flags = value.bool()
    row = dict(observation={"state": value}, action=value[:1], done=flags,
               next=dict(observation={"state": value}, terminated=flags, truncated=~flags, qpos=value))
    with pytest.raises(ValueError, match="batch dimensions"):
        sparse.extend(row)
    assert sparse.empty() and sparse.storage is None


@pytest.mark.parametrize("corruption", ["missing_group", "missing_row", "wrong_ids", "wrong_shape"])
def test_sparse_load_rejects_incomplete_successors(tmp_path, corruption):
    _, sparse = pair(compact=False)
    value = torch.ones(1, 2, 1)
    flags = value.bool()
    sparse.extend(dict(observation={"state": value}, action=value, done=flags,
                       next=dict(observation={"state": value + 1}, terminated=flags, truncated=~flags, qpos=value)))
    sparse.save(tmp_path)
    with h5py.File(tmp_path / "buffer.hdf5", "a") as file:
        if corruption == "missing_group":
            del file["__extras__"]
        elif corruption == "missing_row":
            del file["__extras__/terminal/0"]
        elif corruption == "wrong_ids":
            file["__extras__/terminal/0/env_ids"][:] = [1, 0]
        else:
            del file["__extras__/pending/state"]
            file["__extras__/pending/state"] = torch.zeros(2, 7).numpy()
    with pytest.raises(ValueError, match="Sparse replay"):
        TrajectoryDictBufferMultiDim.load_for_training(tmp_path)


def test_sparse_and_dense_identical_agent_update():
    from test_qv_agent_integration import make_agent, make_batch, Provider
    torch.manual_seed(7)
    agent, dims = make_agent()
    other = copy.deepcopy(agent)
    batch = make_batch(agent, dims)
    kwargs = dict(capacity=4, n_dim=2, explicit_next=True,
                  output_key_t=list(batch.keys() - {"next"}), output_key_tp1=list(batch["next"]))
    buffers = [TrajectoryDictBufferMultiDim(**kwargs, sparse_next=sparse) for sparse in (False, True)]
    first = tree_map(lambda v: v[:4].clone(), batch)
    first["next"]["terminated"][:2] = True
    first["done"] = first["next"]["terminated"].clone()
    second = copy.deepcopy(first)
    second["observation"] = copy.deepcopy(first["next"]["observation"])
    for value in second["observation"].values():
        value[:2] += 1
    second["observation"]["heading"].fill_(3)
    second["observation"]["velocity_command"].zero_()
    for row in (first, second):
        for buffer in buffers:
            buffer.extend(tree_map(lambda v: v[None], row))
    indices = torch.cartesian_prod(torch.arange(2), torch.arange(4))
    batches = [buffer._gather_transitions(indices) for buffer in buffers]
    assert_same(*batches)
    for model, data in zip((agent, other), batches):
        torch.manual_seed(51)
        metrics = model.update({"train": Provider(data), "expert_slicer": Provider(batch, expert=True)}, 1024)
        assert torch.isfinite(metrics["velocity/critic_loss"])
        if model is agent:
            expected = metrics
        else:
            for key in ("velocity/critic_loss", "actor_gradient_local/V_norm"):
                assert_same(metrics[key], expected[key])
    assert_same(agent._model.state_dict(), other._model.state_dict())


def test_memory_saved_uses_actual_tensor_bytes(tmp_path):
    import json
    from pathlib import Path
    cfg = json.loads((Path(__file__).resolve().parents[1] / "pbfm/configs/legacy_large.json").read_text())
    dense, sparse = pair(capacity=100, compact=True)
    for step in range(100):
        value = torch.full((4, 1), step)
        flags = torch.full((1, 4, 1), step % 50 == 49)
        obs = observation(value)
        # Production body/depth dimensions, without allocating a production-size replay.
        obs.update(state=torch.zeros(4, 64), privileged_state=torch.zeros(4, 463),
                   last_action=torch.zeros(4, 29), history_actor=torch.zeros(4, 372),
                   velocity_target_yaw=torch.zeros(4, 3), terrain_priv=torch.zeros(4, 273),
                   depth_image=torch.zeros(4, 3, 18, 32, dtype=torch.uint8))
        row = dict(observation=obs, action=torch.zeros(4, 29), done=flags[0],
                   next=dict(observation=obs, terminated=flags[0], truncated=~flags[0],
                             qpos=torch.zeros(4, 36), qvel=torch.zeros(4, 35)),
                   qpos=torch.zeros(4, 36), qvel=torch.zeros(4, 35),
                   z=torch.zeros(4, 256), heading_z_next=torch.zeros(4, 256),
                   step_count=torch.zeros(4, 1, dtype=torch.long), reward=torch.zeros(4, 1),
                   aux_rewards={key: torch.zeros(4, 1) for key in cfg["agent"]["aux_rewards"]})
        for key in ("terminated", "truncated", "transition_terminated", "transition_truncated",
                    "heading_valid", "heading_context_continues", "velocity_valid"):
            row[key] = flags[0]
        for key in ("heading_context_id", "heading_source_type"):
            row[key] = torch.zeros(4, 1, dtype=torch.long)
        for key, width in (("velocity_next", 3), ("velocity_reward", 1), ("velocity_error_mps", 1),
                           ("heading_next", 2), ("heading_reward", 1), ("root_heading_xy", 2), ("next_root_heading_xy", 2)):
            row[key] = torch.zeros(4, width)
        for buffer in (dense, sparse):
            buffer.extend(tree_map(lambda v: v[None], row))
    def bytes_used(buffer):
        tensors, _ = tree_flatten((buffer.storage, buffer._terminal_observations, buffer._pending_observation))
        return sum(v.numel() * v.element_size() for v in tensors if isinstance(v, torch.Tensor))
    before, after = bytes_used(dense), bytes_used(sparse)
    print(f"\nReplay tensors: dense={before} bytes, sparse={after} bytes, saved={(before-after)/before:.2%}")
    assert after < before * .65
    dense.save(tmp_path / "dense")
    sparse.save(tmp_path / "sparse")
    disk_before = (tmp_path / "dense/buffer.hdf5").stat().st_size
    disk_after = (tmp_path / "sparse/buffer.hdf5").stat().st_size
    print(f"Replay HDF5: dense={disk_before} bytes, sparse={disk_after} bytes, saved={(disk_before-disk_after)/disk_before:.2%}")
    assert disk_after < disk_before * .67
