"""Reward relabeling utilities used by MJLab reward inference.

This module is intentionally self-contained and does not import the legacy G1 gym environment. It only reuses the local MuJoCo reward functions.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import mujoco
import numpy as np
import torch

from pbfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from pbfm.envs.mjlab import rewards as g1_rewards
from pbfm.envs.mjlab.rewards import RewardFunction

TERRAIN_REFERENCE_RAY_INDEX = 58


def _detached_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def canonicalize_terrain_relabel_qpos(
    qpos: np.ndarray,
    next_obs: Any,
    *,
    reference_ray_index: int = TERRAIN_REFERENCE_RAY_INDEX,
    atol: float = 1e-5,
) -> np.ndarray:
    """Translate terrain rollouts to the flat-ground height convention used by rewards."""
    if not isinstance(next_obs, dict) or not ({"terrain_actor", "terrain_priv"} & next_obs.keys()):
        return qpos
    if "privileged_state" not in next_obs:
        raise RuntimeError("terrain reward relabeling requires next privileged_state")

    qpos_array = np.asarray(qpos)
    privileged_state = _detached_numpy(next_obs["privileged_state"])
    if qpos_array.ndim != 2 or qpos_array.shape[1] < 3:
        raise ValueError(f"terrain reward relabeling expects qpos shape [N, >=3], got {qpos_array.shape}")
    if privileged_state.ndim != 2 or privileged_state.shape[0] != qpos_array.shape[0] or privileged_state.shape[1] < 1:
        raise ValueError(
            "terrain reward relabeling expects privileged_state shape [N, >=1] matching qpos; "
            f"got qpos={qpos_array.shape}, privileged_state={privileged_state.shape}"
        )

    root_clearance = privileged_state[:, 0]
    if not np.isfinite(qpos_array).all() or not np.isfinite(root_clearance).all():
        raise RuntimeError("terrain reward relabeling received non-finite qpos or root clearance")

    if "terrain_actor" in next_obs:
        terrain_actor = _detached_numpy(next_obs["terrain_actor"])
        if terrain_actor.ndim != 2 or terrain_actor.shape[0] != qpos_array.shape[0]:
            raise ValueError(
                "terrain reward relabeling expects terrain_actor shape [N, rays] matching qpos; "
                f"got qpos={qpos_array.shape}, terrain_actor={terrain_actor.shape}"
            )
        if not 0 <= reference_ray_index < terrain_actor.shape[1]:
            raise IndexError(
                f"terrain reference ray index {reference_ray_index} is invalid for {terrain_actor.shape[1]} rays"
            )
        center_clearance = terrain_actor[:, reference_ray_index]
        if not np.isfinite(center_clearance).all():
            raise RuntimeError("terrain reward relabeling received non-finite terrain_actor clearances")
        if not np.allclose(center_clearance, root_clearance, atol=atol, rtol=0.0):
            max_error = float(np.max(np.abs(center_clearance - root_clearance)))
            raise RuntimeError(
                "terrain reward relabeling found inconsistent root clearances between privileged_state and "
                f"terrain_actor center ray; max_error={max_error:.6g}"
            )

    canonical_qpos = qpos_array.copy()
    canonical_qpos[:, 2] = root_clearance
    return canonical_qpos


def get_next(field: str, data: Any):
    if "next" in data and field in data["next"]:
        return data["next"][field]
    if f"next_{field}" in data:
        return data[f"next_{field}"]
    raise ValueError(f"No next of {field} found in data.")


def make_reward_from_name(name: str | None) -> RewardFunction:
    for _class_name, reward_cls in inspect.getmembers(g1_rewards, inspect.isclass):
        if not issubclass(reward_cls, RewardFunction) or inspect.isabstract(reward_cls):
            continue
        reward_obj = reward_cls.reward_from_name(name)
        if reward_obj is not None:
            return reward_obj
    raise ValueError(f"Unknown reward name: {name}")


@dataclasses.dataclass(kw_only=True)
class RewardWrapperHV:
    model: Any
    inference_dataset: Any
    num_samples_per_inference: int
    inference_function: str
    max_workers: int
    process_executor: bool = False
    process_context: str = "spawn"
    env_model: str | mujoco.MjModel = "pbfm/data/robots/g1_mjlab/g1_29dof.xml"

    @property
    def device(self) -> Any:
        return self.unwrapped_model.device

    @property
    def unwrapped_model(self):
        if hasattr(self.model, "unwrapped_model"):
            return self.model.unwrapped_model
        return self.model

    def reward_inference(self, task: str) -> torch.Tensor:
        if isinstance(self.env_model, str):
            self.env_model = mujoco.MjModel.from_xml_path(self.env_model)

        if isinstance(self.inference_dataset, TrajectoryDictBufferMultiDim):
            if "qpos" not in self.inference_dataset.output_key_tp1:
                self.inference_dataset.output_key_tp1.append("qpos")
            if "qvel" not in self.inference_dataset.output_key_tp1:
                self.inference_dataset.output_key_tp1.append("qvel")

        if self.num_samples_per_inference >= self.inference_dataset.size() and hasattr(self.inference_dataset, "get_full_buffer"):
            data = self.inference_dataset.get_full_buffer()
        else:
            data = self.inference_dataset.sample(self.num_samples_per_inference)

        qpos = get_next("qpos", data)
        qvel = get_next("qvel", data)
        next_obs = get_next("observation", data)
        action = data["action"]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().detach().numpy()
            qvel = qvel.cpu().detach().numpy()
            action = action.cpu().detach().numpy()
        qpos = canonicalize_terrain_relabel_qpos(qpos, next_obs)

        rewards = relabel(
            self.env_model,
            qpos,
            qvel,
            action,
            make_reward_from_name(task),
            max_workers=self.max_workers,
            process_executor=self.process_executor,
            process_context=self.process_context,
        )

        td = {"reward": torch.tensor(rewards, dtype=torch.float32, device=self.device)}
        if "B" in data:
            td["B_vect"] = data["B"]
        else:
            td["next_obs"] = next_obs
        inference_fn = getattr(self.model, self.inference_function, None)
        if inference_fn is None:
            raise AttributeError(f"Model does not define {self.inference_function!r}")
        return inference_fn(**td).reshape(1, -1)


def _relabel_worker(
    x,
    model: mujoco.MjModel,
    reward_fn: RewardFunction,
):
    qpos, qvel, action = x
    assert len(qpos.shape) > 1
    assert qvel.shape[0] == qpos.shape[0]
    assert qvel.shape[0] == action.shape[0]
    rewards = np.zeros((qpos.shape[0], 1))
    for i in range(qpos.shape[0]):
        rewards[i] = reward_fn(model, qpos[i], qvel[i], action[i])
    return rewards


def relabel(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    reward_fn: RewardFunction,
    max_workers: int = 5,
    process_executor: bool = False,
    process_context: str = "spawn",
):
    chunk_size = int(np.ceil(qpos.shape[0] / max_workers))
    args = [(qpos[i : i + chunk_size], qvel[i : i + chunk_size], action[i : i + chunk_size]) for i in range(0, qpos.shape[0], chunk_size)]
    if max_workers == 1:
        result = [_relabel_worker(args[0], model=model, reward_fn=reward_fn)]
    elif process_executor:
        import multiprocessing

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context(process_context),
        ) as exe:
            f = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
            result = exe.map(f, args)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            f = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
            result = exe.map(f, args)

    return np.concatenate([r for r in result])
