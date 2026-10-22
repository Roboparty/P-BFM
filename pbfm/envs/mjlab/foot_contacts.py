"""RP1 stumble reduction over every foot contact, before force cancellation."""

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mjlab.sensor.sensor import Sensor, SensorCfg


@wp.kernel
def _reduce_stumble(
    nacon: wp.array[int],
    worldids: wp.array[int],
    geoms: wp.array[wp.vec2i],
    nefc: wp.array[int],
    njmax: int,
    foot_geoms: wp.array[int],
    forces: wp.array[wp.spatial_vector],
    cost: wp.array[float],
):
    i = wp.tid()
    if nacon[0] > forces.shape[0]:
        if i < cost.shape[0]:
            cost[i] = wp.inf
        return
    if i < cost.shape[0] and nefc[i] > njmax:
        wp.atomic_max(cost, i, wp.inf)
    if i >= nacon[0]:
        return
    pair = geoms[i]
    if (pair[0] >= 0 and foot_geoms[pair[0]] != 0) or (pair[1] >= 0 and foot_geoms[pair[1]] != 0):
        f = forces[i]
        value = wp.inf
        if wp.isfinite(f[0]) and wp.isfinite(f[1]) and wp.isfinite(f[2]):
            excess = wp.sqrt(f[0] * f[0] + f[1] * f[1]) - wp.abs(f[2])
            value = wp.min(wp.max((excess - 20.0) / 100.0, 0.0), 1.0)
        wp.atomic_max(cost, worldids[i], value)


@dataclass
class FootStumbleSensorCfg(SensorCfg):
    body_names: tuple[str, ...]
    history_length: int

    def build(self):
        return FootStumbleSensor(self)


class FootStumbleSensor(Sensor[torch.Tensor]):
    def __init__(self, cfg: FootStumbleSensorCfg):
        super().__init__()
        if cfg.history_length < 1:
            raise ValueError("Foot contact history must include at least one physics step")
        self.cfg = cfg

    def edit_spec(self, scene_spec, entities) -> None:
        pass

    def initialize(self, mj_model, model, data, device: str) -> None:
        self._model, self._data = model.struct, data.struct
        foot_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.cfg.body_names]
        if not foot_ids or min(foot_ids) < 0:
            raise ValueError(f"Unknown foot bodies: {self.cfg.body_names}")
        self._foot_geoms = wp.array(np.isin(mj_model.geom_bodyid, foot_ids).astype(np.int32), device=device)
        self._contact_ids = wp.array(np.arange(self._data.naconmax, dtype=np.int32), device=device)
        self._forces = wp.empty(self._data.naconmax, dtype=wp.spatial_vector, device=device)
        self._cost = wp.zeros(self._data.nworld, dtype=float, device=device)
        self._history = torch.zeros(self._data.nworld, self.cfg.history_length, device=device)

    def update(self, dt: float) -> None:
        super().update(dt)
        if dt <= 0:
            return
        self._cost.zero_()
        mjwarp.contact_force(self._model, self._data, self._contact_ids, True, self._forces)
        wp.launch(_reduce_stumble, dim=max(self._data.naconmax, self._data.nworld), inputs=[
            self._data.nacon, self._data.contact.worldid, self._data.contact.geom,
            self._data.nefc, self._data.njmax,
            self._foot_geoms, self._forces, self._cost], device=self._cost.device)
        self._history[:, 1:] = self._history[:, :-1].clone()
        self._history[:, 0] = wp.to_torch(self._cost)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._history[slice(None) if env_ids is None else env_ids] = 0

    def _compute_data(self) -> torch.Tensor:
        cost = self._history.amax(dim=1)
        if not torch.isfinite(cost).all():
            raise RuntimeError("Invalid RP1 foot contacts: contact/constraint capacity exceeded or non-finite force")
        return cost
