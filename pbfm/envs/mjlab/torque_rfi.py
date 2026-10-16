"""Per-physics-step external joint torque noise; never part of deployment."""
from dataclasses import dataclass
import math
import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class RfiJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._qfrc_ids = self._entity.indexing.joint_v_adr[self._target_ids]
        self._limits = torch.tensor(cfg.torque_limits, device=env.device).unsqueeze(0)
        if self._limits.shape[1] != len(self._qfrc_ids):
            raise ValueError('RFI torque limits must follow the action joint order')
        if not torch.isfinite(self._limits).all() or not (self._limits > 0).all():
            raise ValueError('RFI torque limits must be finite and positive')
        if not math.isfinite(cfg.rfi_lim) or not 0 <= cfg.rfi_lim <= 1:
            raise ValueError('RFI fraction must be in [0, 1]')
        self.last_noise = torch.zeros((env.num_envs, self._limits.shape[1]), device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.last_noise[ids] = 0
        if env_ids is None:
            self._env.sim.data.qfrc_applied[:, self._qfrc_ids] = 0
        else:
            self._env.sim.data.qfrc_applied[env_ids[:, None], self._qfrc_ids] = 0

    def apply_actions(self):
        super().apply_actions()
        self.last_noise = (2 * torch.rand_like(self.last_noise) - 1) * self.cfg.rfi_lim * self._limits
        # Own these joint-force slots; leave base forces and external pushes
        # untouched, and overwrite rather than accumulate across substeps.
        self._env.sim.data.qfrc_applied[:, self._qfrc_ids] = self.last_noise


@dataclass(kw_only=True)
class RfiJointPositionActionCfg(JointPositionActionCfg):
    torque_limits: tuple[float, ...]
    rfi_lim: float = 0.1

    def build(self, env):
        return RfiJointPositionAction(self, env)
