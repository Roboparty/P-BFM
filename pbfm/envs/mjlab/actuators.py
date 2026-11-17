"""RP1 peak/constant-torque/roll-off envelope; G1 retains MJLab DC motors."""

from dataclasses import dataclass
import math
import torch
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg


def rp1_clip_effort(effort, velocity, peak, keep_velocity, velocity_limit):
    drive = peak * ((velocity_limit - velocity.abs()) / (velocity_limit - keep_velocity)).clamp(0, 1)
    upper = torch.where(velocity >= 0, drive, peak)
    lower = torch.where(velocity <= 0, -drive, -peak)
    return effort.clamp(min=lower, max=upper)


@dataclass(kw_only=True)
class RP1ActuatorCfg(IdealPdActuatorCfg):
    keep_velocity: float
    velocity_limit: float
    rfi_lim: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.keep_velocity < self.velocity_limit:
            raise ValueError("RP1 motor requires 0 <= keep_velocity < velocity_limit")
        if not math.isfinite(self.rfi_lim) or not 0 <= self.rfi_lim <= 1:
            raise ValueError("RP1 RFI must be a finite fraction in [0, 1]")

    def build(self, entity, target_ids, target_names):
        return RP1Actuator(self, entity, target_ids, target_names)


class RP1Actuator(IdealPdActuator):
    def compute(self, cmd):
        self._joint_velocity = cmd.vel
        return super().compute(cmd)

    def _clip_effort(self, effort):
        if self.cfg.rfi_lim:
            effort = effort + (2 * torch.rand_like(effort) - 1) * self.cfg.rfi_lim * self.force_limit
        # Beyond no-load speed, braking remains available but zero demand stays zero.
        return rp1_clip_effort(effort, self._joint_velocity, self.force_limit,
                               self.cfg.keep_velocity, self.cfg.velocity_limit)


def reset_motor_delay(env, env_ids, delay_range):
    """The source RP1 motor group shares one per-episode physics-step lag."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    lags = torch.randint(delay_range[0], delay_range[1] + 1, (len(env_ids),), device=env.device)
    for actuator in env.scene["robot"].actuators:
        actuator.set_lags(lags, env_ids)
