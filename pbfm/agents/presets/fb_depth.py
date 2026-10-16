"""The final PBFM depth preset, sourced from the archived run configuration."""

from __future__ import annotations

import json
import math
from typing import Any

from pbfm.agents.fb_cpr_aux.config import FBcprAuxAgentConfig
from pbfm.utils.resource_paths import CONFIG_ROOT


def load_final_config() -> dict[str, Any]:
    """Return a fresh copy; never fall back to historical generic defaults."""
    return json.loads((CONFIG_ROOT / "legacy_large.json").read_text())


def build_fb_depth_agent(
    *,
    device: str | None = None,
    compile: bool | None = None,
    update_z_every_step: int | None = None,
    lr_scale: float = 1.0,
    clip_grad_norm: float | None = None,
    cartwheel_aux_safe: bool = False,
    heading_context: bool = True,
    heading_reg_coeff: float | None = None,
    behavior_prior: bool = True,
) -> FBcprAuxAgentConfig:
    """Preserve final architecture/rewards unless an override is explicit.

    QV currently requires the heading context and critic. Reject unsupported
    heading ablations rather than silently disabling QV or velocity estimation.
    """
    raw = load_final_config()["agent"]
    train = raw["train"]
    if not heading_context or (heading_reg_coeff is not None and heading_reg_coeff <= 0):
        raise ValueError("Final fb_depth QV requires heading_context=True and positive heading_reg_coeff")
    if not math.isfinite(lr_scale) or lr_scale <= 0:
        raise ValueError("lr_scale must be finite and positive")
    if heading_reg_coeff is not None:
        if not math.isfinite(heading_reg_coeff):
            raise ValueError("heading_reg_coeff must be finite")
        train["reg_coeff_heading"] = float(heading_reg_coeff)
    if device is not None:
        raw["model"]["device"] = device
    if compile is not None:
        raw["compile"] = bool(compile)
    if update_z_every_step is not None:
        if update_z_every_step <= 0:
            raise ValueError("update_z_every_step must be positive")
        train["update_z_every_step"] = int(update_z_every_step)
    if clip_grad_norm is not None:
        if not math.isfinite(clip_grad_norm) or clip_grad_norm < 0:
            raise ValueError("clip_grad_norm must be finite and non-negative")
        train["clip_grad_norm"] = float(clip_grad_norm)
    # D retains the calibrated 5e-6 learning rate; explicit scaling affects
    # Actor, FB and the value critics, including QV.
    for key in train:
        if key.startswith("lr_") and key != "lr_discriminator":
            train[key] *= lr_scale
    train["behavior_prior_enabled"] = bool(behavior_prior)
    if not behavior_prior:
        train["reg_coeff"] = 0.0
    if cartwheel_aux_safe:
        raw["aux_rewards"] = [
            "penalty_torques", "penalty_action_rate", "limits_dof_pos", "limits_torque",
        ]
        raw["aux_rewards_scaling"] = {
            "penalty_action_rate": -0.03, "limits_dof_pos": -10.0,
            "penalty_torques": 0.0, "limits_torque": 0.0,
        }
    return FBcprAuxAgentConfig.model_validate(raw, strict=False)


__all__ = ["build_fb_depth_agent", "load_final_config"]
