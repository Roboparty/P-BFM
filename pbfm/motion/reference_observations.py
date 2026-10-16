"""Reference observation helpers shared by expert and inference paths."""

import torch

from pbfm.utils.torch_utils import quat_rotate_inverse


def reference_base_ang_vel(env, base_quat: torch.Tensor, world_ang_vel: torch.Tensor) -> torch.Tensor:
    """Match simulator actor-observation preprocessing for base angular velocity."""
    local_ang_vel = quat_rotate_inverse(base_quat, world_ang_vel, w_last=True)
    scale = float(env.config.obs.obs_scales.get("base_ang_vel", 1.0))
    return local_ang_vel * scale
