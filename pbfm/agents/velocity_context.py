"""QV v1 coordinate and transition contract.

Commands are metres/second in the robot's *yaw-only* frame, not the tilted
body frame. They are explicit task inputs (vx, vy, valid), never estimates
of the robot's true velocity. Reference world vectors are aligned with the
rollout start yaw using the same fixed rotation as reference headings.

Validity is semantic, assigned to an entire source motion by a reviewed
manifest. In particular, zero reference speed does NOT invalidate a stop.
Random/goal contexts have no invented velocity target. QV v1 is a finite
context task, and never bootstraps through a context switch or environment
reset. An auto-reset observation is not a terminal velocity measurement.
"""
from dataclasses import dataclass
import math
import re
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F


def locomotion_motion_name(name: str) -> bool:
    """V1 explicit source rule, independent of speed and learned success.

    Accept LaFAN walk/run families only. No dance, getup, fight, cartwheel
    or arbitrary goal. Dump accepted names before launching the experiment.
    """
    return bool(re.match(r"^(walk|run)(?:\d|_|$)", Path(str(name)).stem.lower()))


REFERENCE_VELOCITY_CONTRACT = "explicit_reference_finite_aligned_temporal_v3"


def temporal_reference_velocity(
    velocity: Tensor,
    *,
    mode: str = "instantaneous",
    horizon: int = 1,
) -> Tensor:
    """Apply the versioned temporal command contract to one [T,2] reference.

    ``future_mean`` uses the same forward-looking, tail-clipped indexing as
    the behavior latent: frame ``t`` averages ``[t, min(t+horizon, T))``.
    It never crosses a motion boundary because callers pass one motion at a
    time.  Fixed reward/teleoperation commands do not call this function.
    """
    if velocity.ndim != 2 or velocity.shape[-1] != 2 or len(velocity) < 1:
        raise ValueError("Reference velocity must have shape [T,2] with T >= 1")
    if not torch.isfinite(velocity).all():
        raise ValueError("Reference velocity contains nonfinite values")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Reference velocity temporal horizon must be a positive integer")
    if mode == "instantaneous":
        return velocity.clone()
    if mode != "future_mean":
        raise ValueError(f"Unknown reference velocity temporal mode: {mode}")

    count = velocity.shape[0]
    starts = torch.arange(count, device=velocity.device)
    ends = torch.clamp(starts + horizon, max=count)
    prefix = torch.cat((velocity.new_zeros((1, 2)), velocity.cumsum(dim=0)), dim=0)
    totals = prefix[ends] - prefix[starts]
    lengths = (ends - starts).to(dtype=velocity.dtype).unsqueeze(-1)
    return totals / lengths


def reference_velocity_enabled(name: str, policy: str = "walk_run") -> bool:
    """Only call for an actual, time-aligned reference (never random/goal z).

    Stationary phases remain valid. Family labels are not a speed threshold.
    Finite/alignment checks are mandatory at the reference ingestion boundary.
    """
    if policy not in ("walk_run", "all_reference"):
        raise ValueError(f"Unknown reference velocity policy: {policy}")
    return policy == "all_reference" or locomotion_motion_name(name)


def validate_reference_velocity(velocity: Tensor, heading: Tensor) -> None:
    """Fail closed for corrupt/misaligned sources, not for legitimate zero speed."""
    if velocity.ndim != 2 or velocity.shape[-1] != 2 or velocity.shape != heading.shape:
        raise ValueError("Reference root velocity and heading must have identical [T,2] frames")
    if len(velocity) < 1 or not torch.isfinite(velocity).all():
        raise ValueError("Missing/nonfinite reference root velocity")
    unit_heading(heading)


@dataclass(frozen=True)
class VelocityTaskContract:
    version: str = "yaw_xy_huber_finite_context_v1"
    error_scale_mps: float = 0.5
    huber_delta: float = 1.0

    def __post_init__(self):
        for name in ("error_scale_mps", "huber_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def unit_heading(heading: Tensor) -> Tensor:
    if heading.shape[-1] != 2:
        raise ValueError("Heading must contain two horizontal components")
    norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    if not torch.isfinite(heading).all() or (norm < 1e-6).any():
        raise ValueError("Invalid yaw heading; do not silently invent forward")
    return heading / norm


def world_to_yaw(vector_xy: Tensor, heading_xy: Tensor) -> Tensor:
    """Inverse yaw rotation. Preserves vector magnitude, including zero."""
    h = unit_heading(heading_xy)
    x, y = vector_xy.unbind(-1)
    c, s = h.unbind(-1)
    return torch.stack((c * x + s * y, -s * x + c * y), dim=-1)


def yaw_to_world(vector_xy: Tensor, heading_xy: Tensor) -> Tensor:
    h = unit_heading(heading_xy)
    x, y = vector_xy.unbind(-1)
    c, s = h.unbind(-1)
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def align_reference_velocity(reference_world_xy: Tensor, reference_anchor_heading: Tensor,
                             rollout_anchor_heading: Tensor) -> Tensor:
    """Apply a fixed start-yaw alignment, NOT a per-frame heading reorientation."""
    return yaw_to_world(world_to_yaw(reference_world_xy, reference_anchor_heading),
                        rollout_anchor_heading)


def velocity_observation(target_world_xy: Tensor, current_heading_xy: Tensor,
                         valid: Tensor) -> Tensor:
    """Actor input [target_vx, target_vy, valid]. Invalid commands are zeros."""
    local = world_to_yaw(target_world_xy, current_heading_xy)
    valid = valid.to(device=local.device, dtype=torch.bool)
    if valid.shape != local.shape[:-1] + (1,):
        raise ValueError("valid must have shape (..., 1)")
    return torch.cat((torch.where(valid, local, torch.zeros_like(local)), valid.to(local.dtype)), -1)


def velocity_reward(actual_world_xy: Tensor, target_world_xy: Tensor, valid: Tensor,
                    contract: VelocityTaskContract = VelocityTaskContract()) -> Tensor:
    """Fixed-scale summed Huber penalty, evaluated in the shared world plane.

    Rotation-invariant squared L2 is used inside a radial Huber penalty;
    unlike componentwise Huber this cannot change with the world's yaw.
    This function is reward computation, not a differentiable policy loss.
    """
    if actual_world_xy.shape != target_world_xy.shape or actual_world_xy.shape[-1] != 2:
        raise ValueError("actual/target velocity must have equal (..., 2) shapes")
    if valid.shape != actual_world_xy.shape[:-1] + (1,):
        raise ValueError("valid must have shape (..., 1)")
    valid = valid.to(device=actual_world_xy.device, dtype=torch.bool)
    error = torch.where(valid, actual_world_xy - target_world_xy, torch.zeros_like(actual_world_xy))
    scaled_norm = torch.linalg.vector_norm(error / contract.error_scale_mps, dim=-1, keepdim=True)
    if not torch.isfinite(scaled_norm).all():
        raise ValueError("Nonfinite valid velocity measurement")
    return -F.huber_loss(scaled_norm, torch.zeros_like(scaled_norm),
                         reduction="none", delta=contract.huber_delta)


def velocity_transition_masks(*, current_valid: Tensor, next_valid: Tensor,
                              context_id: Tensor, next_context_id: Tensor,
                              terminated: Tensor, truncated: Tensor,
                              outcome_observed: Tensor, gamma: float) -> tuple[Tensor, Tensor]:
    """Return (critic/actor validity, Bellman discount), both shape (N,1).

    outcome_observed is false for an auto-reset next observation unless a
    genuine terminal measurement was separately captured. Never learn the
    last action's reward using the reset robot's velocity.
    """
    tensors = (current_valid, next_valid, context_id, next_context_id,
               terminated, truncated, outcome_observed)
    if any(t.shape != current_valid.shape for t in tensors) or current_valid.shape[-1] != 1:
        raise ValueError("Transition masks and context IDs must have equal (...,1) shapes")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma outside [0,1]")
    usable = current_valid.bool() & outcome_observed.bool()
    continues = (usable & next_valid.bool() & (context_id == next_context_id)
                 & ~terminated.bool() & ~truncated.bool())
    return usable, continues.float() * gamma


def merge_velocity_context(original: Tensor, sampled: Tensor, relabel_mask: Tensor) -> Tensor:
    """Use the exact SAME selection mask as z/heading; no independent sampling."""
    if original.shape != sampled.shape or original.shape[-1] != 3:
        raise ValueError("Velocity contexts must have equal (...,3) shapes")
    if relabel_mask.shape != original.shape[:-1] + (1,):
        raise ValueError("Relabel mask shape does not match the behavior context")
    return torch.where(relabel_mask.bool(), sampled, original)
