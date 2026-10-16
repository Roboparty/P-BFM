"""Deployable task commands shared by independent inference entrypoints.

Reference frame t commands action t; z may encode future frames independently.
Tracking vectors use a fixed start-yaw alignment, never the robot's current yaw
to rotate the *world target*. The latter only expresses it as an Actor input.
"""
import math
import re
from dataclasses import dataclass

import torch

from pbfm.agents.behavior_context import align_heading_sequence, repeat_heading_sequence
from pbfm.agents.velocity_context import (
    reference_velocity_enabled,
    temporal_reference_velocity,
    validate_reference_velocity,
    velocity_observation,
    world_to_yaw,
    yaw_to_world,
)


def aligned_tracking_velocity(reference_velocity, reference_heading, aligned_heading):
    """Same fixed rotation used for heading. Also supports turning repeats."""
    if reference_velocity.shape != reference_heading.shape or reference_heading.shape != aligned_heading.shape:
        raise ValueError("Reference velocity/headings must have identical [T,2] shapes")
    return yaw_to_world(world_to_yaw(reference_velocity, reference_heading), aligned_heading)


def tracking_velocity_valid(motion_name, mode="auto", policy="walk_run"):
    if mode not in ("auto", "valid", "invalid"):
        raise ValueError(f"Unknown velocity context mode: {mode}")
    return mode == "valid" or (mode == "auto" and reference_velocity_enabled(motion_name, policy))


@dataclass
class TrackingCommands:
    heading_world: torch.Tensor
    velocity_world: torch.Tensor
    velocity_valid: torch.Tensor
    motion_name: str
    mode: str
    temporal_mode: str
    temporal_horizon: int

    def velocity_at(self, step, current_heading):
        if not 0 <= step < len(self.heading_world):
            raise IndexError("Tracking context exhausted; do not silently reuse the last command")
        return velocity_observation(self.velocity_world[step:step + 1], current_heading,
                                    self.velocity_valid[step:step + 1])


def build_tracking_commands(reference_heading, reference_velocity, initial_heading, *,
                            motion_name, frames, mode="auto", repeats=1, policy="walk_run",
                            temporal_mode="instantaneous", temporal_horizon=1):
    if frames < 1 or len(reference_heading) < frames:
        raise ValueError("Tracking reference must cover every action frame")
    heading = reference_heading[:frames]
    # Transform the complete motion before selecting action frames.  This is
    # identical to expert-cache construction and preserves the final available
    # reference frame when a tracking z sequence is one frame shorter.
    velocity = temporal_reference_velocity(
        reference_velocity, mode=temporal_mode, horizon=int(temporal_horizon)
    )[:frames]
    if velocity.shape != heading.shape or not torch.isfinite(velocity).all():
        raise ValueError("Missing/nonfinite or misaligned reference root velocity")
    validate_reference_velocity(velocity, heading)
    repeated_heading = repeat_heading_sequence(heading, repeats)
    aligned = align_heading_sequence(repeated_heading.unsqueeze(0), initial_heading,
                                    torch.zeros(1, device=heading.device, dtype=torch.long))[0]
    # Preserve local reference velocity through each turning repetition.
    world_velocity = aligned_tracking_velocity(velocity.repeat(repeats, 1),
                                               heading.repeat(repeats, 1), aligned)
    valid = torch.full((len(aligned), 1), tracking_velocity_valid(motion_name, mode, policy),
                       device=heading.device, dtype=torch.bool)
    return TrackingCommands(
        aligned, world_velocity, valid, str(motion_name), mode,
        str(temporal_mode), int(temporal_horizon),
    )


def tracking_context_arrays(z, commands, initial_heading, dt):
    """World fields are diagnostics; heading0-relative fields transfer to real.

    At playback start align heading0 with the robot's initial IMU yaw. Each
    control tick express the resulting world target in the current IMU yaw.
    Do not feed reference_velocity_heading0_xy directly as velocity_command.
    """
    arrays = {
        "z": z, "heading_world_xy": commands.heading_world,
        "heading_relative_xy": world_to_yaw(commands.heading_world, initial_heading),
        "heading_valid": torch.ones_like(commands.velocity_valid),
        "reference_velocity_world_xy": commands.velocity_world,
        "reference_velocity_heading0_xy": world_to_yaw(commands.velocity_world, initial_heading),
        "initial_heading_world_xy": initial_heading,
        "velocity_valid": commands.velocity_valid,
    }
    return {**{k: v.detach().cpu().numpy() for k, v in arrays.items()},
            "dt": float(dt), "context_contract": "tracking_heading0_yaw_velocity_v1",
            "velocity_context_mode": commands.mode,
            "velocity_temporal_mode": commands.temporal_mode,
            "velocity_temporal_horizon": commands.temporal_horizon,
            "motion_name": commands.motion_name}


def reward_velocity_command(task, current_heading, *, mode="auto"):
    """Map the explicitly defined move-ego task to a yaw-frame command.

    These reward tasks are egocentric (not fixed world vectors). Unsupported
    rewards have no invented velocity target. 'valid' on an unsupported task
    raises rather than silently constructing one. Zero speed is a valid stop.
    """
    if mode not in ("auto", "valid", "invalid"):
        raise ValueError(f"Unknown velocity context mode: {mode}")
    number = r"(-?\d+(?:\.\d*)?)"
    match = re.fullmatch(r"move-ego-" + number + "-" + number, task)
    if match is None:
        low = re.fullmatch(r"move-ego-low" + number + "-" + number + "-" + number, task)
        values = None if low is None else (float(low[2]), float(low[3]))
    else:
        values = (float(match[1]), float(match[2]))
    out = current_heading.new_zeros((*current_heading.shape[:-1], 3))
    if mode == "invalid":
        return out
    if values is None:
        if mode == "valid":
            raise ValueError(f"No explicit horizontal velocity contract for reward {task!r}")
        return out
    angle, speed = values
    angle = math.radians(angle)
    out[..., 0] = speed * math.cos(angle)
    out[..., 1] = speed * math.sin(angle)
    out[..., 2] = 1
    return out
