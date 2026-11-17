"""RP1 source DR semantics that differ from MJLab's default events."""

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def scale_link_density(env, env_ids, scale_range):
    # One uniform density multiplier per link; mass and inertia must stay consistent.
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = env.scene["robot"].indexing.body_ids
    scale = torch.empty((len(env_ids), len(ids)), device=env.device).uniform_(*scale_range)
    mass = torch.as_tensor(env.sim.mj_model.body_mass, device=env.device, dtype=scale.dtype)[ids]
    inertia = torch.as_tensor(env.sim.mj_model.body_inertia, device=env.device, dtype=scale.dtype)[ids]
    env.sim.model.body_mass[env_ids[:, None], ids] = mass * scale
    env.sim.model.body_inertia[env_ids[:, None], ids] = inertia * scale[..., None]


def set_push_velocity(env, env_ids, velocity_range):
    # RP1 overwrites velocity, including z=0; MJLab's standard event adds a kick.
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    velocity = torch.zeros((len(env_ids), 6), device=env.device)
    for i, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        if key in velocity_range:
            velocity[:, i].uniform_(*velocity_range[key])
    env.scene["robot"].write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
