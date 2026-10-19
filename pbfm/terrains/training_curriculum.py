"""Reset-only, training-clock terrain sampling; never changes terrain geometry."""
import torch


def maximum_level(global_step: int, num_rows: int, interval: int) -> int:
    if global_step < 0 or num_rows < 1 or interval < 1:
        raise ValueError('Invalid curriculum clock or terrain shape')
    return min(num_rows - 1, 2 + 2 * (global_step // interval))


def sample_levels(count, *, device, maximum, easy_fraction):
    if not 0 <= easy_fraction <= 1 or maximum < 0:
        raise ValueError('Invalid curriculum sampling settings')
    # Mixture, not a claim that exactly 25% of a small batch is easy.
    levels = torch.randint(maximum + 1, (count,), device=device)
    easy = torch.rand(count, device=device) < easy_fraction
    low = torch.randint(min(2, maximum) + 1, (count,), device=device)
    return torch.where(easy, low, levels)


def apply_reset_curriculum(core, env_ids, target_states=None):
    cfg = core._creation_config
    step = getattr(core, '_training_terrain_global_step', None)
    if (not cfg.terrain_training_curriculum or step is None or target_states is not None
            or not core.terrain_enabled or core.terrain_mode != 'training_mix'):
        return
    terrain = core.mjlab_env.scene['terrain']
    elapsed = max(0, step - cfg.terrain_curriculum_start_global_step)
    maximum = maximum_level(elapsed, terrain.terrain_origins.shape[0], cfg.terrain_curriculum_interval)
    levels = sample_levels(len(env_ids), device=core.device, maximum=maximum,
                           easy_fraction=cfg.terrain_curriculum_easy_fraction)
    terrain.terrain_levels[env_ids] = levels
    origins = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]
    terrain.env_origins[env_ids] = origins
    core.env_origins[env_ids] = origins
