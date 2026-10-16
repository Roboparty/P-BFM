"""Finite-value checks and environment metrics used by training diagnostics."""

import typing as tp

import numpy as np
import torch

from pbfm.envs.mjlab.reset import RESET_REGION_NAMES
from pbfm.envs.mjlab.observations import _terrain_observations
from pbfm.terrains.rp1_simple import RP1_TERRAIN_COMPONENT_NAMES, rp1_center_reset_profile


def _assert_canonical_plane_terrain_priv(observation: tp.Mapping[str, tp.Any], *, label: str) -> None:
    terrain_priv = observation.get("terrain_priv")
    if terrain_priv is None:
        raise RuntimeError(f"{label} is missing terrain_priv")
    if isinstance(terrain_priv, torch.Tensor):
        nonzero = int(torch.count_nonzero(terrain_priv).item())
    else:
        nonzero = int(np.count_nonzero(terrain_priv))
    if nonzero:
        raise RuntimeError(f"{label}.terrain_priv must be analytic zero, found {nonzero} nonzero values")


def _accumulate_metrics(
    total_metrics: dict[str, torch.Tensor] | None,
    metric_update_counts: dict[str, int],
    metrics: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    if total_metrics is None:
        total_metrics = {}
    for key, value in metrics.items():
        value = value.float()
        if key in total_metrics:
            total_metrics[key] = total_metrics[key] + value
        else:
            total_metrics[key] = value.clone()
        metric_update_counts[key] = metric_update_counts.get(key, 0) + 1
    return total_metrics, metric_update_counts


def _tensor_nonfinite_summary(value: tp.Any) -> str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or not torch.is_floating_point(value):
            return None
        finite = torch.isfinite(value)
        if bool(finite.all().item()):
            return None
        bad = ~finite
        bad_count = int(bad.sum().item())
        with torch.no_grad():
            finite_values = value[finite]
            finite_min = float(finite_values.min().item()) if finite_values.numel() > 0 else None
            finite_max = float(finite_values.max().item()) if finite_values.numel() > 0 else None
            bad_indices = bad.nonzero(as_tuple=False)[:5].detach().cpu().tolist()
            bad_values = value[bad][:5].detach().cpu().tolist()
        return (
            f"type=torch shape={tuple(value.shape)} dtype={value.dtype} bad_count={bad_count} "
            f"finite_min={finite_min} finite_max={finite_max} bad_indices={bad_indices} bad_values={bad_values}"
        )
    if isinstance(value, np.ndarray):
        if value.size == 0 or not np.issubdtype(value.dtype, np.floating):
            return None
        finite = np.isfinite(value)
        if bool(finite.all()):
            return None
        bad = ~finite
        bad_count = int(bad.sum())
        finite_values = value[finite]
        finite_min = float(finite_values.min()) if finite_values.size > 0 else None
        finite_max = float(finite_values.max()) if finite_values.size > 0 else None
        bad_indices = np.argwhere(bad)[:5].tolist()
        bad_values = value[bad][:5].tolist()
        return (
            f"type=numpy shape={value.shape} dtype={value.dtype} bad_count={bad_count} "
            f"finite_min={finite_min} finite_max={finite_max} bad_indices={bad_indices} bad_values={bad_values}"
        )
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return None
        return f"type=scalar value={value}"
    return None


def _iter_nonfinite(value: tp.Any, prefix: str = "") -> tp.Iterator[tuple[str, str]]:
    summary = _tensor_nonfinite_summary(value)
    if summary is not None:
        yield prefix or "<root>", summary
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_nonfinite(item, child_prefix)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_nonfinite(item, child_prefix)


def _assert_finite(value: tp.Any, *, label: str, rank: int, local_time: int, global_time: int, optimizer_steps: int) -> None:
    bad = list(_iter_nonfinite(value, label))
    if not bad:
        return
    details = "\n".join(f"  - {path}: {summary}" for path, summary in bad[:20])
    more = "" if len(bad) <= 20 else f"\n  ... {len(bad) - 20} more non-finite fields"
    raise FloatingPointError(
        "Non-finite value detected "
        f"(rank={rank}, local_time={local_time}, global_time={global_time}, optimizer_steps={optimizer_steps}):\n"
        f"{details}{more}"
    )


def _assert_model_finite(module: torch.nn.Module, *, rank: int, local_time: int, global_time: int, optimizer_steps: int) -> None:
    for name, param in module.named_parameters():
        _assert_finite(
            param,
            label=f"model.parameters.{name}",
            rank=rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
        )
    for name, buffer in module.named_buffers():
        _assert_finite(
            buffer,
            label=f"model.buffers.{name}",
            rank=rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
        )


def torso_contact_force_metrics(train_env, *, enabled: bool, threshold: float) -> dict[str, float]:
    if not enabled:
        return {}

    raw_env = getattr(train_env, "_env", train_env)
    simulator = getattr(raw_env, "simulator", None)
    torso_index = getattr(raw_env, "torso_index", None)
    if simulator is None or torso_index is None:
        return {}

    with torch.no_grad():
        torso_force = simulator.contact_forces[:, torso_index, :].float()
        torso_norm = torch.linalg.vector_norm(torso_force, dim=-1)
        torso_mean = torso_force.mean(dim=0)
        torso_force_z = torso_force[:, 2]
        torso_z = simulator._rigid_body_pos[:, torso_index, 2].float()
        contact_mask = torso_norm > threshold
        contact_count = contact_mask.sum()
        num_envs = max(torso_norm.numel(), 1)

        metrics = {
            "torso_contact_force_mean": np.round(torso_norm.mean().item(), 6),
            "torso_contact_force_max": np.round(torso_norm.max().item(), 6),
            "torso_contact_force_p95": np.round(torch.quantile(torso_norm, 0.95).item(), 6),
            "torso_contact_force_contact_count": int(contact_count.item()),
            "torso_contact_force_contact_frac": np.round((contact_count.float() / num_envs).item(), 6),
            "torso_contact_force_mean_x": np.round(torso_mean[0].item(), 6),
            "torso_contact_force_mean_y": np.round(torso_mean[1].item(), 6),
            "torso_contact_force_mean_z": np.round(torso_mean[2].item(), 6),
            "torso_contact_force_z_mean": np.round(torso_force_z.mean().item(), 6),
            "torso_contact_force_z_max": np.round(torso_force_z.max().item(), 6),
            "torso_contact_force_z_p95": np.round(torch.quantile(torso_force_z, 0.95).item(), 6),
            "torso_z_mean": np.round(torso_z.mean().item(), 6),
            "torso_z_min": np.round(torso_z.min().item(), 6),
            "torso_z_p05": np.round(torch.quantile(torso_z, 0.05).item(), 6),
        }
        if contact_count.item() > 0:
            torso_z_contact = torso_z[contact_mask]
            metrics.update(
                {
                    "torso_z_contact_mean": np.round(torso_z_contact.mean().item(), 6),
                    "torso_z_contact_min": np.round(torso_z_contact.min().item(), 6),
                    "torso_z_contact_p05": np.round(torch.quantile(torso_z_contact, 0.05).item(), 6),
                }
            )
        else:
            metrics.update(
                {
                    "torso_z_contact_mean": 0.0,
                    "torso_z_contact_min": 0.0,
                    "torso_z_contact_p05": 0.0,
                }
            )
        return metrics


def terrain_metrics(train_env, *, distributed_sync: bool, world_size: int) -> dict[str, float]:
    raw_env = getattr(train_env, "_env", train_env)
    if not getattr(raw_env, "terrain_enabled", False):
        return {}
    with torch.no_grad():
        root_clearance, _terrain_actor, _terrain_priv = _terrain_observations(raw_env)
        pelvis_world_z = raw_env.body_pos[:, 0, 2]
        root_clearance = root_clearance[:, 0]
        local_ground_z = pelvis_world_z - root_clearance
        # Once tiles are connected, the assigned spawn column no longer
        # describes the physical terrain currently under the robot.
        terrain_ids = raw_env._current_terrain_type_ids()
        metrics: dict[str, float] = {}
        for terrain_id, name in enumerate(raw_env.terrain_component_names):
            mask = terrain_ids == terrain_id
            if not torch.any(mask):
                continue
            for metric_name, values in (
                ("pelvis_world_z", pelvis_world_z),
                ("local_ground_z", local_ground_z),
                ("pelvis_clearance", root_clearance),
            ):
                selected = values[mask]
                prefix = f"terrain/{name}/{metric_name}"
                metrics[f"{prefix}_mean"] = np.round(selected.mean().item(), 6)
                metrics[f"{prefix}_min"] = np.round(selected.min().item(), 6)
                metrics[f"{prefix}_p05"] = np.round(torch.quantile(selected, 0.05).item(), 6)

        boundary_margin = raw_env._terrain_boundary_margin()
        if boundary_margin is not None:
            boundary_min = boundary_margin.min()
            boundary_historical_min = raw_env._terrain_boundary_min.clone()
            boundary_violation_count = raw_env._terrain_boundary_violation_count.clone()
            if distributed_sync and world_size > 1:
                torch.distributed.all_reduce(boundary_min, op=torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(boundary_historical_min, op=torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(boundary_violation_count, op=torch.distributed.ReduceOp.SUM)
            metrics["terrain/boundary_margin_min"] = np.round(boundary_min.item(), 6)
            metrics["terrain/boundary_margin_p05"] = np.round(torch.quantile(boundary_margin, 0.05).item(), 6)
            metrics["terrain/boundary_margin_historical_min"] = np.round(
                boundary_historical_min.item(), 6
            )
            metrics["terrain/boundary_required_margin"] = np.round(
                raw_env._terrain_boundary_required, 6
            )
            metrics["terrain/boundary_violation_count"] = int(boundary_violation_count.item())

        tile_crossings = raw_env._terrain_tile_crossing_count.clone()
        transition_counts = raw_env._terrain_transition_counts.clone()
        reset_region_counts = raw_env._reset_region_counts.clone()
        reset_family_counts = raw_env._reset_terrain_family_counts.clone()
        lie_down_resets = raw_env._lie_down_reset_count.clone()
        if distributed_sync and world_size > 1:
            torch.distributed.all_reduce(tile_crossings, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(transition_counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reset_region_counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reset_family_counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(lie_down_resets, op=torch.distributed.ReduceOp.SUM)
        metrics["terrain/tile_crossing_count"] = int(tile_crossings.item())
        for source_id, source_name in enumerate(raw_env.terrain_component_names):
            for target_id, target_name in enumerate(raw_env.terrain_component_names):
                count = int(transition_counts[source_id, target_id].item())
                if count:
                    metrics[f"terrain/transition/{source_name}_to_{target_name}"] = count
        reset_family_total = int(reset_family_counts.sum().item())
        metrics["reset/family_total"] = reset_family_total
        is_rp1_simple = tuple(raw_env.terrain_component_names) == tuple(
            RP1_TERRAIN_COMPONENT_NAMES
        )
        for family_id, family_name in enumerate(raw_env.terrain_component_names):
            count = int(reset_family_counts[family_id].item())
            metrics[f"reset/family/{family_name}"] = count
            metrics[f"reset/family_fraction/{family_name}"] = (
                float(count / reset_family_total) if reset_family_total else 0.0
            )
            if is_rp1_simple:
                reset_profile, _vertical_direction = rp1_center_reset_profile(family_name)
                metrics[f"reset/profile/{reset_profile}"] = count
        if raw_env._reset_region_labels_are_semantic:
            for region_id, region_name in enumerate(RESET_REGION_NAMES):
                metrics[f"reset/region/{region_name}"] = int(reset_region_counts[region_id].item())
        metrics["reset/lie_down"] = int(lie_down_resets.item())
        return metrics
