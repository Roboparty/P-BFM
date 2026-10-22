"""Tracking inference and video export for PBFM policies.

Policy rollout is rendered from the training environment, while the reference
motion is rendered from the configured robot MJCF with pure MuJoCo qpos playback.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import joblib
import mediapy as media
import mujoco
import numpy as np
import torch
from torch.utils._pytree import tree_map

from pbfm.inference.actor_override import load_actor_override
from pbfm.agents.behavior_context import (
    heading_observation,
    root_heading_xy,
)
from pbfm.agents.load_utils import load_model_from_checkpoint_dir
from pbfm.agents.inference_context import build_tracking_commands, tracking_context_arrays
from pbfm.inference.runtime import (
    MujocoQposRenderer,
    _assign_training_tile,
    _control_to_qpos_order_indices,
    _expert_qpos_from_obs,
    _export_policy_model as _export_policy_model,
    _export_tracking_onnx,
    _resize_nearest,
    _target_states_from_obs,
    _terrain_env_cfg,
    _tracking_z,
    add_bool_arg,
    add_robot_config_manifest_args,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    policy_qpos_from_env,
    resolve_inference_data_and_robot_args,
    resolve_inference_robot_config as _resolve_tracking_robot_config,
)
from pbfm.utils.helpers import get_backward_observation
from pbfm.inference.runtime import load_inference_robot_training
from pbfm.terrains.presets import TRAINING_TERRAIN_COMPONENT_NAMES


def _center_target_states_on_terrain(
    target_states: dict[str, torch.Tensor], env: Any, *, terrain_relative_height: bool = False,
) -> dict[str, torch.Tensor]:
    """Center the reset; optionally preserve clearance above a raised tile origin."""
    if not bool(getattr(env, "terrain_enabled", False)):
        return target_states
    centered = dict(target_states)
    root_states = target_states["root_states"].clone()
    terrain_origins = env.env_origins[: root_states.shape[0], :3].to(
        device=root_states.device,
        dtype=root_states.dtype,
    )
    root_states[:, :2] = terrain_origins[:, :2]
    if terrain_relative_height:
        root_states[:, 2] += terrain_origins[:, 2]
    centered["root_states"] = root_states
    return centered


def run_tracking_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    robot_config: Path | None,
    headless: bool,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    motion_list: list[int],
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    fps: int,
    max_steps: int | None,
    log_every_steps: int,
    max_episode_length_s: float,
    export_onnx: bool,
    actor_override: Path | None = None,
    velocity_context: str = "auto",
    terrain_family: str | None = None,
    difficulty_row: int = 7,
    reset_height_mode: str | None = None,
    disable_heading_input: bool = False,
) -> None:
    if terrain_family is not None and terrain_family not in TRAINING_TERRAIN_COMPONENT_NAMES:
        raise ValueError(f"Unknown terrain family: {terrain_family}")
    if difficulty_row < 0:
        raise ValueError("difficulty_row must be non-negative")
    if reset_height_mode is None:
        reset_height_mode = "terrain" if terrain_family is not None else "reference"
    if reset_height_mode not in ("reference", "terrain"):
        raise ValueError(f"Unknown reset height mode: {reset_height_mode}")
    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    robot_config = _resolve_tracking_robot_config(robot_config, None, model_folder)
    robot_training = load_inference_robot_training(model_folder, robot_config)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    if not robot_xml.exists():
        raise FileNotFoundError(f"Missing robot XML: {robot_xml}")
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    dof_qpos_order_indices, qpos_joint_names = _control_to_qpos_order_indices(robot_training)

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()
    actor_override_info = load_actor_override(model, actor_override) if actor_override is not None else None
    if actor_override_info is not None:
        print(f"[INFO] Loaded read-only Actor override: {actor_override_info}")

    if export_onnx:
        _export_tracking_onnx(model, model_folder / "exported", robot_training)

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=device,
        headless=headless,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=max_episode_length_s,
    )
    if terrain_family is not None:
        env_cfg = _terrain_env_cfg(env_cfg, terrain_family, seed=4831)
    with ExitStack() as resources:
        wrapped_env = env_cfg.build(num_envs=1)[0]
        resources.callback(wrapped_env.close)
        env = wrapped_env._env
        if terrain_family is not None:
            _assign_training_tile(env, family=terrain_family, difficulty_row=difficulty_row)

        output_dir = model_folder / "tracking_inference"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] PBFM tracking inference model_folder={model_folder}")
        print(f"[INFO] Rollout XML={env_cfg.mjcf_path}")
        print(f"[INFO] Motion data={env_cfg.lafan_tail_path}")
        print(f"[INFO] Expert renderer XML={robot_xml}")
        if qpos_joint_names != control_joint_names:
            print(f"[INFO] Expert qpos joint order differs from control order: {qpos_joint_names}")
        print(f"[INFO] device={device} disable_dr={disable_dr} disable_obs_noise={disable_obs_noise} save_mp4={save_mp4}")

        env._motion_lib.load_all_motions()
        env.is_evaluating = True
        expert_renderer = (
            MujocoQposRenderer(
                robot_xml,
                render_size=render_size,
                camera_distance=camera_distance,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                expected_qpos_size=7 + num_dof,
            )
            if save_mp4
            else None
        )
        if expert_renderer is not None:
            resources.callback(expert_renderer.close)
        if save_mp4:
            policy_render_spec = env.mjlab_env.scene.spec
            if terrain_family == "flat":
                # Rendering only: preserve the real scene's lights/materials while
                # removing tiled-floor z-fighting. Physics and depth input stay unchanged.
                policy_render_spec = policy_render_spec.copy()
                terrain_body = policy_render_spec.body("terrain")
                if terrain_body is None:
                    raise RuntimeError("Flat preview scene is missing terrain body")
                policy_render_spec.delete(terrain_body)
                policy_render_spec.worldbody.add_geom(
                    name="inference_flat_slab",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[0.0, 0.0, -0.02],
                    size=[25.0, 25.0, 0.02],
                    rgba=[0.20, 0.20, 0.20, 1.0],
                    contype=0,
                    conaffinity=0,
                )
            policy_renderer = MujocoQposRenderer(
                None,
                render_size=render_size,
                scene_spec=policy_render_spec,
                source_xml_path=robot_xml,
                add_floor=False,
                camera_distance=camera_distance,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                expected_qpos_size=7 + num_dof,
            )
        else:
            policy_renderer = None
        if policy_renderer is not None:
            resources.callback(policy_renderer.close)
        flat_render_origin_xy = (
            env.env_origins[0, :2].detach().cpu().numpy().copy()
            if terrain_family == "flat"
            else None
        )
        for motion_id in motion_list:
            backward_obs, obs_dict = get_backward_observation(env, motion_id, use_root_height_obs=use_root_height_obs)
            z = _tracking_z(model, tree_map(lambda x: x[1:].to(device) if hasattr(x, "to") else x, backward_obs))
            joblib.dump(z.detach().cpu().numpy(), output_dir / f"zs_{motion_id}.pkl")
            print(f"[INFO] Saved z embedding: {output_dir / f'zs_{motion_id}.pkl'}")

            target_states = _target_states_from_obs(obs_dict, device=device, num_dof=num_dof)
            target_states = _center_target_states_on_terrain(
                target_states, env, terrain_relative_height=reset_height_mode == "terrain",
            )
            print(
                f"[RESET_DIAG] mode={reset_height_mode} "
                f"target_root_xyz={target_states['root_states'][0, :3].detach().cpu().tolist()} "
                f"env_origin_xyz={env.env_origins[0, :3].detach().cpu().tolist()}",
                flush=True,
            )
            observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)
            commands = build_tracking_commands(
                root_heading_xy(obs_dict["ref_body_rots"][:, 0].to(device).float()),
                obs_dict["ref_body_vels"][:, 0, :2].to(device).float(),
                root_heading_xy(env.base_quat.float()),
                motion_name=env._motion_lib._motion_data_keys[motion_id],
                frames=len(z), mode=velocity_context,
                policy=getattr(model.cfg, "reference_velocity_policy", "walk_run"),
                temporal_mode=getattr(model.cfg, "reference_velocity_temporal_mode", "instantaneous"),
                temporal_horizon=int(getattr(model.cfg, "reference_velocity_temporal_horizon", 1)),
            )
            heading_enabled = bool(getattr(model.cfg, "heading_context_enabled", False))
            context = tracking_context_arrays(z, commands, root_heading_xy(env.base_quat.float()), env.dt)
            if disable_heading_input or not heading_enabled:
                context["heading_valid"] = np.zeros_like(context["heading_valid"])
            np.savez_compressed(output_dir / f"context_{motion_id}.npz", **context)
            print(f"[INFO] velocity context: motion={commands.motion_name}, mode={velocity_context}, "
                  f"valid={bool(commands.velocity_valid.any())}, QV={getattr(model.cfg, 'velocity_critic_enabled', False)}")
            heading_targets = None
            if heading_enabled:
                heading_targets = commands.heading_world
            episode_len = int(z.shape[0])
            if max_steps is not None:
                episode_len = min(episode_len, int(max_steps))
            expert_qpos = _expert_qpos_from_obs(
                obs_dict,
                num_dof=num_dof,
                dof_qpos_order_indices=dof_qpos_order_indices,
            )
            frames: list[np.ndarray] = []

            print(f"[INFO] Running policy rollout for motion_id={motion_id}, steps={episode_len}", flush=True)
            for step in range(episode_len):
                if heading_targets is not None:
                    if disable_heading_input:
                        observation["heading"] = torch.zeros_like(observation["heading"])
                    else:
                        target = heading_targets[min(step, heading_targets.shape[0] - 1)].unsqueeze(0)
                        valid = torch.ones((1, 1), device=device, dtype=torch.bool)
                        observation["heading"] = heading_observation(
                            root_heading_xy(env.base_quat.float()), target, valid,
                        )
                if bool(getattr(model.cfg, "velocity_critic_enabled", False)):
                    observation["velocity_command"] = commands.velocity_at(step, root_heading_xy(env.base_quat.float()))
                action = model.act(observation, z[step].unsqueeze(0), mean=True)
                observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                if save_mp4:
                    policy_qpos = policy_qpos_from_env(
                        wrapped_env,
                        expected_qpos_size=policy_renderer.input_nq,
                    )
                    if flat_render_origin_xy is not None:
                        policy_qpos = np.asarray(policy_qpos).copy()
                        policy_qpos[:2] -= flat_render_origin_xy
                    policy_frame = policy_renderer.render_qpos(policy_qpos)
                    expert_frame = expert_renderer.render_qpos(expert_qpos[min(step + 1, len(expert_qpos) - 1)])
                    expert_frame = _resize_nearest(expert_frame, policy_frame.shape[0], policy_frame.shape[1])
                    frames.append(np.concatenate([expert_frame, policy_frame], axis=1))

                if step == 0 or (step + 1) == episode_len or (log_every_steps > 0 and (step + 1) % log_every_steps == 0):
                    print(f"[INFO] motion_id={motion_id} rollout/render progress {step + 1}/{episode_len}", flush=True)

                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    print(f"[INFO] Episode ended at step={step}; stopping rollout for motion_id={motion_id}")
                    break

            if save_mp4:
                video_path = output_dir / f"tracking_{motion_id}.mp4"
                if not frames:
                    raise RuntimeError(f"No frames were rendered for motion_id={motion_id}")
                media.write_video(str(video_path), frames, fps=fps)
                print(f"[INFO] Saved side-by-side video: {video_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PBFM tracking inference with MuJoCo expert rendering.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    add_robot_config_manifest_args(parser, purpose="tracking inference")
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save side-by-side expert/policy MP4.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--motion-list", type=int, nargs="+", default=[20])
    parser.add_argument("--terrain-family", choices=TRAINING_TERRAIN_COMPONENT_NAMES, default=None,
                        help="Select an training tile; omit to keep the checkpoint's environment setup.")
    parser.add_argument("--difficulty-row", type=int, default=7,
                        help="Terrain row used with --terrain-family; step height still comes from terrain config.")
    parser.add_argument("--reset-height-mode", choices=("reference", "terrain"), default=None,
                        help="reference: keep root Z; terrain: add tile-origin Z. Default: terrain with a selected family, otherwise reference.")
    parser.add_argument("--disable-heading-input", action="store_true",
                        help="Diagnostic only: mask heading input and mark exported heading context invalid; does not disable velocity.")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rollout/video frames for quick previews.")
    parser.add_argument(
        "--log-every-steps", type=int, default=100, help="Print rollout/render progress every N steps; 0 disables periodic logs."
    )
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    parser.add_argument(
        "--actor-override",
        type=Path,
        default=None,
        help="Actor-only milestone to load in memory; the source full checkpoint is never modified.",
    )
    add_bool_arg(parser, "--export-onnx", True, "Export ONNX next to the checkpoint before inference.")
    parser.add_argument("--velocity-context", choices=("auto", "valid", "invalid"), default="auto",
                        help="auto: use checkpoint reference-velocity policy; valid: explicit reliable reference velocity; invalid: no QV task.")
    return resolve_inference_data_and_robot_args(parser.parse_args(), parser)


def main() -> None:
    args = parse_args()
    run_tracking_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,
        headless=args.headless,
        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        motion_list=args.motion_list,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        fps=args.fps,
        max_steps=args.max_steps,
        log_every_steps=args.log_every_steps,
        max_episode_length_s=args.max_episode_length_s,
        export_onnx=args.export_onnx,
        actor_override=args.actor_override,
        velocity_context=args.velocity_context,
        terrain_family=args.terrain_family,
        difficulty_row=args.difficulty_row,
        reset_height_mode=args.reset_height_mode,
        disable_heading_input=args.disable_heading_input,
    )


if __name__ == "__main__":
    main()
