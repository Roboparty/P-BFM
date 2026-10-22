from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from pbfm.envs.mjlab.core import HumanoidVerseMjlabCore
from pbfm.train import _resolve_training_robot_config, build_ufo_mjlab_config, parse_args as parse_train_args
from pbfm.inference.tracking import (
    _expert_qpos_from_obs,
    _resolve_tracking_robot_config,
    _target_states_from_obs,
    parse_args as parse_tracking_args,
)
from pbfm.robots import load_robot_training_spec, resolve_robot_config_path


def _write_non_g1_config(root: Path) -> Path:
    path = root / "unsupported.yaml"
    path.write_text("name: tiny_train\n")
    return path

class RobotConfigTrainingTest(unittest.TestCase):
    def test_fb_depth_default_uses_final_main_terrain_prior(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit_main_lsgan",
            num_envs=16,
            num_env_steps=2048,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            agent="fb_depth",
        )
        self.assertEqual(cfg.prior_plane_envs, 0)
        self.assertTrue(cfg.agent.train.behavior_prior_enabled)
        self.assertEqual(cfg.agent.train.discriminator_loss, "bce")
        self.assertEqual(cfg.agent.train.discriminator_reward, "log_odds")
        self.assertEqual(cfg.agent.train.reg_coeff, 0.01)

    def test_fb_depth_no_d_disables_prior_collector_and_actor_weight(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit_no_d",
            num_envs=16,
            num_env_steps=2048,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            agent="fb_depth",
            behavior_prior=False,
            prior_plane_envs=0,
        )
        self.assertEqual(cfg.prior_plane_envs, 0)
        self.assertFalse(cfg.agent.train.behavior_prior_enabled)
        self.assertEqual(cfg.agent.train.reg_coeff, 0.0)

    def test_old_g1_default_builds_cfg(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
        )
        self.assertTrue(str(cfg.env.robot_config_path).endswith("configs/robots/g1_29dof.yaml"))
        self.assertTrue(str(cfg.env.mjcf_path).endswith("pbfm/data/robots/g1_mjlab/g1_29dof.xml"))

    def test_explicit_g1_robot_config_builds_cfg(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            robot_config="configs/robots/g1_29dof.yaml",
        )
        self.assertTrue(str(cfg.env.robot_config_path).endswith("configs/robots/g1_29dof.yaml"))

    def test_manifest_robot_config_is_used_when_cli_missing(self) -> None:
        argv = [
            "train.py",
            "--agent",
            "fb_depth",
            "--data-manifest",
            "tests/fixtures/example_robot_state_auto_build.yaml",
            "--gpu-ids",
            "single",
            "--smoke",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_train_args()
        self.assertTrue(str(args.robot_config).endswith("configs/robots/g1_29dof.yaml"))

    def test_cli_rejects_non_g1_robot_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_non_g1_config(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "Supported robots"):
                _resolve_training_robot_config(tiny_robot, "configs/robots/g1_29dof.yaml")

    def test_tracking_manifest_robot_config_is_used_when_cli_missing(self) -> None:
        argv = [
            "tracking_inference.py",
            "--model-folder",
            "/tmp/ufo_unit_model",
            "--data-manifest",
            "tests/fixtures/example_robot_state_auto_build.yaml",
            "--dataset",
            "g1_robot_state_sample",
            "--export-onnx",
            "false",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_tracking_args()
        self.assertTrue(str(args.robot_config).endswith("configs/robots/g1_29dof.yaml"))

    def test_tracking_cli_rejects_non_g1_robot_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_non_g1_config(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "Supported robots"):
                _resolve_tracking_robot_config(tiny_robot, "configs/robots/g1_29dof.yaml")

    def test_aux_rewards_require_two_contact_bodies_unconditionally(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.reward_scales = {}
        cfg = OmegaConf.create(
            {
                "robot": {
                    "contact_bodies": ["left_foot"],
                    "left_ankle_dof_names": ["left_ankle_pitch_joint", "left_ankle_roll_joint"],
                    "right_ankle_dof_names": ["right_ankle_pitch_joint", "right_ankle_roll_joint"],
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "robot.contact_bodies.*biped foot auxiliary terms unconditionally"):
            core._validate_aux_reward_semantics(cfg)

    def test_aux_ankle_reward_requires_both_ankle_fields(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.reward_scales = {"penalty_ankle_roll": -1.0}
        cfg = OmegaConf.create(
            {
                "robot": {
                    "contact_bodies": ["left_foot", "right_foot"],
                    "left_ankle_dof_names": ["left_ankle_pitch_joint"],
                    "right_ankle_dof_names": [],
                }
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "robot.left_ankle_dof_names, robot.right_ankle_dof_names.*penalty_ankle_roll",
        ):
            core._validate_aux_reward_semantics(cfg)


    def test_mjlab_action_input_reorders_policy_actions_to_action_term_order(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.actions = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        core.default_dof_pos_offset = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        core.action_target_scale = torch.tensor([[1.0, 2.0, 1.0, 4.0]])
        core._action_term_dof_indices = torch.tensor([2, 0, 3, 1])

        action_input = core._mjlab_action_input()

        torch.testing.assert_close(action_input, torch.tensor([[33.0, 11.0, 41.0, 21.0]]))

    def test_public_training_loader_rejects_non_g1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            robot_config = _write_non_g1_config(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "Supported robots"):
                load_robot_training_spec(robot_config)
            with self.assertRaisesRegex(ValueError, "Supported robots"):
                build_ufo_mjlab_config(device="cpu", robot_config=robot_config)

    def test_g1_yaml_actuator_missing_joint_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OmegaConf.load(resolve_robot_config_path("configs/robots/g1_29dof.yaml"))
            joints = list(config.training.init_state.default_joint_angles)
            config.training.actuator = {
                "source": "yaml",
                "joints": {joint: dict(effort_limit=1.0, velocity_limit=1.0, armature=0.01, friction=0.01)
                           for joint in joints[:-1]},
            }
            path = Path(tmpdir) / "g1.yaml"
            OmegaConf.save(config, path)
            with self.assertRaisesRegex(ValueError, f"missing parameters for joint '{joints[-1]}'"):
                load_robot_training_spec(path)

    def test_g1_training_rejects_reordered_control_joints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OmegaConf.load(resolve_robot_config_path("configs/robots/g1_29dof.yaml"))
            config.control_joints = list(reversed(list(config.training.init_state.default_joint_angles)))
            path = Path(tmpdir) / "g1.yaml"
            OmegaConf.save(config, path)
            with self.assertRaisesRegex(ValueError, "g1_29dof control joint order must match"):
                load_robot_training_spec(path)

    def test_tracking_shapes_follow_num_dof(self) -> None:
        obs = {
            "ref_body_pos": torch.zeros(4, 1, 3),
            "ref_body_rots": torch.zeros(4, 1, 4),
            "ref_body_vels": torch.zeros(4, 1, 3),
            "ref_body_angular_vels": torch.zeros(4, 1, 3),
            "dof_pos": torch.zeros(4, 2),
            "ref_dof_vel": torch.ones(4, 2),
        }
        obs["ref_body_rots"][..., 3] = 1.0
        qpos = _expert_qpos_from_obs(obs, num_dof=2, dof_qpos_order_indices=torch.tensor([0, 1]).numpy())
        self.assertEqual(qpos.shape, (4, 9))
        target = _target_states_from_obs(obs, device="cpu", num_dof=2)
        self.assertEqual(tuple(target["dof_states"].shape), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
