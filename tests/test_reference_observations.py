from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from pbfm.envs.mjlab.observations import _extend_body_state
from pbfm.training.evaluation.tracking import get_backward_observation as get_mjlab_eval_backward_observation
from pbfm.motion.reference_observations import reference_base_ang_vel
from pbfm.utils.torch_utils import quat_rotate_inverse


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReferenceObservationsTest(unittest.TestCase):
    def test_virtual_head_velocity_matches_training_contract(self) -> None:
        s = 2.0**-0.5
        env = SimpleNamespace(
            num_envs=5, num_extend_bodies=1, extend_body_parent_ids=[0],
            body_pos=torch.zeros(5, 1, 3),
            body_rot=torch.tensor([
                [0., 0., 0., 1.], [0., s, 0., s], [s, 0., 0., s],
                [0., 0., s, s], [0., s, 0., s],
            ]).unsqueeze(1),
            body_vel=torch.tensor([[0., 0., 0.]] * 4 + [[1., 2., 3.]]).unsqueeze(1),
            body_ang_vel=torch.tensor([[0., 1., 0.]] * 4 + [[0., 0., 0.]]).unsqueeze(1),
            extend_body_pos_in_parent=torch.tensor([0., 0., .35]).repeat(5, 1, 1),
            extend_body_rot_in_parent_xyzw=torch.tensor([0., 0., 0., 1.]).repeat(5, 1, 1),
        )
        _, _, vel, _ = _extend_body_state(env)
        expected = torch.tensor([
            [.35, 0., 0.], [.35, 0., 0.], [.35, 0., 0.],
            [.35, 0., 0.], [1., 2., 3.],
        ])
        torch.testing.assert_close(vel[:, -1], expected, atol=1e-6, rtol=0)

    def test_no_virtual_bodies_keeps_state_unchanged(self) -> None:
        env = SimpleNamespace(num_extend_bodies=0,
            body_pos=torch.zeros(2, 3, 3), body_rot=torch.zeros(2, 3, 4),
            body_vel=torch.zeros(2, 3, 3), body_ang_vel=torch.zeros(2, 3, 3))
        for actual, expected in zip(_extend_body_state(env),
                                   (env.body_pos, env.body_rot, env.body_vel, env.body_ang_vel)):
            self.assertIs(actual, expected)

    def test_reference_base_ang_vel_matches_simulator_contract(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(obs_scales={"base_ang_vel": 0.25})))
        sqrt_half = 2.0**-0.5
        base_quat = torch.tensor(
            [
                [0.0, 0.0, sqrt_half, sqrt_half],
                [sqrt_half, 0.0, 0.0, sqrt_half],
            ],
            dtype=torch.float32,
        )
        world_ang_vel = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [-2.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        )

        expected = quat_rotate_inverse(base_quat, world_ang_vel, w_last=True) * 0.25
        actual = reference_base_ang_vel(env, base_quat, world_ang_vel)

        self.assertTrue(torch.allclose(actual, expected))

    def test_reference_base_ang_vel_defaults_to_unit_scale(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(obs_scales={})))
        base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        world_ang_vel = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)

        actual = reference_base_ang_vel(env, base_quat, world_ang_vel)

        self.assertTrue(torch.equal(actual, world_ang_vel))

    def test_mjlab_eval_backward_observation_requires_obs_filter(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(use_obs_filter=False)))

        with self.assertRaisesRegex(ValueError, "requires obs.use_obs_filter=True"):
            get_mjlab_eval_backward_observation(env, motion_id=0, include_last_action=False)

    def test_reference_paths_do_not_directly_assign_world_base_ang_vel(self) -> None:
        paths = [
            "pbfm/motion/expert.py",
            "pbfm/utils/helpers.py",
            "pbfm/training/evaluation/tracking.py",
        ]
        for relative_path in paths:
            with self.subTest(path=relative_path):
                tree = ast.parse((REPO_ROOT / relative_path).read_text())
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    assigns_ref_ang_vel = any(isinstance(target, ast.Name) and target.id == "ref_ang_vel" for target in node.targets)
                    if assigns_ref_ang_vel:
                        self.assertNotEqual(ast.unparse(node.value), "ref_body_angular_vels[:, 0]", relative_path)

    def test_reference_backward_helpers_use_zero_fake_actions(self) -> None:
        paths = [
            "pbfm/utils/helpers.py",
            "pbfm/training/evaluation/tracking.py",
        ]
        for relative_path in paths:
            with self.subTest(path=relative_path):
                tree = ast.parse((REPO_ROOT / relative_path).read_text())
                zero_action_assignments = 0
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    assigns_bogus_actions = any(isinstance(target, ast.Name) and target.id == "bogus_actions" for target in node.targets)
                    if not assigns_bogus_actions:
                        continue
                    value = ast.unparse(node.value)
                    self.assertNotEqual(value, "ref_dof_pos", relative_path)
                    zero_action_assignments += value == "torch.zeros_like(ref_dof_pos)"
                self.assertGreaterEqual(
                    zero_action_assignments,
                    1,
                    relative_path,
                )
