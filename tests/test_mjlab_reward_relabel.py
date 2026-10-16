from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from pbfm.envs.mjlab.rewards import LocomotionReward
from pbfm.inference.relabel import (
    TERRAIN_REFERENCE_RAY_INDEX,
    RewardWrapperHV,
    canonicalize_terrain_relabel_qpos,
    make_reward_from_name,
)


def _terrain_observation(clearance: np.ndarray) -> dict[str, torch.Tensor]:
    batch_size = clearance.shape[0]
    terrain_actor = np.zeros((batch_size, 273), dtype=np.float32)
    terrain_actor[:, TERRAIN_REFERENCE_RAY_INDEX] = clearance
    privileged_state = np.zeros((batch_size, 8), dtype=np.float32)
    privileged_state[:, 0] = clearance
    return {
        "privileged_state": torch.from_numpy(privileged_state),
        "terrain_actor": torch.from_numpy(terrain_actor),
        "terrain_priv": torch.zeros(batch_size, 273),
    }


class TerrainRewardRelabelCanonicalizationTest(unittest.TestCase):
    def test_strict_locomotion_reward_parses_and_prefers_straight_motion(self) -> None:
        reward = make_reward_from_name("move-ego-strict-0-0.7-0")
        self.assertIsInstance(reward, LocomotionReward)
        self.assertTrue(reward.strict_velocity)
        self.assertEqual(reward.move_speed, 0.7)
        self.assertEqual(reward.target_yaw_velocity, 0.0)

        def score(velocity, yaw_velocity):
            def sensor(_model, _data, name):
                if name == "upvector_torso":
                    return np.array([0.073, 0.0, 1.0])
                if name == "imu-angular-velocity":
                    return np.array([0.0, 0.0, yaw_velocity])
                raise AssertionError(name)

            with (
                patch("pbfm.envs.mjlab.rewards.get_xpos", return_value=np.array([0.0, 0.0, 0.82])),
                patch("pbfm.envs.mjlab.rewards.get_xmat", return_value=np.eye(3)),
                patch(
                    "pbfm.envs.mjlab.rewards.get_center_of_mass_linvel",
                    return_value=np.array([velocity[0], velocity[1], 0.0]),
                ),
                patch("pbfm.envs.mjlab.rewards.get_sensor_data", side_effect=sensor),
            ):
                return reward.compute(object(), object())

        target = score((0.7, 0.0), 0.0)
        self.assertGreater(target, score((0.7, 0.4), 0.0))
        self.assertGreater(target, score((0.7, 0.0), 0.5))
        self.assertGreater(target, score((-0.7, 0.0), 0.0))

    def test_terrain_qpos_uses_ground_relative_root_height_without_mutation(self) -> None:
        qpos = np.zeros((3, 36), dtype=np.float32)
        qpos[:, 2] = np.array([0.82, -0.75, 2.8], dtype=np.float32)
        original = qpos.copy()
        clearance = np.array([0.82, 0.65, 0.8], dtype=np.float32)

        canonical = canonicalize_terrain_relabel_qpos(qpos, _terrain_observation(clearance))

        np.testing.assert_array_equal(qpos, original)
        np.testing.assert_allclose(canonical[:, 2], clearance, atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(canonical[:, :2], original[:, :2])
        np.testing.assert_array_equal(canonical[:, 3:], original[:, 3:])

    def test_vertical_translation_produces_identical_canonical_qpos(self) -> None:
        lower = np.zeros((1, 36), dtype=np.float32)
        elevated = lower.copy()
        lower[:, 2] = 0.8
        elevated[:, 2] = 2.8
        observation = _terrain_observation(np.array([0.8], dtype=np.float32))

        lower_canonical = canonicalize_terrain_relabel_qpos(lower, observation)
        elevated_canonical = canonicalize_terrain_relabel_qpos(elevated, observation)

        np.testing.assert_array_equal(lower_canonical, elevated_canonical)

    def test_original_flat_fb_qpos_is_unchanged(self) -> None:
        qpos = np.random.default_rng(7).normal(size=(4, 36)).astype(np.float32)
        observation = {"privileged_state": torch.zeros(4, 8)}
        output = canonicalize_terrain_relabel_qpos(qpos, observation)
        self.assertIs(output, qpos)

    def test_inconsistent_center_clearance_fails(self) -> None:
        qpos = np.zeros((1, 36), dtype=np.float32)
        observation = _terrain_observation(np.array([0.8], dtype=np.float32))
        observation["terrain_actor"][0, TERRAIN_REFERENCE_RAY_INDEX] = 0.7
        with self.assertRaisesRegex(RuntimeError, "inconsistent root clearances"):
            canonicalize_terrain_relabel_qpos(qpos, observation)

    def test_nonfinite_clearance_fails(self) -> None:
        qpos = np.zeros((1, 36), dtype=np.float32)
        observation = _terrain_observation(np.array([np.nan], dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            canonicalize_terrain_relabel_qpos(qpos, observation)

    def test_reward_wrapper_relabels_canonical_next_qpos(self) -> None:
        clearance = np.array([0.65, 0.8], dtype=np.float32)
        next_observation = _terrain_observation(clearance)
        qpos = torch.zeros(2, 36)
        qpos[:, 2] = torch.tensor([-0.75, 2.8])
        sample = {
            "action": torch.zeros(2, 29),
            "next": {
                "qpos": qpos,
                "qvel": torch.zeros(2, 35),
                "observation": next_observation,
            },
        }

        class Dataset:
            def size(self):
                return 2

            def sample(self, _size):
                return sample

        model = SimpleNamespace(
            device="cpu",
            reward_wr_inference=lambda **kwargs: torch.ones(1, 256),
        )
        wrapper = RewardWrapperHV(
            model=model,
            inference_dataset=Dataset(),
            num_samples_per_inference=1,
            inference_function="reward_wr_inference",
            max_workers=1,
            env_model=object(),
        )

        with patch("pbfm.inference.relabel.relabel", return_value=np.ones((2, 1))) as relabel_mock:
            wrapper.reward_inference("move-ego-0-0.7")

        relabeled_qpos = relabel_mock.call_args.args[1]
        np.testing.assert_allclose(relabeled_qpos[:, 2], clearance, atol=0.0, rtol=0.0)


    def test_reward_wrapper_preserves_sampling_cached_embeddings_and_device(self) -> None:
        for full_buffer in (False, True):
            for cached_embedding in (False, True):
                for wrapped_model in (False, True):
                    with self.subTest(full=full_buffer, cached=cached_embedding, wrapped=wrapped_model):
                        next_obs = {"state": torch.zeros(2, 64)}
                        data = {"action": torch.zeros(2, 29), "next": {
                            "qpos": torch.zeros(2, 36), "qvel": torch.zeros(2, 35),
                            "observation": next_obs,
                        }}
                        if cached_embedding:
                            data["B"] = torch.ones(2, 4)
                        dataset = SimpleNamespace(size=lambda: 2, sample=Mock(return_value=data),
                                                  get_full_buffer=Mock(return_value=data))
                        infer = Mock(return_value=torch.arange(4, dtype=torch.float32))
                        model = SimpleNamespace(device="cpu", reward_wr_inference=infer)
                        if wrapped_model:
                            model = SimpleNamespace(unwrapped_model=model, reward_wr_inference=infer)
                        wrapper = RewardWrapperHV(
                            model=model, inference_dataset=dataset,
                            num_samples_per_inference=2 if full_buffer else 1,
                            inference_function="reward_wr_inference", max_workers=1, env_model=object(),
                        )
                        with patch("pbfm.inference.relabel.relabel", return_value=np.ones((2, 1))):
                            result = wrapper.reward_inference("move-ego-0-0.7")
                        torch.testing.assert_close(result, torch.arange(4, dtype=torch.float32).reshape(1, 4))
                        self.assertEqual(dataset.sample.call_count, int(not full_buffer))
                        self.assertEqual(dataset.get_full_buffer.call_count, int(full_buffer))
                        values = infer.call_args.kwargs
                        expected_key = "B_vect" if cached_embedding else "next_obs"
                        self.assertEqual(set(values), {"reward", expected_key})
                        self.assertIs(values[expected_key], data["B"] if cached_embedding else next_obs)
                        self.assertEqual(values["reward"].device.type, "cpu")
                        torch.testing.assert_close(values["reward"], torch.ones(2, 1))


if __name__ == "__main__":
    unittest.main()
