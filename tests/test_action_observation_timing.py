import unittest
from types import SimpleNamespace

import numpy as np

from rl_policy.observations.ufo_policy import (
    prev_actions, prev_actions_history, dof_pos_minus_default_history,
)


class ActionObservationTimingTest(unittest.TestCase):
    def test_current_action_and_history_match_training_tick_order(self):
        state = SimpleNamespace(num_dof=29, joint_pos=np.zeros(29))
        env = SimpleNamespace(num_actions=29, state_processor=state,
            policy_config={"history_update_after_compute": True},
            last_action=np.zeros((1, 29)))
        current = prev_actions(env=env)
        history = prev_actions_history(env=env, steps=4)
        positions = dof_pos_minus_default_history(env=env, steps=4, default_pos=[0]*29)
        expected_actions = np.zeros((4, 29))
        expected_positions = np.zeros((4, 29))
        for tick in range(10):
            # Includes the reset-to-zero transition of a stopped/rearmed policy.
            previous_action = np.full(29, 0 if tick == 6 else tick * .17)
            env.last_action = previous_action[None].copy()
            state.joint_pos = np.full(29, tick * .01)
            np.testing.assert_array_equal(current.compute(), previous_action)
            np.testing.assert_array_equal(history.compute().reshape(4, 29), expected_actions)
            np.testing.assert_array_equal(positions.compute().reshape(4, 29), expected_positions)
            for item in (current, history, positions):
                item.update({"action": env.last_action})
            expected_actions = np.roll(expected_actions, 1, axis=0)
            expected_actions[0] = previous_action
            expected_positions = np.roll(expected_positions, 1, axis=0)
            expected_positions[0] = state.joint_pos

    def test_legacy_update_before_compute_retains_cache_semantics(self):
        env = SimpleNamespace(num_actions=29, state_processor=SimpleNamespace(),
            policy_config={"history_update_after_compute": False},
            last_action=np.full(29, 99.))
        current = prev_actions(env=env)
        current.update({"action": np.arange(29)})
        np.testing.assert_array_equal(current.compute(scale=.5), np.arange(29)*.5)


if __name__ == '__main__':
    unittest.main()
