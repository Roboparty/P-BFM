"""No-actuation regression tests for realtime command heading anchors."""
import unittest
from types import SimpleNamespace

import numpy as np

from rl_policy.observations.ufo_policy import heading


class TeleopHeadingAnchorTests(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(
            state_processor=SimpleNamespace(root_quat_b=None),
            task_type="tracking", ctx_source="zmq", start_motion=True,
            ctx_started_monotonic=1.0,
            tracking_heading_context_available=True,
            tracking_heading_context_valid=True,
        )
        self.obs = heading(env=self.env, mode="behavior_context", dimension=2)

    def sample(self, robot, human, error):
        r, h, e = np.deg2rad([robot, human, error])
        self.env.state_processor.root_quat_b = np.array(
            [np.cos(r / 2), 0, 0, np.sin(r / 2)], dtype=np.float32)
        self.env.tracking_heading_relative_xy = np.array(
            [np.cos(h), np.sin(h)], dtype=np.float32)
        np.testing.assert_allclose(self.obs.compute(),
                                   [1 - np.cos(e), np.sin(e)], atol=1e-6)

    def test_start_ignores_preexisting_human_turn(self):
        self.sample(70, 30, 0)
        self.sample(70, 50, 20)
        self.sample(90, 50, 0)

    def test_stop_and_restart_does_not_double_turn(self):
        self.sample(0, 0, 0)
        self.sample(30, 30, 0)
        self.env.start_motion = False
        self.sample(30, 30, 0)
        self.env.start_motion = True
        self.env.ctx_started_monotonic = 2.0
        self.sample(30, 30, 0)
        self.sample(30, 40, 10)

    def test_repeated_start_without_stopped_tick_reanchors(self):
        self.sample(0, 0, 0)
        self.sample(20, 30, 10)
        self.env.ctx_started_monotonic = 2.0
        self.sample(20, 30, 0)

    def test_x_held_reference_preserves_correction_and_resume_delta(self):
        self.sample(0, 0, 0)
        self.sample(20, 30, 10)
        self.sample(20, 30, 10)  # X pause/resume holds upstream reference yaw.
        self.sample(20, 40, 20)

    def test_first_valid_reference_anchors_both_frames_together(self):
        self.env.tracking_heading_context_valid = False
        self.sample(0, 30, 0)
        self.env.tracking_heading_context_valid = True
        self.sample(20, 30, 0)
        self.sample(20, 40, 10)

    def test_wraparound(self):
        self.sample(0, 179, 0)
        self.sample(0, -179, 2)

    def test_file_context_retains_authored_heading(self):
        self.env.ctx_source = "npz"
        self.sample(0, 30, 30)


if __name__ == "__main__":
    unittest.main()
