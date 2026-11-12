import unittest
import numpy as np
from utils.teleop_velocity import RetargetVelocity, VelocityContextMessage
from scripts.realtime.realtime_z_server import PoseFrame, RelativeHeadingTracker
from utils.pose_timeline import PoseTimeline
from test_teleop_heading_anchor import TeleopHeadingAnchorTests


class EpochTests(TeleopHeadingAnchorTests):
    def test_epoch_resets_even_if_paused_packet_was_missed(self):
        self.env.ctx_latest_reference_epoch = 1
        self.sample(0, 0, 0)
        self.sample(10, 40, 30)
        self.env.ctx_latest_reference_epoch = 3
        self.sample(100, -170, 0)
        self.sample(100, -160, 10)


class VelocityTests(unittest.TestCase):
    def test_repeated_source_does_not_create_zero_speed(self):
        e = RetargetVelocity()
        for i in range(6):
            v, ok = e.update([i*.02, 0, 0], [1, 0], 1+i*.02, True)
        self.assertTrue(ok)
        a, ok = e.update([.10, 0, 0], [1, 0], 1.10, True)
        np.testing.assert_allclose(a, [1, 0], atol=1e-6)
        self.assertTrue(ok)

    def test_jump_requires_rewarm_without_spike(self):
        e = RetargetVelocity()
        for i in range(8): e.update([i*.01, 0, 0], [1, 0], 1+i*.02, True)
        self.assertFalse(e.update([10, 0, 0], [1, 0], 1.16, True)[1])
        self.assertFalse(e.update([10, 0, 0], [1, 0], 1.18, True)[1])
        v, ok = e.update([10, 0, 0], [1, 0], 1.20, True)
        self.assertTrue(ok)
        np.testing.assert_allclose(v, 0)

    def test_window_reduces_position_noise(self):
        rng = np.random.default_rng(5)
        e = RetargetVelocity()
        t = 1 + np.arange(300)*.02
        x = .4*(t-1) + rng.normal(0, .002, len(t))
        estimated = []
        for a, b in zip(t, x):
            v, ok = e.update([b, 0, 0], [1, 0], a, True)
            if ok: estimated.append(v[0])
        self.assertLess(np.std(np.array(estimated)-.4), .6*np.std(np.diff(x)/.02-.4))

    def test_v2_roundtrip(self):
        msg = VelocityContextMessage(z=np.ones(256), heading_relative_xy=[1, 0],
            stream_id=1, sequence=8, source_receive_monotonic_s=3,
            encoder_completion_monotonic_s=3.01, publish_monotonic_s=3.02,
            heading_valid=True, reference_live=True, velocity_heading_xy=[.5, .1],
            velocity_valid=True, reference_epoch=7, reference_sample_s=2.98)
        out = VelocityContextMessage.from_bytes(msg.to_bytes())
        self.assertEqual(out.reference_epoch, 7)
        self.assertEqual(out.reference_sample_s, 2.98)

    def test_timeline_generation_and_processing_delay(self):
        def pose(t, epoch, x):
            return PoseFrame(np.array([x, 0, .8]), np.array([1., 0, 0, 0]),
                np.zeros(29), source_sample_s=t, reference_epoch=epoch)
        a, b = PoseTimeline(), PoseTimeline()
        for i in range(30):
            p = pose(1+i*.02, 0 if i<10 else 2, i*.01)
            out_a = a.update(p, 10+i*.02)
            out_b = b.update(p, 40+i*.04)
            np.testing.assert_allclose(out_a.root_pos, out_b.root_pos)
            self.assertEqual(out_a.sample_time_s, p.source_sample_s)
        self.assertIs(a.update(p, 100), out_a)

if __name__ == '__main__': unittest.main()
