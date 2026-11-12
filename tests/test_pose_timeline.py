import unittest
from dataclasses import dataclass
import numpy as np
import pytest
from utils.pose_timeline import PoseTimeline
@dataclass
class Pose:
 root_pos:object
 root_quat_wxyz:object
 dof_pos:object
 retarget_paused:bool=False
 source_sample_s:float=1.
 sample_time_s:object=None
 transitioning:bool=False
 reference_epoch:int=0

def pose(t=1.,q=0.,paused=False):return Pose(np.array([q,0,0.8]),np.array([1.,0,0,0]),np.full(29,q),paused,t)
class Test(unittest.TestCase):
 def test_repeated_source_and_gap(self):
  x=PoseTimeline();a=x.update(pose(),1.)
  b=x.update(pose(),1.2);np.testing.assert_equal(a.dof_pos,b.dof_pos);self.assertEqual(x.source,1.)
  c=x.update(pose(1.3,1.),1.3);np.testing.assert_equal(c.dof_pos,b.dof_pos);self.assertTrue(c.transitioning)
  d=x.update(pose(1.45,1.),1.45) # gap must not restart while already transitioning
 def test_pause_resume_no_first_jump(self):
  x=PoseTimeline();x.update(pose(1.,1.,True),1.)
  a=x.update(pose(1.02,2.),1.02);np.testing.assert_equal(a.dof_pos,1.)
  for i in range(1,17):a=x.update(pose(1.02+i*.02,2.),1.02+i*.02)
  np.testing.assert_allclose(a.dof_pos,2.);self.assertFalse(a.transitioning)
 def test_common_clock(self):
  x=PoseTimeline();a=x.update(pose(1.),1.1)
  self.assertEqual(a.sample_time_s,1.);self.assertEqual(a.source_sample_s,1.)
 def test_quaternion_sign(self):
  x=PoseTimeline();x.update(pose(),1.);p=pose(1.3);p.root_quat_wxyz*=-1
  for i in range(16):
   p.source_sample_s=1.3+.02*i;a=x.update(p,1.3+.02*i)
   self.assertAlmostEqual(np.linalg.norm(a.root_quat_wxyz),1.)
if __name__=='__main__':unittest.main()


@pytest.mark.parametrize('duration', [0., .1, .3, .7])
def test_configured_transition_and_reference_reset(duration):
    timeline = PoseTimeline(recovery_s=duration)
    timeline.update(pose(1., 0., True), 1.)
    out = timeline.update(pose(1.02, 1., False), 1.02)
    assert out.transitioning == (duration > 0.)
    # Update at source rate; processing latency must not advance the blend.
    for i in range(1, 41):
        t = 1.02 + .02 * i
        out = timeline.update(pose(t, 1.), t + .05)
        assert out.sample_time_s == t
        if t - 1.02 >= duration:
            assert not out.transitioning
            np.testing.assert_allclose(out.dof_pos, 1.)
    timeline.restart()
    out = timeline.update(pose(1.84, 2.), 1.9)
    np.testing.assert_allclose(out.dof_pos, 1. if duration else 2.)


@pytest.mark.parametrize('duration', [-1., float('nan'), float('inf')])
def test_invalid_transition_duration(duration):
    with pytest.raises(ValueError):
        PoseTimeline(recovery_s=duration)
