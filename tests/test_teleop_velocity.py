import unittest
import numpy as np
from utils.teleop_velocity import *
class Tests(unittest.TestCase):
 def test_frames_and_resume(self):
  e=RetargetVelocity()
  self.assertFalse(e.update([0,0,0],[0,1],1,True)[1])
  self.assertFalse(e.update([0,.02,0],[0,1],1.02,True)[1])
  v,valid=e.update([0,.04,0],[0,1],1.04,True)
  np.testing.assert_allclose(v,[1,0],atol=1e-6)
  np.testing.assert_allclose(command_from_heading_error(v,valid,[1,1],True),[[0,1,1]],atol=1e-6)
  e.update([20,30,0],[1,0],1.04,False)
  self.assertFalse(e.update([20,30,0],[1,0],1.06,True)[1])
  self.assertFalse(e.update([20,30,0],[1,0],1.08,True)[1])
  v,valid=e.update([20,30,0],[1,0],1.10,True)
  np.testing.assert_array_equal(v,[0,0]);self.assertTrue(valid)
 def test_gap_jump_and_disabled(self):
  e=RetargetVelocity();e.update([0,0,0],[1,0],1,True)
  self.assertFalse(e.update([1,0,0],[1,0],2,True)[1])
  self.assertFalse(e.update([9,0,0],[1,0],2.02,True)[1])
  np.testing.assert_array_equal(command_from_heading_error([1,2],True,[0,0],False),[[0,0,0]])
  with self.assertRaises(ValueError):command_from_heading_error([1,0],True,[1,10],True)
 def test_packet(self):
  kwargs=dict(z=np.ones(256),heading_relative_xy=[1,0],stream_id=7,sequence=4,source_receive_monotonic_s=1,encoder_completion_monotonic_s=2,publish_monotonic_s=3,heading_valid=True,reference_live=True)
  m=VelocityContextMessage(**kwargs,velocity_heading_xy=[.3,-.2],velocity_valid=True)
  r=VelocityContextMessage.from_bytes(m.to_bytes())
  self.assertTrue(r.velocity_valid);np.testing.assert_allclose(r.velocity_heading_xy,[.3,-.2])
  self.assertFalse(VelocityContextMessage.from_bytes(RealtimeContextMessage(**kwargs).to_bytes()).velocity_valid)
  with self.assertRaises(ValueError):VelocityContextMessage.from_bytes(m.to_bytes()[:-1])
if __name__=='__main__':unittest.main()
