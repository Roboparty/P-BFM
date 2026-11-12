import unittest,tempfile
from unittest.mock import patch
from types import SimpleNamespace as NS
from collections import deque
import numpy as np
from utils import stand_fallback as mod
class Test(unittest.TestCase):
 def setUp(self):
  self.f=tempfile.NamedTemporaryFile(suffix='.npy');np.save(self.f.name,np.ones(256,dtype=np.float32))
  self.o=NS(policy_config={'teleop_auto_resume':True},state_processor=NS(get_latest_direct_depth_latent=lambda **k:None),ctx_latest=np.ones(256),use_policy_action=True,stop_latched=False,start_motion=True,ctx_window=deque(),ctx_norm_ref=16.,pico_only_control=True,pico_control_have_valid_source=True,pico_control_source_timeout_s=.25,ctx_zmq_timeout_s=.2,ctx_zmq_result_timeout_s=.1,ctx_zmq_source_timeout_s=.2,ctx_latest_reference_epoch=2,_warn_throttled=lambda *a,**k:None)
  self.b=mod.StandFallback(self.o,self.f.name); self.clock=patch.object(mod.time,'monotonic',return_value=10.);self.t=self.clock.start();self.addCleanup(self.clock.stop)
  self.b.enter('pose stale');self.b.depth_ok=True
 def tick(self,t,live=True,heading_valid=True):
  self.t.return_value=t
  o=self.o
  for k in ['ctx_last_z_monotonic','ctx_latest_result_monotonic','ctx_latest_source_monotonic','pico_control_last_source_monotonic','pico_control_last_receive_monotonic']:setattr(o,k,t)
  o.ctx_input_message=NS(z=np.arange(256,dtype=np.float32)/100,heading_relative_xy=np.array([1.,0.]),heading_valid=heading_valid,velocity_heading_xy=np.array([1.,0.]),velocity_valid=True,reference_live=live)
  self.b.apply()
 def test_auto_blend(self):
  with patch.object(mod.Path,'write_text'),patch.object(mod.Path,'replace'),patch.object(mod.Path,'read_text',side_effect=lambda: __import__('json').dumps({'token':self.b.request_token,'epoch':2})):
   self.tick(10.1);self.assertFalse(self.o.start_motion)
   self.tick(10.29);self.assertFalse(self.o.start_motion)
   self.tick(10.31);self.assertTrue(self.o.start_motion);self.assertTrue(self.b.active)
   self.tick(11.32);self.assertFalse(self.b.active);self.assertTrue(self.o.start_motion)
 def test_long_gap(self):
  self.tick(13.01);self.assertFalse(self.b.auto_allowed);self.assertFalse(self.o.start_motion)
 def test_pause(self):
  self.tick(10.1,False);self.assertFalse(self.b.auto_allowed)
 def test_transition_waits_without_cancelling_auto_resume(self):
  with patch.object(mod.Path,'write_text'),patch.object(mod.Path,'replace'),patch.object(mod.Path,'read_text',side_effect=lambda: __import__('json').dumps({'token':self.b.request_token,'epoch':2})):
   self.tick(10.1,heading_valid=False);self.tick(10.4,heading_valid=False)
   self.assertTrue(self.b.auto_allowed);self.assertFalse(self.o.start_motion)
   self.assertIsNone(self.b.request_token)
   self.tick(10.5);self.tick(10.71)
   self.assertTrue(self.o.start_motion)
 def test_stop(self):
  self.o.stop_latched=True;self.tick(10.1);self.assertFalse(self.b.active);self.assertFalse(self.o.start_motion)
 def test_no_ack(self):
  with patch.object(mod.Path,'write_text'),patch.object(mod.Path,'replace'),patch.object(mod.Path,'read_text',return_value='{}'):
   self.tick(10.1);self.tick(10.7);self.assertFalse(self.o.start_motion)
 def test_freshness_break(self):
  self.tick(10.1);self.b.depth_ok=False;self.tick(10.4);self.assertIsNone(self.b.fresh_since)
 def test_producer_restart_latches(self):
  self.b.enter('realtime context producer restarted; operator re-arm required')
  self.assertFalse(self.b.auto_allowed)
 def test_explicit_x_cancels(self):
  self.tick(10.1);self.b.cancel_auto();self.tick(10.5)
  self.assertFalse(self.o.start_motion)
 def test_loss_during_resume(self):
  with patch.object(mod.Path,'write_text'),patch.object(mod.Path,'replace'),patch.object(mod.Path,'read_text',side_effect=lambda: __import__('json').dumps({'token':self.b.request_token,'epoch':2})):
   self.tick(10.1);self.tick(10.31);self.assertTrue(self.o.start_motion)
   self.b.enter('pose stale');self.b.depth_ok=False;self.tick(10.4)
   self.assertFalse(self.o.start_motion);self.assertIsNone(self.b.resume_started)
if __name__=='__main__':unittest.main()
