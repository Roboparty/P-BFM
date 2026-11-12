import unittest, tempfile, time, ast
from pathlib import Path
from types import SimpleNamespace as NS
from collections import deque
import numpy as np
try:
 from utils.stand_fallback import StandFallback
except ModuleNotFoundError:
 from stand_fallback import StandFallback

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)/'stand.npy'
  z=np.zeros(256,dtype=np.float32);z[1]=16;np.save(self.path,z)
  self.error=None
  self.result=NS(depth_latent=np.ones(256),capture_monotonic_s=time.monotonic())
  def get(**kw):
   if self.error: raise self.error
   return self.result
  self.o=NS(state_processor=NS(get_latest_direct_depth_latent=get),use_policy_action=True,stop_latched=False,ctx_latest=np.eye(1,256,dtype=np.float32)[0]*16,ctx_norm_ref=16.,ctx_window=deque(),start_motion=True,_warn_throttled=lambda *a,**kw:None,pico_only_control=False)
  self.o.policy_config={'teleop_auto_resume': False}
  self.o.ctx_latest_reference_live=False
  self.o.ctx_last_z_monotonic=None
  self.o.ctx_latest_result_monotonic=None
  self.o.ctx_latest_source_monotonic=None
  self.o.ctx_zmq_timeout_s=.2
  self.o.ctx_zmq_result_timeout_s=.1
  self.o.ctx_zmq_source_timeout_s=.2
  self.f=StandFallback(self.o,self.path)
 def tearDown(self): self.tmp.cleanup()
 def test_depth_hold_and_stand(self):
  self.f.preflight(); cache=self.f.cache
  self.error=RuntimeError('direct-depth latent capture is stale (150 ms)')
  self.f.preflight();self.assertTrue(self.f.active);self.assertIs(self.f.cache,cache)
  self.assertTrue(self.o.use_policy_action);self.assertFalse(self.o.start_motion)
  self.f.started-=2;self.f.apply()
  np.testing.assert_allclose(self.o.ctx_latest,self.f.z)
  self.assertFalse(self.o.ctx_latest_heading_valid);self.assertFalse(self.o.ctx_latest_velocity_valid)
 def test_no_cache_and_bad_data_fail(self):
  self.error=RuntimeError('no direct-depth latent received')
  with self.assertRaises(RuntimeError): self.f.preflight()
  self.error=None;self.f.preflight();self.error=RuntimeError('timestamp is in the future')
  with self.assertRaises(RuntimeError): self.f.preflight()
 def test_recovery_requires_fresh_manual_rearm(self):
  self.f.preflight();self.f.enter('teleop timeout');self.f.started-=2
  self.f.preflight();self.f.apply();self.assertTrue(self.f.active)
  now=time.monotonic()
  for attr in ('ctx_last_z_monotonic','ctx_latest_result_monotonic','ctx_latest_source_monotonic'):setattr(self.o,attr,now-1)
  self.o.ctx_zmq_timeout_s=.2;self.o.ctx_zmq_result_timeout_s=.1;self.o.ctx_zmq_source_timeout_s=.2
  self.assertFalse(self.f.rearm())
  for attr in ('ctx_last_z_monotonic','ctx_latest_result_monotonic','ctx_latest_source_monotonic'):setattr(self.o,attr,now)
  self.assertTrue(self.f.rearm());self.assertFalse(self.f.active)
 def test_stopped_policy_never_enabled(self):
  self.o.use_policy_action=False
  self.assertFalse(self.f.enter('timeout'));self.assertFalse(self.f.active)
  self.o.use_policy_action=True;self.o.stop_latched=True
  self.assertFalse(self.f.enter('timeout'))
 def test_rearm_rejects_future_or_missing_timestamps(self):
  self.f.preflight();self.f.enter('teleop timeout')
  now=time.monotonic()
  attrs=('ctx_last_z_monotonic','ctx_latest_result_monotonic','ctx_latest_source_monotonic')
  for attr in attrs:setattr(self.o,attr,now)
  for attr in attrs:
   setattr(self.o,attr,now+1)
   self.assertFalse(self.f.rearm());self.assertTrue(self.f.active)
   setattr(self.o,attr,now)
  self.o.pico_only_control=True;self.o.pico_control_have_valid_source=True
  self.o.pico_control_source_timeout_s=.25
  self.o.pico_control_last_source_monotonic=now
  self.o.pico_control_last_receive_monotonic=now
  for attr in ('pico_control_last_source_monotonic','pico_control_last_receive_monotonic'):
   for stamp in (now+1,None):
    setattr(self.o,attr,stamp)
    self.assertFalse(self.f.rearm());self.assertTrue(self.f.active)
   setattr(self.o,attr,now)
  self.assertTrue(self.f.rearm())
 def test_transition_norm(self):
  self.f.enter('timeout');self.f.started-=.5;self.f.apply()
  self.assertAlmostEqual(float(np.linalg.norm(self.o.ctx_latest)),16,places=4)
  self.assertGreater(self.o.ctx_latest[0],0);self.assertGreater(self.o.ctx_latest[1],0)
 def test_policy_routes_and_stop_override(self):
  tree=ast.parse((Path('rl_policy/ufo_qv_policy.py') if Path('rl_policy/ufo_qv_policy.py').exists() else Path('policy.py')).read_text())
  methods={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
  for name in ('_check_realtime_z_watchdog','_check_pico_control_watchdog'):
   calls=[n.func.attr for n in ast.walk(methods[name]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
   self.assertIn('enter_input_fallback',calls);self.assertNotIn('enter_safe_stop',calls)
  # Execute actual emergency-stop method with a fake owner: fallback cannot override it.
  m=ast.Module(body=[methods['enter_safe_stop']],type_ignores=[]);ns={ 'np':np }
  exec(compile(ast.fix_missing_locations(m),'policy','exec'),ns)
  self.o.stand_fallback=self.f;self.f.active=True;self.o.last_action=np.ones(29)
  self.o._reset_tracking_state_to_stop=lambda **kw:None
  ns['enter_safe_stop'](self.o,'operator stop')
  self.assertFalse(self.f.active);self.assertFalse(self.o.use_policy_action)

if __name__=='__main__': unittest.main()
