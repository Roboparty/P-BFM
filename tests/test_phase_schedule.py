import unittest
from utils.phase_schedule import next_phase_tick,advance_phase_tick
class PhaseTests(unittest.TestCase):
 def test_shared_grid(self):
  for now in [0.,1.123,9999.987,1000000.03]:
   a=next_phase_tick(now,.02);b=next_phase_tick(a,.02,.011)
   self.assertAlmostEqual(b-a,.011,places=8)
 def test_no_drift(self):
  tick=next_phase_tick(1.234,.02,.011);first=tick
  for i in range(1000):tick=advance_phase_tick(tick,tick+.008,.02,.011)
  self.assertAlmostEqual(tick-first,20.,places=8)
 def test_skip_late(self):
  self.assertAlmostEqual(advance_phase_tick(1.011,1.087,.02,.011),1.091)
 def test_future(self):
  for i in range(10000):
   now=i*.001
   for phase in [0.,.011]:
    n=next_phase_tick(now,.02,phase);self.assertGreater(n,now-1e-12);self.assertLessEqual(n-now,.020000001)
 def test_invalid(self):
  for args in [(0,0,0),(0,.02,.03),(float('nan'),.02,0)]:
   with self.assertRaises(ValueError):next_phase_tick(*args)
if __name__=='__main__':unittest.main()
