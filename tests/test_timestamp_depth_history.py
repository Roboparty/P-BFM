import unittest
import numpy as np
from rl_policy.direct_depth import DirectDepthConfig,DirectDepthHistory
from utils.timestamp_depth_history import TimestampDepthHistory,resolve_encoder_rate
C=DirectDepthConfig(vertical_flip=False,gaussian_enabled=False)
def frame(v):return np.full((36,64),v*2.5/255,np.float32)
class HistoryTests(unittest.TestCase):
 def test_rates(self):
  for rate in (20,30,50,60,75,100,120):
   h=TimestampDepthHistory(C)
   for i in range(rate+1):a=h.append(frame(i),timestamp_s=10+i/rate)
   expected=np.floor((1-np.arange(7,-1,-1)*.1)*rate+1e-8).astype(int)
   np.testing.assert_array_equal(a[:,0,0],expected)
 def test_matches_original_50(self):
  a=TimestampDepthHistory(C);b=DirectDepthHistory(C)
  for i in range(150):np.testing.assert_array_equal(a.append(frame(i),timestamp_s=100+i/50),b.append(frame(i)))
 def test_duplicates(self):
  h=TimestampDepthHistory(C);a=h.append(frame(1),timestamp_s=10)
  for _ in range(100):np.testing.assert_array_equal(a,h.append(frame(99),timestamp_s=10))
  self.assertEqual(len(h.frames),1)
 def test_reset_and_clock_reverse(self):
  h=TimestampDepthHistory(C);h.append(frame(1),timestamp_s=10)
  self.assertTrue((h.append(frame(2),timestamp_s=9)==2).all())
  self.assertTrue((h.append(frame(3),timestamp_s=11,reset=True)==3).all())
 def test_config(self):
  self.assertEqual(resolve_encoder_rate({}),50)
  self.assertEqual(resolve_encoder_rate({'depth_encoder_rate_hz':60}),60)
  self.assertEqual(resolve_encoder_rate({'depth_encoder_rate_hz':60},30),30)
  for x in [0,121,float('nan')]:
   with self.assertRaises(ValueError):resolve_encoder_rate({'depth_encoder_rate_hz':x})
if __name__=='__main__':unittest.main()
