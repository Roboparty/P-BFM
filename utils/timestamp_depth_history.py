"""Depth history sampled by capture time, independently of encoder tick rate."""
from collections import deque
import numpy as np
from rl_policy.direct_depth import DirectDepthConfig,preprocess_direct_depth_frame

class TimestampDepthHistory:
 def __init__(self,config=DirectDepthConfig()):
  config.validate();self.config=config;self.frames=deque()
  self.ages=np.arange(7,-1,-1,dtype=np.float64)*.1 + config.fixed_delay_frames*.02
 def reset(self):self.frames.clear()
 def append(self,depth_m,*,timestamp_s,reset=False):
  timestamp_s=float(timestamp_s)
  if not np.isfinite(timestamp_s) or timestamp_s<0:raise ValueError('Invalid capture timestamp')
  if reset or (self.frames and timestamp_s<self.frames[-1][0]):self.reset()
  if not self.frames or timestamp_s>self.frames[-1][0]:
   self.frames.append((timestamp_s,preprocess_direct_depth_frame(depth_m,self.config)))
  cutoff=timestamp_s-float(self.ages[0])
  while len(self.frames)>1 and self.frames[1][0]<=cutoff+1e-9:self.frames.popleft()
  times=np.array([x[0] for x in self.frames]);targets=timestamp_s-self.ages
  # Causal nearest-previous selection; retain oldest frame to bootstrap history.
  indices=np.maximum(0,np.searchsorted(times,targets+1e-9,side='right')-1)
  return np.stack([self.frames[int(i)][1] for i in indices])

def resolve_encoder_rate(policy_config,override=None):
 rate=float(policy_config.get('depth_encoder_rate_hz',50.) if override is None else override)
 if not np.isfinite(rate) or not 10<=rate<=120:raise ValueError('depth_encoder_rate_hz must be between 10 and 120')
 return rate
