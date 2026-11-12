"""Absolute host-monotonic scheduling shared by depth and policy processes."""
import math

def next_phase_tick(now,period,phase=0.0):
 if not all(math.isfinite(x) for x in (now,period,phase)) or period<=0 or not 0<=phase<period:
  raise ValueError('Require finite time, positive period and 0 <= phase < period')
 return (math.floor((now-phase)/period)+1)*period+phase

def advance_phase_tick(previous,now,period,phase=0.0):
 candidate=previous+period
 return candidate if candidate>=now else next_phase_tick(now,period,phase)
