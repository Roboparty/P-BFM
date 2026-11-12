"""Reference pose clock and bounded recovery blend, independent of inference time."""
from dataclasses import replace
import numpy as np
class PoseTimeline:
 def __init__(self,recovery_s=.3,gap_s=.1):
  if not np.isfinite(recovery_s) or recovery_s<0:raise ValueError('pose transition must be finite and non-negative')
  if not np.isfinite(gap_s) or gap_s<=0:raise ValueError('pose gap must be finite and positive')
  self.recovery_s=recovery_s;self.gap_s=gap_s;self.last=None;self.source=None;self.seed=None;self.began=None;self.restart_pending=False
 def restart(self):
  self.restart_pending=True
 def update(self,pose,now):
  source=pose.source_sample_s
  if not np.isfinite(now) or source is None or not np.isfinite(source):raise ValueError('pose clock missing')
  new=self.source is None or source!=self.source
  if self.last is not None and source == self.source and pose.reference_epoch == self.last.reference_epoch and pose.retarget_paused==self.last.retarget_paused:
   return self.last
  changed=self.last is not None and (pose.retarget_paused!=self.last.retarget_paused or pose.reference_epoch!=self.last.reference_epoch)
  gap=new and self.source is not None and (source-self.source>self.gap_s or source<self.source)
  if self.last is not None and (changed or gap or self.restart_pending):
   self.seed=self.last;self.began=source
  self.restart_pending=False
  self.source=source
  fraction=1. if self.began is None or self.recovery_s==0 else min(1.,max(0.,(source-self.began)/self.recovery_s))
  out=replace(pose,sample_time_s=source,transitioning=fraction<1.)
  if fraction<1.:
   a=fraction*fraction*(3.-2.*fraction);old=self.seed
   q0=old.root_quat_wxyz.astype(np.float64);q1=pose.root_quat_wxyz.astype(np.float64)
   q0/=np.linalg.norm(q0);q1/=np.linalg.norm(q1)
   dot=float(np.dot(q0,q1))
   if dot<0:q1=-q1;dot=-dot
   if dot>.9995:q=(1-a)*q0+a*q1
   else:
    theta=np.arccos(np.clip(dot,-1,1));q=(np.sin((1-a)*theta)*q0+np.sin(a*theta)*q1)/np.sin(theta)
   out=replace(out,root_pos=((1-a)*old.root_pos+a*pose.root_pos).astype(np.float32),dof_pos=((1-a)*old.dof_pos+a*pose.dof_pos).astype(np.float32),root_quat_wxyz=(q/np.linalg.norm(q)).astype(np.float32))
  else:self.seed=None;self.began=None
  self.last=out
  return out
