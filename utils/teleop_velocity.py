"""QV teleop velocity paired with the exact retarget pose used for z."""
import struct
from collections import deque
import numpy as np
from utils.common import RealtimeContextMessage

class VelocityContextMessage(RealtimeContextMessage):
    FOOTER = struct.Struct("<4sfff")
    EPOCH = struct.Struct("<4sQd")
    def __init__(self, *, velocity_heading_xy=None, velocity_valid=False, reference_epoch=None, reference_sample_s=None, **kwargs):
        super().__init__(**kwargs)
        self.reference_epoch = reference_epoch
        self.reference_sample_s = reference_sample_s
        v=np.zeros(2,np.float32) if velocity_heading_xy is None else np.asarray(velocity_heading_xy,np.float32).reshape(2)
        if not np.isfinite(v).all(): raise ValueError("nonfinite velocity")
        self.velocity_valid=bool(velocity_valid)
        self.velocity_heading_xy=v.copy() if self.velocity_valid else np.zeros(2,np.float32)
    def to_bytes(self):
        data = super().to_bytes()+self.FOOTER.pack(b"QVV1",*self.velocity_heading_xy,float(self.velocity_valid))
        if self.reference_epoch is not None:
            if self.reference_sample_s is None or not np.isfinite(self.reference_sample_s):
                raise ValueError("missing reference sample time")
            data += self.EPOCH.pack(b"QVE2", int(self.reference_epoch), float(self.reference_sample_s))
        return data
    @classmethod
    def from_bytes(cls,data):
        n=RealtimeContextMessage._HEADER.size+(256+2)*4
        if len(data)==n:
            base=RealtimeContextMessage.from_bytes(data)
            return cls(**base.__dict__)
        extra = {}
        if len(data) == n + cls.FOOTER.size + cls.EPOCH.size:
            tag, epoch, sample = cls.EPOCH.unpack(data[-cls.EPOCH.size:])
            if tag != b"QVE2" or not np.isfinite(sample) or sample <= 0:
                raise ValueError("invalid reference epoch")
            extra = dict(reference_epoch=epoch, reference_sample_s=sample)
            data = data[:-cls.EPOCH.size]
        if len(data)!=n+cls.FOOTER.size:raise ValueError("QV context size mismatch")
        tag,x,y,valid=cls.FOOTER.unpack(data[n:])
        if tag!=b"QVV1" or valid not in (0.,1.):raise ValueError("invalid QV footer")
        base=RealtimeContextMessage.from_bytes(data[:n])
        if valid and not (base.heading_valid and base.reference_live):raise ValueError("velocity without live heading")
        return cls(**base.__dict__,velocity_heading_xy=[x,y],velocity_valid=bool(valid),**extra)

class RetargetVelocity:
    """Fit recent source-time positions in world XY, then rotate into current yaw.

    A 60 ms causal window (normally four samples at 50 Hz) rejects single-step
    noise without a second low-pass filter. Gaps/jumps require fresh warmup.
    """
    def __init__(self, max_speed=3., max_gap=.2, window_s=.06, min_span=.035):
        self.max_speed=float(max_speed); self.max_gap=float(max_gap)
        self.window_s=float(window_s); self.min_span=float(min_span)
        self.reset()
    def reset(self):
        self.samples=deque()
        self.world_velocity=np.zeros(2,np.float64)
        self.valid=False
    def update(self,position,heading,now,active):
        zero=np.zeros(2,np.float32)
        if not active:
            self.reset(); return zero,False
        p=np.asarray(position,np.float64).reshape(3)
        h=np.asarray(heading,np.float64).reshape(2)
        if not np.isfinite(p).all() or not np.isfinite(h).all() or not np.isfinite(now) or np.linalg.norm(h)<1e-6:
            self.reset(); return zero,False
        now=float(now); h=h/np.linalg.norm(h)
        if self.samples:
            old_t,old_p=self.samples[-1]; dt=now-old_t
            if dt == 0:
                # A repeated source sample neither advances nor erases history.
                c,s=h; v=self.world_velocity
                return np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1]],np.float32),self.valid
            if dt < 0 or dt > self.max_gap or np.linalg.norm(p[:2]-old_p)/dt > self.max_speed:
                self.reset(); self.samples.append((now,p[:2].copy()))
                return zero,False
            if dt < .005:
                return zero,False
        self.samples.append((now,p[:2].copy()))
        while len(self.samples)>1 and now-self.samples[0][0]>self.window_s+1e-6:
            self.samples.popleft()
        if len(self.samples)<3 or now-self.samples[0][0]<self.min_span:
            self.valid=False; return zero,False
        t=np.array([a-now for a,b in self.samples]); t-=t.mean()
        xy=np.array([b for a,b in self.samples])
        v=(t[:,None]*(xy-xy.mean(axis=0))).sum(axis=0)/np.dot(t,t)
        if not np.isfinite(v).all() or np.linalg.norm(v)>self.max_speed:
            self.reset(); self.samples.append((now,p[:2].copy()))
            return zero,False
        self.world_velocity=v; self.valid=True
        c,s=h
        return np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1]],np.float32),True

def command_from_heading_error(velocity,valid,heading_error,active):
    """Heading observation is [1-cos(target-robot), sin(target-robot)]."""
    out=np.zeros((1,3),np.float32)
    if not active or not valid:return out
    v=np.asarray(velocity,np.float32).reshape(2)
    e=np.asarray(heading_error,np.float32).reshape(2)
    if not np.isfinite(v).all() or not np.isfinite(e).all():raise ValueError("nonfinite QV input")
    c,s=1.-float(e[0]),float(e[1])
    if abs(c*c+s*s-1)>1e-3:raise ValueError("QV requires unamplified unit heading error")
    out[0]=[c*v[0]-s*v[1],s*v[0]+c*v[1],1.]
    return out
