"""QV velocity commands in the current yaw frame, aligned at playback start."""
import numpy as np

def unit(h):
    h=np.asarray(h,dtype=np.float32).reshape(2)
    n=float(np.linalg.norm(h))
    if not np.isfinite(h).all() or n<1e-6: raise ValueError('Invalid heading')
    return h/n

def to_world(v,h):
    c,s=unit(h);x,y=np.asarray(v,dtype=np.float32)
    return np.array([c*x-s*y,s*x+c*y],dtype=np.float32)

def to_yaw(v,h):
    c,s=unit(h);x,y=np.asarray(v,dtype=np.float32)
    return np.array([c*x+s*y,-s*x+c*y],dtype=np.float32)

class VelocityRuntime:
    def __init__(self, velocities, validity):
        self.velocities = np.asarray(velocities, dtype=np.float32)
        self.validity = np.asarray(validity).reshape(-1)
        if self.velocities.ndim != 2 or self.velocities.shape[1] != 2 or not np.isfinite(self.velocities).all():
            raise ValueError('Invalid reference velocity')
        if len(self.validity) != len(self.velocities) or not np.isin(self.validity, [0, 1]).all():
            raise ValueError('Invalid velocity validity')
        self.reset()
    def reset(self):
        self.anchor = None
        self.key = None
    def command(self, heading, frame, active, key):
        if not active:
            self.reset()
            return np.zeros((1, 3), dtype=np.float32)
        h = unit(heading)
        if self.anchor is None or self.key != key:
            self.anchor = h.copy()
            self.key = key
        if not 0 <= frame < len(self.velocities):
            raise ValueError('Velocity frame outside context')
        if not self.validity[frame]:
            return np.zeros((1, 3), dtype=np.float32)
        local = to_yaw(to_world(self.velocities[frame], self.anchor), h)
        return np.array([[local[0], local[1], 1]], dtype=np.float32)
