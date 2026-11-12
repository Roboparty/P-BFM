"""Training-camera nearest sampling; RealSense K uses integer pixel centers."""
import numpy as np

class TrainingDepthSampler:
    def __init__(self, raw_k, source_shape, processing_profile="legacy_rp1"):
        if processing_profile not in {"legacy_rp1", "optical_z_maxcrop18_reset_v2"}:
            raise ValueError("Unknown depth processing profile")
        self.processing_profile = processing_profile
        self.shape = tuple(source_shape)
        k = np.asarray(raw_k, dtype=np.float64)
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0,0], k[1,1]) <= 0:
            raise ValueError('Invalid camera intrinsics')
        fx = 64 / (2 * np.tan(np.deg2rad(89.51) / 2))
        fy = 36 / (2 * np.tan(np.deg2rad(58.29) / 2))
        # Equivalent to training (u+.5-32)/fx, (v+.5-18)/fy.
        self.output_k = np.array([[fx,0,31.5],[0,fy,17.5],[0,0,1]], np.float32)
        rows, cols = np.indices((36,32), dtype=np.float64)
        x = (cols + 16 + .5 - 32) / fx
        y = (rows + .5 - 18) / fy
        u = x*k[0,0] + k[0,2]
        v = y*k[1,1] + k[1,2]
        self.u = np.floor(u+.5).astype(np.int64)
        self.v = np.floor(v+.5).astype(np.int64)
        self.covered = (self.u>=0)&(self.u<self.shape[1])&(self.v>=0)&(self.v<self.shape[0])
        self.ray_scale = np.sqrt(1+x*x+y*y)
    def sample(self, depth_m):
        d = np.asarray(depth_m)
        if d.shape != self.shape: raise ValueError('Camera dimensions changed')
        # Only the cropped training rays are sampled. Outside FOV is zero,
        # never border-replicated. Keep the 36x64 wire shape for existing clients.
        z = np.zeros((36,32),np.float32)
        z[self.covered] = d[self.v[self.covered],self.u[self.covered]]
        if self.processing_profile == "optical_z_maxcrop18_reset_v2":
            # Optical-Z clipping, no radial cutoff. Native zero denotes no hit.
            valid = self.covered & np.isfinite(z) & (z > 0)
            out = np.full((36,64), 2.5, np.float32)
            out[:,16:48] = np.where(valid, np.clip(z,0,2.5), 2.5)
            return out
        valid = self.covered & np.isfinite(z) & (z>0) & (z*self.ray_scale<=2.5)
        out = np.zeros((36,64),np.float32)
        out[:,16:48] = np.where(valid,z,0)
        return out
