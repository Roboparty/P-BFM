"""Remove reference-root lateral translation before backward encoding."""
import numpy as np
class ForwardOnlyRoot:
    def __init__(self):
        self.reset()
    def reset(self):
        self.raw = self.filtered = None
        self.sample = None
    def update(self, position, heading, sample, active):
        p = np.asarray(position, dtype=np.float64).reshape(3)
        h = np.asarray(heading, dtype=np.float64).reshape(2)
        if not np.isfinite(p).all() or not np.isfinite(h).all() or not np.isfinite(sample) or np.linalg.norm(h) < 1e-6:
            raise ValueError('Invalid forward-only root reference')
        if not active:
            self.reset()
            return p.astype(np.float32), False
        h = h / np.linalg.norm(h)
        if self.raw is None or sample < self.sample or sample-self.sample > .1:
            self.raw, self.filtered, self.sample = p.copy(), p.copy(), sample
            return p.astype(np.float32), True
        if sample == self.sample:
            return self.filtered.astype(np.float32).copy(), False
        delta = p[:2]-self.raw[:2]
        self.filtered[:2] += h * np.dot(delta, h)
        self.filtered[2] = p[2]
        self.raw, self.sample = p.copy(), sample
        return self.filtered.astype(np.float32).copy(), False
