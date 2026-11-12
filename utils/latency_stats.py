"""Bounded once-per-second latency summaries; no raw-motion recording."""
import os
import time
from collections import defaultdict, deque
import numpy as np


class LatencyStats:
    def __init__(self, label, interval=1.0):
        self.enabled = os.environ.get('PBFM_LATENCY_STATS', '0') == '1'
        self.label = label
        self.interval = interval
        self.start = time.monotonic()
        self.values = defaultdict(lambda: deque(maxlen=512))

    def add(self, **seconds):
        if not self.enabled:
            return
        for key, value in seconds.items():
            if value is not None and np.isfinite(value):
                self.values[key].append(float(value)*1000)
        now = time.monotonic()
        if now - self.start < self.interval:
            return
        parts = []
        for key, values in self.values.items():
            if values:
                a = np.asarray(values)
                parts.append('%s=%.2f/%.2f/%.2f(n=%d)' %
                             (key, a.mean(), np.percentile(a,95), a.max(), len(a)))
        print('[LATENCY %s] ms mean/p95/max: %s' % (self.label, ' '.join(parts)), flush=True)
        self.values.clear()
        self.start = now
