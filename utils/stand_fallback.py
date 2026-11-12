"""Opt-in input-loss standing fallback. Does not replace low-state/stop safety."""
import time
import json
from pathlib import Path
import numpy as np

class StandFallback:
    def __init__(self, owner, path, blend_s=1.0):
        self.owner = owner
        self.z = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
        if self.z.shape != (256,) or not np.isfinite(self.z).all() or np.linalg.norm(self.z) < 1e-6:
            raise ValueError('invalid encoded standing z')
        self.blend_s = float(blend_s)
        if not np.isfinite(self.blend_s) or self.blend_s <= 0:
            raise ValueError('invalid standing blend duration')
        self.fresh_wait_s = float(owner.policy_config.get('teleop_auto_resume_fresh_s', .2))
        if not .05 <= self.fresh_wait_s <= 1.:
            raise ValueError('teleop_auto_resume_fresh_s must be between .05 and 1 second')
        self.active = False
        self.resume_started = None
        self.cache = None
        self.depth_ok = False
        self.original_depth = owner.state_processor.get_latest_direct_depth_latent
        owner.state_processor.get_latest_direct_depth_latent = self.depth

    def enter(self, reason):
        o = self.owner
        if not o.use_policy_action or o.stop_latched:
            return False
        if not self.active:
            self.seed = np.asarray(o.ctx_latest, dtype=np.float32).copy()
            self.started = time.monotonic()
            self.active = True
            self.auto_allowed = bool(o.policy_config.get('teleop_auto_resume', False)) and bool(o.start_motion)
            if 'restarted' in reason:
                self.auto_allowed = False
            self.fresh_since = None
            self.request_token = None
            self.resume_started = None
            o._warn_throttled('_last_stand_fallback_warning', 'STANDING FALLBACK: ' + reason)
        if 'restarted' in reason:
            self.auto_allowed = False
        if self.resume_started is not None:
            self.seed = np.asarray(o.ctx_latest, dtype=np.float32).copy()
            self.started = time.monotonic()
            self.resume_started = None
        self.fresh_since = None
        self.request_token = None
        o.start_motion = False
        return True

    def depth(self, **kwargs):
        try:
            result = self.original_depth(**kwargs)
            latent = np.asarray(result.depth_latent)
            if latent.shape != (256,) or not np.isfinite(latent).all():
                raise ValueError('invalid depth latent')
            self.cache = result
            self.depth_ok = True
            return result
        except RuntimeError as exc:
            self.depth_ok = False
            # Only missing/stale input is recoverable. Future timestamps and
            # malformed data must still fail through the original safety path.
            if not any(s in str(exc) for s in ('stale', 'no direct-depth latent received')):
                raise
            if self.cache is None or not self.enter(str(exc)):
                raise
            self.owner._warn_throttled('_last_held_depth_warning',
                'STANDING FALLBACK: using cached depth; capture age %.2f s' %
                (time.monotonic() - self.cache.capture_monotonic_s), interval_s=5.)
            return self.cache

    def preflight(self):
        self.depth(max_receive_age_s=.1, max_capture_age_s=.12, max_result_age_s=.1)

    def cancel_auto(self):
        self.auto_allowed = False
        self.resume_started = None

    def inputs_fresh(self, now):
        o = self.owner
        fresh = self.depth_ok and all(t is not None and -.01 <= now - t <= limit for t, limit in (
            (o.ctx_last_z_monotonic, o.ctx_zmq_timeout_s),
            (o.ctx_latest_result_monotonic, o.ctx_zmq_result_timeout_s),
            (o.ctx_latest_source_monotonic, o.ctx_zmq_source_timeout_s)))
        if o.pico_only_control:
            fresh = fresh and o.pico_control_have_valid_source and all(
                t is not None and -.01 <= now - t <= o.pico_control_source_timeout_s for t in (
                    o.pico_control_last_source_monotonic, o.pico_control_last_receive_monotonic))
        return fresh

    def apply(self):
        if not self.active:
            return
        o = self.owner
        now = time.monotonic()
        if o.stop_latched or not o.use_policy_action:
            self.cancel_auto()
            self.active = False
            return
        packet = getattr(o, 'ctx_input_message', None)
        if packet is not None:
            o.ctx_latest = packet.z.copy()
            o.ctx_latest_heading_relative_xy = packet.heading_relative_xy.copy()
            o.ctx_latest_heading_valid = bool(packet.heading_valid)
            o.ctx_latest_velocity_heading_xy = packet.velocity_heading_xy.copy()
            o.ctx_latest_velocity_valid = bool(packet.velocity_valid)
            o.ctx_latest_reference_live = bool(packet.reference_live)
        fresh = self.inputs_fresh(now)
        live = bool(o.ctx_latest_reference_live)
        ready = live and bool(o.ctx_latest_heading_valid)
        if not live:
            # Only a freshly received paused packet means intentional pause.
            if o.ctx_last_z_monotonic is not None and now - o.ctx_last_z_monotonic < .03:
                self.cancel_auto()
        if not fresh:
            self.fresh_since = None
            self.request_token = None
            self.resume_started = None
        if now - self.started > 3.0 and self.resume_started is None:
            self.auto_allowed = False
        if not ready:
            self.fresh_since = None
        if self.auto_allowed and fresh and ready:
            if self.fresh_since is None:
                self.fresh_since = now
            if now - self.fresh_since >= self.fresh_wait_s and self.request_token is None:
                self.request_token = str(time.monotonic_ns())
                path = Path('/tmp/pbfm_teleop_recovery_request.json')
                tmp = path.with_suffix('.tmp')
                tmp.write_text(json.dumps({'token': self.request_token}))
                tmp.replace(path)
            if self.request_token is not None and self.resume_started is None:
                try:
                    ack = json.loads(Path('/tmp/pbfm_teleop_recovery_ack.json').read_text())
                except (OSError, ValueError):
                    ack = {}
                if (ack.get('token') == self.request_token and
                        ack.get('epoch') == o.ctx_latest_reference_epoch):
                    self.resume_started = now
                    self.resume_seed = self.held_z.copy()
                    o.start_motion = True
                    o.ctx_started_monotonic = now
                    o._warn_throttled('_last_auto_resume_warning', 'AUTO RESUME: fresh inputs %.2fs; reference reset acknowledged; blending 1s' % self.fresh_wait_s, interval_s=0.)
        if self.resume_started is not None:
            fraction = min(1., max(0., now - self.resume_started))
            z = (1. - fraction) * self.resume_seed + fraction * o.ctx_latest
            o.ctx_latest_velocity_heading_xy = np.asarray(o.ctx_latest_velocity_heading_xy) * fraction
            # Relative heading is re-anchored by the observation on motion start.
            xy = np.asarray(o.ctx_latest_heading_relative_xy)
            angle = np.arctan2(xy[1], xy[0]) * fraction
            o.ctx_latest_heading_relative_xy = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            o.start_motion = True
            if fraction >= 1.:
                self.active = False
                self.resume_started = None
        else:
            fraction = min(1., max(0., (now - self.started) / self.blend_s))
            z = (1. - fraction) * self.seed + fraction * self.z
            o.ctx_latest_heading_valid = False
            o.ctx_latest_velocity_valid = False
            o.ctx_latest_reference_live = False
            o.start_motion = False
        norm = np.linalg.norm(z)
        if norm < 1e-6:
            z = self.z.copy()
            norm = np.linalg.norm(z)
        o.ctx_latest = (z * (o.ctx_norm_ref / norm)).astype(np.float32)
        self.held_z = o.ctx_latest.copy()
        o.ctx_window.clear()

    def rearm(self):
        if not self.active:
            return True
        o = self.owner
        if not self.inputs_fresh(time.monotonic()):
            o._warn_throttled('_last_stand_rearm_warning', 'Standing fallback: inputs not fresh; re-arm rejected')
            return False
        self.active = False
        self.resume_started = None
        o.ctx_window.clear()
        return True
