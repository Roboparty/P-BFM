from .base import Observation

import numpy as np
from loguru import logger
import time
from typing import Any, Dict
from utils.math import quat_rotate_inverse_numpy, quat_rotate_numpy

from rl_policy.direct_depth import DirectDepthConfig, DirectDepthHistory


def _projected_gravity_xy_bias(env) -> np.ndarray:
    """Return the configured additive XY bias in unscaled gravity units."""
    raw_bias = env.policy_config.get("projected_gravity_xy_bias", [0.0, 0.0])
    bias = np.asarray(raw_bias, dtype=np.float32)
    if bias.shape != (2,):
        raise ValueError(
            "projected_gravity_xy_bias must contain exactly [x, y]"
        )
    if not np.isfinite(bias).all():
        raise ValueError("projected_gravity_xy_bias must be finite")
    return bias


def _apply_projected_gravity_xy_bias(
    projected_gravity: np.ndarray, xy_bias: np.ndarray
) -> np.ndarray:
    biased = np.asarray(projected_gravity, dtype=np.float32).copy()
    biased[:2] += xy_bias
    return biased


class base_ang_vel(Observation):
    def compute(self, scale: float = 1.0) -> np.ndarray:
        # base_ang_vel = self.state_processor.root_ang_vel_b   # for medium-size model and large-size model without dr
        base_ang_vel = self.state_processor.root_ang_vel_b  # large-size models with dr0109
        return base_ang_vel * scale

class base_ang_vel_history(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.base_ang_vel_history = np.zeros((self.steps, 3))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.base_ang_vel_history = np.roll(self.base_ang_vel_history, 1, axis=0)
        self.base_ang_vel_history[0, :] = self.state_processor.root_ang_vel_b

    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.base_ang_vel_history.reshape(-1) * scale

class projected_gravity(Observation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.v = np.array([0, 0, -1])
        self.xy_bias = _projected_gravity_xy_bias(self.env)
        logger.info(f"Projected-gravity XY bias={self.xy_bias.tolist()}")

    def compute(self, scale: float = 1.0) -> np.ndarray:
        base_quat = self.state_processor.root_quat_b
        projected_gravity = quat_rotate_inverse_numpy(
            base_quat[None, :], 
            self.v[None, :]
        ).squeeze(0)
        projected_gravity = _apply_projected_gravity_xy_bias(
            projected_gravity, self.xy_bias
        )
        return projected_gravity * scale
    
class projected_gravity_history(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.projected_gravity_history = np.zeros((self.steps, 3))
        self.v = np.array([0, 0, -1])
        self.xy_bias = _projected_gravity_xy_bias(self.env)
    
    def update(self, data: Dict[str, Any]) -> None:
        base_quat = self.state_processor.root_quat_b
        projected_gravity = quat_rotate_inverse_numpy(
            base_quat[None, :], 
            self.v[None, :]
        ).squeeze(0)
        projected_gravity = _apply_projected_gravity_xy_bias(
            projected_gravity, self.xy_bias
        )
        self.projected_gravity_history = np.roll(self.projected_gravity_history, 1, axis=0)
        self.projected_gravity_history[0, :] = projected_gravity

    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.projected_gravity_history.reshape(-1) * scale

class dof_pos_minus_default(Observation):
    def __init__(self, default_pos: list, **kwargs):
        super().__init__(**kwargs)
        self.default_pos = np.array(default_pos)

    def compute(self, scale: float = 1.0) -> np.ndarray:
        return (self.state_processor.joint_pos - self.default_pos) * scale

class dof_pos_minus_default_history(Observation):
    def __init__(self, steps: int, default_pos: list, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.default_pos = np.array(default_pos)
        self.dof_pos_minus_default_history = np.zeros((self.steps, self.state_processor.num_dof))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.dof_pos_minus_default_history = np.roll(self.dof_pos_minus_default_history, 1, axis=0)
        self.dof_pos_minus_default_history[0, :] = self.state_processor.joint_pos - self.default_pos

    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.dof_pos_minus_default_history.reshape(-1) * scale

class dof_vel(Observation):
    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.state_processor.joint_vel * scale
    
class dof_vel_history(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.dof_vel_history = np.zeros((self.steps, self.state_processor.num_dof))
    
    def update(self, data: Dict[str, Any]) -> None: 
        self.dof_vel_history = np.roll(self.dof_vel_history, 1, axis=0)
        self.dof_vel_history[0, :] = self.state_processor.joint_vel
    
    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.dof_vel_history.reshape(-1) * scale

class prev_actions(Observation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_actions = np.zeros((1, self.env.num_actions))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.prev_actions[0, :] = data["action"]

    def compute(self, scale: float = 1.0) -> np.ndarray:
        if getattr(self.env, "policy_config", {}).get("history_update_after_compute", False):
            # History is read before appending this observation, but the
            # standalone last_action must already contain the previous tick's
            # action. Its update() cache still contains the action from t-2.
            return np.asarray(self.env.last_action).reshape(self.env.num_actions) * scale
        return self.prev_actions[0] * scale
    
class prev_actions_history(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps 
        self.prev_actions = np.zeros((self.steps, self.env.num_actions))

    def update(self, data: Dict[str, Any]) -> None:
        self.prev_actions = np.roll(self.prev_actions, 1, axis=0)
        self.prev_actions[0, :] = data["action"]

    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.prev_actions.reshape(-1) * scale


def _root_heading_xy_from_wxyz(root_quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(root_quat, dtype=np.float32).reshape(4)
    if not np.all(np.isfinite(quat)):
        raise RuntimeError("root quaternion contains non-finite values")
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-8:
        raise RuntimeError("root quaternion is degenerate")
    quat = quat / norm
    forward = quat_rotate_numpy(quat[None, :], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))[0]
    heading = np.asarray(forward[:2], dtype=np.float32)
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= 1.0e-6 or not np.isfinite(heading_norm):
        raise RuntimeError("root heading is degenerate")
    return heading / heading_norm


class heading(Observation):
    """BehaviorContext heading command for heading-enabled direct-depth Actors.

    The current Phase-2J/QH direct-depth model uses a zero-centered 2D heading
    observation: ``[1 - cos(error), sin(error)]``.  ``behavior_context`` uses a
    synchronized motion-relative heading sequence for packaged tracking
    contexts and otherwise falls back to the command-start world heading.
    Realtime contexts use the relative turn since this tracking start, with
    robot and reference anchors captured together on the first valid sample.
    """

    def __init__(
        self,
        mode: str = "fixed_initial",
        dimension: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mode = str(mode).strip().lower()
        if self.mode not in {"behavior_context", "fixed_initial", "constant_zero", "invalid_zero"}:
            raise ValueError(
                "heading mode must be behavior_context, fixed_initial, constant_zero, or invalid_zero"
            )
        self.dimension = int(dimension)
        if self.dimension not in {2, 3}:
            raise ValueError("heading dimension must be 2 or 3")
        self.tracking_yaw_gain = float(
            getattr(self.env, "policy_config", {}).get("tracking_heading_yaw_gain", 1.0)
        )
        if not np.isfinite(self.tracking_yaw_gain) or self.tracking_yaw_gain <= 0:
            raise ValueError("tracking_heading_yaw_gain must be finite and positive")
        self._target_heading_xy: np.ndarray | None = None
        self._relative_start_xy: np.ndarray | None = None
        self._tracking_start_key = None
        self._heading_diag_last = None
        self._heading_diag_next = 0.0

    def _diagnose_heading(self, current, target=None, relative=None) -> None:
        if not getattr(self.env, "policy_config", {}).get("teleop_heading_diagnostics", False):
            return
        valid = target is not None
        now = time.monotonic()
        if valid == self._heading_diag_last and now < self._heading_diag_next:
            return
        self._heading_diag_last = valid
        self._heading_diag_next = now + 0.1
        yaw = lambda xy: None if xy is None else round(float(np.rad2deg(np.arctan2(xy[1], xy[0]))), 3)
        error = None if target is None else float(np.rad2deg(np.arctan2(
            current[0]*target[1]-current[1]*target[0], np.dot(current,target))))
        logger.info("TELEOP_HEADING valid={} robot_yaw={} target_yaw={} reference_yaw={} error_deg={} gain={}",
                    valid, yaw(current), yaw(target), yaw(relative), error, self.tracking_yaw_gain)

    def reset(self) -> None:
        self._target_heading_xy = None
        self._relative_start_xy = None
        self._tracking_start_key = None

    def _zero(self) -> np.ndarray:
        return np.zeros(self.dimension, dtype=np.float32)

    def compute(self, scale: float = 1.0) -> np.ndarray:
        if self.mode in {"constant_zero", "invalid_zero"}:
            return self._zero() * float(scale)

        current = _root_heading_xy_from_wxyz(self.state_processor.root_quat_b)
        # Waiting/first-frame hold must not pull yaw back to the previous
        # playback anchor. A fresh anchor is captured on the first playing tick.
        if not bool(getattr(self.env, "start_motion", False)):
            self.reset()
            return self._zero() * float(scale)
        realtime = getattr(self.env, "ctx_source", None) == "zmq"
        start_key = getattr(self.env, "ctx_started_monotonic", None)
        if realtime:
            start_key = (start_key, getattr(self.env, 'ctx_latest_reference_epoch', None))
            reset_after = getattr(self.env, 'teleop_heading_reset_after', 0.)
            source_time = getattr(self.env, 'ctx_latest_source_monotonic', None)
            if reset_after and (source_time is None or source_time <= reset_after):
                self.reset()
                return self._zero() * float(scale)
        if start_key != self._tracking_start_key:
            self.reset()
            self._tracking_start_key = start_key
        if self._target_heading_xy is None:
            self._target_heading_xy = current.copy()

        target = np.asarray(self._target_heading_xy, dtype=np.float32)
        if (
            self.mode == "behavior_context"
            and getattr(self.env, "task_type", None) == "tracking"
            and bool(getattr(self.env, "tracking_heading_context_available", False))
        ):
            if not bool(getattr(self.env, "tracking_heading_context_valid", False)):
                if realtime:
                    # Paused retarget publishes an invalid heading while its
                    # zero-velocity standing reference continues producing z.
                    # The next valid sample anchors robot and human together.
                    self.reset()
                    self._diagnose_heading(current)
                return self._zero() * float(scale)
            relative = np.asarray(
                getattr(self.env, "tracking_heading_relative_xy", None),
                dtype=np.float32,
            ).reshape(2)
            relative_norm = float(np.linalg.norm(relative))
            if not np.isfinite(relative_norm) or relative_norm <= 1.0e-6:
                raise RuntimeError("tracking relative heading is degenerate")
            relative = relative / relative_norm
            if realtime:
                # The producer accumulates yaw across policy stops and X pauses.
                # Anchor both frames together on the first valid command of
                # each tracking start; file contexts retain their authored yaw.
                if self._relative_start_xy is None:
                    self._relative_start_xy = relative.copy()
                    self._target_heading_xy = current.copy()
                start = self._relative_start_xy
                relative = np.array(
                    [np.dot(start, relative),
                     start[0] * relative[1] - start[1] * relative[0]],
                    dtype=np.float32,
                )
                target = self._target_heading_xy
            anchor = target
            target = np.array(
                [
                    relative[0] * anchor[0] - relative[1] * anchor[1],
                    relative[1] * anchor[0] + relative[0] * anchor[1],
                ],
                dtype=np.float32,
            )
        if realtime:
            self._diagnose_heading(current, target, getattr(self.env, "tracking_heading_relative_xy", None))
        cos = float(np.clip(np.dot(current, target), -1.0, 1.0))
        if cos > 1.0 - 1.0e-6:
            cos = 1.0
        sin = float(current[0] * target[1] - current[1] * target[0])
        if getattr(self.env, "task_type", None) == "tracking" and self.tracking_yaw_gain != 1.0:
            # Amplify the signed shortest yaw error, not the encoded vector.
            # Saturate at 90 degrees to avoid wrapping or weakening sin(error)
            # for larger errors after amplification.
            yaw_error = np.arctan2(sin, cos)
            amplified = np.clip(self.tracking_yaw_gain * yaw_error, -np.pi / 2, np.pi / 2)
            cos, sin = float(np.cos(amplified)), float(np.sin(amplified))
        if self.dimension == 2:
            value = np.array([1.0 - cos, sin], dtype=np.float32)
        else:
            value = np.array([cos, sin, 1.0], dtype=np.float32)
        return value * float(scale)


class depth_image(Observation):
    """Direct policy depth input: 37 control ticks sampled into 8 frames.

    A 30/60 Hz camera frame may be reused by more than one 50 Hz policy tick.
    Reuse still advances the ring because its indices are control-time ages,
    not camera-frame ages.  This preserves the trained 0.7 s temporal span.
    """

    def __init__(
        self,
        receive_timeout_ms: float = 100.0,
        capture_timeout_ms: float = 100.0,
        max_depth_m: float = 2.5,
        vertical_flip: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if receive_timeout_ms <= 0.0 or capture_timeout_ms <= 0.0:
            raise ValueError("direct-depth freshness timeouts must be positive")
        self.receive_timeout_s = float(receive_timeout_ms) / 1000.0
        self.capture_timeout_s = float(capture_timeout_ms) / 1000.0
        self.history = DirectDepthHistory(
            DirectDepthConfig(
                max_depth_m=float(max_depth_m),
                vertical_flip=vertical_flip,
            )
        )
        self._last_reset_key: tuple[int, float] | None = None
        self._last_frame_key: tuple[int, float] | None = None
        if not hasattr(self.state_processor, "terrain_depth_socket"):
            raise RuntimeError("depth_image requires TERRAIN_DEPTH_HOST/PORT")

    def reset(self) -> None:
        self.history.reset()
        self._last_reset_key = None
        self._last_frame_key = None

    def compute(self, scale: float = 1.0) -> np.ndarray:
        frame = self.state_processor.get_latest_terrain_depth(
            max_receive_age_s=self.receive_timeout_s,
            max_capture_age_s=self.capture_timeout_s,
        )
        if frame.depth_z.shape != (36, 64):
            raise RuntimeError(
                f"direct-depth producer must publish 36x64, got {frame.depth_z.shape}"
            )
        reset_key = (int(frame.frame_number), float(frame.capture_monotonic_s))
        reset = bool(frame.reset_history) and reset_key != self._last_reset_key
        if reset:
            self._last_reset_key = reset_key
        started = time.perf_counter()
        history = self.history.append(frame.depth_z, reset=reset)
        self.env.latest_depth_preprocess_s = time.perf_counter() - started
        if reset_key != self._last_frame_key:
            self.env.latest_depth_consume_monotonic_s = time.monotonic()
            self._last_frame_key = reset_key
        self.env.latest_depth_frame = frame
        # The exported encoder consumes normalized floating point. ObsGroup
        # concatenation promotes this to the surrounding vector dtype;
        # setup_policy casts it to the engine's declared binding dtype.
        return history.reshape(-1).astype(np.float32) * (float(scale) / 255.0)


class depth_latent(Observation):
    """Direct policy depth latent produced by a separate 50 Hz encoder worker."""

    def __init__(
        self,
        receive_timeout_ms: float = 100.0,
        capture_timeout_ms: float = 120.0,
        result_timeout_ms: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if (
            receive_timeout_ms <= 0.0
            or capture_timeout_ms <= 0.0
            or result_timeout_ms <= 0.0
        ):
            raise ValueError("direct-depth latent freshness timeouts must be positive")
        self.receive_timeout_s = float(receive_timeout_ms) / 1000.0
        self.capture_timeout_s = float(capture_timeout_ms) / 1000.0
        self.result_timeout_s = float(result_timeout_ms) / 1000.0
        self._last_result_key: tuple[int, int] | None = None
        if not hasattr(self.state_processor, "direct_depth_latent_socket"):
            raise RuntimeError("depth_latent requires DIRECT_DEPTH_LATENT_HOST/PORT")

    def compute(self, scale: float = 1.0) -> np.ndarray:
        result = self.state_processor.get_latest_direct_depth_latent(
            max_receive_age_s=self.receive_timeout_s,
            max_capture_age_s=self.capture_timeout_s,
            max_result_age_s=self.result_timeout_s,
        )
        latent = np.asarray(result.depth_latent, dtype=np.float32).reshape(-1)
        if latent.shape != (256,) or not np.all(np.isfinite(latent)):
            raise RuntimeError(f"direct-depth latent has invalid shape/value: {latent.shape}")

        result_key = (int(result.reset_generation), int(result.worker_sequence))
        if result_key != self._last_result_key:
            self.env.latest_depth_latent_consume_monotonic_s = time.monotonic()
            self._last_result_key = result_key
        self.env.latest_depth_latent_result = result
        self.env.latest_depth_frame = result
        self.env.latest_depth_preprocess_s = float(result.preprocess_s)
        self.env.last_depth_encoder_s = float(result.encoder_s)
        return latent * float(scale)
