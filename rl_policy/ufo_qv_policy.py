from __future__ import annotations

import os
import signal
import sys
import time
import numpy as np
from typing import Dict, Any, Type
try:
    from termcolor import colored
except ImportError:
    # Color is cosmetic and must not block the headless onboard dry-run.
    def colored(text, *_args, **_kwargs):
        return text
from sshkeyboard import listen_keyboard
from pathlib import Path
import csv
import gc
import hashlib
import json


sys.path.append(".")


from loguru import logger

# Re-use utilities from the existing sim2real package
from utils.strings import resolve_matching_names_values
from utils.onnx_module import Timer, create_inference_session
from utils.config import load_yaml_config
from utils.manual_goal import ManualGoalSequence
from utils.qv_runtime import VelocityRuntime
from utils.reward_goal_runtime import RewardGoalVelocityRuntime, CONTRACT as REWARD_GOAL_CONTRACT
from utils.latency_stats import LatencyStats
from utils.phase_schedule import next_phase_tick, advance_phase_tick
from utils.stand_fallback import StandFallback
from rl_policy.observations.ufo_policy import _root_heading_xy_from_wxyz
from utils.command_timeout_contract import validate_command_timeout_contract
from utils.teleop_velocity import VelocityContextMessage as RealtimeContextMessage, command_from_heading_error
from rl_policy.observations import Observation, ObsGroup
from rl_policy.utils.state_processor import StateProcessor, validate_state_source
from rl_policy.utils.command_sender import CommandSender


def _default_realtime_z(norm_ref: float) -> np.ndarray:
    z = np.zeros(256, dtype=np.float32)
    z[0] = float(norm_ref)
    return z


def _tracking_frame_from_elapsed(
    elapsed_s: float,
    playback_hz: float,
    start: int,
    end: int,
    context_length: int,
    loop: bool = True,
) -> int:
    """Select the latest context frame without accumulating loop delay."""
    if not np.isfinite(elapsed_s) or elapsed_s < 0.0:
        elapsed_s = 0.0
    if not np.isfinite(playback_hz) or playback_hz <= 0.0:
        raise ValueError("playback_hz must be finite and positive")
    bounded_start = max(0, int(start))
    bounded_end = min(int(end), int(context_length))
    if bounded_end <= bounded_start:
        raise ValueError("tracking playback range is empty")
    span = bounded_end - bounded_start
    elapsed_frames = int(np.floor(float(elapsed_s) * float(playback_hz)))
    if not loop and elapsed_frames >= span:
        return bounded_end
    return bounded_start + elapsed_frames % span


def _tracking_playback_options(config, source, context_dt, control_dt):
    is_file = source == "npz"
    loop = config.get("ctx_playback_loop", not is_file)
    if not isinstance(loop, bool):
        raise ValueError("ctx_playback_loop must be a boolean")
    behavior = config.get("ctx_end_behavior", "hold_first" if is_file and not loop else "stop")
    if behavior not in ("stop", "hold_first"):
        raise ValueError("ctx_end_behavior must be stop or hold_first")
    if behavior == "hold_first" and (not is_file or loop):
        raise ValueError("hold_first requires single-pass file tracking")
    hz = config.get("ctx_playback_hz")
    if hz is None and is_file:
        dt = float(context_dt if context_dt is not None else control_dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("tracking dt must be finite and positive")
        hz = 1.0 / dt
    if hz is not None:
        hz = float(hz)
        if not np.isfinite(hz) or hz <= 0 or not is_file:
            raise ValueError("ctx_playback_hz must be positive and only used with files")
    return loop, behavior, hz


def _require_real_actuation_authorization(
    robot_config: Dict[str, Any], *, dry_run: bool
) -> None:
    """Require explicit operator opt-in before constructing a G1 interface."""
    if dry_run:
        return
    if os.environ.get("UFO_REAL_ROBOT_OK") != "1":
        raise RuntimeError(
            "Refusing real robot actuation without both --nodryrun and UFO_REAL_ROBOT_OK=1."
        )
    try:
        validate_command_timeout_contract(robot_config)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing real robot actuation: invalid command-timeout contract: {exc}"
        ) from exc
    if not bool(robot_config.get("COMMAND_TIMEOUT_VERIFIED", False)):
        logger.warning(
            "COMMAND_TIMEOUT_VERIFIED=false: starting operator-supervised real "
            "mode without a certified firmware low-command timeout. R2 only "
            "works while this policy process is healthy; keep the physical "
            "e-stop operator and mechanical protection ready."
        )


def _inference_backend(policy_config: Dict[str, Any], component: str) -> str:
    backend = str(
        policy_config.get(
            f"{component}_inference_backend",
            policy_config.get("inference_backend", "onnx"),
        )
    ).strip().lower()
    if backend not in {"onnx", "tensorrt"}:
        raise ValueError(f"Unsupported {component} inference backend: {backend!r}")
    return backend


def _direct_depth_observation_mode(policy_config: Dict[str, Any]) -> str:
    policy_obs = policy_config.get("observation", {}).get("policy", {})
    if not isinstance(policy_obs, dict):
        return "image"
    has_depth_image = "depth_image" in policy_obs
    has_depth_latent = "depth_latent" in policy_obs
    if has_depth_image and has_depth_latent:
        raise ValueError(
            "direct-depth policy config must choose either depth_image or depth_latent, not both"
        )
    if has_depth_latent:
        return "latent"
    return "image"


def _model_root_artifact(model_path: str | Path, configured_path: str) -> Path:
    path = Path(str(configured_path)).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    model_root = Path(model_path).expanduser().resolve(strict=False).parent.parent
    return (model_root / path).resolve(strict=False)


def _verify_optional_sha256(
    path: Path, expected: Any, *, label: str
) -> None:
    if expected in {None, ""}:
        return
    normalized = str(expected).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{label} SHA256 must contain 64 lowercase hexadecimal characters")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != normalized:
        raise RuntimeError(f"{label} SHA256 mismatch: {actual} != {normalized}")


def _validate_inference_artifacts(
    model_path: str | Path, policy_config: Dict[str, Any]
) -> None:
    """Fail before constructing any command-capable real-robot interface."""
    actor_onnx = Path(model_path).expanduser().resolve(strict=False)
    if not actor_onnx.is_file():
        raise FileNotFoundError(f"Direct-depth Actor ONNX not found: {actor_onnx}")
    configured_encoder = policy_config.get("direct_depth_encoder_model_path")
    if not configured_encoder:
        raise ValueError("direct_depth_encoder_model_path is required")
    encoder_onnx = Path(str(configured_encoder)).expanduser()
    if not encoder_onnx.is_absolute():
        encoder_onnx = actor_onnx.parent / encoder_onnx
    if not encoder_onnx.is_file():
        raise FileNotFoundError(f"Direct-depth encoder ONNX not found: {encoder_onnx}")
    if _inference_backend(policy_config, "actor") == "tensorrt":
        configured = policy_config.get("tensorrt_actor_engine_path")
        if not configured:
            raise ValueError(
                "tensorrt_actor_engine_path is required when Actor inference uses TensorRT"
            )
        engine = _model_root_artifact(actor_onnx, str(configured))
        if not engine.is_file():
            raise FileNotFoundError(f"Actor TensorRT engine not found: {engine}")
        _verify_optional_sha256(
            engine,
            policy_config.get("tensorrt_actor_engine_sha256"),
            label="Actor TensorRT engine",
        )
    if _inference_backend(policy_config, "depth") == "tensorrt":
        configured = policy_config.get("tensorrt_depth_encoder_engine_path")
        if not configured:
            raise ValueError(
                "tensorrt_depth_encoder_engine_path is required when depth inference uses TensorRT"
            )
        engine = _model_root_artifact(actor_onnx, str(configured))
        if not engine.is_file():
            raise FileNotFoundError(f"Direct-depth encoder TensorRT engine not found: {engine}")
        _verify_optional_sha256(
            engine,
            policy_config.get("tensorrt_depth_encoder_engine_sha256"),
            label="Direct-depth encoder TensorRT engine",
        )


def _resolve_context_path(exp_config: Dict[str, Any]) -> Path:
    path = Path(str(exp_config["ctx_path"])).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve(strict=True)


def _validate_tracking_z(arr: np.ndarray, path: Path) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim < 2 or arr.shape[-1] <= 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"tracking context has invalid shape/value: {arr.shape} ({path})")
    return arr


def _load_tracking_context_bundle(path: Path) -> Dict[str, Any]:
    """Load the synchronized NPZ context without executable pickle payloads."""
    if path.suffix.lower() != ".npz":
        raise ValueError("File contexts must use .npz")
    with np.load(path, allow_pickle=False) as payload:
        if "z" not in payload:
            raise ValueError(f"tracking context {path} is missing z")
        z = _validate_tracking_z(payload["z"], path)
        heading_relative = (
            payload["heading_relative_xy"]
            if "heading_relative_xy" in payload.files
            else None
        )
        heading_valid = (
            payload["heading_valid"] if "heading_valid" in payload.files else None
        )
        dt_value = payload["dt"] if "dt" in payload.files else None
    if (heading_relative is None) != (heading_valid is None):
        raise ValueError(
            f"tracking context {path} must provide heading_relative_xy and heading_valid together"
        )
    if heading_relative is not None:
        heading_relative = np.asarray(heading_relative, dtype=np.float32)
        heading_valid = np.asarray(heading_valid, dtype=np.bool_).reshape(-1, 1)
        if heading_relative.shape != (z.shape[0], 2):
            raise ValueError(
                f"tracking heading shape {heading_relative.shape} does not match z length {z.shape[0]}"
            )
        if heading_valid.shape != (z.shape[0], 1):
            raise ValueError(
                f"tracking heading_valid shape {heading_valid.shape} does not match z length {z.shape[0]}"
            )
        norms = np.linalg.norm(heading_relative, axis=-1)
        if not np.all(np.isfinite(heading_relative)) or not np.allclose(norms, 1.0, atol=1.0e-4):
            raise ValueError("tracking heading_relative_xy must contain finite unit vectors")
    dt = None if dt_value is None else float(np.asarray(dt_value).reshape(()))
    if dt is not None and (not np.isfinite(dt) or dt <= 0.0):
        raise ValueError(f"tracking context dt must be finite and positive, got {dt}")
    return {
        "z": z,
        "heading_relative_xy": heading_relative,
        "heading_valid": heading_valid,
        "dt": dt,
    }



# -------------------------------------------------------------------------------------------------
# High-level RL policy that plugs into the existing framework
# -------------------------------------------------------------------------------------------------
class UFODeployPolicy:
    def __init__(
        self,
        robot_config: Dict[str, Any],
        policy_config: Dict[str, Any],
        exp_config: Dict[str, Any],
        model_path: str,
        rl_rate: int = 50,
        pico_control: bool = False,
        pico_control_addr: str = "tcp://127.0.0.1:28704",
        pico_only_control: bool = False,
        pico_control_source_timeout_ms: float = 250.0,
        dry_run: bool = True,
    ) -> None:
        if exp_config.get("type") != "tracking":
            raise ValueError("Tasks must use type: tracking; reward/goal presets are authored NPZ contexts")
        self.ctx_source = str(exp_config.get("ctx_source", "npz")).lower()
        if self.ctx_source not in {"npz", "zmq"}:
            raise ValueError("ctx_source must be npz or zmq")
        validate_state_source(robot_config)
        robot_config = dict(robot_config)
        self.model_path = str(Path(model_path).expanduser().resolve(strict=False))
        _validate_inference_artifacts(self.model_path, policy_config)
        self.robot_type = str(robot_config["ROBOT_TYPE"])
        self.dry_run = bool(dry_run)
        if (
            bool(policy_config.get("require_dry_run", False))
            and not self.dry_run
        ):
            raise RuntimeError(
                "This policy configuration is restricted to dry-run validation and "
                "cannot be launched with --nodryrun."
            )
        robot_config["_UFO_DRY_RUN"] = self.dry_run
        _require_real_actuation_authorization(robot_config, dry_run=self.dry_run)
        if not self.dry_run:
            if os.environ.get("UNITREE_SDK_LIB"):
                sys.path.append(os.environ["UNITREE_SDK_LIB"])
            import g1_interface
            if getattr(g1_interface, "COMMAND_WATCHDOG_TIMEOUT_MS", None) != 100:
                raise RuntimeError("Rebuild native/g1_interface: the 100 ms command watchdog is required")
            network_interface = robot_config.get("INTERFACE", None)
            self.robot = g1_interface.G1Interface(network_interface)
            try:
                self.robot.set_control_mode(g1_interface.ControlMode.PR)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to set G1 control mode to PR; refusing to start real robot control."
                ) from exc
            robot_config["robot"] = self.robot
        else:
            logger.warning(
                "G1 policy is in dry-run mode: no G1Interface, PR mode, or motor command publisher is created"
            )
        direct_depth_mode = _direct_depth_observation_mode(policy_config)
        # The external CPU encoder publishes timestamped depth latents.
        self.state_processor = StateProcessor(
            robot_config,
            policy_config["isaac_joint_names"],
            subscribe_terrain_depth=(direct_depth_mode == "image"),
            subscribe_direct_depth_latent=(direct_depth_mode == "latent"),
        )
        self.command_sender = CommandSender(robot_config, policy_config)

        self.rl_dt = 1.0 / rl_rate
        self.t = 0

        self.policy_config = policy_config
        self.rl_overrun_warning_factor = float(
            policy_config.get("rl_overrun_warning_factor", 1.0)
        )
        if (
            not np.isfinite(self.rl_overrun_warning_factor)
            or self.rl_overrun_warning_factor <= 0.0
        ):
            raise ValueError("rl_overrun_warning_factor must be finite and positive")
        self.timing_trace_path = os.environ.get("UFO_TIMING_TRACE_PATH")
        self.timing_trace_arm_file = os.environ.get("UFO_TIMING_TRACE_ARM_FILE")
        self.policy_ready_file = os.environ.get("UFO_POLICY_READY_FILE")
        self.timing_trace_records: list[tuple[Any, ...]] = []
        trace_duration_raw = os.environ.get("UFO_TIMING_TRACE_DURATION_S")
        self.timing_trace_duration_s = (
            float(trace_duration_raw) if trace_duration_raw is not None else None
        )
        if self.timing_trace_duration_s is not None and (
            not np.isfinite(self.timing_trace_duration_s)
            or self.timing_trace_duration_s <= 0.0
        ):
            raise ValueError("UFO_TIMING_TRACE_DURATION_S must be finite and positive")
        self.timing_trace_started_monotonic: float | None = None
        self._timing_trace_gpu_frequency_path = Path(
            "/sys/devices/platform/17000000.ga10b/devfreq/17000000.ga10b/cur_freq"
        )
        self._timing_trace_cpu_temperature_path = Path(
            "/sys/class/thermal/thermal_zone0/temp"
        )
        self._timing_trace_gpu_temperature_path = Path(
            "/sys/class/thermal/thermal_zone1/temp"
        )

        self.setup_policy(self.model_path)
        self.obs_cfg = policy_config["observation"]

        self.isaac_joint_names = policy_config["isaac_joint_names"]
        self.num_dofs = len(self.isaac_joint_names)

        default_joint_pos_dict = policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            self.isaac_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_dof_angles = np.zeros(len(self.isaac_joint_names))
        self.last_action  = np.zeros(len(self.isaac_joint_names))
        self.default_dof_angles[joint_indices] = default_joint_pos

        action_scale_cfg = policy_config["action_scale"]
        self.action_scale = np.ones((self.num_dofs))
        self.action_rescale = policy_config["action_rescale"]
        if isinstance(action_scale_cfg, float):
            self.action_scale *= action_scale_cfg
        elif isinstance(action_scale_cfg, dict):
            joint_ids, joint_names, action_scales = resolve_matching_names_values(
                action_scale_cfg, self.isaac_joint_names, preserve_order=True
            )
            self.action_scale[joint_ids] = action_scales
        else:
            raise ValueError(f"Invalid action scale type: {type(action_scale_cfg)}")

        self.policy_joint_names = policy_config["policy_joint_names"]
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [
            self.isaac_joint_names.index(name)
            for name in self.policy_joint_names
        ]

        # Keypress/control state
        self.use_policy_action = False
        self.pico_control_enabled = bool(pico_control)
        self.pico_only_control = bool(pico_only_control)
        if self.pico_only_control and not self.pico_control_enabled:
            raise ValueError("pico_only_control requires pico_control")
        self.pico_control_addr = str(pico_control_addr)
        self.pico_control_source_timeout_s = (
            float(pico_control_source_timeout_ms) / 1000.0
        )
        if (
            not np.isfinite(self.pico_control_source_timeout_s)
            or self.pico_control_source_timeout_s <= 0.0
        ):
            raise ValueError("pico_control_source_timeout_ms must be finite and positive")
        self.pico_control_sock = None
        self.last_pico_buttons: Dict[str, bool] = {}
        self.pico_control_have_valid_source = False
        self.pico_control_last_receive_monotonic: float | None = None
        self.pico_control_last_source_monotonic: float | None = None
        self.pico_control_future_tolerance_s = 0.010
        self.stop_latched = False
        self.pico_enable_released_after_stop = True
        self.stop_latch_enable_released_after_stop = True
        self._last_safe_stop_warning = 0.0
        self._last_invalid_z_warning = 0.0
        self._last_z_timeout_warning = 0.0
        self._last_stop_latch_warning = 0.0
        self._last_low_state_warning = 0.0
        self._exit_requested = False
        self._exit_signal_name: str | None = None
        self.hard_fault_latched = False

        self.first_time_init = True
        self.init_count = 0
        self.get_ready_state = False
        self.has_valid_low_state = False
        self.low_state_start_monotonic = time.monotonic()
        self.low_state_startup_timeout_s = float(
            robot_config.get("LOW_STATE_STARTUP_TIMEOUT_MS", 2000.0)
        ) / 1000.0
        if (
            not np.isfinite(self.low_state_startup_timeout_s)
            or self.low_state_startup_timeout_s <= 0.0
        ):
            raise ValueError("LOW_STATE_STARTUP_TIMEOUT_MS must be finite and positive")

        # Joint limits
        joint_indices, joint_names, joint_pos_lower_limit = (
            resolve_matching_names_values(
                robot_config["joint_pos_lower_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_lower_limit = np.zeros(self.num_dofs)
        self.joint_pos_lower_limit[joint_indices] = joint_pos_lower_limit

        joint_indices, joint_names, joint_pos_upper_limit = (
            resolve_matching_names_values(
                robot_config["joint_pos_upper_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_upper_limit = np.zeros(self.num_dofs)
        self.joint_pos_upper_limit[joint_indices] = joint_pos_upper_limit

        joint_indices, joint_names, joint_velocity_limit = (
            resolve_matching_names_values(
                robot_config["joint_velocity_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_velocity_limit = np.full(self.num_dofs, np.inf)
        self.joint_velocity_limit[joint_indices] = joint_velocity_limit
        self.q_target_slew_safety_factor = float(
            policy_config.get(
                "q_target_slew_safety_factor",
                robot_config.get("Q_TARGET_SLEW_SAFETY_FACTOR", 0.5),
            )
        )
        if not np.isfinite(self.q_target_slew_safety_factor) or self.q_target_slew_safety_factor < 0.0:
            raise ValueError("q_target_slew_safety_factor must be a finite non-negative value")
        self.last_cmd_q: np.ndarray | None = None
        self.action_filter_alpha = float(policy_config.get("action_filter_alpha", 1.0))
        if not np.isfinite(self.action_filter_alpha) or not 0.0 < self.action_filter_alpha <= 1.0:
            raise ValueError("action_filter_alpha must be finite and in (0, 1]")
        self.filtered_policy_action = None
        logger.info(f"Action low-pass alpha={self.action_filter_alpha} (1.0 disables filtering)")

        # ------------------------------------------------------
        # Joystick / keyboard setup (mirrors base_policy logic)
        # ------------------------------------------------------
        if (
            robot_config.get("USE_JOYSTICK", False)
            and not self.dry_run
            and not self.pico_only_control
        ):
            print("Using joystick")
            self.use_joystick = True
            self.wc_msg = None  # type: ignore
            self.last_wc_msg = self.robot.read_wireless_controller()
            print("Wireless Controller Initialized")
        elif self.pico_only_control:
            print("Using Pico controller only (wireless controller input disabled)")
            self.use_joystick = False
        elif sys.stdin.isatty():
            import threading
            print("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(
                target=self.start_key_listener, daemon=True
            )
        else:
            self.use_joystick = False
            logger.warning(
                "stdin is not a TTY; interactive keyboard control is disabled"
            )

        if self.pico_control_enabled:
            self.setup_pico_control()

        # Setup observations after all processors are ready
        self.setup_observations()

        # Initialize variables
        self.exp_config = exp_config
        self.task_type = exp_config['type']
        self.start_motion = False
        self.manual_goals = None
        self.file_standing_idle = bool(exp_config.get('file_standing_idle', False))
        if self.file_standing_idle:
            if self.task_type != 'tracking' or exp_config.get('context_contract') != REWARD_GOAL_CONTRACT or exp_config.get('manual_goal_selection', False):
                raise ValueError('Standing idle requires non-manual reward/goal file context')
            self.file_standing_z = np.asarray(np.load(exp_config['goal_standing_z_path'], allow_pickle=False), dtype=np.float32).reshape(-1)
            if self.file_standing_z.shape != (256,) or not np.isfinite(self.file_standing_z).all():
                raise ValueError('Invalid file-context standing latent')
        if exp_config.get('manual_goal_selection', False):
            if self.task_type != 'tracking' or exp_config.get('context_contract') != REWARD_GOAL_CONTRACT:
                raise ValueError('Manual goals require explicit reward/goal file context')
            self.manual_goals = ManualGoalSequence(exp_config['goal_schedule_path'], exp_config['ctx_path'])
            self.manual_goal_stand_z = np.asarray(np.load(exp_config['goal_standing_z_path'], allow_pickle=False), dtype=np.float32).reshape(-1)
            if self.manual_goal_stand_z.shape != (256,) or not np.isfinite(self.manual_goal_stand_z).all():
                raise ValueError('Invalid manual-goal standing latent')
            logger.info('Manual goals: R1=stand, B=goal1, X=next, Y=reset to standing, final goal holds; count={}', len(self.manual_goals.frames))
        logger.info(f"task_type={self.task_type}")

        # Task-specific setup
        if self.ctx_source == "npz":
            ctx_path = _resolve_context_path(exp_config)
            logger.info(f"Loading tracking context from {ctx_path}")
            exp_config["ctx_path"] = str(ctx_path)
            if ctx_path.suffix.lower() == '.npz':
                with np.load(ctx_path, allow_pickle=False) as data:
                    contract = str(data['context_contract']) if 'context_contract' in data.files else None
                    requested = exp_config.get('context_contract')
                    if contract == REWARD_GOAL_CONTRACT:
                        if requested != REWARD_GOAL_CONTRACT:
                            raise ValueError('Reward/goal NPZ requires explicit context_contract adapter')
                        self.qv_runtime = RewardGoalVelocityRuntime(data)
                    elif requested == REWARD_GOAL_CONTRACT:
                        raise ValueError('Task expects reward/goal NPZ but file contract differs')
            tracking_bundle = _load_tracking_context_bundle(ctx_path)
            self.ctx = tracking_bundle["z"]
            self.ctx_heading_relative_xy = tracking_bundle["heading_relative_xy"]
            self.ctx_heading_valid = tracking_bundle["heading_valid"]
            self.ctx_dt = tracking_bundle["dt"]
            self.ctx_zmq = None
            self.ctx_window = None
            self.ctx_norm_ref = float(np.linalg.norm(self.ctx[0]))
            if self.ctx_heading_relative_xy is not None:
                logger.info(
                    "Loaded synchronized tracking heading sequence: "
                    f"frames={self.ctx.shape[0]}, dt={self.ctx_dt}"
                )
        else:
            import zmq
            from collections import deque

            self.ctx = None
            self.ctx_heading_relative_xy = None
            self.ctx_heading_valid = None
            self.ctx_dt = None
            self.ctx_norm_ref = float(exp_config.get("ctx_norm_ref", 16.0))
            self.ctx_window = deque(maxlen=int(exp_config.get("window_size", 1)))
            self.ctx_zmq_timeout_s = max(
                0.0,
                float(exp_config.get("ctx_zmq_timeout_ms", 200.0)) / 1000.0,
            )
            self.ctx_zmq_source_timeout_s = max(
                0.0,
                float(exp_config.get("ctx_zmq_source_timeout_ms", 200.0)) / 1000.0,
            )
            self.ctx_zmq_result_timeout_s = max(
                0.0,
                float(exp_config.get("ctx_zmq_result_timeout_ms", 100.0)) / 1000.0,
            )
            self.ctx_zmq_future_tolerance_s = max(
                0.0,
                float(exp_config.get("ctx_zmq_future_tolerance_ms", 10.0)) / 1000.0,
            )
            self.ctx_zmq_start_monotonic = time.monotonic()
            self.ctx_last_z_monotonic: float | None = None
            self.ctx_latest_source_monotonic: float | None = None
            self.ctx_latest_result_monotonic: float | None = None
            self.ctx_latest_reference_live = False
            self.ctx_latest_heading_relative_xy = np.zeros(2, dtype=np.float32)
            self.ctx_latest_heading_valid = False
            self.ctx_latest_stream_id: int | None = None
            self.ctx_latest_sequence: int | None = None

            zmq_addr = str(exp_config.get("ctx_zmq_addr", "tcp://127.0.0.1:28711"))
            logger.info(
                f"Tracking context from ZMQ: {zmq_addr} "
                f"(structured z+heading, norm_ref={self.ctx_norm_ref}, "
                f"receive_timeout={self.ctx_zmq_timeout_s * 1000:.0f} ms, "
                f"source_timeout={self.ctx_zmq_source_timeout_s * 1000:.0f} ms, "
                f"result_timeout={self.ctx_zmq_result_timeout_s * 1000:.0f} ms)"
            )
            zctx = zmq.Context.instance()
            sock = zctx.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.setsockopt(zmq.CONFLATE, 1)
            sock.setsockopt(zmq.RCVTIMEO, 0)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(zmq_addr)
            self.ctx_zmq = sock
            self.ctx_latest = _default_realtime_z(self.ctx_norm_ref)

        logger.info(f"t_start={exp_config['start']}, t_end={exp_config['end']}, t_stop={exp_config['stop']}")
        self.t_start = exp_config['start']
        self.t_end = exp_config['end']
        self.t_stop = exp_config['stop']

        logger.info(f"gamma={exp_config['gamma']}, window_size={exp_config['window_size']}")
        self.gamma = exp_config['gamma']  # discount factor
        self.window_size = exp_config['window_size']  # context window size
        self.ctx_holding_first = False
        self.ctx_playback_loop, self.ctx_end_behavior, self.ctx_playback_hz = (
            _tracking_playback_options(exp_config, self.ctx_source, self.ctx_dt, self.rl_dt)
        )
        if self.ctx_playback_hz is not None:
            if (
                not np.isfinite(self.ctx_playback_hz)
                or self.ctx_playback_hz <= 0.0
            ):
                raise ValueError("ctx_playback_hz must be finite and positive")
            if self.ctx_source != "npz":
                raise ValueError("ctx_playback_hz is only supported for file tracking")
            _tracking_frame_from_elapsed(
                0.0,
                self.ctx_playback_hz,
                self.t_start,
                self.t_end,
                self.ctx.shape[0],
            )
            logger.info(
                f"Tracking context uses wall-clock playback at {self.ctx_playback_hz:.1f} Hz"
            )
        self.ctx_started_monotonic: float | None = None
        self.tracking_heading_context_available = (
            self.ctx_source == "zmq" or self.ctx_heading_relative_xy is not None
        )
        self.tracking_heading_relative_xy = None
        self.tracking_heading_context_valid = False

        if hasattr(self, "key_listener_thread"):
            self.key_listener_thread.start()

    def setup_policy(self, model_path):
        direct_depth_mode = _direct_depth_observation_mode(self.policy_config)
        actor_backend = _inference_backend(self.policy_config, "actor")
        actor_engine_path = None
        if actor_backend == "tensorrt":
            configured_engine = self.policy_config.get("tensorrt_actor_engine_path")
            if not configured_engine:
                raise ValueError(
                    "tensorrt_actor_engine_path is required when Actor inference uses TensorRT"
                )
            actor_engine_path = _model_root_artifact(model_path, str(configured_engine))
            if not actor_engine_path.is_file():
                raise FileNotFoundError(f"Actor TensorRT engine not found: {actor_engine_path}")
            logger.info(f"Loading direct-depth Actor TensorRT engine from {actor_engine_path}")
        else:
            logger.info(f"Loading direct-depth Actor ONNX from {model_path}")
        providers = self.policy_config.get("onnx_providers", ["CPUExecutionProvider"])
        if isinstance(providers, str):
            providers = [providers]
        if not isinstance(providers, list) or not all(isinstance(item, str) for item in providers):
            raise ValueError("policy_config['onnx_providers'] must be a string or list of strings")
        if len(providers) == 0:
            providers = ["CPUExecutionProvider"]
        if actor_backend == "onnx":
            logger.info(f"ONNX Runtime providers: {providers}")
        self.onnx_policy_session = create_inference_session(
            model_path,
            backend=actor_backend,
            engine_path=str(actor_engine_path) if actor_engine_path is not None else None,
            providers=providers,
            intra_op_num_threads=self.policy_config.get(
                "onnx_actor_intra_op_num_threads",
                self.policy_config.get("onnx_intra_op_num_threads"),
            ),
            inter_op_num_threads=int(self.policy_config.get("onnx_inter_op_num_threads", 1)),
            allow_spinning=bool(self.policy_config.get("onnx_allow_spinning", False)),
        )

        actor_inputs = {item.name: item for item in self.onnx_policy_session.get_inputs()}
        expected_actor_inputs_without_heading = {
            "state",
            "last_action",
            "history_actor",
            "depth_latent",
            "z",
        }
        expected_actor_inputs_with_heading = expected_actor_inputs_without_heading | {"heading", "velocity_command"}
        if set(actor_inputs) == expected_actor_inputs_without_heading:
            self.direct_depth_heading_dim = 0
        elif set(actor_inputs) == expected_actor_inputs_with_heading:
            heading_shape = list(actor_inputs["heading"].shape)
            if len(heading_shape) != 2 or heading_shape[0] != 1:
                raise RuntimeError(
                    f"direct-depth Actor heading input shape must be [1,D], got {heading_shape}"
                )
            self.direct_depth_heading_dim = int(heading_shape[1])
            if self.direct_depth_heading_dim not in {2, 3}:
                raise RuntimeError(
                    f"direct-depth Actor heading dim must be 2 or 3, got {self.direct_depth_heading_dim}"
                )
        else:
            raise RuntimeError(
                f"direct-depth Actor inputs mismatch: {sorted(actor_inputs)} != "
                f"{sorted(expected_actor_inputs_without_heading)} or "
                f"{sorted(expected_actor_inputs_with_heading)}"
            )
        actor_outputs = self.onnx_policy_session.get_outputs()
        if len(actor_outputs) != 1 or actor_outputs[0].name != "action":
            raise RuntimeError("direct-depth Actor must have one output named action")

        self.onnx_output_name = actor_outputs[0].name
        self.direct_depth_vector_dim = 64 + 29 + 372 + self.direct_depth_heading_dim
        self.direct_depth_image_dim = 8 * 36 * 32
        self.direct_depth_latent_dim = 256
        self.direct_depth_z_dim = 256

        def _numpy_dtype(node_arg):
            mapping = {
                "tensor(float16)": np.float16,
                "tensor(float)": np.float32,
                "tensor(float32)": np.float32,
            }
            try:
                return mapping[node_arg.type]
            except KeyError as exc:
                raise TypeError(
                    f"unsupported inference input dtype {node_arg.type!r} for {node_arg.name}"
                ) from exc

        if direct_depth_mode == "image":
            encoder_path = Path(
                str(self.policy_config["direct_depth_encoder_model_path"])
            ).expanduser()
            if not encoder_path.is_absolute():
                encoder_path = Path(model_path).expanduser().resolve().parent / encoder_path
            depth_backend = _inference_backend(self.policy_config, "depth")
            depth_engine_path = None
            if depth_backend == "tensorrt":
                configured_depth_engine = self.policy_config.get(
                    "tensorrt_depth_encoder_engine_path"
                )
                if not configured_depth_engine:
                    raise ValueError(
                        "tensorrt_depth_encoder_engine_path is required for TensorRT depth inference"
                    )
                depth_engine_path = _model_root_artifact(
                    model_path, str(configured_depth_engine)
                )
                logger.info(
                    f"Loading direct-depth CNN+GRU TensorRT engine from {depth_engine_path}"
                )
            else:
                logger.info(f"Loading direct-depth CNN+GRU ONNX from {encoder_path}")
            if depth_backend == "onnx":
                logger.info(f"ONNX Runtime providers: {providers}")
            self.depth_encoder_session = create_inference_session(
                str(encoder_path),
                backend=depth_backend,
                engine_path=str(depth_engine_path) if depth_engine_path is not None else None,
                providers=providers,
                intra_op_num_threads=self.policy_config.get(
                    "onnx_depth_intra_op_num_threads",
                    self.policy_config.get("onnx_intra_op_num_threads"),
                ),
                inter_op_num_threads=int(self.policy_config.get("onnx_inter_op_num_threads", 1)),
                allow_spinning=bool(self.policy_config.get("onnx_allow_spinning", False)),
            )
            encoder_inputs = self.depth_encoder_session.get_inputs()
            if len(encoder_inputs) != 1 or encoder_inputs[0].name != "depth_image":
                raise RuntimeError("direct-depth encoder must have one input named depth_image")
            encoder_outputs = self.depth_encoder_session.get_outputs()
            if len(encoder_outputs) != 1 or encoder_outputs[0].name != "depth_latent":
                raise RuntimeError("direct-depth encoder must have one output named depth_latent")
            self.depth_output_name = encoder_outputs[0].name
            encoder_input_dtype = _numpy_dtype(encoder_inputs[0])
        else:
            self.depth_encoder_session = None
            self.depth_output_name = "depth_latent"
            encoder_input_dtype = np.float32
            logger.info(
                "Using external direct-depth latent worker; policy loop runs Actor only"
            )

        self.qv_runtime = None
        actor_input_dtypes = {
            name: _numpy_dtype(node_arg) for name, node_arg in actor_inputs.items()
        }

        def policy_act(obs):
            flat = np.asarray(obs, dtype=np.float32)
            expected = (
                self.direct_depth_vector_dim
                + (
                    self.direct_depth_image_dim
                    if direct_depth_mode == "image"
                    else self.direct_depth_latent_dim
                )
                + self.direct_depth_z_dim
            )
            if flat.shape != (1, expected):
                raise ValueError(
                    f"direct-depth policy input must be [1, {expected}], got {flat.shape}"
                )
            cursor = 0
            state = flat[:, cursor : cursor + 64]
            cursor += 64
            last_action = flat[:, cursor : cursor + 29]
            cursor += 29
            history_actor = flat[:, cursor : cursor + 372]
            cursor += 372
            heading = None
            if self.direct_depth_heading_dim:
                heading = flat[:, cursor : cursor + self.direct_depth_heading_dim]
                cursor += self.direct_depth_heading_dim
            if direct_depth_mode == "image":
                depth_image = flat[
                    :, cursor : cursor + self.direct_depth_image_dim
                ].reshape(1, 8, 36, 32)
                cursor += self.direct_depth_image_dim
                depth_started = time.perf_counter()
                depth_latent = self.depth_encoder_session.run(
                    [self.depth_output_name],
                    {"depth_image": depth_image.astype(encoder_input_dtype, copy=False)},
                )[0]
                self.last_depth_encoder_s = time.perf_counter() - depth_started
                self.perf_dict["depth_encoder"] = self.perf_dict.get(
                    "depth_encoder", 0.0
                ) + self.last_depth_encoder_s
            else:
                depth_latent = flat[:, cursor : cursor + self.direct_depth_latent_dim]
                cursor += self.direct_depth_latent_dim
                self.last_depth_encoder_s = float(
                    getattr(self, "last_depth_encoder_s", float("nan"))
                )
            z = flat[:, cursor : cursor + self.direct_depth_z_dim]
            feed = {
                "state": state.astype(actor_input_dtypes["state"], copy=False),
                "last_action": last_action.astype(
                    actor_input_dtypes["last_action"], copy=False
                ),
                "history_actor": history_actor.astype(
                    actor_input_dtypes["history_actor"], copy=False
                ),
                "depth_latent": np.asarray(depth_latent).astype(
                    actor_input_dtypes["depth_latent"], copy=False
                ),
                "z": z.astype(actor_input_dtypes["z"], copy=False),
            }
            if self.direct_depth_heading_dim:
                assert heading is not None
                feed["heading"] = heading.astype(actor_input_dtypes["heading"], copy=False)
            if self.ctx_source == "zmq":
                if not self.policy_config.get("teleop_velocity_enabled", False):
                    raise RuntimeError("Realtime QV requires teleop_velocity_enabled")
                if float(self.policy_config.get("tracking_heading_yaw_gain", 1.0)) != 1.0:
                    raise RuntimeError("QV teleop requires heading yaw gain 1")
                active = self.use_policy_action and self.start_motion and not self.stop_latched
                valid = bool(getattr(self, "ctx_latest_velocity_valid", False)) and bool(getattr(self, "ctx_latest_heading_valid", False))
                feed["velocity_command"] = command_from_heading_error(
                    getattr(self, "ctx_latest_velocity_heading_xy", np.zeros(2)),
                    valid, heading, active)
                # Apply after heading rotation: constrain robot-frame lateral command.
                if self.policy_config.get("teleop_zero_lateral_velocity", False):
                    feed["velocity_command"][:, 1] = 0.0
                if time.monotonic() >= getattr(self, "_qv_teleop_next_log", 0.0):
                    logger.info("TELEOP_VELOCITY active={} valid={} command={} context_sequence={}",
                                active, valid, feed["velocity_command"].tolist(),
                                getattr(self, "ctx_latest_sequence", None))
                    self._qv_teleop_next_log = time.monotonic() + 1.0
            else:
                if self.qv_runtime is None:
                    if self.task_type != 'tracking' or self.ctx_source != 'npz':
                        raise RuntimeError('QV currently requires an authored tracking NPZ context')
                    with np.load(self.exp_config['ctx_path'], allow_pickle=False) as data:
                        velocities = data['reference_velocity_anchor'] if 'reference_velocity_anchor' in data.files else data['reference_velocity_heading0_xy']
                        validity = data['velocity_valid']
                    if len(velocities) != len(self.ctx):
                        raise ValueError('Velocity/z context length mismatch')
                    self.qv_runtime = VelocityRuntime(velocities, validity)
                root_heading = _root_heading_xy_from_wxyz(self.state_processor.root_quat_b)
                active = (self.use_policy_action and self.start_motion and not self.stop_latched)
                feed['velocity_command'] = self.qv_runtime.command(
                    root_heading, self.t, active, self.ctx_started_monotonic)
            actor_started = time.perf_counter()
            result = self.onnx_policy_session.run(
                [self.onnx_output_name], feed
            )[0]
            self.last_actor_network_s = time.perf_counter() - actor_started
            if not np.isfinite(result).all(): raise RuntimeError('Nonfinite QV action')
            return result
        self.policy = policy_act

    def setup_observations(self):
        """Setup observations for policy inference"""
        self.observations: Dict[str, ObsGroup] = {}
        self.reset_callbacks = []
        self.update_callbacks = []
        
        # Create observation instances based on config
        for obs_group, obs_items in self.obs_cfg.items():
            print(f"obs_group: {obs_group}")
            obs_funcs = {}
            for obs_name, obs_config in obs_items.items():
                obs_class: Type[Observation] = Observation.registry[obs_name]
                obs_func = obs_class(env=self, **obs_config)
                obs_funcs[obs_name] = obs_func
                self.reset_callbacks.append(obs_func.reset)
                self.update_callbacks.append(obs_func.update)
                print(f"\t{obs_name}: {obs_config}")
            self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)

    def reset(self):
        for reset_callback in self.reset_callbacks:
            reset_callback()

    def update(self):
        self.state_dict["action"] = self.last_action
        for update_callback in self.update_callbacks:
            update_callback(self.state_dict)

    def _prepare_tracking_heading_context(self) -> None:
        """Expose the heading command aligned with the z frame used this tick."""

        self.tracking_heading_relative_xy = None
        self.tracking_heading_context_valid = False
        holding_first = getattr(self, "ctx_holding_first", False) and self.use_policy_action and not self.stop_latched
        if self.task_type != "tracking" or not (self.start_motion or holding_first):
            return
        ctx_source = getattr(self, "ctx_source", None)
        if ctx_source == "zmq":
            self.tracking_heading_relative_xy = np.asarray(
                getattr(
                    self,
                    "ctx_latest_heading_relative_xy",
                    np.zeros(2, dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(2)
            self.tracking_heading_context_valid = bool(
                getattr(self, "ctx_latest_heading_valid", False)
            )
            return
        if ctx_source != "npz":
            return
        headings = getattr(self, "ctx_heading_relative_xy", None)
        validity = getattr(self, "ctx_heading_valid", None)
        if headings is None or validity is None:
            return
        frame = min(max(int(self.t), 0), int(headings.shape[0]) - 1)
        self.tracking_heading_relative_xy = np.asarray(headings[frame], dtype=np.float32)
        self.tracking_heading_context_valid = bool(validity[frame, 0])

    def _warn_throttled(self, attr_name: str, message: str, interval_s: float = 1.0) -> None:
        now = time.monotonic()
        last = float(getattr(self, attr_name, 0.0))
        if now - last >= interval_s:
            logger.warning(message)
            setattr(self, attr_name, now)

    def _reset_tracking_state_to_stop(self, reset_realtime_z: bool = False) -> None:
        self.ctx_holding_first = False
        self.start_motion = False
        self.ctx_started_monotonic = None
        self.tracking_heading_relative_xy = None
        self.tracking_heading_context_valid = False
        if hasattr(self, "t_stop"):
            self.t = self.t_stop
        if reset_realtime_z and getattr(self, "ctx_source", None) == "zmq":
            self.ctx_latest = _default_realtime_z(float(self.ctx_norm_ref))
            if self.ctx_window is not None:
                self.ctx_window.clear()
            self.ctx_latest_heading_relative_xy = np.zeros(2, dtype=np.float32)
            self.ctx_latest_heading_valid = False
            self.ctx_latest_reference_live = False
            self.ctx_latest_source_monotonic = None
            self.ctx_latest_result_monotonic = None

    def enter_input_fallback(self, reason, **kwargs):
        fallback = getattr(self, 'stand_fallback', None)
        if fallback is not None and fallback.enter(reason):
            return
        self.enter_safe_stop(reason, **kwargs)

    def enter_safe_stop(
        self,
        reason: str,
        *,
        reset_realtime_z: bool = False,
        warning_attr: str = "_last_safe_stop_warning",
    ) -> None:
        if getattr(self, 'stand_fallback', None) is not None:
            self.stand_fallback.active = False
        self.use_policy_action = False
        self.get_ready_state = False
        self.filtered_policy_action = None
        self._reset_tracking_state_to_stop(reset_realtime_z=reset_realtime_z)
        self.last_action = np.zeros_like(self.last_action)
        self._warn_throttled(warning_attr, f"Safe stop: {reason}")

    def enter_stop_latch(self, reason: str) -> None:
        self.stop_latched = True
        self.pico_enable_released_after_stop = False
        self.stop_latch_enable_released_after_stop = False
        self.enter_safe_stop(reason)

    def _close_command_interface(self) -> None:
        sender = getattr(self, "command_sender", None)
        if sender is not None:
            try:
                sender.close()
            except Exception as exc:
                self._warn_throttled(
                    "_last_safe_stop_warning",
                    f"Failed to close command sender: {exc}",
                )
        if hasattr(self, "robot"):
            self.robot = None
        gc.collect()

    def enter_hard_fault(self, reason: str) -> None:
        """Latch an unrecoverable state/transport fault and terminate control."""
        self.hard_fault_latched = True
        self.stop_latched = True
        self.enter_safe_stop(reason)
        self._exit_requested = True
        self._exit_signal_name = "hard fault"
        self._close_command_interface()

    def _handle_low_state_failure(self) -> None:
        reason = str(
            getattr(self.state_processor, "last_low_state_error", "low-state unavailable")
        )
        if not self.has_valid_low_state:
            startup_age_s = time.monotonic() - self.low_state_start_monotonic
            if startup_age_s <= self.low_state_startup_timeout_s:
                self._warn_throttled(
                    "_last_low_state_warning",
                    f"Waiting for fresh low-state: {reason}",
                )
                return
            reason = (
                f"low-state startup timeout after {startup_age_s * 1000.0:.0f} ms: {reason}"
            )
        self.enter_hard_fault(reason)

    def _request_exit(self, signum, _frame) -> None:
        self._exit_requested = True
        try:
            self._exit_signal_name = signal.Signals(signum).name
        except ValueError:
            self._exit_signal_name = str(signum)

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._request_exit)
            except (ValueError, RuntimeError):
                pass

    def _warn_stop_latch_ignored(self, source: str) -> None:
        self._warn_throttled(
            "_last_stop_latch_warning",
            f"{source} ignored while global stop latch is active; release enable/start inputs, then re-arm explicitly.",
        )

    def _clear_stop_latch_for_rearm(self, source: str) -> None:
        self.stop_latched = False
        self.pico_enable_released_after_stop = True
        self.stop_latch_enable_released_after_stop = True
        logger.info(f"{source}: global stop latch cleared")

    def _poll_realtime_z(self) -> bool:
        if getattr(self, "ctx_zmq", None) is None:
            return False

        import zmq

        received_valid_z = False
        while True:
            try:
                raw = self.ctx_zmq.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            try:
                message = RealtimeContextMessage.from_bytes(raw)
            except ValueError as exc:
                self._warn_throttled(
                    "_last_invalid_z_warning",
                    f"Ignoring invalid realtime context packet: {exc}",
                )
                continue

            now = time.monotonic()
            future_tolerance_s = float(
                getattr(self, "ctx_zmq_future_tolerance_s", 0.010)
            )
            if (
                message.source_receive_monotonic_s > now + future_tolerance_s
                or message.encoder_completion_monotonic_s > now + future_tolerance_s
                or message.publish_monotonic_s > now + future_tolerance_s
            ):
                self._warn_throttled(
                    "_last_invalid_z_warning",
                    "Ignoring invalid realtime context: timestamp is in the future",
                )
                continue

            result_timeout_s = float(
                getattr(self, "ctx_zmq_result_timeout_s", 0.100)
            )
            result_age_s = now - message.publish_monotonic_s
            if result_timeout_s > 0.0 and result_age_s > result_timeout_s:
                self._warn_throttled(
                    "_last_invalid_z_warning",
                    f"Ignoring stale realtime context result: {result_age_s * 1000.0:.1f} ms",
                )
                continue

            source_timeout_s = float(
                getattr(self, "ctx_zmq_source_timeout_s", 0.200)
            )
            source_age_s = now - message.source_receive_monotonic_s
            if (
                message.reference_live
                and source_timeout_s > 0.0
                and source_age_s > source_timeout_s
            ):
                self._warn_throttled(
                    "_last_invalid_z_warning",
                    f"Ignoring stale realtime pose source: {source_age_s * 1000.0:.1f} ms",
                )
                continue

            previous_stream_id = getattr(self, "ctx_latest_stream_id", None)
            previous_sequence = getattr(self, "ctx_latest_sequence", None)
            if previous_stream_id == message.stream_id:
                if previous_sequence is not None and message.sequence <= previous_sequence:
                    self._warn_throttled(
                        "_last_invalid_z_warning",
                        "Ignoring duplicate/out-of-order realtime context: "
                        f"sequence={message.sequence}, last={previous_sequence}",
                    )
                    continue
            elif previous_stream_id is not None:
                self.enter_input_fallback(
                    "realtime context producer restarted; operator re-arm required",
                    reset_realtime_z=True,
                    warning_attr="_last_z_timeout_warning",
                )

            self.ctx_input_message = message
            z = message.z

            self.ctx_latest_reference_epoch = message.reference_epoch
            self.ctx_latest_reference_sample_s = message.reference_sample_s
            self.ctx_latest = z.copy()
            self.ctx_latest_heading_relative_xy = message.heading_relative_xy.copy()
            self.ctx_latest_heading_valid = bool(message.heading_valid)
            self.ctx_latest_velocity_heading_xy = message.velocity_heading_xy.copy()
            self.ctx_latest_velocity_valid = bool(message.velocity_valid)
            self.ctx_latest_reference_live = bool(message.reference_live)
            self.ctx_latest_source_monotonic = message.source_receive_monotonic_s
            self.ctx_latest_result_monotonic = message.publish_monotonic_s
            self.ctx_latest_stream_id = message.stream_id
            self.ctx_latest_sequence = message.sequence
            self.ctx_last_z_monotonic = now
            received_valid_z = True

        return received_valid_z

    def _check_realtime_z_watchdog(self) -> None:
        timeout_s = float(getattr(self, "ctx_zmq_timeout_s", 0.0))
        if timeout_s <= 0.0:
            return

        now = time.monotonic()
        last_z_time = getattr(self, "ctx_last_z_monotonic", None)
        reference_time = (
            float(last_z_time)
            if last_z_time is not None
            else float(getattr(self, "ctx_zmq_start_monotonic", now))
        )
        age_s = now - reference_time
        if age_s > timeout_s:
            self.enter_input_fallback(
                f"no valid realtime context for {age_s * 1000:.0f} ms",
                reset_realtime_z=True,
                warning_attr="_last_z_timeout_warning",
            )
            return

        result_monotonic = getattr(self, "ctx_latest_result_monotonic", None)
        result_timeout_s = float(getattr(self, "ctx_zmq_result_timeout_s", 0.0))
        if (
            result_monotonic is not None
            and result_timeout_s > 0.0
            and now - float(result_monotonic) > result_timeout_s
        ):
            self.enter_input_fallback(
                f"realtime context result is stale for "
                f"{(now - float(result_monotonic)) * 1000.0:.0f} ms",
                reset_realtime_z=True,
                warning_attr="_last_z_timeout_warning",
            )
            return

        source_monotonic = getattr(self, "ctx_latest_source_monotonic", None)
        source_timeout_s = float(getattr(self, "ctx_zmq_source_timeout_s", 0.0))
        if (
            bool(getattr(self, "ctx_latest_reference_live", False))
            and source_monotonic is not None
            and source_timeout_s > 0.0
            and now - float(source_monotonic) > source_timeout_s
        ):
            self.enter_input_fallback(
                f"realtime pose source is stale for "
                f"{(now - float(source_monotonic)) * 1000.0:.0f} ms",
                reset_realtime_z=True,
                warning_attr="_last_z_timeout_warning",
            )

    def _apply_q_target_slew_limit(self, q_target: np.ndarray) -> np.ndarray:
        q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
        if self.last_cmd_q is None:
            baseline = np.asarray(self.state_processor.joint_pos, dtype=np.float64).reshape(-1)
            if baseline.shape != q_target.shape or not np.all(np.isfinite(baseline)):
                baseline = q_target.copy()
        else:
            baseline = np.asarray(self.last_cmd_q, dtype=np.float64).reshape(-1)

        max_delta = self.joint_velocity_limit * self.rl_dt * self.q_target_slew_safety_factor
        if max_delta.shape != q_target.shape:
            raise ValueError(
                f"q_target slew limit shape mismatch: {max_delta.shape} vs {q_target.shape}"
            )
        return baseline + np.clip(q_target - baseline, -max_delta, max_delta)

    def _send_safe_hold_command(self) -> bool:
        if not bool(getattr(self, "has_valid_low_state", False)):
            return False
        if not hasattr(self, "state_processor") or not hasattr(self.state_processor, "joint_pos"):
            return False

        hold_q = np.asarray(self.state_processor.joint_pos, dtype=np.float64).reshape(-1)
        if hold_q.shape[0] != self.num_dofs or not np.all(np.isfinite(hold_q)):
            return False

        hold_q = np.clip(hold_q, self.joint_pos_lower_limit, self.joint_pos_upper_limit)
        try:
            self.command_sender.send_command(
                hold_q,
                np.zeros(self.num_dofs),
                np.zeros(self.num_dofs),
            )
            self.last_cmd_q = hold_q.copy()
            return True
        except Exception as exc:
            self._warn_throttled(
                "_last_safe_stop_warning",
                f"Failed to send safe hold command: {exc}",
            )
            return False

    def _reset_file_standing(self) -> None:
        self.start_motion = False
        self.ctx_started_monotonic = None
        self.ctx_holding_first = False
        self.t = self.t_start
        self.tracking_heading_context_valid = False
        self.tracking_heading_relative_xy = None

    def _finish_tracking_clip(self) -> None:
        if getattr(self, 'file_standing_idle', False):
            self._reset_file_standing()
            logger.info('Reward playback complete: default standing reference')
            return
        if getattr(self, "ctx_end_behavior", "stop") == "hold_first":
            # Never re-enable a policy disabled by a stop or input fault.
            # Hold an exact first frame, independently of stop index/window size.
            self.start_motion = False
            self.ctx_started_monotonic = None
            self.t = self.t_start
            self.ctx_holding_first = self.use_policy_action and not self.stop_latched
            logger.info("Tracking clip complete: holding first-frame z with free heading; policy enable unchanged")
        else:
            self.enter_stop_latch("tracking clip complete; holding current posture")

    def prepare_obs_for_rl(self):
        """Prepare observation for policy inference using observation classes"""
        obs_dict: Dict[str, np.ndarray] = {}
        if self.ctx_source == "zmq":
            # Poll before observation construction so heading and z come
            # from the same structured context packet on this tick.
            self._poll_realtime_z()
            self._check_realtime_z_watchdog()
            if getattr(self, 'stand_fallback', None) is not None:
                self.stand_fallback.apply()
        if (
            self.ctx_source == "npz"
            and self.start_motion
            and self.manual_goals is None
            and self.ctx_playback_hz is not None
        ):
            started = self.ctx_started_monotonic
            if started is None:
                started = time.monotonic()
                self.ctx_started_monotonic = started
            self.t = _tracking_frame_from_elapsed(
                time.monotonic() - started,
                self.ctx_playback_hz,
                self.t_start,
                self.t_end,
                self.ctx.shape[0],
                loop=getattr(self, "ctx_playback_loop", True),
            )
            if self.t >= min(self.t_end, self.ctx.shape[0]):
                self._finish_tracking_clip()
        if self.manual_goals is not None:
            self.t = self.manual_goals.frame
        self._prepare_tracking_heading_context()
        history_update_after_compute = bool(
            self.policy_config.get("history_update_after_compute", False)
        )
        if not history_update_after_compute:
            self.update()
        
        for obs_group in self.observations.values():
            obs = obs_group.compute()
            obs_dict[obs_group.name] = obs[None, :].astype(np.float32)

        # PBFM training queries history_actor before inserting the current
        # observation.  Keep this opt-in so legacy deployment models retain
        # their original runtime behavior.
        if history_update_after_compute:
            self.update()
        
        obs = obs_dict[obs_group.name]

        if self.ctx_source == "npz":
            window_size = 1 if getattr(self, "ctx_holding_first", False) else self.window_size
            window = self.ctx[self.t:self.t + window_size]
            discounts = self.gamma ** np.arange(len(window))
            discounts = discounts / np.sum(discounts)
            discounted_avg = np.sum(window * discounts[:, np.newaxis], axis=0)
            discounted_avg = (
                discounted_avg
                / (np.linalg.norm(discounted_avg, axis=-1) + 1e-8)
                * float(np.linalg.norm(self.ctx[0]))
            )
        else:
            from collections import deque

            if self.ctx_window.maxlen != int(self.window_size):
                old = list(self.ctx_window)
                self.ctx_window = deque(old[-int(self.window_size):], maxlen=int(self.window_size))

            self.ctx_window.appendleft(self.ctx_latest)
            if len(self.ctx_window) > 0:
                window = np.stack(list(self.ctx_window), axis=0)
            else:
                window = self.ctx_latest[None, :]
            discounts = self.gamma ** np.arange(window.shape[0])
            discounts = discounts / np.sum(discounts)
            discounted_avg = np.sum(window * discounts[:, np.newaxis], axis=0)
            discounted_avg = (
                discounted_avg
                / (np.linalg.norm(discounted_avg) + 1e-8)
                * float(self.ctx_norm_ref)
            )

        if self.exp_config.get('context_contract') == REWARD_GOAL_CONTRACT:
            # Authored reward/goal latents must not be renormalized to frame 0.
            discounted_avg = (self.manual_goal_stand_z.copy()
                              if self.manual_goals is not None and not self.start_motion
                              else self.ctx[self.t].copy())
            if getattr(self, 'file_standing_idle', False) and not self.start_motion:
                discounted_avg = self.file_standing_z.copy()
        inputs = np.concatenate([obs, discounted_avg[np.newaxis, :]], axis=-1).astype(np.float32)
        
        if self.use_policy_action:
            if self.start_motion and self.t < self.t_end:
                if self.ctx_playback_hz is None:
                    self.t += 1
                    if self.ctx_source == "npz":
                        self.t = self.t % self.ctx.shape[0]
                if self.total_inference_cnt % 100 == 0:
                    logger.info(f"context_step={self.t}")
            else:
                self.t = self.t_start if getattr(self, "ctx_holding_first", False) else self.t_stop
                self.start_motion = False
                self.ctx_started_monotonic = None
        return obs_dict, inputs

    def get_init_target(self):
        if self.init_count > 500:
            self.init_count = 500

        # interpolate from current dof_pos to default angles
        dof_pos = self.state_processor.joint_pos
        progress = self.init_count / 500
        q_target = dof_pos + (self.default_dof_angles - dof_pos) * progress
        self.init_count += 1
        return q_target

    def run(self) -> int:
        self._install_signal_handlers()
        self._exit_requested = False
        self._exit_signal_name = None
        total_inference_cnt = 0
        state_dict = {}
        state_dict["action"] = np.zeros(self.num_actions)
        self.state_dict = state_dict
        self.total_inference_cnt = total_inference_cnt
        self.perf_dict = {}
        self.latency_stats = LatencyStats("policy")
        if self.policy_ready_file:
            ready_path = Path(self.policy_ready_file).expanduser()
            if not ready_path.parent.is_dir():
                raise RuntimeError(
                    f"Policy ready marker parent does not exist: {ready_path.parent}"
                )
            ready_path.write_text("ready\n", encoding="utf-8")
            logger.info(f"Policy initialization ready marker: {ready_path}")

        try:
            phase_lock = bool(self.policy_config.get("depth_policy_phase_lock", False))
            phase = float(self.policy_config.get("depth_policy_phase_ms", 11.0)) / 1000.0
            if phase_lock and abs(self.rl_dt - .02) > 1e-6:
                raise ValueError("Phase-locked policy requires 50 Hz")
            next_run_time = next_phase_tick(time.monotonic(), self.rl_dt, phase) if phase_lock else time.monotonic()
            
            while not self._exit_requested:
                sleep_s = next_run_time - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                self._schedule_lag_s = max(0.0, time.monotonic() - next_run_time)
                self._rl_step_scheduled()
                if (
                    self.timing_trace_started_monotonic is not None
                    and self.timing_trace_duration_s is not None
                    and time.monotonic() - self.timing_trace_started_monotonic
                    >= self.timing_trace_duration_s
                ):
                    self._exit_requested = True
                    self._exit_signal_name = "timing trace duration complete"
                
                if phase_lock:
                    next_run_time = advance_phase_tick(next_run_time, time.monotonic(), self.rl_dt, phase)
                else:
                    next_run_time = max(next_run_time + self.rl_dt, time.monotonic())
                self.total_inference_cnt += 1

                if self.total_inference_cnt % 100 == 0:
                    if bool(self.policy_config.get("log_performance", False)):
                        timing = ", ".join(
                            f"{name}={elapsed * 10.0:.2f}ms"
                            for name, elapsed in sorted(self.perf_dict.items())
                        )
                        logger.info(f"Policy timing mean over 100 ticks: {timing}")
                    self.perf_dict = {}
            if self._exit_requested:
                logger.info(f"{self._exit_signal_name or 'signal'} received, exiting policy loop.")
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt, exiting policy loop.")
        finally:
            self.enter_safe_stop("policy loop exiting")
            if not self.hard_fault_latched:
                self._send_safe_hold_command()
            self._close_command_interface()
            self._write_timing_trace()
        return 2 if self.hard_fault_latched else 0

    @staticmethod
    def _read_sysfs_float(path: Path, scale: float = 1.0) -> float:
        try:
            return float(path.read_text(encoding="utf-8").strip()) * float(scale)
        except (OSError, ValueError):
            return float("nan")

    def _append_timing_trace(
        self,
        *,
        tick: int,
        prepare_low_state_s: float,
        prepare_obs_s: float,
        actor_s: float,
        control_s: float,
        loop_start: float,
    ) -> float:
        if not self.timing_trace_path:
            return time.perf_counter() - loop_start
        if self.timing_trace_arm_file and not Path(
            self.timing_trace_arm_file
        ).expanduser().is_file():
            return time.perf_counter() - loop_start
        gpu_hz = self._read_sysfs_float(self._timing_trace_gpu_frequency_path)
        cpu_c = self._read_sysfs_float(self._timing_trace_cpu_temperature_path, 1.0e-3)
        gpu_c = self._read_sysfs_float(self._timing_trace_gpu_temperature_path, 1.0e-3)
        total_s = time.perf_counter() - loop_start
        trace_monotonic = time.monotonic()
        depth_frame = getattr(self, "latest_depth_frame", None)
        if depth_frame is None:
            capture_s = camera_receive_s = producer_publish_s = consume_s = float("nan")
            camera_delivery_ms = producer_processing_ms = float("nan")
            producer_to_policy_ms = capture_to_policy_ms = float("nan")
            capture_age_ms = receive_age_ms = float("nan")
            frame_number = -1
            reused_frame = False
        else:
            capture_s = float(depth_frame.capture_monotonic_s)
            camera_receive_s = float(depth_frame.camera_receive_monotonic_s)
            producer_publish_s = float(depth_frame.producer_publish_monotonic_s)
            consume_s = float(
                getattr(
                    self,
                    "latest_depth_latent_consume_monotonic_s",
                    getattr(self, "latest_depth_consume_monotonic_s", trace_monotonic),
                )
            )
            camera_delivery_ms = (camera_receive_s - capture_s) * 1000.0
            producer_processing_ms = (producer_publish_s - camera_receive_s) * 1000.0
            producer_to_policy_ms = (consume_s - producer_publish_s) * 1000.0
            capture_to_policy_ms = (consume_s - capture_s) * 1000.0
            capture_age_ms = (trace_monotonic - capture_s) * 1000.0
            received_s = getattr(
                self.state_processor,
                "latest_direct_depth_latent_monotonic",
                getattr(self.state_processor, "latest_terrain_depth_monotonic", None),
            )
            receive_age_ms = (
                (trace_monotonic - float(received_s)) * 1000.0
                if received_s is not None
                else float("nan")
            )
            frame_number = int(
                getattr(depth_frame, "frame_number", getattr(depth_frame, "source_frame_number", -1))
            )
            frame_key = (frame_number, capture_s)
            reused_frame = frame_key == getattr(
                self, "_timing_last_depth_frame_key", None
            )
            self._timing_last_depth_frame_key = frame_key
        if self.timing_trace_started_monotonic is None:
            self.timing_trace_started_monotonic = trace_monotonic
        self.timing_trace_records.append(
            (
                int(tick),
                trace_monotonic,
                prepare_low_state_s * 1000.0,
                float(getattr(self, "latest_depth_preprocess_s", float("nan"))) * 1000.0,
                float(getattr(self, "last_depth_encoder_s", float("nan"))) * 1000.0,
                float(getattr(self, "last_actor_network_s", float("nan"))) * 1000.0,
                prepare_obs_s * 1000.0,
                actor_s * 1000.0,
                control_s * 1000.0,
                total_s * 1000.0,
                int(total_s > self.rl_dt),
                int(reused_frame),
                capture_s,
                camera_receive_s,
                producer_publish_s,
                consume_s,
                camera_delivery_ms,
                producer_processing_ms,
                producer_to_policy_ms,
                capture_to_policy_ms,
                capture_age_ms,
                receive_age_ms,
                frame_number,
                gpu_hz / 1.0e6,
                cpu_c,
                gpu_c,
            )
        )
        return total_s

    def _write_timing_trace(self) -> None:
        if not self.timing_trace_path or not self.timing_trace_records:
            return
        path = Path(self.timing_trace_path).expanduser()
        if not path.parent.is_dir():
            logger.error(f"Timing trace parent directory does not exist: {path.parent}")
            return
        header = (
            "tick",
            "host_monotonic_s",
            "prepare_low_state_ms",
            "direct_depth_preprocess_ms",
            "depth_encoder_ms",
            "actor_network_ms",
            "prepare_obs_total_ms",
            "policy_inference_total_ms",
            "control_flow_ms",
            "total_loop_ms",
            "overrun",
            "reused_depth_frame",
            "depth_capture_monotonic_s",
            "depth_camera_receive_monotonic_s",
            "depth_producer_publish_monotonic_s",
            "depth_policy_first_consume_monotonic_s",
            "depth_camera_delivery_ms",
            "depth_producer_processing_ms",
            "depth_producer_to_policy_ms",
            "depth_capture_to_policy_ms",
            "depth_capture_age_ms",
            "depth_receive_age_ms",
            "depth_frame_number",
            "gpu_frequency_mhz",
            "cpu_temperature_c",
            "gpu_temperature_c",
        )
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(self.timing_trace_records)
            logger.info(
                f"Wrote {len(self.timing_trace_records)} timing ticks to {path}"
            )
        except OSError as exc:
            logger.error(f"Could not write timing trace {path}: {exc}")

    def _filter_policy_action(self, action: np.ndarray) -> np.ndarray:
        alpha = getattr(self, "action_filter_alpha", 1.0)
        if alpha == 1.0:
            self.filtered_policy_action = None
            return action
        if not self.use_policy_action or self.get_ready_state:
            self.filtered_policy_action = None
            return action
        previous = getattr(self, "filtered_policy_action", None)
        if previous is None:
            # Seed from first valid command; never reuse pre-stop motion.
            filtered = action.copy()
        else:
            if previous.shape != action.shape:
                raise ValueError("action filter shape changed")
            filtered = alpha * action + (1.0 - alpha) * previous
        if not np.all(np.isfinite(filtered)):
            raise ValueError("non-finite filtered policy action")
        self.filtered_policy_action = filtered.copy()
        return filtered

    def _rl_step_scheduled(self):
        if not hasattr(self, 'stand_fallback'):
            path = self.policy_config.get('standing_fallback_z_path')
            self.stand_fallback = StandFallback(self, path) if path else None
        loop_start = time.perf_counter()

        stage_start = time.perf_counter()
        with Timer(self.perf_dict, "prepare_low_state"):
            wireless_r2_stop = False
            if self.use_joystick:
                # print(f"Debug::process_joystick:")
                wireless_r2_stop = self.process_joystick_input()
            if self.pico_control_enabled and not wireless_r2_stop:
                self.process_pico_control_input()
                self._check_pico_control_watchdog()

            if not self.state_processor._prepare_low_state():
                self._handle_low_state_failure()
                return
            self.has_valid_low_state = True
        prepare_low_state_s = time.perf_counter() - stage_start
            
        try:
            stage_start = time.perf_counter()
            with Timer(self.perf_dict, "prepare_obs"):
                # Prepare observations
                if self.stand_fallback is not None:
                    self.stand_fallback.preflight()
                obs_dict, observations = self.prepare_obs_for_rl()
                self.state_dict["is_init"] = np.zeros(1, dtype=bool)
            prepare_obs_s = time.perf_counter() - stage_start
            observation_ready_monotonic = time.monotonic()
            observation_depth_message = getattr(self.state_processor, "latest_direct_depth_latent", None)

            stage_start = time.perf_counter()
            with Timer(self.perf_dict, "policy"):  
                # Inference
                action = np.asarray(self.policy(observations), dtype=np.float32)
                if not np.all(np.isfinite(action)):
                    self.enter_safe_stop("non-finite policy action")
                    self.state_dict["action"] = np.zeros(self.num_actions)
                    self._send_safe_hold_command()
                    return
                # Clip policy action
                action = self._filter_policy_action(action.clip(-1, 1))
                action_scaled = self.action_rescale * action
                self.last_action = action_scaled
            actor_s = time.perf_counter() - stage_start
        except Exception as e:
            self.enter_safe_stop(f"policy inference failed: {e}")
            self.state_dict["action"] = np.zeros(self.num_actions)
            self._send_safe_hold_command()
            return

        stage_start = time.perf_counter()
        with Timer(self.perf_dict, "rule_based_control_flow"):
            # rule based control flow
            if self.get_ready_state:
                q_target = self.get_init_target()
            elif not self.use_policy_action:
                q_target = self.state_processor.joint_pos
            else:
                policy_action = np.zeros((self.num_dofs))
                policy_action[self.controlled_joint_indices] = action_scaled
                policy_action = policy_action * self.action_scale
                q_target = policy_action + self.default_dof_angles

            # Clip q target
            # print(self.joint_pos_lower_limit)
            q_target = np.clip(
                q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
            )
            if not np.all(np.isfinite(q_target)):
                self.enter_safe_stop("non-finite q_target before slew limit")
                self._send_safe_hold_command()
                return

            q_target = self._apply_q_target_slew_limit(q_target)
            q_target = np.clip(
                q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
            )
            if not np.all(np.isfinite(q_target)):
                self.enter_safe_stop("non-finite q_target after slew limit")
                self._send_safe_hold_command()
                return

            # Send command
            cmd_q = q_target
            cmd_dq = np.zeros(self.num_dofs)
            cmd_tau = np.zeros(self.num_dofs)
            self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)
            self.last_cmd_q = np.asarray(cmd_q, dtype=np.float64).copy()
        control_s = time.perf_counter() - stage_start
        measurement_now = time.monotonic()
        depth_message = getattr(self.state_processor, "latest_direct_depth_latent", None)
        source_t = getattr(self, "ctx_latest_source_monotonic", None)
        sample_t = getattr(self, "ctx_latest_reference_sample_s", None)
        self.latency_stats.add(schedule_lag=getattr(self, "_schedule_lag_s", 0.0),
            lowstate=prepare_low_state_s, observation=prepare_obs_s,
            actor=actor_s, control_send=control_s, cycle=time.perf_counter()-loop_start,
            depth_age=measurement_now-depth_message.capture_monotonic_s if depth_message is not None else None,
            depth_at_obs=observation_ready_monotonic-observation_depth_message.capture_monotonic_s if observation_depth_message is not None else None,
            latent_to_obs=observation_ready_monotonic-observation_depth_message.completion_monotonic_s if observation_depth_message is not None else None,
            source_age=measurement_now-source_t if source_t is not None else None,
            reference_age=measurement_now-sample_t if sample_t is not None else None)

        elapsed = self._append_timing_trace(
            tick=self.total_inference_cnt,
            prepare_low_state_s=prepare_low_state_s,
            prepare_obs_s=prepare_obs_s,
            actor_s=actor_s,
            control_s=control_s,
            loop_start=loop_start,
        )
        overrun_warning_factor = getattr(self, "rl_overrun_warning_factor", 1.0)
        if elapsed > self.rl_dt * overrun_warning_factor:
            self._warn_throttled(
                "_last_rl_overrun_warning",
                f"RL step took {elapsed:.6f} seconds, expected "
                f"<={self.rl_dt * overrun_warning_factor:.6f} seconds",
            )


    def setup_pico_control(self):
        """Subscribe to Pico controller button JSON from the onboard teleop server."""
        try:
            import zmq
        except ImportError as exc:
            logger.warning(f"Pico control disabled because pyzmq is unavailable: {exc}")
            self.pico_control_enabled = False
            return

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, 0)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.pico_control_addr)
        self.pico_control_sock = sock
        logger.info(f"Pico policy control enabled: {self.pico_control_addr}")
        logger.info(
            "Pico controls: A=init, A+B=enable policy/start tracking, "
            "B=latched stop, X=freeze/resume retarget upstream"
        )

    def process_pico_control_input(self):
        if self.pico_control_sock is None:
            return

        try:
            import zmq
        except ImportError:
            return

        latest_buttons = None
        latest_source_monotonic = None
        latest_receive_monotonic = None
        while True:
            try:
                raw = self.pico_control_sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                logger.debug(f"Pico control receive failed: {exc}")
                return

            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            buttons = payload.get("controller_buttons")
            try:
                source_monotonic = float(payload["source_monotonic_s"])
            except (KeyError, TypeError, ValueError):
                continue
            receive_monotonic = time.monotonic()
            source_age_s = receive_monotonic - source_monotonic
            if (
                not np.isfinite(source_monotonic)
                or source_age_s < -self.pico_control_future_tolerance_s
                or source_age_s > self.pico_control_source_timeout_s
            ):
                continue
            if isinstance(buttons, dict):
                latest_buttons = {str(k): bool(v) for k, v in buttons.items()}
                latest_source_monotonic = source_monotonic
                latest_receive_monotonic = receive_monotonic

        if latest_buttons is None:
            return

        self.pico_control_have_valid_source = True
        self.pico_control_last_source_monotonic = latest_source_monotonic
        self.pico_control_last_receive_monotonic = latest_receive_monotonic
        self.handle_pico_buttons(latest_buttons)

    def _check_pico_control_watchdog(self) -> None:
        """Fail closed when Pico is the only policy controller and its stream is lost."""
        if not self.pico_only_control:
            return
        if not self.pico_control_have_valid_source:
            self._warn_throttled(
                "_last_pico_control_warning",
                "Waiting for fresh Pico controller input; policy enable is unavailable",
            )
            return

        now = time.monotonic()
        source_age_s = now - float(self.pico_control_last_source_monotonic)
        receive_age_s = now - float(self.pico_control_last_receive_monotonic)
        if max(source_age_s, receive_age_s) <= self.pico_control_source_timeout_s:
            return
        if not self.stop_latched:
            self.enter_input_fallback(
                "Pico controller stream stale "
                f"(source={source_age_s * 1000.0:.1f} ms, "
                f"receive={receive_age_s * 1000.0:.1f} ms)"
            )

    def handle_pico_buttons(self, buttons: Dict[str, bool]):
        prev = self.last_pico_buttons
        x = bool(buttons.get("left_key_one", False))
        if x and not bool(prev.get("left_key_one", False)):
            fallback = getattr(self, 'stand_fallback', None)
            if fallback is not None and fallback.active:
                fallback.cancel_auto()
            logger.info("Pico X: reference producer owns the reset; waiting for its epoch")
        a = bool(buttons.get("right_key_one", False))
        b = bool(buttons.get("right_key_two", False))

        prev_a = bool(prev.get("right_key_one", False))
        prev_b = bool(prev.get("right_key_two", False))

        combo = a and b
        prev_combo = prev_a and prev_b
        if self.stop_latched and not a and not b:
            self.pico_enable_released_after_stop = True
            self.stop_latch_enable_released_after_stop = True

        if combo and not prev_combo:
            if self.stop_latched:
                if not self.stop_latch_enable_released_after_stop:
                    self._warn_stop_latch_ignored("Pico A+B")
                    self.last_pico_buttons = buttons
                    return
                self._clear_stop_latch_for_rearm("Pico A+B")
            logger.info("Pico A+B: enable policy and start tracking")
            self.handle_joystick_button("R1")
            self.handle_joystick_button("B")
        elif a and not prev_a and not b:
            logger.info("Pico A: init")
            self.handle_joystick_button("A")
        elif b and not prev_b and not a:
            logger.info("Pico B: stop policy")
            self.handle_joystick_button("R2")
        self.last_pico_buttons = buttons


    def process_joystick_input(self):
        """Poll current wireless controller state and translate to high-level key events."""
        try:
            self.wc_msg = self.robot.read_wireless_controller()
        except Exception:
            return False

        if self.wc_msg is None:
            return False

        r2_stop_pressed = bool(self.wc_msg.R2)
        if r2_stop_pressed:
            self.enter_stop_latch("G1 wireless R2 stop")
            self.last_wc_msg = self.wc_msg
            return True

        if self.stop_latched:
            enable_or_start_pressed = bool(self.wc_msg.R1 or self.wc_msg.B)
            if not enable_or_start_pressed:
                self.stop_latch_enable_released_after_stop = True

            combo = bool(self.wc_msg.R1 and self.wc_msg.B)
            prev_combo = bool(self.last_wc_msg.R1 and self.last_wc_msg.B)
            if combo and not prev_combo:
                if self.stop_latch_enable_released_after_stop:
                    self._clear_stop_latch_for_rearm("Wireless R1+B")
                    self.handle_joystick_button("R1")
                    if getattr(self, 'manual_goals', None) is None and not getattr(self, 'file_standing_idle', False):
                        self.handle_joystick_button("B")
                    else:
                        logger.info('Re-armed in standing mode; release buttons, then press B separately')
                else:
                    self._warn_stop_latch_ignored("Wireless R1+B")
                self.last_wc_msg = self.wc_msg
                return False

            if (
                (self.wc_msg.R1 and not self.last_wc_msg.R1)
                or (self.wc_msg.B and not self.last_wc_msg.B)
            ):
                self._warn_stop_latch_ignored("Wireless enable/start")
                self.last_wc_msg = self.wc_msg
                return False

        # print(f"wc_msg.A: {self.wc_msg.A}")
        if self.wc_msg.A and not self.last_wc_msg.A:
            self.handle_joystick_button("A")
        if self.wc_msg.B and not self.last_wc_msg.B:
            self.handle_joystick_button("B")
        if self.wc_msg.X and not self.last_wc_msg.X:
            self.handle_joystick_button("X")
        if self.wc_msg.Y and not self.last_wc_msg.Y:
            self.handle_joystick_button("Y")
        if self.wc_msg.L1 and not self.last_wc_msg.L1:
            self.handle_joystick_button("L1")
        if self.wc_msg.L2 and not self.last_wc_msg.L2:
            self.handle_joystick_button("L2")
        if self.wc_msg.R1 and not self.last_wc_msg.R1:
            self.handle_joystick_button("R1")
        
        self.last_wc_msg = self.wc_msg
        return False


    def handle_joystick_button(self, cur_key):
        fallback = getattr(self, 'stand_fallback', None)
        if fallback is not None and fallback.active:
            if cur_key in ('R1', 'B') and not fallback.rearm():
                return
            if cur_key in ('A', 'X'):
                self._warn_throttled('_last_stand_rearm_warning', 'Standing fallback: use A+B after inputs recover, or B alone to stop')
                return
        if getattr(self, 'file_standing_idle', False):
            if cur_key == 'B' and (self.stop_latched or not self.use_policy_action):
                logger.warning('Playback ignored: initialize and enable policy first')
                return
            if cur_key == 'X':
                self._reset_file_standing()
                logger.info('Default standing reference; press B to restart playback')
                return
            if cur_key == 'R1' and not self.stop_latched:
                self._reset_file_standing()
        if getattr(self, 'manual_goals', None) is not None and cur_key == 'Y':
            self.manual_goals.index = 0
            self.start_motion = False
            self.ctx_started_monotonic = None
            self.ctx_holding_first = False
            self.t = self.manual_goals.frame
            self.tracking_heading_context_valid = False
            self.tracking_heading_relative_xy = None
            logger.info('MANUAL_GOAL reset to 1/{}; standing reference selected; enable and stop latch unchanged', len(self.manual_goals.frames))
            return
        if getattr(self, 'manual_goals', None) is not None and cur_key in ('B', 'X'):
            if self.stop_latched or not self.use_policy_action:
                logger.warning('Manual goal selection ignored: initialize and enable policy first')
                return
            if cur_key == 'X':
                if not self.start_motion:
                    logger.warning('Press B to enter current goal before X next')
                    return
                self.manual_goals.next()
            else:
                self.manual_goals.index = 0
            self.ctx_holding_first = False
            self.start_motion = True
            self.t = self.manual_goals.frame
            if self.ctx_started_monotonic is None:
                self.ctx_started_monotonic = time.monotonic()
            entry = self.manual_goals.entries[self.manual_goals.index]
            logger.info('MANUAL_GOAL {}/{} motion={} source_frame={} context_frame={}',
                        self.manual_goals.index+1, len(self.manual_goals.frames),
                        entry['motion'], entry['frame'], self.t)
            return
        if cur_key in ("A", "X"):
            self.ctx_holding_first = False
        if cur_key == "R1":
            if self.stop_latched:
                self._warn_stop_latch_ignored("R1 enable")
                return
            if getattr(self, 'manual_goals', None) is not None:
                self.start_motion = False
                self.ctx_started_monotonic = None
                self.ctx_holding_first = False
                logger.info('MANUAL_GOAL standing reference; press B for goal 1')
            logger.info("Using policy actions")
            self.use_policy_action = True
            self.get_ready_state = False    
            self.t = self.t_stop

        elif cur_key == "R2":
            self.enter_stop_latch("R2 stop command")
            logger.info(colored("Actions set to zero", "blue"))
        elif cur_key == "A":
            self.get_ready_state = True
            self.init_count = 0
            logger.info(colored("Setting to init state (do this when robot was in a bad shape)", "blue"))
        elif cur_key == "B":
            if self.stop_latched:
                self._warn_stop_latch_ignored("B start")
                return
            logger.info("Starting motion")
            self.ctx_holding_first = False
            self.start_motion = True
            self.t = self.t_start
            self.ctx_started_monotonic = time.monotonic()
        elif cur_key == "X":
            self.start_motion = False
            self.t = self.t_stop
            self.ctx_started_monotonic = None
            logger.info("Resetting to stop state")

        # Debug print for kp level tuning
        if cur_key in ["Y+left", "Y+right", "A+left", "A+right"]:
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))

    # ----------------------------- Keyboard handling -----------------------------
    def _warn_unsupported_keyboard_key(self, keycode: str) -> None:
        logger.warning(f"Keyboard key '{keycode}' is not supported by this release runtime")

    def _set_debug_kp_level(self, *, delta: float | None = None, value: float | None = None) -> None:
        if value is None:
            value = self.command_sender.kp_level + float(delta or 0.0)
        value = max(0.0, float(value))
        self.command_sender.kp_level = value
        logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
        logger.info(
            colored(
                f"Debug kp: {np.array2string(self.command_sender.joint_kp_unitree, precision=4)}",
                "green",
            )
        )

    def start_key_listener(self):
        """Start a key listener using sshkeyboard (same as BasePolicy)."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except Exception as e:
                logger.warning(f"Keyboard key {keycode}. Error: {e}")

        listen_keyboard(on_press=on_press, sequential=True)

    def handle_keyboard_button(self, keycode):
        controls = {"]": "R1", "[": "B", "i": "A", "o": "R2", "p": "X"}
        if self.manual_goals is not None:
            controls.update({"n": "X", "p": "Y"})
        if keycode in controls:
            self.handle_joystick_button(controls[keycode])
        elif keycode in {"w", "s", "a", "d", "q", "e", "z"}:
            self._warn_unsupported_keyboard_key(keycode)
        elif keycode == "5":
            self._set_debug_kp_level(delta=-0.01)
        elif keycode == "6":
            self._set_debug_kp_level(delta=0.01)
        elif keycode == "4":
            self._set_debug_kp_level(delta=-0.1)
        elif keycode == "7":
            self._set_debug_kp_level(delta=0.1)
        elif keycode == "0":
            self._set_debug_kp_level(value=1.0)
        else:
            self._warn_unsupported_keyboard_key(keycode)
        

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1_real.yaml", help="robot config file"
    )
    parser.add_argument(
        "--policy_config", type=str, help="policy config file"
    )
    parser.add_argument(
        "--model_path", type=str, help="model path"
    )
    parser.add_argument(
        "--task", type=str, help="task YAML path", default="track-walk"
    )
    parser.add_argument(
        "--ctx_path",
        type=str,
        default=None,
        help="override tracking ctx_path; supports synchronized .npz",
    )
    parser.add_argument(
        "--ctx_source",
        type=str,
        default=None,
        help="override tracking ctx_source, npz or zmq",
    )
    parser.add_argument(
        "--pico-control", action="store_true",
        help="Enable Pico controller buttons from the onboard teleop server for policy mode control.",
    )
    parser.add_argument(
        "--pico-control-addr", type=str, default="tcp://127.0.0.1:28704",
        help="ZMQ SUB address for Pico controller button JSON.",
    )
    parser.add_argument(
        "--pico-only-control",
        action="store_true",
        help="Disable G1 wireless/keyboard controls and use only fresh Pico button input.",
    )
    parser.add_argument(
        "--pico-control-source-timeout-ms",
        type=float,
        default=250.0,
        help="Latch stop if Pico-only controller source data exceeds this age.",
    )
    parser.add_argument(
        "--nodryrun",
        action="store_true",
        help=(
            "Enable real G1 motor commands. Requires ROBOT_TYPE=g1_real, "
            "UFO_REAL_ROBOT_OK=1, and an explicit command-timeout contract. "
            "An unverified contract runs with a prominent operator-supervised warning."
        ),
    )

    args = parser.parse_args()

    policy_config = load_yaml_config(args.policy_config)
    
    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)

    with open(args.task, 'r') as file:
        exp_config = yaml.load(file, Loader=yaml.FullLoader)
    if args.ctx_path is not None:
        exp_config["ctx_path"] = args.ctx_path
    if args.ctx_source is not None:
        exp_config["ctx_source"] = args.ctx_source
    model_path = args.model_path

    policy = UFODeployPolicy(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=model_path,
        exp_config=exp_config,
        rl_rate=50,
        pico_control=bool(args.pico_control),
        pico_control_addr=str(args.pico_control_addr),
        pico_only_control=bool(args.pico_only_control),
        pico_control_source_timeout_ms=float(args.pico_control_source_timeout_ms),
        dry_run=not bool(args.nodryrun),
    )
    raise SystemExit(policy.run())
