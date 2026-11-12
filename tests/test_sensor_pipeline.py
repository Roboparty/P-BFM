"""Recorded simulation replay plus synthetic raw-camera integration; never actuates."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import onnxruntime as ort
import pytest
from scipy.spatial.transform import Rotation
import zmq

from rl_policy.direct_depth import DirectDepthConfig
from rl_policy.ufo_qv_policy import UFODeployPolicy
from utils.common import LowStateMessage, DirectDepthLatentMessage
from utils.config import load_yaml_config
from utils.latency_stats import LatencyStats
from utils.timestamp_depth_history import TimestampDepthHistory
from utils.training_depth_rays import TrainingDepthSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'model/g1_policy'


def test_lowstate_bridge_rejects_duplicates_and_invalid_imu(monkeypatch):
    from scripts.onboard.realsense_g1_terrain_producer import LowStateBridge
    packets = []
    publisher = SimpleNamespace(send=lambda data, **kwargs: packets.append(LowStateMessage.from_bytes(data)))
    bridge = LowStateBridge(publisher)
    clock = [10.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    message = SimpleNamespace(tick=100,
        imu_state=SimpleNamespace(quaternion=[1., 0., 0., 0.], gyroscope=[.1, .2, .3]),
        motor_state=[SimpleNamespace(q=i*.001, dq=i*.002, tau_est=0.) for i in range(29)])
    bridge.on_low_state(message)
    assert len(packets) == 1 and packets[0].capture_monotonic_s == 10.
    np.testing.assert_allclose(packets[0].joint_positions, np.arange(29)*.001)
    bridge.on_low_state(message)
    assert len(packets) == 1 and 'duplicate' in bridge.last_error
    clock[0] += .005
    message.tick = 105
    bridge.on_low_state(message)
    assert len(packets) == 2 and packets[-1].capture_monotonic_s == pytest.approx(10.005)
    clock[0] += .005
    message.tick = 1  # Controller restart: re-anchor the source clock.
    bridge.on_low_state(message)
    assert len(packets) == 3 and packets[-1].capture_monotonic_s == pytest.approx(10.010)
    for quaternion in ([0., 0., 0., 0.], [np.nan, 0., 0., 0.]):
        message.tick += 1
        message.imu_state.quaternion = quaternion
        bridge.on_low_state(message)
        assert len(packets) == 3 and 'quaternion' in bridge.last_error


class WireQueue:
    def __init__(self):
        self.packets = deque()

    def recv(self, flags=0):
        if not self.packets:
            raise zmq.Again()
        return self.packets.popleft()

    def close(self, *args, **kwargs):
        pass


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_yaml_config(MODEL / 'policy.yaml')
    cfg['actor_inference_backend'] = 'onnx'
    runtime = UFODeployPolicy(load_yaml_config(ROOT / 'config/robot/g1_real.yaml'), cfg,
                              load_yaml_config(ROOT / 'config/tasks/a417.yaml'),
                              str(MODEL / 'onnx/actor.onnx'), dry_run=True)
    for attr in ['low_state_socket', 'direct_depth_latent_socket']:
        getattr(runtime.state_processor, attr).close(linger=0)
        setattr(runtime.state_processor, attr, WireQueue())
    assert not hasattr(runtime, 'robot')
    runtime.state_dict = {'action': np.zeros(29)}
    runtime.total_inference_cnt = 0
    runtime.perf_dict = {}
    runtime.latency_stats = LatencyStats('replay')
    runtime.use_policy_action = runtime.start_motion = True
    runtime.get_ready_state = runtime.stop_latched = False
    runtime.ctx_started_monotonic = 10.
    clock = [10.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    encoder = ort.InferenceSession(str(MODEL / 'onnx/depth_encoder.onnx'), options, providers=['CPUExecutionProvider'])
    yield runtime, encoder, clock
    runtime._close_command_interface()


def push_state(runtime, q, dq, quaternion, gyro, t, sequence):
    indices = runtime.state_processor.joint_indices_in_source
    source_q, source_dq = np.zeros(29), np.zeros(29)
    source_q[indices], source_dq[indices] = q, dq
    packet = LowStateMessage(quaternion, gyro, source_q, source_dq,
                             tick=sequence, capture_monotonic_s=t)
    runtime.state_processor.low_state_socket.packets.append(packet.to_bytes())
    assert runtime.state_processor._prepare_low_state()


def push_depth(runtime, latent, t, sequence):
    packet = DirectDepthLatentMessage(depth_latent=latent, source_frame_number=sequence,
        worker_sequence=sequence, reset_generation=0, source_timestamp_s=t,
        capture_monotonic_s=t, camera_receive_monotonic_s=t, sync_ready_monotonic_s=t,
        producer_publish_monotonic_s=t, worker_start_monotonic_s=t,
        completion_monotonic_s=t, preprocess_s=0., encoder_s=0., worker_total_s=0.)
    runtime.state_processor.direct_depth_latent_socket.packets.append(packet.to_bytes())


def test_recorded_simulation_lowstate_to_actor(pipeline):
    runtime, encoder, clock = pipeline
    with np.load(ROOT / 'tests/fixtures/a417_sim_replay.npz', allow_pickle=False) as trace:
        for i, state in enumerate(trace['state']):
            clock[0] = 10. + i * .02 + 1e-7
            gravity = state[58:61]
            gravity = gravity / np.linalg.norm(gravity)
            pitch = np.arcsin(gravity[0])
            roll = np.arctan2(-gravity[1], -gravity[2])
            target = runtime.ctx_heading_relative_xy[i]
            error = np.arctan2(trace['heading'][i, 1], 1. - trace['heading'][i, 0])
            yaw = np.arctan2(target[1], target[0]) - error
            xyzw = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
            push_state(runtime, state[:29] + runtime.default_dof_angles, state[29:58],
                       xyzw[[3, 0, 1, 2]], state[61:64] * 4., clock[0], i)
            # Replay the recorded previous action; this is not a physics rollout.
            runtime.last_action = trace['last_action'][i].copy()
            image = trace['depth_image'][i:i+1].astype(np.float32) / 255.
            latent = encoder.run(None, {'depth_image': image})[0]
            push_depth(runtime, latent, clock[0], i)
            _, obs = runtime.prepare_obs_for_rl()
            # Reconstructing a normalized quaternion adds float32 rounding.
            np.testing.assert_allclose(obs[0, :64], state, atol=2e-6, rtol=0)
            np.testing.assert_array_equal(obs[0, 64:93], trace['last_action'][i])
            np.testing.assert_allclose(obs[0, 93:465], trace['history_actor'][i], atol=2e-6, rtol=0)
            np.testing.assert_allclose(obs[0, 465:467], trace['heading'][i], atol=1.1e-6, rtol=0)
            np.testing.assert_allclose(obs[0, -256:], trace['z'][i], atol=1e-6, rtol=0)
            action = runtime.policy(obs)
            np.testing.assert_allclose(action[0], trace['expected_action'][i], atol=2e-5, rtol=0)


def test_raw_camera_to_actor_and_stale_inputs(pipeline):
    runtime, encoder, clock = pipeline
    config = DirectDepthConfig.from_policy_config(runtime.policy_config)
    intrinsics = np.array([[240., 0., 239.5], [0., 240., 134.5], [0., 0., 1.]])
    sampler = TrainingDepthSampler(intrinsics, (270, 480), config.processing_profile)
    history = TimestampDepthHistory(config)
    for i in range(45):
        clock[0] = 10. + i * .02 + 1e-7
        raw = np.full((270, 480), 1. + i * .01, np.float32)
        raw[:20] = 0.
        raw[:, 240:] = 3.  # Far side and invalid top must use maximum distance.
        depth = history.append(sampler.sample(raw), timestamp_s=clock[0])
        assert depth.shape == (8, 18, 32) and depth.dtype == np.uint8
        assert np.all(depth[:, :, -1] == 255)
        latent = encoder.run(None, {'depth_image': depth[None].astype(np.float32) / 255.})[0]
        push_depth(runtime, latent, clock[0], i)
        push_state(runtime, runtime.default_dof_angles, np.zeros(29), [1., 0., 0., 0.], np.zeros(3), clock[0], i)
        runtime._rl_step_scheduled()
        assert runtime.use_policy_action and not runtime.stop_latched
        assert np.isfinite(runtime.last_cmd_q).all()
    # Fresh low-state with a stale depth packet must stop tracking.
    clock[0] += .2
    push_state(runtime, runtime.default_dof_angles, np.zeros(29), [1., 0., 0., 0.], np.zeros(3), clock[0], 100)
    runtime._rl_step_scheduled()
    assert not runtime.use_policy_action and not runtime.start_motion
    # Re-receiving the same old state must not make its capture timestamp fresh.
    clock[0] += .2
    assert not runtime.state_processor._prepare_low_state()
