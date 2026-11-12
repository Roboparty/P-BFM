"""Single-checkpoint packaging checks; never construct a motor interface."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rl_policy.ufo_qv_policy import UFODeployPolicy, _load_tracking_context_bundle
from utils.config import load_yaml_config
from scripts.verify_release import verify_config

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'model/g1_policy'
TASKS = [entry['task'] for entry in json.loads((MODEL / 'tasks.json').read_text())]


@pytest.mark.parametrize('xml', [
    'data/robots/g1/g1_29dof_freebase.xml',
    'data/robots/g1/g1_29dof_pbfm_training.xml',
    'data/robots/g1/scene_29dof_freebase.xml',
    'scripts/teleop/motion_tracking_retarget/assets/g1.xml',
])
def test_deployment_xml_loads_with_retained_assets(xml):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(ROOT / xml))
    assert model.nq == 36 and model.nv == 35


def test_retarget_configuration_loads():
    from scripts.teleop.motion_tracking_retarget.robot_config import load_teleop_robot_config
    config = load_teleop_robot_config('g1')
    assert config.dof_count == 29 and config.ctrl_fps == 50


def test_actor_builder_uses_release_paths_and_fp32(tmp_path):
    trtexec = tmp_path / 'trtexec'
    trtexec.write_text('#!/usr/bin/env python3\n'
        'import json, os, pathlib, sys\n'
        'pathlib.Path(os.environ["BUILD_ARGUMENTS"]).write_text(json.dumps(sys.argv[1:]))\n'
        'for arg in sys.argv[1:]:\n'
        '    if arg.startswith("--saveEngine="): pathlib.Path(arg.split("=",1)[1]).write_bytes(b"test")\n')
    trtexec.chmod(0o755)
    lowered = tmp_path / 'actor_trt85.onnx'
    lowered.touch()
    env = dict(os.environ, TRTEXEC=str(trtexec), MODEL_ROOT=str(MODEL),
               OUTPUT_DIR=str(tmp_path), SKIP_LOWERING='1', BUILD_ARGUMENTS=str(tmp_path / 'args.json'))
    for key in ('ACTOR_ENGINE', 'ACTOR_ONNX', 'ACTOR_LOWERED', 'ACTOR_SHAPES'):
        env.pop(key, None)
    subprocess.run(['bash', str(ROOT / 'scripts/onboard/build_tensorrt_engines.sh')],
                   cwd='/tmp', env=env, capture_output=True, text=True, check=True)
    args = json.loads((tmp_path / 'args.json').read_text())
    assert f'--onnx={lowered}' in args
    assert f'--saveEngine={tmp_path / "actor_fp32.engine"}' in args
    assert '--noTF32' in args and '--fp16' not in args


def test_launcher_requires_explicit_motor_opt_in():
    env = dict(os.environ, G1_INTERFACE='unused-test-interface', PYTHON_BIN=sys.executable)
    env.pop('UFO_REAL_ROBOT_OK', None)
    result = subprocess.run(['bash', str(ROOT / 'run.sh'), 'real', 'a417'],
                            cwd='/tmp', env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'explicitly enable motor control' in result.stderr


def test_camera_contract_environment():
    result = subprocess.run(['bash', '-c', 'source "$1"; env', 'bash', str(ROOT / 'scripts/deploy_env.sh')],
                            cwd='/tmp', check=True, capture_output=True, text=True)
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    assert values['DEPTH_TRAINING_RAYS'] == '1'
    assert values['DEPTH_PROCESSING_PROFILE'] == 'optical_z_maxcrop18_reset_v2'
    assert values['BACKWARD_ONNX_SHA256'] == json.loads((MODEL / 'manifest.json').read_text())['onnx_sha256']['backward_encoder.onnx']


@pytest.mark.parametrize('override', [
    'direct_depth_preprocess:\n  gaussian_enabled: false\n',
    'joint_kp:\n  waist_yaw_joint: 1\n',
    'history_update_after_compute: false\n',
])
def test_verify_checks_actual_policy_overrides(tmp_path, override):
    config = tmp_path / 'policy.yaml'
    config.write_text(f'extends: {MODEL / "policy.yaml"}\n' + override)
    with pytest.raises(ValueError, match='deployment contract'):
        verify_config(config, ROOT / 'config/robot/g1_real.yaml')


def test_verify_environment_override_is_not_ignored(tmp_path):
    config = tmp_path / 'policy.yaml'
    config.write_text(f'extends: {MODEL / "policy.yaml"}\ndirect_depth_preprocess:\n  gaussian_enabled: false\n')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_release.py')],
                            env=dict(os.environ, POLICY_CONFIG=str(config)), capture_output=True, text=True)
    assert result.returncode != 0
    assert 'deployment contract: direct_depth_preprocess' in result.stderr


def test_verify_resolves_configured_engine(tmp_path):
    config = tmp_path / 'policy.yaml'
    engine = tmp_path / 'custom.engine'
    config.write_text(f'extends: {MODEL / "policy.yaml"}\ntensorrt_actor_engine_path: {engine}\n')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_release.py'),
                             '--policy-config', str(config), '--print-engine-path'],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(engine)


def test_runtime_cannot_verify_a_different_engine(tmp_path):
    config = tmp_path / 'policy.yaml'
    config.write_text(f'extends: {MODEL / "policy.yaml"}\ntensorrt_actor_engine_path: {tmp_path / "actual.engine"}\n')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_release.py'),
                            '--policy-config', str(config), '--runtime',
                            '--task-config', str(ROOT / 'config/tasks/a417.yaml'),
                            '--engine', str(tmp_path / 'different.engine')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'differs from the configured runtime engine' in result.stderr


def test_robot_limit_override_is_checked(tmp_path):
    config = tmp_path / 'robot.yaml'
    config.write_text(f'extends: {ROOT / "config/robot/g1_real.yaml"}\njoint_velocity_limit: {{".*": 10000}}\n')
    with pytest.raises(ValueError, match='deployment contract: joint_velocity_limit'):
        verify_config(MODEL / 'policy.yaml', config)


def test_launcher_rejects_unverified_trailing_overrides():
    env = dict(os.environ, G1_INTERFACE='unused-test-interface', UFO_REAL_ROBOT_OK='1')
    result = subprocess.run(['bash', str(ROOT / 'run.sh'), 'real', 'a417', '--policy_config', '/tmp/other.yaml'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2 and 'accepts only TASK' in result.stderr


@pytest.mark.parametrize('override', [False, True])
def test_teleop_launcher_preserves_explicit_policy_config(tmp_path, override):
    # Intercept before the real launcher; this test never opens the motor interface.
    flock = tmp_path / 'flock'
    flock.write_text('#!/bin/sh\nprintf "%s\\n" "$POLICY_CONFIG"\n')
    flock.chmod(0o755)
    env = dict(os.environ, G1_INTERFACE='unused', UFO_REAL_ROBOT_OK='1',
               PATH=f'{tmp_path}:{os.environ["PATH"]}')
    env.pop('POLICY_CONFIG', None)
    expected = str(MODEL / 'teleop_policy.yaml')
    if override:
        expected = env['POLICY_CONFIG'] = str(tmp_path / 'custom-teleop.yaml')
    result = subprocess.run(['bash', str(ROOT / 'run.sh'), 'real', 'teleop'],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == expected


def test_context_requires_npz_before_deserialization(tmp_path):
    path = tmp_path / 'context.pkl'
    path.write_bytes(b'not a supported context')
    with pytest.raises(ValueError, match='must use .npz'):
        _load_tracking_context_bundle(path)
    path = tmp_path / 'context.npz'
    np.savez(path, z=np.ones((2, 256)), heading_relative_xy=np.array([[1., 0.]]),
             heading_valid=np.ones(2), dt=.02)
    with pytest.raises(ValueError, match='heading shape'):
        _load_tracking_context_bundle(path)


@pytest.mark.parametrize('task', [{'type': 'reward'}, {'type': 'tracking', 'ctx_source': 'pkl'}])
def test_unsupported_task_rejected_before_hardware_setup(task):
    with pytest.raises(ValueError, match='Tasks must|ctx_source must'):
        UFODeployPolicy({}, {}, task, 'unused.onnx', dry_run=False)


@pytest.mark.parametrize('robot', [
    {'ROBOT_TYPE': 'g1_29dof', 'LOW_STATE_SOURCE': 'zmq'},
    {'ROBOT_TYPE': 'g1_real', 'LOW_STATE_SOURCE': 'direct'},
    {'ROBOT_TYPE': 'other', 'LOW_STATE_SOURCE': 'zmq'},
])
def test_unsupported_state_transport_rejected_before_hardware(robot):
    from rl_policy.utils.state_processor import StateProcessor
    with pytest.raises(ValueError, match='Release requires ROBOT_TYPE=g1_real and LOW_STATE_SOURCE=zmq'):
        StateProcessor(robot, [])
    with pytest.raises(ValueError, match='Release requires ROBOT_TYPE=g1_real and LOW_STATE_SOURCE=zmq'):
        UFODeployPolicy(robot, {}, {'type': 'tracking'}, 'unused.onnx', dry_run=False)


def test_native_command_mapping_and_close_without_hardware():
    from rl_policy.utils.command_sender import CommandSender
    from utils.strings import unitree_joint_names
    written = []
    robot = SimpleNamespace(
        create_zero_command=lambda: SimpleNamespace(**{
            key: [0.] * 29 for key in ('q_target', 'dq_target', 'tau_ff', 'kp', 'kd')}),
        write_low_command=written.append)
    config = load_yaml_config(MODEL / 'policy.yaml')
    config['isaac_joint_names'] = unitree_joint_names[::-1]
    sender = CommandSender({'ROBOT_TYPE': 'g1_real', 'robot': robot}, config)
    sender.kp_level = .5
    q = np.linspace(-.1, .1, 29)
    sender.send_command(q, q * 2, q * 3)
    assert len(written) == 1
    np.testing.assert_array_equal(written[0].q_target, q[::-1])
    np.testing.assert_array_equal(written[0].dq_target, q[::-1] * 2)
    np.testing.assert_array_equal(written[0].tau_ff, q[::-1] * 3)
    hip = unitree_joint_names.index('left_hip_pitch_joint')
    assert written[0].kp[hip] == pytest.approx(99.09843 * .5)
    assert written[0].kd[hip] == pytest.approx(6.3088)
    sender.close()
    sender.close()
    with pytest.raises(RuntimeError, match='closed'):
        sender.send_command(q, q, q)
    assert len(written) == 1


def test_command_sender_dry_run_never_accesses_motor_interface():
    from rl_policy.utils.command_sender import CommandSender
    sender = CommandSender({'ROBOT_TYPE': 'g1_real', '_UFO_DRY_RUN': True},
                           load_yaml_config(MODEL / 'policy.yaml'))
    sender.send_command(np.zeros(29), np.zeros(29), np.zeros(29))
    assert not hasattr(sender, 'robot')
    sender.close()


@pytest.mark.parametrize('mode', ['tracking', 'reward', 'goal'])
def test_keyboard_uses_shared_controls_and_stop_latch(mode):
    from utils.manual_goal import ManualGoalSequence
    runtime = UFODeployPolicy.__new__(UFODeployPolicy)
    runtime.manual_goals = None
    if mode == 'goal':
        runtime.manual_goals = ManualGoalSequence(
            MODEL / 'contexts/01_goal21/schedule.json', MODEL / 'contexts/01_goal21/context.npz')
    runtime.file_standing_idle = mode == 'reward'
    runtime.use_policy_action = False
    runtime.stop_latched = False
    runtime.start_motion = False
    runtime.t_start = runtime.t_stop = 0
    runtime.ctx_source = 'npz'
    runtime.last_action = np.ones(29)
    runtime.handle_keyboard_button(']')
    assert runtime.use_policy_action and not runtime.start_motion
    runtime.handle_keyboard_button('[')
    assert runtime.start_motion
    if mode == 'goal':
        runtime.handle_keyboard_button('n')
        assert runtime.manual_goals.index == 1
    runtime.handle_keyboard_button('p')
    assert not runtime.start_motion and runtime.t == 0
    if mode == 'goal':
        assert runtime.manual_goals.index == 0
    runtime.handle_keyboard_button('o')
    assert runtime.stop_latched and not runtime.use_policy_action
    runtime.handle_keyboard_button(']')
    runtime.handle_keyboard_button('[')
    assert not runtime.use_policy_action and not runtime.start_motion


def test_keyboard_listener_handles_blocking_api(monkeypatch):
    calls = []
    def listen(*, on_press, sequential):
        assert sequential
        on_press('i')
        # sshkeyboard returns None after listening stops.
    monkeypatch.setattr('rl_policy.ufo_qv_policy.listen_keyboard', listen)
    UFODeployPolicy.start_key_listener(SimpleNamespace(handle_keyboard_button=calls.append))
    assert calls == ['i']


@pytest.mark.parametrize('task', TASKS)
def test_task_constructs_and_infers_without_hardware(task, monkeypatch):
    monkeypatch.chdir(ROOT)
    policy_config = load_yaml_config(MODEL / ('teleop_policy.yaml' if task == 'teleop' else 'policy.yaml'))
    policy_config['actor_inference_backend'] = 'onnx'
    runtime = UFODeployPolicy(
        robot_config=load_yaml_config(ROOT / 'config/robot/g1_real.yaml'),
        policy_config=policy_config,
        exp_config=load_yaml_config(ROOT / f'config/tasks/{task}.yaml'),
        model_path=str(MODEL / 'onnx/actor.onnx'), dry_run=True)
    assert runtime.dry_run and not hasattr(runtime, 'robot')
    runtime.state_processor = SimpleNamespace(root_quat_b=np.array([1., 0., 0., 0.]))
    runtime.use_policy_action = True
    runtime.start_motion = True
    runtime.stop_latched = False
    runtime.ctx_started_monotonic = 1.
    runtime.t = 0
    if task == 'teleop':
        runtime.ctx_latest_velocity_valid = True
        runtime.ctx_latest_heading_valid = True
        runtime.ctx_latest_velocity_heading_xy = np.array([.8, .1], np.float32)
    with np.load(MODEL / 'onnx/reference_cases.npz') as data:
        encoder = __import__('onnxruntime').InferenceSession(str(MODEL / 'onnx/depth_encoder.onnx'), providers=['CPUExecutionProvider'])
        latent = encoder.run(None, {'depth_image': data['depth_image'][0]})[0]
        flat = np.concatenate([data[k][0] for k in ('state', 'last_action', 'history_actor', 'heading')] + [latent, data['z'][0]], axis=-1)
        action = runtime.policy(flat)
        assert action.shape == (1, 29) and np.isfinite(action).all()
    if runtime.ctx_zmq is not None:
        runtime.ctx_zmq.close(linger=0)
