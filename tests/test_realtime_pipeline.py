"""Exercise the real z-server process with synthetic retarget packets over loopback."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import zmq

from scripts.realtime.realtime_z_server import G1_29DOF_DEFAULT_POSE_RAD
from utils.teleop_velocity import VelocityContextMessage

ROOT = Path(__file__).resolve().parents[1]


def test_pause_follow_gap_and_reference_reanchor(tmp_path):
    context = zmq.Context()
    request, reply, subscriber = [context.socket(kind) for kind in (zmq.PULL, zmq.PUSH, zmq.SUB)]
    for socket in (request, reply, subscriber):
        socket.setsockopt(zmq.LINGER, 0)
    request_port = request.bind_to_random_port('tcp://127.0.0.1')
    reply_port = reply.bind_to_random_port('tcp://127.0.0.1')
    # Subscriber binds so the test owns all ports before the child starts.
    subscriber_port = subscriber.bind_to_random_port('tcp://127.0.0.1')
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')
    log = (tmp_path / 'z.log').open('w+')
    process = None
    try:
        # The normal publisher binds; a tiny in-process forwarder is unnecessary:
        # release this reserved port immediately before starting the single child.
        subscriber.unbind(f'tcp://127.0.0.1:{subscriber_port}')
        subscriber.connect(f'tcp://127.0.0.1:{subscriber_port}')
        # Keep the test away from an operator's real recovery handshake files.
        runner = '''
import os
from pathlib import Path
from scripts.realtime import realtime_z_server as server
def isolated_path(value):
    path = Path(value)
    if str(path).startswith('/tmp/pbfm_teleop_recovery_'):
        return Path(os.environ['PBFM_TEST_RECOVERY_DIR']) / path.name
    return path
server.Path = isolated_path
server.main()
'''
        command = [sys.executable, '-u', '-c', runner,
            '--teleop_req', f'tcp://127.0.0.1:{request_port}',
            '--teleop_rep', f'tcp://127.0.0.1:{reply_port}',
            '--z_bind', f'tcp://127.0.0.1:{subscriber_port}',
            '--backward_onnx', str(ROOT / 'model/g1_policy/onnx/backward_encoder.onnx'),
            '--mujoco_xml', str(ROOT / 'data/robots/g1/g1_29dof_pbfm_training.xml'),
            '--initial-mode', 'follow', '--publish-qv-velocity', '--root_height_obs',
            '--wall-clock-dt', '--no-freeze-static-pose', '--fix-quat-continuity',
            '--angvel-delta-frame', 'world', '--pose-transition-ms', '100',
            '--max-retarget-age-ms', '200', '--max-z-delta', '.75']
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PBFM_TEST_RECOVERY_DIR=str(tmp_path)))

        def phase(duration, paused, epoch, yaw, send=True):
            packets = []
            started = time.monotonic()
            while time.monotonic() - started < duration:
                assert process.poll() is None, (tmp_path / 'z.log').read_text()
                if request.poll(5):
                    while request.poll(0):
                        request.recv()
                    if send:
                        now = time.monotonic_ns()
                        payload = {'frames': [{'root_pos': [0., 0., .78],
                            'root_quat': [np.cos(yaw/2), 0., 0., np.sin(yaw/2)],
                            'dof_pos': G1_29DOF_DEFAULT_POSE_RAD.tolist()}],
                            'retarget_paused': paused, 'reference_epoch': epoch,
                            'source_receive_monotonic_ns': now, 'sample_monotonic_ns': now,
                            'retarget_age_ms': 0}
                        reply.send_string(json.dumps(payload))
                while subscriber.poll(0):
                    packet = VelocityContextMessage.from_bytes(subscriber.recv())
                    assert np.isfinite(packet.z).all()
                    packets.append((time.monotonic()-started, packet))
            return packets

        paused = phase(1.5, True, 0, 0.)
        assert paused and not any(p.heading_valid for _, p in paused)
        live = phase(.5, False, 1, 1.2)
        live = [(age, p) for age, p in live if p.reference_epoch > paused[-1][1].reference_epoch]
        valid = [p for _, p in live if p.heading_valid]
        assert valid
        np.testing.assert_allclose(valid[0].heading_relative_xy, [1., 0.], atol=1e-5)
        assert not live[0][1].heading_valid  # The configured transition is actually used.
        assert live[0][1].reference_live  # Recovery must not be mistaken for an X pause.
        assert valid[0].reference_sample_s - live[0][1].reference_sample_s >= .08
        paused = phase(.5, True, 2, 1.2)
        assert not paused[-1][1].heading_valid
        live = phase(.5, False, 3, -1.1)
        valid = [p for _, p in live if p.heading_valid]
        assert valid
        np.testing.assert_allclose(valid[0].heading_relative_xy, [1., 0.], atol=1e-5)
        stale = phase(.55, False, 3, -1.1, send=False)
        assert not [p for age, p in stale if age > .35]
        recovered = phase(.5, False, 3, -1.1)
        assert any(p.heading_valid for _, p in recovered)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        log.close()
        for socket in (request, reply, subscriber):
            socket.close()
        context.term()
