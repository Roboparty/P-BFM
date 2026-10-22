"""Compare the effective G1 configuration with a separate published checkout."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_g1_configuration_matches_published_checkout():
    baseline = os.environ.get("PBFM_G1_BASELINE")
    if not baseline:
        pytest.skip("set PBFM_G1_BASELINE to the G1-only checkout")
    candidate = Path(__file__).resolve().parents[1]
    program = '''
import dataclasses, enum, json
import numpy as np
from omegaconf import OmegaConf
from pbfm.train import build_ufo_mjlab_config
from pbfm.envs.mjlab.task_config import _compose_humanoidverse_config
from pbfm.envs.mjlab.config import make_mjlab_ufo_env_cfg
from pbfm.training.checkpoint import _depth_observation_contract
from pbfm.utils.resource_paths import PROJECT_ROOT
OmegaConf.register_new_resolver('len', len, replace=True)

def canonical(value):
    if dataclasses.is_dataclass(value):
        return {f.name: canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, slice):
        return {'slice': [value.start, value.stop, value.step]}
    if isinstance(value, enum.Enum):
        return value.name
    if callable(value):
        return value.__module__ + '.' + value.__qualname__
    if isinstance(value, str):
        return value.replace(str(PROJECT_ROOT), '<CODE>')
    return value

cfg = build_ufo_mjlab_config(device='cpu',num_envs=4,data_path='g1_motions.pkl')
e = cfg.env
task, _ = _compose_humanoidverse_config(num_envs=4,relative_config_path=e.relative_config_path,
    hydra_overrides=e.hydra_overrides,lafan_tail_path=e.lafan_tail_path,data_mix_weights=e.data_mix_weights,
    disable_obs_noise=False,disable_domain_randomization=False,max_episode_length_s=None,
    root_height_obs=e.root_height_obs,robot_training=e.robot_training,direct_depth_profile=e.direct_depth_profile)
sim = make_mjlab_ufo_env_cfg(task,num_envs=4,seed=4831,mjcf_path=e.mjcf_path,auto_reset=False,
                           robot_training=e.robot_training)
print('PARITY_JSON=' + json.dumps(canonical(dict(train=cfg.model_dump(),
    task=OmegaConf.to_container(task,resolve=True), actions=sim.actions, events=sim.events,
    actuators=sim.scene.entities['robot'].articulation.actuators,
    depth=_depth_observation_contract(cfg))), sort_keys=True))
'''
    snapshots = []
    for root in (Path(baseline), candidate):
        run = subprocess.run([sys.executable, "-c", program], cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)}, text=True, capture_output=True)
        assert run.returncode == 0, run.stdout + run.stderr
        snapshots.append(json.loads(next(line.removeprefix("PARITY_JSON=")
            for line in run.stdout.splitlines() if line.startswith("PARITY_JSON="))))
    assert snapshots[0] == snapshots[1]
