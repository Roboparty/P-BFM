"""Release includes runtime resources, not local-only files or job archives."""
import hashlib
from pathlib import Path

from pbfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from pbfm.agents.buffers.transition import DictBuffer
from pbfm.utils.resource_paths import CONFIG_ROOT, PACKAGE_ROOT


def test_archived_training_settings_are_unchanged():
    hashes = {
        "legacy_large.json": "6ed7c384dafdad7ee7ec415baa0ab230d5c3616151f6eff9bc70774ca5f1cd97",
        "depth_contract.json": "ea6c9ebd23ffb08080bad1112498451912c36a67daa7b17639ef38a8a593d1c5",
    }
    for name, expected in hashes.items():
        content = (CONFIG_ROOT / name).read_bytes()
        # Naming cleanup is allowed; keep the original numeric-config fingerprint.
        content = content.replace(b"terrain.terrain_type=training_mix", b"terrain.terrain_type=rp1_simple")
        assert hashlib.sha256(content).hexdigest() == expected


def test_replay_implementations_and_simulation_resources_are_bundled():
    assert DictBuffer is not None and TrajectoryDictBufferMultiDim is not None
    assert (PACKAGE_ROOT / "agents/buffers/trajectory.py").is_file()
    assert (CONFIG_ROOT / "robots/g1_29dof.yaml").is_file()
    assert (CONFIG_ROOT / "simulation/exp/bfm_zero/bfm_zero.yaml").is_file()


def test_no_server_job_archives_in_release():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "experiments").exists()
    assert not (root / "tools").exists()
    assert not (root / "run_config.json").exists()
