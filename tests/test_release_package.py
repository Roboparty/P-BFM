import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from scripts import package_release


def test_source_and_model_assets_are_separate_and_verifiable(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    models = root / 'model/g1_policy'
    (models / 'onnx').mkdir(parents=True)
    weights = {}
    for name in ('actor.onnx', 'depth_encoder.onnx', 'backward_encoder.onnx'):
        path = models / 'onnx' / name
        path.write_bytes(name.encode())
        weights[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (models / 'manifest.json').write_text(json.dumps({'onnx_sha256': weights, 'asset_sha256': {}}))
    (root / 'README.md').write_text('release')
    inventory = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()) + ['FILES.txt']
    (root / 'FILES.txt').write_text('\n'.join(inventory)+'\n')
    monkeypatch.setattr(package_release, 'ROOT', root)
    assets = package_release.package(tmp_path / 'output')
    with tarfile.open(assets[0]) as archive:
        assert not any(name.endswith('.onnx') for name in archive.getnames())
        assert 'PBFM-deploy/README.md' in archive.getnames()
    with tarfile.open(assets[1]) as archive:
        assert set(archive.getnames()) == {f'PBFM-deploy/model/g1_policy/onnx/{name}' for name in weights}
    for line in (tmp_path / 'output/SHA256SUMS').read_text().splitlines():
        sha, name = line.split()
        assert sha == package_release.digest(tmp_path / 'output' / name)
    # Model corruption must fail before replacing release assets.
    (models / 'onnx/actor.onnx').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='hash mismatch'):
        package_release.package(tmp_path / 'output')


def test_package_refuses_output_inside_source(tmp_path, monkeypatch):
    monkeypatch.setattr(package_release, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='outside'):
        package_release.package(tmp_path / 'build')
