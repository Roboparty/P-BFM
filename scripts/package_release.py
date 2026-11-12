"""Build local source/model release assets from the reviewed FILES.txt inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def package(output):
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('Release output must be outside the source repository')
    manifest = json.loads((ROOT / 'model/g1_policy/manifest.json').read_text())
    weights = {f'model/g1_policy/onnx/{name}': sha for name, sha in manifest['onnx_sha256'].items()}
    for relative, expected in (weights | manifest['asset_sha256']).items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Asset hash mismatch: {relative}')
    inventory = (ROOT / 'FILES.txt').read_text().splitlines()
    paths = []
    for relative in inventory:
        path = ROOT / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink() or ROOT not in path.resolve().parents or not path.is_file():
            raise ValueError(f'Invalid release inventory entry: {relative}')
        if path.suffix in ('.engine', '.so', '.log', '.pyc') or any(p in ('.git', '.venv', '__pycache__', 'build', 'logs') for p in path.relative_to(ROOT).parts):
            raise ValueError(f'Local build/runtime file in inventory: {relative}')
        if path.suffix == '.onnx' and relative not in weights:
            raise ValueError(f'Unexpected model in inventory: {relative}')
        if relative.startswith('config/local'):
            raise ValueError(f'Local configuration in release inventory: {relative}')
        paths.append(relative)
    if not set(weights).issubset(paths):
        raise ValueError('Model files are missing from FILES.txt')
    output.mkdir(parents=True, exist_ok=True)
    assets = []
    for name, selected in [('pbfm-g1-source.tar.gz', [p for p in paths if p not in weights]),
                           ('pbfm-g1-models.tar.gz', list(weights))]:
        destination = output / name
        with tarfile.open(destination, 'w:gz') as archive:
            for relative in sorted(selected):
                archive.add(ROOT / relative, arcname='PBFM-deploy/' + relative, recursive=False)
        assets.append(destination)
    (output / 'SHA256SUMS').write_text(''.join(f'{digest(path)}  {path.name}\n' for path in assets))
    return assets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    for path in package(parser.parse_args().output_dir):
        print(path)
