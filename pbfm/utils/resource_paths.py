"""Resolve bundled resources without rewriting archived configuration files."""

from __future__ import annotations

import os
from pathlib import Path

import pbfm


PACKAGE_ROOT = Path(pbfm.__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"


def remap_resource_path(raw_path: str | os.PathLike[str]) -> Path:
    """Relocate known code resources; existing absolute/external paths win.

    Only bundled package data and configurations are relocatable. In
    particular, an external motion-data path is never resolved by basename.
    """
    path = Path(os.path.expandvars(os.fspath(raw_path))).expanduser()
    if path.is_absolute() and path.exists():
        return path
    parts = path.parts
    indices = range(len(parts) - 1) if path.is_absolute() else range(min(1, len(parts) - 1))
    for index in indices:
        root, resource = parts[index : index + 2]
        if root in {"humanoidverse", "pbfm"} and resource in {"data", "config", "configs"}:
            suffix = parts[index + 1 :]
            if ".." not in suffix:
                if resource == "config":
                    return CONFIG_ROOT.joinpath("simulation", *suffix[1:])
                if resource == "configs":
                    return CONFIG_ROOT.joinpath(*suffix[1:])
                return PACKAGE_ROOT.joinpath(*suffix)
        if root in {"code", "UFO", "UFO-g1depth", "PBFM"} and resource == "configs":
            suffix = parts[index + 1 :]
            if ".." not in suffix:
                return CONFIG_ROOT.joinpath(*suffix[1:])
    if not path.is_absolute() and parts and parts[0] == "configs" and ".." not in parts:
        return CONFIG_ROOT.joinpath(*parts[1:])
    return path


def resolve_resource_path(raw_path: str | os.PathLike[str], *, base_dir: Path | None = None) -> Path:
    """Keep config-relative/cwd lookup, then fall back to bundled resources."""
    path = Path(os.path.expandvars(os.fspath(raw_path))).expanduser()
    if path.is_absolute():
        return remap_resource_path(path).resolve()
    candidates = [] if base_dir is None else [Path(base_dir) / path]
    candidates.append(Path.cwd() / path)
    mapped = remap_resource_path(path)
    if mapped != path:
        candidates.append(mapped)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[-1]).resolve()
