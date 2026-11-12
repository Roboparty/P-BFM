"""YAML configuration loading with relative inheritance."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping with optional relative ``extends`` inheritance."""

    def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def load(current: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        current = current.expanduser().resolve()
        if current in stack:
            chain = " -> ".join(str(item) for item in (*stack, current))
            raise ValueError(f"Cyclic YAML config inheritance: {chain}")
        with current.open() as file:
            payload = yaml.load(file, Loader=yaml.FullLoader)
        if not isinstance(payload, dict):
            raise ValueError(f"YAML config must contain a mapping: {current}")
        parent = payload.pop("extends", None)
        if parent is None:
            return payload
        parent_path = Path(str(parent)).expanduser()
        if not parent_path.is_absolute():
            parent_path = current.parent / parent_path
        return merge(load(parent_path, (*stack, current)), payload)

    return load(Path(path), ())

