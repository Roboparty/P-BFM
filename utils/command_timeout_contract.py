"""Validation for the evidence-backed G1 low-command timeout contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


def _sha256(value: Any, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return normalized


def validate_command_timeout_contract(robot_config: dict[str, Any]) -> None:
    verified = bool(robot_config.get("COMMAND_TIMEOUT_VERIFIED", False))
    contract = robot_config.get("COMMAND_TIMEOUT_CONTRACT")
    if not isinstance(contract, dict):
        raise ValueError("COMMAND_TIMEOUT_CONTRACT must be an explicit mapping")
    required_trials = int(contract.get("required_repeat_trials", 0))
    if required_trials < 3:
        raise ValueError("COMMAND_TIMEOUT_CONTRACT requires at least 3 repeat trials")
    reports = contract.get("report_sha256", [])
    if not isinstance(reports, list):
        raise ValueError("COMMAND_TIMEOUT_CONTRACT.report_sha256 must be a list")

    status = str(contract.get("status", "")).strip().lower()
    if not verified:
        if status != "unverified":
            raise ValueError(
                "false COMMAND_TIMEOUT_VERIFIED requires contract status=unverified"
            )
        return

    if status != "verified":
        raise ValueError(
            "true COMMAND_TIMEOUT_VERIFIED requires contract status=verified"
        )
    text_fields = (
        "target_robot_id",
        "firmware_version",
        "interface",
        "safe_behavior",
    )
    for field in text_fields:
        if not str(contract.get(field, "")).strip():
            raise ValueError(f"COMMAND_TIMEOUT_CONTRACT.{field} is required")
    _sha256(
        contract.get("g1_interface_sha256", ""),
        "COMMAND_TIMEOUT_CONTRACT.g1_interface_sha256",
    )
    if len(reports) < required_trials:
        raise ValueError(
            "COMMAND_TIMEOUT_CONTRACT does not contain enough repeat-trial reports"
        )
    for index, digest in enumerate(reports):
        _sha256(digest, f"COMMAND_TIMEOUT_CONTRACT.report_sha256[{index}]")
    timeout_min = float(contract.get("timeout_min_ms", float("nan")))
    timeout_max = float(contract.get("timeout_max_ms", float("nan")))
    if not (
        np.isfinite(timeout_min)
        and np.isfinite(timeout_max)
        and 0.0 < timeout_min <= timeout_max
    ):
        raise ValueError(
            "COMMAND_TIMEOUT_CONTRACT timeout_min_ms/timeout_max_ms are invalid"
        )
    verified_utc = str(contract.get("verified_utc", "")).strip()
    try:
        parsed = datetime.fromisoformat(verified_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "COMMAND_TIMEOUT_CONTRACT.verified_utc must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError("COMMAND_TIMEOUT_CONTRACT.verified_utc must include a timezone")
