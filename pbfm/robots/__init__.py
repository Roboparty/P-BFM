"""Robot specification helpers for robot-aware motion data."""

from pbfm.robots.mujoco_parser import load_robot_spec
from pbfm.robots.robot_spec import RobotSpec
from pbfm.robots.training_spec import (
    RobotTrainingSpec,
    assert_robot_configs_compatible,
    load_robot_training_spec,
    resolve_robot_config_path,
    validate_robot,
)

__all__ = [
    "RobotSpec",
    "RobotTrainingSpec",
    "assert_robot_configs_compatible",
    "load_robot_spec",
    "load_robot_training_spec",
    "resolve_robot_config_path",
    "validate_robot",
]
