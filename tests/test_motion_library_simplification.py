"""G1 motion loading keeps the original XML order and unextended body count."""

from pathlib import Path
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from pbfm.motion.library.motion_lib_robot import MotionLibRobot
from pbfm.motion.library.torch_humanoid_batch import Humanoid_Batch


def _g1_motion_config():
    package = Path(__file__).resolve().parents[1] / "pbfm"
    cfg = OmegaConf.load(package / "configs/simulation/robot/g1/g1_29dof_hard_waist.yaml").robot.motion
    cfg.asset.assetRoot = str(package / "data/robots/g1_mjlab")
    return cfg


@pytest.mark.parametrize("extend_head", [False, True])
def test_g1_xml_body_order_and_local_transforms(extend_head):
    cfg = _g1_motion_config()
    if not extend_head:
        cfg.extend_config = []
    fk = Humanoid_Batch(cfg)
    assert fk.body_names == list(cfg.body_names)
    assert fk.num_bodies == 30
    assert fk.num_bodies_augment == 30 + int(extend_head)
    assert fk.num_dof == 29
    # Recorded from SkeletonTree before its duplicate body-count parse was removed.
    assert fk.mjcf_data["parent_indices"].tolist() == [
        -1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 0, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 15, 23, 24, 25, 26, 27, 28,
    ]
    root = ET.parse(fk.mjcf_file).getroot()
    bodies = list(root.find("worldbody/body").iter("body"))
    assert fk.body_names == [body.attrib["name"] for body in bodies]
    for key, attribute, default in (
        ("local_translation", "pos", "0 0 0"),
        ("local_rotation", "quat", "1 0 0 0"),
    ):
        expected = torch.tensor([
            [float(value) for value in body.get(attribute, default).split()] for body in bodies
        ])
        torch.testing.assert_close(fk.mjcf_data[key], expected, rtol=0, atol=0)
    joints = root.findall("worldbody/.//joint")
    expected_axes = torch.tensor([[float(value) for value in joint.attrib["axis"].split()] for joint in joints])
    torch.testing.assert_close(fk.dof_axis, expected_axes, rtol=0, atol=0)


def test_g1_motion_initialization_and_loading(tmp_path):
    cfg = _g1_motion_config()
    dof_pos = np.arange(4 * 29, dtype=np.float32).reshape(4, 29) / 1000
    motion_file = tmp_path / "tiny_motion.pkl"
    joblib.dump({"tiny": {
        "root_trans_offset": np.zeros((4, 3), dtype=np.float32),
        "pose_aa": np.zeros((4, 30, 3), dtype=np.float32),
        "dof_pos": dof_pos,
        "fps": 50,
    }}, motion_file)
    cfg.motion_file = str(motion_file)
    library = MotionLibRobot(cfg, num_envs=1, device="cpu")
    assert library._num_unique_motions == 1
    assert library.mesh_parsers.num_bodies == 30
    library.load_all_motions()
    assert library.num_joints == library.num_bodies == 30
    assert library.num_motions() == 1
    assert library.gts.shape == (4, 30, 3)
    assert library.gts_t.shape == (4, 31, 3)
    torch.testing.assert_close(library.dof_pos, torch.from_numpy(dof_pos), rtol=0, atol=0)
    state = library.get_motion_state(torch.tensor([0]), torch.tensor([1 / 50]))
    torch.testing.assert_close(state["dof_pos"], torch.from_numpy(dof_pos[1:2]))
    assert state["rg_pos"].shape == (1, 30, 3)
