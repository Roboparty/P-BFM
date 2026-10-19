import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from pbfm.evaluation import terrain
from pbfm.inference import goal, reward, runtime, tracking
from pbfm.terrains.presets import TRAINING_TERRAIN_COMPONENT_NAMES


@pytest.mark.parametrize("option", ["--difficulty-row", "--rp1-difficulty-row"])
def test_terrain_evaluation_accepts_canonical_and_archived_row_option(option, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "terrain", "--model-folder", "/tmp/pbfm-test", "--prompt-type", "tracking", option, "7",
    ])
    assert terrain.parse_args().difficulty_row == 7


@pytest.mark.parametrize("entrypoint", [tracking, goal, reward, terrain])
def test_public_inference_rejects_legacy_perception_checkpoint(entrypoint, monkeypatch, capsys):
    args = ["inference", "--model-folder", "/tmp/pbfm-test", "--perception-checkpoint", "/tmp/legacy.pt"]
    if entrypoint is terrain:
        args += ["--prompt-type", "tracking"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        entrypoint.parse_args()
    assert error.value.code == 2
    assert "unrecognized arguments: --perception-checkpoint" in capsys.readouterr().err


def test_tracking_preserves_both_reset_height_modes_and_source_reset():
    target = {"root_states": torch.tensor([[1.0, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0]])}
    env = SimpleNamespace(terrain_enabled=True, env_origins=torch.tensor([[10.0, 20.0, 0.6]]))
    tracked = tracking._center_target_states_on_terrain(target, env)
    previewed = tracking._center_target_states_on_terrain(target, env, terrain_relative_height=True)
    torch.testing.assert_close(tracked["root_states"][0, :3], torch.tensor([10.0, 20.0, 0.8]))
    torch.testing.assert_close(previewed["root_states"][0, :3], torch.tensor([10.0, 20.0, 1.4]))
    torch.testing.assert_close(target["root_states"][0, :3], torch.tensor([1.0, 2.0, 0.8]))


def test_saved_non_g1_config_cannot_bypass_inference_validation(tmp_path):
    config = {"env": {"robot_training": {"robot": {"name": "other_robot", "control_joint_names": []}}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Only G1"):
        runtime.load_mjlab_env_cfg(
            tmp_path, data_path=None, device="cpu", headless=True,
            disable_dr=True, disable_obs_noise=True, max_episode_length_s=10.0,
        )


@pytest.mark.parametrize("family", [None, *TRAINING_TERRAIN_COMPONENT_NAMES])
def test_tracking_cli_exposes_explicit_terrain_options(family, monkeypatch):
    args = ["tracking", "--model-folder", "/tmp/pbfm-test"]
    if family is not None:
        args += ["--terrain-family", family, "--difficulty-row", "3",
                 "--reset-height-mode", "reference", "--disable-heading-input"]
    monkeypatch.setattr(sys, "argv", args)
    parsed = tracking.parse_args()
    assert parsed.terrain_family == family
    assert parsed.difficulty_row == (7 if family is None else 3)
    assert parsed.reset_height_mode == (None if family is None else "reference")
    assert parsed.disable_heading_input == (family is not None)


@pytest.mark.parametrize(
    ("options", "expected_z", "failure"),
    [({}, 0.8, None),
     ({"terrain_family": "flat", "difficulty_row": 3}, 1.4, None),
     ({"terrain_family": "low_stairs_down", "disable_heading_input": True}, 1.4, None),
     ({"terrain_family": "low_stairs_up", "reset_height_mode": "reference"}, 0.8, None),
     ({"terrain_family": "low_stairs_up"}, 1.4, "tile"),
     ({}, 0.8, "renderer setup"),
     ({}, 0.8, "renderer close")],
)
def test_tracking_generic_and_training_rollouts_keep_context_export_and_rendering(options, expected_z, failure, monkeypatch, tmp_path):
    (tmp_path / "checkpoint").mkdir()
    xml = tmp_path / "g1.xml"
    xml.touch()
    joint_names = list(runtime.G1_MJLAB_DOF_NAMES)
    robot = SimpleNamespace(robot=SimpleNamespace(
        xml_path=str(xml), control_joint_names=joint_names,
        joint_qpos_addr={name: index + 7 for index, name in enumerate(joint_names)},
    ))
    scene = Mock()
    env = SimpleNamespace(
        terrain_enabled=True, env_origins=torch.tensor([[10.0, 20.0, 0.6]]),
        base_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]), dt=0.02,
        mjlab_env=SimpleNamespace(scene=SimpleNamespace(spec=scene)),
        _motion_lib=SimpleNamespace(load_all_motions=Mock(), _motion_data_keys=["dance1_subject2"]),
    )
    observation = {"heading": torch.ones(1, 2), "velocity_command": torch.zeros(1, 3)}
    wrapped = SimpleNamespace(
        _env=env, reset=Mock(return_value=(observation, {})),
        step=Mock(return_value=(observation, 0.0, False, False, {})), close=Mock(),
    )
    cfg = SimpleNamespace(build=Mock(return_value=(wrapped, {})), mjcf_path=str(xml), lafan_tail_path="motions.pkl")
    inputs = []

    def act(obs, z, mean):
        assert mean
        inputs.append({key: value.clone() for key, value in obs.items()})
        return torch.zeros(1, 29)

    model = SimpleNamespace(
        cfg=SimpleNamespace(seq_length=1, heading_context_enabled=True, velocity_critic_enabled=True,
                            reference_velocity_policy="all_reference"),
        backward_map=lambda obs: torch.ones(len(obs["state"]), 2), project_z=lambda z: z,
        to=Mock(), eval=Mock(), act=act,
    )
    rotations = torch.tensor([[[0.0, 0.0, 0.0, 1.0]],
                              [[0.0, 0.0, 2 ** -0.5, 2 ** -0.5]],
                              [[0.0, 0.0, 2 ** -0.5, 2 ** -0.5]]])
    expert = {
        "ref_body_pos": torch.tensor([[[1.0, 2.0, 0.8]]]).repeat(3, 1, 1),
        "ref_body_rots": rotations,
        "ref_body_vels": torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]]),
        "ref_body_angular_vels": torch.zeros(3, 1, 3),
        "dof_pos": torch.zeros(3, 29), "ref_dof_vel": torch.zeros(3, 29),
    }
    expert_renderer = Mock(input_nq=36, render_qpos=Mock(return_value=np.full((2, 2, 3), 11, dtype=np.uint8)))
    policy_renderer = Mock(input_nq=36, render_qpos=Mock(return_value=np.full((3, 4, 3), 22, dtype=np.uint8)))
    renderer = Mock(side_effect=[expert_renderer, policy_renderer])
    qpos = np.zeros(36)
    qpos[:2] = [10.0, 20.0]
    export, tile = Mock(), Mock()
    terrain_cfg, video = Mock(return_value=cfg), Mock()
    patches = {
        "_resolve_tracking_robot_config": Mock(return_value=tmp_path / "g1.yaml"),
        "load_robot_training_spec": Mock(return_value=robot),
        "load_model_from_checkpoint_dir": Mock(return_value=model),
        "load_mjlab_env_cfg": Mock(return_value=(cfg, False)),
        "get_backward_observation": Mock(return_value=({"state": torch.zeros(3, 1)}, expert)),
        "_export_tracking_onnx": export, "_terrain_env_cfg": terrain_cfg, "_assign_training_tile": tile,
        "MujocoQposRenderer": renderer, "policy_qpos_from_env": Mock(return_value=qpos),
    }
    for name, value in patches.items():
        monkeypatch.setattr(tracking, name, value)
    monkeypatch.setattr(tracking.media, "write_video", video)
    for name, value in {"RP1_ISOLATED_FAMILY": "boxes", "RP1_PREVIEW_DIFFICULTY_ROW": "0",
                        "PBFM_PREVIEW_DISABLE_HEADING": "1"}.items():
        monkeypatch.setenv(name, value)

    if failure == "tile":
        tile.side_effect = RuntimeError("tile assignment failed")
    elif failure == "renderer setup":
        renderer.side_effect = [expert_renderer, RuntimeError("renderer setup failed")]
    elif failure == "renderer close":
        policy_renderer.close.side_effect = RuntimeError("renderer close failed")
    with pytest.raises(RuntimeError, match=failure) if failure else nullcontext():
        tracking.run_tracking_inference(
            model_folder=tmp_path, data_path=None, robot_config=None, headless=True, device="cpu",
            save_mp4=True, disable_dr=True, disable_obs_noise=True, motion_list=[0], render_size=16,
            camera_distance=3.5, camera_azimuth=57.0, camera_elevation=-12.0, fps=50,
            max_steps=2, log_every_steps=0, max_episode_length_s=10.0, export_onnx=True, **options,
        )
    if failure:
        wrapped.close.assert_called_once_with()
        if failure == "tile":
            renderer.assert_not_called()
        else:
            expert_renderer.close.assert_called_once_with()
        if failure == "renderer close":
            policy_renderer.close.assert_called_once_with()
        return

    family = options.get("terrain_family")
    if family is None:
        terrain_cfg.assert_not_called()
        tile.assert_not_called()
    else:
        terrain_cfg.assert_called_once_with(cfg, family, seed=4831)
        tile.assert_called_once_with(env, family=family, difficulty_row=options.get("difficulty_row", 7))
    reset = wrapped.reset.call_args.kwargs["target_states"]["root_states"]
    torch.testing.assert_close(reset[0, :3], torch.tensor([10.0, 20.0, expected_z]))
    export.assert_called_once_with(model, tmp_path / "exported", robot)
    assert len(inputs) == wrapped.step.call_count == 2
    torch.testing.assert_close(inputs[0]["velocity_command"], torch.tensor([[1.0, 0.0, 1.0]]))
    torch.testing.assert_close(inputs[1]["velocity_command"], torch.tensor([[0.0, 1.0, 1.0]]), atol=1e-6, rtol=0.0)
    heading_disabled = options.get("disable_heading_input", False)
    expected_heading = torch.zeros(2, 2) if heading_disabled else torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    torch.testing.assert_close(torch.cat([obs["heading"] for obs in inputs]), expected_heading, atol=1e-6, rtol=0.0)
    with np.load(tmp_path / "tracking_inference/context_0.npz") as context:
        assert np.all(context["heading_valid"] == (not heading_disabled))
        assert context["velocity_valid"].all()
        np.testing.assert_array_equal(context["z"], np.ones((2, 2)))

    for call in renderer.call_args_list:
        assert (call.kwargs["camera_distance"], call.kwargs["camera_azimuth"], call.kwargs["camera_elevation"]) == (3.5, 57.0, -12.0)
    render_spec = renderer.call_args_list[1].kwargs["scene_spec"]
    scene.delete.assert_not_called()
    scene.worldbody.add_geom.assert_not_called()
    if family == "flat":
        scene.copy.assert_called_once_with()
        assert render_spec is scene.copy.return_value
        render_spec.delete.assert_called_once_with(render_spec.body.return_value)
        assert render_spec.worldbody.add_geom.call_args.kwargs["contype"] == 0
        np.testing.assert_array_equal(policy_renderer.render_qpos.call_args.args[0][:2], [0.0, 0.0])
    else:
        scene.copy.assert_not_called()
        assert render_spec is scene
    np.testing.assert_array_equal(qpos[:2], [10.0, 20.0])
    frames = video.call_args.args[1]
    assert len(frames) == 2 and frames[0].shape == (3, 8, 3)
    assert np.all(frames[0][:, :4] == 11) and np.all(frames[0][:, 4:] == 22)
    assert video.call_args.kwargs["fps"] == 50
    expert_renderer.close.assert_called_once_with()
    policy_renderer.close.assert_called_once_with()
    wrapped.close.assert_called_once_with()
