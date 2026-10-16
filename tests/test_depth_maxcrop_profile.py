import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from pbfm.perception.instinct_direct_depth import (
    MAXCROP_PROFILE, RP1DirectDepthConfig, RP1DirectDepthRuntime,
    depth_config_for_profile, preprocess_rp1_depth, raycast_ranges_to_image_plane,
)
from pbfm.perception.depth_camera import (
    InstallationRandomizedRayCastSensor, make_depth_camera_sensor_cfg,
    optical_rays_from_intrinsics, source_from_optical_rotation,
)
from pbfm.agents.nn_models import DepthEncoder, _FixedAdaptiveAvgPool2x4


def test_maxcrop_white_invalid_and_far():
    cfg = depth_config_for_profile(MAXCROP_PROFILE)
    cfg.validate()
    for value in (float('nan'), float('inf'), -1., 3., 10.):
        image = torch.full((2, 36, 64), value)
        out = preprocess_rp1_depth(image, cfg, enable_noise=True)
        assert out.shape == (2, 18, 32)
        assert (out == 255).all()


def test_new_profile_has_no_image_randomness_and_correct_crop_sigma():
    cfg = depth_config_for_profile(MAXCROP_PROFILE)
    image = torch.linspace(0, 3, 36*64).reshape(1, 36, 64)
    a = preprocess_rp1_depth(image, cfg, enable_noise=True)
    b = preprocess_rp1_depth(image, cfg, enable_noise=False)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    g = torch.exp(-torch.arange(-1, 2).float().square()/2)
    g /= g.sum()
    expected = F.conv2d(F.pad(image.clamp(0, 2.5)[:, None, 18:, 16:-16], (1,1,1,1), mode='reflect'),
                        (g[:, None]*g[None, :])[None, None])[:, 0]
    torch.testing.assert_close(a, expected.div(2.5).mul(255).round().to(torch.uint8), atol=0, rtol=0)


def test_oblique_ray_is_not_discarded_before_optical_z_clipping():
    cfg = depth_config_for_profile(MAXCROP_PROFILE)
    camera = cfg.camera_config()
    rays = optical_rays_from_intrinsics(camera, device='cpu', dtype=torch.float32)
    distances = 2.3 / rays[..., 2]
    assert distances.max() > 2.5
    assert cfg.raycast_max_distance > distances.max()
    sensor = make_depth_camera_sensor_cfg(camera, raycast_max_distance=cfg.raycast_max_distance)
    assert sensor.max_distance > distances.max()
    depth = raycast_ranges_to_image_plane(distances[None], camera, invalid_to_max=True)
    torch.testing.assert_close(depth, torch.full_like(depth, 2.3))
    assert (raycast_ranges_to_image_plane(torch.full_like(distances[None], -1), camera, invalid_to_max=True) == 2.5).all()


@pytest.mark.parametrize('height', [18, 36])
def test_depth_encoder_shape_and_pool_equivalence(height):
    encoder = DepthEncoder(height=height, latent_dim=16)
    out = encoder(torch.ones(2, 8, height, 32, dtype=torch.uint8)*128)
    assert out.shape == (2, 16)
    feature = torch.randn(2, 64, (height+3)//4, 8)
    torch.testing.assert_close(encoder.frame_cnn[-1](feature), F.adaptive_avg_pool2d(feature, (2,4)))


def test_old_weights_load_without_parameter_changes():
    old, new = DepthEncoder(height=36), DepthEncoder(height=18)
    new.load_state_dict(old.state_dict(), strict=True)
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())


@pytest.mark.parametrize('height', [18, 36])
@pytest.mark.parametrize('stride,features', [(2,128), (1,256)])
def test_configurable_encoder_shapes_and_gradients(height, stride, features):
    encoder = DepthEncoder(height=height, first_conv_stride=stride, frame_feature_dim=features)
    depth = torch.randint(0, 256, (2,8,height,32), dtype=torch.uint8)
    first = encoder.frame_cnn[0](depth[:,0:1].float()/255)
    assert first.shape == (2,16,(height+stride-1)//stride,(32+stride-1)//stride)
    feature = encoder.frame_cnn[:-1](depth[:,0:1].float()/255)
    assert tuple(feature.shape[-2:]) == encoder.feature_map_shape()
    torch.testing.assert_close(encoder.frame_cnn[-1](feature), F.adaptive_avg_pool2d(feature,(2,4)))
    assert encoder.frame_projection[1].in_features == 512
    assert encoder.frame_projection[1].out_features == features
    assert encoder.gru.input_size == features and encoder.gru.hidden_size == 256
    out = encoder(depth)
    assert out.shape == (2,256)
    torch.testing.assert_close(out, encoder(depth.float()/255))
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_legacy_config_defaults_and_new_architecture_checkpoint_incompatibility():
    from pbfm.agents.nn_models import DirectDepthActorArchiConfig
    cfg = DirectDepthActorArchiConfig()
    assert cfg.depth_first_conv_stride == 2 and cfg.depth_frame_feature_dim == 128
    old = DepthEncoder()
    new = DepthEncoder(height=18, first_conv_stride=1, frame_feature_dim=256)
    with pytest.raises(RuntimeError, match='size mismatch'):
        new.load_state_dict(old.state_dict(), strict=True)


def test_reset_resamples_only_selected_installations():
    sensor = object.__new__(InstallationRandomizedRayCastSensor)
    bounds = tuple(math.radians(x) for x in (2, 10, 2))
    sensor.cfg = SimpleNamespace(position_error=.01, angle_error_rad=0., angle_error_rpy_rad=bounds)
    sensor._position_delta = torch.zeros(128, 3)
    sensor._rotation_delta = torch.eye(3).repeat(128, 1, 1)
    sensor.randomize_installation(torch.arange(128))
    old_pos, old_rot = sensor._position_delta.clone(), sensor._rotation_delta.clone()
    sensor.randomize_installation(torch.arange(64))
    assert sensor._position_delta.abs().max() <= .01
    assert not torch.equal(old_pos[:64], sensor._position_delta[:64])
    assert torch.equal(old_pos[64:], sensor._position_delta[64:])
    assert torch.equal(old_rot[64:], sensor._rotation_delta[64:])
    R = sensor._rotation_delta
    rpy = torch.stack((torch.atan2(R[:,2,1], R[:,2,2]), torch.asin(-R[:,2,0]), torch.atan2(R[:,1,0], R[:,0,0])), -1)
    assert (rpy.abs() <= torch.tensor(bounds)+1e-6).all()


def test_profile_checkpoint_mismatch_fail_closed():
    from pbfm.training.checkpoint import _depth_observation_contract, _validate_checkpoint_depth_observation_contract
    actor = SimpleNamespace(name='direct_depth')
    cfg = SimpleNamespace(agent=SimpleNamespace(model=SimpleNamespace(archi=SimpleNamespace(actor=actor))),
                          env=SimpleNamespace(direct_depth_profile='legacy_rp1'))
    old = _depth_observation_contract(cfg)
    cfg.env.direct_depth_profile = MAXCROP_PROFILE
    with pytest.raises(RuntimeError):
        _validate_checkpoint_depth_observation_contract({'depth_observation_contract':old}, cfg)
    new = _depth_observation_contract(cfg)
    assert new['output_shape'] == [18,32] and new['installation_resample'] == 'reset'


def test_history_uses_new_shape_and_resets():
    cfg = depth_config_for_profile(MAXCROP_PROFILE)
    rt = RP1DirectDepthRuntime(2, 'cpu', cfg, enable_noise=True)
    sensor = SimpleNamespace(data=SimpleNamespace(distances=torch.ones(2,36*64)))
    rt.reset_from_sensor(sensor, torch.arange(2))
    rt.append_from_sensor(sensor)
    assert rt.observation().shape == (2,8,18,32)
    sensor.data.distances.fill_(-1)
    rt.reset_from_sensor(sensor, torch.arange(2))
    assert (rt.observation() == 255).all()


def test_compact_replay_reconstructs_18x32_without_changing_storage_contract():
    from pbfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
    replay = TrajectoryDictBufferMultiDim(capacity=10, device='cpu', n_dim=2,
        end_key='truncated', output_key_t=['observation'], output_key_tp1=['observation'],
        compact_depth_history=True)
    newest = torch.arange(6,dtype=torch.uint8).reshape(6,1,1,1,1)
    depth = newest.expand(6,1,8,18,32).clone()
    done = torch.zeros(6,1,1,dtype=torch.bool)
    done[-1]=True
    replay.extend({'observation':{'depth_image':depth,'state':torch.arange(6).float().reshape(6,1,1)},
                   'terminated':torch.zeros_like(done),'truncated':done})
    assert replay.storage['observation']['depth_image'].shape == (10,1,18,32)
    restored = replay.get_full_buffer()['observation']['depth_image']
    assert restored.shape == (5,8,18,32)
    torch.testing.assert_close(restored[-1,:,0,0],torch.tensor([0,0,0,0,0,0,0,4],dtype=torch.uint8))


def test_small_agent_update_and_production_export_18x32(tmp_path):
    pytest.importorskip('onnxruntime')
    import gymnasium as gym
    import numpy as np
    from test_qv_agent_integration import make_agent, make_batch, Provider
    from pbfm.agents.fb_cpr_aux.config import FBcprAuxAgentConfig
    from pbfm.inference.export.depth_policy import export_direct_depth_policy
    old, dims = make_agent()
    cfg = old.cfg.model_dump()
    cfg['model']['archi']['actor'].update(depth_height=18, depth_processing_profile=MAXCROP_PROFILE,
                                         velocity_supervision_updates_depth=True,
                                         depth_first_conv_stride=1, depth_frame_feature_dim=256)
    dims['depth_image'] = (8, 18, 32)
    space = gym.spaces.Dict({k:gym.spaces.Box(-np.inf,np.inf,shape=v,dtype=np.float32) for k,v in dims.items()})
    agent = FBcprAuxAgentConfig.model_validate(cfg).build(space,29)
    batch = make_batch(agent,dims)
    batch['observation']['depth_image'] = torch.randint(0,256,(8,8,18,32),dtype=torch.uint8)
    batch['next']['observation']['depth_image'] = batch['observation']['depth_image'].clone()
    metrics = agent.update({'train':Provider(batch),'expert_slicer':Provider(batch,expert=True)},1024)
    assert torch.isfinite(metrics['velocity/critic_loss'])
    assert metrics['actor_gradient_local/V_norm'] > 0
    metadata = export_direct_depth_policy(agent._model.eval(), tmp_path/'policy.onnx')
    assert metadata['depth_input_contract']['processing_profile'] == MAXCROP_PROFILE
    assert metadata['depth_shape'] == [8,18,32]
    assert metadata['depth_encoder_architecture']['first_conv_stride'] == 1
    assert metadata['depth_encoder_architecture']['frame_feature_dim'] == 256
    assert len(metadata['verification']) == 12
