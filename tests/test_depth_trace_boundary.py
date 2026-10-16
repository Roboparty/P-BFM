import numpy as np
import pytest
import torch
from test_qv_agent_integration import make_agent, make_batch

from pbfm.inference.export.depth_policy import depth_to_onnx_float


def test_byte_conversion_and_reference_input_are_independent():
    from scripts.verify_depth_policy_trace import prepare_trace_batch
    depth = np.array([0, 128, 255], dtype=np.uint8).reshape(1, 1, 1, 3)
    original = depth.copy()
    data = {'depth_image': depth, 'z': np.zeros((1, 8), dtype=np.float32)}
    obs, z, feed = prepare_trace_batch(data, ['depth_image', 'z'], 'depth_image', 0, 1)
    assert obs['depth_image'].dtype == torch.uint8
    assert obs['depth_image'][0, 0, 0, 1].item() == 128
    assert feed['depth_image'].dtype == np.float32
    np.testing.assert_allclose(feed['depth_image'], [[[ [0., 128./255., 1.] ]]], atol=1e-7)
    np.testing.assert_array_equal(depth, original)
    assert not np.shares_memory(feed['depth_image'], depth)
    assert z.dtype == torch.float32


def test_float_depth_is_not_divided_twice():
    depth = np.array([0., 128./255., 1.], dtype=np.float32)
    np.testing.assert_array_equal(depth_to_onnx_float(depth), depth)


@pytest.mark.parametrize('depth', [
    np.array([128.], dtype=np.float32), np.array([-1.], dtype=np.float32),
    np.array([np.nan], dtype=np.float32), np.array([np.inf], dtype=np.float32),
])
def test_out_of_range_float_depth_is_rejected(depth):
    with pytest.raises(ValueError, match='normalized'):
        depth_to_onnx_float(depth)


def test_nonbyte_integer_depth_is_not_guessed():
    with pytest.raises(TypeError):
        depth_to_onnx_float(np.array([128], dtype=np.int16))


def test_uint8_trace_end_to_end_matches_saved_rollout_and_detects_corruption(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    from pbfm.utils.helpers import export_meta_policy_as_onnx
    from scripts.verify_depth_policy_trace import verify_trace
    agent, dims = make_agent()
    model = agent._model.eval()
    batch = make_batch(agent, dims)
    obs = {k: v[:3] for k, v in batch['observation'].items()}
    shape = obs['depth_image'].shape
    obs['depth_image'] = (torch.arange(int(np.prod(shape))) % 256).to(torch.uint8).reshape(shape)
    z = batch['z'][:3]
    # Anchor from the true raw-byte model, not the converted ONNX feed.
    estimate = model.estimate_velocity(obs).numpy()
    capture = []
    hook = model._actor.depth_encoder.frame_cnn.register_forward_pre_hook(
        lambda _module, args: capture.append(args[0].detach().clone()))
    expected_action = model.act(obs, z).numpy()
    hook.remove()
    assert capture[0].min().item() == 0 and capture[0].max().item() == 1
    assert torch.isclose(capture[0].flatten()[128], torch.tensor(128./255.))
    export_meta_policy_as_onnx(model, tmp_path, 'actor.onnx')
    session = ort.InferenceSession(str(tmp_path/'actor.onnx'), providers=['CPUExecutionProvider'])
    names = [i.name for i in session.get_inputs()]
    trace = {k: (z if k == 'z' else obs[k]).numpy() for k in names}
    trace['velocity_estimate'] = estimate
    trace['velocity_true_yaw'] = np.zeros_like(estimate)
    path = tmp_path/'trace.npz'
    np.savez_compressed(path, **trace)
    report = verify_trace(model, session, path)
    assert report['source_depth_dtype'] == 'uint8'
    assert report['source_depth_max'] == 255 and report['onnx_depth_max'] == 1
    assert report['pytorch_depth_dtype_preserved']
    assert report['max_abs_pytorch_vs_saved_estimate_error'] < 1e-6
    assert report['max_abs_onnx_vs_saved_estimate_error'] < 2e-4
    actual_action = session.run(None, {k: depth_to_onnx_float(v) if k == 'depth_image' else v
                                      for k, v in trace.items() if k in names})[0]
    np.testing.assert_allclose(actual_action, expected_action, atol=2e-4, rtol=2e-4)
    # A test that only compares two identically preprocessed paths misses this.
    trace['velocity_estimate'] = estimate + .1
    np.savez_compressed(path, **trace)
    with pytest.raises(AssertionError, match='saved rollout velocity_estimate'):
        verify_trace(model, session, path)


def test_export_rejects_depth_normalizer_that_breaks_byte_equivalence():
    from pbfm.inference.export.depth_policy import DirectDepthPolicyWrapper
    agent, _ = make_agent()
    agent._model._obs_normalizer._normalizers['depth_image'] = torch.nn.ReLU()
    with pytest.raises(ValueError, match='identity depth'):
        DirectDepthPolicyWrapper(agent._model)
