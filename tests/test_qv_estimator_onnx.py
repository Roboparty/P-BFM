import numpy as np
import pytest
import torch
from test_qv_agent_integration import make_agent, make_batch


def test_production_backward_verification_accepts_pruned_last_action(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    from pbfm.inference.export.backward_encoder import export_backward_encoder_from_model
    agent, _ = make_agent()
    output = tmp_path/'backward.onnx'
    export_backward_encoder_from_model(agent._model.eval(), output, verify=True)
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    assert 'last_action' not in [x.name for x in session.get_inputs()]


def test_production_export_includes_depth_normalization_and_estimator(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    from pbfm.agents.normalizers import BatchNormNormalizerConfig
    from pbfm.utils.helpers import export_meta_policy_as_onnx
    agent, dims = make_agent()
    model = agent._model.eval()
    # Nontrivial trained statistics: identity-only parity misses preprocessing bugs.
    normalizer = BatchNormNormalizerConfig().build(model.obs_space['state']).eval()
    normalizer._normalizer.running_mean.fill_(.7)
    normalizer._normalizer.running_var.fill_(1.9)
    model._obs_normalizer._normalizers['state'] = normalizer
    before = {k: v.clone() for k, v in model.state_dict().items()}
    metadata = export_meta_policy_as_onnx(model, tmp_path, 'production.onnx')
    assert metadata['format'] == 'direct_depth_full_v1'
    assert len(metadata['verification']) == 12
    assert {c['source_depth_dtype'] for c in metadata['verification']} == {'float32', 'uint8'}
    assert metadata['depth_input_contract']['onnx_range'] == [0.0, 1.0]
    session = ort.InferenceSession(str(tmp_path/'production.onnx'), providers=['CPUExecutionProvider'])
    names = [i.name for i in session.get_inputs()]
    assert 'depth_image' in names and 'velocity_command' in names and 'heading' in names
    assert not {'privileged_state', 'velocity_target_yaw', 'terrain_priv'} & set(names)
    batch = make_batch(agent, dims)
    for size in [1, 2, 3]:
        obs = {k: v[:size] for k,v in batch['observation'].items()}
        obs['velocity_command'] = torch.tensor([[0.,0.,0.], [0.,0.,1.], [.8,0.,1.]])[:size]
        z = batch['z'][:size]
        feed = {k: (z if k=='z' else obs[k]).numpy() for k in names}
        action, estimate = session.run(None, feed)
        np.testing.assert_allclose(action, model.act(obs,z).numpy(), atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(estimate, model.estimate_velocity(obs).numpy(), atol=2e-4, rtol=2e-4)
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_deployable_estimated_velocity_onnx(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    import onnx
    agent, dims = make_agent()
    actor = agent._model._actor.eval()
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = actor
        def forward(self,state,last_action,history_actor,heading,velocity_command,depth_latent,z):
            dist = self.actor.forward_from_depth_latent(
                dict(state=state,last_action=last_action,history_actor=history_actor,
                     heading=heading,velocity_command=velocity_command),z,.05,depth_latent)
            return dist.mean,dist.velocity_estimate
    wrapper = Wrapper()
    keys = ['state','last_action','history_actor','heading','velocity_command','depth_latent','z']
    sizes = [64,29,372,2,3,256,8]
    inputs = tuple(torch.randn(2,s) for s in sizes)
    path = tmp_path/'actor_with_velocity.onnx'
    torch.onnx.export(wrapper,inputs,str(path),opset_version=17,dynamo=False,
                      input_names=keys,output_names=['action','estimated_velocity'],
                      dynamic_axes={k:{0:'batch'} for k in keys+['action','estimated_velocity']})
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    assert {i.name for i in session.get_inputs()}==set(keys)
    for batch in [1,2,3]:
        values=tuple(torch.randn(batch,s) for s in sizes)
        ref=wrapper(*values)
        actual=session.run(None,{k:v.detach().numpy() for k,v in zip(keys,values)})
        for a,b in zip(ref,actual):
            np.testing.assert_allclose(a.detach().numpy(),b,rtol=1e-4,atol=1e-5)
