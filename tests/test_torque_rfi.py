from types import SimpleNamespace
import pytest
import torch
from mjlab.envs.mdp.actions import JointPositionAction
from pbfm.envs.mjlab.torque_rfi import RfiJointPositionAction, RfiJointPositionActionCfg


def make_term(monkeypatch, fraction=.1, limits=(10.,20.,30.)):
    env=SimpleNamespace(device='cpu',num_envs=4,
        sim=SimpleNamespace(data=SimpleNamespace(qfrc_applied=torch.full((4,9),7.))))
    def initialize(self,cfg,env):
        self.cfg,self._env=cfg,env
        self._entity=SimpleNamespace(indexing=SimpleNamespace(joint_v_adr=torch.tensor([6,7,8])))
        self._target_ids=torch.arange(3)
    monkeypatch.setattr(JointPositionAction,'__init__',initialize)
    monkeypatch.setattr(JointPositionAction,'apply_actions',lambda self:None)
    monkeypatch.setattr(JointPositionAction,'reset',lambda self,env_ids=None:None)
    cfg=RfiJointPositionActionCfg(entity_name='robot',actuator_names=('.*',),torque_limits=limits,rfi_lim=fraction)
    return RfiJointPositionAction(cfg,env),env


def test_noise_resamples_is_bounded_and_never_accumulates(monkeypatch):
    term,env=make_term(monkeypatch)
    term.apply_actions()
    old=term.last_noise.clone()
    for _ in range(30):
        term.apply_actions()
        assert (term.last_noise.abs() <= .1*term._limits).all()
        torch.testing.assert_close(env.sim.data.qfrc_applied[:,6:],term.last_noise)
        assert (env.sim.data.qfrc_applied[:,:6]==7).all()
    assert not torch.equal(old,term.last_noise)


def test_partial_reset_clears_only_selected_joint_forces(monkeypatch):
    term,env=make_term(monkeypatch)
    term.apply_actions(); old=term.last_noise.clone()
    term.reset(torch.tensor([0,2]))
    assert (env.sim.data.qfrc_applied[[0,2],6:]==0).all()
    torch.testing.assert_close(term.last_noise[[1,3]],old[[1,3]])
    torch.testing.assert_close(env.sim.data.qfrc_applied[[1,3],6:],old[[1,3]])
    term.reset()
    assert (env.sim.data.qfrc_applied[:,6:]==0).all()
    assert (env.sim.data.qfrc_applied[:,:6]==7).all()


@pytest.mark.parametrize('fraction',[-.1,1.1,float('nan')])
def test_invalid_fraction_rejected(monkeypatch,fraction):
    with pytest.raises(ValueError,match='fraction'):
        make_term(monkeypatch,fraction=fraction)


def test_bad_limit_order_or_values_rejected(monkeypatch):
    with pytest.raises(ValueError,match='order'):
        make_term(monkeypatch,limits=(10.,20.))
    with pytest.raises(ValueError,match='positive'):
        make_term(monkeypatch,limits=(10.,-1.,30.))
