"""Exercise the actual Warp contact-force API on CPU MuJoCo contacts."""

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import pytest
import torch
import warp as wp
from mjlab.sim.sim_data import WarpBridge

from pbfm.envs.mjlab.foot_contacts import FootStumbleSensorCfg


@pytest.fixture
def contact_sensor():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <geom name="wall" type="plane" size="2 2 .1" pos="0 .045 0" quat=".7071067812 .7071067812 0 0"/>
      <body name="foot_body" pos="0 0 .02"><freejoint/>
        <geom name="foot" type="capsule" size=".05" fromto="-.2 0 0 .2 0 0" mass="10"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with wp.ScopedDevice("cpu"):
        mw = mjwarp.put_model(model)
        dw = mjwarp.put_data(model, data, nworld=2, nconmax=8, njmax=64)
    # put_data repeats contacts but tiles solver rows; align this host-upload fixture.
    for field in ("frame", "friction", "dim", "geom"):
        target = wp.to_torch(getattr(dw.contact, field))
        source = torch.as_tensor(np.asarray(getattr(data.contact, field))).reshape(data.ncon, *target.shape[1:])
        target[:2 * data.ncon] = source.repeat(2, *([1] * (source.ndim - 1)))
    sensor = FootStumbleSensorCfg(name="foot_stumble", body_names=("foot_body",), history_length=4).build()
    sensor.initialize(model, WarpBridge(mw), WarpBridge(dw), "cpu")
    return sensor, model, data


def test_all_contacts_include_weaker_side_force_and_history(contact_sensor):
    sensor, model, data = contact_sensor
    forces = []
    for i in range(data.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        forces.append(data.contact[i].frame.reshape(3, 3).T @ force[:3])
    forces = np.asarray(forces)
    costs = np.clip((np.linalg.norm(forces[:, :2], axis=1) - abs(forces[:, 2]) - 20) / 100, 0, 1)
    assert data.ncon == 4 and costs.max() == 1
    assert costs[np.linalg.norm(forces, axis=1).argsort()[-2:]].max() == 0
    sensor.update(.005)
    torch.testing.assert_close(sensor.data, torch.ones(2))
    sensor._data.nacon.zero_()
    sensor.update(0.)
    sensor.reset(torch.tensor([0]))
    torch.testing.assert_close(sensor.data, torch.tensor([0., 1.]))
    for _ in range(3):
        sensor.update(.005)
        torch.testing.assert_close(sensor.data, torch.tensor([0., 1.]))
    sensor.update(.005)
    assert sensor.data.eq(0).all()


def test_overflow_and_nonfinite_contact_force_fail_loudly(contact_sensor):
    sensor, _, _ = contact_sensor
    wp.to_torch(sensor._data.nacon).fill_(sensor._data.naconmax + 1)
    sensor.update(.005)
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        _ = sensor.data
    sensor.reset()
    wp.to_torch(sensor._data.nacon).fill_(8)
    wp.to_torch(sensor._data.efc.force).fill_(float("nan"))
    sensor.update(.005)
    with pytest.raises(RuntimeError, match="non-finite"):
        _ = sensor.data


def test_nonfoot_contacts_are_excluded(contact_sensor):
    sensor, _, _ = contact_sensor
    sensor._foot_geoms.zero_()
    sensor.update(.005)
    assert sensor.data.eq(0).all()


def test_constraint_overflow_is_not_silently_clipped(contact_sensor):
    sensor, _, _ = contact_sensor
    wp.to_torch(sensor._data.nefc)[0] = sensor._data.njmax + 1
    sensor.update(.005)
    with pytest.raises(RuntimeError, match="constraint capacity"):
        _ = sensor.data
