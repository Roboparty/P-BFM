"""Robot interfaces and CPU physics; GPU training requires a separate smoke test."""
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from pbfm.robots import load_robot_training_spec
from pbfm.train import build_ufo_mjlab_config
from pbfm.envs.mjlab.task_config import _compose_humanoidverse_config
from pbfm.envs.mjlab.config import make_mjlab_ufo_env_cfg, _direct_depth_config
from pbfm.envs.mjlab.actuators import RP1ActuatorCfg, rp1_clip_effort
from pbfm.motion.library.torch_humanoid_batch import Humanoid_Batch
from pbfm.inference.runtime import resolve_inference_robot_config, load_mjlab_env_cfg, write_mjlab_relabel_xml, _export_policy_model

ROBOT = "configs/robots/rp1_24dof.yaml"


@pytest.fixture(scope="module")
def rp1():
    spec = load_robot_training_spec(ROBOT)
    train = build_ufo_mjlab_config(device="cpu", num_envs=2, smoke=True,
                                  robot_config=ROBOT, data_path="rp1_motions.pkl")
    e = train.env
    task, _ = _compose_humanoidverse_config(
        num_envs=2, relative_config_path=e.relative_config_path,
        hydra_overrides=e.hydra_overrides, lafan_tail_path=e.lafan_tail_path,
        data_mix_weights=e.data_mix_weights, disable_obs_noise=False,
        disable_domain_randomization=False, max_episode_length_s=None,
        root_height_obs=e.root_height_obs, robot_training=e.robot_training,
        direct_depth_profile=e.direct_depth_profile)
    return spec, train, task


def test_rp1_requires_own_data():
    with pytest.raises(ValueError, match="RP1-retargeted"):
        build_ufo_mjlab_config(device="cpu", robot_config=ROBOT)


def test_shared_algorithm_and_g1_defaults_unchanged(rp1):
    spec, train, task = rp1
    baseline = build_ufo_mjlab_config(device="cpu", num_envs=2, smoke=True, data_path="g1_motions.pkl")
    expected = baseline.agent.model_dump()
    expected.update(aux_rewards=list(spec.robot_options["aux_rewards_scaling"]),
                    aux_rewards_scaling=spec.robot_options["aux_rewards_scaling"])
    expected["model"]["archi"]["actor"].update(depth_processing_profile="rp1_pufo", depth_width=36)
    assert train.agent.model_dump() == expected
    assert len(spec.robot.control_joint_names) == 24
    assert len(spec.robot.body_names) == 25
    assert task.robot.control.action_rescale is False
    assert spec.default_joint_angles["left_elbow_joint"] == -1.2
    assert task.robot.control.stiffness.left_knee_joint == 120
    depth = _direct_depth_config(task)
    assert depth.mount_body == "waist_yaw_link"
    assert depth.mount_pos == (.09175, .011, .3982)
    assert (depth.num_output_frames, depth.output_height, depth.output_width) == (8,18,36)


def test_rp1_scene_and_all_actuators_compile(rp1):
    from mjlab.scene import Scene
    spec, train, task = rp1
    cfg = make_mjlab_ufo_env_cfg(task, num_envs=2, seed=1,
        mjcf_path=spec.robot.xml_path, auto_reset=False, robot_training=spec.to_env_dict())
    motors = cfg.scene.entities["robot"].articulation.actuators
    assert len(motors) == 24 and all(isinstance(m, RP1ActuatorCfg) for m in motors)
    cfg.scene.terrain = None
    cfg.scene.spec_fn = None
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    assert model.nu == 24 and model.nq == 31
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], spec.effort_limits)
    # Test the compiled physical model, not only YAML dimensions.
    model.opt.timestep = .002
    data = mujoco.MjData(model)
    data.qpos[2] = 1.5
    data.qpos[3] = 1
    data.qpos[7:] = list(spec.default_joint_angles.values())
    for _ in range(50):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert data.qpos[2] < 1.5


def test_rp1_clips_absolute_pd_targets_not_raw_actions(rp1):
    from types import SimpleNamespace as NS
    spec, _, task = rp1
    cfg = make_mjlab_ufo_env_cfg(task, num_envs=2, seed=1,
        mjcf_path=spec.robot.xml_path, auto_reset=False, robot_training=spec.to_env_dict())
    names = spec.robot.control_joint_names
    default = torch.tensor(list(spec.default_joint_angles.values())).repeat(2, 1)
    robot = NS(find_joints_by_actuator_names=lambda _: (list(range(24)), names),
               data=NS(default_joint_pos=default))
    term = cfg.actions["actions"].build(NS(scene={"robot": robot}, num_envs=2, device="cpu"))
    raw = torch.tensor([8., -8.])[:, None].expand(2, 24)
    term.process_actions(raw)
    limits = torch.tensor([spec.robot.joint_ranges[name] for name in names])
    expected = (default + .4 * raw).clamp(limits[:, 0], limits[:, 1])
    torch.testing.assert_close(term._processed_actions, expected)
    torch.testing.assert_close(term.raw_action, raw)
    assert (expected[0] == limits[:, 1]).any()  # Hard limits, not 90% reward limits.


def test_motor_braking_and_no_load_speed():
    v = torch.tensor([-30., -10., 0., 10., 30.])
    peak = torch.full_like(v, 100.)
    assert torch.equal(rp1_clip_effort(torch.zeros_like(v), v, peak, 10., 20.), torch.zeros_like(v))
    assert torch.equal(rp1_clip_effort(peak, v, peak, 10., 20.), torch.tensor([100.,100.,100.,100.,0.]))
    assert torch.equal(rp1_clip_effort(-peak, v, peak, 10., 20.), torch.tensor([0.,-100.,-100.,-100.,-100.]))


def test_rp1_motion_library_loading(rp1,tmp_path):
    import joblib
    from pbfm.motion.library.motion_lib_robot import MotionLibRobot
    spec,_,task = rp1
    cfg=copy.deepcopy(task.robot.motion)
    motion=dict(robot_name=spec.robot.name, joint_names=list(spec.robot.control_joint_names),
                body_names=list(spec.robot.body_names),fps=50,
                root_trans_offset=np.tile([0.,0.,.85],(4,1)).astype(np.float32),
                pose_aa=np.zeros((4,25,3),np.float32),dof_pos=np.zeros((4,24),np.float32))
    path=tmp_path/'motion.pkl';joblib.dump({'walk':motion},path)
    cfg.motion_file=str(path)
    library=MotionLibRobot(cfg,num_envs=1,device='cpu')
    library.load_all_motions()
    state=library.get_motion_state(torch.tensor([0]),torch.tensor([.02]))
    assert state['dof_pos'].shape == (1,24)
    assert state['rg_pos_t'].shape == (1,25,3)


def test_fk_preserves_all_24_axes_and_matches_mujoco(rp1):
    spec, _, task = rp1
    fk = Humanoid_Batch(task.robot.motion, device="cpu")
    assert fk.has_freejoint and fk.dof_axis.shape == (24,3)
    model = mujoco.MjModel.from_xml_path(spec.robot.xml_path)
    q = torch.tensor(list(spec.default_joint_angles.values()))
    pose = torch.zeros(1,2,25,3)
    pose[:,:,1:] = q[:,None] * fk.dof_axis
    trans = torch.tensor([[[0.,0.,.85],[0.,0.,.85]]])
    result = fk.fk_batch(pose, trans)
    data = mujoco.MjData(model)
    data.qpos[:3] = trans[0,0].numpy()
    data.qpos[3:7] = [1,0,0,0]
    data.qpos[7:] = q.numpy()
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(result.global_translation[0,0].numpy(), data.xpos[1:], atol=2e-6)
    motion = dict(robot_name="rp1_24dof",joint_names=list(spec.robot.control_joint_names),
                  body_names=list(spec.robot.body_names),pose_aa=pose[0].numpy())
    fk.validate_reference(motion)
    with pytest.raises(ValueError, match="does not match"):
        fk.validate_reference({**motion,"robot_name":"g1_29dof"})
    with pytest.raises(ValueError, match="joint_names"):
        fk.validate_reference({**motion,"joint_names":motion["joint_names"][::-1]})


def test_checkpoint_selects_robot_and_rejects_wrong_override(rp1, tmp_path):
    spec, train, _ = rp1
    (tmp_path/"config.json").write_text(train.model_dump_json())
    assert resolve_inference_robot_config(None,None,tmp_path) == spec.config_path
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_inference_robot_config("configs/robots/g1_29dof.yaml",None,tmp_path)
    with pytest.raises(ValueError, match="checkpoint"):
        load_mjlab_env_cfg(tmp_path, device="cpu", robot_config=Path("configs/robots/g1_29dof.yaml"),
                          data_path=None,headless=True,disable_dr=True,disable_obs_noise=True,max_episode_length_s=10)


@pytest.mark.parametrize("name", ["g1_29dof", "rp1_24dof"])
def test_inference_and_export_keep_saved_robot_calibration(name, tmp_path, monkeypatch):
    from pbfm.inference import runtime
    from types import SimpleNamespace
    spec = load_robot_training_spec(f"configs/robots/{name}.yaml")
    saved = spec.to_env_dict()
    saved.update(action_scale=.3)
    saved["stiffness"]["left_knee_joint"] = 100.
    saved["default_joint_angles"]["left_knee_joint"] += .02
    saved["camera"] = {"mount_body": spec.torso_name, "mount_pos": [.11175, .011, .3982],
                       "camera_frame_quat_wxyz": [0., .5, 0., .866]}
    saved["robot"]["xml_path"] = "/unavailable-server/" + Path(spec.robot.xml_path).name
    (tmp_path / "config.json").write_text(json.dumps({"env": {"robot_training": saved, "lafan_tail_path": "motion.pkl"}}))
    restored = runtime.load_inference_robot_training(tmp_path, spec.config_path)
    expected = copy.deepcopy(saved)
    expected["robot"]["xml_path"] = spec.robot.xml_path
    assert restored.to_env_dict() == expected
    env, _ = load_mjlab_env_cfg(tmp_path, device="cpu", data_path=None, headless=True,
        disable_dr=True, disable_obs_noise=True, max_episode_length_s=10.)
    assert env.robot_training == expected
    count = len(spec.robot.control_joint_names)
    monkeypatch.setattr(runtime, "export_meta_policy_as_onnx", lambda *a, **kw: {"output_action_dim": count})
    meta = runtime._export_policy_model(SimpleNamespace(action_dim=count,
        cfg=SimpleNamespace(archi=SimpleNamespace(z_dim=256))), tmp_path / "export", restored)
    assert meta["robot_calibration"] == expected
    saved["robot"]["joint_ranges"]["left_knee_joint"][0] += .1
    with pytest.raises(ValueError, match="joint limits"):
        load_robot_training_spec(spec.config_path, saved=saved)


def test_reward_and_goal_entrypoints(rp1,tmp_path):
    from pbfm.inference.reward import _resolve_reward_tasks
    from pbfm.inference.goal import _find_goal_json
    from pbfm.envs.mjlab.rewards import LocomotionReward
    spec,_,_ = rp1
    tasks,_ = _resolve_reward_tasks(None,spec)
    path = write_mjlab_relabel_xml(Path(spec.robot.xml_path), tmp_path, spec.robot.control_joint_names,
                                  spec.robot.name,root_body_name=spec.robot.base_body)
    model = mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    assert np.isfinite(LocomotionReward.reward_from_name(tasks[0]).compute(model,data))
    with pytest.raises(ValueError, match="locomotion"):
        _resolve_reward_tasks(["raise-arms"],spec)
    with pytest.raises(ValueError, match="explicit"):
        _find_goal_json(None,num_dof=24,robot_name=spec.robot.name)
    goals=tmp_path/"goals.json";goals.write_text('[{"motion_id":0,"frames":[0]}]')
    assert _find_goal_json(goals,num_dof=24,robot_name=spec.robot.name) == goals


def test_rp1_update_checkpoint_and_production_onnx(rp1,tmp_path):
    from test_qv_agent_integration import make_agent, make_batch, Provider
    agent,dims = make_agent(num_dof=24, robot_train=rp1[1])
    batch=make_batch(agent,dims)
    metrics=agent.update({'train':Provider(batch),'expert_slicer':Provider(batch,expert=True)},1024)
    assert torch.isfinite(metrics['velocity/critic_loss'])
    agent.save(tmp_path/'checkpoint')
    loaded=type(agent).load(tmp_path/'checkpoint',device='cpu')
    for a,b in zip(agent._model.parameters(),loaded._model.parameters()):
        torch.testing.assert_close(a,b)
    metadata=_export_policy_model(loaded._model,tmp_path/'export',rp1[0])
    assert metadata['output_action_dim'] == 24
    assert metadata['robot_name'] == 'rp1_24dof'
    assert metadata['robot_calibration']['action_scale'] == .4
    assert metadata['depth_shape'] == [8,18,36]
    assert metadata['depth_input_contract']['normalization_range_m'] == [.1,2.5]
    assert metadata['verification']


def test_rp1_rewards_and_dr_reach_actual_configs(rp1):
    spec, train, task = rp1
    weights = dict(penalty_action_rate=-.2, sound_suppression_acc_per_foot=-.005,
        joint_acc_l2=-1e-6, joint_vel_l2=-1e-3, limits_dof_pos=-10., penalty_body_impact=-1.,
        penalty_slippage=-4., penalty_ankle_roll=-1., feet_stumble=-2.)
    assert train.agent.aux_rewards_scaling == task.rewards.reward_scales == weights
    assert task.rewards.body_impact_static_gain == 0
    assert task.rewards.reward_limit.soft_dof_pos_limit == .9
    assert task.robot.penalize_contacts_on == ['waist_yaw_link','hip','shoulder']
    dr = task.domain_rand
    assert dr.kp_range == dr.kd_range == [.7,1.3]
    assert dr.link_mass_range == [.8,1.2] and dr.friction_range == [.3,1.7]
    assert dr.base_com_range.x == dr.base_com_range.y == dr.base_com_range.z == [-.03,.03]
    assert dr.push_interval_s == [1,3] and dr.max_push_vel_xy == dr.max_push_ang_vel == 1.
    assert not dr.randomize_default_dof_pos and not dr.randomize_ctrl_delay
    assert not task.lie_down_init
    cfg = make_mjlab_ufo_env_cfg(task,num_envs=2,seed=1,mjcf_path=spec.robot.xml_path,
                               auto_reset=False,robot_training=spec.to_env_dict())
    motors = cfg.scene.entities['robot'].articulation.actuators
    assert all((m.delay_min_lag,m.delay_max_lag,m.delay_hold_prob,m.rfi_lim) == (0,2,1.,.1) for m in motors)
    assert cfg.actions['actions'].__class__.__name__ == 'JointPositionActionCfg'
    assert any(s.name == 'foot_stumble' for s in cfg.scene.sensors)
    assert not any(s.name == 'foot_collision_contacts' for s in cfg.scene.sensors)
    assert cfg.events['motor_delay'].mode == cfg.events['random_pd_gains'].mode == 'reset'
    for name in ('random_link_pseudo_inertia','random_base_com','random_geom_friction'):
        assert cfg.events[name].mode == 'startup'
    assert cfg.events['push_robots'].func.__name__ == 'set_push_velocity'
    from pbfm.training.checkpoint import _depth_observation_contract
    contract = _depth_observation_contract(train)
    assert contract['output_shape'] == [18,36]
    assert contract['installation_resample'] == 'startup'
    assert contract['normalization_range_m'] == [.1,2.5]


def test_rp1_disable_dr_and_noise(rp1):
    spec,train,_ = rp1
    e=train.env
    task,_ = _compose_humanoidverse_config(num_envs=2,relative_config_path=e.relative_config_path,
        hydra_overrides=e.hydra_overrides,lafan_tail_path=e.lafan_tail_path,data_mix_weights=None,
        disable_obs_noise=True,disable_domain_randomization=True,max_episode_length_s=None,
        root_height_obs=e.root_height_obs,robot_training=e.robot_training,direct_depth_profile=e.direct_depth_profile)
    cfg=make_mjlab_ufo_env_cfg(task,num_envs=2,seed=1,mjcf_path=spec.robot.xml_path,
        auto_reset=False,robot_training=spec.to_env_dict())
    assert not cfg.events
    assert all(m.rfi_lim == m.delay_max_lag == 0 for m in cfg.scene.entities['robot'].articulation.actuators)
    assert all(value == 0 for value in task.obs.noise_scales.values())
    depth=_direct_depth_config(task)
    assert depth.position_error == 0 and depth.angle_error_rpy_rad == (0.,0.,0.)


def test_rp1_impact_excludes_static_load_and_uses_com_acceleration():
    from pbfm.envs.mjlab.aux_rewards import body_contact_severity,foot_impact_cost
    force=torch.tensor([[[0.,0.,100.]]]);weight=torch.tensor([100.]);vel=torch.zeros_like(force)
    assert body_contact_severity(force,vel,weight,static_gain=0).item() == 0
    assert body_contact_severity(force,vel,weight).item() > 0  # G1 unchanged
    vel[...,2]=1.
    assert body_contact_severity(force,vel,weight,static_gain=0).item() > 0
    v=torch.tensor([[[0.,0.,2.],[0.,0.,10.]]]);acc=v*2
    contacts=torch.tensor([[[0.,0.,1.],[0.,0.,0.]]])
    torch.testing.assert_close(foot_impact_cost(v,acc,contacts),torch.tensor([20.]))
    # A free-falling body's real world acceleration is g, not zero.
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="b" pos="0 0 2"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody><sensor><framelinacc name="acc" objtype="body" objname="b"/></sensor></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_step(model,data)
    np.testing.assert_allclose(data.sensordata + model.opt.gravity, data.qacc[:3], atol=1e-10)


def test_rp1_reward_collection_produces_all_nine_channels(rp1):
    from types import SimpleNamespace as NS
    from pbfm.envs.mjlab.aux_rewards import _compute_reward
    weights=rp1[1].agent.aux_rewards_scaling
    zero=torch.zeros(2,4);quat=torch.tensor([0.,0.,0.,1.]).repeat(2,2,1)
    gravity=torch.tensor([0.,0.,-9.81])
    forces=torch.zeros(2,2,3);forces[:,:,2]=100.
    sensors={f'robot/foot_com_acc_{i}':NS(data=-gravity.repeat(2,1)) for i in range(2)}
    core=NS(num_envs=2,device='cpu',config=rp1[2],reward_scales=weights,
        reward_penalty_reward_names=set(),use_reward_penalty_curriculum=False,
        contact_forces=forces,contact_force_history=forces[:,:,None],feet_indices=torch.tensor([0,1]),
        torques=zero,actions=zero,last_actions=zero,dof_pos=zero,dof_vel=torch.ones_like(zero),
        dof_pos_limits=torch.tensor([[-1.,1.]]).repeat(4,1),dof_vel_limits=torch.ones(4)*10,
        torque_limits=torch.ones(4)*100,penalised_contact_indices=[0,1],terrain_aware_auxiliary=True,
        robot_weight=torch.ones(2)*100,left_ankle_dof_indices=[0,1],right_ankle_dof_indices=[2,3],
        body_rot=quat,body_vel=torch.zeros(2,2,3),base_quat=quat[:,0],
        gravity_vec=torch.tensor([0.,0.,-1.]).repeat(2,1),forward_vec=torch.tensor([1.,0.,0.]).repeat(2,1),
        _body_ids=[0,1],_joint_ids=[0,1,2,3],
        robot=NS(data=NS(joint_acc=torch.ones_like(zero)*2,body_com_vel_w=torch.zeros(2,2,6))),
        mjlab_env=NS(scene=NS(sensors=sensors),sim=NS(mj_model=NS(opt=NS(gravity=gravity.numpy())))),
        _read_foot_collision_force_history=lambda: forces[:,:,None],virtual_terrain_edges=None,
        feet_height_sensor=None)
    reward,aux=_compute_reward(core,pre_body_vel=torch.zeros(2,2,3))
    assert set(weights).issubset(aux)
    assert all(torch.isfinite(aux[key]).all() for key in weights)
    torch.testing.assert_close(aux['joint_vel_l2'],torch.full((2,),4.))
    torch.testing.assert_close(aux['joint_acc_l2'],torch.full((2,),16.))
    assert aux['sound_suppression_acc_per_foot'].eq(0).all()
    assert aux['penalty_body_impact'].eq(0).all()
    torch.testing.assert_close(reward,torch.full((2,),-4e-3-16e-6))
    sensors['foot_stumble'] = NS(data=torch.tensor([.4, .8]))
    updated_reward, aux = _compute_reward(core, pre_body_vel=torch.zeros(2,2,3))
    torch.testing.assert_close(aux['feet_stumble'], torch.tensor([.4, .8]))
    torch.testing.assert_close(updated_reward - reward, torch.tensor([-.8, -1.6]))


def test_rp1_torque_noise_is_clipped_and_motor_delay_is_shared(monkeypatch):
    from types import SimpleNamespace
    from pbfm.envs.mjlab.actuators import RP1Actuator,reset_motor_delay
    motor=SimpleNamespace(cfg=RP1ActuatorCfg(target_names_expr=('joint',),stiffness=1.,damping=1.,
        effort_limit=100.,keep_velocity=10.,velocity_limit=20.,rfi_lim=.1),
        force_limit=torch.tensor([100.]),_joint_velocity=torch.tensor([20.]))
    monkeypatch.setattr(torch,'rand_like',lambda x: torch.ones_like(x))
    assert RP1Actuator._clip_effort(motor,torch.tensor([99.])).item() == 0  # No drive at no-load speed.
    assert RP1Actuator._clip_effort(motor,torch.tensor([-200.])).item() == -100.
    calls=[]
    motors=[SimpleNamespace(set_lags=lambda lag,ids: calls.append((lag.clone(),ids.clone()))) for _ in range(24)]
    env=SimpleNamespace(num_envs=10,device='cpu',scene={'robot':SimpleNamespace(actuators=motors)})
    reset_motor_delay(env,torch.tensor([1,4,9]),(0,2))
    assert len(calls) == 24 and calls[0][0].shape == (3,)
    assert all(torch.equal(calls[0][0],lag) and torch.equal(ids,torch.tensor([1,4,9])) for lag,ids in calls)


def test_rp1_randomization_does_not_accumulate():
    from types import SimpleNamespace as NS
    from pbfm.envs.mjlab.randomization import scale_link_density,set_push_velocity
    target=NS(body_mass=torch.ones(3,2),body_inertia=torch.ones(3,2,3))
    writes=[]
    robot=NS(indexing=NS(body_ids=torch.tensor([0,1])),
             write_root_link_velocity_to_sim=lambda v,env_ids: writes.append((v,env_ids)))
    env=NS(num_envs=3,device='cpu',scene={'robot':robot},sim=NS(model=target,
        mj_model=NS(body_mass=np.array([2.,3.]),body_inertia=np.ones((2,3))*4)))
    for _ in range(2):
        scale_link_density(env,torch.tensor([1]),(1.2,1.2))
        torch.testing.assert_close(target.body_mass[1],torch.tensor([2.4,3.6]))
        torch.testing.assert_close(target.body_inertia[1],torch.full((2,3),4.8))
    assert target.body_mass[0].tolist() == [1.,1.]
    set_push_velocity(env,torch.tensor([1]),{'x':(1.,1.),'yaw':(-1.,-1.)})
    assert writes[0][0].tolist() == [[1.,0.,0.,0.,0.,-1.]]


def test_rp1_depth_shape_range_and_noise_switch(monkeypatch):
    from pbfm.perception.instinct_direct_depth import depth_config_for_profile,preprocess_depth,_rp1_pufo_noise
    cfg=depth_config_for_profile('rp1_pufo')
    assert cfg.crop == (18,0,14,14) and cfg.position_error == .02
    np.testing.assert_allclose(np.rad2deg(cfg.angle_error_rpy_rad),[3,10,3])
    raw=torch.full((3,36,64),1.3)
    clean=preprocess_depth(raw,cfg,enable_noise=False)
    assert clean.dtype == torch.uint8 and clean.shape == (3,18,36)
    assert ((clean.float()-128).abs() <= 1).all()
    assert preprocess_depth(torch.full_like(raw,float('nan')),cfg,enable_noise=False).eq(255).all()
    assert preprocess_depth(torch.zeros_like(raw),cfg,enable_noise=False).eq(0).all()
    monkeypatch.setattr('pbfm.perception.instinct_direct_depth.random.random',lambda:1.)
    monkeypatch.setattr(torch,'randn_like',lambda x: torch.ones_like(x))
    x=torch.tensor([[[.05,.1,1.,2.5,3.]]])
    torch.testing.assert_close(_rp1_pufo_noise(x),torch.tensor([[[.05,.11,1.01,2.51,3.]]]))
    assert not torch.equal(preprocess_depth(raw,cfg,enable_noise=True),clean)


@pytest.mark.parametrize("disable_dr", [False, True])
def test_rp1_gpu_rollout_and_partial_reset(tmp_path, disable_dr):
    import os
    if os.environ.get("PBFM_TEST_RP1_GPU") != "1" or not torch.cuda.is_available():
        pytest.skip("requires PBFM_TEST_RP1_GPU=1 and PBFM_TEST_RP1_DATA")
    from pbfm.utils.helpers import get_backward_observation
    from pbfm.training.workspace import transition_successor
    cfg = build_ufo_mjlab_config(device="cuda:0", work_dir=str(tmp_path), num_envs=4,
        smoke=True, robot_config=ROBOT, data_path=os.environ["PBFM_TEST_RP1_DATA"],
        terrain_mode="training_mix", disable_dr=disable_dr, disable_obs_noise=disable_dr)
    env, _ = cfg.env.build(num_envs=4)
    try:
        obs, _ = env.reset(to_numpy=False)
        core = env.base_env
        assert obs["state"].shape == (4, 54)
        assert obs["privileged_state"].shape == (4, 373)
        assert obs["depth_image"].shape == (4, 8, 18, 36)
        assert obs["depth_image"].dtype == torch.uint8
        reference, _ = get_backward_observation(core, 0, use_root_height_obs=cfg.env.root_height_obs)
        assert reference["state"].shape[-1] == 54
        assert reference["privileged_state"].shape[-1] == 373
        captured = {}
        reset = core.reset_finished

        def capture(ids):
            captured["ids"] = ids.clone()
            captured["qpos"], _ = env._get_qpos_qvel(to_numpy=False)
            captured["depth"] = core._direct_depth_runtime.observation().clone()
            reset(ids)

        core.reset_finished = capture
        for step in range(12):
            if step == 5:
                core.mjlab_env.episode_length_buf[0] = 100000
            obs, reward, terminated, truncated, info = env.step(torch.zeros_like(core.actions), to_numpy=False)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert set(cfg.agent.aux_rewards).issubset(info["aux_rewards"])
            assert all(torch.isfinite(value).all() for value in info["aux_rewards"].values())
            if step == 5:
                assert truncated[0]
                successor = transition_successor(obs, info)
                ids = captured["ids"]
                torch.testing.assert_close(successor["qpos"][ids], captured["qpos"][ids])
                torch.testing.assert_close(successor["observation"]["depth_image"][ids], captured["depth"][ids])
                assert obs["time"][0].item() == 0
    finally:
        env.close()
