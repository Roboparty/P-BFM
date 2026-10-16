from types import SimpleNamespace

import torch

from pbfm.envs.mjlab.config import HumanoidVerseMjlabConfig
from pbfm.envs.mjlab.core import HumanoidVerseMjlabCore
from pbfm.envs.mjlab.observations import get_observation
from pbfm.envs.mjlab.vector_env import HumanoidVerseMjlabVectorEnv


def test_environment_creation_and_vector_interface_have_separate_owners():
    assert HumanoidVerseMjlabConfig.__module__ == "pbfm.envs.mjlab.config"
    assert HumanoidVerseMjlabCore.__module__ == "pbfm.envs.mjlab.core"
    assert HumanoidVerseMjlabVectorEnv.__module__ == "pbfm.envs.mjlab.vector_env"
    config = HumanoidVerseMjlabConfig(lafan_tail_path="unused", device="cpu")
    assert config.relative_config_path == "exp/bfm_zero/bfm_zero"


def test_observation_noise_order_and_history_before_update():
    events = []
    previous = {"actions": torch.full((1, 1, 2), 10.0), "dof_pos": torch.full((1, 1, 2), 20.0)}

    def query(key):
        events.append(("query", key))
        return previous[key]

    def add(key, value):
        events.append(("add", key))
        previous[key] = value[:, None].clone()

    history_config = {"dof_pos": 1, "actions": 1}
    raw_values = {
        "actions": torch.full((1, 2), 1.0),
        "base_ang_vel": torch.full((1, 2), 2.0),
        "dof_pos": torch.full((1, 2), 3.0),
        "dof_vel": torch.full((1, 2), 4.0),
        "projected_gravity": torch.full((1, 2), 5.0),
        "history_actor": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
    }
    core = SimpleNamespace(
        **{key: value for key, value in raw_values.items() if key != "history_actor"},
        config=SimpleNamespace(obs=SimpleNamespace(
            obs_scales={}, noise_scales=dict.fromkeys(raw_values, 0.1),
            obs_auxiliary={"history_actor": history_config},
        )),
        _creation_config=SimpleNamespace(
            evaluation_fast_path=True, velocity_context_enabled=True,
            include_velocity_estimator_target=False,
        ),
        default_dof_pos=torch.zeros(1, 2), default_dof_pos_offset=torch.zeros(1, 2),
        history_handler=SimpleNamespace(query=query, add=add),
        is_evaluating=False, add_noise_currculum=False,
        episode_length_buf=torch.tensor([7]), num_envs=1, device="cpu", terrain_enabled=False,
    )
    torch.manual_seed(841)
    expected = {key: value + (torch.rand_like(value) * 2.0 - 1.0) * 0.1 for key, value in raw_values.items()}
    torch.manual_seed(841)
    observation = get_observation(core, to_numpy=False)

    torch.testing.assert_close(observation["state"], torch.cat([
        expected[key] for key in ("dof_pos", "dof_vel", "projected_gravity", "base_ang_vel")
    ], dim=-1))
    torch.testing.assert_close(observation["last_action"], expected["actions"])
    torch.testing.assert_close(observation["history_actor"], expected["history_actor"])
    torch.testing.assert_close(observation["heading"], torch.zeros(1, 2))
    torch.testing.assert_close(observation["velocity_command"], torch.zeros(1, 3))
    assert events == [("query", "actions"), ("query", "dof_pos"), ("add", "dof_pos"), ("add", "actions")]
