# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Agent checkpoint serialization with the existing config and state keys."""

import json
import pickle
from pathlib import Path

import safetensors.torch
import torch

from pbfm.utils.gym_spaces import json_to_space, space_to_json


def load_agent(config_class, path: str, device: str | None = None):
    path = Path(path)
    with (path / "config.json").open() as f:
        loaded_config = json.load(f)
    if device is not None:
        loaded_config["model"]["device"] = device

    if (path / "init_kwargs.pkl").exists():
        # Load arguments from a pickle file
        with (path / "init_kwargs.pkl").open("rb") as f:
            args = pickle.load(f)
        obs_space = args["obs_space"]
        action_dim = args["action_dim"]
    else:
        # load argeuments from a json file
        with (path / "init_kwargs.json").open("r") as f:
            args = json.load(f)
        obs_space = json_to_space(args["obs_space"])
        action_dim = args["action_dim"]

    # JSON has no tuple type, so strict construction cannot reload fields
    # such as direct_depth.proprio_keys after a checkpoint save. Keep
    # strict validation for newly-built configs and coerce JSON containers
    # only at this serialization boundary.
    config = config_class.model_validate(loaded_config, strict=False)
    agent = config.build(obs_space, action_dim)
    optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True, map_location=device)
    for k, v in optimizers.items():
        getattr(agent, k).load_state_dict(v)
    model_state = safetensors.torch.load_file(path / "model/model.safetensors", device=device or config.model.device)
    agent._model.load_state_dict(model_state, strict=False)
    del model_state
    agent._model.train()
    agent._model.requires_grad_(True)
    return agent


def save_agent(agent, output_folder: str) -> None:
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    json_dump = agent.cfg.model_dump()
    with (output_folder / "config.json").open("w+") as f:
        json.dump(json_dump, f, indent=4)
    # save optimizer
    torch.save(
        agent.optimizer_dict,
        output_folder / "optimizers.pth",
    )
    # save model
    model_folder = output_folder / "model"
    model_folder.mkdir(exist_ok=True)
    agent._model.save(output_folder=str(model_folder))

    # Save the arguments required to create this agent (in addition to the config)
    init_kwargs = {
        "obs_space": space_to_json(agent.obs_space),
        "action_dim": agent.action_dim,
    }
    with (output_folder / "init_kwargs.json").open("w") as f:
        json.dump(init_kwargs, f, indent=4)
