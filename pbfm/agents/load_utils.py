"""Load the PBFM agent or its model from a checkpoint directory."""

import json
from pathlib import Path
from typing import Any

from pbfm.agents.fb_cpr_aux.agent import FBcprAuxAgent
from pbfm.agents.fb_cpr_aux.model import FBcprAuxModel


def load_agent_from_checkpoint_dir(checkpoint_dir: str, device: str = "cpu") -> Any:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")
    with (checkpoint_dir / "config.json").open("r") as f:
        config = json.load(f)

    agent_name = config["name"]
    if agent_name != "FBcprAuxAgent":
        raise ValueError(f"Unknown agent name: {agent_name}. Available: ['FBcprAuxAgent']")
    agent = FBcprAuxAgent.load(checkpoint_dir, device=device)
    return agent


def load_model_from_checkpoint_dir(checkpoint_dir: str, device: str = "cpu") -> Any:
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "model" / "config.json").open("r") as f:
        config = json.load(f)
        
    model_name = config["name"]
    if model_name != "FBcprAuxModel":
        raise ValueError(f"Unknown model name: {model_name}. Available: ['FBcprAuxModel']")
    model = FBcprAuxModel.load(checkpoint_dir / "model", device=device)
    return model
