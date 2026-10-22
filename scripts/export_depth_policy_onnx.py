"""Export an immutable checkpoint using the same production path as tracking."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pbfm.agents.load_utils import load_model_from_checkpoint_dir
from pbfm.inference.runtime import _export_policy_model, load_inference_robot_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, help="Run directory containing config.json; defaults to checkpoint parent.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    folder = args.model_folder or args.checkpoint.parent
    if not (folder / "config.json").is_file():
        parser.error("Robot calibration requires the run config.json; supply --model-folder")
    robot_training = load_inference_robot_training(folder)
    model = load_model_from_checkpoint_dir(args.checkpoint, device="cpu").eval()
    metadata = _export_policy_model(model, args.output.parent, robot_training, output_name=args.output.name)
    print(f"Verified {args.output}: {metadata['verification']}")


if __name__ == "__main__":
    main()
