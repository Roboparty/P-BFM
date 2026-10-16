"""Export an immutable checkpoint using the same production path as tracking."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pbfm.agents.load_utils import load_model_from_checkpoint_dir
from pbfm.utils.helpers import export_meta_policy_as_onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_model_from_checkpoint_dir(args.checkpoint, device="cpu").eval()
    metadata = export_meta_policy_as_onnx(model, args.output.parent, args.output.name)
    print(f"Verified {args.output}: {metadata['verification']}")


if __name__ == "__main__":
    main()
