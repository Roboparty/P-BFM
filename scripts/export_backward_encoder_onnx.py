#!/usr/bin/env python3
"""Export the PBFM backward latent encoder as a standalone ONNX model."""

from __future__ import annotations

from pbfm.inference.export.backward_encoder import export_backward_encoder_from_checkpoint


main = export_backward_encoder_from_checkpoint


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
