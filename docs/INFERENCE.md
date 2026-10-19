# Inference and deployment export

The only policy inference entry points are:

```bash
python -m pbfm.inference.tracking --help
python -m pbfm.inference.goal --help
python -m pbfm.inference.reward --help
```

Run from the repository root after installation. Supply the run directory with
`--model-folder`; it must contain the selected checkpoint. Motion data and weights
are external. Tracking includes camera, terrain selection, terrain-relative reset
and motion-reference/policy comparison options; a second preview program is not
needed. For a training terrain tile, for example:

```bash
python -m pbfm.inference.tracking --model-folder /path/to/run \
  --motion-list 20 --save-mp4 \
  --terrain-family low_stairs_down --difficulty-row 7
```

Terrain geometry is read from that checkpoint's configuration. The family name
alone does not specify the direction of travel: center-outward traversal of
`low_stairs_down` goes uphill. Check reset position, heading and actual geometry
before labelling a video.

## Context

Tracking advances an eight-frame reference latent window (shortened at the end).
Heading and valid planar-velocity targets use the same aligned reference. A valid
zero-velocity target means stop; an invalid target means no velocity instruction.
Goal and reward latents do not automatically create reference velocity targets.
Use the CLI's velocity-context options deliberately for custom motion names.

Keep the exported latent, heading/velocity fields, validity flags, frame rate and
coordinate convention together. A latent alone is not the complete input contract
of a command-conditioned policy. The policy always uses its estimated current
velocity, not simulator velocity labels, during deployment.

## Export

```bash
python -m scripts.export_depth_policy_onnx \
  --checkpoint /path/to/run/checkpoint --output /path/to/export/policy.onnx
python -m scripts.export_backward_encoder_onnx \
  --model-folder /path/to/run --output /path/to/export/backward_encoder.onnx --device cpu
python -m scripts.verify_depth_policy_trace --help
```

The exported `policy.onnx` includes **depth encoding, the velocity
estimator, observation normalization and the Actor**. It is not a split actor-only
graph. Backward encoding is a separate export. No critic or simulator-only velocity
label is an input to the deployed policy. Inspect the generated metadata for the
precise names and shapes instead of reusing another checkpoint's interface.

ONNX receives normalized float depth in [0,1]. Environment uint8 depth must be
converted to float32 and divided by 255 once; already-normalized float depth must
not be divided again. Real camera preprocessing, temporal history, context frame
alignment, action scaling and PD control remain outside the graph. ONNX parity
tests do not validate those real-robot integration steps.
